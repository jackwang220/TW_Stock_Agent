"""LLM 正向選股 (positive tilt) harness — 不是 veto,是「看多分數→加倉」。

哲學(使用者洞察):台股新聞是「報哪些會漲」不是「預警哪些會跌」→ veto 死路(已證 -490pp),
所以這版把 LLM 的多空判斷當成 edge 的「tilt 乘數」:LLM 越看多 → 部位放越大;
LLM 越看空 → 部位縮(但不歸零,因為新聞報空通常是過期的已跌完消息)。

候選 = H+反彈雙引擎每日 top-N(與 exp_60d_entry_compare/exp_60d_llm 完全相同);
baseline = 不加 LLM 的純引擎(⑤買收賣開,各窗皆正 alpha 的最佳執行);
tilt 版 = edge_tilted = edge * (1 + tilt * tanh((bull-bear)/3)),掃 tilt 強度找甜蜜點。

曝險中性:tilt 乘數以 1.0 為中心(tanh∈[-1,1]),整體 avg_expo 幾乎不動 → 比較公平。
報告同時印 baseline / tilt 兩者 avg_expo + 換手 + 平均持股,確認不是高曝險虛假贏。

用法:
    uv run python scripts/news_llm_select.py                       # 用快取 bull-bear(零成本),掃 tilt
    uv run python scripts/news_llm_select.py --provider openai --model gpt-4o-mini --days 60 --rerun
    uv run python scripts/news_llm_select.py --tilt 0.1,0.3,0.6,1.0,2.0

第一輪允許輕微洩漏(見報告 leakage 註記):新聞窗 ≤ 決策日,但 finmind 新聞時戳可能含當日盤後標題。
"""
from __future__ import annotations
import sys, os, json, argparse, importlib.util, math
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

v5  = _load("v5",  ROOT / "scripts/exp_step1_v5.py")
v6  = _load("v6",  ROOT / "scripts/exp_step1_v6.py")
ec  = _load("ec",  ROOT / "scripts/exp_60d_entry_compare.py")
cpv = _load("cpv", ROOT / "scripts/crossperiod_validate.py")
features, _factors, oh = v5.features, v5._factors, v5.oh
h_score = ec.h_score
sim5 = ec.sim_buyclose_sellopen          # ⑤買收賣開:各窗皆正 alpha 的最佳執行引擎
_finmind_stock = cpv._finmind_stock

END = "2026-06-08"
# 新聞窗短(finmind 2022+ 標題、google 近期最足)→ 重點看短窗;長窗仍跑供 regime 對照
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252), ("1年半", 378), ("2年", 504)]
# 5 regime(對齊 v15 報告的切法)
REGIMES = {
    "2021復甦":   ("2021-06-01", "2021-12-31"),
    "2022空頭":   ("2022-01-01", "2022-12-31"),
    "2023復甦":   ("2023-01-01", "2023-12-31"),
    "2024-25多頭": ("2024-01-01", "2025-06-30"),
    "2025下-26":  ("2025-07-01", "2026-06-08"),
}
TOPN = 4
CACHE = DATA_DIR / "_60dllm_debate.json"   # 既有快取:{date_code:[verdict,bull,bear]} 220 筆(同窗)


def tilt_mult(bull, bear, tilt, lo=0.0, hi=None):
    """正向選股 tilt:LLM 越看多(bull-bear 大)→ 乘數越大。
    中心 1.0(tanh(0)=0),tilt 控制強度。允許上不封頂(hi=None)、下限不歸零(lo)。"""
    m = 1.0 + tilt * math.tanh((bull - bear) / 3.0)
    m = max(lo, m)
    if hi is not None:
        m = min(hi, m)
    return m


def build_candidates(codes, names, feats, twii_feat, reb_cache, turn_pct, sig_days):
    """每日 H+反彈雙引擎 top-N 候選 → {date: [(score, code), ...]}。"""
    regime_bull = {
        d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
        for d in sig_days
    }
    cands = {}
    for d in sig_days:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        sc = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            v = (h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                 if bull else reb_cache.get(c, {}).get(d, 0.0))
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        cands[d] = sc[:TOPN]
    return cands, regime_bull


def load_llm_cache():
    """讀既有辯論快取:{(d,c): (verdict, bull, bear)}。center%/conf 沒存 → tilt 只用 bull-bear。"""
    res = {}
    if CACHE.exists():
        for k, v in json.loads(CACHE.read_text(encoding="utf-8")).items():
            if not v:
                continue
            d, c = k.rsplit("_", 1)
            res[(d, c)] = (v[0], int(v[1]), int(v[2]))
    return res


def rerun_llm(pairs, names, provider, model, workers=1):
    """重跑 LLM(正向):存完整 (verdict, bull, bear, center%, conf) 到 provider 專屬快取。"""
    # 切後端
    import tw_stock_agent.debate.bear as bear_mod
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    _orig_cfg = bear_mod.cfg
    def _cfg(key, default=None):
        if key == "llm.provider":
            return provider
        if key in ("llm.debate", "debate"):
            return model or _orig_cfg(key, default)
        return _orig_cfg(key, default)
    bear_mod.cfg = _cfg
    from tw_stock_agent.debate.bear import run_debate

    pcache = DATA_DIR / f"_news_select_{provider}_{(model or 'def').replace('/','-')}.json"
    res = {}
    if pcache.exists():
        for k, v in json.loads(pcache.read_text(encoding="utf-8")).items():
            d, c = k.rsplit("_", 1); res[(d, c)] = tuple(v) if v else None
    todo = [p for p in pairs if p not in res]
    logger.info(f"重跑 LLM {provider}/{model}:候選去重 {len(pairs)},快取已有 {len(pairs)-len(todo)},待跑 {len(todo)}")

    import time as _t
    throttle = float(os.environ.get("NEWS_LLM_THROTTLE", "0"))   # 秒/候選,避開 TPM 限流

    def _one(p):
        d, c = p
        try:
            if throttle:
                _t.sleep(throttle)
            stock = _finmind_stock(c, names.get(c, c), d)
            if stock is None:
                return p, None
            r = run_debate(stock, [], historical_date=date.fromisoformat(d))
            return p, (r.verdict, int(r.bull_score), int(r.bear_score),
                       float(r.predicted_center_pct), float(r.prediction_confidence))
        except Exception as e:
            logger.warning(f"  {c}@{d} {str(e)[:50]}"); return p, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for p, v in ex.map(_one, todo):
            res[p] = v; done += 1
            if done % 10 == 0:
                pcache.write_text(json.dumps(
                    {f"{d}_{c}": list(val) if val else None for (d, c), val in res.items()},
                    ensure_ascii=False), encoding="utf-8")
                logger.info(f"  LLM {done}/{len(todo)}")
    pcache.write_text(json.dumps(
        {f"{d}_{c}": list(val) if val else None for (d, c), val in res.items()},
        ensure_ascii=False), encoding="utf-8")
    bear_mod.cfg = _orig_cfg
    return res   # {(d,c): (verdict, bull, bear, center, conf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="openai|claude|gemini(省略=只用既有快取的 bull-bear)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--days", type=int, default=60, help="新聞窗(僅供記錄;回測仍跑全窗多窗呈現)")
    ap.add_argument("--tilt", default="0.1,0.3,0.6,1.0,2.0", help="tilt 強度掃描,逗號分隔")
    ap.add_argument("--rerun", action="store_true", help="重跑 LLM(會花 API 成本;否則用既有快取)")
    ap.add_argument("--rerun-days", type=int, default=90, help="--rerun 時只對最近 N 交易日的候選跑 LLM(控成本)")
    ap.add_argument("--use-center", action="store_true", help="tilt 改用 center%%xconf(需 --rerun 的完整快取)")
    args = ap.parse_args()
    tilts = [float(x) for x in args.tilt.split(",")]

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info(f"載入特徵({len(codes)} 支)...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    opens, closes = {}, {}
    for c in codes + ["0050"]:
        o = oh(c); opens[c] = {d: o[d]["open"] for d in o}; closes[c] = {d: o[d]["close"] for d in o}
    alld = sorted({d for c in codes for d in closes.get(c, {}) if d <= END})
    sig_days = alld[-max(n for _, n in WINDOWS):]
    logger.info(f"訊號範圍 {sig_days[0]} ~ {sig_days[-1]}({len(sig_days)} 日)")

    logger.info("反彈訊號 + 漲停日...")
    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); cl_list = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    cands, regime_bull = build_candidates(codes, names, feats, twii_feat, reb_cache, turn_pct, sig_days)
    n_cand = sum(len(v) for v in cands.values())
    logger.info(f"候選 {n_cand} 筆(每日 top-{TOPN})")

    # ── LLM 分數 ──
    pairs = list({(d, c) for d in sig_days for _, c in cands[d]})
    if args.rerun and args.provider:
        rerun_window = set(sig_days[-args.rerun_days:])
        pairs_run = [p for p in pairs if p[0] in rerun_window]
        logger.info(f"--rerun-days={args.rerun_days}:LLM 只跑 {len(pairs_run)}/{len(pairs)} 候選(最近 {args.rerun_days} 日)")
        full = rerun_llm(pairs_run, names, args.provider, args.model)
        llm = {k: (v[0], v[1], v[2]) for k, v in full.items() if v}    # (verdict,bull,bear)
        center = {k: (v[3], v[4]) for k, v in full.items() if v}        # (center%,conf)
    else:
        llm = load_llm_cache(); center = {}
        logger.info(f"用既有快取 bull-bear:{len(llm)} 筆有評分(候選去重 {len(pairs)})")

    cover = sum(1 for p in pairs if p in llm)
    logger.info(f"LLM 覆蓋 {cover}/{len(pairs)} = {cover/max(1,len(pairs))*100:.0f}%(沒覆蓋的 tilt=1 中性)")

    # ── 0050 基準(各窗 + 各 regime)──
    def bench_for(day_set):
        dd = sorted(d for d in day_set if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    bench_win = {wl: bench_for(set(sig_days[-n:])) for wl, n in WINDOWS}
    regime_days = {rn: set(d for d in sig_days if s <= d <= e) for rn, (s, e) in REGIMES.items()}
    bench_reg = {rn: bench_for(dd) for rn, dd in regime_days.items()}

    def rows_for_tilt(tilt, day_set, use_center):
        rw = []
        for d in sig_days:
            if d not in day_set:
                continue
            for v, c in cands[d]:
                e = v / 100.0
                if tilt == 0.0:
                    rw.append((d, c, e)); continue
                if use_center and (d, c) in center:
                    cen, conf = center[(d, c)]
                    # center%(預期漲幅) x conf 當看多強度;>0 加倉、<0 縮(以 0 為中心)
                    mult = max(0.0, 1.0 + tilt * math.tanh((cen * conf) / 1.5))
                    rw.append((d, c, e * mult)); continue
                if (d, c) in llm:
                    _, bull, bear = llm[(d, c)]
                    rw.append((d, c, e * tilt_mult(bull, bear, tilt)))
                else:
                    rw.append((d, c, e))   # 無 LLM → 中性
        return rw

    def run_sim(rw):
        return sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)

    # baseline (tilt=0) + 各 tilt
    configs = [("baseline(無LLM)", 0.0)] + [(f"tilt×{t}", t) for t in tilts]
    if center:
        configs += [(f"center×{t}", t) for t in tilts]   # 若有完整快取,加 center 變體

    res = {}   # (label, window_or_regime) -> sim dict
    for label, tilt in configs:
        use_center = label.startswith("center")
        for wl, n in WINDOWS:
            res[(label, wl)] = run_sim(rows_for_tilt(tilt, set(sig_days[-n:]), use_center))
        for rn, dd in regime_days.items():
            res[(label, rn)] = run_sim(rows_for_tilt(tilt, dd, use_center))
        logger.info(f"{label} 完成")

    # ── 報告(v15 多窗格式)──
    key = args.provider or "cache_bullbear"
    if args.model:
        key += "_" + args.model.replace("/", "-")
    RPT = ROOT / "reports" / f"news_llm_{key}.md"
    wlabels = [wl for wl, _ in WINDOWS]
    rlabels = list(REGIMES.keys())
    allcols = wlabels + rlabels

    def alpha(label, col):
        r = res.get((label, col))
        bench = bench_win.get(col, bench_reg.get(col))
        return (r["ret"] - bench) if (r and bench is not None) else None

    L = [f"# News LLM 正向選股(tilt 非 veto)— {key}\n",
         f"> 候選 H+反彈雙引擎每日 top-{TOPN}｜⑤買收賣開執行｜結束 {END}｜{len(codes)} 檔｜還原價\n",
         f"> tilt = edge × (1 + t·tanh((bull−bear)/3));baseline=t0 純引擎｜資金 DCA 15000+1000/日上限5萬\n",
         f"> 成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到｜ALPHA = 策略 − 同資金 DCA 0050\n",
         f"> LLM 覆蓋 {cover}/{len(pairs)}({cover/max(1,len(pairs))*100:.0f}%);新聞源 finmind 標題(2022+),"
         "近期 google｜⚠️ 第一輪允許輕微洩漏(新聞時戳可能含當日盤後)\n",
         "> 0050 基準: " + " ".join(f"{wl}{bench_win[wl]:+.0f}%" for wl in wlabels) + " | "
         + " ".join(f"{rn}{bench_reg[rn]:+.0f}%" for rn in rlabels) + "\n",
         "## ALPHA %(扣 0050 beta)\n",
         "| 變體 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手(2年) | 持股(2年) | 曝險(2年) |",
         "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|"]
    for label, _ in configs:
        cells = []
        avals = []
        for col in allcols:
            a = alpha(label, col)
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None:
                avals.append(a)
        worst = f"{min(avals):+.0f}" if avals else "—"
        mean = f"{sum(avals)/len(avals):+.0f}" if avals else "—"
        r2y = res.get((label, "2年"))
        turn = f"{r2y['turn']:.0f}x" if r2y else "—"
        pos = f"{r2y.get('avg_pos', 0):.1f}" if r2y else "—"
        expo = f"{r2y.get('avg_expo', 0)*100:.0f}%" if r2y else "—"
        L.append(f"| {label} | " + " | ".join(cells) + f" | **{worst}** | {mean} | {turn} | {pos} | {expo} |")

    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    for label, _ in configs:
        cells = [f"{res[(label,col)]['ret']:+.0f}" if res.get((label, col)) else "—" for col in allcols]
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    # uplift vs baseline
    L += ["", "## Uplift = tilt alpha − baseline alpha(pp,逐窗)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    base_a = {col: alpha("baseline(無LLM)", col) for col in allcols}
    for label, _ in configs:
        if label.startswith("baseline"):
            continue
        cells = []
        for col in allcols:
            a = alpha(label, col); b = base_a.get(col)
            cells.append(f"{a-b:+.0f}" if (a is not None and b is not None) else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += ["", "## 判讀\n",
          "- tilt 各窗 uplift 全正且隨 tilt 單調上升 → 正向選股訊號真實存在。",
          "- uplift 接近 0 或無單調 → bull-bear 沒選股力(可能 finmind 標題太弱,要 center%/全文)。",
          "- 注意曝險欄:tilt 與 baseline 曝險接近才是公平比較(否則是高曝險虛假贏)。"]
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    # console 摘要
    for label, _ in configs:
        a2 = alpha(label, "2年"); a60 = alpha(label, "60天"); a90 = alpha(label, "90天")
        logger.success(f"{label}: 60天α{a60:+.0f} 90天α{a90:+.0f} 2年α{a2:+.0f}"
                       if None not in (a2, a60, a90) else f"{label}: (部分窗無資料)")


if __name__ == "__main__":
    main()
