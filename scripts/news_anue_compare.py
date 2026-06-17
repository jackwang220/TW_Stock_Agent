"""Anue 全文 vs 標題-only 的 LLM 正向選股對照。

決策閘:有了「內文」,LLM 選股能不能出現「只有標題」時沒有的真 alpha?
- 候選/回測/基準 全部重用 news_llm_select.py 的機制(同 H+反彈雙引擎 top-4、⑤買收賣開、同資金 DCA、扣成本)。
- 新聞改用鉅亨 Anue(scripts/news_anue.py),leak-safe(articles_for 以 as_of 過濾 publishAt)。
- 兩個變體:A=只餵標題;B=餵標題+內文(截斷控 token)。各自打 gpt-4o-mini 取 bull/bear,當 tilt 乘進 edge。
- LLM 分數快取到 data/_anue_compare_scores.json(可續跑)。
- baseline=不加 LLM。輸出 reports/news_anue_compare.md。

用法: uv run python scripts/news_anue_compare.py [--llm-days 45] [--tilt 0.1,0.3,0.6,1.0] [--max-body 1500]
"""
from __future__ import annotations
import sys, os, json, argparse, math, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal

import importlib.util as _u
def _load(name, path):
    spec = _u.spec_from_file_location(name, path); m = _u.module_from_spec(spec)
    spec.loader.exec_module(m); return m

nls = _load("nls", ROOT / "scripts/news_llm_select.py")   # 重用候選/回測/基準
na  = _load("na",  ROOT / "scripts/news_anue.py")          # Anue 抓取器

features, _factors, oh = nls.features, nls._factors, nls.oh
build_candidates, tilt_mult = nls.build_candidates, nls.tilt_mult
sim5, v6 = nls.sim5, nls.v6
WINDOWS, REGIMES, TOPN, END = nls.WINDOWS, nls.REGIMES, nls.TOPN, nls.END

SCORE_CACHE = DATA_DIR / "_anue_compare_scores.json"   # {f"{d}_{c}_{variant}": {bull,bear,n}}


def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYS = ("你是嚴謹的台股短線分析師。根據提供的個股近期新聞,判斷該股『隔日』偏多/偏空的力道。"
       "只看新聞透露的前瞻催化劑或利空,不要用你不知道的資訊。"
       "輸出 JSON:{\"bull\":0-10 看多力道,\"bear\":0-10 看空力道}。沒有實質新聞就兩者都給 1。")


def score_one(client, model, code, name, d, news_text):
    user = f"個股:{code} {name}\n決策日:{d}(只用此日之前的新聞)\n近期新聞:\n{news_text or '(無)'}"
    try:
        r = client.chat.completions.create(
            model=model, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            max_tokens=120,
        )
        j = json.loads(r.choices[0].message.content)
        return {"bull": float(j.get("bull", 1)), "bear": float(j.get("bear", 1))}
    except Exception as e:
        logger.warning(f"  score {code}@{d}: {str(e)[:60]}")
        return None


def build_news(code, d, max_body):
    """回 (title_only_text, title_body_text)。leak-safe:articles_for 以 as_of=d 過濾。"""
    try:
        arts = na.articles_for(code, d, window_days=5)
    except Exception as e:
        logger.warning(f"  anue {code}@{d}: {str(e)[:50]}"); return "", ""
    arts = arts[:6]
    titles = "\n".join(f"- {a.get('title','')}" for a in arts if a.get("title"))
    blocks = []
    for a in arts:
        t = a.get("title", ""); b = (a.get("body", "") or "")[:max_body]
        blocks.append(f"標題:{t}\n內文:{b}" if b else f"標題:{t}")
    return titles, "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-days", type=int, default=45, help="只對最近 N 交易日候選跑 LLM(控成本)")
    ap.add_argument("--tilt", default="0.1,0.3,0.6,1.0")
    ap.add_argument("--max-body", type=int, default=1500)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=3)
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
        o = oh(c); ds = sorted(d for d in o if d <= END); cl = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl.append(o[d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"] * 100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    cands, _ = build_candidates(codes, names, feats, twii_feat, reb_cache, turn_pct, sig_days)
    n_cand = sum(len(v) for v in cands.values())
    logger.info(f"候選 {n_cand} 筆(每日 top-{TOPN})")

    # LLM 只跑最近 N 日候選
    llm_days = set(sig_days[-args.llm_days:])
    pairs = sorted({(d, c) for d in sig_days if d in llm_days for _, c in cands[d]})
    logger.info(f"LLM 候選 {len(pairs)} 筆(最近 {args.llm_days} 日)× 2 變體")

    # 先 prefetch 這段日期的清單(只 list 不 body,body 由 articles_for 按需抓候選股)
    logger.info("prefetch Anue 清單...")
    try:
        na.prefetch_range(sorted(llm_days)[0], sorted(llm_days)[-1], fetch_bodies=False)
    except Exception as e:
        logger.warning(f"prefetch: {str(e)[:80]}")

    # 讀分數快取
    scores = {}
    if SCORE_CACHE.exists():
        scores = json.loads(SCORE_CACHE.read_text(encoding="utf-8"))
    client = _openai_client()
    throttle = float(os.environ.get("NEWS_LLM_THROTTLE", "0.2"))

    todo = []
    news_cache = {}  # (d,c) -> (title_text, body_text)
    for (d, c) in pairs:
        t, b = build_news(c, d, args.max_body)
        news_cache[(d, c)] = (t, b)
        for variant, txt in (("title", t), ("body", b)):
            k = f"{d}_{c}_{variant}"
            if k not in scores:
                todo.append((d, c, variant, txt))
    logger.info(f"待打 LLM {len(todo)} 筆(快取已有 {len(pairs)*2 - len(todo)})")

    def _run(item):
        d, c, variant, txt = item
        if throttle: time.sleep(throttle)
        sc = score_one(client, args.model, c, names.get(c, c), d, txt)
        return f"{d}_{c}_{variant}", sc

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, sc in ex.map(_run, todo):
            scores[k] = sc; done += 1
            if done % 20 == 0:
                SCORE_CACHE.write_text(json.dumps(scores, ensure_ascii=False), encoding="utf-8")
                logger.info(f"  LLM {done}/{len(todo)}")
    SCORE_CACHE.write_text(json.dumps(scores, ensure_ascii=False), encoding="utf-8")

    cov_t = sum(1 for (d, c) in pairs if scores.get(f"{d}_{c}_title"))
    cov_b = sum(1 for (d, c) in pairs if scores.get(f"{d}_{c}_body"))
    logger.info(f"覆蓋 title {cov_t}/{len(pairs)}  body {cov_b}/{len(pairs)}")

    # ── 基準 ──
    def bench_for(day_set):
        dd = sorted(d for d in day_set if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    bench_win = {wl: bench_for(set(sig_days[-n:])) for wl, n in WINDOWS}
    regime_days = {rn: set(d for d in sig_days if s <= d <= e) for rn, (s, e) in REGIMES.items()}
    bench_reg = {rn: bench_for(dd) for rn, dd in regime_days.items()}

    def rows_for(tilt, variant, day_set):
        rw = []
        for d in sig_days:
            if d not in day_set: continue
            for v, c in cands[d]:
                e = v / 100.0
                sc = scores.get(f"{d}_{c}_{variant}") if (tilt and variant) else None
                if sc:
                    rw.append((d, c, e * tilt_mult(sc["bull"], sc["bear"], tilt)))
                else:
                    rw.append((d, c, e))   # baseline 或無分數 → 中性
        return rw

    def run_sim(rw):
        return sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)

    configs = [("baseline(無LLM)", 0.0, None)]
    for t in tilts:
        configs.append((f"標題only×{t}", t, "title"))
    for t in tilts:
        configs.append((f"標題+內文×{t}", t, "body"))

    res = {}
    for label, tilt, variant in configs:
        for wl, n in WINDOWS:
            res[(label, wl)] = run_sim(rows_for(tilt, variant, set(sig_days[-n:])))
        for rn, dd in regime_days.items():
            res[(label, rn)] = run_sim(rows_for(tilt, variant, dd))
        logger.info(f"{label} 完成")

    # ── 報告 ──
    wlabels = [wl for wl, _ in WINDOWS]; rlabels = list(REGIMES.keys()); allcols = wlabels + rlabels
    def alpha(label, col):
        r = res.get((label, col)); bench = bench_win.get(col, bench_reg.get(col))
        return (r["ret"] - bench) if (r and bench is not None) else None

    L = ["# Anue 全文 vs 標題-only — LLM 正向選股對照\n",
         f"> 候選 H+反彈雙引擎每日 top-{TOPN}｜⑤買收賣開｜結束 {END}｜{len(codes)} 檔｜模型 {args.model}\n",
         f"> 新聞源=鉅亨 Anue(全文,leak-safe publishAt<決策日)｜LLM 只跑最近 {args.llm_days} 交易日候選\n",
         f"> 覆蓋 title {cov_t}/{len(pairs)} body {cov_b}/{len(pairs)}｜tilt=edge×(1+t·tanh((bull−bear)/3))\n",
         f"> ALPHA=策略−同資金DCA0050｜成本 買0.14%/賣0.44%+滑0.1%+漲停買不到｜⚠️ 第一輪允許輕微洩漏\n",
         "> 0050 基準: " + " ".join(f"{wl}{bench_win[wl]:+.0f}%" for wl in wlabels) + " | "
         + " ".join(f"{rn}{bench_reg[rn]:+.0f}%" for rn in rlabels) + "\n",
         "## ALPHA %\n",
         "| 變體 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手(2年) | 持股 | 曝險 |",
         "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|"]
    for label, _, _ in configs:
        cells, av = [], []
        for col in allcols:
            a = alpha(label, col); cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None: av.append(a)
        r2 = res.get((label, "2年"))
        L.append(f"| {label} | " + " | ".join(cells)
                 + f" | **{min(av):+.0f}** | {sum(av)/len(av):+.0f} | "
                 + (f"{r2['turn']:.0f}x | {r2.get('avg_pos',0):.1f} | {r2.get('avg_expo',0)*100:.0f}% |" if r2 else "— | — | — |"))

    L += ["", "## Uplift = tilt alpha − baseline alpha(pp)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    base_a = {col: alpha("baseline(無LLM)", col) for col in allcols}
    for label, _, _ in configs:
        if label.startswith("baseline"): continue
        cells = []
        for col in allcols:
            a = alpha(label, col); b = base_a.get(col)
            cells.append(f"{a-b:+.0f}" if (a is not None and b is not None) else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += ["", "## 判讀(關鍵問題:內文有沒有讓 LLM 選股出現標題時沒有的真 uplift)\n",
          "- 若『標題+內文』各 tilt uplift 明顯 > 『標題only』且隨 tilt 單調 → 內文帶來真選股力,值得上 TEJ 買歷史全文。",
          "- 若兩者都接近 0 / 非單調 → 連內文都救不了,LLM 選股放生。",
          "- 注意曝險欄要可比(否則高曝險多頭虛假贏)。"]
    RPT = ROOT / "reports" / "news_anue_compare.md"
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")
    for label, _, _ in configs:
        a60, a90, a2 = alpha(label, "60天"), alpha(label, "90天"), alpha(label, "2年")
        if None not in (a60, a90, a2):
            logger.success(f"{label}: 60天α{a60:+.0f} 90天α{a90:+.0f} 2年α{a2:+.0f}")


if __name__ == "__main__":
    main()
