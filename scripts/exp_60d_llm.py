"""60D 三組對比:①無LLM ②LLM純拒絕(veto) ③LLM調配比(size)。
策略=H+反彈雙引擎;辯論=正/反/裁判(run_debate);新聞=google(只有標題,看方向用)。
真實成交:隔日開盤價+滑價+漲停買不到+手續費(v6.sim_real)。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.debate.bear import run_debate

v5 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py"))
importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py").loader.exec_module(v5)
v6 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v6", ROOT/"scripts/exp_step1_v6.py"))
importlib.util.spec_from_file_location("v6", ROOT/"scripts/exp_step1_v6.py").loader.exec_module(v6)
cpv = importlib.util.module_from_spec(importlib.util.spec_from_file_location("cpv", ROOT/"scripts/crossperiod_validate.py"))
importlib.util.spec_from_file_location("cpv", ROOT/"scripts/crossperiod_validate.py").loader.exec_module(cpv)
features, _factors = v5.features, v5._factors
_finmind_stock = cpv._finmind_stock

START, END, NDAYS = "2025-06-01", "2026-06-08", 60
TOPN = 4

def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("特徵...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    v5._OH = OH
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    sig_days = alld[-NDAYS:]
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in list(codes)+["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in list(codes)+["0050"]}
    reb_cache, limitup = {}, {}
    for c in codes:
        ds = sorted(OH[c]); cl = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl.append(OH[c][d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
            if j > 0 and OH[c][ds[j-1]]["close"] > 0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1 >= 0.095: s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20") and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in sig_days}

    # 每日雙引擎 top-N 候選
    cands = {}
    for d in sig_days:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        sc = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            v = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5)) if bull else reb_cache.get(c, {}).get(d, 0.0)
            if v > 0: sc.append((v, c))
        sc.sort(reverse=True); cands[d] = sc[:TOPN]

    # 辯論(每個候選 d,c)+ 快取(可續跑)
    pairs = list({(d, c) for d in sig_days for _, c in cands[d]})
    CACHE = DATA_DIR / "_60dllm_debate.json"
    res = {}
    if CACHE.exists():
        for k, v in json.loads(CACHE.read_text(encoding="utf-8")).items():
            d, c = k.split("_"); res[(d, c)] = tuple(v) if v else None
    todo = [p for p in pairs if p not in res]
    logger.info(f"60D候選 {sum(len(v) for v in cands.values())} 筆,去重 {len(pairs)},快取已有 {len(pairs)-len(todo)},待辯 {len(todo)}(google新聞)")
    def deb(p):
        d, c = p
        try:
            stock = _finmind_stock(c, names.get(c, c), d)
            if stock is None: return p, None
            r = run_debate(stock, [], historical_date=date.fromisoformat(d))
            return p, (r.verdict, r.bull_score, r.bear_score)
        except Exception as e:
            logger.warning(f"  {c}@{d} {str(e)[:40]}"); return p, None
    RPT = ROOT / "reports" / "exp_60d_llm.md"
    def write_progress(done, total):
        graded = [v for v in res.values() if v]
        rej = sum(1 for v in graded if v[0] == "REJECT")
        pas = sum(1 for v in graded if v[0] == "PASS")
        recent = [(f"{d}", c, names.get(c, c)[:4], v[0], v[1], v[2]) for (d, c), v in list(res.items())[-8:] if v]
        P = [f"# 60D 三組LLM測試 — 辯論進度(邊跑邊更新)\n",
             f"> 候選 {len(pairs)} 筆｜**已辯論 {len(graded)}/{len(pairs)}**｜還在跑…\n",
             f"## 目前辯論統計\n- PASS {pas}｜REJECT {rej}｜REJECT率 {rej/max(1,len(graded))*100:.0f}%\n",
             "## 最近幾筆(日期 代號 名 verdict bull bear)\n", "| 日期 | 股 | verdict | bull | bear |", "|---|---|---|---|---|"]
        for dd, c, nm, vd, bl, be in recent:
            P.append(f"| {dd} | {c} {nm} | {vd} | {bl} | {be} |")
        P.append("\n*(三組報酬對比要等全部辯論完才算)*")
        RPT.write_text("\n".join(P), encoding="utf-8")
    done = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        for p, v in ex.map(deb, todo):
            res[p] = v; done += 1
            if done % 10 == 0:
                CACHE.write_text(json.dumps({f"{d}_{c}": list(val) if val else None for (d, c), val in res.items()}, ensure_ascii=False), encoding="utf-8")
                write_progress(done, len(todo))
                logger.info(f"  辯論 {done}/{len(todo)}")
    CACHE.write_text(json.dumps({f"{d}_{c}": list(val) if val else None for (d, c), val in res.items()}, ensure_ascii=False), encoding="utf-8")

    def size_mult(bull, bear):
        return max(0.7, min(1.3, 1 + 0.3*math.tanh((bull-bear)/3.0)))
    rows = {"①無LLM": [], "②LLM純拒絕": [], "③LLM調配比": []}
    for d in sig_days:
        for v, c in cands[d]:
            e = v/100
            rows["①無LLM"].append((d, c, e))
            verdict = res.get((d, c))
            if verdict is None:
                rows["②LLM純拒絕"].append((d, c, e)); rows["③LLM調配比"].append((d, c, e)); continue
            vd, bull, bear = verdict
            if vd != "REJECT": rows["②LLM純拒絕"].append((d, c, e))
            rows["③LLM調配比"].append((d, c, e*size_mult(bull, bear)))

    bench = v6.bench_0050(opens["0050"], closes["0050"], sig_days)
    L = ["# 60D 三組對比:無LLM / LLM純拒絕 / LLM調配比(H+反彈雙引擎,google新聞,真實成交)\n",
         f"> 窗口 {sig_days[0]}~{sig_days[-1]}｜0050基準 {bench:+.1f}%｜隔日開盤成交+滑價+漲停買不到+手續費\n",
         "> ⚠️ google只有標題、clickbait污染 → 只看方向,不能定案\n",
         "## 結果\n", "| 組別 | 本金報酬率 | ALPHA(扣0050) | Sharpe | MDD | 換手x |", "|---|---|---|---|---|---|"]
    out = {}
    for lab, rw in rows.items():
        r = v6.sim_real(rw, opens, closes, limitup); out[lab] = r
        if r:
            L.append(f"| {lab} | {r['ret']:+.1f}% | **{r['ret']-bench:+.1f}%** | {r['sharpe']:.2f} | -{r['mdd']:,.0f} | {r['turn']:.0f}x |")
    rej = sum(1 for v in res.values() if v and v[0] == "REJECT")
    L += ["", f"## 辯論統計\n- 候選去重 {len(pairs)} 筆,REJECT {rej} 筆({rej/max(1,len(pairs))*100:.0f}%)",
          "## 判讀\n- veto 比無LLM 高 → LLM 排雷有用;低 → 砍到財路。",
          "- 調配比 比無LLM 高 → LLM 微調倉位有用。", "- 三組差很小 → google標題下 LLM 沒加值(符合預期,要 live 全文才知)。"]
    (ROOT/"reports"/"exp_60d_llm.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/exp_60d_llm.md")
    for lab in rows:
        r = out.get(lab)
        if r: logger.success(f"{lab}: 報酬{r['ret']:+.1f}% alpha{r['ret']-bench:+.1f}%")

if __name__ == "__main__":
    main()