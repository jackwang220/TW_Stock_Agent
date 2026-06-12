"""測 LLM 否決層:從雙引擎(H雙引擎B純切)實際選股裡,挑大賺/大賠各一批,跑 LLM 辯論看 verdict。
問:① 賺錢的被 REJECT 多不多(亂擋財路?)② 賠錢的被 REJECT 多不多(幫擋刀?)。用 FinMind 新聞(2021-2025有料)。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import Counter
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
cpv = importlib.util.module_from_spec(importlib.util.spec_from_file_location("cpv", ROOT/"scripts/crossperiod_validate.py"))
importlib.util.spec_from_file_location("cpv", ROOT/"scripts/crossperiod_validate.py").loader.exec_module(cpv)
features, _factors = v5.features, v5._factors
_finmind_stock = cpv._finmind_stock

START, END = "2021-01-01", "2025-11-30"     # 避開 2026 FinMind 新聞洞
WIN_TH, N_EACH = 0.08, 40

def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("特徵/反彈/成交值...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    v5._OH = OH
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    reb_cache = {}
    for c in codes:
        ds = sorted(d for d in OH[c] if d <= END); cl = []; m = {}
        for d in ds:
            cl.append(OH[c][d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
        reb_cache[c] = m
    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20") and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    # 每日雙引擎 top-3 + 5日後報酬
    picks = []   # (date, code, fwd5)
    for d in alld:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        day = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5)); rb = reb_cache.get(c, {}).get(d, 0.0)
            sc = hh if bull else rb
            if sc > 0: day.append((sc, c))
        day.sort(reverse=True)
        for _, c in day[:3]:
            ds = sorted(x for x in OH[c] if x >= d)
            if len(ds) > 5:
                fwd = OH[c][ds[5]]["close"]/OH[c][ds[0]]["close"]-1
                picks.append((d, c, fwd))
    winners = [p for p in picks if p[2] > WIN_TH]
    losers = [p for p in picks if p[2] < -WIN_TH]
    def sample(lst):
        lst = sorted(lst, key=lambda x: x[0]); step = max(1, len(lst)//N_EACH)
        return lst[::step][:N_EACH]
    win_s, los_s = sample(winners), sample(losers)
    logger.info(f"雙引擎選股 {len(picks)} 筆:大賺{len(winners)} 大賠{len(losers)} → 抽樣各 {len(win_s)}/{len(los_s)} 跑LLM")

    def debate(p):
        d, c, fwd = p
        try:
            stock = _finmind_stock(c, names.get(c, c), d)
            if stock is None: return None
            r = run_debate(stock, [], historical_date=date.fromisoformat(d))
            return (d, c, fwd, r.verdict, r.bear_score)
        except Exception as e:
            logger.warning(f"  {c}@{d} 失敗 {str(e)[:40]}"); return None
    with ThreadPoolExecutor(max_workers=8) as ex:
        win_r = [x for x in ex.map(debate, win_s) if x]
        los_r = [x for x in ex.map(debate, los_s) if x]

    def summ(rows, label):
        vc = Counter(r[3] for r in rows); n = len(rows)
        rej = vc.get("REJECT", 0); passc = vc.get("PASS", 0)
        avg_all = sum(r[2] for r in rows)/n*100 if n else 0
        kept = [r for r in rows if r[3] != "REJECT"]
        avg_kept = sum(r[2] for r in kept)/len(kept)*100 if kept else 0
        return (label, n, passc, vc.get("WARN",0), rej, rej/n*100 if n else 0, avg_all, avg_kept, len(kept))

    L = ["# LLM 否決層測試:該擋的擋了沒、不該擋的亂擋沒(雙引擎H純切選股,FinMind新聞,2021-2025)\n",
         f"> 5日後 >+{WIN_TH*100:.0f}%=大賺 / <-{WIN_TH*100:.0f}%=大賠｜verdict: bear≥7→REJECT\n",
         "## 結果\n", "| 組別 | 樣本 | PASS | WARN | REJECT | **REJECT率** | 原均5日報酬 | 留下的(非REJECT)均報酬 |",
         "|---|---|---|---|---|---|---|---|"]
    for rows, lab in [(win_r, "💰 大賺的"), (los_r, "💸 大賠的")]:
        s = summ(rows, lab)
        L.append(f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} | **{s[5]:.0f}%** | {s[6]:+.1f}% | {s[7]:+.1f}%(留{s[8]}) |")
    L += ["", "## 判讀",
          "- 大賺組 REJECT率 **低** = LLM 沒亂擋財路(好);**高** = LLM 把你的賺錢股擋掉(壞)。",
          "- 大賠組 REJECT率 **高** = LLM 幫你擋刀(好);**低** = LLM 沒用。",
          "- 若『留下的均報酬』比『原均報酬』高 → LLM 過濾淨效果是正的。"]
    (ROOT/"reports"/"exp_llm_veto.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/exp_llm_veto.md")
    for rows, lab in [(win_r, "大賺"), (los_r, "大賠")]:
        s = summ(rows, lab); logger.success(f"{lab}: REJECT率 {s[5]:.0f}% | 原{s[6]:+.1f}%→留{s[7]:+.1f}%")

if __name__ == "__main__":
    main()