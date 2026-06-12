"""Step1 v9:H+反彈 雙引擎(空頭用反彈保命、多頭用H爆發,regime切換)。把那張表兩邊的優點都拿到。
動能引擎換成 H(成交值,多頭最強),反彈引擎保命。跨5regime測alpha。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

v5 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py"))
importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py").loader.exec_module(v5)
v6 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v6", ROOT/"scripts/exp_step1_v6.py"))
importlib.util.spec_from_file_location("v6", ROOT/"scripts/exp_step1_v6.py").loader.exec_module(v6)
features, _factors = v5.features, v5._factors

START = "2021-01-01"
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
# (多頭 H, 多頭 reb, 空頭 H, 空頭 reb)
DUAL = {"H雙引擎A": (1.0,0.4,0.2,1.4), "H雙引擎B純切": (1.0,0.0,0.0,1.5), "H雙引擎C": (1.0,0.6,0.3,1.3)}

def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("全史特徵...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    v5._OH = OH
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c]})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in list(codes)+["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in list(codes)+["0050"]}
    logger.info("反彈/漲停/成交值...")
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
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20") and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    def rows_dual(w):
        whb, wrb, whs, wrs = w; out = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5)); rb = reb_cache.get(c, {}).get(d, 0.0)
                sc = max(hh*whb, rb*wrb) if bull else max(hh*whs, rb*wrs)
                if sc > 0: out.append((d, c, sc/100))
        return out
    def rows_pure(kind):
        out = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20")
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                rb = reb_cache.get(c, {}).get(d, 0.0)
                sc = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5)) if kind == "H純" else rb
                if sc > 0: out.append((d, c, sc/100))
        return out

    bench = {lab: v6.bench_0050(opens["0050"], closes["0050"], [d for d in alld if s <= d <= e]) for lab, s, e in REGIMES}
    allv = list(DUAL) + ["H純", "反彈純"]
    res = {}
    for vn in allv:
        rows = rows_dual(DUAL[vn]) if vn in DUAL else rows_pure(vn)
        for lab, s, e in REGIMES:
            res[(vn, lab)] = v6.sim_real([r for r in rows if s <= r[0] <= e], opens, closes, limitup)
        logger.info(f"{vn} 完成")

    def alpha(vn, lab):
        r = res.get((vn, lab)); return (r["ret"]-bench[lab]) if r else None
    L = ["# Step1 v9:H+反彈雙引擎(空頭反彈保命/多頭H爆發)— 跨5regime ALPHA(清洗價/真實成交)\n",
         "> 0050:" + " ".join(f"{lab}{bench[lab]:+.0f}%" for lab,_,_ in REGIMES) + "｜權重=(多頭H,reb/空頭H,reb)\n",
         "## ALPHA %(正=贏大盤;要『最差別太負』+『2022保命』+『多頭噴』)\n",
         "| 變體 | 2021 | 2022空頭 | 2023 | 2024-25 | 2025下-26 | 最差 | 平均 |",
         "|---|---|---|---|---|---|---|---|"]
    for vn in allv:
        vals = [alpha(vn, lab) for lab,_,_ in REGIMES]
        L.append(f"| {vn} | " + " | ".join(f"{v:+.0f}" for v in vals) + f" | **{min(vals):+.0f}** | {sum(vals)/len(vals):+.0f} |")
    L += ["", "## 原始報酬 %\n", "| 變體 | 2021 | 2022空頭 | 2023 | 2024-25 | 2025下-26 |", "|---|---|---|---|---|---|",
          "| 0050大盤 | " + " | ".join(f"{bench[lab]:+.0f}" for lab,_,_ in REGIMES) + " |"]
    for vn in allv:
        L.append(f"| {vn} | " + " | ".join(f"{res[(vn,lab)]['ret']:+.0f}" if res.get((vn,lab)) else "—" for lab,_,_ in REGIMES) + " |")
    (ROOT/"reports"/"exp_step1_v9.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/exp_step1_v9.md")

if __name__ == "__main__":
    main()