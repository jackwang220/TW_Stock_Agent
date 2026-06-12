"""Step1 v7 多 regime ALPHA(含 2022 真實空頭),真實版隔日開盤成交+滑價+漲停+手續費+降換手。
看防禦型策略(G大盤濾網/反彈)是不是在熊市靠避跌賺回正 alpha。清洗後乾淨價。
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
features = v5.features

START = "2021-01-01"
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
SHOW = ["H","G","D+G","C+G","反彈only","D","B","A"]

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("全史特徵(2021~)...")
    # 用全史 OHLCV
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050", start=START)
    v5._OH = OH; features.__globals__["_OH"] = OH   # 讓 features/oh 共用全史
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c]})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes}
    opens["0050"] = {d: OH["0050"][d]["open"] for d in OH["0050"]}
    closes["0050"] = {d: OH["0050"][d]["close"] for d in OH["0050"]}
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

    bench = {lab: v6.bench_0050(opens["0050"], closes["0050"], [d for d in alld if s <= d <= e]) for lab, s, e in REGIMES}
    res = {}
    for vn in SHOW:
        if vn == "反彈only":
            rows_full = [(d, c, reb_cache[c][d]/100) for c in codes for d in reb_cache[c]]
        else:
            rows_full = v5.build_rows(codes, names, feats, twii_feat, reb_cache, turn_pct, v5.VAR[vn], alld)
        for lab, s, e in REGIMES:
            rw = [r for r in rows_full if s <= r[0] <= e]
            res[(vn, lab)] = v6.sim_real(rw, opens, closes, limitup)
        logger.info(f"{vn} 完成")

    def alpha(vn, lab):
        r = res.get((vn, lab)); return (r["ret"]-bench[lab]) if r else None
    L = ["# Step1 v7 多 regime ALPHA(含2022空頭,真實成交,清洗後乾淨價)\n",
         "> 看防禦策略是否在熊市(2022)靠避跌贏 buy&hold｜0050各regime報酬:" +
         " ".join(f"{lab}{bench[lab]:+.0f}%" for lab,_,_ in REGIMES) + "\n",
         "## ALPHA %(策略 - 0050;正=贏大盤)\n",
         "| 變體 | " + " | ".join(lab for lab,_,_ in REGIMES) + " |",
         "|---|" + "---|"*len(REGIMES)]
    for vn in SHOW:
        L.append(f"| {vn} | " + " | ".join(f"{alpha(vn,lab):+.0f}" if alpha(vn,lab) is not None else "—" for lab,_,_ in REGIMES) + " |")
    L += ["", "## 參考:原始報酬 %\n", "| 變體 | " + " | ".join(lab for lab,_,_ in REGIMES) + " |", "|---|"+"---|"*len(REGIMES)]
    L.append(f"| 0050大盤 | " + " | ".join(f"{bench[lab]:+.0f}" for lab,_,_ in REGIMES) + " |")
    for vn in SHOW:
        L.append(f"| {vn} | " + " | ".join(f"{res[(vn,lab)]['ret']:+.0f}" if res.get((vn,lab)) else "—" for lab,_,_ in REGIMES) + " |")
    (ROOT/"reports"/"exp_step1_v7.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/exp_step1_v7.md")

if __name__ == "__main__":
    main()