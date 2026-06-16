"""Step1 v8:regime-aware 動能+反彈雙引擎。動能=D突破底,反彈=驗證過的反彈分,
用 0050 vs 20MA 當開關動態調權重(多頭偏動能/空頭偏反彈)。跨 5 regime 測 alpha,目標:都不大輸+2022保命。
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
ec = importlib.util.module_from_spec(importlib.util.spec_from_file_location("ec", ROOT/"scripts/exp_60d_entry_compare.py"))
importlib.util.spec_from_file_location("ec", ROOT/"scripts/exp_60d_entry_compare.py").loader.exec_module(ec)
sim5 = ec.sim_buyclose_sellopen   # ⑤:收盤買 + 開盤賣 + 開盤補買
features, _factors, clamp = v5.features, v5._factors, v5.clamp

START = "2021-01-01"
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
# 雙引擎權重:(多頭 mom,多頭 reb,空頭 mom,空頭 reb)
DUAL = {"雙引擎A溫和": (1.0,0.7,0.6,1.3), "雙引擎B激進": (1.0,0.4,0.3,1.5), "雙引擎C均衡": (1.0,0.9,0.7,1.2)}

def mom_break(ff):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (t+rs+vo+ri+ma+br)/6*100

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
    logger.info("反彈/漲停...")
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
    regime_bull = {d: (twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20") and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    def rows_dual(w):
        wmb, wrb, wms, wrs = w; out = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                mm = mom_break(_factors(f[d], ir)); rb = reb_cache.get(c, {}).get(d, 0.0)
                sc = max(mm*wmb, rb*wrb) if bull else max(mm*wms, rb*wrs)
                if sc > 0: out.append((d, c, sc/100))
        return out
    # 對照組
    def rows_cmp(kind):
        out = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                rb = reb_cache.get(c, {}).get(d, 0.0); mm = mom_break(_factors(f[d], ir))
                if kind == "純D動能": sc = mm
                elif kind == "反彈only": sc = rb
                elif kind == "D+regime": sc = max(mm*(1.0 if bull else 0.7), rb)
                else: sc = 0
                if sc > 0: out.append((d, c, sc/100))
        return out

    bench = {lab: v6.bench_0050(opens["0050"], closes["0050"], [d for d in alld if s <= d <= e]) for lab, s, e in REGIMES}
    allv = list(DUAL) + ["D+regime","純D動能","反彈only"]
    res = {}
    for vn in allv:
        rows = rows_dual(DUAL[vn]) if vn in DUAL else rows_cmp(vn)
        for lab, s, e in REGIMES:
            res[(vn, lab)] = sim5([r for r in rows if s <= r[0] <= e], opens, closes, limitup)
        logger.info(f"{vn} 完成")

    def alpha(vn, lab):
        r = res.get((vn, lab)); return (r["ret"]-bench[lab]) if r else None
    L = ["# Step1 v8 雙引擎(動能+反彈,regime切換)— 跨5regime ALPHA(⑤執行:收盤買+開盤賣買,還原價)\n",
         "> 0050各regime:" + " ".join(f"{lab}{bench[lab]:+.0f}%" for lab,_,_ in REGIMES) +
         "｜雙引擎權重=(多頭mom,reb / 空頭mom,reb)\n",
         "## ALPHA %(正=贏大盤;看『最差regime』別太負 + 2022空頭要保命)\n",
         "| 變體 | 2021 | 2022空頭 | 2023 | 2024-25 | 2025下-26 | 最差 |",
         "|---|---|---|---|---|---|---|"]
    for vn in allv:
        vals = [alpha(vn, lab) for lab,_,_ in REGIMES]
        valid = [v for v in vals if v is not None]
        cells = " | ".join(f"{v:+.0f}" if v is not None else "—" for v in vals)
        worst = f"{min(valid):+.0f}" if valid else "—"
        L.append(f"| {vn} | {cells} | **{worst}** |")
    L += ["", "## 原始報酬 %\n", "| 變體 | 2021 | 2022空頭 | 2023 | 2024-25 | 2025下-26 |", "|---|---|---|---|---|---|",
          "| 0050大盤 | " + " | ".join(f"{bench[lab]:+.0f}" for lab,_,_ in REGIMES) + " |"]
    for vn in allv:
        L.append(f"| {vn} | " + " | ".join(f"{res[(vn,lab)]['ret']:+.0f}" if res.get((vn,lab)) else "—" for lab,_,_ in REGIMES) + " |")
    (ROOT/"reports"/"exp_step1_v8.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/exp_step1_v8.md")

if __name__ == "__main__":
    main()