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
ec = importlib.util.module_from_spec(importlib.util.spec_from_file_location("ec", ROOT/"scripts/exp_60d_entry_compare.py"))
importlib.util.spec_from_file_location("ec", ROOT/"scripts/exp_60d_entry_compare.py").loader.exec_module(ec)
sim5 = ec.sim_buyclose_sellopen   # ⑤:收盤買 + 開盤賣 + 開盤補買(rebalance)
features, _factors = v5.features, v5._factors

START = "2021-01-01"
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
# 固定策略=H雙引擎B純切(多頭純H動能、空頭純反彈)。只改「多頭偵測」門檻,擋掉2022空頭假多頭。
DETECT = ["原版c>20MA", "加ret20>0", "加ma5>ma20", "加季線c>60MA", "全都要"]

def is_bull(tf, mode):
    """判斷今天大盤算不算多頭(可打H);否則=空頭(打反彈)。"""
    c, m20, m5, m60, r20 = tf.get("close"), tf.get("ma20"), tf.get("ma5"), tf.get("ma60"), tf.get("ret20")
    if not c or not m20 or math.isnan(m20):
        return False
    base = c > m20
    r20ok = r20 is not None and not math.isnan(r20)
    m5ok = m5 is not None and not math.isnan(m5)
    m60ok = m60 is not None and not math.isnan(m60)
    if mode == "原版c>20MA":   return base
    if mode == "加ret20>0":    return base and r20ok and r20 > 0
    if mode == "加ma5>ma20":   return base and m5ok and m5 > m20
    if mode == "加季線c>60MA": return base and m60ok and c > m60
    if mode == "全都要":       return base and r20ok and r20 > 0 and m5ok and m5 > m20 and m60ok and c > m60
    return base

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
    # 固定 H雙引擎B純切(多頭純H、空頭純反彈),只變「多頭偵測」門檻
    def rows_for(mode):
        out = []; bdays = 0
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20")
            bull = is_bull(twii_feat.get(d, {}), mode)
            if bull: bdays += 1
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                if bull:
                    sc = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))   # 多頭純H
                else:
                    sc = reb_cache.get(c, {}).get(d, 0.0) * 1.5                          # 空頭純反彈
                if sc > 0: out.append((d, c, sc/100))
        return out, bdays

    bench = {lab: v6.bench_0050(opens["0050"], closes["0050"], [d for d in alld if s <= d <= e]) for lab, s, e in REGIMES}
    res, bull_pct = {}, {}
    for mode in DETECT:
        rows, bdays = rows_for(mode)
        bull_pct[mode] = bdays / len(alld) * 100
        for lab, s, e in REGIMES:
            res[(mode, lab)] = sim5([r for r in rows if s <= r[0] <= e], opens, closes, limitup)
        logger.info(f"{mode} 完成(多頭日{bull_pct[mode]:.0f}%)")

    def alpha(mode, lab):
        r = res.get((mode, lab)); return (r["ret"]-bench[lab]) if r else None
    def compound(mode):
        p = 1.0
        for lab, _, _ in REGIMES:
            r = res.get((mode, lab))
            p *= (1 + (r["ret"] if r else 0)/100)
        return (p-1)*100
    bench_comp = (math.prod((1+bench[l]/100) for l, _, _ in REGIMES)-1)*100

    L = ["# 用 H 避開 2022 空頭 — 固定 H雙引擎(多頭純H/空頭純反彈) × 5種空頭偵測\n",
         "> ⑤執行(收盤買+開盤賣買)、還原價、真實成交｜只改『多頭』門檻,讓空頭反彈別騙它進場打H\n",
         f"> 0050各regime:" + " ".join(f"{lab}{bench[lab]:+.0f}%" for lab,_,_ in REGIMES) + f"｜0050全期複合 +{bench_comp:.0f}%\n",
         "## ALPHA %(正=贏大盤;重點:2022空頭別再 -26%)\n",
         "| 多頭偵測門檻 | 多頭日% | 2021 | **2022空頭** | 2023 | 2024-25 | 2025下-26 | 最差 |",
         "|---|---|---|---|---|---|---|---|"]
    for mode in DETECT:
        vals = [alpha(mode, lab) for lab,_,_ in REGIMES]
        valid = [v for v in vals if v is not None]
        c = [f"{v:+.0f}" if v is not None else "—" for v in vals]
        worst = f"{min(valid):+.0f}" if valid else "—"
        L.append(f"| {mode} | {bull_pct[mode]:.0f}% | {c[0]} | **{c[1]}** | {c[2]} | {c[3]} | {c[4]} | **{worst}** |")
    L += ["", "## 原始報酬 %(未扣大盤)+ 全期複合\n",
          "| 多頭偵測門檻 | 2021 | **2022空頭** | 2023 | 2024-25 | 2025下-26 | **全期複合** |", "|---|---|---|---|---|---|---|",
          f"| 0050大盤 | " + " | ".join(f"{bench[lab]:+.0f}" for lab,_,_ in REGIMES) + f" | **+{bench_comp:.0f}%** |"]
    for mode in DETECT:
        cells = " | ".join(f"{res[(mode,lab)]['ret']:+.0f}" if res.get((mode,lab)) else "—" for lab,_,_ in REGIMES)
        cc = cells.split(" | ")
        L.append(f"| {mode} | {cc[0]} | **{cc[1]}** | {cc[2]} | {cc[3]} | {cc[4]} | **+{compound(mode):.0f}%** |")
    (ROOT/"reports"/"exp_h_avoid2022.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/exp_h_avoid2022.md")

if __name__ == "__main__":
    main()