"""Step1 v11:固定 H雙引擎 A/B/C,測『0050 趨勢判斷規則 × 回看天數』對 alpha 的影響。
多頭→打 H(h_score),空頭→打反彈(reb);多/空開關由 0050 決定。
比較 6 種 0050 判斷(MA20/MA60/MA120/MA20>MA60/季+排列/ret60),看能否讓空頭少受傷、整體更穩。
引擎 = ⑤ 收盤買+開盤賣買;成本買0.14/賣0.44+滑價0.1+漲停買不到;全史2021~還原價;112檔。
ALPHA = 策略 − 同資金 DCA 進 0050(每欄各自的日期區間)。
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

def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

v5 = _load("v5", "scripts/exp_step1_v5.py")
v6 = _load("v6", "scripts/exp_step1_v6.py")
ec = _load("ec", "scripts/exp_60d_entry_compare.py")
features, _factors = v5.features, v5._factors
sim5 = ec.sim_buyclose_sellopen
bench_0050 = v6.bench_0050

START, END = "2021-01-01", "2026-06-08"
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
DUAL = {"H雙引擎A": (1.0,0.4,0.2,1.4), "H雙引擎B純切": (1.0,0.0,0.0,1.5), "H雙引擎C": (1.0,0.6,0.3,1.3)}
DETECTORS = ["c>MA20(基準)", "c>MA60(季線)", "c>MA120(半年線)",
             "MA20>MA60(均線多頭)", "c>MA60&MA20>MA60", "ret60>0"]


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info("載入全史還原 OHLCV 2021~ ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    logger.info("特徵 ...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c]})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes + ["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes + ["0050"]}

    logger.info("反彈/漲停 ...")
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

    # ── 0050 趨勢判斷:用 0050 收盤序列算各種規則,carry-forward 到 alld ──
    ds0 = sorted(OH["0050"]); c0 = [OH["0050"][d]["close"] for d in ds0]
    def ma(i, k): return sum(c0[i-k+1:i+1])/k if i >= k-1 else float("nan")
    raw = {}   # date -> dict of detector booleans
    for i, d in enumerate(ds0):
        m20, m60, m120 = ma(i,20), ma(i,60), ma(i,120)
        r60 = (c0[i]/c0[i-60]-1) if i >= 60 else float("nan")
        raw[d] = {
            "c>MA20(基準)":        (not math.isnan(m20)) and c0[i] > m20,
            "c>MA60(季線)":        (not math.isnan(m60)) and c0[i] > m60,
            "c>MA120(半年線)":     (not math.isnan(m120)) and c0[i] > m120,
            "MA20>MA60(均線多頭)": (not math.isnan(m20)) and (not math.isnan(m60)) and m20 > m60,
            "c>MA60&MA20>MA60":    (not math.isnan(m60)) and (not math.isnan(m20)) and c0[i] > m60 and m20 > m60,
            "ret60>0":             (not math.isnan(r60)) and r60 > 0,
        }
    # carry-forward 到所有 alld(理論上 alld⊆ds0,保險起見)
    bull_maps = {det: {} for det in DETECTORS}
    last = {det: False for det in DETECTORS}
    for d in alld:
        if d in raw:
            for det in DETECTORS: last[det] = raw[d][det]
        for det in DETECTORS: bull_maps[det][d] = last[det]
    bull_pct = {det: sum(bull_maps[det][d] for d in alld)/len(alld)*100 for det in DETECTORS}

    # turn_pct(112檔內)
    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    def rows_dual(w, bull_map):
        whb, wrb, whs, wrs = w; out = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = bull_map[d]
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                rb = reb_cache.get(c, {}).get(d, 0.0)
                sc = max(hh*whb, rb*wrb) if bull else max(hh*whs, rb*wrs)
                if sc > 0: out.append((d, c, sc/100))
        return out

    # 欄位日期 + 0050 基準
    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    res = {}   # (variant, detector, col) -> sim
    for vn, w in DUAL.items():
        for det in DETECTORS:
            rows = rows_dual(w, bull_maps[det])
            for col in COLS:
                ds = col_dates[col]
                res[(vn, det, col)] = sim5([x for x in rows if x[0] in ds], opens, closes, limitup)
        logger.info(f"{vn} 完成")

    def a(vn, det, col):
        r = res.get((vn, det, col)); return (r["ret"] - bench[col]) if r else None

    L = ["# Step1 v11 — 固定 H雙引擎 A/B/C,測『0050 趨勢判斷 × 回看天數』\n",
         f"> ⑤執行｜結束{END}｜還原價(全史2021~)｜手續費買0.14/賣0.44+滑價0.1+漲停買不到｜112檔｜ALPHA=策略−同資金DCA 0050\n",
         "> 多頭→打H,空頭→打反彈;開關=0050各規則。**重點看 2022空頭 + 最差欄**(慢判斷應救回空頭,但拖慢多頭起漲)\n",
         "> 0050 各欄基準: " + " ".join(f"{col}{bench[col]:+.0f}%" for col in COLS) + "\n",
         "> 各判斷的『多頭日%』: " + " ".join(f"{det.split('(')[0]}={bull_pct[det]:.0f}%" for det in DETECTORS) + "\n"]
    head = "| 0050判斷 | 多頭日% | " + " | ".join(COLS) + " | 最差 | 平均 |"
    sep = "|" + "---|" * (len(COLS) + 4)
    for vn in DUAL:
        L += [f"\n## {vn}\n", head, sep]
        for det in DETECTORS:
            vals = [a(vn, det, col) for col in COLS]; valid = [v for v in vals if v is not None]
            cells = " | ".join(f"{v:+.0f}" if v is not None else "—" for v in vals)
            w = f"{min(valid):+.0f}" if valid else "—"; av = f"{sum(valid)/len(valid):+.0f}" if valid else "—"
            L.append(f"| {det} | {bull_pct[det]:.0f}% | {cells} | **{w}** | {av} |")
    REPORT = ROOT / "reports" / "exp_step1_v11.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")


if __name__ == "__main__":
    main()
