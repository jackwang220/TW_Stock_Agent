"""趨勢gate實驗 — 「MA5≤MA20 沒趨勢就硬切0」改成幾種較軟的處理,回測比較。
動機:強茂今天漲停但 MA5 還沒站上 MA20 → H分=0(斷崖)。測軟化會不會更好。
變體(只影響多頭H;空頭反彈不變。B純切、⑤收盤買開盤只賣、incumbent1.75、116檔):
  V0 gate0   : MA5≤MA20 → 0(現行,斷崖)
  V1 nogate  : 拿掉gate,trend因子[0,1](低於門檻 trend=0,其餘因子RS/量/RSI/MACD照算)
  V2 signed  : 拿掉gate,trend因子[-1,1](下降趨勢給負分,連續穿過門檻)
  V3 pen50   : 拿掉gate,低於門檻者「完整分數×0.5」(半信半疑)
  V4 pen70   : 同上但×0.7
ALPHA=策略−同資金DCA0050;6窗+5regime;換手取2年。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove(); logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v5 = _load("v5", "scripts/exp_step1_v5.py"); v6 = _load("v6", "scripts/exp_step1_v6.py"); ec = _load("ec", "scripts/exp_60d_entry_compare.py")
features = v5.features; sim5 = ec.sim_buyclose_sellopen; bench_0050 = v6.bench_0050
clamp = lambda x, lo=0.0, hi=1.0: max(lo, min(hi, x))

START, END = "2021-01-01", "2026-06-08"; INC = 1.75
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252), ("1年半", 378), ("2年", 504)]
REGIMES = [("2021復甦", "2021-04-01", "2021-12-31"), ("2022空頭", "2022-01-01", "2022-12-31"),
           ("2023復甦", "2023-01-01", "2023-12-31"), ("2024-25多頭", "2024-01-01", "2025-06-30"),
           ("2025下-26", "2025-07-01", "2026-06-08")]

def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t + 0.35*rs + 0.15*vo + 0.10*ri + 0.05*ma) * 100 * (0.8 + 0.4*tp)

def factors_variant(feat, ir, mode):
    """回 (因子tuple, 分數乘數) 或 None。"""
    if not feat.get("ma20") or math.isnan(feat["ma20"]) or math.isnan(feat.get("rsi", float("nan"))):
        return None
    above = feat["ma5"] > feat["ma20"]
    if mode == "gate0" and not above:
        return None
    rs = (1+feat["ret20"])/(1+ir) if (ir is not None and abs(1+ir) > 1e-6 and not math.isnan(feat["ret20"])) else 1.0
    raw = (feat["ma5"]/feat["ma20"] - 1)/0.05
    f_trend = clamp(raw, -1.0, 1.0) if mode == "signed" else clamp(raw)
    f_rs = clamp((rs-0.9)/0.2)
    f_vol = clamp((feat["volr"]-1.2)/0.8) if not math.isnan(feat["volr"]) else 0.0
    r = feat["rsi"]; f_rsi = clamp((r-40)/30) if r <= 70 else clamp(1-(r-70)/20)
    f_macd = 1.0 if feat["macdh"] > 0 else 0.4
    f_break = clamp((feat["close"]/feat["high60"]-0.90)/0.10) if feat["high60"] else 0.0
    bias = feat["close"]/feat["ma20"]-1
    mult = 1.0
    if not above:
        mult = {"pen50": 0.5, "pen70": 0.7}.get(mode, 1.0)
    return ((f_trend, f_rs, f_vol, f_rsi, f_macd, f_break, bias), mult)

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入 {len(codes)} 檔 ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    twii = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c]})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes + ["0050"]}
    closes_p = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes + ["0050"]}
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

    # 預算每日 tp(成交值百分位)
    tp_by_day = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d].get("turn", 0) > 0), key=lambda x: x[1])
        tp_by_day[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    def build_rows(mode):
        rows = []
        for d in alld:
            tf = twii.get(d)
            if not tf or not tf.get("ma20") or math.isnan(tf["ma20"]): continue
            bull = tf["close"] > tf["ma20"]; ir = tf.get("ret20"); tp = tp_by_day[d]
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                if bull:
                    fv = factors_variant(f[d], ir, mode)
                    sc = h_score(fv[0], tp.get(c, 0.5)) * fv[1] if fv else 0.0
                else:
                    sc = reb_cache.get(c, {}).get(d, 0.0) * 1.5
                if sc > 0: rows.append((d, c, sc/100))
        return rows

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes_p["0050"], sorted(col_dates[col])) for col in COLS}

    VARIANTS = [("V0 gate0(現行斷崖)", "gate0"), ("V1 nogate(trend[0,1])", "nogate"),
                ("V2 signed(trend[-1,1])", "signed"), ("V3 pen50(低門檻×0.5)", "pen50"),
                ("V4 pen70(低門檻×0.7)", "pen70")]
    res = {}
    for lab, mode in VARIANTS:
        rows = build_rows(mode)
        for col in COLS:
            ds = col_dates[col]
            res[(lab, col)] = sim5([x for x in rows if x[0] in ds], opens, closes_p, limitup,
                                   incumbent=INC, sell_mode="exit_only", open_buy="none")
        logger.info(f"  {lab} 完成")

    def a(lab, col):
        r = res.get((lab, col)); return (r["ret"]-bench[col]) if r else None
    L = ["# 趨勢gate實驗 — 沒趨勢(MA5≤MA20)的處理 vs 現行斷崖切0\n",
         f"> B純切｜⑤收盤買+開盤只賣｜incumbent{INC}｜116檔｜結束{END}｜DCA\n",
         "> 0050各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) |",
         "|" + "---|" * (len(COLS) + 4)]
    for lab, _ in VARIANTS:
        vals = [a(lab, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res.get((lab, "2年")); turn = r2["turn"] if r2 else 0
        L.append(f"| {lab} | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {turn:.1f}x |")
    L += ["", "> 判讀:軟化(V1-V4)要平均不輸 V0 且最差不更糟,才值得改;否則斷崖切0(只騎確立趨勢)就是對的。"]
    REPORT = ROOT / "reports" / "exp_trend_gate.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); logger.success(f"報告 → {REPORT}")

if __name__ == "__main__":
    main()
