"""Step1 v22 — 日線 MACD 死叉出場(可回測)。
現況:H 分數已含 MACD 軟因子(macdh>0→1.0, <0→0.4),但不會硬出場。
測:在 B純切 上加「MACD 死叉(macdh<0)→ 分數歸零」→ 掉出名單 → 隔天開盤賣(也不進場已死叉的)。
變體:① 當日 macdh<0 出場  ② 連2日 macdh<0 才出(較不洗)。對照 plain H。
112檔｜⑤收盤買+開盤只賣｜incumbent1.5｜DCA｜6窗+5regime。
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


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("載入全史 ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c]})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes + ["0050"]}
    closes_p = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes + ["0050"]}

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

    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    # 預算每檔「macdh 連續<0 天數」(算死叉幾天)
    dead_days = {}
    for c in codes:
        ds = sorted(feats[c]); run = 0
        for d in ds:
            mh = feats[c][d].get("macdh", float("nan"))
            run = (run + 1) if (not math.isnan(mh) and mh < 0) else 0
            dead_days[(c, d)] = run

    def build_rows(macd_exit):   # macd_exit: 0=不用, 1=當日死叉出, 2=連2日死叉出
        rows = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                if macd_exit and dead_days.get((c, d), 0) >= macd_exit:
                    hh = 0.0                                  # MACD 死叉 → 出場/不進(只動多頭H)
                rb = reb_cache.get(c, {}).get(d, 0.0)
                sc = hh if bull else rb*1.5
                if sc > 0: rows.append((d, c, sc/100))
        return rows

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes_p["0050"], sorted(col_dates[col])) for col in COLS}

    VARIANTS = [("plain H(MACD僅軟因子)", 0),
                ("+MACD死叉出場(當日)", 1),
                ("+MACD死叉出場(連2日)", 2)]
    logger.info("回測 ...")
    res = {}
    for lab, me in VARIANTS:
        rows = build_rows(me)
        for col in COLS:
            ds = col_dates[col]
            res[(lab, col)] = sim5([x for x in rows if x[0] in ds], opens, closes_p, limitup,
                                   incumbent=1.5, sell_mode="exit_only", open_buy="none")
        logger.info(f"  {lab} 完成")

    def a(lab, col):
        r = res.get((lab, col)); return (r["ret"]-bench[col]) if r else None
    L = ["# Step1 v22 — 日線 MACD 死叉出場 vs plain H(112檔,B純切)\n",
         "> 死叉=macdh<0 → 該檔分數歸零 → 掉出名單 → 隔天開盤賣(也不進已死叉的);只動多頭H｜⑤｜incumbent1.5\n",
         "> ALPHA=策略−同資金DCA0050;換手取2年\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) |",
         "|" + "---|" * (len(COLS) + 4)]
    for lb in [v[0] for v in VARIANTS]:
        vals = [a(lb, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res.get((lb, "2年")); turn = r2["turn"] if r2 else 0
        L.append(f"| {lb} | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {turn:.1f}x |")
    L += ["", "> MACD死叉出場要『平均不降 + 最差季變好(早跑掉跌的)』才有價值;若平均掉很多=被洗掉太多賣太早。"]
    (ROOT / "reports" / "exp_step1_v22.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/exp_step1_v22.md")


if __name__ == "__main__":
    main()
