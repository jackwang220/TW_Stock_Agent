"""Step1 v21 — 修正族群因子:① 幅度型乘數(取代百分位,近0自動關閉、極端才放大,符合「明顯趨勢才算」)
② 掃相關窗 60/120/250 天(不再固定120)。超額動能(−大盤)。universe=v3(118,有被動元件群)。
乘數 = clip(1 + 超額族群動能 × STRENGTH, LO, HI)。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from pathlib import Path
import numpy as np
import pandas as pd

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

UNIV = "base_universe_v3.json"
START, END = "2021-01-01", "2026-06-08"; K = 8; MOM = 60
STRENGTH, LO, HI = 0.6, 0.7, 1.4
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
CORR_WINS = [60, 90, 120, 150, 180, 250]


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def main():
    u = json.loads((DATA_DIR/UNIV).read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入 {UNIV} ({len(codes)}檔) 全史 ...")
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

    close = pd.DataFrame({c: pd.Series(closes_p[c]) for c in codes}).reindex(alld).sort_index()
    mkt = pd.Series(closes_p["0050"]).reindex(alld).sort_index()
    ret1 = close.pct_change(); retM = close.pct_change(MOM); mretM = mkt.pct_change(MOM)
    idx = {c: i for i, c in enumerate(codes)}

    def mult_for(corr_win):
        """超額族群動能 → 幅度型乘數(近0=1.0,極端才放大)。回 {date:{code:mult}}。"""
        gm = pd.DataFrame(np.nan, index=close.index, columns=codes)
        mf = {}
        for d in alld: mf.setdefault(d[:7], d)
        for ym, rd in mf.items():
            win = ret1.loc[:rd].iloc[:-1].tail(corr_win)
            if len(win) < max(40, corr_win // 2): continue
            cmat = win.corr(); W = np.zeros((len(codes), len(codes)))
            for c in codes:
                col = cmat[c].drop(c).dropna()
                if len(col) < K: continue
                for p in col.nlargest(K).index: W[idx[c], idx[p]] = 1.0/K
            md = [d for d in alld if d[:7] == ym]
            gm.loc[md] = np.nan_to_num(retM.loc[md, codes].values) @ W.T
        gm = gm.sub(mretM, axis=0)                       # 超額(−大盤)
        out = {}
        for d in alld:
            row = gm.loc[d]
            mm = (1 + STRENGTH * row).clip(LO, HI)       # 幅度型:近0→1.0,正大→放大,負→壓低
            out[d] = {c: mm[c] for c in codes if not math.isnan(mm[c])}
        return out

    def build_rows(gmult):
        rows = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
            gm_d = gmult.get(d, {}) if gmult else {}
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                if gmult: hh *= gm_d.get(c, 1.0)
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

    VARIANTS = [("plain H(無族群)", None)]
    for cw in CORR_WINS:
        VARIANTS.append((f"幅度族群|相關窗{cw}日", mult_for(cw)))

    logger.info("回測 ...")
    res = {}
    for lab, gmult in VARIANTS:
        rows = build_rows(gmult)
        for col in COLS:
            ds = col_dates[col]
            res[(lab, col)] = sim5([x for x in rows if x[0] in ds], opens, closes_p, limitup,
                                   incumbent=1.5, sell_mode="exit_only", open_buy="none")
        logger.info(f"  {lab} 完成")

    def a(lab, col):
        r = res.get((lab, col)); return (r["ret"]-bench[col]) if r else None
    L = [f"# Step1 v21 — 修正族群因子(幅度乘數+掃相關窗) | {UNIV}({len(codes)}檔)\n",
         f"> 幅度乘數=clip(1+超額動能×{STRENGTH},{LO},{HI});近0→1.0(因子自動關)、極端才放大。B純切｜⑤｜incumbent1.5\n",
         "> ALPHA=策略−同資金DCA0050;換手取2年\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) |",
         "|" + "---|" * (len(COLS) + 4)]
    for lb in [v[0] for v in VARIANTS]:
        vals = [a(lb, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res.get((lb, "2年")); turn = r2["turn"] if r2 else 0
        L.append(f"| {lb} | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {turn:.1f}x |")
    L += ["", "> 幅度乘數要贏 plain H(尤其平均+最差不退步),才證明『百分位壓平+固定120窗』是之前失敗主因。"]
    (ROOT / "reports" / "exp_step1_v21.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/exp_step1_v21.md")


if __name__ == "__main__":
    main()
