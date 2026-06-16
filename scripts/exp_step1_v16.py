"""Step1 v16 — 族群動能因子接進策略 + 回測驗證(IC→實際alpha)。
Phase1 收集關係:每月用過去120日報酬算相關,每檔取前K=8鄰居(point-in-time,純數據)。
Phase2 族群動能:鄰居的平均 N 日報酬(排除自己)。
Phase3 接分數:族群動能當天百分位 → 乘數(lo+span×pct),乘進「多頭 H 分數」(空頭反彈不動)。
Phase4 回測:plain H(B純切) vs H×族群乘數,同框架(⑤收盤買+開盤只賣、incumbent1.5、112檔、6窗+5regime)。
比 alpha/最差季/換手;含敏感度(動能窗20/60、乘數強度)。
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

START, END = "2021-01-01", "2026-06-08"
K = 8
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

    logger.info("載入全史還原 OHLCV 2021~ ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    logger.info("特徵 ...")
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

    # ── Phase1+2:族群關係(每月相關前K) + 族群動能(鄰居平均 N 日報酬)──
    logger.info("Phase1/2:族群關係 + 族群動能 ...")
    close = pd.DataFrame({c: pd.Series(closes_p[c]) for c in codes}).reindex(alld).sort_index()
    ret1 = close.pct_change()
    idx = {c: i for i, c in enumerate(codes)}
    def gm_df(mom):
        retm = close.pct_change(mom)
        gm = pd.DataFrame(np.nan, index=close.index, columns=codes)
        mf = {}
        for d in alld: mf.setdefault(d[:7], d)
        for ym, rd in mf.items():
            win = ret1.loc[:rd].iloc[:-1].tail(120)
            if len(win) < 60: continue
            cmat = win.corr(); W = np.zeros((len(codes), len(codes)))
            for c in codes:
                col = cmat[c].drop(c).dropna()
                if len(col) < K: continue
                for p in col.nlargest(K).index: W[idx[c], idx[p]] = 1.0/K
            md = [d for d in alld if d[:7] == ym]
            gm.loc[md] = retm.loc[md, codes].values @ W.T
        return gm
    GM = {60: gm_df(60), 20: gm_df(20)}

    def mult_map(mom, lo, span):
        gm = GM[mom]; out = {}
        for d in alld:
            row = gm.loc[d].dropna()
            if len(row) < 5: continue
            r = row.rank(pct=True)
            out[d] = {c: lo + span*r[c] for c in row.index}
        return out

    # ── rows builder:B純切;group_mult 乘進多頭 H 分數 ──
    def build_rows(gmult):
        rows = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
            gm_d = gmult.get(d, {}) if gmult else {}
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                if gmult: hh *= gm_d.get(c, 1.0)               # 族群乘數只動多頭H
                rb = reb_cache.get(c, {}).get(d, 0.0)
                sc = hh if bull else rb*1.5                    # B純切
                if sc > 0: rows.append((d, c, sc/100))
        return rows

    VARIANTS = [
        ("plain H(B純切,無族群)", None),
        ("H×族群60日(乘0.8~1.2)", mult_map(60, 0.8, 0.4)),
        ("H×族群60日(乘0.7~1.3 強)", mult_map(60, 0.7, 0.6)),
        ("H×族群20日(乘0.8~1.2)", mult_map(20, 0.8, 0.4)),
    ]

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes_p["0050"], sorted(col_dates[col])) for col in COLS}

    logger.info("Phase4:回測 ...")
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

    L = ["# Step1 v16 — 族群動能因子接進 H + 回測(plain H vs H×族群乘數)\n",
         f"> B純切｜⑤收盤買+開盤只賣｜incumbent1.5｜112檔｜結束{END}｜DCA｜族群=相關前{K}(每月、過去120日)\n",
         "> 族群乘數只乘多頭H分數(空頭反彈不動);ALPHA=策略−同資金DCA0050;換手取2年\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) |",
         "|" + "---|" * (len(COLS) + 4)]
    ranked = sorted([v[0] for v in VARIANTS],
                    key=lambda lb: min((a(lb, c) for c in COLS if a(lb, c) is not None), default=-999), reverse=True)
    for lb in ranked:
        vals = [a(lb, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res.get((lb, "2年")); turn = r2["turn"] if r2 else 0
        L.append(f"| {lb} | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {turn:.1f}x |")
    L += ["", "> 解讀:族群版要在『平均』『最差季』**贏過 plain H** 且換手沒爆,才值得上線;否則 IC 雖正但接進策略沒實益。"]
    REPORT = ROOT / "reports" / "exp_step1_v16.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")


if __name__ == "__main__":
    main()
