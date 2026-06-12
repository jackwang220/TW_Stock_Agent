"""驗證 H(成交值聚光燈):① 集中度(扣掉前N大貢獻股還剩多少)② 換 20日均成交值(較乾淨)③ H+G/D×turn 組合。
全扣真實手續費 + 漲停買不到 + 降換手(inc1.5+hyst5 / hyst5)。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_prices

_v3s = importlib.util.spec_from_file_location("v3", ROOT / "scripts/exp_step1_v3.py")
v3 = importlib.util.module_from_spec(_v3s); _v3s.loader.exec_module(v3)
_r60s = importlib.util.spec_from_file_location("r60", ROOT / "scripts/run_backtest_60d.py")
r60 = importlib.util.module_from_spec(_r60s); _r60s.loader.exec_module(r60)
features, _factors, oh, clamp = v3.features, v3._factors, v3.oh, v3.clamp

END = "2026-06-08"
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
FEE_BUY, FEE_SELL = 0.001425, 0.004425

def score(kind, ff, ex):
    """kind: H_d / H_20 / H+G / DxT / G / D+G。回傳 0-1 的 edge(=score/100)。"""
    reb = ex["reb"]; m = 0.0
    if ff is not None:
        t, rs, vo, ri, ma, br, bias = ff
        if kind in ("H_d","H_20","H+G"):   # trendRS base
            m = 0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma
        elif kind in ("DxT","D+G"):         # break6 base
            m = (t+rs+vo+ri+ma+br)/6
        elif kind == "G":
            m = (t+rs+vo+ri+ma)/5
        m *= 100
        if kind in ("H_d","DxT"): m *= (0.8+0.4*ex["turn_d"])
        if kind == "H_20": m *= (0.8+0.4*ex["turn_20"])
        if kind == "H+G": m *= (0.8+0.4*ex["turn_d"])*ex["regime_mult"]
        if kind in ("G","D+G"): m *= ex["regime_mult"] if kind != "G" else ex["regime_mult"]
        if kind == "G": m = m  # eq5×regime 已含
    return max(m, reb)/100

def sim(rows, panel, limitup, incumbent, hyst, track=False):
    sig = defaultdict(dict); tickers = set()
    for d, tk, e in rows:
        tickers.add(tk); sig[d][tk] = e
    if not sig: return None
    def price(tk, d):
        pr = panel.get(tk, {}); ds = [x for x in pr if x <= d]; return pr[max(ds)] if ds else None
    first, last = min(sig), max(sig)
    cal = sorted({d for tk in tickers for d in panel.get(tk, {}) if first <= d <= last})
    if not cal: return None
    cash = contributed = prev_eq = traded = fees = 0.0
    shares, last_edge = {}, {}; day_pnl = []
    pnl_by_tk = defaultdict(float)   # 每檔貢獻(現金流法)
    for i, d in enumerate(cal):
        add = r60.INITIAL_CAPITAL if i == 0 else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contributed) if contributed < r60.MAX_CONTRIBUTION else 0.0)
        cash += add; contributed += add
        port = cash + sum(shares[tk]*(price(tk, d) or 0) for tk in shares)
        todays = sig.get(d, {}); edges = {}
        for tk, e in todays.items(): edges[tk] = e; last_edge[tk] = e
        for tk in shares:
            if tk not in edges:
                e = last_edge.get(tk, 0.0)*r60.EDGE_DECAY; edges[tk] = e; last_edge[tk] = e
        rk = lambda tk: edges[tk]*(incumbent if tk in shares else 1.0)
        ranked = sorted([tk for tk in edges if edges[tk] > 0], key=rk, reverse=True)
        sel = ranked[:r60.MAX_SIGNALS]
        if len(ranked) > r60.MAX_SIGNALS and rk(ranked[r60.MAX_SIGNALS]) >= rk(ranked[r60.MAX_SIGNALS-1])*r60.TIE_RATIO:
            sel = ranked[:r60.MAX_SIGNALS+1]
        confs = [todays[tk] for tk in sel if tk in todays]; avg = sum(confs)/len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg)) if sel else 0.0
        wsum = sum(edges[tk] for tk in sel); targets = {}
        if wsum > 0 and expo > 0:
            for tk in sel: targets[tk] = port*expo*(edges[tk]/wsum)
        for tk in set(shares) | set(targets):
            p = price(tk, d)
            if not p: continue
            cur = shares.get(tk, 0.0)*p; tgt = targets.get(tk, 0.0); delta = tgt-cur
            if delta > r60.DAILY_ADD_CAP: delta = r60.DAILY_ADD_CAP; tgt = cur+delta
            if abs(delta) < hyst*port: continue
            if delta > 0 and d in limitup.get(tk, set()): continue
            fee = (FEE_BUY if delta > 0 else FEE_SELL)*abs(delta)
            cash -= delta + fee; traded += abs(delta); fees += fee
            pnl_by_tk[tk] -= delta + fee                    # 買花錢(負)、賣收錢(正)
            if tgt <= 1e-6: shares.pop(tk, None)
            else: shares[tk] = tgt/p
        equity = cash + sum(shares[tk]*(price(tk, d) or 0) for tk in shares)
        day_pnl.append(equity - prev_eq - add); prev_eq = equity
    for tk in shares: pnl_by_tk[tk] += shares[tk]*(price(tk, last) or 0)   # 期末持股市值算回
    total = sum(day_pnl); active = [x for x in day_pnl if abs(x) > 1e-9]
    cum = peak = mdd = 0.0
    for x in day_pnl: cum += x; peak = max(peak, cum); mdd = max(mdd, peak-cum)
    if len(active) > 1:
        mm = sum(active)/len(active); sd = math.sqrt(sum((x-mm)**2 for x in active)/len(active))
        shp = (mm/sd*math.sqrt(252)) if sd > 0 else 0.0
    else: shp = 0.0
    out = {"ret": total/contributed*100 if contributed else 0, "mdd": mdd, "sharpe": shp,
           "turn": traded/contributed if contributed else 0, "total_pnl": total, "contributed": contributed}
    if track: out["pnl_by_tk"] = dict(pnl_by_tk)
    return out

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("特徵/反彈/漲停/成交值(日+20日)/價格...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    cal = [d for d in sorted(twii_feat) if d <= END][-WINDOWS[-1][1]:]
    reb_cache, limitup, turn20series = {}, {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); closes = []; m = {}; s = set()
        tser = pd.Series([o[d]["close"]*o[d].get("volume", 0) for d in ds], index=ds)
        t20 = tser.rolling(20).mean()
        turn20series[c] = {ds[i]: t20.iloc[i] for i in range(len(ds))}
        for j, d in enumerate(ds):
            closes.append(o[d]["close"])
            if len(closes) >= 25:
                try:
                    g = rebound_signal(closes, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095: s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct_d, turn_pct_20 = {}, {}
    for d in cal:
        vd = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct_d[d] = {c: (i+1)/len(vd) for i, (c, _) in enumerate(vd)}
        v2 = sorted(((c, turn20series[c][d]) for c in codes if d in turn20series.get(c, {}) and turn20series[c][d] and not math.isnan(turn20series[c][d])), key=lambda x: x[1])
        turn_pct_20[d] = {c: (i+1)/len(v2) for i, (c, _) in enumerate(v2)} if v2 else {}
    panel = {c: get_daily_prices(c) for c in codes}

    def rows_for(kind, dates):
        out = []
        for d in dates:
            tf = twii_feat.get(d, {}); ir = tf.get("ret20")
            rm = 1.0 if (tf.get("close") and tf.get("ma20") and tf["close"] > tf["ma20"]) else 0.7
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                ex = {"regime_mult": rm, "turn_d": turn_pct_d.get(d, {}).get(c, 0.5),
                      "turn_20": turn_pct_20.get(d, {}).get(c, 0.5), "reb": reb_cache.get(c, {}).get(d, 0.0)}
                e = score(kind, _factors(f[d], ir), ex)
                if e > 0: out.append((d, c, e))
        return out

    INC, HY = 1.5, 0.05
    L = ["# H 驗證(扣費+降換手 inc1.5+hyst5)\n"]
    # ── ① 集中度 ──
    L += ["## ① H 集中度檢查(每檔貢獻;扣掉前N大還剩多少)\n",
          "| 窗口 | 總損益 | 前1大 | 前3大 | 前3大佔比 | 扣前3大後報酬 |", "|---|---|---|---|---|---|"]
    for wl, n in [("1年",252),("2年",504)]:
        wd = set(cal[-n:]); r = sim(rows_for("H_d", [x for x in cal if x in wd]), panel, limitup, INC, HY, track=True)
        pt = sorted(r["pnl_by_tk"].items(), key=lambda x: -x[1])
        top3 = sum(v for _, v in pt[:3]); tot = r["total_pnl"]
        ex_ret = (tot - top3)/r["contributed"]*100
        topn = ", ".join(f"{names.get(tk,tk)[:3]}{v/tot*100:.0f}%" for tk, v in pt[:5])
        L.append(f"| {wl} | {tot:,.0f} | {pt[0][1]/tot*100:.0f}% | {top3/tot*100:.0f}% | — | {ex_ret:+.1f}% |")
        L.append(f"|   └ 前5大貢獻 | {topn} ||||")
    # ── ② 量度 + ③ 組合 ──
    L += ["", "## ②③ 換乾淨量度 + 組合(扣費後本金報酬率 %)\n",
          "| 變體 | 60天 | 90天 | 半年 | 1年 | 1年半 | 2年 |", "|---|---|---|---|---|---|---|"]
    for kind, lab in [("H_d","H 日成交值"),("H_20","H 20日均成交值"),("H+G","H+G"),("DxT","D×成交值"),("G","G(對照)"),("D+G","D+G(對照)")]:
        cells = []
        for wl, n in WINDOWS:
            wd = set(cal[-n:]); r = sim(rows_for(kind, [x for x in cal if x in wd]), panel, limitup, INC, HY)
            cells.append(f"{r['ret']:+.0f}" if r else "—")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    REPORT = ROOT / "reports" / "exp_h_verify.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")

if __name__ == "__main__":
    main()