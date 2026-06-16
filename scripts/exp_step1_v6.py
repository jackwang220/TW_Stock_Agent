"""Step1 v6 真實版:隔日開盤成交(非收盤)+滑價+漲停買不到+手續費+降換手,加 ALPHA(扣0050同資金) + 集中度。
ALPHA = 策略報酬% - 「同資金模型 DCA 進 0050」報酬%(剝掉大盤beta)。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal

_v5s = importlib.util.spec_from_file_location("v5", ROOT / "scripts/exp_step1_v5.py")
v5 = importlib.util.module_from_spec(_v5s); _v5s.loader.exec_module(v5)
_r60s = importlib.util.spec_from_file_location("r60", ROOT / "scripts/run_backtest_60d.py")
r60 = importlib.util.module_from_spec(_r60s); _r60s.loader.exec_module(r60)
features, _factors, oh, clamp = v5.features, v5._factors, v5.oh, v5.clamp

END = "2026-06-08"
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
FEE_BUY, FEE_SELL, SLIP = 0.001425, 0.004425, 0.001
INC, HY = 1.5, 0.05

def sim_real(rows, opens, closes, limitup, incumbent=INC, hyst=HY, track=False):
    """隔日開盤成交 + 滑價 + 漲停(隔日開盤)買不到 + 手續費。"""
    sig = defaultdict(dict); tickers = set()
    for d, tk, e in rows: tickers.add(tk); sig[d][tk] = e
    if not sig: return None
    alld = sorted({d for tk in tickers for d in closes.get(tk, {})})
    first, last = min(sig), max(sig)
    cal = [d for d in alld if first <= d <= last]
    if len(cal) < 2: return None
    def cl(tk, d):
        c = closes.get(tk, {}); ds = [x for x in c if x <= d]; return c[max(ds)] if ds else None
    cash = contributed = prev_eq = traded = fees = 0.0
    shares, last_edge = {}, {}; day_pnl = []; pnl_tk = defaultdict(float)
    for i, d in enumerate(cal):
        add = r60.INITIAL_CAPITAL if i == 0 else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contributed) if contributed < r60.MAX_CONTRIBUTION else 0.0)
        cash += add; contributed += add
        if i+1 >= len(cal):
            eq = cash + sum(shares[tk]*(cl(tk, d) or 0) for tk in shares)
            day_pnl.append(eq - prev_eq - add); prev_eq = eq; break
        e = cal[i+1]
        port = cash + sum(shares[tk]*(cl(tk, d) or 0) for tk in shares)
        todays = sig.get(d, {}); edges = {}
        for tk, ed in todays.items(): edges[tk] = ed; last_edge[tk] = ed
        for tk in shares:
            if tk not in edges:
                ed = last_edge.get(tk, 0.0)*r60.EDGE_DECAY; edges[tk] = ed; last_edge[tk] = ed
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
            op = opens.get(tk, {}).get(e)
            if not op or op <= 0: continue
            pc = cl(tk, d)
            cur = shares.get(tk, 0.0)*op; tgt = targets.get(tk, 0.0); delta = tgt-cur
            if delta > r60.DAILY_ADD_CAP: delta = r60.DAILY_ADD_CAP; tgt = cur+delta
            if abs(delta) < hyst*port: continue
            if delta > 0 and pc and op/pc-1 >= 0.095: continue        # 隔日開盤漲停 → 買不到
            slip_cost = abs(delta)*SLIP
            fee = (FEE_BUY if delta > 0 else FEE_SELL)*abs(delta)
            cash -= delta + slip_cost + fee; traded += abs(delta); fees += fee+slip_cost
            pnl_tk[tk] -= delta + slip_cost + fee
            if tgt <= 1e-6: shares.pop(tk, None)
            else: shares[tk] = tgt/op
        eq = cash + sum(shares[tk]*(cl(tk, e) or 0) for tk in shares)
        day_pnl.append(eq - prev_eq - add); prev_eq = eq
    for tk in shares: pnl_tk[tk] += shares[tk]*(cl(tk, cal[-1]) or 0)
    total = sum(day_pnl); active = [x for x in day_pnl if abs(x) > 1e-9]
    cum = peak = mdd = 0.0
    for x in day_pnl: cum += x; peak = max(peak, cum); mdd = max(mdd, peak-cum)
    if len(active) > 1:
        mm = sum(active)/len(active); sd = math.sqrt(sum((x-mm)**2 for x in active)/len(active))
        shp = (mm/sd*math.sqrt(252)) if sd > 0 else 0.0
    else: shp = 0.0
    out = {"ret": total/contributed*100 if contributed else 0, "mdd": mdd, "sharpe": shp,
           "turn": traded/contributed if contributed else 0, "total_pnl": total}
    if track: out["pnl_tk"] = dict(pnl_tk)
    return out

def bench_0050(opens, closes, dates):
    """同資金模型(15000+1000/日上限5萬)DCA 進 0050,隔日開盤買、滑價、持有到底。"""
    cal = [d for d in dates if d in closes]
    if len(cal) < 2: return 0.0
    sh = cash = contributed = 0.0
    for i, d in enumerate(cal):
        add = r60.INITIAL_CAPITAL if i == 0 else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contributed) if contributed < r60.MAX_CONTRIBUTION else 0.0)
        cash += add; contributed += add
        if i+1 < len(cal):
            op = opens.get(cal[i+1])
            if op and op > 0 and cash > 0:
                sh += cash/(op*(1+SLIP)); cash = 0.0
    final = cash + sh*closes[cal[-1]]
    return (final - contributed)/contributed*100 if contributed else 0.0

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("特徵/反彈/漲停/成交值/開收盤...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    cal = [d for d in sorted(twii_feat) if d <= END][-504:]
    opens, closes = {}, {}
    o50 = oh("0050"); opens["0050"] = {d: o50[d]["open"] for d in o50}; closes["0050"] = {d: o50[d]["close"] for d in o50}
    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); closes[c] = {d: o[d]["close"] for d in ds}
        opens[c] = {d: o[d]["open"] for d in ds}; closesl = []; m = {}; s = set()
        for j, d in enumerate(ds):
            closesl.append(o[d]["close"])
            if len(closesl) >= 25:
                try:
                    g = rebound_signal(closesl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095: s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct = {}
    for d in cal:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)}

    VAR = dict(v5.VAR); VAR["反彈only"] = {"base": "none"}
    bench = {wl: bench_0050(opens["0050"], closes["0050"], cal[-n:]) for wl, n in WINDOWS}

    # ⑤執行引擎(收盤買+開盤賣買):延遲載入 ec(避免頂層循環 import)
    ecs = importlib.util.spec_from_file_location("ec", ROOT/"scripts/exp_60d_entry_compare.py")
    ecm = importlib.util.module_from_spec(ecs); ecs.loader.exec_module(ecm)
    sim5 = ecm.sim_buyclose_sellopen

    res, conc = {}, {}
    for vn, cfg in VAR.items():
        if cfg["base"] == "none":
            rows_full = [(d, c, reb_cache[c][d]/100) for c in codes for d in reb_cache[c] if d in set(cal)]
        else:
            rows_full = v5.build_rows(codes, names, feats, twii_feat, reb_cache, turn_pct, cfg, cal)
        for wl, n in WINDOWS:
            wd = set(cal[-n:]); rw = [r for r in rows_full if r[0] in wd]
            res[(vn, wl)] = sim5(rw, opens, closes, limitup)   # ⑤:收盤買/開盤賣買
        logger.info(f"{vn} 完成")

    def a(vn, wl): r = res.get((vn, wl)); return (r["ret"]-bench[wl]) if r else None
    order = sorted(VAR, key=lambda vn: min((a(vn, wl) for wl in ["1年","1年半","2年"] if a(vn, wl) is not None), default=-999), reverse=True)
    L = ["# Step1 v6 — ⑤執行(收盤買+開盤賣+開盤補買)真實成交｜16變體 × 6窗口\n",
         f"> 結束{END}｜112檔｜還原價｜手續費買0.14%/賣0.44%+滑價0.1%+漲停買不到｜資金15000+1000/日上限5萬\n",
         f"> ALPHA=策略報酬-同資金DCA進0050｜0050基準:" +
         " ".join(f"{wl}{bench[wl]:+.0f}%" for wl,_ in WINDOWS) + "\n",
         "## 🎯 ALPHA %(扣大盤beta;排序=1年/1年半/2年最差alpha)\n",
         "| 變體 | 60天 | 90天 | 半年 | 1年 | 1年半 | 2年 |", "|---|---|---|---|---|---|---|"]
    for vn in order:
        cells = " | ".join(f"{a(vn,wl):+.0f}" if a(vn,wl) is not None else "—" for wl,_ in WINDOWS)
        L.append(f"| {vn} | {cells} |")
    L += ["", "## 參考:真實版「原始報酬%」(未扣大盤)\n",
          "| 變體 | 60天 | 90天 | 半年 | 1年 | 1年半 | 2年 |", "|---|---|---|---|---|---|---|"]
    for vn in order:
        L.append(f"| {vn} | " + " | ".join(f"{res[(vn,wl)]['ret']:+.0f}" if res.get((vn,wl)) else "—" for wl,_ in WINDOWS) + " |")
    REPORT = ROOT / "reports" / "exp_step1_v6.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    logger.success("0050基準 2年 " + f"{bench['2年']:+.0f}%")

if __name__ == "__main__":
    main()