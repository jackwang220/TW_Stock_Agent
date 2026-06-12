"""Step1 v4:聚焦 C/D/L + 大盤濾網 G 組合,加 1年半窗口,加手續費+漲停買不到。
問題:① C/D/L 在 1年半長怎樣 ② +G 能不能把 2年的 overfit 補起來 ③ 扣手續費後誰活(換檔越兇扣越多)。
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

_v3s = importlib.util.spec_from_file_location("v3", ROOT / "scripts/exp_step1_v3.py")
v3 = importlib.util.module_from_spec(_v3s); _v3s.loader.exec_module(v3)
_r60s = importlib.util.spec_from_file_location("r60", ROOT / "scripts/run_backtest_60d.py")
r60 = importlib.util.module_from_spec(_r60s); _r60s.loader.exec_module(r60)
features, _factors, oh, clamp = v3.features, v3._factors, v3.oh, v3.clamp

END = "2026-06-08"
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
FEE_BUY, FEE_SELL = 0.001425, 0.004425
REPORT = ROOT / "reports" / "exp_step1_v4.md"

# 變體 = base + 修飾旗標。L=C+D=break6+anti。+G=加 regime。
VAR = {
 "A基準":     dict(base="eq5"),
 "C防追高":   dict(base="eq5", anti=True),
 "D突破":     dict(base="break6"),
 "L(C+D)":    dict(base="break6", anti=True),
 "G大盤濾網": dict(base="eq5", regime=True),
 "C+G":       dict(base="eq5", anti=True, regime=True),
 "D+G":       dict(base="break6", regime=True),
 "L+G":       dict(base="break6", anti=True, regime=True),
}

def mom_score(cfg, ff, ex):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    if cfg["base"] == "eq5": s = (t+rs+vo+ri+ma)/5
    elif cfg["base"] == "break6": s = (t+rs+vo+ri+ma+br)/6
    else: s = 0.0
    s *= 100
    if cfg.get("anti"):
        s *= (1.0 if bias <= 0.12 else clamp(1-(bias-0.12)/0.13, 0.5, 1.0))
    if cfg.get("regime"): s *= ex["regime_mult"]
    return s

def build_rows(codes, names, feats, twii_feat, reb_cache, cfg, dates):
    rows = []
    for d in dates:
        tf = twii_feat.get(d, {}); ir = tf.get("ret20")
        regime_mult = 1.0 if (tf.get("close") and tf.get("ma20") and tf["close"] > tf["ma20"]) else 0.7
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            sc = max(mom_score(cfg, _factors(f[d], ir), {"regime_mult": regime_mult}),
                     reb_cache.get(c, {}).get(d, 0.0))
            if sc <= 0: continue
            rows.append({"date": d, "ticker": c, "name": names.get(c, c),
                         "predicted_center_pct": "100", "prediction_confidence": f"{sc/100:.4f}",
                         "predicted_direction": "up", "llm_verdict": "PASS",
                         "pattern_type": "", "bull_score": "", "bear_score": ""})
    return rows

def paper_trade_cost(rows, limitup):
    """= _paper_trade + 手續費(買FEE_BUY/賣FEE_SELL)+ 漲停當日不買 + 換手率統計。"""
    from collections import defaultdict
    from tw_stock_agent.tools.finmind_client import get_daily_prices
    sig_by_date = defaultdict(dict); tickers = set()
    for r in rows:
        d, tk = r["date"], r["ticker"]; tickers.add(tk)
        center = float(r["predicted_center_pct"] or 0); conf = float(r["prediction_confidence"] or 0)
        sig_by_date[d][tk] = {"name": r["name"], "conf": conf,
                              "edge": max(0.0, center)/100*max(0.0, conf)}
    if not sig_by_date: return r60._empty_trade_result()
    panel = {tk: get_daily_prices(tk) for tk in tickers}
    def price(tk, d):
        pr = panel.get(tk, {}); ds = [x for x in pr if x <= d]; return pr[max(ds)] if ds else None
    first, last = min(sig_by_date), max(sig_by_date)
    cal = sorted({d for pr in panel.values() for d in pr if first <= d <= last})
    if not cal: return r60._empty_trade_result()
    cash = contributed = prev_eq = traded = fees = 0.0
    shares, last_edge = {}, {}; day_pnl, eq_curve = [], []
    for i, d in enumerate(cal):
        add = r60.INITIAL_CAPITAL if i == 0 else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contributed) if contributed < r60.MAX_CONTRIBUTION else 0.0)
        cash += add; contributed += add
        port = cash + sum(shares[tk]*(price(tk, d) or 0) for tk in shares)
        todays = sig_by_date.get(d, {}); edges, conf_of = {}, {}
        for tk, s in todays.items(): edges[tk] = s["edge"]; conf_of[tk] = s["conf"]; last_edge[tk] = s["edge"]
        for tk in shares:
            if tk not in edges:
                e = last_edge.get(tk, 0.0)*r60.EDGE_DECAY; edges[tk] = e; last_edge[tk] = e
        rank = lambda tk: edges[tk]*(r60.INCUMBENT_BONUS if tk in shares else 1.0)
        ranked = sorted([tk for tk in edges if edges[tk] > 0], key=rank, reverse=True)
        sel = ranked[:r60.MAX_SIGNALS]
        if len(ranked) > r60.MAX_SIGNALS and rank(ranked[r60.MAX_SIGNALS]) >= rank(ranked[r60.MAX_SIGNALS-1])*r60.TIE_RATIO:
            sel = ranked[:r60.MAX_SIGNALS+1]
        confs = [conf_of[tk] for tk in sel if tk in conf_of]; avg_conf = sum(confs)/len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg_conf)) if sel else 0.0
        wsum = sum(edges[tk] for tk in sel); targets = {}
        if wsum > 0 and expo > 0:
            for tk in sel: targets[tk] = port*expo*(edges[tk]/wsum)
        for tk in set(shares) | set(targets):
            p = price(tk, d)
            if not p: continue
            cur = shares.get(tk, 0.0)*p; tgt = targets.get(tk, 0.0); delta = tgt-cur
            if delta > r60.DAILY_ADD_CAP: delta = r60.DAILY_ADD_CAP; tgt = cur+delta
            if delta > 0 and d in limitup.get(tk, set()): continue   # 漲停買不到 → 不調整
            fee = (FEE_BUY if delta > 0 else FEE_SELL)*abs(delta)
            cash -= delta + fee; traded += abs(delta); fees += fee
            if tgt <= 1e-6: shares.pop(tk, None)
            else: shares[tk] = tgt/p
        equity = cash + sum(shares[tk]*(price(tk, d) or 0) for tk in shares)
        pnl = equity - prev_eq - add; prev_eq = equity
        day_pnl.append(pnl); eq_curve.append(equity)
    total_pnl = sum(day_pnl); active = [x for x in day_pnl if abs(x) > 1e-9]
    cum = peak = mdd = 0.0
    for x in day_pnl: cum += x; peak = max(peak, cum); mdd = max(mdd, peak-cum)
    if len(active) > 1:
        m = sum(active)/len(active); sd = math.sqrt(sum((x-m)**2 for x in active)/len(active))
        sharpe = (m/sd*math.sqrt(252)) if sd > 0 else 0.0
    else: sharpe = 0.0
    return {"total_pnl": total_pnl, "return_pct": total_pnl/contributed*100 if contributed else 0,
            "final_equity": eq_curve[-1] if eq_curve else 0, "max_dd": mdd, "sharpe": sharpe,
            "win_days": sum(1 for x in active if x > 0), "total_days": len(active),
            "total_contributed": contributed, "fees": fees, "turnover_x": traded/contributed if contributed else 0}

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("特徵 + 反彈 + 漲停表...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    cal = [d for d in sorted(twii_feat) if d <= END][-WINDOWS[-1][1]:]
    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); closes = []; m = {}; s = set()
        for j, d in enumerate(ds):
            closes.append(o[d]["close"])
            if len(closes) >= 25:
                try:
                    sig = rebound_signal(closes, turns.get(c, 0.0))
                    if sig.get("fired"): m[d] = sig["score"]*100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095: s.add(d)
        reb_cache[c] = m; limitup[c] = s

    grid = {}  # (vname, win, mode) -> pt
    for vn, cfg in VAR.items():
        rows_full = build_rows(codes, names, feats, twii_feat, reb_cache, cfg, cal)
        for wl, n in WINDOWS:
            wd = set(cal[-n:]); rw = [r for r in rows_full if r["date"] in wd]
            grid[(vn, wl, "净")] = r60._paper_trade(rw)
            grid[(vn, wl, "费")] = paper_trade_cost(rw, limitup)
        logger.info(f"{vn} 完成")

    def table(mode, key, fmt, title):
        L = [f"## {title}\n", "| 變體 | 60天 | 90天 | 半年 | 1年 | 1年半 | 2年 |", "|---|---|---|---|---|---|---|"]
        for vn in VAR:
            L.append(f"| {vn} | " + " | ".join(fmt(grid[(vn, wl, mode)][key]) for wl, _ in WINDOWS) + " |")
        return L
    L = ["# Step1 v4:C/D/L × 大盤濾網G,加1年半窗口 + 真實手續費(買0.14%/賣0.44%)+漲停買不到\n",
         f"> 結束 {END}｜112檔｜資金15000+1000/日(上限5萬)｜最多3檔｜沿用_paper_trade的抱法\n",
         "## 【扣手續費後】本金報酬率 %（這才是實戰)\n",
         "| 變體 | 60天 | 90天 | 半年 | 1年 | 1年半 | 2年 |", "|---|---|---|---|---|---|---|"]
    for vn in VAR:
        L.append(f"| {vn} | " + " | ".join(f"{grid[(vn,wl,'费')]['return_pct']:+.1f}" for wl,_ in WINDOWS) + " |")
    L += [""] + table("费", "sharpe", lambda x: f"{x:.2f}", "【扣手續費後】Sharpe")
    L += [""] + table("费", "max_dd", lambda x: f"-{x:,.0f}", "【扣手續費後】最大回撤")
    L += ["", "## 換手率(traded/投入,越高換越兇=越被手續費咬)+ 2年付了多少手續費\n",
          "| 變體 | 2年換手x | 2年手續費TWD | 淨報酬2年 | 費報酬2年 | 手續費吃掉 |", "|---|---|---|---|---|---|"]
    for vn in VAR:
        g_net = grid[(vn,"2年","净")]; g_fee = grid[(vn,"2年","费")]
        L.append(f"| {vn} | {g_fee['turnover_x']:.1f}x | -{g_fee['fees']:,.0f} | {g_net['return_pct']:+.1f}% "
                 f"| {g_fee['return_pct']:+.1f}% | {g_net['return_pct']-g_fee['return_pct']:.1f}pp |")
    L += ["", "## 參考:【無手續費】本金報酬率 %\n",
          "| 變體 | 60天 | 90天 | 半年 | 1年 | 1年半 | 2年 |", "|---|---|---|---|---|---|---|"]
    for vn in VAR:
        L.append(f"| {vn} | " + " | ".join(f"{grid[(vn,wl,'净')]['return_pct']:+.1f}" for wl,_ in WINDOWS) + " |")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    for vn in VAR:
        logger.success(f"{vn}: 費報酬 " + " ".join(f"{wl}{grid[(vn,wl,'费')]['return_pct']:+.0f}" for wl,_ in WINDOWS)
                       + f" | 換手{grid[(vn,'2年','费')]['turnover_x']:.1f}x")

if __name__ == "__main__":
    main()