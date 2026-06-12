"""Step1 v5:15 變體 × 4 降換手設定 × 6 窗口,全部扣真實手續費。找「扣費後最穩的真實贏家」。
降換手:incumbent(換手門檻)+ hysteresis(差距<band×權益就不交易)。
排名 = min(1年, 1年半, 2年 的扣費報酬)(獎勵長期穩定為正)。
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
from tw_stock_agent.tools.finmind_client import get_daily_prices

_v3s = importlib.util.spec_from_file_location("v3", ROOT / "scripts/exp_step1_v3.py")
v3 = importlib.util.module_from_spec(_v3s); _v3s.loader.exec_module(v3)
_r60s = importlib.util.spec_from_file_location("r60", ROOT / "scripts/run_backtest_60d.py")
r60 = importlib.util.module_from_spec(_r60s); _r60s.loader.exec_module(r60)
features, _factors, oh, clamp = v3.features, v3._factors, v3.oh, v3.clamp

END = "2026-06-08"
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
FEE_BUY, FEE_SELL = 0.001425, 0.004425

VAR = {
 "A": dict(base="eq5"), "B": dict(base="trendRS"), "C": dict(base="eq5", anti=True),
 "D": dict(base="break6"), "E": dict(base="eq5", reb=True), "F": dict(base="break6", vcp=True),
 "G": dict(base="eq5", regime=True), "H": dict(base="trendRS", turn=True),
 "I": dict(base="eq5", reb=True, panic=True), "J": dict(base="trendRS", anti=True, regime=True),
 "K": dict(base="break6", vcp=True, turn=True), "L": dict(base="break6", anti=True),
 "C+G": dict(base="eq5", anti=True, regime=True), "D+G": dict(base="break6", regime=True),
 "L+G": dict(base="break6", anti=True, regime=True),
}
CHURN = {"base": (1.2, 0.0), "inc1.5": (1.5, 0.0), "hyst5": (1.2, 0.05), "inc1.5+hyst5": (1.5, 0.05)}

def total_score(cfg, ff, ex):
    reb = ex["reb"]
    if cfg.get("panic") and ex["panic"] and reb > 0: reb = 100.0
    if cfg.get("reb"): reb = min(100.0, reb*1.2)
    m = 0.0
    if ff is not None:
        t, rs, vo, ri, ma, br, bias = ff
        b = cfg["base"]
        if b == "eq5": m = (t+rs+vo+ri+ma)/5
        elif b == "trendRS": m = 0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma
        elif b == "break6": m = (t+rs+vo+ri+ma+br)/6
        m *= 100
        if cfg.get("anti"): m *= (1.0 if bias <= 0.12 else clamp(1-(bias-0.12)/0.13, 0.5, 1.0))
        if cfg.get("regime"): m *= ex["regime_mult"]
        if cfg.get("vcp"): m *= (1.0 if ex["vcp"] else 0.5)
        if cfg.get("turn"): m *= (0.8 + 0.4*ex["turn_pct"])
    return max(m, reb)

def build_rows(codes, names, feats, twii_feat, reb_cache, turn_pct, cfg, dates):
    rows = []
    for d in dates:
        tf = twii_feat.get(d, {}); ir = tf.get("ret20")
        regime_mult = 1.0 if (tf.get("close") and tf.get("ma20") and tf["close"] > tf["ma20"]) else 0.7
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            fd = f[d]
            ex = {"regime_mult": regime_mult, "turn_pct": turn_pct.get(d, {}).get(c, 0.5),
                  "vcp": fd["vcp"], "reb": reb_cache.get(c, {}).get(d, 0.0),
                  "panic": fd["gap"] <= -0.03 and (not math.isnan(fd["volr"]) and fd["volr"] >= 1.5)}
            sc = total_score(cfg, _factors(fd, ir), ex)
            if sc <= 0: continue
            rows.append((d, c, sc/100))
    return rows

def sim(rows, panel, limitup, incumbent, hyst):
    sig = defaultdict(dict); tickers = set()
    for d, tk, conf in rows:
        tickers.add(tk); sig[d][tk] = conf   # edge = conf(=score/100), center=100 → edge=conf
    if not sig: return None
    def price(tk, d):
        pr = panel.get(tk, {}); ds = [x for x in pr if x <= d]; return pr[max(ds)] if ds else None
    first, last = min(sig), max(sig)
    cal = sorted({d for tk in tickers for d in panel.get(tk, {}) if first <= d <= last})
    if not cal: return None
    cash = contributed = prev_eq = traded = fees = 0.0
    shares, last_edge = {}, {}; day_pnl = []
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
        confs = [todays[tk] for tk in sel if tk in todays]; avg_conf = sum(confs)/len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg_conf)) if sel else 0.0
        wsum = sum(edges[tk] for tk in sel); targets = {}
        if wsum > 0 and expo > 0:
            for tk in sel: targets[tk] = port*expo*(edges[tk]/wsum)
        for tk in set(shares) | set(targets):
            p = price(tk, d)
            if not p: continue
            cur = shares.get(tk, 0.0)*p; tgt = targets.get(tk, 0.0); delta = tgt-cur
            if delta > r60.DAILY_ADD_CAP: delta = r60.DAILY_ADD_CAP; tgt = cur+delta
            if abs(delta) < hyst*port: continue                       # 遲滯帶:差距太小不動
            if delta > 0 and d in limitup.get(tk, set()): continue    # 漲停買不到
            fee = (FEE_BUY if delta > 0 else FEE_SELL)*abs(delta)
            cash -= delta + fee; traded += abs(delta); fees += fee
            if tgt <= 1e-6: shares.pop(tk, None)
            else: shares[tk] = tgt/p
        equity = cash + sum(shares[tk]*(price(tk, d) or 0) for tk in shares)
        day_pnl.append(equity - prev_eq - add); prev_eq = equity
    total = sum(day_pnl); active = [x for x in day_pnl if abs(x) > 1e-9]
    cum = peak = mdd = 0.0
    for x in day_pnl: cum += x; peak = max(peak, cum); mdd = max(mdd, peak-cum)
    if len(active) > 1:
        m = sum(active)/len(active); sd = math.sqrt(sum((x-m)**2 for x in active)/len(active))
        shp = (m/sd*math.sqrt(252)) if sd > 0 else 0.0
    else: shp = 0.0
    return {"ret": total/contributed*100 if contributed else 0, "mdd": mdd, "sharpe": shp,
            "turn": traded/contributed if contributed else 0, "fees": fees}

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("特徵/反彈/漲停/成交值/價格面板...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    cal = [d for d in sorted(twii_feat) if d <= END][-WINDOWS[-1][1]:]
    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); closes = []; m = {}; s = set()
        for j, d in enumerate(ds):
            closes.append(o[d]["close"])
            if len(closes) >= 25:
                try:
                    g = rebound_signal(closes, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095: s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct = {}
    for d in cal:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        n = len(vals); turn_pct[d] = {c: (i+1)/n for i, (c, _) in enumerate(vals)}
    panel = {c: get_daily_prices(c) for c in codes}

    res = {}   # (var, churn, win) -> sim
    for vn, cfg in VAR.items():
        rows_full = build_rows(codes, names, feats, twii_feat, reb_cache, turn_pct, cfg, cal)
        for cn, (inc, hy) in CHURN.items():
            for wl, n in WINDOWS:
                wd = set(cal[-n:])
                res[(vn, cn, wl)] = sim([r for r in rows_full if r[0] in wd], panel, limitup, inc, hy)
        logger.info(f"{vn} 完成")

    # 排名:min(1年,1年半,2年 扣費報酬)
    def g(vn, cn, wl, k="ret"):
        r = res.get((vn, cn, wl)); return r[k] if r else -999
    board = []
    for vn in VAR:
        for cn in CHURN:
            robust = min(g(vn,cn,"1年"), g(vn,cn,"1年半"), g(vn,cn,"2年"))
            board.append((robust, vn, cn))
    board.sort(reverse=True)

    L = ["# Step1 v5:15變體 × 4降換手 × 6窗口,全扣真實手續費(買0.14%/賣0.44%)+漲停買不到\n",
         f"> 結束{END}｜112檔｜資金15000+1000/日(上限5萬)｜最多3檔｜排名=min(1年,1年半,2年扣費報酬)=長期最穩為正\n",
         "## 🏆 排行榜(扣費後,前 22 名;穩定=min(1年/1年半/2年)高)\n",
         "| 排名 | 變體 | 降換手 | 60天 | 90天 | 半年 | 1年 | 1年半 | 2年 | 穩定分 | 2年換手x |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (robust, vn, cn) in enumerate(board[:22], 1):
        cells = " | ".join(f"{g(vn,cn,wl):+.0f}" for wl,_ in WINDOWS)
        L.append(f"| {i} | {vn} | {cn} | {cells} | **{robust:+.0f}** | {g(vn,cn,'2年','turn'):.0f}x |")
    L += ["", "## 降換手對換手率/扣費報酬的效果(看 2年)\n",
          "| 變體 | base換手 | base費後2年 | inc1.5+hyst5換手 | 後者費後2年 |", "|---|---|---|---|---|"]
    for vn in ["A","D","G","D+G","L","C+G","L+G"]:
        L.append(f"| {vn} | {g(vn,'base','2年','turn'):.0f}x | {g(vn,'base','2年'):+.0f}% "
                 f"| {g(vn,'inc1.5+hyst5','2年','turn'):.0f}x | {g(vn,'inc1.5+hyst5','2年'):+.0f}% |")
    REPORT = ROOT / "reports" / "exp_step1_v5.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    logger.success("前5名: " + " || ".join(f"{vn}/{cn} 穩{robust:+.0f}" for robust, vn, cn in board[:5]))

if __name__ == "__main__":
    main()