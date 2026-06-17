"""retest_turnover_kill — 換手 dose-response 甜蜜點掃描(診斷 turnover_kill 結論)。

原 §3 結論:每天再平衡換手 156-208x、真實費吃掉 100-136pp、多策略由正翻負。
原方法瑕疵:只測 incumbent 1.0/1.5/2.0 三點;沒連續掃 incumbent;
           沒掃「遲滯帶寬/最小交易門檻(switch_cost_mult)」;沒加最小持有天數。

本腳本:固定 B純切雙引擎 + ⑤收盤買開盤只賣(exit_only,對齊實盤),全史2021~還原價,
        對每個 (incumbent, switch_cost_mult, min_hold) 組合跑 6 窗口 + 5 regime,
        報告 換手x vs 淨ALPHA(扣0050 buy&hold),找甜蜜點並檢查跨 regime 穩定性。

引擎重用 v14 的資料管線;sim 用本檔自帶版(複製 sim_buyclose_sellopen 並加 min_hold/switch_cost_mult),
不改既有 exp_*.py。
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
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

v5 = _load("v5", "scripts/exp_step1_v5.py")
v6 = _load("v6", "scripts/exp_step1_v6.py")
ec = _load("ec", "scripts/exp_60d_entry_compare.py")
features, _factors = v5.features, v5._factors
bench_0050 = v6.bench_0050
r60 = ec.r60
FEE_BUY, FEE_SELL, SLIP = ec.FEE_BUY, ec.FEE_SELL, ec.SLIP
ROUNDTRIP_COST = ec.ROUNDTRIP_COST

START, END = "2021-01-01", "2026-06-08"
WINDOWS = [("1年", 252), ("1年半", 378), ("2年", 504)]
REGIMES = [("2021復甦", "2021-04-01", "2021-12-31"), ("2022空頭", "2022-01-01", "2022-12-31"),
           ("2023復甦", "2023-01-01", "2023-12-31"), ("2024-25多頭", "2024-01-01", "2025-06-30"),
           ("2025下-26", "2025-07-01", "2026-06-08")]


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def sim(rows, opens, closes, limitup, incumbent, switch_cost_mult=1.0, min_hold=0):
    """B純切 exit_only。加 min_hold(最小持有天數,持有天數<min_hold時不允許賣出/掉出)
    與 switch_cost_mult(遲滯帶寬:交易門檻 = mult × 0.78% 來回成本)。"""
    sig = defaultdict(dict); tickers = set()
    for d, tk, e in rows:
        tickers.add(tk); sig[d][tk] = e
    if not sig: return None
    alld = sorted({d for tk in tickers for d in closes.get(tk, {})})
    first, last = min(sig), max(sig)
    cal = [d for d in alld if first <= d <= last]
    if len(cal) < 2: return None
    def cl(tk, d):
        c = closes.get(tk, {}); ds = [x for x in c if x <= d]; return c[max(ds)] if ds else None
    thresh = switch_cost_mult * ROUNDTRIP_COST
    cash = contributed = prev_eq = traded = fees = 0.0
    shares = {}; last_edge = {}; hold_days = {}; day_pnl = []; pos_track = []
    def _buy(tk, amt, px):
        nonlocal cash, traded, fees
        slip_cost = amt*SLIP; fee = FEE_BUY*amt
        cash -= amt + slip_cost + fee; traded += amt; fees += fee + slip_cost
        shares[tk] = shares.get(tk, 0.0) + amt/px
    for i, d in enumerate(cal):
        add = (r60.INITIAL_CAPITAL if i == 0 else
               (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contributed) if contributed < r60.MAX_CONTRIBUTION else 0.0))
        cash += add; contributed += add
        for tk in shares: hold_days[tk] = hold_days.get(tk, 0) + 1
        if i+1 >= len(cal):
            eq = cash + sum(shares[tk]*(cl(tk, d) or 0) for tk in shares)
            day_pnl.append(eq - prev_eq - add); break
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
        # min_hold:持有未滿天數的持股強制保留在名單內(不准賣)
        if min_hold > 0:
            for tk in list(shares):
                if hold_days.get(tk, 0) < min_hold and tk not in sel:
                    sel.append(tk)
        confs = [todays[tk] for tk in sel if tk in todays]; avg = sum(confs)/len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg)) if sel else 0.0
        wsum = sum(edges[tk] for tk in sel); targets = {}
        if wsum > 0 and expo > 0:
            for tk in sel: targets[tk] = port*expo*(edges[tk]/wsum)
        added_today = {}
        # A @ d 收盤買
        for tk in sel:
            cp = closes.get(tk, {}).get(d)
            if not cp or cp <= 0: continue
            cur = shares.get(tk, 0.0)*cp
            cap_left = r60.DAILY_ADD_CAP - added_today.get(tk, 0.0)
            buy_amt = min(targets.get(tk, 0.0)-cur, cap_left)
            if buy_amt <= 0 or buy_amt < thresh*port: continue
            if d in limitup.get(tk, set()): continue
            _buy(tk, buy_amt, cp); added_today[tk] = added_today.get(tk, 0.0)+buy_amt
        # B1 @ d+1 開盤只賣(出場)
        for tk in list(shares):
            op = opens.get(tk, {}).get(e)
            if not op or op <= 0: continue
            tgt = targets.get(tk, 0.0)
            if tgt > 1e-6: continue                       # exit_only:名單內不賣
            if min_hold > 0 and hold_days.get(tk, 0) < min_hold: continue  # 未滿最小持有天不賣
            cur = shares[tk]*op; delta = tgt - cur
            if delta >= 0: continue
            if abs(delta) < thresh*port: continue
            sell_amt = abs(delta); slip_cost = sell_amt*SLIP; fee = FEE_SELL*sell_amt
            cash += sell_amt - slip_cost - fee; traded += sell_amt; fees += fee + slip_cost
            shares[tk] = tgt/op
            if shares[tk] <= 1e-6: shares.pop(tk, None); hold_days.pop(tk, None)
        invested = sum(shares[tk]*(cl(tk, e) or 0) for tk in shares)
        eq = cash + invested
        pos_track.append(len([tk for tk in shares if shares[tk] > 1e-6]))
        day_pnl.append(eq - prev_eq - add); prev_eq = eq
    total = sum(day_pnl)
    return {"ret": total/contributed*100 if contributed else 0,
            "turn": traded/contributed if contributed else 0,
            "avg_pos": sum(pos_track)/len(pos_track) if pos_track else 0.0}


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("載入全史還原 OHLCV ...")
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
    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    rows = []  # B純切
    for d in alld:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            rb = reb_cache.get(c, {}).get(d, 0.0)
            sc = hh if bull else rb*1.5
            if sc > 0: rows.append((d, c, sc/100))

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    def run(inc, scm, mh):
        out = {}
        for col in COLS:
            ds = col_dates[col]
            out[col] = sim([x for x in rows if x[0] in ds], opens, closes, limitup,
                           incumbent=inc, switch_cost_mult=scm, min_hold=mh)
        return out

    def alpha(res, col):
        r = res.get(col); return (r["ret"]-bench[col]) if r else None

    L = ["# retest_turnover_kill — 換手 dose-response 甜蜜點掃描\n",
         f"> B純切雙引擎｜⑤收盤買開盤只賣(exit_only)｜還原價｜112檔｜DCA(15000+1000/日上限5萬)\n",
         f"> ALPHA=策略−同資金DCA0050;成本 買{FEE_BUY*100:.3f}%/賣{FEE_SELL*100:.3f}%/滑價{SLIP*100:.1f}%+漲停買不到\n",
         "> 0050 基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n"]

    # ── 掃描1:incumbent 連續(switch_cost_mult=1.0 預設,min_hold=0)──
    logger.info("=== 掃描1:incumbent 連續 ===")
    L += ["## 掃描1 — incumbent 連續(switch_cost_mult=1.0, min_hold=0)\n",
          "| incumbent | " + " | ".join(COLS) + " | 最差 | 平均 | 換手2年 | 持股2年 |",
          "|" + "---|"*(len(COLS)+5)]
    inc_results = {}
    for inc in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
        res = run(inc, 1.0, 0); inc_results[inc] = res
        vals = [alpha(res, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" for x in vals)
        r2 = res.get("2年")
        L.append(f"| {inc} | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {r2['turn']:.0f}x | {r2['avg_pos']:.1f} |")
        logger.info(f"  inc={inc} done")

    # ── 掃描2:switch_cost_mult 遲滯帶寬連續(inc=1.5,min_hold=0)──
    logger.info("=== 掃描2:switch_cost_mult 遲滯帶寬 ===")
    L += ["", "## 掃描2 — 遲滯帶寬 switch_cost_mult(交易門檻=mult×0.78%來回成本;inc=1.5)\n",
          "| scm(門檻%) | " + " | ".join(COLS) + " | 最差 | 平均 | 換手2年 | 持股2年 |",
          "|" + "---|"*(len(COLS)+5)]
    for scm in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        res = run(1.5, scm, 0)
        vals = [alpha(res, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" for x in vals)
        r2 = res.get("2年")
        L.append(f"| {scm}({scm*ROUNDTRIP_COST*100:.1f}%) | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {r2['turn']:.0f}x | {r2['avg_pos']:.1f} |")
        logger.info(f"  scm={scm} done")

    # ── 掃描3:最小持有天數(inc=1.5,scm=1.0)──
    logger.info("=== 掃描3:min_hold ===")
    L += ["", "## 掃描3 — 最小持有天數 min_hold(inc=1.5, scm=1.0)\n",
          "| min_hold | " + " | ".join(COLS) + " | 最差 | 平均 | 換手2年 | 持股2年 |",
          "|" + "---|"*(len(COLS)+5)]
    for mh in [0, 2, 3, 5, 8, 13]:
        res = run(1.5, 1.0, mh)
        vals = [alpha(res, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" for x in vals)
        r2 = res.get("2年")
        L.append(f"| {mh}天 | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {r2['turn']:.0f}x | {r2['avg_pos']:.1f} |")
        logger.info(f"  mh={mh} done")

    REPORT = ROOT / "reports" / "retest_turnover_kill.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")


if __name__ == "__main__":
    main()
