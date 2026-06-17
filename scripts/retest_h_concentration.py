"""retest_h_concentration:嚴格檢定 H(成交值聚光燈)是 robust edge 還是「壓中記憶體循環的運氣」。

對照原結論「77-88% 利潤集中3檔=運氣」做三個真正的穩健性檢定(不是事後減 PnL):
 (a) 單股權重上限 CAP(25%/15%):限制曝險集中度後重跑,看 H alpha 還在嗎。
 (b) 真·leave-N-out:把貢獻前 N 名「整檔逐出 universe」後完整重跑(資金真的釋放、重新配置),
     而非原報告「總 PnL 減掉前3大 PnL / 同分母」的偽剔除。
 (c) 非記憶體子集:把記憶體股(華邦/南亞科/國巨/南電)整批逐出後重跑,看 edge 是否消失。
全扣真實手續費(買0.1425%/賣0.4425%);扣 0050 buy&hold 做 alpha;報告換手與每檔貢獻。

重用 v5 引擎(features/_factors),sim 重寫加單股 CAP + 可禁名單。
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

_v5s = importlib.util.spec_from_file_location("v5", ROOT / "scripts/exp_step1_v5.py")
v5 = importlib.util.module_from_spec(_v5s); _v5s.loader.exec_module(v5)
_r60s = importlib.util.spec_from_file_location("r60", ROOT / "scripts/run_backtest_60d.py")
r60 = importlib.util.module_from_spec(_r60s); _r60s.loader.exec_module(r60)
features, _factors, oh = v5.features, v5._factors, v5.oh

END = "2026-06-08"
WINDOWS = [("1年", 252), ("1年半", 378), ("2年", 504)]
FEE_BUY, FEE_SELL = 0.001425, 0.004425


def h_edge(ff, turn_pct):
    """H = trendRS base × 成交值百分位調整。回傳 edge(0-1)。"""
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    m = (0.35*t + 0.35*rs + 0.15*vo + 0.10*ri + 0.05*ma) * 100
    m *= (0.8 + 0.4*turn_pct)
    return max(m, 0.0) / 100


def sim(rows, panel, limitup, incumbent, hyst, ban=None, stock_cap=None, track=False):
    """ban: set 直接逐出 universe(訊號與持股都剔除);stock_cap: 單股權重上限(占組合)。"""
    ban = ban or set()
    sig = defaultdict(dict); tickers = set()
    for d, tk, e in rows:
        if tk in ban: continue
        tickers.add(tk); sig[d][tk] = e
    if not sig: return None
    def price(tk, d):
        pr = panel.get(tk, {}); ds = [x for x in pr if x <= d]; return pr[max(ds)] if ds else None
    first, last = min(sig), max(sig)
    cal = sorted({d for tk in tickers for d in panel.get(tk, {}) if first <= d <= last})
    if not cal: return None
    cash = contributed = prev_eq = traded = fees = 0.0
    shares, last_edge = {}, {}; day_pnl = []
    pnl_by_tk = defaultdict(float)
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
            if stock_cap is not None:                  # 單股權重上限(占組合);超過砍到上限
                capv = port*stock_cap
                for tk in list(targets):
                    if targets[tk] > capv: targets[tk] = capv
        for tk in set(shares) | set(targets):
            p = price(tk, d)
            if not p: continue
            cur = shares.get(tk, 0.0)*p; tgt = targets.get(tk, 0.0); delta = tgt-cur
            if delta > r60.DAILY_ADD_CAP: delta = r60.DAILY_ADD_CAP; tgt = cur+delta
            if abs(delta) < hyst*port: continue
            if delta > 0 and d in limitup.get(tk, set()): continue
            fee = (FEE_BUY if delta > 0 else FEE_SELL)*abs(delta)
            cash -= delta + fee; traded += abs(delta); fees += fee
            pnl_by_tk[tk] -= delta + fee
            if tgt <= 1e-6: shares.pop(tk, None)
            else: shares[tk] = tgt/p
        equity = cash + sum(shares[tk]*(price(tk, d) or 0) for tk in shares)
        day_pnl.append(equity - prev_eq - add); prev_eq = equity
    for tk in shares: pnl_by_tk[tk] += shares[tk]*(price(tk, last) or 0)
    total = sum(day_pnl)
    out = {"ret": total/contributed*100 if contributed else 0, "turn": traded/contributed if contributed else 0,
           "total_pnl": total, "contributed": contributed}
    if track: out["pnl_by_tk"] = dict(pnl_by_tk)
    return out


def bench_0050(panel, cal):
    """0050 buy&hold,同樣的每日定額注資(報酬率%)。"""
    p = panel.get("0050", {})
    ds = [d for d in cal if d in p or any(x <= d for x in p)]
    def price(d):
        xs = [x for x in p if x <= d]; return p[max(xs)] if xs else None
    contributed = units = 0.0
    for i, d in enumerate(cal):
        add = r60.INITIAL_CAPITAL if i == 0 else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contributed) if contributed < r60.MAX_CONTRIBUTION else 0.0)
        contributed += add; pr = price(d)
        if pr: units += add/pr
    last = price(cal[-1]); val = units*last if last else 0.0
    return (val-contributed)/contributed*100 if contributed else 0.0


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    MEM = {c for c in codes if any(k in names[c] for k in ["華邦", "南亞科", "國巨", "南電"])}
    logger.info(f"記憶體股 {len(MEM)}: " + ", ".join(names[c] for c in MEM))
    logger.info("特徵/反彈/漲停/成交值/價格...")
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
        n = len(vals); turn_pct[d] = {c: (i+1)/n for i, (c, _) in enumerate(vals)} if vals else {}
    panel = {c: get_daily_prices(c) for c in codes}; panel["0050"] = get_daily_prices("0050")

    def rows_for(dates):
        out = []
        for d in dates:
            ir = twii_feat.get(d, {}).get("ret20")
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                e = h_edge(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                if e > 0: out.append((d, c, e))
        return out

    INC, HY = 1.5, 0.05
    allrows = rows_for(cal)
    bench = {wl: bench_0050(panel, cal[-n:]) for wl, n in WINDOWS}

    L = ["# H 集中度穩健性嚴格檢定(retest_h_concentration)\n",
         "> 全扣手續費(買0.14%/賣0.44%)+漲停買不到｜inc1.5+hyst5｜alpha=H−0050買抱｜2021~2026-06-08(大多頭,缺長空頭)\n",
         f"> 0050買抱報酬: " + " ".join(f"{wl}{bench[wl]:+.0f}%" for wl, _ in WINDOWS) + "\n"]

    # ── baseline + 每檔貢獻 ──
    L += ["## 0. baseline H + 貢獻分解\n",
          "| 窗口 | H報酬% | alpha vs 0050 | 換手x | 前1 | 前3佔比 | 前5貢獻股 |", "|---|---|---|---|---|---|---|"]
    base_pt = {}
    for wl, n in WINDOWS:
        wd = set(cal[-n:]); r = sim([x for x in allrows if x[0] in wd], panel, limitup, INC, HY, track=True)
        pt = sorted(r["pnl_by_tk"].items(), key=lambda x: -x[1]); base_pt[wl] = pt
        tot = r["total_pnl"]; top1 = pt[0][1]/tot*100; top3 = sum(v for _, v in pt[:3])/tot*100
        top5 = ", ".join(f"{names.get(tk,tk)[:3]}{v/tot*100:.0f}%" for tk, v in pt[:5])
        L.append(f"| {wl} | {r['ret']:+.0f} | {r['ret']-bench[wl]:+.0f} | {r['turn']:.0f}x | {top1:.0f}% | {top3:.0f}% | {top5} |")

    # ── (a) 單股權重上限 ──
    L += ["", "## (a) 單股權重上限 CAP(限制集中度後 H alpha 還在?)\n",
          "| 窗口 | CAP無 | CAP25% | CAP15% | CAP無alpha | CAP25%alpha | CAP15%alpha |", "|---|---|---|---|---|---|---|"]
    for wl, n in WINDOWS:
        wd = set(cal[-n:]); rows = [x for x in allrows if x[0] in wd]
        r0 = sim(rows, panel, limitup, INC, HY)
        r25 = sim(rows, panel, limitup, INC, HY, stock_cap=0.25)
        r15 = sim(rows, panel, limitup, INC, HY, stock_cap=0.15)
        L.append(f"| {wl} | {r0['ret']:+.0f} | {r25['ret']:+.0f} | {r15['ret']:+.0f} "
                 f"| {r0['ret']-bench[wl]:+.0f} | {r25['ret']-bench[wl]:+.0f} | {r15['ret']-bench[wl]:+.0f} |")

    # ── (b) 真·leave-N-out:整檔逐出後重跑 ──
    L += ["", "## (b) 真·leave-N-out(把貢獻前N名整檔逐出 universe 後完整重跑)\n",
          "| 窗口 | 原始 | 逐出前1 | 逐出前3 | 逐出前5 | 逐出前3 alpha | 逐出前5 alpha | 逐出名單(前3) |",
          "|---|---|---|---|---|---|---|---|"]
    for wl, n in WINDOWS:
        wd = set(cal[-n:]); rows = [x for x in allrows if x[0] in wd]
        pt = base_pt[wl]; topall = [tk for tk, _ in pt]
        r0 = sim(rows, panel, limitup, INC, HY)
        r1 = sim(rows, panel, limitup, INC, HY, ban=set(topall[:1]))
        r3 = sim(rows, panel, limitup, INC, HY, ban=set(topall[:3]))
        r5 = sim(rows, panel, limitup, INC, HY, ban=set(topall[:5]))
        ban3 = ", ".join(names.get(tk, tk)[:3] for tk in topall[:3])
        L.append(f"| {wl} | {r0['ret']:+.0f} | {r1['ret']:+.0f} | {r3['ret']:+.0f} | {r5['ret']:+.0f} "
                 f"| {r3['ret']-bench[wl]:+.0f} | {r5['ret']-bench[wl]:+.0f} | {ban3} |")

    # ── (c) 非記憶體子集 ──
    L += ["", "## (c) 非記憶體子集(把記憶體股整批逐出後重跑)\n",
          f"> 逐出: {', '.join(names[c] for c in MEM)}\n",
          "| 窗口 | 全universe | 去記憶體 | 去記憶體 alpha | 去記憶體後前5貢獻股 |", "|---|---|---|---|---|"]
    for wl, n in WINDOWS:
        wd = set(cal[-n:]); rows = [x for x in allrows if x[0] in wd]
        r0 = sim(rows, panel, limitup, INC, HY)
        rm = sim(rows, panel, limitup, INC, HY, ban=set(MEM), track=True)
        pt = sorted(rm["pnl_by_tk"].items(), key=lambda x: -x[1]); tot = rm["total_pnl"]
        top5 = ", ".join(f"{names.get(tk,tk)[:3]}{v/tot*100:.0f}%" for tk, v in pt[:5])
        L.append(f"| {wl} | {r0['ret']:+.0f} | {rm['ret']:+.0f} | {rm['ret']-bench[wl]:+.0f} | {top5} |")

    REPORT = ROOT / "reports" / "retest_h_concentration.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
