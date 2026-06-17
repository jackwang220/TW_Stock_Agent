"""retest: rebound_offregime —— 實作並回測 proposed_fix(方向一).

被否決結論: 反彈引擎 off-regime「主動倒賠」(2021 多頭 -16.7%)。
診斷 verdict = method_flawed: 真兇是『出場機制』(TP=0 + 硬 -12% 停損(含跳空穿價)
   + max_hold=5 的等權引擎),不是訊號離開 regime 會倒賠。

proposed_fix(方向一,可立即回測): 純反彈不要用硬 -12% 停損 + 5 日到期出場,
   改用 canonical ⑤ 引擎的 EDGE_DECAY 自然老化出場。

本腳本在『同一份反彈訊號 + 同一份 universe + 同一段 regime + 同樣真實成本』下,
直接把【修正前 baseline = 等權 SL-12/hold5 隔日開盤】對上
              【修正後 fix    = канonical ⑤(收盤買/開盤賣 + EDGE_DECAY 老化, 無硬停損)】,
逐 regime 報:
   - 報酬率(扣成本)、ALPHA(−0050 buy&hold, DCA 同資金模型)
   - 最差 regime alpha
   - 換手(turnover ×)
   - 集中度(top-3 個股佔 |毛利| 比重) —— 防 skew 紅旗
   - 平均曝險(avg_expo) —— 區分「低曝險少賺」vs「高曝險真倒賠」/ 提示曝險中性問題

方法論護欄:
  1. 防洩漏: rebound_signal 只吃 ≤ 當日收盤序列;進場用隔日開盤/當日收盤,無未來 bar。
  2. OOS: 全期皆 held-out(反彈參數來自 edge_scanner OOS,本回測未再調參)。
  3. 扣基準: alpha = ret − 0050 同期 DCA buy&hold。
  4. 扣真實成本: 買0.1425/賣0.4425/滑價0.1, 漲停買不到。
  5. 防 skew: 報 top-3 個股毛利集中度。
  6. 換手: 報 turnover。
  7. 曝險中性: 報兩引擎 avg_expo;若不同則於結論點明,不可直接比 alpha 高低。

資料限制: 2021-2026 為大多頭, 僅 2022 一段真空頭, 任何降曝險/避險效果難證,
   結論中明確標註。
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
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

v5 = _load("v5", "scripts/exp_step1_v5.py")
v6 = _load("v6", "scripts/exp_step1_v6.py")
r60 = _load("r60", "scripts/run_backtest_60d.py")
ec = _load("ec", "scripts/exp_60d_entry_compare.py")
features = v5.features
bench_0050 = v6.bench_0050

START, END = "2021-01-01", "2026-06-08"
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]

# baseline 等權引擎成本(與 backtest_realistic 一致: 買賣合計 0.5% + 0.1% 滑價)
B_FEE, B_SLIP, MAX_POS = 0.005, 0.001, 3
INIT, DAILY, MAXC = 15000.0, 1000.0, 50000.0


def baseline_eqweight(OH, codes, fire, s_date, e_date, alld, sl=-0.12, max_hold=5):
    """修正前: 等權、隔日開盤進場、硬 SL(含跳空穿價用開盤)、max_hold 到期出場。
    回傳 ret%, turnover×, 個股毛利 dict(供集中度), 平均曝險。"""
    cal = [d for d in alld if s_date <= d <= e_date]
    if len(cal) < 2: return None
    cash = contributed = INIT
    pos = []                         # {tk,eidx,eprice,shares,cost}
    traded = 0.0
    pnl_by_tk = defaultdict(float)   # 已實現毛利(扣成本)按股
    expo_track = []
    for di, d in enumerate(cal):
        add = (min(DAILY, MAXC-contributed) if contributed < MAXC else 0.0) if di > 0 else 0.0
        cash += add; contributed += add
        # 出場
        for p in list(pos):
            bar = OH[p["tk"]].get(d)
            if not bar: continue
            held = di - p["eidx"]; ep = p["eprice"]; xp = None
            if sl is not None and bar["low"] <= ep*(1+sl):
                xp = min(ep*(1+sl), bar["open"])      # 跳空穿價 → 用更差的開盤
            elif held >= max_hold:
                xp = bar["close"]
            if xp is not None:
                xp *= (1-B_SLIP); fee = B_FEE*p["shares"]*ep
                proceeds = p["shares"]*xp - fee
                cash += proceeds; traded += p["shares"]*xp
                pnl_by_tk[p["tk"]] += proceeds - p["cost"]
                pos.remove(p)
        # 進場: 前一日觸發 → 今日開盤
        if di > 0:
            prev = cal[di-1]; held_tks = {p["tk"] for p in pos}
            eq = cash + sum(q["shares"]*(OH[q["tk"]].get(d,{}).get("close",0)) for q in pos)
            cands = [c for c in codes if prev in fire[c] and c not in held_tks and OH[c].get(d)]
            for c in cands:
                if len(pos) >= MAX_POS: break
                bar = OH[c].get(d); op = bar["open"]
                if not op or op <= 0: continue
                if OH[c].get(prev) and OH[c][prev]["close"] > 0 and op/OH[c][prev]["close"]-1 >= 0.095:
                    continue                          # 開盤漲停買不到
                budget = min(eq*0.9/MAX_POS, cash)
                if budget < 1000: continue
                px = op*(1+B_SLIP); sh = budget/px
                cost = sh*px; cash -= cost; traded += sh*op
                pos.append({"tk": c, "eidx": di, "eprice": op, "shares": sh, "cost": cost})
        invested = sum(q["shares"]*(OH[q["tk"]].get(d,{}).get("close",0)) for q in pos)
        eqv = cash + invested
        expo_track.append(invested/eqv if eqv > 0 else 0.0)
    # 結算未平倉(以最後收盤計入毛利)
    for p in pos:
        cp = OH[p["tk"]].get(cal[-1],{}).get("close",0)
        pnl_by_tk[p["tk"]] += p["shares"]*cp - p["cost"]
    final = cash + sum(p["shares"]*(OH[p["tk"]].get(cal[-1],{}).get("close",0)) for p in pos)
    ret = (final-contributed)/contributed*100 if contributed else 0.0
    return {"ret": ret, "turn": traded/contributed if contributed else 0.0,
            "pnl_by_tk": dict(pnl_by_tk),
            "avg_expo": sum(expo_track)/len(expo_track) if expo_track else 0.0}


def concentration(pnl_by_tk):
    """top-3 個股 |毛利| 佔總 |毛利| 比重(%);越高越像運氣/skew。"""
    if not pnl_by_tk: return 0.0
    mags = sorted((abs(v) for v in pnl_by_tk.values()), reverse=True)
    tot = sum(mags)
    return sum(mags[:3])/tot*100 if tot > 0 else 0.0


# ── 修正後: канonical ⑤ 引擎,並 instrument 個股毛利供集中度 ──
def sim5_instrumented(rows, opens, closes, limitup):
    """複製 ec.sim_buyclose_sellopen 的邏輯,額外回傳 pnl_by_tk(個股已實現+未實現毛利)。
    參數與 канonical 雙引擎反彈腿一致: incumbent=1.5, EDGE_DECAY 老化, switch_cost_mult=1.0, 無硬停損。"""
    INC, SLIP = ec.INC, ec.SLIP
    FEE_BUY, FEE_SELL = ec.FEE_BUY, ec.FEE_SELL
    thresh = 1.0 * ec.ROUNDTRIP_COST
    sig = defaultdict(dict); tickers = set()
    for d, tk, e in rows:
        tickers.add(tk); sig[d][tk] = e
    if not sig: return None
    alld = sorted({d for tk in tickers for d in closes.get(tk, {})})
    first, last = min(sig), max(sig)
    cal = [d for d in alld if first <= d <= last]
    if len(cal) < 2: return None
    def cl(tk, d):
        c = closes.get(tk, {}); ds = [x for x in c if x <= d]
        return c[max(ds)] if ds else None
    cash = contributed = prev_eq = traded = fees = 0.0
    shares = {}; last_edge = {}; day_pnl = []; expo_track = []
    cost_basis = defaultdict(float)        # 累計買入成本(含費)
    proceeds_acc = defaultdict(float)      # 累計賣出收入(扣費)
    def _buy(tk, amt, px):
        nonlocal cash, traded, fees
        slip_cost = amt*SLIP; fee = FEE_BUY*amt
        cash -= amt + slip_cost + fee; traded += amt; fees += fee + slip_cost
        shares[tk] = shares.get(tk, 0.0) + amt/px
        cost_basis[tk] += amt + slip_cost + fee
    for i, d in enumerate(cal):
        add = (r60.INITIAL_CAPITAL if i == 0
               else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contributed)
                     if contributed < r60.MAX_CONTRIBUTION else 0.0))
        cash += add; contributed += add
        if i+1 >= len(cal):
            eq = cash + sum(shares[tk]*(cl(tk,d) or 0) for tk in shares)
            day_pnl.append(eq - prev_eq - add); break
        e = cal[i+1]
        port = cash + sum(shares[tk]*(cl(tk,d) or 0) for tk in shares)
        todays = sig.get(d, {}); edges = {}
        for tk, ed in todays.items(): edges[tk] = ed; last_edge[tk] = ed
        for tk in shares:
            if tk not in edges:
                ed = last_edge.get(tk, 0.0)*r60.EDGE_DECAY
                edges[tk] = ed; last_edge[tk] = ed
        rk = lambda tk: edges[tk]*(INC if tk in shares else 1.0)
        ranked = sorted([tk for tk in edges if edges[tk] > 0], key=rk, reverse=True)
        sel = ranked[:r60.MAX_SIGNALS]
        if (len(ranked) > r60.MAX_SIGNALS
                and rk(ranked[r60.MAX_SIGNALS]) >= rk(ranked[r60.MAX_SIGNALS-1])*r60.TIE_RATIO):
            sel = ranked[:r60.MAX_SIGNALS+1]
        confs = [todays[tk] for tk in sel if tk in todays]
        avg = sum(confs)/len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg)) if sel else 0.0
        wsum = sum(edges[tk] for tk in sel); targets = {}
        if wsum > 0 and expo > 0:
            for tk in sel: targets[tk] = port*expo*(edges[tk]/wsum)
        added_today = {}
        # A @ d 收盤
        for tk in sel:
            cp = closes.get(tk, {}).get(d)
            if not cp or cp <= 0: continue
            cur = shares.get(tk, 0.0)*cp
            cap_left = r60.DAILY_ADD_CAP - added_today.get(tk, 0.0)
            buy_amt = min(targets.get(tk, 0.0)-cur, cap_left)
            if buy_amt <= 0 or buy_amt < thresh*port: continue
            if d in limitup.get(tk, set()): continue
            _buy(tk, buy_amt, cp); added_today[tk] = added_today.get(tk, 0.0)+buy_amt
        # B① @ d+1 開盤 賣/減碼/出場
        for tk in list(shares):
            op = opens.get(tk, {}).get(e)
            if not op or op <= 0: continue
            cur = shares[tk]*op; tgt = targets.get(tk, 0.0)
            delta = tgt - cur
            if delta >= 0: continue
            if abs(delta) < thresh*port: continue
            sell_amt = abs(delta); slip_cost = sell_amt*SLIP; fee = FEE_SELL*sell_amt
            cash += sell_amt - slip_cost - fee; traded += sell_amt; fees += fee+slip_cost
            proceeds_acc[tk] += sell_amt - slip_cost - fee
            shares[tk] = tgt/op
            if shares[tk] <= 1e-6: shares.pop(tk, None)
        # B② @ d+1 開盤 補買
        for tk in sel:
            op = opens.get(tk, {}).get(e); cp = closes.get(tk, {}).get(d)
            if not op or op <= 0 or not cp or cp <= 0: continue
            gap = op/cp - 1
            cur = shares.get(tk, 0.0)*op
            cap_left = r60.DAILY_ADD_CAP - added_today.get(tk, 0.0)
            buy_amt = min(targets.get(tk, 0.0)-cur, cap_left)
            if buy_amt <= 0 or buy_amt < thresh*port: continue
            if gap >= 0.095: continue
            _buy(tk, buy_amt, op); added_today[tk] = added_today.get(tk, 0.0)+buy_amt
        invested = sum(shares[tk]*(cl(tk,e) or 0) for tk in shares)
        eq = cash + invested
        expo_track.append(invested/eq if eq > 0 else 0.0)
        day_pnl.append(eq - prev_eq - add); prev_eq = eq
    # 個股毛利 = 已實現(賣出收入) + 期末未實現市值 − 累計成本
    pnl_by_tk = {}
    for tk in set(cost_basis) | set(proceeds_acc):
        mv = shares.get(tk, 0.0)*(cl(tk, cal[-1]) or 0)
        pnl_by_tk[tk] = proceeds_acc.get(tk, 0.0) + mv - cost_basis.get(tk, 0.0)
    total = sum(day_pnl)
    return {"ret": total/contributed*100 if contributed else 0.0,
            "turn": traded/contributed if contributed else 0.0,
            "pnl_by_tk": pnl_by_tk,
            "avg_expo": sum(expo_track)/len(expo_track) if expo_track else 0.0}


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入全史({len(codes)}支)...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050", start=START)
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in list(codes)+["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in list(codes)+["0050"]}

    logger.info("反彈訊號(score)+觸發日+漲停日...")
    reb_cache = {}; fire = {}; limitup = {}
    for c in codes:
        ds = sorted(OH[c]); cl = []; m = {}; s = set(); lu = set()
        for j, d in enumerate(ds):
            cl.append(OH[c][d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"]*100; s.add(d)
                except Exception: pass
            if j > 0 and OH[c][ds[j-1]]["close"] > 0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1 >= 0.095:
                lu.add(d)
        reb_cache[c] = m; fire[c] = s; limitup[c] = lu

    # 修正後 ⑤ 引擎的 rows(同 entry_compare 反彈腿尺度 *1.5),全 always-on(純反彈)
    def rows_for(s_date, e_date):
        out = []
        for d in alld:
            if not (s_date <= d <= e_date): continue
            for c in codes:
                sc = reb_cache.get(c, {}).get(d, 0.0)*1.5
                if sc > 0: out.append((d, c, sc/100))
        return out

    bench = {lab: bench_0050(opens["0050"], closes["0050"], [d for d in alld if s <= d <= e])
             for lab, s, e in REGIMES}

    logger.info("跑 修正前(等權 SL-12/hold5)...")
    base = {lab: baseline_eqweight(OH, codes, fire, s, e, alld) for lab, s, e in REGIMES}
    logger.info("跑 修正後(канonical ⑤ EDGE_DECAY 無硬停損)...")
    fix = {lab: sim5_instrumented(rows_for(s, e), opens, closes, limitup) for lab, s, e in REGIMES}

    def alpha(res, lab):
        r = res.get(lab); return (r["ret"]-bench[lab]) if r else None
    def worst(res):
        vs = [alpha(res, lab) for lab,_,_ in REGIMES]; vs = [v for v in vs if v is not None]
        return min(vs) if vs else None

    labs = [lab for lab,_,_ in REGIMES]
    L = ["# retest: rebound_offregime — 修正前 vs 修正後(出場機制)\n",
         "> proposed_fix 方向一: 純反彈把『硬 -12% 停損 + max_hold=5』換成 канonical ⑤ 的 EDGE_DECAY 自然老化出場。",
         "> 同一份反彈訊號(leak-safe, ≤當日)、同 universe(112)、同 regime、真實成本(買0.14/賣0.44/滑0.1)、漲停買不到。",
         "> 修正前=等權隔日開盤進場+SL-12+hold5(=被否決引擎);修正後=收盤買/開盤賣+incumbent1.5+EDGE_DECAY+無硬停損。\n",
         f"> 0050 各 regime(同資金 DCA buy&hold): " + " ".join(f"{lab}{bench[lab]:+.0f}%" for lab in labs) + "\n",
         "## ① 報酬率 %(扣成本)\n",
         "| 引擎 | " + " | ".join(labs) + " |",
         "|---|" + "|".join(["---"]*len(labs)) + "|"]
    L.append("| 修正前 SL-12/hold5 | " + " | ".join(f"{base[lab]['ret']:+.0f}" if base[lab] else "—" for lab in labs) + " |")
    L.append("| 修正後 ⑤EDGE_DECAY | " + " | ".join(f"{fix[lab]['ret']:+.0f}" if fix[lab] else "—" for lab in labs) + " |")
    L.append("| 0050 DCA | " + " | ".join(f"{bench[lab]:+.0f}" for lab in labs) + " |")

    L += ["", "## ② ALPHA %(− 0050 DCA)  ← 主判據\n",
          "| 引擎 | " + " | ".join(labs) + " | 最差 |",
          "|---|" + "|".join(["---"]*len(labs)) + "|---|"]
    L.append("| 修正前 SL-12/hold5 | " + " | ".join(f"{alpha(base,lab):+.0f}" if alpha(base,lab) is not None else "—" for lab in labs) + f" | **{worst(base):+.0f}** |")
    L.append("| 修正後 ⑤EDGE_DECAY | " + " | ".join(f"{alpha(fix,lab):+.0f}" if alpha(fix,lab) is not None else "—" for lab in labs) + f" | **{worst(fix):+.0f}** |")

    L += ["", "## ③ 換手 turnover ×(越低越不靠高換手偷報酬)\n",
          "| 引擎 | " + " | ".join(labs) + " |",
          "|---|" + "|".join(["---"]*len(labs)) + "|"]
    L.append("| 修正前 | " + " | ".join(f"{base[lab]['turn']:.1f}" if base[lab] else "—" for lab in labs) + " |")
    L.append("| 修正後 | " + " | ".join(f"{fix[lab]['turn']:.1f}" if fix[lab] else "—" for lab in labs) + " |")

    L += ["", "## ④ 集中度 = top-3 個股佔 |毛利| %(>~70% = skew 紅旗)\n",
          "| 引擎 | " + " | ".join(labs) + " |",
          "|---|" + "|".join(["---"]*len(labs)) + "|"]
    L.append("| 修正前 | " + " | ".join(f"{concentration(base[lab]['pnl_by_tk']):.0f}" if base[lab] else "—" for lab in labs) + " |")
    L.append("| 修正後 | " + " | ".join(f"{concentration(fix[lab]['pnl_by_tk']):.0f}" if fix[lab] else "—" for lab in labs) + " |")

    L += ["", "## ⑤ 平均曝險 %(曝險中性檢查: 兩引擎不同則 alpha 不可直接比大小)\n",
          "| 引擎 | " + " | ".join(labs) + " |",
          "|---|" + "|".join(["---"]*len(labs)) + "|"]
    L.append("| 修正前 | " + " | ".join(f"{base[lab]['avg_expo']*100:.0f}" if base[lab] else "—" for lab in labs) + " |")
    L.append("| 修正後 | " + " | ".join(f"{fix[lab]['avg_expo']*100:.0f}" if fix[lab] else "—" for lab in labs) + " |")

    L += ["", "## 結論\n",
          f"- 被否決原版(修正前 SL-12/hold5)最差 regime alpha = **{worst(base):+.0f}%**;"
          f"修正後(⑤ EDGE_DECAY 無硬停損)最差 = **{worst(fix):+.0f}%**。",
          "- 「2021 多頭主動倒賠」: 看 ① 報酬率 2021 復甦欄 —— 修正前 vs 修正後,正負號是否翻轉。",
          "- 曝險中性: 兩引擎 avg_expo 若差很多, 高曝險在多頭會虛假占優, 解讀 alpha 要扣這層。",
          "- 資料限制: 2021-2026 為大多頭、僅 2022 一段真空頭; 出場放寬=拉長曝險, 其『保命』代價在本資料無法被空頭證偽。"]
    (ROOT/"reports"/"retest_rebound_offregime_fix.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/retest_rebound_offregime_fix.md")
    print("\n".join(L))


if __name__ == "__main__":
    main()
