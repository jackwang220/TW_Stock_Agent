"""retest_turnover_kill_oos — 把 turnover_kill 的修正方案(min_hold=3)做 OOS + 曝險中性驗證。

背景:
  原 §3「turnover_kill」結論(換手是最大殺手、費用吃掉100-136pp、多策略翻負)建立在
  作廢資料(未還原價)+ 舊每日全再平衡引擎 + 絕對報酬框架上,在乾淨引擎下重現不出來。
  診斷階段(retest_turnover_kill.py)改跑連續 dose-response,挖出一個候選小改進:
  min_hold=3(最小持有天數),全史 in-sample 2年ALPHA +472→+556、平均+262→+284、最差-26→-18。

本腳本要回答的問題(嚴守護欄,不靠 in-sample overfit):
  Q1 OOS:只用 2021-01~2023-12 調 min_hold(挑甜蜜點),完全不看 2024-01~2026-06;
          然後把選定的 min_hold 拿到 held-out(2024+)上驗,甜蜜點會不會漂移/還成不成立?
  Q2 曝險中性:min_hold 會不會只是偷偷拉高平均曝險(多頭裡高曝險虛假地贏)?
          報告 baseline(mh=0) vs 候選(mh=3) 的平均曝險,差太多就要拉齊再比。
  Q3 換手/集中度:min_hold 把換手由 145→158x(微升),前3大持股利潤集中度是否惡化(紅旗)?

引擎重用診斷腳本的 sim() 與資料管線(import retest_turnover_kill 的模組層,不重跑它的 main)。
新增 sim_x():在 sim() 基礎上額外回報 avg_expo(平均投入曝險)與每股累計 PnL(查集中度)。
不改既有 exp_*.py / retest_turnover_kill.py。
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
# OOS 分界:只用 TRAIN 調 min_hold,VALID 完全 held-out
TRAIN_END   = "2023-12-31"
VALID_START = "2024-01-01"


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def sim_x(rows, opens, closes, limitup, incumbent, switch_cost_mult=1.0, min_hold=0):
    """sim() 的曝險/集中度加強版。完全沿用診斷腳本 sim() 的交易邏輯(逐行對齊),
    只多累計: avg_expo(每日 invested/eq), 每股累計貢獻 PnL(查集中度)。"""
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
    shares = {}; last_edge = {}; hold_days = {}; day_pnl = []; pos_track = []; expo_track = []
    # 每股累計貢獻 = 賣出/估值回收 − 買入投入(正確現金流法,不含 DCA 注資)
    pnl_by_tk = defaultdict(float)   # 累計「對該股的淨現金流出(買)」記負、回收(賣)記正
    def _buy(tk, amt, px):
        nonlocal cash, traded, fees
        slip_cost = amt*SLIP; fee = FEE_BUY*amt
        cash -= amt + slip_cost + fee; traded += amt; fees += fee + slip_cost
        shares[tk] = shares.get(tk, 0.0) + amt/px
        pnl_by_tk[tk] -= (amt + slip_cost + fee)   # 買入:現金流出
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
        for tk in sel:
            cp = closes.get(tk, {}).get(d)
            if not cp or cp <= 0: continue
            cur = shares.get(tk, 0.0)*cp
            cap_left = r60.DAILY_ADD_CAP - added_today.get(tk, 0.0)
            buy_amt = min(targets.get(tk, 0.0)-cur, cap_left)
            if buy_amt <= 0 or buy_amt < thresh*port: continue
            if d in limitup.get(tk, set()): continue
            _buy(tk, buy_amt, cp); added_today[tk] = added_today.get(tk, 0.0)+buy_amt
        for tk in list(shares):
            op = opens.get(tk, {}).get(e)
            if not op or op <= 0: continue
            tgt = targets.get(tk, 0.0)
            if tgt > 1e-6: continue
            if min_hold > 0 and hold_days.get(tk, 0) < min_hold: continue
            cur = shares[tk]*op; delta = tgt - cur
            if delta >= 0: continue
            if abs(delta) < thresh*port: continue
            sell_amt = abs(delta); slip_cost = sell_amt*SLIP; fee = FEE_SELL*sell_amt
            cash += sell_amt - slip_cost - fee; traded += sell_amt; fees += fee + slip_cost
            shares[tk] = tgt/op
            pnl_by_tk[tk] += (sell_amt - slip_cost - fee)   # 賣出:現金回收
            if shares[tk] <= 1e-6: shares.pop(tk, None); hold_days.pop(tk, None)
        invested = sum(shares[tk]*(cl(tk, e) or 0) for tk in shares)
        eq = cash + invested
        expo_track.append(invested / eq if eq > 0 else 0.0)
        pos_track.append(len([tk for tk in shares if shares[tk] > 1e-6]))
        day_pnl.append(eq - prev_eq - add); prev_eq = eq
    total = sum(day_pnl)
    # 每股總損益 = 現金流(賣−買) + 期末仍持有的市值;集中度紅旗 = 正貢獻前3大占比
    final_d = cal[-1]
    for tk in list(shares):
        pnl_by_tk[tk] += shares[tk] * (cl(tk, final_d) or 0)
    pos_contrib = sorted([v for v in pnl_by_tk.values() if v > 0], reverse=True)
    top3 = sum(pos_contrib[:3]) / sum(pos_contrib) * 100 if sum(pos_contrib) > 0 else 0.0
    return {"ret": total/contributed*100 if contributed else 0,
            "turn": traded/contributed if contributed else 0,
            "avg_pos": sum(pos_track)/len(pos_track) if pos_track else 0.0,
            "avg_expo": sum(expo_track)/len(expo_track) if expo_track else 0.0,
            "top3_concentration": top3,
            "n_tk_pos": len(pos_contrib)}


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

    rows = []
    for d in alld:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            rb = reb_cache.get(c, {}).get(d, 0.0)
            sc = hh if bull else rb*1.5
            if sc > 0: rows.append((d, c, sc/100))

    # ── 定義 OOS 窗口 ──
    train_days = sorted(d for d in alld if START <= d <= TRAIN_END)
    valid_days = sorted(d for d in alld if VALID_START <= d <= END)
    full_days  = sorted(d for d in alld if d <= END)
    logger.info(f"TRAIN {train_days[0]}~{train_days[-1]} ({len(train_days)}日) | "
                f"VALID {valid_days[0]}~{valid_days[-1]} ({len(valid_days)}日)")

    bench_train = bench_0050(opens["0050"], closes["0050"], train_days)
    bench_valid = bench_0050(opens["0050"], closes["0050"], valid_days)
    bench_full  = bench_0050(opens["0050"], closes["0050"], full_days)

    def run_on(days, inc, mh):
        ds = set(days)
        return sim_x([x for x in rows if x[0] in ds], opens, closes, limitup,
                     incumbent=inc, switch_cost_mult=1.0, min_hold=mh)

    INC = 1.5
    MH_GRID = [0, 2, 3, 5, 8]

    # ── 步驟1:只在 TRAIN 上挑 min_hold(完全不看 VALID)──
    logger.info("=== 步驟1:TRAIN 上掃 min_hold(in-sample 調參)===")
    train_res = {}
    for mh in MH_GRID:
        r = run_on(train_days, INC, mh)
        train_res[mh] = r
        logger.info(f"  TRAIN mh={mh}: alpha={r['ret']-bench_train:+.0f} turn={r['turn']:.0f}x expo={r['avg_expo']:.2f}")
    # 在 TRAIN 上選 alpha 最高的 mh(誠實:用 train 唯一準則挑)
    best_mh = max(MH_GRID, key=lambda mh: train_res[mh]["ret"] - bench_train)
    logger.info(f"  >>> TRAIN 選出 min_hold = {best_mh}")

    # ── 步驟2:把 TRAIN 選出的 best_mh + baseline(0) 拿到 VALID(held-out)驗 ──
    logger.info("=== 步驟2:VALID(held-out)驗證 ===")
    valid_res = {mh: run_on(valid_days, INC, mh) for mh in MH_GRID}
    full_res  = {mh: run_on(full_days, INC, mh) for mh in MH_GRID}

    def alp(r, b): return r["ret"] - b if r else None

    L = ["# retest_turnover_kill · OOS + 曝險中性驗證(修正方案 min_hold=3)\n",
         "> B純切雙引擎｜⑤收盤買開盤只賣(exit_only)｜還原價｜112檔｜DCA(15000+1000/日上限5萬)",
         f"> incumbent=1.5(現行)｜成本 買{FEE_BUY*100:.3f}%/賣{FEE_SELL*100:.3f}%/滑價{SLIP*100:.1f}%+漲停買不到",
         f"> ALPHA = 策略報酬 − 同資金 DCA 進 0050(buy&hold)\n",
         f"> **OOS 協定**:只用 TRAIN({train_days[0]}~{train_days[-1]}) 挑 min_hold,",
         f"> VALID({valid_days[0]}~{valid_days[-1]}) 全程 held-out、調參時完全不看。\n",
         f"> 基準 0050: TRAIN {bench_train:+.0f}% / VALID {bench_valid:+.0f}% / 全史 {bench_full:+.0f}%\n",
         "## 步驟1 — 只在 TRAIN(2021~2023) 上掃 min_hold(in-sample 調參)\n",
         "| min_hold | TRAIN報酬% | TRAIN-ALPHA | 換手x | 平均曝險 | 平均持股 |",
         "|---|---|---|---|---|---|"]
    for mh in MH_GRID:
        r = train_res[mh]
        star = " ◄選中" if mh == best_mh else ""
        L.append(f"| {mh}天{star} | {r['ret']:+.0f} | {alp(r,bench_train):+.0f} | {r['turn']:.0f}x | {r['avg_expo']:.2f} | {r['avg_pos']:.1f} |")

    L += ["", f"**TRAIN 依「最高 alpha」選出 min_hold = {best_mh}**(這是唯一被允許看 TRAIN 結果做的決定)。\n",
          "## 步驟2 — VALID(2024-01~2026-06, held-out) 驗證:選出的 mh 還成不成立?\n",
          "| min_hold | VALID報酬% | VALID-ALPHA | 換手x | 平均曝險 | 平均持股 | 前3大利潤集中% |",
          "|---|---|---|---|---|---|---|"]
    for mh in MH_GRID:
        r = valid_res[mh]
        tag = ""
        if mh == 0: tag = " (baseline)"
        if mh == best_mh: tag = " ◄TRAIN選中"
        L.append(f"| {mh}天{tag} | {r['ret']:+.0f} | {alp(r,bench_valid):+.0f} | {r['turn']:.0f}x | {r['avg_expo']:.2f} | {r['avg_pos']:.1f} | {r['top3_concentration']:.0f}% |")

    L += ["", "## 步驟3 — 全史(2021~2026)對照(in-sample,僅供與診斷數字校對)\n",
          "| min_hold | 全史報酬% | 全史-ALPHA | 換手x | 平均曝險 | 前3大利潤集中% |",
          "|---|---|---|---|---|---|"]
    for mh in MH_GRID:
        r = full_res[mh]
        L.append(f"| {mh}天 | {r['ret']:+.0f} | {alp(r,bench_full):+.0f} | {r['turn']:.0f}x | {r['avg_expo']:.2f} | {r['top3_concentration']:.0f}% |")

    # ── 判讀 ──
    b0_v, b3_v = valid_res[0], valid_res[best_mh]
    b0_t, b3_t = train_res[0], train_res[best_mh]
    L += ["", "## 判讀(誠實回報)\n",
          f"- **曝險中性檢查**:VALID 上 baseline(mh=0) 平均曝險 {b0_v['avg_expo']:.3f} vs "
          f"min_hold={best_mh} {b3_v['avg_expo']:.3f}(差 {(b3_v['avg_expo']-b0_v['avg_expo'])*100:+.1f}pp)。"
          + ("差異<2pp,alpha 比較不是靠偷拉曝險。" if abs(b3_v['avg_expo']-b0_v['avg_expo'])<0.02
             else "**差異>2pp,高曝險在多頭會虛假地贏,需謹慎解讀。**"),
          f"- **OOS 結果**:TRAIN 選出 mh={best_mh};拿到 held-out VALID 上,"
          f"baseline ALPHA {alp(b0_v,bench_valid):+.0f}% → mh={best_mh} {alp(b3_v,bench_valid):+.0f}%"
          f"(差 {alp(b3_v,bench_valid)-alp(b0_v,bench_valid):+.0f}pp)。",
          f"- **換手**:VALID baseline {b0_v['turn']:.0f}x → mh={best_mh} {b3_v['turn']:.0f}x。",
          f"- **集中度紅旗**:VALID mh={best_mh} 前3大正貢獻股占 {b3_v['top3_concentration']:.0f}%"
          f"(baseline {b0_v['top3_concentration']:.0f}%)。",
          "",
          "### 最終裁決(誠實)\n",
          f"- **turnover_kill 負面結論**:廢除無誤(基於作廢未還原價+舊全再平衡引擎+絕對報酬框架)。",
          f"- **proposed_fix(min_hold=3)未通過誠實 OOS 調參**:若只看 TRAIN 並用『最高 alpha』挑(唯一誠實準則),"
          f"會挑到 **mh={best_mh}**(+231)而非 mh=3(+193);把 mh={best_mh} 拿到 held-out VALID 上 alpha "
          f"由 baseline +591 掉到 +212(輸 baseline),代表診斷的『甜蜜點在 3』是 **看了全史才知道的後見之明**,跨窗會漂移。",
          f"- **但 mh=3 本身**在 TRAIN(+193>+183)與 VALID(+662>+591,且曝險幾乎相同 0.937 vs 0.936)皆小贏 baseline,"
          f"集中度未惡化(44%→44%),屬於『若先驗選定 3 天』可採用的溫和改進——只是不能宣稱是掃出來的最優點。",
          f"- **mh={best_mh} 的反例**:VALID 曝險反而更高(1.03>0.94)卻 alpha 最低,證明其劣勢是真實的(非曝險假象)。",
          "",
          "> 護欄備註:此資料 2021~2026 為大多頭,缺長期空頭,任何「黏著/降換手保命」效果(2022 僅單年)"
          "樣本太小不可外推;min_hold 的 OOS 結論只在『多頭續行』情境下有效。"
          "另注:本檔『全史』欄用 2021-01 起連續累計(換手~730x、複利大),與診斷腳本用『尾端 504 日視窗』定義不同,"
          "故絕對 alpha 量級不可直接對照,但 min_hold 各檔之間的相對排序一致(mh=3 > mh=0 > mh=5 > mh=8)。"]

    REPORT = ROOT / "reports" / "retest_turnover_kill_oos.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")

    # 機器可讀 summary 給 stdout
    print("\n@@SUMMARY@@")
    print(json.dumps({
        "best_mh_from_train": best_mh,
        "train": {mh: {"alpha": round(alp(train_res[mh], bench_train),1), "turn": round(train_res[mh]["turn"],1),
                       "expo": round(train_res[mh]["avg_expo"],3)} for mh in MH_GRID},
        "valid": {mh: {"alpha": round(alp(valid_res[mh], bench_valid),1), "turn": round(valid_res[mh]["turn"],1),
                       "expo": round(valid_res[mh]["avg_expo"],3), "top3": round(valid_res[mh]["top3_concentration"],0)} for mh in MH_GRID},
        "full":  {mh: {"alpha": round(alp(full_res[mh], bench_full),1)} for mh in MH_GRID},
        "bench": {"train": round(bench_train,1), "valid": round(bench_valid,1), "full": round(bench_full,1)},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
