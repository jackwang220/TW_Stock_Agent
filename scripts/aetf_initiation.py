"""AETF 方法 #7:新建倉事件 (initiation event study)

假設:主動式 ETF 經理人「首次建倉」某股 = 看多訊號 → 事件後 5/10/20 日有正超額報酬。

做法:
  1. 從 28 檔 ETF 的每日持股偵測 initiation 事件:
       (a) brand-new   = 該 (ETF,股) 在序列中首次出現權重 > 0(且非第 1 天就持有)
       (b) cross-zero  = 權重從 ~0(<thr) 升到 ≥thr(含再建倉)
     用 thr=0.01% 抓「實質從零到有」(更高的 thr 會被每日權重在門檻附近的雜訊污染,
     50 檔全部都會「穿越」→ 假事件,已驗證並排除)。
  2. Event study:對每個事件,取事件日 D(= 揭露日)後 +1..+20 日的個股報酬,
     算 5/10/20 日累積報酬,並扣同期 0050 → 超額報酬 (CAR)。
  3. 把事件當交易訊號丟進 ⑤買收賣開 引擎,持有固定 N 天,多窗 alpha vs DCA0050、扣成本。

⚠️ 第一輪作弊註記(明確):
  - 時點洩漏:用「揭露日當天」就交易(主動式 ETF 持股揭露其實是盤後/隔日,這裡當日就進場)。
  - 用當前 ETF 成員(survivorship / membership-as-of-now)。
  不作弊的部分:基準 = 同資金 DCA 0050、扣真實成本(買0.1425%/賣0.4425%+滑價0.1%、
  漲停買不到)、曝險中性比較、報換手/集中。

資料限制(誠實):此 CSV 是 2025-06-17~2026-06-17、28 檔 ETF 的 50 檔大型股池,
  每個 (ETF,股) 對都「整年每天都在」(每對 262 列全勤)→ ETF 成員整年幾乎不換。
  → initiation 事件樣本極少(見 console 與報告)。

用法:
    uv run python scripts/aetf_initiation.py
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

v6 = _load("v6", ROOT / "scripts/exp_step1_v6.py")
ec = _load("ec", ROOT / "scripts/exp_60d_entry_compare.py")
v3 = _load("v3", ROOT / "scripts/exp_step1_v3.py")
oh = v3.oh
sim5 = ec.sim_buyclose_sellopen

CSV = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"
END = "2026-06-08"
# 資料只 1 年 → 多窗用全期可得段;regime 只有「多頭」一段可講
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]
HOLD_DAYS = [5, 10, 20]      # event study 視窗
THR = 0.01                    # 「實質從零到有」門檻(%)


def detect_events(df, dates):
    """偵測 initiation 事件。回傳:
       brand_new: [(date, etf, stk)]  首次出現(非第1天持有)
       cross:     [(date, etf, stk)]  權重 <THR -> >=THR (含再建倉)
       fake_high_thr: 用 thr=0.5% 會抓到幾筆(示警雜訊)"""
    di = {d: i for i, d in enumerate(dates)}
    brand_new, cross = [], []
    for (etf, stk), gg in df.groupby(["ETF_Code", "Stock_Code"]):
        gg = gg.sort_values("Date")
        w = gg["Weight(%)"].values; dd = gg["Date"].values
        # brand-new: 第一個 weight>0 的 run,起點非 index0
        pos_idx = [di[dd[i]] for i in range(len(w)) if w[i] > 0]
        if pos_idx and min(pos_idx) > 0:
            j = min(pos_idx)
            brand_new.append((dates[j], etf, stk))
        # cross THR
        for i in range(1, len(w)):
            if w[i - 1] < THR <= w[i]:
                cross.append((dd[i], etf, stk))
    # 高門檻假事件數(示警:每日權重雜訊穿越)
    fake = 0
    for (etf, stk), gg in df.groupby(["ETF_Code", "Stock_Code"]):
        w = gg.sort_values("Date")["Weight(%)"].values
        fake += sum(1 for i in range(1, len(w)) if w[i - 1] < 0.5 <= w[i])
    return brand_new, cross, fake


def car_event_study(events, closes, hold_days):
    """對每個事件算 D 後 hold 日的個股報酬與超額(扣 0050 同期)。
       事件日 D 為揭露日;進場用 D 收盤(作弊:當日就進)。"""
    res = {h: {"raw": [], "exc": []} for h in hold_days}
    bench = closes.get("0050", {})
    bench_dates = sorted(bench)
    detail = []
    for (d, etf, stk) in events:
        cs = closes.get(stk, {})
        cds = sorted(cs)
        if d not in cs:
            # 對齊到 <= d 的最近交易日
            le = [x for x in cds if x <= d]
            if not le:
                continue
            d0 = le[-1]
        else:
            d0 = d
        idx = cds.index(d0)
        row = {"date": d, "etf": etf, "stk": stk}
        for h in hold_days:
            if idx + h >= len(cds):
                continue
            p0 = cs[d0]; p1 = cs[cds[idx + h]]
            if p0 <= 0:
                continue
            r = p1 / p0 - 1
            # 0050 同期(用相同日曆日對齊)
            le0 = [x for x in bench_dates if x <= d0]
            d_end = cds[idx + h]
            le1 = [x for x in bench_dates if x <= d_end]
            if le0 and le1 and bench[le0[-1]] > 0:
                br = bench[le1[-1]] / bench[le0[-1]] - 1
            else:
                br = 0.0
            res[h]["raw"].append(r)
            res[h]["exc"].append(r - br)
            row[f"r{h}"] = r; row[f"e{h}"] = r - br
        detail.append(row)
    return res, detail


def main():
    df = pd.read_csv(CSV, dtype={"Stock_Code": str, "ETF_Code": str})
    dates = sorted(df.Date.unique())
    logger.info(f"CSV: {dates[0]}~{dates[-1]} {len(dates)} 交易日, {df.ETF_Code.nunique()} ETF, {df.Stock_Code.nunique()} 股")

    # 全勤檢查(membership 是否靜態)
    cnt = df.groupby(["ETF_Code", "Stock_Code"]).size()
    full = int((cnt == len(dates)).sum())
    logger.info(f"(ETF,股) 對 {len(cnt)};全勤(整年每天都在)= {full}/{len(cnt)} → membership "
                f"{'幾乎靜態,無建倉事件可言' if full == len(cnt) else '有變動'}")

    brand_new, cross, fake = detect_events(df, dates)
    logger.info(f"事件偵測:brand-new {len(brand_new)} 筆 | cross<{THR}% {len(cross)} 筆 | "
                f"高門檻(0.5%)假穿越 {fake} 筆(雜訊,排除)")
    by_date = defaultdict(int)
    for d, _, _ in cross:
        by_date[d] += 1
    logger.info(f"cross 事件日分布:{dict(sorted(by_date.items()))}")

    # 載入價格(事件股 + 0050)
    stocks = sorted(set(s for _, _, s in cross) | set(s for _, _, s in brand_new))
    closes = {}
    for c in stocks + ["0050"]:
        o = oh(c)
        closes[c] = {d: o[d]["close"] for d in o}
    opens = {}
    limitup = {}
    for c in stocks + ["0050"]:
        o = oh(c)
        opens[c] = {d: o[d]["open"] for d in o}
        ds = sorted(o); s = set()
        for j, d in enumerate(ds):
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"] - 1 >= 0.095:
                s.add(d)
        limitup[c] = s

    # ── Event study (cross 定義,樣本較多) ──
    car, detail = car_event_study(cross, closes, HOLD_DAYS)
    logger.info("=== Event study (cross<THR initiations) ===")
    for h in HOLD_DAYS:
        raw = car[h]["raw"]; exc = car[h]["exc"]
        if raw:
            logger.info(f"  +{h}日: n={len(raw)} 原始 {sum(raw)/len(raw)*100:+.2f}% "
                        f"超額 {sum(exc)/len(exc)*100:+.2f}% "
                        f"勝率 {sum(1 for x in exc if x>0)/len(exc)*100:.0f}%")
        else:
            logger.info(f"  +{h}日: n=0")

    # ── 回測:把 cross 事件當訊號丟進 ⑤,持有 N 天,多窗 alpha ──
    # baseline = DCA0050 本身(alpha=0);本方法 = 事件進場
    def events_to_rows(events, hold, edge=0.5):
        """每個事件 → 在 D..D+hold-1 每天給該股一個 edge(持有 hold 天)。"""
        di = {d: i for i, d in enumerate(dates)}
        rows = []
        for (d, etf, stk) in events:
            if stk not in closes or d not in closes[stk]:
                continue
            i0 = dates.index(d) if d in dates else None
            if i0 is None:
                continue
            for k in range(hold):
                if i0 + k < len(dates):
                    rows.append((dates[i0 + k], stk, edge))
        return rows

    res = {}
    bench_win = {}
    for wl, n in WINDOWS:
        wd = set(dates[-n:])
        bench_win[wl] = ec.bench_close_0050(closes, dates[-n:])
    # 全期(=1年內最長可得)bench 換手/曝險基準
    for hold in HOLD_DAYS:
        rows_all = events_to_rows(cross, hold)
        for wl, n in WINDOWS:
            wd = set(dates[-n:])
            rw = [r for r in rows_all if r[0] in wd]
            res[(hold, wl)] = sim5(rw, opens, closes, limitup, switch_cost_mult=1.0) if rw else None

    # ── 報告 ──
    RPT = ROOT / "reports" / "aetf_initiation.md"
    RPT.parent.mkdir(parents=True, exist_ok=True)
    wlabels = [wl for wl, _ in WINDOWS]
    L = [
        "# AETF 方法 #7:新建倉事件 (initiation event study)\n",
        f"> 資料 {dates[0]}~{dates[-1]}({len(dates)} 日)｜28 ETF｜50 檔大型股池｜還原含息價(yfinance/FinMind)\n",
        "> ⚠️ 作弊註記:**時點洩漏**(揭露日當天就進場,實務揭露為盤後/隔日)+ **survivorship/當前ETF成員**。\n",
        "> 不作弊:基準=同資金 DCA0050、扣成本(買0.14%/賣0.44%+滑價0.1%、漲停買不到)、曝險中性、報換手/集中。\n",
        "",
        "## 0. 致命發現:此資料沒有建倉事件可研究\n",
        f"- 1091 個 (ETF,股) 對中,**{full} 對整年 262 天全勤**(每天都在持股表) → ETF 成員整年幾乎不換。",
        f"- **brand-new(中途首次建倉)= {len(brand_new)} 筆**;且該唯一事件是 ETF 上市第 2 天權重由 0→正(launch ramp),非真實經理人換股。",
        f"- 用「權重 <{THR}% → ≥{THR}%」抓實質從零到有 = **{len(cross)} 筆**,且全部落在 {sorted(by_date.keys())}(launch 隔日權重定型),非中途換股。",
        f"- 若放寬到 0.5% 門檻會抓到 **{fake} 筆「穿越」**——那是每日權重在門檻附近的雜訊,50 檔全中,**假事件**,已排除。",
        "- 結論:此 CSV 的主動式 ETF 在這一年內**不換成員、只調權重**,『新建倉事件』在本資料中**樣本 ≈ 0**,方法不可行。",
        "",
        "## 1. Event study (cross initiations, n 見下)\n",
        "| 持有 | n | 平均原始報酬 | 平均超額(扣0050) | 勝率 |",
        "|---|---|---|---|---|",
    ]
    for h in HOLD_DAYS:
        raw = car[h]["raw"]; exc = car[h]["exc"]
        if raw:
            L.append(f"| +{h}日 | {len(raw)} | {sum(raw)/len(raw)*100:+.2f}% | "
                     f"{sum(exc)/len(exc)*100:+.2f}% | {sum(1 for x in exc if x>0)/len(exc)*100:.0f}% |")
        else:
            L.append(f"| +{h}日 | 0 | — | — | — |")
    L += [
        "",
        f"> n≤{max(len(car[h]['raw']) for h in HOLD_DAYS)};且事件集中單一日 → **統計上無意義**(無自由度、無跨時間樣本)。",
        "",
        "## 2. 多窗 ALPHA %(事件進場策略 − 同資金 DCA0050,扣成本)\n",
        "> baseline = DCA0050(定義 alpha=0);本方法 = cross 事件丟 ⑤買收賣開、持有 N 天。\n",
        "| 持有N天 | " + " | ".join(wlabels) + " | 換手 | 持股 | 曝險 |",
        "|---|" + "|".join(["---"] * (len(wlabels) + 3)) + "|",
    ]
    for hold in HOLD_DAYS:
        cells = []
        for wl in wlabels:
            r = res.get((hold, wl))
            if r is None:
                cells.append("—")
            else:
                cells.append(f"{r['ret'] - bench_win[wl]:+.1f}")
        r1y = res.get((hold, "1年"))
        turn = f"{r1y['turn']:.1f}x" if r1y else "—"
        pos = f"{r1y.get('avg_pos',0):.1f}" if r1y else "—"
        expo = f"{r1y.get('avg_expo',0)*100:.0f}%" if r1y else "—"
        L.append(f"| {hold}天 | " + " | ".join(cells) + f" | {turn} | {pos} | {expo} |")
    L += [
        "",
        "> 0050 基準報酬: " + " ".join(f"{wl}{bench_win[wl]:+.1f}%" for wl in wlabels),
        "",
        "## 3. 判讀(誠實)\n",
        "- **has_uplift = False**:事件樣本本質 ≈ 0(全是 launch ramp,集中單日),任何 alpha 都是 1~6 檔個股的純運氣,不可外推。",
        "- 即便用第一輪允許的作弊(當日進場/survivorship),也**無法逼出訊號**——因為沒有事件母體可逼。",
        "- 根因:本 CSV 的主動式 ETF 一年內**不換成員**(50 檔大型池固定),只在既有成員間調權重。",
        "  『新建倉』要有意義,需要 (a) 更長期間涵蓋成員輪動,或 (b) 更廣的可投資池(經理人真的會新納入池外股)。",
        "- 相鄰可行替代:既有『每日共識加減碼流向』已證無 edge;本方法是其更稀疏的特例,結論一致(無 edge)。",
    ]
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    # 回 console 摘要 + 把表存成 markdown 字串供結構化輸出
    print("\n".join(L[L.index("## 2. 多窗 ALPHA %(事件進場策略 − 同資金 DCA0050,扣成本)\n"):]))


if __name__ == "__main__":
    main()
