"""方法 #3：策略參數微調 — 逆向出 1-2 檔策略明確 ETF 的規則後掃參數。

哲學:主動式 ETF 的持股 ≈ 「從某個產業池中挑 K 檔大型股、用某訊號加權、每 R 天再平衡」。
逆向出這條可複製規則,掃參數(K/訊號/權重/再平衡),看調過版本能否同時贏:
  (a) 原 ETF       — 直接照它每日實際持股×實際權重 buy&hold(這是「原ETF」基準)
  (b) H 引擎       — base_universe 上的 H+反彈雙引擎(baseline)
  (c) tuned rule   — 在該 ETF 持股池上掃出來的最佳規則

挑選的 2 檔(由 data/_aetf_profile.json 的 HHI/產業決定):
  • 00979A : 最集中(HHI 542, top5 41%)、電子工業 31% → 「集中科技/AI 供應鏈」原型
  • 00985A : 金融保險 34%、較分散(HHI 383)        → 「高息/金融」原型

═══════════════════════ 作弊註記(第一輪刻意允許,逼訊號)═══════════════════════
  [CHEAT-1 時點洩漏]   用「同日揭露持股」當日就交易。實務揭露是盤後 → 這是 look-ahead。
                       (原 ETF 基準與 tuned rule 的持股池都來自當日 CSV 揭露)
  [CHEAT-2 成員存活]   tuned rule 的選股池 = 該 ETF *當前/各日揭露*的成員(point-in-time
                       但仍是「已知是這檔 ETF 會持有的股票」)→ survivorship / in-sample 池。
  [CHEAT-3 in-sample]  最佳參數是在「全期(1年)」上掃出來的 → in-sample 調參。
                       (有 uplift 才進 Verify 階段做 leak-free / OOS)
  ── 不准作弊處(嚴守)──
  • 基準  : alpha = 策略 − 同期同資金 DCA 0050(還原含息價)
  • 成本  : 買0.1425%/賣0.4425%/滑價0.1%、漲停買不到(沿用 ⑤ 引擎)
  • 曝險  : 三方都報 avg_expo,高曝險虛假贏會被抓出
  • 報換手/集中(avg_pos)/曝險

⚠️ 資料天花板:窗內 TWII +107%、0050 +124%(實為極端多頭,非價格 bug;已對 yfinance
   交叉驗證)。1 個多頭年、池偏大型股 → 結論帶「缺空頭」天花板,空頭行為未知。

用法: uv run python scripts/aetf_tune.py
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

v5  = _load("v5",  ROOT / "scripts/exp_step1_v5.py")
v6  = _load("v6",  ROOT / "scripts/exp_step1_v6.py")
ec  = _load("ec",  ROOT / "scripts/exp_60d_entry_compare.py")
r60 = _load("r60", ROOT / "scripts/run_backtest_60d.py")
features, _factors, oh = v5.features, v5._factors, v5.oh
h_score = ec.h_score
sim5 = ec.sim_buyclose_sellopen        # ⑤ 買收賣開:各窗皆正 alpha 的最佳執行
FEE_BUY, FEE_SELL, SLIP = ec.FEE_BUY, ec.FEE_SELL, ec.SLIP

END = "2026-06-16"   # ETF 持股到 2026-06-17;訊號/價需 d+1 估值,留一日
# ETF 資料僅 1 年(2025-06-17~);長窗就是全期,短窗從尾端切。regime 用可得段。
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]
REGIMES = {
    "前段25H2": ("2025-06-17", "2025-12-31"),
    "後段26H1": ("2026-01-01", "2026-06-16"),
}
TARGET_ETFS = ["00979A", "00985A"]
CSV = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"


# ───────────────────────── 載入 ETF 每日持股 ─────────────────────────
def load_etf_holdings():
    """回傳 {etf: {date: {code: weight_frac}}};weight 正規化成和=1。"""
    import csv
    h = defaultdict(lambda: defaultdict(dict))
    with open(CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                w = float(row["Weight(%)"])
            except (ValueError, KeyError):
                continue
            h[row["ETF_Code"]][row["Date"]][row["Stock_Code"]] = w
    out = {}
    for etf, byd in h.items():
        out[etf] = {}
        for d, wm in byd.items():
            s = sum(wm.values())
            out[etf][d] = {c: w / s for c, w in wm.items()} if s > 0 else {}
    return out


# ───────────────────────── 0050 DCA 基準(收盤,對齊 ⑤ 估值口徑) ─────────────────────────
def bench_0050(opens, closes, days):
    return v6.bench_0050(opens["0050"], closes["0050"], sorted(d for d in days if d in closes.get("0050", {})))


# ───────────────────────── (a) 原 ETF buy&hold(照實際每日持股權重) ─────────────────────────
def sim_etf_holdings(holdings_byd, opens, closes, cal):
    """每日把資金按 ETF 揭露權重配置(收盤成交,目標權重再平衡,扣成本)。
    [CHEAT-1 時點洩漏: 用當日揭露權重當日收盤就配]。資金模型同 ⑤(初始+每日+上限)。"""
    cal = [d for d in cal if d in closes.get("0050", {})]
    if len(cal) < 2:
        return None
    cash = contributed = prev_eq = traded = fees = 0.0
    shares = defaultdict(float)
    day_pnl, expo_track, pos_track = [], [], []

    def cl(tk, d):
        c = closes.get(tk, {}); ds = [x for x in c if x <= d]
        return c[max(ds)] if ds else None

    for i, d in enumerate(cal):
        add = (r60.INITIAL_CAPITAL if i == 0
               else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION - contributed)
                     if contributed < r60.MAX_CONTRIBUTION else 0.0))
        cash += add; contributed += add
        if i + 1 >= len(cal):
            eq = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)
            day_pnl.append(eq - prev_eq - add); break
        e = cal[i + 1]
        port = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)
        # 取 <= d 的最近一次揭露權重
        wd = None
        for dd in sorted(holdings_byd):
            if dd <= d:
                wd = holdings_byd[dd]
        if not wd:
            day_pnl.append(0.0); continue
        targets = {tk: port * w for tk, w in wd.items() if closes.get(tk, {}).get(d)}
        for tk in set(shares) | set(targets):
            cp = closes.get(tk, {}).get(d)
            if not cp or cp <= 0:
                continue
            cur = shares[tk] * cp; tgt = targets.get(tk, 0.0); delta = tgt - cur
            if abs(delta) < 0.002 * port:           # 小再平衡不動,省成本
                continue
            if delta > 0 and d in LIMITUP.get(tk, set()):   # 漲停買不到
                continue
            fee = (FEE_BUY if delta > 0 else FEE_SELL) * abs(delta) + abs(delta) * SLIP
            cash -= delta + fee; traded += abs(delta); fees += fee
            shares[tk] = tgt / cp
            if shares[tk] <= 1e-9:
                shares.pop(tk, None)
        invested = sum(shares[tk] * (cl(tk, e) or 0) for tk in shares)
        eq = cash + invested
        expo_track.append(invested / eq if eq > 0 else 0.0)
        pos_track.append(sum(1 for tk in shares if shares[tk] > 1e-9))
        day_pnl.append(eq - prev_eq - add); prev_eq = eq
    return _stats(day_pnl, contributed, traded, fees, expo_track, pos_track)


def _stats(day_pnl, contributed, traded, fees, expo_track, pos_track):
    total = sum(day_pnl); active = [x for x in day_pnl if abs(x) > 1e-9]
    cum = peak = mdd = 0.0
    for x in day_pnl:
        cum += x; peak = max(peak, cum); mdd = max(mdd, peak - cum)
    if len(active) > 1:
        mm = sum(active) / len(active)
        sd = math.sqrt(sum((x - mm) ** 2 for x in active) / len(active))
        shp = (mm / sd * math.sqrt(252)) if sd > 0 else 0.0
    else:
        shp = 0.0
    return {"ret": total / contributed * 100 if contributed else 0, "mdd": mdd, "sharpe": shp,
            "turn": traded / contributed if contributed else 0, "fees": fees,
            "avg_expo": sum(expo_track) / len(expo_track) if expo_track else 0.0,
            "avg_pos": sum(pos_track) / len(pos_track) if pos_track else 0.0}


# ───────────────────────── (c) tuned rule:在 ETF 持股池掃參數 ─────────────────────────
# 規則: 每日從「該 ETF 揭露成員池(<=d)」取訊號 top-K,edge = 訊號分數,丟進 ⑤ 引擎。
#       訊號可選 mom20/mom60/hscore/lowvol/equal;K/再平衡用 ⑤ 自己的 sizing。
def build_rule_rows(pool_codes_byd, feats, sig_days, signal, topk, twii_feat, turn_pct):
    """回傳 ⑤ 引擎吃的 rows=[(d,code,edge)]。每日只在該日有效池內選 top-K。"""
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in sig_days}
    rows = []
    for d in sig_days:
        pool = pool_codes_byd.get(d, set())
        if not pool:
            continue
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        sc = []
        for c in pool:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            fd = f[d]
            if signal == "hscore":
                v = h_score(_factors(fd, ir), turn_pct.get(d, {}).get(c, 0.5))
            elif signal == "mom20":   # 20日報酬動能
                r = fd.get("ret20", 0.0)
                v = r * 100 if (r is not None and not math.isnan(r)) else 0.0
            elif signal == "trend":   # 站上 MA60 的乖離(中期趨勢)
                ma60 = fd.get("ma60"); cl = fd.get("close")
                v = (cl / ma60 - 1) * 100 if (ma60 and not math.isnan(ma60) and ma60 > 0) else 0.0
            elif signal == "rsi":     # RSI 強度(順勢)
                rs = fd.get("rsi", 0.0)
                v = rs if (rs is not None and not math.isnan(rs)) else 0.0
            else:  # equal
                v = 1.0
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:topk]:
            rows.append((d, c, max(v, 1e-6) / 100.0 if signal != "equal" else 0.01))
    return rows


def main():
    logger.info("載入 ETF 持股 + base_universe...")
    holdings = load_etf_holdings()
    bu = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    bu_codes = list(bu.keys())
    turns = {c: bu[c].get("avg_turnover", 0.0) for c in bu_codes}

    # 全市場池 = base_universe ∪ 兩檔 ETF 出現過的股票(取在 base_universe 內的=有特徵)
    etf_stocks = set()
    for etf in TARGET_ETFS:
        for d, wm in holdings[etf].items():
            etf_stocks |= set(wm.keys())
    all_codes = sorted(set(bu_codes) | (etf_stocks & set(bu_codes)))
    logger.info(f"特徵載入 {len(all_codes)} 檔(ETF 持股 {len(etf_stocks)},其中在 base_universe={len(etf_stocks & set(bu_codes))})...")

    twii_feat = features("0050")
    feats = {c: features(c) for c in all_codes}
    opens, closes = {}, {}
    for c in all_codes + ["0050"]:
        o = oh(c); opens[c] = {d: o[d]["open"] for d in o}; closes[c] = {d: o[d]["close"] for d in o}

    alld = sorted({d for c in all_codes for d in closes.get(c, {}) if d <= END})
    # ETF 資料起點
    etf_start = min(d for etf in TARGET_ETFS for d in holdings[etf])
    sig_days = [d for d in alld if d >= etf_start]
    logger.info(f"訊號範圍 {sig_days[0]} ~ {sig_days[-1]}({len(sig_days)} 日,ETF 起 {etf_start})")

    # 漲停日 + 反彈 + 成交百分位(H 引擎要用)
    global LIMITUP
    LIMITUP = {}
    reb_cache = {}
    for c in all_codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); cl_list = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; LIMITUP[c] = s
    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in all_codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    # ── baseline: H+反彈雙引擎(全 base_universe) ──
    logger.info("baseline H+反彈雙引擎 rows...")
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in sig_days}
    h_rows = []
    TOPN = 4
    for d in sig_days:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        sc = []
        for c in bu_codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            v = (h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                 if bull else reb_cache.get(c, {}).get(d, 0.0))
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:TOPN]:
            h_rows.append((d, c, v / 100.0))

    # 每日 ETF 有效池(<=d 最近揭露)
    def pool_byd(etf):
        dates = sorted(holdings[etf])
        out = {}
        for d in sig_days:
            wd = None
            for dd in dates:
                if dd <= d:
                    wd = holdings[etf][dd]
            out[d] = set(c for c in (wd or {}) if c in feats) if wd else set()
        return out

    # ── 掃參數 ──
    SIGNALS = ["hscore", "mom20", "trend", "rsi", "equal"]
    TOPKS = [3, 5, 8, 12]

    def windows_and_regimes(day_set):
        return day_set

    def run_window(rows, day_set):
        rw = [r for r in rows if r[0] in day_set]
        return sim5(rw, opens, closes, LIMITUP, switch_cost_mult=1.0)

    # baseline alpha 各窗
    def alpha_of(sim, day_set):
        if not sim:
            return None
        b = bench_0050(opens, closes, day_set)
        return sim["ret"] - b

    win_days = {wl: set(sig_days[-n:]) for wl, n in WINDOWS}
    reg_days = {rn: set(d for d in sig_days if s <= d <= e) for rn, (s, e) in REGIMES.items()}
    allcols = [wl for wl, _ in WINDOWS] + list(REGIMES.keys())
    col_days = {**win_days, **reg_days}

    # H baseline sims
    H_sim = {col: run_window(h_rows, days) for col, days in col_days.items()}

    results = {}   # etf -> {...}
    for etf in TARGET_ETFS:
        logger.info(f"=== {etf} ===")
        pbd = pool_byd(etf)
        # (a) 原 ETF buy&hold
        etf_sim = {col: sim_etf_holdings(holdings[etf], opens, closes, sorted(days))
                   for col, days in col_days.items()}
        # (c) tuned: 掃 signal×K,以 1年窗 alpha 為目標選最佳(in-sample 調參=CHEAT-3)
        sweep = {}
        for sig in SIGNALS:
            for k in TOPKS:
                rows = build_rule_rows(pbd, feats, sig_days, sig, k, twii_feat, turn_pct)
                sim_1y = run_window(rows, win_days["1年"])
                a1y = alpha_of(sim_1y, win_days["1年"])
                sweep[(sig, k)] = (a1y, rows)
        # 選 1年 alpha 最佳
        best_key = max(sweep, key=lambda kk: (sweep[kk][0] if sweep[kk][0] is not None else -1e9))
        best_rows = sweep[best_key][1]
        tuned_sim = {col: run_window(best_rows, days) for col, days in col_days.items()}
        # 也記錄全 sweep 的 1年 alpha 看單調性
        sweep_1y = {kk: sweep[kk][0] for kk in sweep}
        results[etf] = {"etf_sim": etf_sim, "tuned_sim": tuned_sim,
                        "best_key": best_key, "sweep_1y": sweep_1y}
        logger.info(f"  {etf} best={best_key} 1年α(tuned)={sweep[best_key][0]:+.0f}")

    # ── 報告 ──
    RPT = ROOT / "reports" / "aetf_tune.md"
    RPT.parent.mkdir(parents=True, exist_ok=True)
    bench = {col: bench_0050(opens, closes, days) for col, days in col_days.items()}
    L = [
        "# 方法 #3 策略參數微調 — tuned rule vs 原ETF vs H\n",
        f"> 結束 {END}｜⑤買收賣開執行｜資金 15000+1000/日(上限5萬)｜還原含息價\n",
        "> ALPHA = 策略 − 同期同資金 DCA 0050｜成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到\n",
        "> 作弊(第一輪刻意): [C1]同日揭露當日交易(時點洩漏) [C2]選股池=ETF揭露成員(survivorship) "
        "[C3]最佳參數在全期1年in-sample掃出\n",
        "> 不准作弊: 基準/成本/曝險中性全部照實。報換手/持股(avg_pos)/曝險。\n",
        f"> ⚠️ 窗內 0050 DCA 基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in allcols)
        + "(極端多頭年,缺空頭→天花板偏樂觀)\n",
        "## ALPHA %(扣 0050 DCA)\n",
        "| 策略 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手(1年) | 持股(1年) | 曝險(1年) |",
        "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|",
    ]

    def add_row(label, simdict):
        avals = []; cells = []
        for col in allcols:
            a = alpha_of(simdict.get(col), col_days[col])
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None:
                avals.append(a)
        s1 = simdict.get("1年")
        worst = f"{min(avals):+.0f}" if avals else "—"
        mean = f"{sum(avals)/len(avals):+.0f}" if avals else "—"
        turn = f"{s1['turn']:.0f}x" if s1 else "—"
        pos = f"{s1.get('avg_pos',0):.1f}" if s1 else "—"
        expo = f"{s1.get('avg_expo',0)*100:.0f}%" if s1 else "—"
        L.append(f"| {label} | " + " | ".join(cells) + f" | **{worst}** | {mean} | {turn} | {pos} | {expo} |")

    add_row("H雙引擎(baseline)", H_sim)
    for etf in TARGET_ETFS:
        add_row(f"原{etf} buy&hold", results[etf]["etf_sim"])
        bk = results[etf]["best_key"]
        add_row(f"tuned{etf}[{bk[0]},K{bk[1]}]", results[etf]["tuned_sim"])

    # 原始報酬
    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 策略 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    def ret_row(label, simdict):
        cells = [f"{simdict[c]['ret']:+.0f}" if simdict.get(c) else "—" for c in allcols]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    ret_row("H雙引擎", H_sim)
    for etf in TARGET_ETFS:
        ret_row(f"原{etf}", results[etf]["etf_sim"])
        bk = results[etf]["best_key"]
        ret_row(f"tuned{etf}[{bk[0]},K{bk[1]}]", results[etf]["tuned_sim"])
    L.append("| 0050 DCA基準 | " + " | ".join(f"{bench[c]:+.0f}" for c in allcols) + " |")

    # uplift vs 兩個對照
    L += ["", "## Uplift:tuned − max(原ETF, H)  逐窗(pp)\n",
          "> 要同時贏原ETF與H才算真 uplift → 取兩者較高者當門檻。\n",
          "| ETF | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    for etf in TARGET_ETFS:
        cells = []
        for col in allcols:
            at = alpha_of(results[etf]["tuned_sim"].get(col), col_days[col])
            ae = alpha_of(results[etf]["etf_sim"].get(col), col_days[col])
            ah = alpha_of(H_sim.get(col), col_days[col])
            if None in (at, ae, ah):
                cells.append("—")
            else:
                cells.append(f"{at - max(ae, ah):+.0f}")
        L.append(f"| tuned{etf} | " + " | ".join(cells) + " |")

    # sweep 單調性(1年 alpha)
    L += ["", "## 參數掃描(1年 alpha %,in-sample=C3)— 看訊號×K 單調性\n"]
    for etf in TARGET_ETFS:
        sw = results[etf]["sweep_1y"]
        sigs = sorted({k[0] for k in sw}); ks = sorted({k[1] for k in sw})
        L.append(f"### {etf}(★=選中 {results[etf]['best_key']})\n")
        L.append("| signal\\K | " + " | ".join(f"K{k}" for k in ks) + " |")
        L.append("|---|" + "|".join(["---"] * len(ks)) + "|")
        for sig in sigs:
            row = []
            for k in ks:
                a = sw.get((sig, k))
                star = "★" if (sig, k) == results[etf]["best_key"] else ""
                row.append(f"{a:+.0f}{star}" if a is not None else "—")
            L.append(f"| {sig} | " + " | ".join(row) + " |")
        L.append("")

    L += ["## 判讀\n",
          "- tuned 同窗 uplift 全正 → 在該 ETF 持股池上,可調出同時贏原ETF與H的規則(但帶 C1/C2/C3 作弊)。",
          "- uplift≈0 或負 → 微調沒額外 edge,主動式 ETF 的選股無法被簡單規則複製超越。",
          "- 看曝險欄:三方曝險接近才公平;tuned 若靠高曝險贏 → 虛假。",
          "- 看 sweep 表:若最佳是角點/孤峰(鄰格負)→ 過擬合;平滑高原才穩健。",
          "- ⚠️ 一個多頭年、大型股池 → 缺空頭,uplift 是樂觀上界。"]
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    # console 摘要
    for etf in TARGET_ETFS:
        for col in ["60天", "1年"]:
            at = alpha_of(results[etf]["tuned_sim"].get(col), col_days[col])
            ae = alpha_of(results[etf]["etf_sim"].get(col), col_days[col])
            ah = alpha_of(H_sim.get(col), col_days[col])
            logger.success(f"{etf} {col}: tuned α{at:+.0f} | 原ETF α{ae:+.0f} | H α{ah:+.0f} | "
                           f"uplift vs max {at-max(ae,ah):+.0f}")


if __name__ == "__main__":
    main()
