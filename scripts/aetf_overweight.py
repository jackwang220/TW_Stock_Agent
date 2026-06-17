"""方法 #4：主動權重 alpha(靜態超配,非流向)

核心問句:28 檔主動式 ETF 集體「重押」(相對基準權重的主動超配)的股票,
之後是否有正向超額報酬?這跟已證實無 edge 的「每日流向 tilt」是不同維度
(長期信念/結構,而非當日加減碼方向)。

定義:
  w_etf(c,d)   = 28 檔 ETF 中持有 c 的權重「平均」(ETF-blend 組合權重),
                 再對池內正規化使 Σ=1。代表「主動經理人共識組合」的配重。
  w_bench(c,d) = 同池(50 大型股)以流動性 rank(avg_turnover)為比例的基準權重
                 (0050 近似:市值/流動性加權)。
  aow(c,d)     = w_etf − w_bench  (主動超配,>0=經理人比基準更看重)
  aow_s(c,d)   = aow 的 L 日尾端均值(「靜態信念」平滑,降低日流向雜訊)

驗證:
  1) IC:aow_s(d) 對前向 H 日報酬的橫斷面 rank-IC(多 horizon)。
  2) 分位回測:每日依 aow_s 分 Q 組,看 Q5(高超配)−Q1 前向報酬差。
  3) 策略回測:用 aow_s rank 當 edge 餵 sim5(⑤買收賣開,與 baseline 同引擎),
     多窗 alpha vs 同資金 DCA 0050、扣成本、報換手/集中/曝險。
  baseline = H+反彈雙引擎(專案既定 baseline)。

⚠️ 第一輪允許的作弊(cheat_used,Verify 階段再修):
  - 時點洩漏:ETF 持股當日揭露其實在盤後 → 本輪用「揭露日當日」weight 直接交易
    (應 lag ≥1 日)。已提供 --lag 開關量化洩漏影響。
  - membership/survivorship:用「當前」28 檔 ETF 成員與其 50 檔持股池
    (這些 ETF 多在 2025 才上市,池本身就是事後已知的贏家集合)。
  - 基準近似:w_bench 用流動性 rank 近似 0050 權重(無真實 0050 成分權重檔)。
不作弊的部分:alpha 一律扣同期同資金 DCA 0050;成本買0.1425%/賣0.4425%+滑價0.1%;
  漲停買不到;曝險中性比較(sim5 內建 EXPOSURE_CAP/FLOOR,與 baseline 同 sizing)。

用法:
    uv run python scripts/aetf_overweight.py
    uv run python scripts/aetf_overweight.py --smooth 20 --lag 1   # leak-free 探針
"""
from __future__ import annotations
import sys, json, argparse, importlib.util, math
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal


def _load(name, path):
    m = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    importlib.util.spec_from_file_location(name, path).loader.exec_module(m)
    return m

v5 = _load("v5", ROOT / "scripts/exp_step1_v5.py")
v6 = _load("v6", ROOT / "scripts/exp_step1_v6.py")
ec = _load("ec", ROOT / "scripts/exp_60d_entry_compare.py")
features, _factors, oh = v5.features, v5._factors, v5.oh
h_score = ec.h_score
sim5 = ec.sim_buyclose_sellopen   # ⑤買收賣開:各窗皆正 alpha 的最佳執行

END = "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]
# 資料只 2025-06-17~2026-06-17 → regime 只切「可得段」(多頭年,無空頭)
REGIMES = {
    "2025下半": ("2025-07-01", "2025-12-31"),
    "2026上半": ("2026-01-01", END),
}
ETF_CSV = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"
TOPN = 4   # 與 baseline 同:每日最多 top-N 候選


# ── 載入 ETF-blend 權重 & 主動超配 ─────────────────────────────────────────────
def build_overweight(smooth: int, lag: int):
    """回傳:
      aow_s[d][c]     = 平滑後主動超配(已 lag)
      pool            = 池內所有 stock code(50)
      etf_dates       = ETF 資料的交易日(升冪)
    """
    df = pd.read_csv(ETF_CSV, dtype={"Stock_Code": str, "ETF_Code": str})
    df["w"] = df["Weight(%)"].astype(float)
    pool = sorted(df["Stock_Code"].unique())
    etf_dates = sorted(df["Date"].unique())

    # 每日:ETF-blend 權重 = 各 ETF 內權重平均(對「持有它的 ETF」平均,反映共識強度
    # 與持有家數),再對池正規化 Σ=1。
    # 基準權重:以 avg_turnover(流動性 rank 比例)近似 0050 市值/流動性加權。
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    turn = {c: u.get(c, {}).get("avg_turnover", 0.0) for c in pool}
    # 池內缺 turnover 的給池中位數(避免 0)
    med = sorted(v for v in turn.values() if v > 0)
    med = med[len(med) // 2] if med else 1.0
    for c in pool:
        if turn[c] <= 0:
            turn[c] = med
    tot_turn = sum(turn[c] for c in pool)
    w_bench = {c: turn[c] / tot_turn for c in pool}   # 靜態基準權重(全期 turnover,池內)

    aow_raw: dict[str, dict[str, float]] = {}   # d -> {c: aow}
    for d, g in df.groupby("Date"):
        # ETF-blend:每股對「持有它的 ETF」取權重平均(=該股在主動經理組合的代表配重)
        blend = g.groupby("Stock_Code")["w"].mean()
        s = blend.sum()
        if s <= 0:
            continue
        w_etf = (blend / s).to_dict()   # 正規化 Σ=1
        aow_raw[d] = {c: w_etf.get(c, 0.0) - w_bench.get(c, 0.0) for c in pool}

    # 平滑(尾端 L 日均值)+ lag
    dlist = sorted(aow_raw)
    aow_s: dict[str, dict[str, float]] = {}
    for i, d in enumerate(dlist):
        win = dlist[max(0, i - smooth + 1): i + 1]
        sm = {}
        for c in pool:
            vals = [aow_raw[w][c] for w in win if c in aow_raw[w]]
            sm[c] = sum(vals) / len(vals) if vals else 0.0
        aow_s[d] = sm
    # lag:決策日 d 用 d-lag 的超配(模擬盤後揭露 → 隔日才可用)
    if lag > 0:
        lagged = {}
        for i, d in enumerate(dlist):
            if i - lag >= 0:
                lagged[d] = aow_s[dlist[i - lag]]
        aow_s = lagged
    return aow_s, pool, dlist, w_bench, aow_raw


# ── IC & 分位分析(用本金前向報酬) ──────────────────────────────────────────────
def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    rx = _rank(xs); ry = _rank(ys)
    mx = sum(rx) / n; my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def fwd_ret(closes, c, d, h, daylist):
    """c 在 d → d+h 的本金前向報酬(用收盤,還原價)。"""
    idx = daylist.get(d)
    if idx is None or idx + h >= len(daylist["_list"]):
        return None
    d2 = daylist["_list"][idx + h]
    p0 = closes.get(c, {}).get(d); p1 = closes.get(c, {}).get(d2)
    if p0 and p1 and p0 > 0:
        return p1 / p0 - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=int, default=20, help="超配平滑窗(交易日)")
    ap.add_argument("--lag", type=int, default=0, help="決策用 d-lag 的超配(0=作弊同日;1=leak-free探針)")
    ap.add_argument("--scan", action="store_true", help="掃 smooth/lag 找最佳與單調性")
    args = ap.parse_args()

    # 價格(池 + 0050)
    pool_probe, _, _, _, _ = build_overweight(args.smooth, 0)
    pool = pool_probe and sorted({c for d in pool_probe for c in pool_probe[d]})
    df = pd.read_csv(ETF_CSV, dtype={"Stock_Code": str})
    pool = sorted(df["Stock_Code"].unique())
    logger.info(f"池 {len(pool)} 檔;載入價格...")
    opens, closes = {}, {}
    for c in pool + ["0050"]:
        o = oh(c)
        opens[c] = {d: o[d]["open"] for d in o}
        closes[c] = {d: o[d]["close"] for d in o}

    # 漲停日(sim5 需要)
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    turns = {c: u.get(c, {}).get("avg_turnover", 0.0) for c in pool}
    limitup = {}
    for c in pool:
        o = oh(c); ds = sorted(o)
        s = set()
        for j, d in enumerate(ds):
            if j > 0 and o[ds[j - 1]]["close"] > 0 and o[d]["close"] / o[ds[j - 1]]["close"] - 1 >= 0.095:
                s.add(d)
        limitup[c] = s

    # 交易日(以 0050 為準,且在 ETF 資料範圍內)
    etf_dates = sorted(df["Date"].unique())
    alld = sorted({d for c in pool for d in closes.get(c, {}) if etf_dates[0] <= d <= END})
    daylist = {"_list": alld}
    for i, d in enumerate(alld):
        daylist[d] = i
    logger.info(f"交易日 {alld[0]}~{alld[-1]} ({len(alld)})")

    # ── baseline:H+反彈雙引擎(專案既定 baseline) ──
    logger.info("建 baseline(H+反彈雙引擎)候選...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in pool}
    reb_cache = {}
    for c in pool:
        o = oh(c); ds = sorted(o); cl = []; m = {}
        for d in ds:
            cl.append(o[d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
        reb_cache[c] = m
    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in pool
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}
    base_rows = []
    for d in alld:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        sc = []
        for c in pool:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            v = (h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                 if bull else reb_cache.get(c, {}).get(d, 0.0))
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:TOPN]:
            base_rows.append((d, c, v / 100))

    # ── 0050 基準(各窗 + regime) ──
    def bench_for(dd):
        dd = sorted(d for d in dd if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    bench_win = {wl: bench_for([d for d in alld[-n:]]) for wl, n in WINDOWS}
    regime_days = {rn: [d for d in alld if s <= d <= e] for rn, (s, e) in REGIMES.items()}
    bench_reg = {rn: bench_for(dd) for rn, dd in regime_days.items()}

    # ── 超配方法:aow_s rank → edge 餵 sim5 ──
    def build_ow_rows(aow_s):
        """每日取 aow_s 最高 top-N 作候選,edge = 正規化的超配強度(>0)。
        為與 baseline edge 量級對齊(sim5 用 edge 當 confidence/sizing),
        把 top-N 的 aow 線性映到 [0.4,0.9](EXPOSURE_FLOOR~CAP 區間)。"""
        rows = []
        for d in alld:
            ow = aow_s.get(d)
            if not ow:
                continue
            cand = [(ow[c], c) for c in pool if c in closes and d in closes[c] and ow[c] > 0]
            if not cand:
                continue
            cand.sort(reverse=True)
            top = cand[:TOPN]
            vmax = top[0][0]; vmin = top[-1][0]
            span = (vmax - vmin) or 1.0
            for v, c in top:
                e = 0.5 + 0.4 * ((v - vmin) / span)   # 0.5~0.9
                rows.append((d, c, e))
        return rows

    def run(rows, dd):
        s = set(dd)
        rw = [r for r in rows if r[0] in s]
        return sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)

    # ── IC & 分位(用 args.smooth/lag) ──
    aow_s, *_ = build_overweight(args.smooth, args.lag)
    logger.info("計算 IC & 分位...")
    ic_rows = {}
    for h in (5, 10, 20, 60):
        ics = []
        for d in alld:
            ow = aow_s.get(d)
            if not ow:
                continue
            xs, ys = [], []
            for c in pool:
                fr = fwd_ret(closes, c, d, h, daylist)
                if fr is not None and c in ow:
                    xs.append(ow[c]); ys.append(fr)
            if len(xs) >= 8:
                ics.append(spearman(xs, ys))
        ic_rows[h] = (sum(ics) / len(ics) if ics else 0.0, len(ics))

    # 分位(Q5-Q1)前向 20 日報酬
    def quantile_spread(h=20, nq=5):
        buckets = {q: [] for q in range(nq)}
        for d in alld:
            ow = aow_s.get(d)
            if not ow:
                continue
            arr = [(ow[c], c) for c in pool if c in closes and d in closes.get(c, {})]
            arr.sort()
            n = len(arr)
            if n < nq * 2:
                continue
            for qi in range(nq):
                lo = qi * n // nq; hi = (qi + 1) * n // nq
                for _, c in arr[lo:hi]:
                    fr = fwd_ret(closes, c, d, h, daylist)
                    if fr is not None:
                        buckets[qi].append(fr)
        means = {q: (sum(v) / len(v) * 100 if v else 0.0) for q, v in buckets.items()}
        return means
    qmeans = quantile_spread()

    # ── 多窗回測:baseline vs 超配 ──
    ow_rows = build_ow_rows(aow_s)
    allcols = [wl for wl, _ in WINDOWS] + list(REGIMES.keys())
    col_days = {wl: alld[-n:] for wl, n in WINDOWS}
    col_days.update(regime_days)

    res = {}
    for label, rows in (("baseline(H雙引擎)", base_rows), ("超配aow", ow_rows)):
        for col in allcols:
            res[(label, col)] = run(rows, col_days[col])
        logger.info(f"{label} 完成")

    def alpha(label, col):
        r = res.get((label, col))
        b = bench_win.get(col, bench_reg.get(col))
        return (r["ret"] - b) if (r and b is not None) else None

    if args.scan:
        scan_results = []
        for sm in (5, 10, 20, 40):
            for lg in (0, 1):
                a_s, *_ = build_overweight(sm, lg)
                rw = build_ow_rows(a_s)
                a = {}
                for col in allcols:
                    r = run(rw, col_days[col]); b = bench_win.get(col, bench_reg.get(col))
                    a[col] = (r["ret"] - b) if r and b is not None else None
                vals = [v for v in a.values() if v is not None]
                scan_results.append((sm, lg, a, min(vals) if vals else None,
                                     sum(vals) / len(vals) if vals else None))
        logger.info("=== SCAN smooth×lag (mean/worst alpha over cols) ===")
        for sm, lg, a, w, m in scan_results:
            logger.info(f"  smooth={sm} lag={lg}: mean={m:+.1f} worst={w:+.1f} | "
                        + " ".join(f"{k}{v:+.0f}" for k, v in a.items() if v is not None))

    # ── 報告(v15 多窗格式) ──
    RPT = ROOT / "reports" / "aetf_overweight.md"
    RPT.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# 方法#4 主動權重 alpha(靜態超配,非流向)\n",
         f"> 28 ETF-blend 權重 − 流動性基準權重 = 主動超配;平滑{args.smooth}日 lag={args.lag}"
         f"｜候選每日 top-{TOPN} 高超配｜⑤買收賣開執行｜結束 {END}｜池 {len(pool)} 檔｜還原價\n",
         f"> baseline = H+反彈雙引擎(專案既定)｜成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到"
         f"｜ALPHA = 策略 − 同資金 DCA 0050｜資金 15000+1000/日上限5萬\n",
         f"> ⚠️ 作弊(第一輪):time-leak(lag={args.lag},0=當日揭露當日交易)、"
         f"membership/survivorship(當前28ETF+50池,多2025才上市)、基準用流動性近似0050\n",
         "> 0050 基準: " + " ".join(f"{wl}{bench_win[wl]:+.0f}%" for wl, _ in WINDOWS) + " | "
         + " ".join(f"{rn}{bench_reg[rn]:+.0f}%" for rn in REGIMES) + "\n",
         "## IC:主動超配 aow_s 對前向報酬(rank-IC, 全池橫斷面)\n",
         "| horizon | 5日 | 10日 | 20日 | 60日 |",
         "|---|---|---|---|---|",
         "| rank-IC | " + " | ".join(f"{ic_rows[h][0]:+.3f}" for h in (5, 10, 20, 60)) + " |",
         "| 樣本日數 | " + " | ".join(f"{ic_rows[h][1]}" for h in (5, 10, 20, 60)) + " |",
         "",
         "## 分位:依 aow_s 分 5 組,前向 20 日平均報酬 %(Q1=最低超配, Q5=最高)\n",
         "| Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 |",
         "|---|---|---|---|---|---|",
         "| " + " | ".join(f"{qmeans[q]:+.2f}" for q in range(5))
         + f" | **{qmeans[4]-qmeans[0]:+.2f}** |",
         "",
         "## ALPHA %(扣 0050)\n",
         "| 變體 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手 | 持股 | 曝險 |",
         "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|"]
    for label in ("baseline(H雙引擎)", "超配aow"):
        cells = []; avals = []
        for col in allcols:
            a = alpha(label, col)
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None:
                avals.append(a)
        ref = res.get((label, "1年")) or res.get((label, allcols[0]))
        worst = f"{min(avals):+.0f}" if avals else "—"
        mean = f"{sum(avals)/len(avals):+.0f}" if avals else "—"
        turn = f"{ref['turn']:.1f}x" if ref else "—"
        pos = f"{ref.get('avg_pos',0):.1f}" if ref else "—"
        expo = f"{ref.get('avg_expo',0)*100:.0f}%" if ref else "—"
        L.append(f"| {label} | " + " | ".join(cells) + f" | **{worst}** | {mean} | {turn} | {pos} | {expo} |")

    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    for label in ("baseline(H雙引擎)", "超配aow"):
        L.append(f"| {label} | " + " | ".join(
            f"{res[(label,col)]['ret']:+.0f}" if res.get((label, col)) else "—" for col in allcols) + " |")

    L += ["", "## Uplift = 超配 alpha − baseline alpha(pp,逐窗)\n",
          "| " + " | ".join(allcols) + " |",
          "|" + "|".join(["---"] * len(allcols)) + "|"]
    cells = []
    for col in allcols:
        a = alpha("超配aow", col); b = alpha("baseline(H雙引擎)", col)
        cells.append(f"{a-b:+.0f}" if (a is not None and b is not None) else "—")
    L.append("| " + " | ".join(cells) + " |")

    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    # console
    logger.info("IC(5/10/20/60): " + " ".join(f"{ic_rows[h][0]:+.3f}" for h in (5, 10, 20, 60)))
    logger.info(f"Q5-Q1(20日): {qmeans[4]-qmeans[0]:+.2f}%")
    for col in allcols:
        a = alpha("超配aow", col); b = alpha("baseline(H雙引擎)", col)
        if a is not None and b is not None:
            logger.info(f"  {col}: 超配α{a:+.0f} baseline α{b:+.0f} uplift{a-b:+.0f}")


if __name__ == "__main__":
    main()
