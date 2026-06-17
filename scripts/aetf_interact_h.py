"""方法 #10：經理人背書 × H 交互濾網(⭐)

核心問句:不是獨立用經理人訊號(已證每日流向 tilt 無 edge),而是把它疊在 H 上當
「交互濾網」——H+反彈雙引擎選出的候選,若同時被多檔主動式 ETF 經理人「重押/加碼」
→ 加碼;H 看多但經理人正在「倒貨」→ 降級。測「兩者同意」的交互效應 vs 純 H baseline。

設計:
  base edge   = H+反彈雙引擎每日 top-N 候選分數(與專案既定 baseline 完全相同)。
  背書分 endorse(d,c) = 三個經理人維度的標準化合成(用「揭露日當日」資料,見作弊註記):
    breadth   = 持有 c 的 ETF 家數 / 28          (共識廣度)
    conviction= c 在持有它的 ETF 內平均權重 (重押程度,池內 z-score)
    momentum  = 近 M 日 Daily_Change(%) 累積       (加碼>0 / 倒貨<0)
  交互乘數 mult(d,c):
    "boost"   = 1 + k·max(0, e_z)                  (只在經理人也看好時加碼;不背書=中性 1.0)
    "bidir"   = 1 + k·tanh(e_z)                    (背書加碼 / 倒貨降級,雙向;曝險中性)
    其中 e_z = 0.5·breadth_z + 0.3·conviction_z + 0.2·momentum_z (池內橫斷面 z)
  tilted edge = base_edge × mult,餵 sim5(⑤買收賣開,與 baseline 同引擎/同 sizing)。
  baseline    = mult≡1 的純 H(交互關閉)。

驗證:
  1) 交互 IC:在「H 候選子集」上,endorse 對前向 H 日報酬的 rank-IC(才是交互該看的條件 IC)。
  2) 多窗 alpha vs 同資金 DCA 0050、扣成本、報換手/集中/曝險,uplift = 交互 − baseline。
  3) 掃 k(交互強度)× M(加碼動能窗)× mode 找最佳與單調性。

⚠️ 第一輪允許的作弊(cheat_used,Verify 階段再修):
  - 時點洩漏:ETF 持股當日揭露其實在盤後 → 本輪「揭露日當日」weight/Daily_Change 直接用
    (應 lag≥1 日)。提供 --lag 量化。H 特徵本身無洩漏(收盤後決策、⑤隔日開盤腿)。
  - membership/survivorship:用「當前」28 檔 ETF 成員與其 50 檔持股池(多 2025 才上市)。
不作弊:alpha 一律扣同期同資金 DCA 0050;成本 買0.1425%/賣0.4425%+滑價0.1%;漲停買不到;
  曝險中性比較(交互乘數以 1.0 為中心,baseline 同 sizing,報告印兩者曝險確認)。

用法:
    uv run python scripts/aetf_interact_h.py
    uv run python scripts/aetf_interact_h.py --scan
    uv run python scripts/aetf_interact_h.py --k 0.6 --mom 10 --mode bidir --lag 1
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
N_ETF = 28


# ── 載入經理人背書分(breadth/conviction/momentum 合成 z) ──────────────────────
def build_endorse(mom_win: int, lag: int):
    """回傳 endorse_z[d][c] = 池內橫斷面合成 z 分(已 lag)。
    breadth   = 持有家數 / 28
    conviction= 在持有它的 ETF 內平均權重
    momentum  = 近 mom_win 日 Daily_Change(%) 累積(加碼正/倒貨負)
    三項各自做池內橫斷面 z,再 0.5/0.3/0.2 合成。"""
    df = pd.read_csv(ETF_CSV, dtype={"Stock_Code": str, "ETF_Code": str})
    df["w"] = df["Weight(%)"].astype(float)
    df["dc"] = df["Daily_Change(%)"].astype(float)
    pool = sorted(df["Stock_Code"].unique())
    dlist = sorted(df["Date"].unique())

    # 每日原始三維
    raw: dict[str, dict[str, dict]] = {}   # d -> {c: {"b":breadth,"v":conv,"m":dc_today}}
    for d, g in df.groupby("Date"):
        breadth = g.groupby("Stock_Code")["ETF_Code"].nunique()
        conv = g.groupby("Stock_Code")["w"].mean()           # 持有它的 ETF 內平均權重
        dc = g.groupby("Stock_Code")["dc"].sum()             # 當日全 ETF 加總權重變動
        raw[d] = {c: {"b": float(breadth.get(c, 0)) / N_ETF,
                      "v": float(conv.get(c, 0.0)),
                      "m": float(dc.get(c, 0.0))} for c in pool}

    # momentum = 近 mom_win 日 dc 累積
    mom: dict[str, dict[str, float]] = {}
    for i, d in enumerate(dlist):
        win = dlist[max(0, i - mom_win + 1): i + 1]
        mom[d] = {c: sum(raw[w][c]["m"] for w in win) for c in pool}

    def _z(vals):
        v = list(vals)
        n = len(v)
        if n < 2:
            return [0.0] * n
        mu = sum(v) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / n) or 1.0
        return [(x - mu) / sd for x in v]

    endorse: dict[str, dict[str, float]] = {}
    for d in dlist:
        bz = _z(raw[d][c]["b"] for c in pool)
        vz = _z(raw[d][c]["v"] for c in pool)
        mz = _z(mom[d][c] for c in pool)
        endorse[d] = {c: 0.5 * bz[i] + 0.3 * vz[i] + 0.2 * mz[i] for i, c in enumerate(pool)}

    # lag:盤後揭露 → 決策日 d 只能用 d-lag 的背書
    if lag > 0:
        lagged = {}
        for i, d in enumerate(dlist):
            if i - lag >= 0:
                lagged[d] = endorse[dlist[i - lag]]
        endorse = lagged
    return endorse, pool, dlist


# ── IC 工具 ──────────────────────────────────────────────────────────────────
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
    idx = daylist.get(d)
    if idx is None or idx + h >= len(daylist["_list"]):
        return None
    d2 = daylist["_list"][idx + h]
    p0 = closes.get(c, {}).get(d); p1 = closes.get(c, {}).get(d2)
    if p0 and p1 and p0 > 0:
        return p1 / p0 - 1
    return None


def mult_of(ez, k, mode):
    """交互乘數,以 1.0 為中心。"""
    if mode == "boost":
        return 1.0 + k * max(0.0, ez)          # 只在經理人也看好時加碼
    if mode == "bidir":
        return 1.0 + k * math.tanh(ez)         # 背書加碼 / 倒貨降級(雙向,曝險中性)
    if mode == "gate":
        return 1.0 + k if ez > 0 else max(0.2, 1.0 - k)   # 硬門檻:同意放大/不同意縮
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=0.6, help="交互強度")
    ap.add_argument("--mom", type=int, default=10, help="加碼動能窗(交易日)")
    ap.add_argument("--mode", default="bidir", choices=["boost", "bidir", "gate"])
    ap.add_argument("--lag", type=int, default=0, help="0=作弊同日揭露當日交易;1=leak-free探針")
    ap.add_argument("--scan", action="store_true", help="掃 k×mom×mode×lag 找最佳與單調性")
    args = ap.parse_args()

    # 池(只取有價格的,與 baseline 一致;缺 base_universe 的 6 檔仍試抓 oh)
    df = pd.read_csv(ETF_CSV, dtype={"Stock_Code": str})
    pool_all = sorted(df["Stock_Code"].unique())
    logger.info(f"ETF 池 {len(pool_all)} 檔;載入價格...")
    opens, closes = {}, {}
    pool = []
    for c in pool_all:
        try:
            o = oh(c)
        except Exception:
            o = None
        if o and len(o) > 30:
            opens[c] = {d: o[d]["open"] for d in o}
            closes[c] = {d: o[d]["close"] for d in o}
            pool.append(c)
    o0 = oh("0050"); opens["0050"] = {d: o0[d]["open"] for d in o0}; closes["0050"] = {d: o0[d]["close"] for d in o0}
    logger.info(f"有價格池 {len(pool)} 檔")

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    turns = {c: u.get(c, {}).get("avg_turnover", 0.0) for c in pool}

    # 漲停日
    limitup = {}
    for c in pool:
        o = oh(c); ds = sorted(o); s = set()
        for j, d in enumerate(ds):
            if j > 0 and o[ds[j - 1]]["close"] > 0 and o[d]["close"] / o[ds[j - 1]]["close"] - 1 >= 0.095:
                s.add(d)
        limitup[c] = s

    # 交易日(在 ETF 資料範圍內)
    etf_dates = sorted(df["Date"].unique())
    alld = sorted({d for c in pool for d in closes.get(c, {}) if etf_dates[0] <= d <= END})
    daylist = {"_list": alld}
    for i, d in enumerate(alld):
        daylist[d] = i
    logger.info(f"交易日 {alld[0]}~{alld[-1]} ({len(alld)})")

    # ── baseline:H+反彈雙引擎 base edge(專案既定 baseline) ──
    logger.info("建 H+反彈雙引擎候選(baseline)...")
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

    # 每日 H 候選(分數),保留 (d, c, base_edge[0..]) 與每日候選清單供交互
    cand_by_day: dict[str, list] = {}   # d -> [(score, c), ...] top-N
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
        cand_by_day[d] = sc[:TOPN]

    n_cand = sum(len(v) for v in cand_by_day.values())
    logger.info(f"H 候選 {n_cand} 筆(每日 top-{TOPN})")

    base_rows = [(d, c, v / 100.0) for d in alld for v, c in cand_by_day[d]]

    def build_interact_rows(endorse, k, mode):
        """在 H 候選上疊交互乘數。endorse 缺(d 被 lag 掉或 c 不在)→ 中性 1.0。"""
        rows = []
        for d in alld:
            ez_map = endorse.get(d, {})
            for v, c in cand_by_day[d]:
                ez = ez_map.get(c, 0.0)
                rows.append((d, c, (v / 100.0) * mult_of(ez, k, mode)))
        return rows

    # ── 0050 基準 ──
    def bench_for(dd):
        dd = sorted(d for d in dd if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    allcols = [wl for wl, _ in WINDOWS] + list(REGIMES.keys())
    col_days = {wl: alld[-n:] for wl, n in WINDOWS}
    regime_days = {rn: [d for d in alld if s <= d <= e] for rn, (s, e) in REGIMES.items()}
    col_days.update(regime_days)
    bench = {col: bench_for(col_days[col]) for col in allcols}

    def run(rows, dd):
        s = set(dd)
        return sim5([r for r in rows if r[0] in s], opens, closes, limitup, switch_cost_mult=1.0)

    # ── 交互條件 IC:只在 H 候選子集上,endorse 對前向報酬 ──
    endorse, _, _ = build_endorse(args.mom, args.lag)
    logger.info("計算交互條件 IC(僅 H 候選子集)...")
    ic_rows = {}
    for h in (5, 10, 20):
        ics = []
        for d in alld:
            ez = endorse.get(d)
            if not ez:
                continue
            xs, ys = [], []
            for _, c in cand_by_day[d]:
                fr = fwd_ret(closes, c, d, h, daylist)
                if fr is not None and c in ez:
                    xs.append(ez[c]); ys.append(fr)
            if len(xs) >= 4:
                ics.append(spearman(xs, ys))
        ic_rows[h] = (sum(ics) / len(ics) if ics else 0.0, len(ics))

    # ── 多窗回測:baseline vs 交互 ──
    inter_rows = build_interact_rows(endorse, args.k, args.mode)
    res = {}
    for label, rows in (("baseline(純H)", base_rows), (f"交互×k{args.k}_{args.mode}", inter_rows)):
        for col in allcols:
            res[(label, col)] = run(rows, col_days[col])
        logger.info(f"{label} 完成")
    inter_label = f"交互×k{args.k}_{args.mode}"

    def alpha(label, col):
        r = res.get((label, col)); b = bench.get(col)
        return (r["ret"] - b) if (r and b is not None) else None

    if args.scan:
        logger.info("=== SCAN k×mom×mode×lag (mean/worst uplift over cols) ===")
        base_a = {col: alpha("baseline(純H)", col) for col in allcols}
        scan_log = []
        for lg in (0, 1):
            for mw in (5, 10, 20):
                e_s, _, _ = build_endorse(mw, lg)
                for md in ("boost", "bidir", "gate"):
                    for kk in (0.3, 0.6, 1.0):
                        rw = build_interact_rows(e_s, kk, md)
                        ups = []
                        cellmap = {}
                        for col in allcols:
                            r = run(rw, col_days[col]); b = bench.get(col)
                            a = (r["ret"] - b) if r and b is not None else None
                            up = (a - base_a[col]) if (a is not None and base_a[col] is not None) else None
                            cellmap[col] = up
                            if up is not None:
                                ups.append(up)
                        mean = sum(ups) / len(ups) if ups else None
                        worst = min(ups) if ups else None
                        scan_log.append((lg, mw, md, kk, mean, worst, cellmap))
                        logger.info(f"  lag={lg} mom={mw} {md:5s} k={kk}: mean_uplift={mean:+.1f} worst={worst:+.1f} | "
                                    + " ".join(f"{c}{v:+.0f}" for c, v in cellmap.items() if v is not None))
        # 最佳(以 mean uplift)
        best = max((s for s in scan_log if s[4] is not None), key=lambda s: s[4], default=None)
        if best:
            logger.success(f"最佳 mean uplift: lag={best[0]} mom={best[1]} mode={best[2]} k={best[3]} mean={best[4]:+.1f} worst={best[5]:+.1f}")

    # ── 報告(v15 多窗格式) ──
    RPT = ROOT / "reports" / "aetf_interact_h.md"
    RPT.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# 方法#10 經理人背書 × H 交互濾網\n",
         f"> H+反彈雙引擎候選 × 經理人背書乘數(breadth0.5/conviction0.3/momentum0.2 池內z)"
         f"｜k={args.k} mode={args.mode} mom={args.mom}日 lag={args.lag}｜每日 top-{TOPN}｜⑤買收賣開｜結束 {END}｜池 {len(pool)} 檔｜還原價\n",
         f"> baseline = 純 H(交互乘數≡1)｜成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到"
         f"｜ALPHA = 策略 − 同資金 DCA 0050｜資金 15000+1000/日上限5萬\n",
         f"> ⚠️ 作弊(第一輪):time-leak(lag={args.lag},0=當日揭露當日交易)、"
         f"membership/survivorship(當前28ETF+50池,多2025才上市);H 特徵本身無洩漏\n",
         "> 0050 基準: " + " ".join(f"{col}{bench[col]:+.0f}%" for col in allcols) + "\n",
         "## 交互條件 IC:H 候選子集內 endorse 對前向報酬(rank-IC)\n",
         "| horizon | 5日 | 10日 | 20日 |",
         "|---|---|---|---|",
         "| rank-IC | " + " | ".join(f"{ic_rows[h][0]:+.3f}" for h in (5, 10, 20)) + " |",
         "| 樣本日數 | " + " | ".join(f"{ic_rows[h][1]}" for h in (5, 10, 20)) + " |",
         "",
         "## ALPHA %(扣 0050)\n",
         "| 變體 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手 | 持股 | 曝險 |",
         "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|"]
    for label in ("baseline(純H)", inter_label):
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
    for label in ("baseline(純H)", inter_label):
        L.append(f"| {label} | " + " | ".join(
            f"{res[(label,col)]['ret']:+.0f}" if res.get((label, col)) else "—" for col in allcols) + " |")

    L += ["", "## Uplift = 交互 alpha − baseline alpha(pp,逐窗)\n",
          "| " + " | ".join(allcols) + " |",
          "|" + "|".join(["---"] * len(allcols)) + "|"]
    cells = []
    for col in allcols:
        a = alpha(inter_label, col); b = alpha("baseline(純H)", col)
        cells.append(f"{a-b:+.0f}" if (a is not None and b is not None) else "—")
    L.append("| " + " | ".join(cells) + " |")

    L += ["", "## 判讀\n",
          "- 交互 IC>0 且 uplift 各窗正、隨 k 單調 → 「H×經理人同意」有真實交互 edge。",
          "- IC≈0、uplift 隨機正負 → 背書濾網對 H 候選沒額外資訊(與每日流向 tilt 無 edge 一致)。",
          "- 曝險欄:交互乘數以 1.0 為中心,與 baseline 曝險接近才是公平比較。"]
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    logger.info("交互IC(5/10/20): " + " ".join(f"{ic_rows[h][0]:+.3f}" for h in (5, 10, 20)))
    for col in allcols:
        a = alpha(inter_label, col); b = alpha("baseline(純H)", col)
        if a is not None and b is not None:
            logger.info(f"  {col}: 交互α{a:+.0f} baseline α{b:+.0f} uplift{a-b:+.0f}")


if __name__ == "__main__":
    main()
