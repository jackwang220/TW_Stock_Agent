"""對抗式 leak-free 驗證:方法#10 經理人背書 × H 交互濾網。

把第一輪的作弊全改嚴格:
  1) 時點洩漏 → ETF 持股盤後揭露,決策日 D 只能用「真正前一交易日 D-1(或更早)」的背書。
     原檔的 lag 是用 ETF-date list 位移,但 18 個 ETF date 不是交易日 → 基準錯位。
     本檔改用「交易日對齊的 forward lag」:對每個交易決策日 d,用 ≤ d-lag 個交易日內
     最近一次有揭露的背書分。lag≥1 才是 leak-free 下限(揭露在盤後)。
  2) in-sample 參數挑選 → 不掃描挑 max。用「設計時就固定」的 k=0.6/mom=10/bidir
     (曝險中性那組,非 scan 冠軍 lag1-mom5-boost-k0.6)。同時印出鄰近參數穩定性。
  3) survivorship/membership → 1 年資料無法回避(28ETF 多 2025 才上市、50 大型股是事後贏家)。
     明確標記為「無法移除的結構性作弊」,並用 placebo 檢定看 uplift 是否只是擾動權重的噪音。

額外稽核:
  - 曝險中性:bidir(以1.0為中心)vs boost(只放大>1.0,曝險膨脹) 兩者曝險都印。
  - 集中 skew:印 avg_pos(平均持股數)。
  - Placebo:把 endorse 分數在池內「跨股隨機洗牌」(保留分佈、破壞個股對應),
    重抽多次看 uplift 分佈;若真 uplift 落在 placebo 分佈內 → 無真實 edge。

用法:
    uv run python scripts/aetf_interact_h_leakfree.py
"""
from __future__ import annotations
import sys, json, importlib.util, math, random
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
sim5 = ec.sim_buyclose_sellopen

END = "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]
REGIMES = {"2025下半": ("2025-07-01", "2025-12-31"), "2026上半": ("2026-01-01", END)}
ETF_CSV = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"
TOPN = 4
N_ETF = 28


def _z(vals):
    v = list(vals); n = len(v)
    if n < 2: return [0.0] * n
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / n) or 1.0
    return [(x - mu) / sd for x in v]


def build_endorse_raw(mom_win: int):
    """回傳 endorse_by_etfdate[etf_date][c] = 池內合成 z(尚未 lag)。"""
    df = pd.read_csv(ETF_CSV, dtype={"Stock_Code": str, "ETF_Code": str})
    df["w"] = df["Weight(%)"].astype(float)
    df["dc"] = df["Daily_Change(%)"].astype(float)
    pool = sorted(df["Stock_Code"].unique())
    dlist = sorted(df["Date"].unique())
    raw = {}
    for d, g in df.groupby("Date"):
        breadth = g.groupby("Stock_Code")["ETF_Code"].nunique()
        conv = g.groupby("Stock_Code")["w"].mean()
        dc = g.groupby("Stock_Code")["dc"].sum()
        raw[d] = {c: {"b": float(breadth.get(c, 0)) / N_ETF,
                      "v": float(conv.get(c, 0.0)),
                      "m": float(dc.get(c, 0.0))} for c in pool}
    mom = {}
    for i, d in enumerate(dlist):
        win = dlist[max(0, i - mom_win + 1): i + 1]
        mom[d] = {c: sum(raw[w][c]["m"] for w in win) for c in pool}
    endorse = {}
    for d in dlist:
        bz = _z(raw[d][c]["b"] for c in pool)
        vz = _z(raw[d][c]["v"] for c in pool)
        mz = _z(mom[d][c] for c in pool)
        endorse[d] = {c: 0.5 * bz[i] + 0.3 * vz[i] + 0.2 * mz[i] for i, c in enumerate(pool)}
    return endorse, pool, dlist


def lag_align(endorse_by_etfdate, etf_dates, trade_days, lag):
    """leak-free 對齊:決策交易日 d 只能用「揭露日 ≤ (d 的前 lag 個交易日)」最近一次背書。
    lag=1 → 用嚴格早於 d 的最近揭露(盤後揭露下限)。"""
    # 每個揭露(etf_date)對應「最早可交易的交易日索引」:該 etf_date 之後第 lag 個交易日
    td_index = {d: i for i, d in enumerate(trade_days)}
    # 對交易日 d,可用的揭露 = 揭露日 e 使得 e 在交易日序上 ≤ d 的前 lag 位
    # 先把每個揭露日 map 到「它在交易日軸上的位置」(取 ≤ 它的最後一個交易日)
    aligned = {}
    for di, d in enumerate(trade_days):
        # 允許使用的最新揭露日:其交易日位置 ≤ di - lag
        cutoff_idx = di - lag
        if cutoff_idx < 0:
            continue
        cutoff_day = trade_days[cutoff_idx]
        # 找 ≤ cutoff_day 的最近一個有揭露的 etf_date
        cand = [e for e in etf_dates if e <= cutoff_day and e in endorse_by_etfdate]
        if cand:
            aligned[d] = endorse_by_etfdate[max(cand)]
    return aligned


def mult_of(ez, k, mode):
    if mode == "boost":
        return 1.0 + k * max(0.0, ez)
    if mode == "bidir":
        return 1.0 + k * math.tanh(ez)
    return 1.0


def main():
    df = pd.read_csv(ETF_CSV, dtype={"Stock_Code": str})
    pool_all = sorted(df["Stock_Code"].unique())
    etf_dates = sorted(df["Date"].unique())
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

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    turns = {c: u.get(c, {}).get("avg_turnover", 0.0) for c in pool}

    limitup = {}
    for c in pool:
        o = oh(c); ds = sorted(o); s = set()
        for j, d in enumerate(ds):
            if j > 0 and o[ds[j - 1]]["close"] > 0 and o[d]["close"] / o[ds[j - 1]]["close"] - 1 >= 0.095:
                s.add(d)
        limitup[c] = s

    alld = sorted({d for c in pool for d in closes.get(c, {}) if etf_dates[0] <= d <= END})
    daylist = {"_list": alld}
    for i, d in enumerate(alld):
        daylist[d] = i
    logger.info(f"交易日 {alld[0]}~{alld[-1]} ({len(alld)})")

    # baseline H 雙引擎候選
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
    cand_by_day = {}
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
    base_rows = [(d, c, v / 100.0) for d in alld for v, c in cand_by_day[d]]

    def build_rows(aligned, k, mode):
        rows = []
        for d in alld:
            ez_map = aligned.get(d, {})
            for v, c in cand_by_day[d]:
                rows.append((d, c, (v / 100.0) * mult_of(ez_map.get(c, 0.0), k, mode)))
        return rows

    allcols = [wl for wl, _ in WINDOWS] + list(REGIMES.keys())
    col_days = {wl: alld[-n:] for wl, n in WINDOWS}
    col_days.update({rn: [d for d in alld if s <= d <= e] for rn, (s, e) in REGIMES.items()})

    def bench_for(dd):
        dd = sorted(d for d in dd if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    bench = {col: bench_for(col_days[col]) for col in allcols}

    def run(rows, dd):
        s = set(dd)
        return sim5([r for r in rows if r[0] in s], opens, closes, limitup, switch_cost_mult=1.0)

    base_res = {col: run(base_rows, col_days[col]) for col in allcols}

    # ── 主驗證:a-priori 固定參數 k=0.6 mom=10 bidir,lag 0/1/2 對比 ──
    L = ["# 對抗式 leak-free 驗證:方法#10 經理人背書 × H 交互濾網\n",
         f"> 池 {len(pool)} 檔｜每日 top-{TOPN}｜⑤買收賣開｜結束 {END}｜還原價｜ALPHA=策略−同資金DCA0050｜成本 買0.14%/賣0.44%+滑0.1%+漲停買不到\n",
         "> baseline = 純 H 雙引擎(交互≡1)。a-priori 固定參數 k=0.6 mom=10 **bidir**(曝險中性那組,非 scan 冠軍)。\n",
         "> ⚠️ **無法移除的結構性作弊(1年資料先天)**:membership/survivorship(當前28ETF多2025才上市、50檔是事後大型贏家)。本檔用 placebo 量化其影響。\n",
         "> 0050 基準: " + " ".join(f"{col}{bench[col]:+.0f}%" for col in allcols) + "\n",
         "## baseline α / 曝險 / 持股(参照)\n",
         "| 窗 | " + " | ".join(allcols) + " |",
         "|---|" + "|".join(["---"]*len(allcols)) + "|",
         "| baseline α% | " + " | ".join(f"{base_res[col]['ret']-bench[col]:+.0f}" for col in allcols) + " |",
         "| 曝險 | " + " | ".join(f"{base_res[col].get('avg_expo',0)*100:.0f}%" for col in allcols) + " |",
         "| 持股 | " + " | ".join(f"{base_res[col].get('avg_pos',0):.1f}" for col in allcols) + " |",
         ""]

    k, mom, mode = 0.6, 10, "bidir"
    endorse_raw, pool_e, _ = build_endorse_raw(mom)
    L += ["## leak-free 主結果:uplift = 交互α − baselineα(pp),lag 0(作弊)→1→2\n",
          "> lag=0 同日揭露當日交易=洩漏;lag≥1=盤後揭露下限(嚴格 leak-free,交易日對齊)。\n",
          "| lag | " + " | ".join(allcols) + " | 平均 | 最差 | 交互曝險(1年) | 交互持股(1年) |",
          "|---|" + "|".join(["---"]*(len(allcols)+4)) + "|"]
    lag_cells = {}
    for lag in (0, 1, 2):
        aligned = lag_align(endorse_raw, etf_dates, alld, lag)
        rows = build_rows(aligned, k, mode)
        ups = []; cells = []
        for col in allcols:
            r = run(rows, col_days[col])
            up = (r["ret"] - bench[col]) - (base_res[col]["ret"] - bench[col])
            cells.append(f"{up:+.0f}"); ups.append(up)
        r1 = run(rows, col_days["1年"])
        lag_cells[lag] = ups
        L.append(f"| {lag} | " + " | ".join(cells) +
                 f" | {sum(ups)/len(ups):+.1f} | {min(ups):+.0f} | {r1.get('avg_expo',0)*100:.0f}% | {r1.get('avg_pos',0):.1f} |")
        logger.info(f"lag={lag} uplift " + " ".join(f"{c}{u:+.0f}" for c,u in zip(allcols,ups)) + f" mean={sum(ups)/len(ups):+.1f}")

    # ── 鄰近參數穩定性(leak-free lag=1):看是否單調/穩定還是噪音 ──
    L += ["", "## 參數穩定性(leak-free lag=1):平均uplift 隨 k×mode\n",
          "> 若隨機正負、非單調 → in-sample 挑 max 是過擬合。\n",
          "| mode\\k | 0.3 | 0.6 | 1.0 |", "|---|---|---|---|"]
    aligned1 = lag_align(endorse_raw, etf_dates, alld, 1)
    for md in ("boost", "bidir"):
        row = [f"| {md} "]
        for kk in (0.3, 0.6, 1.0):
            rows = build_rows(aligned1, kk, md)
            ups = [(run(rows, col_days[col])["ret"] - bench[col]) - (base_res[col]["ret"] - bench[col]) for col in allcols]
            row.append(f"| {sum(ups)/len(ups):+.1f}")
        L.append("".join(row) + " |")

    # ── Placebo:洗牌 endorse(破壞個股對應、保留分佈) ──
    logger.info("placebo 洗牌檢定(lag=1, k0.6 bidir)...")
    real_ups = lag_cells.get(1)
    real_mean = sum(real_ups) / len(real_ups)
    real_1y = real_ups[allcols.index("1年")]
    rng = random.Random(42)
    placebo_means = []; placebo_1y = []
    NPLAC = 200
    for _ in range(NPLAC):
        # 每個交易日:把 aligned 該日的分數在 pool 內隨機重指派
        shuffled = {}
        for d, ez_map in aligned1.items():
            vals = list(ez_map.values())
            keys = list(ez_map.keys())
            rng.shuffle(vals)
            shuffled[d] = dict(zip(keys, vals))
        rows = build_rows(shuffled, k, mode)
        ups = [(run(rows, col_days[col])["ret"] - bench[col]) - (base_res[col]["ret"] - bench[col]) for col in allcols]
        placebo_means.append(sum(ups)/len(ups))
        placebo_1y.append(ups[allcols.index("1年")])
    placebo_means.sort(); placebo_1y.sort()
    def pctile_rank(arr, v):
        return sum(1 for x in arr if x <= v) / len(arr)
    pr_mean = pctile_rank(placebo_means, real_mean)
    pr_1y = pctile_rank(placebo_1y, real_1y)
    L += ["", "## Placebo 檢定:把背書分跨股洗牌(保留分佈、破壞個股訊息),lag=1 k0.6 bidir\n",
          f"> 真實 uplift 若落在洗牌分佈內 → 「背書濾網」沒有個股級資訊,只是擾動權重的噪音。n={NPLAC}\n",
          "| 指標 | 真實 | placebo均值 | placebo 5% | placebo 95% | 真實所在百分位 |",
          "|---|---|---|---|---|---|",
          f"| 平均uplift | {real_mean:+.1f} | {sum(placebo_means)/len(placebo_means):+.1f} | {placebo_means[int(0.05*NPLAC)]:+.1f} | {placebo_means[int(0.95*NPLAC)]:+.1f} | {pr_mean*100:.0f}% |",
          f"| 1年uplift | {real_1y:+.1f} | {sum(placebo_1y)/len(placebo_1y):+.1f} | {placebo_1y[int(0.05*NPLAC)]:+.1f} | {placebo_1y[int(0.95*NPLAC)]:+.1f} | {pr_1y*100:.0f}% |"]

    L += ["", "## 判讀\n",
          f"- lag=0→1 主結果見上;leak-free 後留多少 uplift。",
          f"- placebo:真實平均uplift 百分位 {pr_mean*100:.0f}%、1年 {pr_1y*100:.0f}%。>5% 且 <95% → 與隨機擾動無異。"]

    RPT = ROOT / "reports" / "aetf_interact_h_leakfree.md"
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")
    logger.info(f"placebo: real_mean={real_mean:+.1f} pctile={pr_mean*100:.0f}% | real_1y={real_1y:+.1f} pctile={pr_1y*100:.0f}%")


if __name__ == "__main__":
    main()
