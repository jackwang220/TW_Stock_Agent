"""#9 類股輪動擇時 — 用「經理人集體往哪類股搬錢」當輪動訊號。

哲學:個股流向已證無 edge(etf_consensus_probe rank-IC≈0),但類股層級雜訊較小。
做法:每日把 28 檔主動式 ETF 的持股加總成「類股總權重」(= sum Weight% across all ETFs),
類股資金流 flow_s(t) = 類股總權重(t) − 類股總權重(t−L)(L=lookback)。
flow>0 = 經理人集體加碼該類股(資金流入) → 超配;flow<0 = 流出 → 低配。

把類股流向轉成個股 edge:選 flow 排名前 K 的類股,對其中可交易的個股(在 base_universe)
給 edge ∝ 類股 flow(類股內個股等權 或 依當前 ETF 權重)。edge 餵進 ⑤買收賣開引擎,
引擎負責成本/曝險/換手/選 top-3。baseline = 0050 同資金 DCA(由 v6.bench_0050 算)。

執行/成本/曝險/換手由既有 sim5 引擎處理(不作弊):買0.14%/賣0.44%+滑價0.1%+漲停買不到。
ALPHA = 策略報酬 − 同資金 DCA 0050。

═══ 第一輪作弊註記(cheat_used)═══
  1. 時點洩漏:ETF 持股是盤後揭露,本版用「同日揭露的持股當日(收盤)就交易」
     → flow_s(t) 用到 t 日的持股,而 t 日持股實際要 t 日盤後才知道。⑤引擎收盤腿用 close[d]
     成交,所以等於拿當天盤後才知道的持股去當天收盤下單 = 時點洩漏(約 1 日)。
  2. 當前 ETF 成員 / 池固定:用的是這 28 檔 ETF 當前的 50 檔成員池,survivorship。
  3. 類股映射用 in-sample 全期 industry 標籤(不變,影響極小)。
  * 基準(DCA0050)、成本、曝險中性、換手 — 全部不作弊。

用法:
    uv run python scripts/aetf_rotation.py                  # 掃 lookback × topK × 加權
    uv run python scripts/aetf_rotation.py --quick          # 只跑最佳設定
"""
from __future__ import annotations
import sys, json, importlib.util, argparse
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

v3 = _load("v3", ROOT / "scripts/exp_step1_v3.py")
v6 = _load("v6", ROOT / "scripts/exp_step1_v6.py")
ec = _load("ec", ROOT / "scripts/exp_60d_entry_compare.py")
oh = v3.oh
sim5 = ec.sim_buyclose_sellopen

END = "2026-06-08"   # 對齊既有報告(資料到 06-17,留尾巴給隔日開盤估值)
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]
# 資料僅 2025-06-17~2026-06-17 → 只有兩個可得 regime 段
REGIMES = {
    "2025下半": ("2025-07-01", "2025-12-31"),
    "2026上半": ("2026-01-01", END),
}
CSV = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"


def load_sector_flows():
    """回傳 (sector_weight piv[Date×industry], ind map{code:industry}, pool_by_ind{ind:[codes]})。
    sector_weight = 每日 sum(Weight%) across 全部 28 檔 ETF(經理人集體曝險)。"""
    df = pd.read_csv(CSV, dtype={"Stock_Code": str, "ETF_Code": str})
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    idx = json.loads((DATA_DIR / "tw_stock_index.json").read_text(encoding="utf-8"))
    ind = {}
    for c in df.Stock_Code.unique():
        if c in u:
            ind[c] = u[c].get("industry")
        elif c in idx:
            ind[c] = idx[c].get("industry")
        else:
            ind[c] = "其他"
    df["ind"] = df.Stock_Code.map(ind)
    piv = df.groupby(["Date", "ind"])["Weight(%)"].sum().unstack().fillna(0.0).sort_index()
    # 池內可交易(在 base_universe)個股 by 類股, 並存每檔當日權重供權重模式用
    uset = set(u.keys())
    pool_by_ind = defaultdict(list)
    for c in df.Stock_Code.unique():
        if c in uset:
            pool_by_ind[ind[c]].append(c)
    # 每日每檔的 ETF 加總權重(供 weighted 模式)
    stock_w = df.groupby(["Date", "Stock_Code"])["Weight(%)"].sum().unstack().fillna(0.0).sort_index()
    return piv, dict(ind), dict(pool_by_ind), stock_w


def build_rotation_rows(piv, pool_by_ind, stock_w, sig_days, lookback, topk, weight_mode):
    """每日:類股 flow = 類股總權重(t) − (t−lookback);取 flow 排名 top-K 的類股,
    對其中可交易個股給 edge。edge 標準化到 ~0..1(⑤引擎吃 edge 當權重/信心)。

    weight_mode:
      "equal"   = 類股內個股等權,edge = 正規化 flow(類股級)
      "stockw"  = edge ∝ 類股flow × (該股當日 ETF 權重 / 類股權重) — 偏向經理人重壓的個股
    rows = [(date, code, edge)]。
    """
    dlist = list(piv.index)
    pos = {d: i for i, d in enumerate(dlist)}
    rows = []
    for d in sig_days:
        if d not in pos:
            continue
        i = pos[d]
        if i < lookback:
            continue
        d0 = dlist[i - lookback]
        flow = (piv.loc[d] - piv.loc[d0])   # Series industry->Δweight
        # 只取流入(flow>0)且有可交易個股的類股
        cand = [(s, flow[s]) for s in flow.index if flow[s] > 0 and pool_by_ind.get(s)]
        cand.sort(key=lambda x: -x[1])
        cand = cand[:topk]
        if not cand:
            continue
        fmax = max(f for _, f in cand)
        if fmax <= 0:
            continue
        for s, f in cand:
            sect_edge = f / fmax   # 0..1, 流入越強 edge 越大
            codes = pool_by_ind[s]
            if weight_mode == "equal":
                for c in codes:
                    rows.append((d, c, max(0.05, sect_edge)))
            else:  # stockw
                wser = stock_w.loc[d] if d in stock_w.index else None
                wsum = sum((wser.get(c, 0.0) if wser is not None else 1.0) for c in codes) or 1.0
                for c in codes:
                    w = (wser.get(c, 0.0) if wser is not None else 1.0)
                    rows.append((d, c, max(0.05, sect_edge * (w / wsum) * len(codes))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    logger.info("載入類股流向...")
    piv, ind, pool_by_ind, stock_w = load_sector_flows()
    pool_codes = sorted({c for cs in pool_by_ind.values() for c in cs})
    logger.info(f"可交易池 {len(pool_codes)} 檔, {len(pool_by_ind)} 類股, 日期 {len(piv)}")

    logger.info("載入價格(開/收盤/漲停)...")
    opens, closes, limitup = {}, {}, {}
    for c in pool_codes + ["0050"]:
        o = oh(c); ds = sorted(d for d in o if d <= END)
        opens[c] = {d: o[d]["open"] for d in ds}
        closes[c] = {d: o[d]["close"] for d in ds}
        s = set()
        for j, d in enumerate(ds):
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095:
                s.add(d)
        limitup[c] = s

    # 訊號日 = 池股共同交易日(≤END)
    sig_days = sorted({d for c in pool_codes for d in closes.get(c, {}) if d <= END})
    logger.info(f"訊號範圍 {sig_days[0]} ~ {sig_days[-1]} ({len(sig_days)} 日)")

    # 基準: 0050 同資金 DCA(各窗 + 各 regime)
    def bench_for(day_set):
        dd = sorted(d for d in day_set if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    bench_win = {wl: bench_for(set(sig_days[-n:])) for wl, n in WINDOWS}
    regime_days = {rn: set(d for d in sig_days if s <= d <= e) for rn, (s, e) in REGIMES.items()}
    bench_reg = {rn: bench_for(dd) for rn, dd in regime_days.items()}

    # 參數掃描
    if args.quick:
        configs = [(10, 3, "equal")]
    else:
        configs = []
        for lb in [3, 5, 10, 20]:
            for tk in [2, 3, 5]:
                for wm in ["equal", "stockw"]:
                    configs.append((lb, tk, wm))

    allcols = [wl for wl, _ in WINDOWS] + list(REGIMES.keys())
    results = {}   # label -> {col: sim}
    rows_cache = {}
    for lb, tk, wm in configs:
        label = f"L{lb}_K{tk}_{wm}"
        rows_full = build_rotation_rows(piv, pool_by_ind, stock_w, sig_days, lb, tk, wm)
        rows_cache[label] = rows_full
        per = {}
        for wl, n in WINDOWS:
            wd = set(sig_days[-n:]); rw = [r for r in rows_full if r[0] in wd]
            per[wl] = sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)
        for rn, dd in regime_days.items():
            rw = [r for r in rows_full if r[0] in dd]
            per[rn] = sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)
        results[label] = per
        logger.info(f"{label} 完成")

    def alpha(label, col):
        r = results[label].get(col)
        b = bench_win.get(col, bench_reg.get(col))
        return (r["ret"] - b) if (r and b is not None) else None

    # 排序:1年 alpha
    def sortkey(label):
        a = alpha(label, "1年")
        return a if a is not None else -999
    order = sorted(results.keys(), key=sortkey, reverse=True)

    L = ["# AETF #9 類股輪動擇時 — 經理人集體類股資金流 超配/低配\n",
         f"> 訊號={{28檔ETF類股總權重 Δ(lookback)}}的 top-K 流入類股｜⑤買收賣開執行｜結束 {END}｜池 {len(pool_codes)} 檔｜還原價\n",
         f"> baseline=同資金 DCA 0050｜成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到｜ALPHA=策略−DCA0050\n",
         f"> 資金 期初15000+1000/日上限5萬｜⑤引擎選 top-3(分數接近放行4)\n",
         "> ⚠️ 作弊(第一輪): (1)時點洩漏~1日(同日揭露持股當日收盤就交易;ETF持股實為盤後揭露)"
         " (2)當前ETF成員/池固定=survivorship (3)類股標籤用全期\n",
         "> 0050 基準: " + " ".join(f"{wl}{bench_win[wl]:+.0f}%" for wl, _ in WINDOWS) + " | "
         + " ".join(f"{rn}{bench_reg[rn]:+.0f}%" for rn in REGIMES) + "\n",
         "## ALPHA %(扣 0050 beta;排序=1年 alpha)\n",
         "| 設定(L=lookback,K=topK) | " + " | ".join(allcols) + " | 最差 | 平均 | 換手(1年) | 持股(1年) | 曝險(1年) |",
         "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|"]
    for label in order:
        cells, avals = [], []
        for col in allcols:
            a = alpha(label, col)
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None:
                avals.append(a)
        worst = f"{min(avals):+.0f}" if avals else "—"
        mean = f"{sum(avals)/len(avals):+.0f}" if avals else "—"
        r1y = results[label].get("1年")
        turn = f"{r1y['turn']:.1f}x" if r1y else "—"
        pos = f"{r1y.get('pos_avg', r1y.get('avg_pos', 0)):.1f}" if r1y else "—"
        expo = f"{r1y.get('avg_expo', 0)*100:.0f}%" if r1y else "—"
        L.append(f"| {label} | " + " | ".join(cells) + f" | **{worst}** | {mean} | {turn} | {pos} | {expo} |")

    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 設定 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    for label in order:
        cells = [f"{results[label][col]['ret']:+.0f}" if results[label].get(col) else "—" for col in allcols]
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    best = order[0]
    L += ["", "## 判讀\n",
          f"- 最佳設定 {best}: 1年 alpha {alpha(best,'1年'):+.0f}pp。",
          "- 註:資料僅 1 個多頭年、池偏大型股 → 缺空頭壓力測試,結論有「缺空頭」天花板。",
          "- 若各窗 alpha 多為負/接近 0 → 類股輪動流向無擇時 edge(與個股流向 rank-IC≈0 一致)。"]
    RPT = ROOT / "reports" / "aetf_rotation.md"
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    # console 摘要
    for label in order[:6]:
        a = {c: alpha(label, c) for c in ["60天", "90天", "半年", "1年"]}
        logger.success(f"{label}: " + " ".join(f"{k}{v:+.0f}" if v is not None else f"{k}—" for k, v in a.items()))
    return results, order, alpha, bench_win, bench_reg, allcols


if __name__ == "__main__":
    main()
