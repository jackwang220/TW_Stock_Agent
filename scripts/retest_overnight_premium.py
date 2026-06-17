"""RETEST: overnight_premium 重現診斷。

原始 exp_overnight_study.py / exp_overnight_gradient.py 從未跑成功,因為它們讀
base_universe_v2.json(不存在;只有 base_universe.json 112檔)。本腳本:
  - 直接掃 data/finmind_cache 內所有 TaiwanStockPrice_*.json(離線,零網路)
  - 用 get_daily_ohlcv(同還原+清洗管線)→ 重現 study(A/B/C) + gradient(梯度)
  - avg_turnover 由快取自身 amount 平均算(取代缺失 universe 的成交值分層)
不改既有 exp_*.py。
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2021-01-01"
CACHE = DATA_DIR / "finmind_cache"


def main():
    codes = sorted({re.match(r"TaiwanStockPrice_(.+)\.json", p.name).group(1)
                    for p in CACHE.glob("TaiwanStockPrice_*.json")})
    print(f"快取股票 {len(codes)} 檔 ...")

    all_on, all_id = [], []
    lu_next_on, lu_next_id = [], []
    nonlu_next_on, nonlu_next_id = [], []
    # 成交值分層用
    tv = {}
    series = {}   # code -> (op, cl) arrays
    used = 0
    for c in codes:
        o = get_daily_ohlcv(c, start=START)
        if not o or len(o) < 60:
            continue
        ds = sorted(d for d in o if d >= START)
        if len(ds) < 60:
            continue
        op = np.array([o[d]["open"] for d in ds], float)
        cl = np.array([o[d]["close"] for d in ds], float)
        am = np.array([o[d].get("amount", 0) for d in ds], float)
        tv[c] = float(np.nanmean(am)) if len(am) else 0.0
        series[c] = (op, cl)
        used += 1
    print(f"有效(>=60bar) {used} 檔")

    # 成交值三分位
    posv = [v for v in tv.values() if v > 0]
    qs = np.quantile(posv, [1/3, 2/3]) if posv else (0, 0)
    def grp(c):
        v = tv.get(c, 0)
        return "大型" if v >= qs[1] else ("中型" if v >= qs[0] else "小型")
    by_grp = {"大型": [[], []], "中型": [[], []], "小型": [[], []]}

    # gradient 桶
    BINS = [("<= -5%", -1.0, -0.05), ("-5~-2%", -0.05, -0.02), ("-2~0%", -0.02, 0.0),
            ("0~2%", 0.0, 0.02), ("2~5%", 0.02, 0.05), ("5~7%", 0.05, 0.07),
            ("7~9.4%(強可買)", 0.07, 0.094), (">=9.4%(鎖漲停)", 0.094, 1.0)]
    bins_on = {b[0]: [] for b in BINS}
    bins_id = {b[0]: [] for b in BINS}

    for c, (op, cl) in series.items():
        g = grp(c)
        on = op[1:] / cl[:-1] - 1
        idr = cl[1:] / op[1:] - 1
        for i in range(len(on)):
            if not (np.isfinite(on[i]) and np.isfinite(idr[i])):
                continue
            all_on.append(on[i]); all_id.append(idr[i])
            by_grp[g][0].append(on[i]); by_grp[g][1].append(idr[i])
        for t in range(1, len(op) - 1):
            if cl[t-1] <= 0 or cl[t] <= 0 or op[t+1] <= 0:
                continue
            today = cl[t] / cl[t-1] - 1
            n_on = op[t+1] / cl[t] - 1
            n_id = cl[t+1] / op[t+1] - 1
            if not (np.isfinite(n_on) and np.isfinite(n_id)):
                continue
            lu = today >= 0.095
            (lu_next_on if lu else nonlu_next_on).append(n_on)
            (lu_next_id if lu else nonlu_next_id).append(n_id)
            for lab, lo, hi in BINS:
                if lo <= today < hi:
                    bins_on[lab].append(n_on); bins_id[lab].append(n_id); break

    def m(x): return float(np.mean(x)) * 100 if x else float("nan")
    def med(x): return float(np.median(x)) * 100 if x else float("nan")

    print("\n===== A. 全體:隔夜 vs 日間 =====")
    print(f"  全體 隔夜均 {m(all_on):+.4f}% (中位 {med(all_on):+.4f}%) 日間均 {m(all_id):+.4f}% (中位 {med(all_id):+.4f}%) n={len(all_on):,}")
    print(f"  -> 隔夜年化~{m(all_on)*252:+.1f}%  日間年化~{m(all_id)*252:+.1f}%")
    print("\n===== B. 成交值分層 =====")
    for g in ["大型", "中型", "小型"]:
        print(f"  {g} 隔夜均 {m(by_grp[g][0]):+.4f}% 日間均 {m(by_grp[g][1]):+.4f}% n={len(by_grp[g][0]):,}")
    print("\n===== C. 鎖漲停隔日 =====")
    print(f"  鎖漲停後 t+1隔夜 {m(lu_next_on):+.4f}% t+1日間 {m(lu_next_id):+.4f}% n={len(lu_next_on):,}")
    print(f"  未漲停後 t+1隔夜 {m(nonlu_next_on):+.4f}% t+1日間 {m(nonlu_next_id):+.4f}% n={len(nonlu_next_on):,}")
    print("\n===== 梯度: 今日漲跌幅 -> 隔夜 / 隔日日間 =====")
    print(f"{'今日漲跌幅':<18}{'隔夜均%':>10}{'隔日日間%':>12}{'n':>10}")
    for lab, _, _ in BINS:
        print(f"{lab:<18}{m(bins_on[lab]):>+10.4f}{m(bins_id[lab]):>+12.4f}{len(bins_on[lab]):>10,}")

    out = [
        "# RETEST overnight_premium (cache 直掃, 離線)\n",
        f"> {used}檔 | 還原日線 | {START}~ | 隔夜=昨收->今開, 日間=今開->今收\n",
        "## A. 全體",
        f"- 隔夜 {m(all_on):+.4f}%/日 (年化~{m(all_on)*252:+.1f}%) | 日間 {m(all_id):+.4f}%/日 (年化~{m(all_id)*252:+.1f}%) | n={len(all_on):,}\n",
        "## B. 成交值分層", "| 類 | 隔夜均% | 日間均% | n |", "|---|---|---|---|"]
    for g in ["大型", "中型", "小型"]:
        out.append(f"| {g} | {m(by_grp[g][0]):+.4f} | {m(by_grp[g][1]):+.4f} | {len(by_grp[g][0]):,} |")
    out += ["", "## C. 鎖漲停隔日", "| 情境 | t+1隔夜% | t+1日間% | n |", "|---|---|---|---|",
            f"| 鎖漲停後 | {m(lu_next_on):+.4f} | {m(lu_next_id):+.4f} | {len(lu_next_on):,} |",
            f"| 未漲停後 | {m(nonlu_next_on):+.4f} | {m(nonlu_next_id):+.4f} | {len(nonlu_next_on):,} |",
            "", "## 梯度", "| 今日漲跌幅 | 隔夜均% | 隔日日間% | n |", "|---|---|---|---|"]
    for lab, _, _ in BINS:
        out.append(f"| {lab} | {m(bins_on[lab]):+.4f} | {m(bins_id[lab]):+.4f} | {len(bins_on[lab]):,} |")
    (ROOT / "reports" / "retest_overnight_premium.md").write_text("\n".join(out), encoding="utf-8")
    print("\n報告 -> reports/retest_overnight_premium.md")


if __name__ == "__main__":
    main()
