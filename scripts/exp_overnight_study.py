"""隔夜 vs 日間報酬分解研究(學術②:Overnight vs Intraday returns)。
驗證你 ⑤「買收盤吃隔夜溢價」的根基,並測:
  A. 全體:隔夜(昨收→今開)vs 日間(今開→今收)平均報酬——隔夜溢價存不存在?
  B. 依成交值分大/中/小:溢價是否集中在中小型(實務說大型股無效)?
  C. 鎖漲停反轉:t 日鎖漲停 → t+1 隔夜(開高?)+ t+1 日間(倒貨反轉?)
  D. 動能條件:昨日強勢(日間漲幅前10%)→ 今日隔夜溢價多大?
資料=base_universe_v2.json(259檔,跨大小型);還原日線;2021~。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2021-01-01"


def main():
    u = json.loads((DATA_DIR / "base_universe_v2.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    print(f"載入 {len(codes)} 檔 ...")
    rows = []   # (code, date, overnight, intraday, prev_limitup, prev_intraday_strong, turnover_grp)
    # 成交值分層
    tv = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    qs = np.quantile([v for v in tv.values() if v > 0], [1/3, 2/3])
    def grp(c):
        v = tv[c]
        return "大型" if v >= qs[1] else ("中型" if v >= qs[0] else "小型")

    all_on, all_id = [], []
    by_grp = {"大型": [[], []], "中型": [[], []], "小型": [[], []]}
    lu_next_on, lu_next_id = [], []      # 鎖漲停隔日
    nonlu_next_on, nonlu_next_id = [], []
    strong_next_on = []                  # 昨日日間強勢→今隔夜

    for c in codes:
        o = get_daily_ohlcv(c, start=START)
        if not o or len(o) < 60: continue
        ds = sorted(o)
        op = np.array([o[d]["open"] for d in ds], float)
        cl = np.array([o[d]["close"] for d in ds], float)
        on = op[1:] / cl[:-1] - 1          # 隔夜:昨收→今開
        idr = cl[1:] / op[1:] - 1          # 日間:今開→今收
        c2c = cl[1:] / cl[:-1] - 1         # 昨收→今收(判漲停用)
        idr_prev_full = cl / op - 1        # 每日日間(對齊)
        g = grp(c)
        for i in range(len(on)):
            if not (np.isfinite(on[i]) and np.isfinite(idr[i])): continue
            all_on.append(on[i]); all_id.append(idr[i])
            by_grp[g][0].append(on[i]); by_grp[g][1].append(idr[i])
            # 前一日(index i 對應 ds[i+1];前一日=ds[i])是否鎖漲停(c2c at i 對應 ds[i+1]?)
        # 鎖漲停反轉:找 t 日 close-to-close >= 9.5%(視為當日鎖漲停),看 t+1
        for t in range(1, len(ds) - 1):
            lu = cl[t] / cl[t-1] - 1 >= 0.095
            n_on = op[t+1] / cl[t] - 1
            n_id = cl[t+1] / op[t+1] - 1
            if not (np.isfinite(n_on) and np.isfinite(n_id)): continue
            if lu:
                lu_next_on.append(n_on); lu_next_id.append(n_id)
            else:
                nonlu_next_on.append(n_on); nonlu_next_id.append(n_id)

    def m(x): return np.mean(x) * 100 if x else float("nan")
    def show(label, on, idr):
        print(f"  {label:<8} 隔夜均 {m(on):+.3f}%  日間均 {m(idr):+.3f}%  (n={len(on):,})")

    print("\n===== A. 全體:隔夜 vs 日間平均報酬 =====")
    show("全體", all_on, all_id)
    print(f"  → 隔夜年化≈{m(all_on)*252:+.0f}%  日間年化≈{m(all_id)*252:+.0f}%")
    print("\n===== B. 依成交值(大/中/小型) =====")
    for g in ["大型", "中型", "小型"]:
        show(g, by_grp[g][0], by_grp[g][1])
    print("\n===== C. 鎖漲停(t日)→ 隔日(t+1)反轉 =====")
    show("鎖漲停後", lu_next_on, lu_next_id)
    show("未漲停後", nonlu_next_on, nonlu_next_id)
    print("  → 若『鎖漲停後 隔夜大正、日間轉負』= 開高出貨反轉(⑤ 賣開盤的依據)")

    out = [
        "# 隔夜 vs 日間報酬研究(學術②;驗證 ⑤ 隔夜溢價)\n",
        f"> 259檔｜還原日線｜2021~｜隔夜=昨收→今開, 日間=今開→今收\n",
        "## A. 全體平均", f"- 隔夜 {m(all_on):+.3f}%/日(年化≈{m(all_on)*252:+.0f}%)｜日間 {m(all_id):+.3f}%/日(年化≈{m(all_id)*252:+.0f}%)\n",
        "## B. 依成交值", "| 類 | 隔夜均% | 日間均% |", "|---|---|---|"]
    for g in ["大型", "中型", "小型"]:
        out.append(f"| {g} | {m(by_grp[g][0]):+.3f} | {m(by_grp[g][1]):+.3f} |")
    out += ["", "## C. 鎖漲停隔日反轉", "| 情境 | t+1隔夜均% | t+1日間均% |", "|---|---|---|",
            f"| 鎖漲停後 | {m(lu_next_on):+.3f} | {m(lu_next_id):+.3f} |",
            f"| 未漲停後 | {m(nonlu_next_on):+.3f} | {m(nonlu_next_id):+.3f} |"]
    (ROOT / "reports" / "overnight_study.md").write_text("\n".join(out), encoding="utf-8")
    print("\n報告 → reports/overnight_study.md")


if __name__ == "__main__":
    main()
