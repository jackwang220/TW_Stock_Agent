"""Live 單日『新資金』配置：讀 signal_log 指定（預設最新）日期的訊號，套用與回測
相同的期望值加權 + 風控，輸出『今天該把今天的新資金配到哪』。

資金模型（與 run_backtest_60d.py / live_portfolio.py 同步）：
  - 第一天注入 INITIAL_CAPITAL（15,000），其後每個訊號日 +DAILY_BUDGET（1,000）；
    總投入達 MAX_CONTRIBUTION（50,000）後不再投入新資金。
  - 無單股持有上限；單股當天加碼 ≤ DAILY_ADD_CAP（15,000）。

注意：本工具只算『今天這筆新資金』要怎麼放（無狀態、不重平衡既有持股）。
      完整的累積帳本與每日再平衡請用 live_portfolio.py。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tw_stock_agent.config import SIGNAL_LOG

# 與 run_backtest_60d.py / live_portfolio.py 一致的資金模型（請保持同步）
INITIAL_CAPITAL  = 15000.0   # 第一天一次性注入
DAILY_BUDGET     = 1000.0    # 其後每個訊號日加碼
MAX_CONTRIBUTION = 50000.0   # 總投入上限，達到後不再投入新資金
MAX_SIGNALS     = 3
DAILY_ADD_CAP   = 15000.0    # 單支股票當天加碼上限（已取消持有上限）
EXPOSURE_CAP    = 0.90
EXPOSURE_FLOOR  = 0.30
TIE_RATIO       = 0.90

START_DATE = "2026-06-10"    # 帳本起始日（與 live_portfolio.py 同步）


def _contrib_through(idx: int) -> float:
    """到第 idx 個訊號日（0-based，含）為止的累計總投入。"""
    if idx < 0:
        return 0.0
    return min(MAX_CONTRIBUTION, INITIAL_CAPITAL + DAILY_BUDGET * idx)


def main(target_date: str | None = None) -> None:
    rows = list(csv.DictReader(open(SIGNAL_LOG, encoding="utf-8")))
    if not rows:
        print("signal_log 是空的")
        return

    # 所有 >= START_DATE 的訊號日（升冪）→ 決定今天是第幾天、該注入多少新資金
    all_dates = sorted({r.get("date", "") for r in rows if r.get("date", "") >= START_DATE})
    if not all_dates:
        print(f"signal_log 裡沒有 >= {START_DATE} 的訊號。")
        return
    d = target_date or all_dates[-1]
    day_idx = len([x for x in all_dates if x < d])     # 0-based：今天是第幾個訊號日
    pool = _contrib_through(day_idx) - _contrib_through(day_idx - 1)   # 今天的新資金

    day = [r for r in rows if r.get("date", "") == d]
    print(f"\n{'='*64}")
    print(f"  Live 新資金配置 — 訊號日 {d}（第 {day_idx + 1} 個交易日，預測下一交易日）")
    print(f"  今日新資金：{pool:,.0f}　｜　無持有上限、單股當天加碼 <={DAILY_ADD_CAP:,.0f}"
          f"｜曝險 {EXPOSURE_FLOOR:.0%}-{EXPOSURE_CAP:.0%}｜最多 {MAX_SIGNALS} 檔")
    print(f"{'='*64}")

    if pool <= 0:
        print(f"\n→ 總投入已達上限 {MAX_CONTRIBUTION:,.0f}，今日不再投入新資金。")
        print("  （既有持股的再平衡請看 live_portfolio.py）")
        return

    # up 訊號（predicted_direction=up 且 verdict≠REJECT）
    cands = []
    for r in day:
        if r.get("predicted_direction", "") != "up":
            continue
        if r.get("verdict", "") == "REJECT":
            continue
        try:
            center = float(r.get("predicted_center_pct", "") or 0)
            conf = float(r.get("prediction_confidence", "") or 0)
        except (ValueError, TypeError):
            center, conf = 0.0, 0.0
        edge = max(0.0, center) / 100.0 * max(0.0, conf)
        if edge <= 0:
            continue
        cands.append({"ticker": r["ticker"], "name": r.get("name", ""),
                      "edge": edge, "conf": conf, "center": center,
                      "bull": r.get("bull_score", ""), "bear": r.get("bear_score", "")})

    print(f"\n當日 PASS 訊號 {len(day)} 筆，其中可進場（up 且非 REJECT）{len(cands)} 筆")
    if not cands:
        print(f"→ 今日無 up 訊號 → 全數抱現金 {pool:,.0f}（這是紀律，不是失誤）")
        return

    # 依 edge 排名選股
    cands.sort(key=lambda x: x["edge"], reverse=True)
    selected = cands[:MAX_SIGNALS]
    if len(cands) > MAX_SIGNALS and cands[MAX_SIGNALS]["edge"] >= cands[MAX_SIGNALS - 1]["edge"] * TIE_RATIO:
        selected = cands[:MAX_SIGNALS + 1]

    # 曝險：信心驅動，clamp 到 [floor, cap]
    avg_conf = sum(s["conf"] for s in selected) / len(selected)
    exposure = min(EXPOSURE_CAP, max(EXPOSURE_FLOOR, avg_conf))

    wsum = sum(s["edge"] for s in selected)
    print(f"\n選中 {len(selected)} 檔，平均信心 {avg_conf:.2f} → 目標曝險 {exposure:.0%}"
          f"（投 {pool*exposure:,.0f}，留現金 {pool*(1-exposure):,.0f}）\n")
    print(f"{'代號':<6}{'名稱':<10}{'多/空':<8}{'信心':<6}{'預測%':<7}{'edge':<7}{'配置TWD':>8}")
    print("-" * 60)
    invested = 0.0
    for s in selected:
        raw = pool * exposure * (s["edge"] / wsum)
        amt = min(raw, DAILY_ADD_CAP)     # 單股當天加碼上限（無持有上限）
        invested += amt
        print(f"{s['ticker']:<6}{s['name']:<10}{str(s['bull'])+'/'+str(s['bear']):<8}"
              f"{s['conf']:<6.2f}{s['center']:<+7.1f}{s['edge']:<7.3f}{amt:>8.0f}")
    print("-" * 60)
    print(f"{'合計投入':<40}{invested:>8.0f}")
    print(f"{'留現金':<40}{pool - invested:>8.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)