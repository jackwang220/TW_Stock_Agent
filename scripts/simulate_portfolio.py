"""模擬交易系統：用 LLM 回測數據回放交易，評估每日固定預算策略的績效。

資料來源：data/backtest_llm_results.csv（216+ 筆 LLM 預測 + 實際漲跌）

使用範例：
    # 每天最多投入 50000，持有 1 天，只買 PASS + predicted=up
    uv run python scripts/simulate_portfolio.py --daily-budget 50000

    # 帶信心門檻 + 5 日持有
    uv run python scripts/simulate_portfolio.py --daily-budget 100000 --hold-days 5 --min-confidence 0.6

    # 包含 WARN 訊號
    uv run python scripts/simulate_portfolio.py --daily-budget 50000 --signal-filter pass-warn
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LLM_CSV = DATA_DIR / "backtest_llm_results.csv"


# ──────────────────────────────────────────────
# 資料載入
# ──────────────────────────────────────────────

def load_signals(
    signal_filter: str = "pass",      # "pass" | "pass-warn" | "all"
    direction_filter: str = "up",     # "up" | "all"
    min_confidence: float = 0.0,
    include_reject: bool = False,
) -> list[dict]:
    """載入並篩選 LLM 訊號。"""
    if not LLM_CSV.exists():
        sys.exit(f"[ERROR] 找不到 {LLM_CSV}，請先執行 backtest_with_llm.py")

    rows = []
    with open(LLM_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # 需要有實際回報才能模擬
            ret_1d = r.get("return_1d", "")
            ret_5d = r.get("return_5d", "")
            if not ret_1d or not ret_5d:
                continue

            pass_level = r.get("pass_level", "")
            if signal_filter == "pass" and pass_level != "PASS":
                continue
            if signal_filter == "pass-warn" and pass_level not in ("PASS", "WARN"):
                continue

            predicted = r.get("predicted_direction", "")
            if direction_filter == "up" and predicted != "up":
                continue

            verdict = r.get("llm_verdict", "")
            if not include_reject and verdict == "REJECT":
                continue

            try:
                conf = float(r.get("prediction_confidence") or 0)
            except ValueError:
                conf = 0.0
            if conf < min_confidence:
                continue

            try:
                close = float(r["close_price"])
                r1 = float(ret_1d)
                r5 = float(ret_5d)
            except (ValueError, KeyError):
                continue

            rows.append({
                "date": r["date"],
                "ticker": r["ticker"],
                "name": r["name"],
                "pass_level": pass_level,
                "pattern_type": r.get("pattern_type", ""),
                "predicted_direction": predicted,
                "prediction_confidence": conf,
                "verdict": verdict,
                "close_price": close,
                "return_1d": r1,
                "return_5d": r5,
                "twii_1d": float(r["twii_1d"]) if r.get("twii_1d") else None,
                "alpha_1d": float(r["alpha_1d"]) if r.get("alpha_1d") else None,
            })

    rows.sort(key=lambda x: x["date"])
    return rows


# ──────────────────────────────────────────────
# 模擬引擎
# ──────────────────────────────────────────────

class Portfolio:
    def __init__(self, initial_capital: float, max_positions: int):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.open_positions: list[dict] = []   # 持倉中
        self.closed_trades: list[dict] = []    # 已平倉
        self.daily_values: list[dict] = []     # 每日淨值

    def total_value(self) -> float:
        pos_value = sum(p["cost"] for p in self.open_positions)
        return self.cash + pos_value

    def record_day(self, date: str):
        self.daily_values.append({
            "date": date,
            "portfolio_value": self.total_value(),
            "cash": self.cash,
            "open_positions": len(self.open_positions),
        })


def simulate(
    signals: list[dict],
    initial_capital: float = 500_000,
    daily_budget: float = 50_000,
    hold_days: int = 1,
    max_positions: int = 10,
    position_mode: str = "equal",   # "equal" = daily_budget/n_signals; "fixed" = daily_budget as per-position
    per_position_amount: float = 0,  # 只有 position_mode="fixed" 時用
) -> Portfolio:
    port = Portfolio(initial_capital, max_positions)

    # 按日分組訊號
    by_date: dict[str, list[dict]] = defaultdict(list)
    for s in signals:
        by_date[s["date"]].append(s)

    all_dates = sorted(by_date.keys())

    for date in all_dates:
        # 1. 平倉到期部位
        still_open = []
        for pos in port.open_positions:
            if pos["exit_date"] <= date:
                ret_pct = pos["return_1d"] if hold_days == 1 else pos["return_5d"]
                pnl = pos["cost"] * ret_pct / 100
                exit_value = pos["cost"] + pnl
                port.cash += exit_value
                port.closed_trades.append({
                    **pos,
                    "exit_date": date,
                    "return_pct": ret_pct,
                    "pnl": pnl,
                    "exit_value": exit_value,
                })
            else:
                still_open.append(pos)
        port.open_positions = still_open

        # 2. 今日可用新訊號
        today_signals = by_date[date]
        available_slots = max_positions - len(port.open_positions)
        if available_slots <= 0 or not today_signals:
            port.record_day(date)
            continue

        # 最多取 available_slots 個訊號（信心高的優先）
        today_signals = sorted(today_signals, key=lambda x: -x["prediction_confidence"])
        today_signals = today_signals[:available_slots]

        # 3. 計算每筆部位金額
        if position_mode == "fixed":
            per_pos = per_position_amount if per_position_amount > 0 else daily_budget
        else:
            per_pos = min(daily_budget / len(today_signals), daily_budget)

        # 4. 買進
        for sig in today_signals:
            cost = min(per_pos, port.cash)  # 不超過現金
            if cost < 1000:  # 太小不交易
                continue
            port.cash -= cost

            # exit_date 僅用於判斷順序，以 date 字串排序近似
            port.open_positions.append({
                "entry_date": date,
                "exit_date": _add_trading_days(date, hold_days, all_dates),
                "ticker": sig["ticker"],
                "name": sig["name"],
                "pass_level": sig["pass_level"],
                "pattern_type": sig["pattern_type"],
                "confidence": sig["prediction_confidence"],
                "close_price": sig["close_price"],
                "cost": cost,
                "return_1d": sig["return_1d"],
                "return_5d": sig["return_5d"],
                "twii_1d": sig["twii_1d"],
            })

        port.record_day(date)

    # 強制平倉所有剩餘部位（以最後可用回報計）
    for pos in port.open_positions:
        ret_pct = pos["return_1d"] if hold_days == 1 else pos["return_5d"]
        pnl = pos["cost"] * ret_pct / 100
        port.cash += pos["cost"] + pnl
        port.closed_trades.append({
            **pos,
            "exit_date": "final",
            "return_pct": ret_pct,
            "pnl": pnl,
            "exit_value": pos["cost"] + pnl,
        })
    port.open_positions = []

    return port


def _add_trading_days(date: str, n: int, all_dates: list[str]) -> str:
    """從 all_dates 中往後找第 n 個交易日。"""
    if date not in all_dates:
        return date
    idx = all_dates.index(date)
    target = idx + n
    if target < len(all_dates):
        return all_dates[target]
    return all_dates[-1]


# ──────────────────────────────────────────────
# 績效指標
# ──────────────────────────────────────────────

def compute_metrics(port: Portfolio) -> dict:
    trades = port.closed_trades
    if not trades:
        return {}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # 最大回撤
    values = [d["portfolio_value"] for d in port.daily_values]
    max_dd = 0.0
    peak = port.initial_capital
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # 夏普（簡化：日報酬）
    daily_rets = []
    prev_v = port.initial_capital
    for d in port.daily_values:
        v = d["portfolio_value"]
        if prev_v > 0:
            daily_rets.append((v - prev_v) / prev_v)
        prev_v = v
    if len(daily_rets) > 1:
        mean_r = sum(daily_rets) / len(daily_rets)
        var_r = sum((r - mean_r) ** 2 for r in daily_rets) / len(daily_rets)
        std_r = math.sqrt(var_r)
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0
    else:
        sharpe = 0

    final_value = port.total_value()
    total_return = (final_value - port.initial_capital) / port.initial_capital * 100

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win_twd": avg_win,
        "avg_loss_twd": avg_loss,
        "total_pnl": total_pnl,
        "initial_capital": port.initial_capital,
        "final_value": final_value,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
    }


# ──────────────────────────────────────────────
# 報告輸出
# ──────────────────────────────────────────────

def print_report(port: Portfolio, metrics: dict, args: argparse.Namespace):
    sep = "=" * 60
    print(sep)
    print("  模擬交易績效報告")
    print(sep)
    print(f"  初始資金：    {metrics['initial_capital']:>12,.0f} TWD")
    print(f"  最終資產：    {metrics['final_value']:>12,.0f} TWD")
    print(f"  總報酬率：    {metrics['total_return_pct']:>+11.2f}%")
    print(f"  最大回撤：    {metrics['max_drawdown_pct']:>11.2f}%")
    print(f"  Sharpe：      {metrics['sharpe']:>11.2f}")
    print()
    print(f"  交易次數：    {metrics['total_trades']:>12}")
    print(f"  勝率：        {metrics['win_rate']*100:>11.1f}%")
    print(f"  獲利因子：    {metrics['profit_factor']:>11.2f}")
    print(f"  平均獲利：    {metrics['avg_win_twd']:>+12,.0f} TWD")
    print(f"  平均虧損：    {metrics['avg_loss_twd']:>+12,.0f} TWD")
    print()
    print(f"  策略設定")
    print(f"    每日預算：  {args.daily_budget:>12,.0f} TWD")
    print(f"    持有天數：  {args.hold_days:>12} 日")
    print(f"    訊號過濾：  {args.signal_filter:>12}")
    print(f"    方向過濾：  {args.direction_filter:>12}")
    print(f"    信心門檻：  {args.min_confidence:>11.2f}")
    print(f"    最大持倉：  {args.max_positions:>12} 檔")
    print(sep)

    # 每日淨值
    print("\n  每日資產變化（取樣）：")
    daily = port.daily_values
    step = max(1, len(daily) // 15)
    for d in daily[::step]:
        bar_len = int((d["portfolio_value"] / metrics["initial_capital"] - 0.8) * 100)
        bar = "█" * max(0, bar_len)
        print(f"  {d['date']}  {d['portfolio_value']:>10,.0f}  {bar}")

    # Top 5 最賺
    print("\n  前 5 筆最賺：")
    top = sorted(port.closed_trades, key=lambda x: -x["pnl"])[:5]
    for t in top:
        print(f"    {t['entry_date']} {t['ticker']} {t['name'][:6]:<6}  "
              f"ret={t['return_pct']:+.2f}%  pnl={t['pnl']:+,.0f}")

    # Top 5 最賠
    print("\n  前 5 筆最賠：")
    bot = sorted(port.closed_trades, key=lambda x: x["pnl"])[:5]
    for t in bot:
        print(f"    {t['entry_date']} {t['ticker']} {t['name'][:6]:<6}  "
              f"ret={t['return_pct']:+.2f}%  pnl={t['pnl']:+,.0f}")

    # 寫交易紀錄 CSV
    out_csv = DATA_DIR / "portfolio_sim_trades.csv"
    fieldnames = ["entry_date", "exit_date", "ticker", "name", "pass_level",
                  "pattern_type", "confidence", "close_price", "cost",
                  "return_pct", "pnl", "exit_value"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for t in sorted(port.closed_trades, key=lambda x: x["entry_date"]):
            w.writerow(t)
    print(f"\n  交易明細已存：{out_csv}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LLM 訊號模擬交易回測")
    p.add_argument("--initial-capital", type=float, default=500_000,
                   help="初始資金（預設 500,000 TWD）")
    p.add_argument("--daily-budget", type=float, default=50_000,
                   help="每日最多投入金額（預設 50,000 TWD）")
    p.add_argument("--hold-days", type=int, default=1, choices=[1, 5],
                   help="持有天數：1 或 5（預設 1）")
    p.add_argument("--max-positions", type=int, default=10,
                   help="最大同時持倉檔數（預設 10）")
    p.add_argument("--signal-filter", default="pass",
                   choices=["pass", "pass-warn", "all"],
                   help="訊號過濾：pass / pass-warn / all")
    p.add_argument("--direction-filter", default="up",
                   choices=["up", "all"],
                   help="方向過濾：只買 up 預測 / 全部")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="最低 LLM 信心值（0~1，預設 0）")
    p.add_argument("--include-reject", action="store_true",
                   help="包含 LLM verdict=REJECT 的訊號")
    p.add_argument("--position-mode", default="equal",
                   choices=["equal", "fixed"],
                   help="部位分配：equal=均分；fixed=每筆固定 --daily-budget 元")
    return p.parse_args()


def main():
    args = parse_args()

    signals = load_signals(
        signal_filter=args.signal_filter,
        direction_filter=args.direction_filter,
        min_confidence=args.min_confidence,
        include_reject=args.include_reject,
    )
    print(f"篩選後訊號：{len(signals)} 筆")
    if not signals:
        print("無符合條件的訊號，請放寬過濾條件。")
        return

    port = simulate(
        signals=signals,
        initial_capital=args.initial_capital,
        daily_budget=args.daily_budget,
        hold_days=args.hold_days,
        max_positions=args.max_positions,
        position_mode=args.position_mode,
        per_position_amount=args.daily_budget,
    )

    metrics = compute_metrics(port)
    print_report(port, metrics, args)


if __name__ == "__main__":
    main()
