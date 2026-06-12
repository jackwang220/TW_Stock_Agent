"""回測評估指標（移植自 shang-che/stock-analysis）。"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def monthly_win_rate(monthly_returns: np.ndarray) -> float:
    r = np.asarray(monthly_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def direction_accuracy(returns: np.ndarray) -> float:
    """多少比例的訊號方向正確（return > 0）。"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    total = np.prod(1 + r)
    if total <= 0:
        return float("nan")
    return float(total ** (periods_per_year / len(r)) - 1)


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    downside = r[r < 0]
    if len(downside) == 0:
        return float("inf")
    mean_ret = r.mean()
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std == 0:
        return float("inf")
    return float(np.sqrt(periods_per_year) * mean_ret / downside_std)


def print_summary(returns: np.ndarray, label: str = "Strategy") -> None:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  訊號數：     {len(r)}")
    print(f"  方向準確率： {direction_accuracy(r):.1%}")
    print(f"  平均報酬：   {r.mean()*100:.2f}%")
    print(f"  Sharpe：     {sharpe_ratio(r):.2f}")
    print(f"  Sortino：    {sortino_ratio(r):.2f}")
    print(f"  最大回撤：   {max_drawdown(r)*100:.1f}%")
    print(f"{'='*40}\n")
