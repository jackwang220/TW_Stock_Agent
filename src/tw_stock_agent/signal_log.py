"""訊號記錄 + 報酬回填：data/signal_log.csv。"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

from tw_stock_agent.config import SIGNAL_LOG
from tw_stock_agent.data.price_data import get_ohlcv

FIELDS = [
    "date", "ticker", "name", "source_news", "bfs_depth",
    "volume_ratio", "ma5_gt_ma20", "rs_20d",
    "pattern_type", "rsi_14",
    "bear_score", "bull_score", "evidence_level", "verdict",
    "close_price",
    # 隔日漲跌預測
    "predicted_direction", "predicted_low_pct", "predicted_high_pct",
    "predicted_center_pct", "prediction_confidence",
    # 實際報酬（回填）
    "return_1d", "return_5d", "return_10d", "return_20d", "max_drawdown_10d",
]


def append_signals(debated: list[dict], today: str) -> None:
    """把當天 debated 清單寫入 signal_log.csv（報酬欄留空，等回填）。"""
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SIGNAL_LOG.exists()
    with SIGNAL_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for s in debated:
            writer.writerow({
                "date": today,
                "ticker": s.get("code", ""),
                "name": s.get("name", ""),
                "source_news": s.get("source_news", "")[:80],
                "bfs_depth": s.get("bfs_depth", 0),
                "volume_ratio": f"{s.get('volume_ratio', 0):.3f}",
                "ma5_gt_ma20": s.get("ma5_gt_ma20", False),
                "rs_20d": f"{s.get('rs_20d', 1.0):.3f}",
                "pattern_type": s.get("pattern_type", ""),
                "rsi_14": f"{s.get('rsi_14', 0):.1f}",
                "bear_score": s.get("bear_score", ""),
                "bull_score": s.get("bull_score", ""),
                "evidence_level": s.get("evidence_level", ""),
                "verdict": s.get("verdict", ""),
                "close_price": f"{s.get('close_price', 0):.2f}",
                "predicted_direction": s.get("predicted_direction", "neutral"),
                "predicted_low_pct": f"{s.get('predicted_low_pct', 0.0):.2f}",
                "predicted_high_pct": f"{s.get('predicted_high_pct', 0.0):.2f}",
                "predicted_center_pct": f"{s.get('predicted_center_pct', 0.0):.2f}",
                "prediction_confidence": f"{s.get('prediction_confidence', 0.0):.2f}",
                "return_1d": "",
                "return_5d": "", "return_10d": "", "return_20d": "", "max_drawdown_10d": "",
            })


def remove_signals(target_date: str) -> int:
    """刪掉 signal_log.csv 中某日的所有紀錄（as_of 重生前先清掉舊的，避免重複）。

    Returns:
        刪掉的列數
    """
    if not SIGNAL_LOG.exists():
        return 0
    with SIGNAL_LOG.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r.get("date") != target_date]
    removed = len(rows) - len(kept)
    if removed:
        with SIGNAL_LOG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)
    return removed


def backfill_returns(as_of: date | None = None) -> int:
    """回填 signal_log.csv 中已到期的報酬欄位。

    Returns:
        更新的列數
    """
    if not SIGNAL_LOG.exists():
        return 0
    if as_of is None:
        as_of = date.today()

    rows = []
    updated = 0
    with SIGNAL_LOG.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    for row in rows:
        signal_date = date.fromisoformat(row["date"])
        # 只回填報酬欄有空值的列
        needs_update = any(row.get(f, "") == "" for f in ["return_1d", "return_5d", "return_10d", "return_20d"])
        if not needs_update:
            continue
        ticker = row["ticker"]
        yf_ticker = f"{ticker}.TW"

        for days, col in [(1, "return_1d"), (5, "return_5d"), (10, "return_10d"), (20, "return_20d")]:
            if row.get(col, "") != "":
                continue
            target_date = signal_date + timedelta(days=days + 3)  # 加 buffer（含週末）
            if target_date > as_of:
                continue  # 還沒到
            df = get_ohlcv(yf_ticker)
            if df.empty or len(df) < 5:
                continue
            # 找 signal_date 的 idx
            try:
                idx = df.index.searchsorted(str(signal_date))
                if idx >= len(df) - days:
                    continue
                entry = df["Close"].iloc[idx]
                exit_p = df["Close"].iloc[idx + days]
                row[col] = f"{(exit_p / entry - 1) * 100:.2f}"
                updated += 1
            except Exception:
                pass

        # max_drawdown_10d
        if row.get("max_drawdown_10d", "") == "":
            target_date = signal_date + timedelta(days=12)
            if target_date <= as_of:
                try:
                    df = get_ohlcv(yf_ticker)
                    idx = df.index.searchsorted(str(signal_date))
                    window = df["Close"].iloc[idx:idx + 10]
                    if len(window) >= 5:
                        entry = window.iloc[0]
                        dd = (window.min() / entry - 1) * 100
                        row["max_drawdown_10d"] = f"{dd:.2f}"
                        updated += 1
                except Exception:
                    pass

    # 重寫
    with SIGNAL_LOG.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return updated


def load_signal_log() -> list[dict]:
    """載入 signal_log.csv 為 list of dict。"""
    if not SIGNAL_LOG.exists():
        return []
    with SIGNAL_LOG.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
