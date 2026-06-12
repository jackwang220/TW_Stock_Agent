"""從 backtest_one_month.csv 建立每支股票的「歷史型態表現資料庫」。

使用方式：
    uv run python scripts/build_stock_profile.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from tw_stock_agent.config import DATA_DIR

DB_PATH = DATA_DIR / "stock_profile.db"
QUANT_CSV = DATA_DIR / "backtest_one_month.csv"


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pattern_stats (
            ticker      TEXT,
            name        TEXT,
            pattern_type TEXT,
            n           INTEGER,
            avg_return  REAL,
            std_return  REAL,
            win_rate    REAL,
            avg_alpha   REAL,
            last_date   TEXT,
            PRIMARY KEY (ticker, pattern_type)
        );
        CREATE TABLE IF NOT EXISTS ticker_overview (
            ticker          TEXT PRIMARY KEY,
            name            TEXT,
            total_signals   INTEGER,
            overall_win_rate REAL,
            avg_return      REAL,
            avg_alpha       REAL,
            last_updated    TEXT
        );
    """)
    conn.commit()


def build_profile() -> None:
    if not QUANT_CSV.exists():
        logger.error(f"找不到 {QUANT_CSV}，請先執行 backtest_one_month.py")
        sys.exit(1)

    rows = []
    with QUANT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("return_1d"):
                continue
            rows.append(r)

    logger.info(f"載入 {len(rows)} 筆有回報資料的訊號")

    # 分組：(ticker, pattern_type)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    names: dict[str, str] = {}

    for r in rows:
        ticker = r["ticker"]
        pattern = r.get("pattern_type") or "none"
        names[ticker] = r.get("name", ticker)
        try:
            ret = float(r["return_1d"])
        except ValueError:
            continue
        alpha = None
        if r.get("alpha_1d"):
            try:
                alpha = float(r["alpha_1d"])
            except ValueError:
                pass
        groups[(ticker, pattern)].append({
            "return_1d": ret,
            "alpha_1d": alpha,
            "date": r["date"],
        })

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _create_tables(conn)
    cur = conn.cursor()

    # pattern_stats
    ticker_all: dict[str, list] = defaultdict(list)
    ticker_alpha: dict[str, list] = defaultdict(list)
    ticker_dates: dict[str, list] = defaultdict(list)

    for (ticker, pattern), items in groups.items():
        returns = [i["return_1d"] for i in items]
        alphas = [i["alpha_1d"] for i in items if i["alpha_1d"] is not None]
        n = len(returns)
        avg_ret = mean(returns)
        std_ret = stdev(returns) if n > 1 else 0.0
        win_rate = sum(1 for r in returns if r > 0.2) / n
        avg_alpha = mean(alphas) if alphas else None
        last_date = max(i["date"] for i in items)

        cur.execute("""
            INSERT OR REPLACE INTO pattern_stats
            (ticker, name, pattern_type, n, avg_return, std_return, win_rate, avg_alpha, last_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, names.get(ticker, ""), pattern, n, avg_ret, std_ret, win_rate, avg_alpha, last_date))

        ticker_all[ticker].extend(returns)
        ticker_alpha[ticker].extend(alphas)
        ticker_dates[ticker].append(last_date)

    # ticker_overview
    for ticker, returns in ticker_all.items():
        n = len(returns)
        avg_ret = mean(returns)
        win_rate = sum(1 for r in returns if r > 0.2) / n
        alphas = ticker_alpha[ticker]
        avg_alpha = mean(alphas) if alphas else None
        last_updated = max(ticker_dates[ticker])

        cur.execute("""
            INSERT OR REPLACE INTO ticker_overview
            (ticker, name, total_signals, overall_win_rate, avg_return, avg_alpha, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, names.get(ticker, ""), n, win_rate, avg_ret, avg_alpha, last_updated))

    conn.commit()
    conn.close()

    n_tickers = len(ticker_all)
    n_patterns = sum(len(set(p for (t, p) in groups if t == tk)) for tk in ticker_all)
    logger.success(f"完成：{n_tickers} 支股票，{len(groups)} 條型態統計 → {DB_PATH}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    build_profile()
