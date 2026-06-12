"""規則回測：對歷史 6 個月跑 Node 4-5 量化規則（無 LLM）。

使用方式：
    uv run python scripts/backtest_screener.py
    uv run python scripts/backtest_screener.py --months 3 --tickers 2059,3324,3017

輸出：data/backtest_rules.csv + 終端機指標摘要
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from loguru import logger

from tw_stock_agent.backtest.metrics import print_summary
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.data.price_data import get_ohlcv, validate_price_data
from tw_stock_agent.screener.patterns import check_volume_breakout

OUTPUT = DATA_DIR / "backtest_rules.csv"

# 預設回測標的（供應鏈核心股）
DEFAULT_TICKERS = [
    {"code": "2059", "name": "川湖科技", "yf_ticker": "2059.TW"},
    {"code": "3324", "name": "雙鴻工程", "yf_ticker": "3324.TW"},
    {"code": "3017", "name": "奇鋐科技", "yf_ticker": "3017.TW"},
    {"code": "3533", "name": "嘉澤端子", "yf_ticker": "3533.TW"},
    {"code": "6125", "name": "廣運機械", "yf_ticker": "6125.TW"},
    {"code": "3653", "name": "健策精密", "yf_ticker": "3653.TW"},
    {"code": "3037", "name": "欣興電子", "yf_ticker": "3037.TW"},
    {"code": "8046", "name": "南亞電路板", "yf_ticker": "8046.TWO"},
    {"code": "5274", "name": "信驊科技", "yf_ticker": "5274.TW"},
    {"code": "6669", "name": "緯穎科技", "yf_ticker": "6669.TW"},
    {"code": "2330", "name": "台積電",   "yf_ticker": "2330.TW"},
]

FIELDS = [
    "date", "ticker", "name", "rule",
    "close_price", "volume_ratio", "rs_20d",
    "return_5d", "return_10d", "return_20d",
]


def run_backtest(
    tickers: list[dict],
    start: date,
    end: date,
    output: Path = OUTPUT,
) -> list[dict]:
    rows: list[dict] = []
    current = start

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        as_of = current.isoformat()
        for t in tickers:
            df = get_ohlcv(t["yf_ticker"], as_of_date=as_of)
            if df.empty or len(df) < 60:
                current += timedelta(days=1)
                continue

            ok, _ = validate_price_data(df, current)
            if not ok:
                current += timedelta(days=1)
                continue

            fired, _ = check_volume_breakout(df)
            if not fired:
                current += timedelta(days=1)
                continue

            close = float(df["Close"].iloc[-1])

            # 事後回填報酬（使用全量資料）
            df_full = get_ohlcv(t["yf_ticker"])
            ret: dict[str, str] = {}
            for days, col in [(5, "return_5d"), (10, "return_10d"), (20, "return_20d")]:
                try:
                    idx = df_full.index.searchsorted(str(current))
                    if idx + days < len(df_full):
                        entry = float(df_full["Close"].iloc[idx])
                        exit_p = float(df_full["Close"].iloc[idx + days])
                        ret[col] = f"{(exit_p / entry - 1) * 100:.2f}"
                    else:
                        ret[col] = ""
                except Exception:
                    ret[col] = ""

            rows.append({
                "date": as_of,
                "ticker": t["code"],
                "name": t["name"],
                "rule": "volume_breakout",
                "close_price": f"{close:.2f}",
                "volume_ratio": "",
                **ret,
            })
            logger.info(f"  SIGNAL {as_of} {t['code']} {t['name']}  10d={ret.get('return_10d','?')}%")

        current += timedelta(days=1)

    # 存 CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest quantitative screener rules")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--tickers", type=str, default="",
                        help="逗號分隔的股票代號，例如 2059,3324（不填則用預設清單）")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    end = date.today() - timedelta(days=10)
    start = end - timedelta(days=args.months * 30)

    tickers = DEFAULT_TICKERS
    if args.tickers:
        codes = [c.strip() for c in args.tickers.split(",")]
        tickers = [t for t in DEFAULT_TICKERS if t["code"] in codes]
        if not tickers:
            logger.error(f"No matching tickers for: {args.tickers}")
            sys.exit(1)

    logger.info(f"Backtesting {len(tickers)} tickers from {start} to {end}")
    rows = run_backtest(tickers, start, end)

    if not rows:
        logger.warning("No signals fired in this period.")
        return

    # ── 指標 ──────────────────────────────────────────────────
    ret10 = np.array([float(r["return_10d"]) / 100
                      for r in rows if r.get("return_10d")])
    print_summary(ret10, label="volume_breakout rule (10-day return)")
    logger.success(f"Results saved → {OUTPUT}")


if __name__ == "__main__":
    main()
