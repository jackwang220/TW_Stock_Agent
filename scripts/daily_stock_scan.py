"""每日股票掃描入口。

使用方式：
    uv run python scripts/daily_stock_scan.py
    uv run python scripts/daily_stock_scan.py --backfill   # 回填歷史報酬
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from tw_stock_agent.pipeline.graph import run_daily_scan
from tw_stock_agent.signal_log import backfill_returns


def main() -> None:
    parser = argparse.ArgumentParser(description="TW Stock Agent daily scan")
    parser.add_argument("--backfill", action="store_true", help="只跑報酬回填，不跑掃描")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    if args.backfill:
        logger.info("Running return backfill on signal_log.csv...")
        updated = backfill_returns()
        logger.info(f"Backfilled {updated} cells.")
        return

    logger.info("Starting daily stock scan...")
    report_path = run_daily_scan()
    if report_path:
        logger.success(f"Report saved: {report_path}")
        # 順便跑回填
        updated = backfill_returns()
        if updated:
            logger.info(f"Also backfilled {updated} return cells in signal_log.")
    else:
        logger.warning("No report generated (no candidates passed screening?)")


if __name__ == "__main__":
    main()
