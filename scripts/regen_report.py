"""用統一系統 point-in-time 重生某日報告(只用 ≤該日的資料/新聞),取代舊報告。

用法:
    python scripts/regen_report.py 2026-06-09

= run_daily_scan(as_of=日期):候選股=當時流動性前N(base_universe)、新聞≤該日、
  技術/反彈/籌碼全 as_of、辯論用現在的邏輯。報告寫到 reports/<日期>.md(舊的先備份)。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger

from tw_stock_agent.config import REPORTS_DIR
from tw_stock_agent.pipeline.graph import run_daily_scan


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python scripts/regen_report.py 2026-06-09")
        sys.exit(1)
    as_of = sys.argv[1]

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    # 備份舊報告
    old = REPORTS_DIR / f"{as_of}.md"
    if old.exists():
        bak = REPORTS_DIR / f"{as_of}.OLD.md"
        shutil.copy2(old, bak)
        logger.info(f"舊報告已備份 → {bak.name}")

    logger.info(f"=== 統一系統 point-in-time 重生 {as_of}(只用 ≤{as_of} 的資料)===")
    path = run_daily_scan(as_of=as_of)
    logger.info(f"完成 → {path}")


if __name__ == "__main__":
    main()