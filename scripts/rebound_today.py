"""掃基本盤,列出『今天』(或指定日)觸發跌深反彈訊號的股票,依期望報酬排序。

用法:
    python scripts/rebound_today.py            # 最新
    python scripts/rebound_today.py 2025-04-07 # 指定日(回看驗證)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from loguru import logger
logger.remove()

from tw_stock_agent.config import DATA_DIR, TW_STOCK_INDEX
from tw_stock_agent.tools.rebound_signal import rebound_signal_for_code


def main():
    as_of = sys.argv[1] if len(sys.argv) > 1 else None
    base = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    names = json.loads(TW_STOCK_INDEX.read_text(encoding="utf-8"))

    hits = []
    for code, info in base.items():
        try:
            sig = rebound_signal_for_code(code, info.get("avg_turnover", 0), as_of=as_of)
        except Exception:
            continue
        if sig["fired"]:
            hits.append((code, info, sig))

    hits.sort(key=lambda x: -x[2]["rebound_edge"])
    print(f"\n{'='*72}")
    print(f"  跌深反彈訊號 — {as_of or '最新'}　共 {len(hits)} 檔(基本盤 {len(base)} 檔中)")
    print(f"{'='*72}")
    if not hits:
        print("  今日無觸發(沒有大型股跌深)。")
        return
    print(f"  {'代號':<6}{'名稱':<9}{'層級':<7}{'劑量':<14}{'勝率':<6}{'期望淨5日':<9}{'信心'}")
    print("  " + "-" * 66)
    for code, info, s in hits:
        nm = names.get(code, {}).get("name", code)[:5]
        print(f"  {code:<6}{nm:<9}{s['tier']:<7}{s['depth']:<14}{s['win_rate']*100:.0f}%   "
              f"{s['rebound_edge']*100:+.1f}%     {s['score']:.2f}")


if __name__ == "__main__":
    main()