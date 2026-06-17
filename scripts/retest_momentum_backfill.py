"""Backfill OHLCV cache to 2021 for the full 112-stock universe + 0050.
一次性、限流內分批 (FinMind ~300/hr; 我們約 113*2≈226 calls)。force_refresh 重抓 2021-01-01 起全史。
跑完後 retest_momentum_beta3.py 才能在 held-out 2021/2022/2023 做全universe IC。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
codes = list(u.keys()) + ["0050"]
print(f"backfilling {len(codes)} codes to 2021-01-01 ...", flush=True)
ok = bad = skipped = 0
for i, c in enumerate(codes, 1):
    # only refresh if cache doesn't already reach 2021
    try:
        cur = get_daily_ohlcv(c, start="2021-01-01")
        ds = sorted(cur)
        if ds and ds[0] <= "2021-06-30":
            skipped += 1
            continue
    except Exception:
        pass
    try:
        d = get_daily_ohlcv(c, start="2021-01-01", force_refresh=True)
        ds = sorted(d)
        if ds and ds[0] <= "2021-12-31":
            ok += 1
        else:
            bad += 1
            print(f"  [{i}] {c}: only {ds[0] if ds else 'EMPTY'} (no 2021)", flush=True)
        time.sleep(0.5)
    except Exception as e:
        bad += 1
        print(f"  [{i}] {c}: ERR {e}", flush=True)
        time.sleep(1.0)
    if i % 20 == 0:
        print(f"  ...{i}/{len(codes)} (ok={ok} bad={bad} skip={skipped})", flush=True)
print(f"DONE ok={ok} bad={bad} skipped={skipped}", flush=True)
