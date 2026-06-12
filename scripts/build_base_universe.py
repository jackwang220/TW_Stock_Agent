"""建立「基本盤」清單 data/base_universe.json:每天強制量化掃描的液性高股票。

基本盤 ≈ 0050(上市大型) + 0051(上市中型) + 富櫃50/006201(上櫃大型) + 00947(IC設計),
但不爬不穩定的 ETF 頁面,改用「平均成交額排名」複製(這些 ETF 本來就是市值/流動性排名)。

兩個模式：
  python scripts/build_base_universe.py            # B:用現有快取 + CORE 大型股,立即可用
  python scripts/build_base_universe.py --fetch-all # A:先抓完全市場(撞限流自動睡),再算完整排名

說明：所有 CORE 代號都會對權威字典(tw_stock_index.json)驗證存在、名稱從字典拉,
      不存在的會被跳過並回報,杜絕捏造代號。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from tw_stock_agent.config import TW_STOCK_INDEX, DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

CACHE = DATA_DIR / "finmind_cache"
OUT = DATA_DIR / "base_universe.json"

TWSE_TOP = 150     # ≈ 0050 + 0051
TPEX_TOP = 50      # ≈ 富櫃50 (006201)
LIQ_MIN = 5e7      # 基本盤最低門檻:日均成交 ≥ 5000 萬

# 知名大型股(僅代號;名稱一律從權威字典拉、逐一驗證存在 → 不可能捏造)
CORE = [
    # 電子權值
    "2330", "2317", "2454", "2308", "2382", "2303", "3711", "2357", "2327", "3008",
    "2379", "2345", "3034", "2376", "2395", "3017", "2474", "2059", "6669", "3231",
    "2356", "2409", "3481", "2344", "2337", "2408", "5274", "6488", "8046", "3037",
    "3443", "2301", "2347", "2360", "4938", "2377", "3653", "3324", "6446",
    # 金融
    "2891", "2881", "2882", "2884", "2886", "2885", "2892", "2880", "2890", "2887",
    "2883", "5880", "5871", "2812",
    # 傳產 / 電信 / 航運
    "2412", "4904", "3045", "2603", "2615", "2609", "1303", "1301", "1326", "2002",
    "1216", "2207", "2105", "9910", "2912", "6505", "1101", "1102",
]


def avg_turnover(code: str) -> float | None:
    """從 FinMind 快取算近 ~120 交易日平均成交額(Trading_money,台幣)。"""
    f = CACHE / f"TaiwanStockPrice_{code}.json"
    if not f.exists():
        return None
    try:
        rows = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    amts = [float(r.get("Trading_money") or 0) for r in rows[-120:]]
    amts = [a for a in amts if a > 0]
    return mean(amts) if len(amts) >= 20 else None


def fetch_all(codes: list[str]) -> None:
    """A 模式:抓完全市場 OHLCV;撞 FinMind 限流(連續失敗)自動睡到重置再續。"""
    n = len(codes)
    fail = 0
    for i, c in enumerate(codes, 1):
        if (CACHE / f"TaiwanStockPrice_{c}.json").exists():
            continue
        d = get_daily_ohlcv(c)
        if not d:
            fail += 1
            if fail >= 20:        # 連續 20 檔空 → 八成撞限流,睡 1 小時再續
                print(f"  [{i}/{n}] 疑似限流,睡 3600s 等重置...", flush=True)
                time.sleep(3600)
                fail = 0
        else:
            fail = 0
        if i % 100 == 0:
            print(f"  fetch {i}/{n}", flush=True)
        time.sleep(0.1)


def build(fetch: bool) -> None:
    idx = json.loads(TW_STOCK_INDEX.read_text(encoding="utf-8"))
    if fetch:
        print(f"A 模式:抓全市場 {len(idx)} 檔(撞限流會自動睡)...", flush=True)
        fetch_all(list(idx))

    # 算流動性排名
    turnover = {c: avg_turnover(c) for c in idx}
    turnover = {k: v for k, v in turnover.items() if v}
    print(f"有成交額資料(已快取)的: {len(turnover)} / {len(idx)} 檔", flush=True)

    twse = sorted([c for c in turnover if idx[c]["market"] == "TWSE"],
                  key=lambda c: -turnover[c])[:TWSE_TOP]
    tpex = sorted([c for c in turnover if idx[c]["market"] == "TPEX"],
                  key=lambda c: -turnover[c])[:TPEX_TOP]
    # 只保留達流動性門檻的
    twse = [c for c in twse if turnover[c] >= LIQ_MIN]
    tpex = [c for c in tpex if turnover[c] >= LIQ_MIN]

    core_ok = [c for c in CORE if c in idx]
    core_missing = [c for c in CORE if c not in idx]

    codes = set(twse) | set(tpex) | set(core_ok)
    base = {}
    for c in sorted(codes):
        info = idx[c]
        src = []
        if c in twse:
            src.append(f"TWSE-top{TWSE_TOP}")
        if c in tpex:
            src.append(f"TPEX-top{TPEX_TOP}")
        if c in core_ok:
            src.append("core")
        base[c] = {
            "name": info["name"], "market": info["market"],
            "industry": info.get("industry", ""),
            "avg_turnover": round(turnover.get(c, 0)),
            "source": "+".join(src),
        }

    OUT.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    n_twse = sum(1 for v in base.values() if v["market"] == "TWSE")
    n_tpex = sum(1 for v in base.values() if v["market"] == "TPEX")
    print(f"\n基本盤建立完成 → {OUT}")
    print(f"  共 {len(base)} 檔（TWSE {n_twse} / TPEX {n_tpex}）")
    print(f"  來源:流動性排名 TWSE-top{TWSE_TOP}={len(twse)}, TPEX-top{TPEX_TOP}={len(tpex)}, CORE={len(core_ok)}")
    if core_missing:
        print(f"  ⚠️ CORE 中字典找不到(已跳過,不捏造): {core_missing}")
    # 抽樣顯示前 15 大
    top = sorted(base, key=lambda c: -base[c]["avg_turnover"])[:15]
    print("  成交額前 15:")
    for c in top:
        b = base[c]
        print(f"    {c} {b['name'][:6]:8s} {b['market']} 日均成交{b['avg_turnover']/1e8:.1f}億 [{b['source']}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-all", action="store_true", help="A 模式:先抓完全市場再排名")
    args = ap.parse_args()
    build(fetch=args.fetch_all)


if __name__ == "__main__":
    main()