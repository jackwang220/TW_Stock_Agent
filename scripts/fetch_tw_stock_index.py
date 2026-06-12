"""抓取 TWSE + TPEX 上市櫃公司名單，產生 data/tw_stock_index.json。

主來源：FinMind TaiwanStockInfo（opendata.twse.com.tw 從本機無法連線，已棄用）。
格式：{"2059": {"name": "川湖科技", "aliases": ["川湖"], "yf_ticker": "2059.TW",
              "market": "TWSE", "industry": "..."}, ...}

只收「4 碼普通股」（排除 ETF 00xx、權證 6 碼、興櫃 emerging、特別股）。

使用方式：
    python scripts/fetch_tw_stock_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tw_stock_agent.config import TW_STOCK_INDEX, get_settings

# ── 手動別名對照（補強新聞比對；FinMind 已是短名，多數情況短名即可命中）──────────
ALIASES: dict[str, list[str]] = {
    "台積電": ["TSMC", "台積"],
    "聯發科": ["MTK"],
    "鴻海": ["FOXCONN"],
    "日月光投控": ["日月光", "ASE"],
    "聯華電子": ["聯電", "UMC"],
    "瑞昱": ["Realtek"],
    "南亞科技": ["南亞科"],
    "南亞電路板": ["南電"],
}

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def _headers() -> dict:
    tok = get_settings().finmind_token
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def fetch_finmind_stockinfo() -> list[dict]:
    """FinMind TaiwanStockInfo：全上市櫃 + 興櫃清單。"""
    r = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInfo"},
                     headers=_headers(), timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != 200:
        raise RuntimeError(f"FinMind TaiwanStockInfo failed: {payload.get('msg')}")
    return payload.get("data", [])


def _is_common_stock(code: str) -> bool:
    """只收 4 碼數字普通股（排除 ETF 0050/00xxx、權證 6 碼、TDR、特別股 等）。"""
    return code.isdigit() and len(code) == 4 and not code.startswith("0")


def build_index() -> dict[str, dict]:
    print("Fetching FinMind TaiwanStockInfo ...")
    data = fetch_finmind_stockinfo()
    print(f"  raw rows: {len(data)}")

    index: dict[str, dict] = {}
    skipped_emerging = skipped_noncommon = 0
    for it in data:
        code = (it.get("stock_id") or "").strip()
        typ = (it.get("type") or "").strip()          # twse / tpex / emerging
        name = (it.get("stock_name") or "").strip().rstrip("*").strip()
        if typ not in ("twse", "tpex"):
            skipped_emerging += 1
            continue
        if not _is_common_stock(code):
            skipped_noncommon += 1
            continue
        if not name:
            continue
        market = "TWSE" if typ == "twse" else "TPEX"
        suffix = ".TW" if typ == "twse" else ".TWO"
        # 去重：同 stock_id 可能多列（改名歷史），保留最後一筆（最新名稱）
        index[code] = {
            "name": name,
            "aliases": ALIASES.get(name, []),
            "yf_ticker": f"{code}{suffix}",
            "market": market,
            "industry": (it.get("industry_category") or "").strip(),
        }

    n_twse = sum(1 for v in index.values() if v["market"] == "TWSE")
    n_tpex = sum(1 for v in index.values() if v["market"] == "TPEX")
    print(f"  skipped: emerging/other={skipped_emerging}, non-common(ETF/權證/特別股)={skipped_noncommon}")
    print(f"  kept common stocks: TWSE={n_twse}, TPEX={n_tpex}, total={len(index)}")
    return index


def main() -> None:
    TW_STOCK_INDEX.parent.mkdir(parents=True, exist_ok=True)
    index = build_index()
    TW_STOCK_INDEX.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(index)} companies -> {TW_STOCK_INDEX}")
    # 驗證重要代號（含先前缺失的上市大型股）
    for code in ["2330", "2327", "3008", "2408", "2059", "3711", "2454", "8299"]:
        info = index.get(code)
        print(f"  {code}: {info['name'] + ' / ' + info['market'] if info else 'NOT FOUND'}")


if __name__ == "__main__":
    main()