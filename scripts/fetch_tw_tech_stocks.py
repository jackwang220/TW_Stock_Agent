"""抓取台灣上市櫃電子/科技類股清單，存到 data/tw_tech_stocks.json。

涵蓋 TWSE + TPEX 以下產業別：
  半導體業、電子業、光電業、電腦及週邊設備業、通信網路業、
  電子零組件業、電子通路業、資訊服務業、其他電子業

使用方式：
    uv run python scripts/fetch_tw_tech_stocks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tw_stock_agent.config import DATA_DIR

OUTPUT = DATA_DIR / "tw_tech_stocks.json"

HEADERS = {"User-Agent": "Mozilla/5.0"}

TECH_SECTORS = {
    "半導體業",
    "電子業",
    "光電業",
    "電腦及週邊設備業",
    "通信網路業",
    "電子零組件業",
    "電子通路業",
    "資訊服務業",
    "其他電子業",
}


def fetch_twse() -> list[dict]:
    """抓 TWSE 上市公司清單（含產業別）。"""
    url = "https://opendata.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] TWSE fetch failed: {e}")
        return []


def fetch_tpex() -> list[dict]:
    """抓 TPEX 上櫃公司清單（含產業別）。"""
    url = "https://opendata.twse.com.tw/v1/opendata/t187ap04_L"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] TPEX (t187ap04_L) fetch failed: {e}, trying fallback...")

    # fallback: 從 TPEX openapi 抓（不一定有產業別，用代號範圍補）
    url2 = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
    try:
        r2 = requests.get(url2, headers=HEADERS, timeout=20)
        r2.raise_for_status()
        raw = r2.json()
        # 轉換成跟 TWSE 格式一致
        result = []
        for item in raw:
            code = (item.get("SecuritiesCompanyCode") or item.get("股票代號") or "").strip()
            name = (item.get("CompanyName") or item.get("公司名稱") or "").strip()
            if code and name:
                result.append({
                    "股票代號": code,
                    "公司簡稱": name,
                    "產業別": item.get("IndustryCategoryCode") or item.get("產業別") or "",
                    "_market": "TPEX",
                })
        return result
    except Exception as e2:
        print(f"  [WARN] TPEX fallback also failed: {e2}")
        return []


def _is_tech_code(code: str) -> bool:
    """當 API 沒有回傳產業別時，用代號範圍猜測是否為電子/科技類股。"""
    if not code.isdigit():
        return False
    n = int(code)
    # TPEX 電子/科技常見範圍
    return (3000 <= n <= 3999) or (4500 <= n <= 4999) or (5200 <= n <= 5500) or (6100 <= n <= 6999) or (8000 <= n <= 8299)


def _load_twse_from_index():
    """TWSE API 失敗時，從已下載的 tw_stock_index.json 補 TWSE 上市股票。

    tw_stock_index.json 的 market 欄位已正確標注 TWSE/TPEX，
    直接用 market=TWSE 篩選即可，不需代號範圍猜測。
    """
    idx_path = DATA_DIR / "tw_stock_index.json"
    if not idx_path.exists():
        return
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    for code, info in idx.items():
        if info.get("market") == "TWSE":
            yield code, {
                "name": info["name"],
                "sector": "電子業",
                "yf_ticker": info["yf_ticker"],
                "market": "TWSE",
            }


def build_tech_list() -> dict[str, dict]:
    index: dict[str, dict] = {}

    print("Fetching TWSE tech stocks...")
    twse = fetch_twse()
    twse_ok = False
    for item in twse:
        code = (item.get("股票代號") or "").strip()
        name = (item.get("公司簡稱") or "").strip()
        sector = (item.get("產業別") or "").strip()
        if not code or not code.isdigit() or not name:
            continue
        if sector in TECH_SECTORS:
            index[code] = {
                "name": name,
                "sector": sector,
                "yf_ticker": f"{code}.TW",
                "market": "TWSE",
            }
            twse_ok = True

    if not twse_ok:
        print("  TWSE API failed, loading from tw_stock_index.json fallback...")
        for code, info in _load_twse_from_index():
            index.setdefault(code, info)
    print(f"  TWSE tech stocks: {sum(1 for v in index.values() if v['market']=='TWSE')}")

    print("Fetching TPEX tech stocks...")
    tpex = fetch_tpex()
    added_tpex = 0
    for item in tpex:
        code = (item.get("股票代號") or "").strip()
        name = (item.get("公司簡稱") or "").strip()
        sector = (item.get("產業別") or "").strip()
        if not code or not code.isdigit() or not name or code in index:
            continue
        is_tech = (sector in TECH_SECTORS) or (not sector and _is_tech_code(code))
        if is_tech:
            index[code] = {
                "name": name,
                "sector": sector or "電子業",
                "yf_ticker": f"{code}.TWO",
                "market": "TPEX",
            }
            added_tpex += 1
    print(f"  TPEX tech stocks: {added_tpex}")

    return index


def main() -> None:
    index = build_tech_list()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    sectors = {}
    for v in index.values():
        s = v.get("sector", "unknown")
        sectors[s] = sectors.get(s, 0) + 1

    print(f"\n總計 {len(index)} 支台灣科技股 → {OUTPUT}")
    print("各產業別：")
    for s, n in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
