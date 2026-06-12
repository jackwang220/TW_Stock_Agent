"""從新聞文字中精確比對台股 ticker（stock code + 公司名 + 別名）。

Strategy（借鑑 shang-che/stock-analysis）：
- 股票代號（數字）：word boundary regex，避免 2330 match 到 23300
- 公司名稱 / 別名：substring match
- LLM 只處理 ticker_match 沒命中但標題有動能詞/產業詞的文章
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tw_stock_agent.config import TW_STOCK_INDEX


def load_ticker_book(path: Path | None = None) -> dict[str, dict]:
    """載入 tw_stock_index.json。

    格式：{"2059": {"name": "川湖科技", "aliases": ["川湖"], "yf_ticker": "2059.TW"}, ...}
    """
    p = path or TW_STOCK_INDEX
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def match_tickers(text: str, ticker_book: dict[str, dict]) -> list[str]:
    """從 text 找出所有能確信比對到的 ticker codes。

    Returns:
        List of ticker codes (e.g. ["2059", "3324"]), deduplicated, insertion-order.
    """
    matched: list[str] = []
    for ticker, info in ticker_book.items():
        # 股票代號：word boundary（\b2059\b 不會 match 23300）
        if re.search(rf"\b{re.escape(ticker)}\b", text):
            if ticker not in matched:
                matched.append(ticker)
            continue
        # 公司名稱 + 別名：substring
        candidates = [info.get("name", "")] + list(info.get("aliases", []))
        for cand in candidates:
            if cand and cand in text:
                if ticker not in matched:
                    matched.append(ticker)
                break
    return matched


def enrich_candidates(tickers: list[str], ticker_book: dict[str, dict]) -> list[dict]:
    """把 ticker code list 轉成帶有 name / yf_ticker 的完整 dict。"""
    results = []
    for t in tickers:
        info = ticker_book.get(t, {})
        results.append({
            "code": t,
            "name": info.get("name", t),
            "yf_ticker": info.get("yf_ticker", f"{t}.TW"),
            "market": info.get("market", "TWSE"),
            "aliases": info.get("aliases", []),
        })
    return results
