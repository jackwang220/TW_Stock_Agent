"""特殊市場狀態：漲跌停偵測 + 處置股 / 注意股查詢。

TWSE / TPEX 官方 API 為 JavaScript 渲染，無法靜態爬取。
改用 Google News RSS 搜尋「{code} 處置」，從新聞標題 regex 抓結束日期。
這樣每支股票都會被自動掃描，不需要人工告知。

使用方式：
    from tw_stock_agent.market_status import get_stock_market_status
    status = get_stock_market_status("3026", "禾伸堂", close=601.0, prev_close=626.0)
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date

logger = logging.getLogger(__name__)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_pub_date(pub: str) -> date | None:
    """從新聞 published 字串解析出日期（支援 ISO 與 RFC822 / 'DD Mon YYYY'）。"""
    if not pub:
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", pub)            # 2026-06-09
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})", pub)  # 09 Jun 2026
    if m:
        mon = _MONTHS.get(m[2].lower())
        if mon:
            try:
                return date(int(m[3]), mon, int(m[1]))
            except ValueError:
                return None
    return None

# ── 記憶體快取（30 分鐘，避免同一次 scan 重複打 Google News）────────────────
_cache: dict[str, object] = {}
_cache_ts: dict[str, float] = {}
_CACHE_TTL = 1800


def _get_cache(key: str):
    if key in _cache and time.time() - _cache_ts.get(key, 0) < _CACHE_TTL:
        return _cache[key]
    return None


def _set_cache(key: str, val) -> None:
    _cache[key] = val
    _cache_ts[key] = time.time()


# ── 處置日期解析 ──────────────────────────────────────────────────────────────

def _parse_disposal_end_date(title: str, today: date) -> date | None:
    """從新聞標題解析處置結束日期。

    支援格式：
      - MM/DD到MM/DD（起到訖）
      - 至6月11日 / 至06月11日
      - 至MM/DD
      - YYYY/MM/DD
    """
    year = today.year

    # 格式1：start到end，例如「05/26到06/11」
    m = re.search(r'(\d{1,2})/(\d{1,2})到(\d{1,2})/(\d{1,2})', title)
    if m:
        em, ed = int(m.group(3)), int(m.group(4))
        try:
            d = date(year, em, ed)
            # 如果結束日在今天之前超過 60 天，可能是跨年，嘗試 year+1
            if (today - d).days > 60:
                d = date(year + 1, em, ed)
            return d
        except ValueError:
            pass

    # 格式2：至6月11日
    m = re.search(r'至\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', title)
    if m:
        em, ed = int(m.group(1)), int(m.group(2))
        try:
            d = date(year, em, ed)
            if (today - d).days > 60:
                d = date(year + 1, em, ed)
            return d
        except ValueError:
            pass

    # 格式3：至MM/DD
    m = re.search(r'至\s*(\d{1,2})/(\d{1,2})', title)
    if m:
        em, ed = int(m.group(1)), int(m.group(2))
        try:
            d = date(year, em, ed)
            if (today - d).days > 60:
                d = date(year + 1, em, ed)
            return d
        except ValueError:
            pass

    # 格式4：YYYY/MM/DD（標題直接含年）
    m = re.search(r'(\d{4})/(\d{1,2})/(\d{1,2})', title)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    return None


# ── Google News 處置偵測（主力邏輯）─────────────────────────────────────────

def fetch_disposal_via_news(code: str, name_zh: str,
                            as_of: date | None = None) -> dict:
    """透過 Google News 搜尋判斷股票是否為處置股。

    TWSE/TPEX 官方 API 為 JS 渲染，改用 Google News RSS。
    搜尋「{code} 處置」與「{name} 處置股」，從最新標題解析結束日期，
    判斷 as_of（預設今日）是否仍在處置期間內。

    回測模式（傳入 as_of）會只採用 published <= as_of 的新聞，避免 look-ahead；
    無法解析發布日期的新聞在回測模式下一律捨棄（寧可漏報，不可洩漏未來）。

    Returns:
        {
            "is_disposal": bool,
            "until": "YYYY/MM/DD" or "",
            "days_until_end": int or None,   # 幾天後出關（已出關為負數）
            "reason": str,
            "source_title": str,
        }
    """
    today = as_of or date.today()
    backtest = as_of is not None
    cache_key = f"disposal_news_{code}_{today.isoformat()}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    from tw_stock_agent.news.scanner import search_company_news

    result = {"is_disposal": False, "until": "", "days_until_end": None,
              "reason": "", "source_title": ""}

    # 搜尋近 45 天的處置公告新聞
    # 三條查詢都已包含股票代號或名稱，直接信任 Google 搜尋結果
    queries = [f"{code} 處置股公告", f"{code} 列入處置", f"{name_zh} 處置股"]
    articles = []
    for q in queries:
        arts = search_company_news(q, max_articles=8, max_age_hours=1080, snippet_chars=150)
        articles.extend(arts)

    # 只要標題含「處置」即保留（查詢本身已限定股票，不需再過濾標題內容）
    relevant = [art for art in articles if "處置" in art.get("title", "")]

    # 回測模式：剔除發布日期晚於 as_of（或無法解析日期）的新聞，杜絕 look-ahead
    if backtest:
        filtered = []
        for art in relevant:
            pub_d = _parse_pub_date(art.get("published", "") or "")
            if pub_d is not None and pub_d <= today:
                filtered.append(art)
        relevant = filtered

    if not relevant:
        _set_cache(cache_key, result)
        return result

    # 收集所有新聞的解析日期，取最大值（延長處置會有多筆公告）
    best_end: date | None = None
    best_title = ""
    has_undated_recent = False  # 有近期但無法解析日期的新聞

    for art in relevant:
        title = art.get("title", "")
        pub = art.get("published", "") or ""
        end_date = _parse_disposal_end_date(title, today)

        if end_date is not None:
            if best_end is None or end_date > best_end:
                best_end = end_date
                best_title = title
        else:
            # 無法解析結束日期但有明確的延長/處置關鍵字且為近期新聞
            # 「近期」= 發布日在 as_of 前 45 天內（相對 as_of，回測一致）
            pub_d = _parse_pub_date(pub)
            is_recent = pub_d is not None and 0 <= (today - pub_d).days <= 45
            if is_recent and any(kw in title for kw in ["延長", "再次", "繼續處置", "處置股"]):
                has_undated_recent = True

    # 判定：以最晚結束日期為準
    if best_end is not None:
        days_left = (best_end - today).days
        if days_left >= -1:  # 今日仍在處置期間（含當日容錯）
            result = {
                "is_disposal": True,
                "until": best_end.strftime("%Y/%m/%d"),
                "days_until_end": days_left,
                "reason": "異常交易（價量異常）",
                "source_title": best_title,
            }
    elif has_undated_recent:
        result = {
            "is_disposal": True,
            "until": "查看公告確認",
            "days_until_end": None,
            "reason": "異常交易（見新聞公告）",
            "source_title": relevant[0].get("title", ""),
        }

    _set_cache(cache_key, result)
    logger.debug(f"disposal_via_news {code}: {result}")
    return result


# ── 漲跌停偵測 ────────────────────────────────────────────────────────────────

def detect_limit(close: float, prev_close: float) -> str | None:
    """偵測漲停 (+9.5%) 或跌停 (-9.5%)，容錯四捨五入誤差。"""
    if prev_close <= 0:
        return None
    pct = (close - prev_close) / prev_close * 100
    if pct >= 9.5:
        return "limit_up"
    if pct <= -9.5:
        return "limit_down"
    return None


# ── 整合查詢（主要對外 API）────────────────────────────────────────────────────

def get_stock_market_status(
    code: str,
    name_zh: str,
    close: float,
    prev_close: float,
    market: str = "TW",
    as_of: date | None = None,
) -> dict:
    """查詢股票特殊市場狀態（漲跌停、處置股）。

    處置股偵測改用 Google News RSS 搜尋，不依賴 TWSE/TPEX JS 渲染頁面。
    每支股票都會自動被搜尋，不需要人工告知。

    Returns:
        {
            "limit": "limit_up" | "limit_down" | None,
            "limit_label": "漲停" | "跌停" | None,
            "is_disposal": bool,
            "disposal_until": str,
            "disposal_days_left": int | None,   # 幾天後出關
            "disposal_reason": str,
            "disposal_source": str,             # 來源新聞標題
            "summary": str,
        }
    """
    limit = detect_limit(close, prev_close)
    disposal = fetch_disposal_via_news(code, name_zh, as_of=as_of)

    parts: list[str] = []
    limit_label = None

    if limit == "limit_up":
        limit_label = "漲停"
        parts.append("[漲停] 當日漲停板 (+10%)")
    elif limit == "limit_down":
        limit_label = "跌停"
        parts.append("[跌停] 當日跌停板 (-10%)")

    if disposal["is_disposal"]:
        until = disposal["until"]
        days = disposal["days_until_end"]
        s = "[處置股]"
        if until and until != "查看公告確認":
            s += f" 至 {until}"
        if days is not None:
            if days == 0:
                s += "（今日出關）"
            elif days > 0:
                s += f"（{days} 天後出關）"
        parts.append(s)

    return {
        "limit": limit,
        "limit_label": limit_label,
        "is_disposal": disposal["is_disposal"],
        "disposal_until": disposal["until"],
        "disposal_days_left": disposal["days_until_end"],
        "disposal_reason": disposal["reason"],
        "disposal_source": disposal["source_title"],
        "summary": "；".join(parts),
    }
