"""RSS 新聞掃描器：抓取 + 初篩 + tenacity 重試。"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from tw_stock_agent.config import cfg

# ── 來源定義 ──────────────────────────────────────────────────────────────────

SOURCES: dict[str, dict[str, str]] = {
    # ★ 最可靠（2026-05 實測存活）
    "twse_official": {
        "name": "TWSE 官方公告",
        "url": "https://www.twse.com.tw/rwd/zh/news/feed?type=rss",
    },
    "udn_money": {
        "name": "經濟日報 股市",
        "url": "https://money.udn.com/rssfeed/news/1001/5588",
    },
    # 廣泛版面
    "ctee_tech": {
        "name": "工商時報 科技",
        "url": "https://ctee.com.tw/feed",
    },
    # Yahoo 主題關鍵字
    "yahoo_tsmc": {
        "name": "Yahoo｜台積電",
        "url": "https://tw.stock.yahoo.com/rss?q=tsmc",
    },
    "yahoo_thermal": {
        "name": "Yahoo｜散熱",
        "url": "https://tw.stock.yahoo.com/rss?q=%E6%95%A3%E7%86%B1",
    },
    "yahoo_server": {
        "name": "Yahoo｜伺服器",
        "url": "https://tw.stock.yahoo.com/rss?q=%E4%BC%BA%E6%9C%8D%E5%99%A8",
    },
    "yahoo_abf": {
        "name": "Yahoo｜ABF",
        "url": "https://tw.stock.yahoo.com/rss?q=ABF",
    },
    "yahoo_cowos": {
        "name": "Yahoo｜CoWoS",
        "url": "https://tw.stock.yahoo.com/rss?q=CoWoS",
    },
    "yahoo_hbm": {
        "name": "Yahoo｜HBM",
        "url": "https://tw.stock.yahoo.com/rss?q=HBM",
    },
    "yahoo_ai_chip": {
        "name": "Yahoo｜AI 晶片",
        "url": "https://tw.stock.yahoo.com/rss?q=AI%E6%99%B6%E7%89%87",
    },
    "yahoo_pcb": {
        "name": "Yahoo｜PCB 載板",
        "url": "https://tw.stock.yahoo.com/rss?q=PCB",
    },
    "yahoo_semiconductor": {
        "name": "Yahoo｜半導體",
        "url": "https://tw.stock.yahoo.com/rss?q=%E5%8D%8A%E5%B0%8E%E9%AB%94",
    },
}

# ── 標題初篩關鍵字 ─────────────────────────────────────────────────────────────

MOMENTUM_WORDS = ["漲", "創高", "訂單", "法說", "受惠", "擴產", "受益", "買超", "營收", "新高", "出貨"]
SECTOR_WORDS = [
    "AI", "伺服器", "散熱", "CoWoS", "HBM", "ABF", "載板", "機殼",
    "連接器", "液冷", "水冷", "Blackwell", "GB200", "NVL", "供應鏈",
    "先進封裝", "晶片", "GPU", "TPU",
    # 宏觀/市場事件
    "SpaceX", "Starlink", "衛星", "IPO", "上市", "法說會",
    "Nvidia", "NVDA", "輝達", "AMD", "Intel", "高通", "蘋果",
    "關稅", "出口管制", "美中", "半導體禁令",
]


def is_relevant(title: str) -> bool:
    return any(w in title for w in MOMENTUM_WORDS) or any(w in title for w in SECTOR_WORDS)


# ── 內部 fetch ──────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT = (4, 8)
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TW-Stock-Agent/1.0)"}


def _clean_html(raw: str) -> str:
    soup = BeautifulSoup(raw or "", "html.parser")
    for tag in soup(["script", "style", "img", "svg", "iframe"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
    return text


def _fetch_feed(url: str) -> feedparser.FeedParserDict:
    r = requests.get(url, headers=_HEADERS, timeout=_DEFAULT_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return feedparser.parse(r.content)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       reraise=True)
def _fetch_feed_with_retry(url: str) -> feedparser.FeedParserDict:
    return _fetch_feed(url)


def _scan_one(src_key: str, max_age_hours: int, max_articles: int, snippet_chars: int
              ) -> tuple[str, list[dict]]:
    meta = SOURCES.get(src_key)
    if not meta:
        return src_key, []
    try:
        d = _fetch_feed_with_retry(meta["url"])
    except Exception as exc:
        return src_key, [{"source": src_key, "title": "[ERROR]", "link": meta["url"],
                          "published": "", "content": str(exc), "_error": True}]

    now = time.time()
    out: list[dict] = []
    for e in (d.entries or [])[:max_articles]:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        # 時效性過濾
        pub = e.get("published_parsed")
        if pub and max_age_hours:
            age_h = (now - time.mktime(pub)) / 3600
            if age_h > max_age_hours:
                continue
        # 內文擷取
        raw_html = (e.get("content") or [{}])[0].get("value", "") or e.get("summary", "")
        text = _clean_html(raw_html)[:snippet_chars]
        link = urljoin(meta["url"], (e.get("link") or "").strip())
        out.append({
            "source": src_key,
            "source_name": meta["name"],
            "title": title,
            "link": link,
            "published": e.get("published", ""),
            "content": text,
        })
    return src_key, out


# ── 個別公司新聞搜尋 ──────────────────────────────────────────────────────────

def search_company_news(
    query: str,
    max_articles: int = 10,
    max_age_hours: int = 72,
    snippet_chars: int = 200,
) -> list[dict]:
    """用 Google News RSS 搜尋特定公司/關鍵字新聞，回傳含連結的文章列表。

    Args:
        query: 搜尋字串，例如 "聯發科 砍單" 或 "台積電 虧損 OR 衰退"
        max_articles: 最多回傳幾筆
        max_age_hours: 只取這幾小時內的文章

    Returns:
        List of {title, link, published, snippet}
    """
    from urllib.parse import quote_plus

    url = (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    )
    try:
        d = _fetch_feed_with_retry(url)
    except Exception as exc:
        return [{"title": f"[搜尋失敗] {exc}", "link": "", "published": "", "snippet": ""}]

    now = time.time()
    results: list[dict] = []
    for e in (d.entries or [])[:max_articles * 2]:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        pub = e.get("published_parsed")
        if pub and max_age_hours:
            if (now - time.mktime(pub)) / 3600 > max_age_hours:
                continue
        link = (e.get("link") or "").strip()
        raw_html = (e.get("content") or [{}])[0].get("value", "") or e.get("summary", "")
        snippet = _clean_html(raw_html)[:snippet_chars]
        results.append({
            "title": title,
            "link": link,
            "published": e.get("published", ""),
            "snippet": snippet,
        })
        if len(results) >= max_articles:
            break
    return results


def search_news_around_date(
    query: str,
    target_date,           # datetime.date
    window_days: int = 1,
    max_articles: int = 15,
    snippet_chars: int = 200,
) -> list[dict]:
    """搜尋『截止 target_date（含）為止』的歷史新聞（回測專用，防未來數據洩漏）。

    使用 Google News 的 after:/before: 日期運算子，繞過 max_age_hours 限制。

    【時間窗（盤後決策慣例，與回測 D→D+1 對齊）】
    - 訊號日 D 的量化資料用 D 收盤計算，預測 D+1；決策時點在 D 收盤後（例：當晚 23:30）。
    - 可看的新聞 = 截止 D 當天（含 D 盤後新聞），**嚴禁看到 D+1**（否則等於看到隔日結果）。
    - Google News 的 before: 為『嚴格小於』，故 before = D+1 → 最新只到 D 當天。
    - after = D-(window_days+1) → 往前留幾天當背景（含週末 buffer）。
      window_days=1 時約取 [D-1, D] 兩天的新聞。

    Args:
        target_date: 訊號日 D（datetime.date）
        window_days: 往前回看的天數（預設 1）
    """
    from datetime import timedelta
    # ≤ D：含 D 當天盤後新聞，排除 D+1（杜絕「看到隔天漲跌結果」的洩漏）
    after = (target_date - timedelta(days=window_days + 1)).isoformat()
    before = (target_date + timedelta(days=1)).isoformat()
    date_query = f"{query} after:{after} before:{before}"
    # 不限 max_age_hours（舊文章不應被時效過濾）
    return search_company_news(
        date_query,
        max_articles=max_articles,
        max_age_hours=0,        # 0 = 停用時效過濾
        snippet_chars=snippet_chars,
    )


# ── 多策略個股新聞搜尋 ────────────────────────────────────────────────────────

def search_company_news_enhanced(
    name_zh: str,
    code: str = "",
    name_en: str = "",
    max_articles: int = 20,
    max_age_hours: int = 72,
    snippet_chars: int = 250,
) -> list[dict]:
    """多策略搜尋：中文名 + 代號 + 法說/電話會議 + 英文名稱。

    策略：
    - 一般新聞用 max_age_hours（預設 72h）
    - 法說 / earnings call 固定用 7 天（168h），避免法說只辦一次卻太舊
    - 若結果 < 5 篇，自動以 7 天重試主要查詢（fallback）
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    _CONF_HOURS = 168  # 法說/earnings call 固定 7 天

    # (query, max_n, age_hours)
    queries: list[tuple[str, int, int]] = [
        (name_zh,                      max_articles // 2 + 2, max_age_hours),
        (f"{name_zh} 法說",            8,                     _CONF_HOURS),
        (f"{name_zh} 法人說明會",      5,                     _CONF_HOURS),
        (f"{name_zh} 電話會議",        5,                     _CONF_HOURS),
    ]
    if code:
        queries.append((f"{code} {name_zh}", 6, max_age_hours))
        queries.append((f"{code} 法說",      5, _CONF_HOURS))
    if name_en:
        queries.append((name_en,                          8, max_age_hours))
        queries.append((f"{name_en} earnings call",       5, _CONF_HOURS))
        queries.append((f"{name_en} investor day",        4, _CONF_HOURS))

    seen: set[str] = set()
    results: list[dict] = []

    def _fetch(q: str, n: int, age: int) -> list[dict]:
        return search_company_news(
            q, max_articles=n,
            max_age_hours=age,
            snippet_chars=snippet_chars,
        )

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_fetch, q, n, age): (q, n, age) for q, n, age in queries}
        for fut in _as_completed(futures, timeout=30):
            try:
                for item in (fut.result() or []):
                    title = item.get("title", "")
                    if title and "[搜尋失敗]" not in title and title not in seen:
                        seen.add(title)
                        results.append(item)
            except Exception:
                pass

    # Fallback：若新聞太少，用 7 天重試主要中文名查詢
    if len(results) < 5 and max_age_hours < 168:
        try:
            for item in search_company_news(
                name_zh, max_articles=12, max_age_hours=168, snippet_chars=snippet_chars
            ):
                title = item.get("title", "")
                if title and "[搜尋失敗]" not in title and title not in seen:
                    seen.add(title)
                    item["_extended"] = True
                    results.append(item)
        except Exception:
            pass

    # 依發布時間排序（最新在前）
    results.sort(key=lambda a: a.get("published", "") or "", reverse=True)
    return results[:max_articles]


# ── 公開 API ───────────────────────────────────────────────────────────────────

def scan_news(
    *,
    sources: list[str] | None = None,
    max_age_hours: int | None = None,
    max_articles: int | None = None,
    snippet_chars: int | None = None,
    deadline_sec: int = 20,
) -> list[dict]:
    """掃描所有 RSS 來源，回傳通過初篩的新聞列表。

    Returns:
        List of article dicts with keys: source, source_name, title, link, published, content
    """
    selected = sources or list(SOURCES.keys())
    max_age = max_age_hours if max_age_hours is not None else cfg("news.max_age_hours", 48)
    max_art = max_articles if max_articles is not None else cfg("news.max_articles_per_source", 30)
    snippet = snippet_chars if snippet_chars is not None else cfg("news.snippet_chars", 300)

    all_articles: list[dict] = []
    workers = min(6, len(selected))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_scan_one, k, max_age, max_art, snippet): k for k in selected}
        try:
            for fut in as_completed(futures, timeout=deadline_sec):
                _, items = fut.result()
                all_articles.extend(items)
        except TimeoutError:
            pass

    # 去重（同標題）
    seen: set[str] = set()
    deduped: list[dict] = []
    for a in all_articles:
        key = a["title"].strip()
        if key not in seen and not a.get("_error"):
            seen.add(key)
            deduped.append(a)

    return deduped
