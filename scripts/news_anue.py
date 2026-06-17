"""Anue 鉅亨 全文新聞抓取器(leak-safe、含快取與限流)。

驗證過的 API(headers 帶 User-Agent，API 另帶 Origin/Referer):
- 清單: GET https://api.cnyes.com/media/api/v1/newslist/category/tw_stock
        ?startAt={unix}&endAt={unix}&limit=30&page={n}
        回傳 j['items']['data']=list,每筆 newsId/title/publishAt(unix秒)/
        market=[{code,name,symbol:'TWS:2330:STOCK'}](該篇提及的個股,可能 None)。
        分頁看 j['items']['last_page']。
- 全文: GET https://news.cnyes.com/news/id/{newsId}(HTML)
        → trafilatura.extract(html, output_format='txt', include_comments=False)

功能:給日期區間 → 抓清單 → 用 market[].code 對映 base_universe.json 個股 →
      抓全文 → 快取 data/anue_cache/(清單 by day + 文章 body by newsId)。
禮貌限流:每 request sleep ~0.5-0.8s、失敗指數退避重試。
leak-safe:每篇記 publishAt(unix);提供 articles_for(code, as_of, ...) 只取
      publishAt < 決策日盤後截點 的文章(嚴格 <,排除當天)。

用法(可獨立執行做煙霧測試):
    uv run python scripts/news_anue.py --start 2026-05-01 --end 2026-06-08
    uv run python scripts/news_anue.py --code 2330 --as-of 2026-06-05
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

try:
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tw_stock_agent.config import DATA_DIR

CACHE_DIR = DATA_DIR / "anue_cache"
LIST_DIR = CACHE_DIR / "list"          # 每日清單 {YYYY-MM-DD}.json
BODY_DIR = CACHE_DIR / "body"          # 每篇全文 {newsId}.json
for _d in (CACHE_DIR, LIST_DIR, BODY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LIST_URL = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"
ART_URL = "https://news.cnyes.com/news/id/{nid}"
HEADERS_API = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Origin": "https://news.cnyes.com",
    "Referer": "https://news.cnyes.com/",
}
HEADERS_HTML = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

THROTTLE = 0.6      # 秒/request
MAX_RETRY = 4


def _get(url, headers, params=None, as_json=True):
    """禮貌 GET + 指數退避重試。回傳 json dict 或 text;失敗回 None。"""
    delays = [2, 4, 8, 16]
    for att in range(MAX_RETRY):
        try:
            time.sleep(THROTTLE)
            r = requests.get(url, headers=headers, params=params, timeout=45)
            if r.status_code == 200:
                return r.json() if as_json else r.text
            if r.status_code in (403, 404):
                return None
            # 429 / 5xx → 退避重試
        except Exception:
            pass
        if att < MAX_RETRY - 1:
            time.sleep(delays[att])
    return None


# ── 清單(按日抓,逐日快取) ────────────────────────────────────────────────

def _day_bounds(day: str) -> tuple[int, int]:
    d = dt.date.fromisoformat(day)
    start = int(dt.datetime(d.year, d.month, d.day, 0, 0, 0).timestamp())
    end = int(dt.datetime(d.year, d.month, d.day, 23, 59, 59).timestamp())
    return start, end


def fetch_list_day(day: str, force: bool = False) -> list[dict]:
    """抓某一天(本地時區)的 tw_stock 新聞清單,全分頁。逐日快取。
    回傳精簡 list:[{newsId, title, publishAt, codes:[..], summary}]。"""
    cache_p = LIST_DIR / f"{day}.json"
    if cache_p.exists() and not force:
        try:
            return json.loads(cache_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    start, end = _day_bounds(day)
    out, page, last = [], 1, 1
    while page <= last:
        params = {"startAt": start, "endAt": end, "limit": 30, "page": page}
        j = _get(LIST_URL, HEADERS_API, params=params, as_json=True)
        if not j or "items" not in j:
            break
        items = j["items"]
        last = items.get("last_page", page) or page
        for a in items.get("data", []) or []:
            mkt = a.get("market") or []
            codes = []
            for m in mkt:
                sym = m.get("symbol", "")
                # 只收台股(TWS:CODE:STOCK)
                if sym.startswith("TWS:") and m.get("code"):
                    codes.append(str(m["code"]))
            out.append({
                "newsId": a.get("newsId"),
                "title": a.get("title", ""),
                "publishAt": a.get("publishAt", 0),
                "codes": codes,
                "summary": a.get("summary", ""),
            })
        page += 1
    cache_p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def fetch_list_range(start: str, end: str, force: bool = False) -> list[dict]:
    """抓 [start, end](含)每一天的清單,合併。"""
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    all_items, d = [], s
    while d <= e:
        items = fetch_list_day(d.isoformat(), force=force)
        all_items.extend(items)
        d += dt.timedelta(days=1)
    return all_items


# ── 全文(逐篇快取) ──────────────────────────────────────────────────────

def fetch_body(news_id, force: bool = False) -> str:
    """抓單篇全文(trafilatura 萃取純文字)。逐篇快取。失敗回 ''。"""
    if not news_id:
        return ""
    cache_p = BODY_DIR / f"{news_id}.json"
    if cache_p.exists() and not force:
        try:
            return json.loads(cache_p.read_text(encoding="utf-8")).get("body", "")
        except Exception:
            pass
    html = _get(ART_URL.format(nid=news_id), HEADERS_HTML, as_json=False)
    body = ""
    if html and trafilatura is not None:
        try:
            body = trafilatura.extract(html, output_format="txt",
                                       include_comments=False) or ""
        except Exception:
            body = ""
    cache_p.write_text(json.dumps({"body": body}, ensure_ascii=False), encoding="utf-8")
    return body


# ── leak-safe 對映 / 過濾 ──────────────────────────────────────────────────

def _cutoff_unix(as_of: str) -> int:
    """決策日盤後截點:嚴格 < 該截點 才算 leak-safe。
    第一輪慣例:as_of 當天 00:00(即只用 < as_of 當天 的新聞,排除當天盤後)。
    """
    d = dt.date.fromisoformat(str(as_of)[:10])
    return int(dt.datetime(d.year, d.month, d.day, 0, 0, 0).timestamp())


def articles_for(code: str, as_of: str, window_days: int = 5,
                 max_articles: int = 6, with_body: bool = True) -> list[dict]:
    """回傳某股、截止 as_of(嚴格 < 當天 00:00)的 Anue 文章(含全文 body)。

    - 從 [as_of-window_days, as_of] 的清單快取挑出 codes 含 code 的文章。
    - publishAt < cutoff 才收(leak-safe)。
    - 取最近 max_articles 篇,可選帶 body。
    回傳 [{title, link, published, content, publishAt}]。
    """
    code = str(code)
    cutoff = _cutoff_unix(as_of)
    as_d = dt.date.fromisoformat(str(as_of)[:10])
    start = (as_d - dt.timedelta(days=window_days)).isoformat()
    items = fetch_list_range(start, as_d.isoformat())

    hits = []
    for a in items:
        if code not in a.get("codes", []):
            continue
        pa = a.get("publishAt", 0) or 0
        if pa >= cutoff:        # leak-safe:嚴格排除當天及之後
            continue
        hits.append(a)
    # 去重 + 最近優先
    seen, uniq = set(), []
    for a in sorted(hits, key=lambda x: x.get("publishAt", 0), reverse=True):
        nid = a.get("newsId")
        if nid in seen:
            continue
        seen.add(nid)
        uniq.append(a)
    uniq = uniq[:max_articles]

    out = []
    for a in uniq:
        pub_iso = dt.datetime.fromtimestamp(a.get("publishAt", 0)).isoformat()
        body = fetch_body(a["newsId"]) if with_body else ""
        out.append({
            "title": a.get("title", ""),
            "link": ART_URL.format(nid=a.get("newsId")),
            "published": pub_iso,
            "content": body or a.get("summary", ""),
            "publishAt": a.get("publishAt", 0),
        })
    return out


def prefetch_range(start: str, end: str, fetch_bodies: bool = True,
                   universe_codes: set | None = None, force: bool = False) -> dict:
    """預抓一段期間的清單(+可選全文),回傳統計。供 compare 跑前暖快取。"""
    items = fetch_list_range(start, end, force=force)
    tagged = [a for a in items if a.get("codes")]
    in_univ = [a for a in tagged
               if universe_codes is None
               or any(c in universe_codes for c in a["codes"])]
    n_body = 0
    if fetch_bodies:
        for a in in_univ:
            b = fetch_body(a["newsId"])
            if b:
                n_body += 1
    return {"days": (start, end), "list_total": len(items),
            "tagged_stock": len(tagged), "in_universe": len(in_univ),
            "bodies_fetched": n_body if fetch_bodies else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--code")
    ap.add_argument("--as-of")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-body", action="store_true")
    args = ap.parse_args()

    if args.code and args.as_of:
        arts = articles_for(args.code, args.as_of, with_body=not args.no_body)
        print(f"== {args.code} @ {args.as_of}: {len(arts)} 篇(leak-safe) ==")
        for a in arts:
            print(f"- [{a['published'][:16]}] {a['title']}")
            if a["content"]:
                print(f"    body {len(a['content'])} chars: {a['content'][:80]}...")
        return

    if args.start and args.end:
        u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
        stats = prefetch_range(args.start, args.end, fetch_bodies=not args.no_body,
                               universe_codes=set(u.keys()), force=args.force)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    ap.error("需 --start/--end 或 --code/--as-of")


if __name__ == "__main__":
    main()
