"""FinMind API 客戶端 — 三大法人 & 月營收

資料本地快取到 data/finmind_cache/，避免回測時反覆打 API。
所有查詢函式支援 as_of_date，確保歷史回測不看未來資料。
"""
from __future__ import annotations

import csv
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from tw_stock_agent.config import DATA_DIR, get_settings

BASE_URL  = "https://api.finmindtrade.com/api/v4/data"
CACHE_DIR = DATA_DIR / "finmind_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 三大法人機構名稱對照
INST_FOREIGN  = "Foreign_Investor"
INST_TRUST    = "Investment_Trust"
INST_DEALER   = "Dealer"


_TOKENS: list[str] | None = None
_tok_idx = 0


def _token_pool() -> list[str]:
    """可用 token 清單(雙帳號分攤額度)。"""
    global _TOKENS
    if _TOKENS is None:
        s = get_settings()
        toks = [s.finmind_token, getattr(s, "finmind_token2", "")]
        _TOKENS = [t for t in toks if t] or [""]
    return _TOKENS


def _headers() -> dict:
    pool = _token_pool()
    return {"Authorization": f"Bearer {pool[0]}"} if pool[0] else {}


def _fetch_api(dataset: str, data_id: str, start_date: str, end_date: str) -> list[dict]:
    """呼叫 FinMind API：多 token 輪流(分攤額度)+ 撞 402 自動切換。失敗回 []。"""
    global _tok_idx
    pool = _token_pool()
    params = {"dataset": dataset, "data_id": data_id,
              "start_date": start_date, "end_date": end_date}
    n = len(pool)
    for _ in range(n):
        tok = pool[_tok_idx % n]
        _tok_idx = (_tok_idx + 1) % n        # 輪流:每次換一支,分攤每小時額度
        headers = {"Authorization": f"Bearer {tok}"} if tok else {}
        try:
            r = requests.get(BASE_URL, params=params, headers=headers, timeout=20)
            if r.status_code == 402:         # 此 token 額度用完 → 換下一支
                continue
            r.raise_for_status()
            payload = r.json()
            if payload.get("status") != 200:
                if payload.get("status") == 402:
                    continue
                logger.warning(f"FinMind {dataset} {data_id}: status={payload.get('status')} msg={payload.get('msg','')}")
                return []
            return payload.get("data", [])
        except Exception as e:
            if "402" in str(e):
                continue
            logger.error(f"FinMind API error {dataset} {data_id}: {e}")
            return []
    return []        # 所有 token 都被限流


# ── 快取層 ────────────────────────────────────────────────────────────────────

def _cache_path(dataset: str, ticker: str) -> Path:
    return CACHE_DIR / f"{dataset}_{ticker}.json"


def _load_cache(dataset: str, ticker: str) -> list[dict] | None:
    p = _cache_path(dataset, ticker)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _save_cache(dataset: str, ticker: str, rows: list[dict]) -> None:
    _cache_path(dataset, ticker).write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )


def _get_data(dataset: str, ticker: str,
              start: str = "2024-01-01",
              force_refresh: bool = False) -> list[dict]:
    """取得資料（優先用快取，快取不存在或 force_refresh 則呼叫 API）。"""
    cached = None if force_refresh else _load_cache(dataset, ticker)
    if cached is not None:
        return cached
    today = date.today().isoformat()
    rows = _fetch_api(dataset, ticker, start, today)
    if rows:
        _save_cache(dataset, ticker, rows)
        logger.debug(f"  FinMind cached {dataset} {ticker}: {len(rows)} rows")
    return rows


# ── 歷史新聞(TaiwanStockNews,per-stock,防未來洩漏)─────────────────────────────

def _news_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"news_{ticker}.json"


def get_historical_news(ticker: str, target_date, window_days: int = 1,
                        max_articles: int = 15) -> list[dict]:
    """FinMind TaiwanStockNews:某股截止 target_date(含)的歷史新聞,零未來洩漏。

    - 單日分段抓(FinMind 多日會 "size too large")+ 逐日 JSON 快取(news_<ticker>.json)。
    - 只回傳 date <= target_date 的新聞(嚴禁看到隔日結果)。FinMind 涵蓋 2022→今。
    回傳 [{title, link, published, source, content}](content 空,FinMind 只有標題)。
    """
    from datetime import timedelta
    if hasattr(target_date, "isoformat"):
        target_date = target_date.isoformat()
    target_date = str(target_date)[:10]
    start = (date.fromisoformat(target_date) - timedelta(days=window_days))

    cache = {}
    p = _news_cache_path(ticker)
    if p.exists():
        try:
            cache = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    days, d = [], start
    end = date.fromisoformat(target_date)
    while d <= end:
        days.append(d.isoformat()); d += timedelta(days=1)

    changed = False
    for day in days:
        if day not in cache:                       # 單日抓(安全,不會 too large)
            rows = _fetch_api("TaiwanStockNews", ticker, day, day)
            cache[day] = [{"title": r.get("title", ""), "link": r.get("link", ""),
                           "source": r.get("source", ""), "date": r.get("date", "")}
                          for r in rows]
            changed = True
    if changed:
        p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    out = []
    for day in days:
        for r in cache.get(day, []):
            if str(r.get("date", ""))[:10] < target_date:      # 防洩漏:嚴格 < 決策日(排除當天盤後新聞)
                out.append({"title": r.get("title", ""), "link": r.get("link", ""),
                            "published": r.get("date", ""), "source": r.get("source", ""),
                            "content": ""})
    out.sort(key=lambda a: a.get("published", ""), reverse=True)
    return out[:max_articles]


# ── 三大法人 ──────────────────────────────────────────────────────────────────

def _foreign_flow_severity(current_5d: int, daily_zhang: list[int]) -> tuple[str, str]:
    """依個股『自身歷史』的 5 日外資淨額分布，判斷目前流量的嚴重程度。

    用百分位（不是絕對張數）→ 大型股小型股一致；只有相對該股歷史極端的
    賣超才標『嚴重出貨』。回傳 (tier, note)。
    """
    if len(daily_zhang) >= 6:
        roll5 = [sum(daily_zhang[i:i + 5]) for i in range(len(daily_zhang) - 4)]
    else:
        roll5 = daily_zhang[:]
    if len(roll5) < 8:
        # 歷史樣本太少 → 一律保守視為中性，避免誤判
        return ("中性", "（歷史樣本不足，僅供參考）")

    below = sum(1 for x in roll5 if x <= current_5d)
    pct = below / len(roll5) * 100   # 百分位：越低=賣超越極端，越高=買超越極端
    if current_5d < 0 and pct <= 10:
        return ("嚴重出貨", f"近5日外資淨流量處於該股歷史最賣超的前 {pct:.0f}%（恐慌性出貨）")
    if current_5d < 0 and pct <= 30:
        return ("中度調節", f"外資賣超偏多但仍在常見範圍（歷史百分位 {pct:.0f}%）")
    if current_5d < 0:
        return ("輕微賣超", "外資小幅調節，屬正常週轉")
    if pct >= 90:
        return ("明顯買超", f"外資大舉買超，處於歷史前 {100 - pct:.0f}%（強力進場）")
    return ("中性", "外資流量在正常範圍")


def get_institutional_flow(
    ticker: str,
    as_of_date: Optional[date] = None,
    lookback_days: int = 10,
    force_refresh: bool = False,
) -> dict:
    """
    回傳截止 as_of_date 前 lookback_days 個交易日的三大法人淨買超統計。

    回傳格式:
    {
        "foreign_net_5d": int,      # 外資近5日淨買超(張)
        "foreign_net_10d": int,     # 外資近10日淨買超(張)
        "foreign_consecutive": int, # 外資連續買超天數(正=買,負=賣)
        "trust_net_5d": int,        # 投信近5日淨買超(張)
        "total_net_5d": int,        # 三大法人合計近5日淨買超(張)
        "summary_text": str,        # LLM 可讀摘要
        "days_available": int,      # 實際取得天數
    }
    """
    cutoff = (as_of_date or date.today()).isoformat()
    rows = _get_data("TaiwanStockInstitutionalInvestorsBuySell", ticker,
                     start="2024-01-01", force_refresh=force_refresh)

    # 篩選截止日前的資料
    rows = [r for r in rows if r["date"] < cutoff]
    if not rows:
        return _empty_inst_result()

    # 取最近 lookback_days 個交易日（按日期分組）
    by_date: dict[str, dict[str, int]] = {}
    for r in rows:
        d = r["date"]
        if d not in by_date:
            by_date[d] = {}
        net = int(r.get("buy", 0)) - int(r.get("sell", 0))
        by_date[d][r["name"]] = by_date[d].get(r["name"], 0) + net

    sorted_dates = sorted(by_date.keys())[-lookback_days:]
    if not sorted_dates:
        return _empty_inst_result()

    recent5  = sorted_dates[-5:]  if len(sorted_dates) >= 5  else sorted_dates
    recent10 = sorted_dates[-10:] if len(sorted_dates) >= 10 else sorted_dates

    def net_sum_zhang(dates: list[str], inst: str) -> int:
        # FinMind 單位是「股」，除以 1000 轉為「張」
        return sum(by_date[d].get(inst, 0) for d in dates) // 1000

    foreign_net_5d  = net_sum_zhang(recent5,  INST_FOREIGN)
    foreign_net_10d = net_sum_zhang(recent10, INST_FOREIGN)
    trust_net_5d    = net_sum_zhang(recent5,  INST_TRUST)
    total_net_5d    = (foreign_net_5d + trust_net_5d
                       + net_sum_zhang(recent5, INST_DEALER))

    # 連續買超天數（從最近一天往前算）
    # 用所有歷史日期，不受 lookback_days 限制
    all_dates = sorted(by_date.keys())
    consecutive = 0
    for d in reversed(all_dates):
        net = by_date[d].get(INST_FOREIGN, 0)
        if net == 0:
            break
        if consecutive == 0:
            consecutive = 1 if net > 0 else -1
        elif consecutive > 0 and net > 0:
            consecutive += 1
        elif consecutive < 0 and net < 0:
            consecutive -= 1
        else:
            break

    # 外資流量嚴重程度（相對個股自身歷史；只有極端賣超才標「嚴重出貨」）
    foreign_daily_zhang = [by_date[d].get(INST_FOREIGN, 0) // 1000 for d in all_dates]
    sev_tier, sev_note = _foreign_flow_severity(foreign_net_5d, foreign_daily_zhang)

    # 組合 LLM 可讀摘要
    trend = "連續買超" if consecutive > 0 else ("連續賣超" if consecutive < 0 else "中性")
    summary = (
        f"【三大法人（截止 {cutoff}）】\n"
        f"外資近5日淨買超 {foreign_net_5d:+,} 張，近10日 {foreign_net_10d:+,} 張\n"
        f"外資流量嚴重程度：【{sev_tier}】{sev_note}\n"
        f"外資{trend} {abs(consecutive)} 天\n"
        f"投信近5日淨買超 {trust_net_5d:+,} 張\n"
        f"三大法人合計近5日 {total_net_5d:+,} 張"
    )

    return {
        "foreign_net_5d":      foreign_net_5d,
        "foreign_net_10d":     foreign_net_10d,
        "foreign_consecutive": consecutive,
        "foreign_severity":    sev_tier,
        "trust_net_5d":        trust_net_5d,
        "total_net_5d":        total_net_5d,
        "summary_text":        summary,
        "days_available":      len(sorted_dates),
    }


def _empty_inst_result() -> dict:
    return {
        "foreign_net_5d": 0, "foreign_net_10d": 0,
        "foreign_consecutive": 0, "trust_net_5d": 0,
        "total_net_5d": 0, "summary_text": "", "days_available": 0,
    }


# ── 月營收 YoY ────────────────────────────────────────────────────────────────

def get_revenue_yoy(
    ticker: str,
    as_of_date: Optional[date] = None,
    force_refresh: bool = False,
) -> dict:
    """
    回傳截止 as_of_date 前最新公告月份的營收 YoY 成長率。

    月營收公告時間：每月 10 日前公告上月營收。
    as_of_date = 2026-03-20 → 可以看到 2 月份營收（2026-02，約 3/10 公告）

    回傳格式:
    {
        "yoy_pct": float,           # YoY 成長率 (%)
        "revenue": int,             # 當月營收（元）
        "revenue_month": str,       # "2026-02"
        "mom_pct": float,           # MoM 成長率 (%)
        "consecutive_growth": int,  # 連續YoY正成長月數
        "summary_text": str,
    }
    """
    cutoff = as_of_date or date.today()
    # 月營收約 10 日前公告，所以可看到的最新月份是上月（若今天 > 10 日）或上上月
    if cutoff.day >= 10:
        max_available_month = date(cutoff.year, cutoff.month, 1) - timedelta(days=1)
    else:
        tmp = date(cutoff.year, cutoff.month, 1) - timedelta(days=1)
        max_available_month = date(tmp.year, tmp.month, 1) - timedelta(days=1)

    rows = _get_data("TaiwanStockMonthRevenue", ticker,
                     start="2023-01-01", force_refresh=force_refresh)

    # 篩選可用月份（revenue_year, revenue_month 組合 <= max_available_month）
    usable = []
    for r in rows:
        ry, rm = int(r["revenue_year"]), int(r["revenue_month"])
        row_date = date(ry, rm, 1)
        if row_date <= max_available_month:
            usable.append((row_date, int(r["revenue"])))

    if len(usable) < 2:
        return _empty_rev_result()

    usable.sort(key=lambda x: x[0])
    latest_date, latest_rev = usable[-1]

    # YoY：找去年同月
    yoy_date = date(latest_date.year - 1, latest_date.month, 1)
    yoy_rows = [r for r in usable if r[0] == yoy_date]
    yoy_pct  = None
    if yoy_rows:
        prev_rev = yoy_rows[0][1]
        yoy_pct = (latest_rev / prev_rev - 1) * 100 if prev_rev else None

    # MoM
    mom_pct = None
    if len(usable) >= 2:
        prev_rev_mom = usable[-2][1]
        mom_pct = (latest_rev / prev_rev_mom - 1) * 100 if prev_rev_mom else None

    # 連續 YoY 正成長月數
    consecutive = 0
    for i in range(len(usable) - 1, -1, -1):
        cur_date, cur_rev = usable[i]
        py_date = date(cur_date.year - 1, cur_date.month, 1)
        py = [r for r in usable if r[0] == py_date]
        if not py:
            break
        if cur_rev > py[0][1]:
            consecutive += 1
        else:
            break

    rev_label = f"{latest_date.year}-{latest_date.month:02d}"
    yoy_str   = f"{yoy_pct:+.1f}%" if yoy_pct is not None else "N/A"
    mom_str   = f"{mom_pct:+.1f}%" if mom_pct is not None else "N/A"
    summary = (
        f"【月營收（{rev_label}，截止 {cutoff.isoformat()}）】\n"
        f"YoY {yoy_str}，MoM {mom_str}\n"
        f"連續YoY正成長 {consecutive} 個月"
    )

    return {
        "yoy_pct":            yoy_pct,
        "revenue":            latest_rev,
        "revenue_month":      rev_label,
        "mom_pct":            mom_pct,
        "consecutive_growth": consecutive,
        "summary_text":       summary,
    }


def _empty_rev_result() -> dict:
    return {
        "yoy_pct": None, "revenue": 0,
        "revenue_month": "", "mom_pct": None,
        "consecutive_growth": 0, "summary_text": "",
    }


# ── 日線價格（回測用：價格面板 + 漲跌停偵測）──────────────────────────────────

def _sanitize_ohlcv(oh: dict) -> dict:
    """清洗未還原股價:① V形壞tick(暴跌隔日暴彈/暴漲隔日暴跌)內插 ② >±10.5%跳動=除權息/分割/減資→回溯還原。
    台股單日收盤對收盤理論上 ≤±10%(漲跌停),超過者必為公司行為或壞資料。最新價不變,只調歷史。"""
    if not oh:
        return oh
    ds = sorted(oh)
    c = [float(oh[d]["close"]) for d in ds]
    n = len(c)
    for i in range(1, n - 1):                       # Pass1: V形壞tick
        if c[i - 1] > 0 and c[i] > 0 and c[i + 1] > 0:
            down = c[i] / c[i - 1] - 1; up = c[i + 1] / c[i] - 1
            if (down < -0.35 and up > 0.5) or (down > 1.0 and up < -0.4):
                c[i] = (c[i - 1] + c[i + 1]) / 2
    for i in range(n):                              # close<=0 壞值
        if c[i] <= 0:
            c[i] = c[i - 1] if i > 0 else next((x for x in c if x > 0), 1.0)
    factor = 1.0; adj = [1.0] * n                   # Pass2: 回溯還原
    for i in range(n - 1, -1, -1):
        adj[i] = factor
        if i > 0 and c[i - 1] > 0:
            r = c[i] / c[i - 1] - 1
            if r < -0.105 or r > 0.11:
                factor *= c[i] / c[i - 1]
    out: dict[str, dict] = {}
    for i, d in enumerate(ds):
        raw = oh[d]; f = adj[i]
        rc = float(raw["close"]); ratio = (c[i] / rc) if rc else 1.0
        out[d] = {"open": float(raw["open"]) * ratio * f, "high": float(raw["high"]) * ratio * f,
                  "low": float(raw["low"]) * ratio * f, "close": c[i] * f,
                  "volume": float(raw.get("volume", 0)) / f if f else float(raw.get("volume", 0)),
                  "amount": float(raw.get("amount", 0))}
    return out


def get_daily_prices(
    ticker: str,
    start: str = "2024-01-01",
    force_refresh: bool = False,
) -> dict[str, float]:
    """回傳 {date: close} 的日線收盤價對照表（已清洗+還原，本地快取）。"""
    oh = get_daily_ohlcv(ticker, start=start, force_refresh=force_refresh)
    return {d: v["close"] for d, v in oh.items()}


def get_daily_ohlcv(
    ticker: str,
    start: str = "2024-01-01",
    force_refresh: bool = False,
) -> dict[str, dict]:
    """回傳 {date: {open, high, low, close, volume, amount}} 的日線 OHLCV（本地快取）。

    複用 TaiwanStockPrice 快取（與 get_daily_prices 同一份），跨股研究/型態用。
    """
    rows = _get_data("TaiwanStockPrice", ticker, start=start, force_refresh=force_refresh)
    out: dict[str, dict] = {}
    for r in rows:
        c = r.get("close")
        if c is None:
            continue
        try:
            out[r["date"]] = {
                "open":   float(r.get("open")  or c),
                "high":   float(r.get("max")   or c),
                "low":    float(r.get("min")   or c),
                "close":  float(c),
                "volume": float(r.get("Trading_Volume") or 0),
                "amount": float(r.get("Trading_money")  or 0),
            }
        except (ValueError, TypeError):
            continue
    return _sanitize_ohlcv(out)


def get_close_on_or_before(ticker: str, as_of: date) -> tuple[str, float] | None:
    """取得 as_of（含）當日或之前最近一個交易日的收盤價。回傳 (date_str, close) 或 None。"""
    prices = get_daily_prices(ticker)
    cutoff = as_of.isoformat()
    valid = sorted(d for d in prices if d <= cutoff)
    if not valid:
        return None
    d = valid[-1]
    return d, prices[d]


def get_prev_close(ticker: str, as_of: date) -> float | None:
    """取得 as_of 前一個交易日的收盤價（用於漲跌停判斷的基準）。"""
    prices = get_daily_prices(ticker)
    cutoff = as_of.isoformat()
    valid = sorted(d for d in prices if d < cutoff)
    if not valid:
        return None
    return prices[valid[-1]]


def build_price_panel(
    tickers: list[str],
    start: str = "2025-01-01",
) -> dict[str, dict[str, float]]:
    """一次建好所有股票的 {ticker: {date: close}} 面板（回測再平衡用）。"""
    return {t: get_daily_prices(t, start=start) for t in tickers}


# ── 批量預抓（回測前執行）────────────────────────────────────────────────────

def prefetch_all(tickers: list[str], delay: float = 0.3) -> None:
    """批量預抓所有股票的三大法人和月營收資料並快取，回測前執行一次即可。"""
    logger.info(f"預抓 FinMind 資料：{len(tickers)} 支股票...")
    for i, t in enumerate(tickers, 1):
        logger.info(f"  [{i}/{len(tickers)}] {t}")
        _get_data("TaiwanStockInstitutionalInvestorsBuySell", t,
                  start="2024-01-01", force_refresh=True)
        time.sleep(delay)
        _get_data("TaiwanStockMonthRevenue", t,
                  start="2023-01-01", force_refresh=True)
        time.sleep(delay)
    logger.info("預抓完成。")