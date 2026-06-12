"""LangGraph Node 函式：Node 1 ~ Node 7。"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from tw_stock_agent.config import COMPANIES_JSON, REPORTS_DIR, TW_STOCK_INDEX, cfg, get_settings
from tw_stock_agent.data.graph_service import GraphService
from tw_stock_agent.data.price_data import get_index_return, get_ohlcv, validate_price_data
from tw_stock_agent.debate.bear import run_debate
from tw_stock_agent.news.scanner import scan_news
from tw_stock_agent.news.ticker_match import enrich_candidates, load_ticker_book, match_tickers
from tw_stock_agent.pipeline.state import ScanState
from tw_stock_agent.screener.patterns import analyze_pattern
from tw_stock_agent.screener.quantitative import screen_batch
from tw_stock_agent.signal_log import append_signals, remove_signals

# ── Singletons（一次載入，重複使用）─────────────────────────────────────────
_graph: GraphService | None = None
_ticker_book: dict | None = None


def _get_graph() -> GraphService:
    global _graph
    if _graph is None:
        _graph = GraphService(COMPANIES_JSON)
    return _graph


def _get_ticker_book() -> dict:
    global _ticker_book
    if _ticker_book is None:
        _ticker_book = load_ticker_book(TW_STOCK_INDEX)
    return _ticker_book


# ── Node 1：fetch_news ──────────────────────────────────────────────────────

def fetch_news_node(state: ScanState) -> dict:
    if state.get("as_of"):
        # as_of 模式:不抓 live 新聞(辯論層自己抓 ≤as_of 的歷史新聞);候選股改用固定掃描池
        logger.info(f"Node 1: as_of={state['as_of']} → 跳過 live 新聞掃描")
        return {"raw_news": []}
    logger.info("Node 1: fetching news (no keyword filter)...")
    articles = scan_news()
    logger.info(f"  fetched {len(articles)} articles (unfiltered, dedup only)")
    return {"raw_news": articles}


# ── Node 2：extract_entities ────────────────────────────────────────────────

def extract_entities_node(state: ScanState) -> dict:
    as_of = state.get("as_of")
    if as_of:
        # as_of 模式:候選股 = 當時(point-in-time)流動性前 N(base_universe),零未來偏誤
        from tw_stock_agent.universe import universe_as_of
        top_n = cfg("universe.as_of_top_n", 30)
        candidates = universe_as_of(as_of, top_n=top_n)
        logger.info(f"Node 2: as_of={as_of} → 掃描池=流動性前 {len(candidates)} 大(base_universe)")
        return {"candidates": candidates, "raw_news": []}
    logger.info("Node 2: extracting ticker entities...")
    ticker_book = _get_ticker_book()
    if not ticker_book:
        logger.warning("  tw_stock_index.json not found — run scripts/fetch_tw_stock_index.py first")

    found: set[str] = set()

    for news in state["raw_news"]:
        text = news["title"] + " " + news.get("content", "")[:200]
        tickers = match_tickers(text, ticker_book)
        if tickers:
            found.update(tickers)

    candidates = enrich_candidates(list(found), ticker_book)
    # 加上 source_news 追蹤（以第一筆匹配的新聞為準）
    ticker_to_news: dict[str, str] = {}
    for news in state["raw_news"]:
        text = news["title"] + " " + news.get("content", "")[:200]
        for c in candidates:
            if c["code"] not in ticker_to_news:
                if match_tickers(text, {c["code"]: ticker_book.get(c["code"], {})}):
                    ticker_to_news[c["code"]] = news["title"][:80]

    for c in candidates:
        c["source_news"] = ticker_to_news.get(c["code"], "")
        c["bfs_depth"] = 0

    # 保留 title+link+content+published 給辯論節點（LLM 需要全部內容判斷相關性）
    slim_news = [
        {
            "title": a["title"],
            "link": a.get("link", ""),
            "published": a.get("published", ""),
            "content": a.get("content", "")[:300],  # 節省 token，只傳前 300 字
        }
        for a in state["raw_news"]
    ]
    logger.info(f"  found {len(candidates)} candidates from {len(slim_news)} unfiltered articles")
    return {"candidates": candidates, "raw_news": slim_news}


# ── Node 3：supply_chain_bfs ────────────────────────────────────────────────

def supply_chain_bfs_node(state: ScanState) -> dict:
    if state.get("as_of"):
        # as_of 模式:掃描池已固定(流動性前N),不做新聞驅動的供應鏈擴張
        logger.info("Node 3: as_of 模式 → 跳過 BFS 擴張,維持固定掃描池")
        return {"bfs_expanded": state["candidates"]}
    logger.info("Node 3: BFS supply chain expansion...")
    graph = _get_graph()

    # code → graph node nickname（修正：圖節點用 nickname，不是股票全名）
    code_to_nick = graph.get_code_to_nickname()
    seeds: list[str] = []
    for c in state["candidates"]:
        code = c.get("code", "")
        nick = code_to_nick.get(code, "")
        if nick and nick not in seeds:
            seeds.append(nick)
    logger.debug(f"  BFS seeds: {seeds}")

    bfs_results = graph.bfs_beneficiaries(seeds)
    # 合併：原有 candidates + BFS 新發現（去重）
    existing_codes = {c["code"] for c in state["candidates"]}
    new_stocks = [r for r in bfs_results if r["code"] not in existing_codes]

    all_candidates = list(state["candidates"])
    for r in new_stocks:
        all_candidates.append({
            "code": r["code"],
            "name": r["name"],
            "yf_ticker": r["yf_ticker"],
            "nickname": r["nickname"],
            "bfs_depth": r["bfs_depth"],
            "via_path": r["via_path"],
            "source_news": "",
        })

    # companies.json 裡的台股永遠加入（watchlist）── 不依賴新聞命中
    watchlist_added = 0
    current_codes = {c["code"] for c in all_candidates}
    for node, data in graph.graph.nodes(data=True):
        code = data.get("code", "")
        if not code or not code.isdigit():
            continue  # 跳過美股/韓股
        if code in current_codes:
            continue
        country = data.get("country", "TW")
        suffix = ".TW" if country == "TW" else ".TWO"
        all_candidates.append({
            "code": code,
            "name": data.get("name_zh", node),
            "yf_ticker": f"{code}{suffix}",
            "nickname": node,
            "bfs_depth": 0,
            "via_path": "",
            "source_news": "[watchlist]",
        })
        current_codes.add(code)
        watchlist_added += 1

    logger.info(f"  BFS added {len(new_stocks)} + watchlist {watchlist_added} = total {len(all_candidates)}")
    return {"bfs_expanded": all_candidates}


# ── Node 4：quantitative_screen ─────────────────────────────────────────────

def quantitative_screen_node(state: ScanState) -> dict:
    logger.info("Node 4: quantitative screening...")
    as_of = state.get("as_of")
    today = date.fromisoformat(as_of) if as_of else date.today()
    results = screen_batch(state["bfs_expanded"], today=today, as_of_date=as_of or None)
    screened = []
    for r, orig in zip(results, state["bfs_expanded"]):
        merged = {**orig,
                  "close_price": r.close_price,
                  "volume_ratio": r.volume_ratio,
                  "ma5_gt_ma20": r.ma5_gt_ma20,
                  "rs_20d": r.rs_20d,
                  "weeks_52_warn": r.weeks_52_warn,
                  "pass_level": r.pass_level,
                  "screen_reason": r.reason,
                  "_screen_passed": r.passed}  # 私有標記，給後續節點用
        screened.append(merged)
        lvl = r.pass_level
        logger.info(f"  {r.code} {orig.get('name','')} → {lvl}  ({r.reason})")
    n_pass = sum(1 for s in screened if s["_screen_passed"])
    logger.info(f"  {n_pass}/{len(screened)} PASS/WARN (REJECT={len(screened)-n_pass})")
    return {"screened": screened}


# ── Node 5：pattern_match ───────────────────────────────────────────────────

def pattern_match_node(state: ScanState) -> dict:
    logger.info("Node 5: K-line pattern matching...")
    with_pattern = []
    pattern_count = 0
    as_of = state.get("as_of") or None
    for stock in state["screened"]:
        if not stock.get("_screen_passed"):
            continue  # REJECT 不送進辯論
        df = get_ohlcv(stock["yf_ticker"], as_of_date=as_of)
        result = analyze_pattern(stock, df) if not df.empty else None
        has_p = result.has_pattern if result else False
        pattern_count += int(has_p)
        merged = {
            **stock,
            "pattern_type":   result.pattern_type   if result else "none",
            "pattern_detail": result.pattern_detail if result else "no data",
            "rsi_14":         result.rsi_14         if result else 0.0,
            "macd_hist":      result.macd_hist       if result else 0.0,
        }
        # PASS → 送辯論（量化全過關，值得深度分析）
        # WARN + pattern → 送辯論
        # WARN + no pattern → 跳過 LLM，節省 token
        if stock.get("pass_level") == "PASS" or has_p:
            with_pattern.append(merged)
        else:
            logger.debug(f"  {stock['code']} WARN+no_pattern → skip debate")
    logger.info(f"  pattern={pattern_count}/{len(with_pattern)} have breakout/GC signal")
    return {"with_pattern": with_pattern}


# ── Node 6：bear_debate ─────────────────────────────────────────────────────

def _debate_score(s: dict) -> float:
    """量化分數,給 Top-N 篩選用:PASS 優先,再看相對強度/量比/型態。
    讓 universe 放大時,只有分數最高的少數股進入(貴的)LLM 辯論,成本不隨 universe 爆。"""
    lvl = {"PASS": 2, "WARN": 1}.get(s.get("pass_level", ""), 0)
    rs = float(s.get("rs_20d", 1.0) or 1.0)
    vr = min(float(s.get("volume_ratio", 0) or 0), 5.0)
    pat = 1.0 if s.get("pattern_type", "none") != "none" else 0.0
    return lvl * 10 + rs * 2 + vr + pat


def bear_debate_node(state: ScanState) -> dict:
    logger.info("Node 6: Bear/Bull debate...")
    debated = []
    all_recent_news = state.get("raw_news", [])  # 全部未篩選新聞，title+link+content
    logger.info(f"  passing {len(all_recent_news)} unfiltered articles to LLM for relevance judgment")

    # ── Top-N 硬上限:量化分數排序後只取前 N 檔做辯論 ──────────────────────────
    # 這是讓「大 universe」可行的關鍵:不管篩出幾百檔,LLM 成本鎖死在 N 檔。
    candidates = state["with_pattern"]
    max_n = cfg("debate.max_candidates", 25)
    if len(candidates) > max_n:
        candidates = sorted(candidates, key=_debate_score, reverse=True)[:max_n]
        logger.info(f"  Top-N 上限:{len(state['with_pattern'])} → {max_n} 檔進入辯論(量化分數排序)")

    as_of = state.get("as_of")
    hist_date = date.fromisoformat(as_of) if as_of else None   # as_of → 辯論用 ≤該日新聞/技術/反彈
    for stock in candidates:
        logger.info(f"  debating {stock['code']} {stock['name']}...")
        try:
            result = run_debate(stock, all_recent_news, historical_date=hist_date)
            # 最終 verdict = 取 debate 和量化評級中較悲觀的那個
            # PASS < WARN < REJECT
            _order = {"PASS": 0, "WARN": 1, "REJECT": 2}
            quant_level = stock.get("pass_level", "WARN")
            debate_verdict = result.verdict
            final_verdict = debate_verdict if _order.get(debate_verdict, 1) >= _order.get(quant_level, 1) else quant_level
            merged = {**stock,
                      "bear_score": result.bear_score,
                      "bull_score": result.bull_score,
                      "evidence_level": result.evidence_level,
                      "top_risks": result.top_risks,
                      "catalysts": result.catalysts,
                      "verdict": final_verdict,
                      "bear_reason": result.bear_reason,
                      "bull_reason": result.bull_reason,
                      "negative_news": result.negative_news,
                      "positive_news": result.positive_news,
                      "predicted_direction": result.predicted_direction,
                      "predicted_low_pct": result.predicted_low_pct,
                      "predicted_high_pct": result.predicted_high_pct,
                      "predicted_center_pct": result.predicted_center_pct,
                      "prediction_confidence": result.prediction_confidence,
                      "prediction_key_factor": result.prediction_key_factor,
                      }
            debated.append(merged)
            logger.info(f"    → quant={quant_level} debate={debate_verdict} final={final_verdict} bear={result.bear_score}")
        except Exception as exc:
            logger.error(f"  debate failed for {stock['code']}: {exc}")
            fallback_verdict = stock.get("pass_level", "WARN")
            merged = {**stock, "bear_score": 0, "bull_score": 0,
                      "verdict": fallback_verdict,
                      "bear_reason": f"[debate_error] {str(exc)[:120]}",
                      "bull_reason": "debate unavailable",
                      "evidence_level": "", "top_risks": [], "catalysts": [],
                      "negative_news": [], "positive_news": [],
                      "predicted_direction": "neutral", "predicted_low_pct": 0.0,
                      "predicted_high_pct": 0.0, "predicted_center_pct": 0.0,
                      "prediction_confidence": 0.0, "prediction_key_factor": ""}
            debated.append(merged)

    return {"debated": debated}


# ── Node 7：generate_report ─────────────────────────────────────────────────

def generate_report_node(state: ScanState) -> dict:
    logger.info("Node 7: generating report...")
    from tw_stock_agent.report import build_report

    decision = state.get("as_of") or datetime.now().strftime("%Y-%m-%d")
    trade_date = state.get("trade_date") or decision   # signal_log/帳本用交易日
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{decision}.md"       # 報告檔以決策日命名

    report_md = build_report(state["debated"], decision,
                             screened=state.get("screened", []))
    report_path.write_text(report_md, encoding="utf-8")

    # signal_log 以「交易日」命名(先清掉該交易日舊紀錄,確保可重複執行不重複)
    remove_signals(trade_date)
    append_signals(state["debated"], trade_date)

    logger.info(f"  report saved → {report_path}(決策日)｜signal_log 標 {trade_date}(交易日)")
    return {"report_path": str(report_path)}
