"""LangGraph StateGraph 組裝。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from tw_stock_agent.config import CHECKPOINT_DB
from tw_stock_agent.pipeline.nodes import (
    bear_debate_node,
    extract_entities_node,
    fetch_news_node,
    generate_report_node,
    pattern_match_node,
    quantitative_screen_node,
    supply_chain_bfs_node,
)
from tw_stock_agent.pipeline.state import ScanState


def build_graph():
    """建立並 compile LangGraph，附 SQLite checkpoint。"""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    g = StateGraph(ScanState)
    g.add_node("fetch_news",    fetch_news_node)
    g.add_node("extract",       extract_entities_node)
    g.add_node("bfs",           supply_chain_bfs_node)
    g.add_node("screen",        quantitative_screen_node)
    g.add_node("pattern",       pattern_match_node)
    g.add_node("bear_debate",   bear_debate_node)
    g.add_node("output",        generate_report_node)

    g.set_entry_point("fetch_news")
    g.add_edge("fetch_news", "extract")
    g.add_edge("extract",    "bfs")
    g.add_edge("bfs",        "screen")
    # 若無候選股直接跳輸出
    g.add_conditional_edges(
        "screen",
        lambda s: "pattern" if any(
            x.get("_screen_passed") for x in s.get("screened", [])
        ) else "output",
    )
    g.add_edge("pattern",    "bear_debate")
    g.add_edge("bear_debate","output")
    g.add_edge("output",     END)

    return g.compile(checkpointer=checkpointer)


def make_config(as_of: str | None = None) -> dict:
    """thread_id 用掃描日期，確保不跨日污染 checkpoint（as_of 模式用 as_of 日）。"""
    tid = as_of or datetime.now().strftime("%Y-%m-%d")
    return {"configurable": {"thread_id": tid}}


def run_daily_scan(as_of: str | None = None) -> str:
    """執行一次完整掃描（一套系統:live 與歷史重生/回測同一條路）。

    日期自動解析(market_calendar.resolve_session):
      - as_of=None → 用現在時間自動判定「決策日」(盤後算今天、半夜/盤前算昨天最近收盤),
        交易日=決策日的下一交易日。所以晚上11點(6/9)跟半夜1點(6/10)都解析成 決策6/9→交易6/10。
      - as_of="2026-06-09" → 手動指定決策日=6/9,交易日=6/10。
    報告檔以「決策日」命名;signal_log/帳本以「交易日」命名。掃描池=base_universe,
    篩選/型態/辯論/反彈/報告全部以 ≤決策日 的資料進行(零洩漏)。
    """
    from tw_stock_agent.market_calendar import resolve_session

    decision, trade = resolve_session(as_of)
    app = build_graph()
    config = make_config(decision)
    initial: ScanState = {
        "as_of": decision,          # 決策日(資料截止日)
        "trade_date": trade,        # 交易日(signal_log/帳本用)
        "raw_news": [],
        "candidates": [],
        "bfs_expanded": [],
        "screened": [],
        "with_pattern": [],
        "debated": [],
        "report_path": "",
    }
    result = app.invoke(initial, config=config)
    return result.get("report_path", "")
