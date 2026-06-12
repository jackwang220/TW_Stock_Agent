"""LangGraph ScanState 定義。"""
from __future__ import annotations

from typing import TypedDict


class ScanState(TypedDict, total=False):
    # 決策日(資料截止日,ISO)：報告檔以此命名；篩選/辯論用 ≤此日資料
    as_of: str
    # 交易日(決策日的下一交易日,ISO)：signal_log / 帳本以此命名
    trade_date: str

    # Node 1 → Node 2
    raw_news: list[dict]        # Node 2 後清空（只留 title+link），避免 checkpoint 膨脹

    # Node 2 → Node 3
    candidates: list[dict]      # {code, name, yf_ticker, source_news, bfs_depth=0}

    # Node 3 → Node 4
    bfs_expanded: list[dict]    # 加入供應鏈受益股後（bfs_depth 1-3）

    # Node 4 → Node 5
    screened: list[dict]        # 通過量化篩選（含 volume_ratio, ma5_gt_ma20, rs_20d）

    # Node 5 → Node 6
    with_pattern: list[dict]    # 有 K 線型態支撐（含 pattern_type, pattern_detail）

    # Node 6 → Node 7
    debated: list[dict]         # Bear 辯論後（含 bear_score, verdict, top_risks）

    # Node 7 output
    report_path: str
