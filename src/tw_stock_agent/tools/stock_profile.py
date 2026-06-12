"""個股歷史型態表現查詢工具。

從 stock_profile.db 取得過去每支股票出現特定 K 線型態後的統計，
供 Bear/Bull/Predict prompt 注入歷史依據。

回測模式（as_of_date 有值）：從 backtest_one_month.csv 即時計算截止日前的統計，
避免 look-ahead bias。
"""
from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

from tw_stock_agent.config import DATA_DIR

DB_PATH  = DATA_DIR / "stock_profile.db"
CSV_PATH = DATA_DIR / "backtest_one_month.csv"

# ── CSV 一次載入快取（記憶體內，避免每筆回測重複讀檔）────────────────────────
_csv_cache: list[dict] | None = None


def _load_csv() -> list[dict]:
    global _csv_cache
    if _csv_cache is None:
        if not CSV_PATH.exists():
            _csv_cache = []
        else:
            with CSV_PATH.open(encoding="utf-8") as f:
                _csv_cache = list(csv.DictReader(f))
    return _csv_cache


def _connect() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH)


# ── 回測專用：從 CSV 即時計算截止日前的統計 ───────────────────────────────────

def _profile_from_csv(ticker: str, current_pattern: str, as_of_date: date) -> str:
    rows = _load_csv()
    cutoff = as_of_date.isoformat()

    # 只取截止日前、return_1d 已知的訊號
    ticker_rows = [
        r for r in rows
        if r.get("ticker") == ticker
        and r.get("date", "") < cutoff
        and r.get("return_1d", "").strip() not in ("", "?")
    ]
    if not ticker_rows:
        return ""

    # 整體統計
    returns = []
    for r in ticker_rows:
        try:
            returns.append(float(r["return_1d"]))
        except (ValueError, TypeError):
            pass
    if not returns:
        return ""

    total_n  = len(returns)
    overall_wr = sum(1 for x in returns if x > 0.2) / total_n
    avg_ret  = sum(returns) / total_n

    # Alpha（若有）
    alphas = []
    for r in ticker_rows:
        try:
            a = r.get("alpha_1d", "").strip()
            if a and a != "?":
                alphas.append(float(a))
        except (ValueError, TypeError):
            pass
    avg_alpha: float | None = (sum(alphas) / len(alphas)) if alphas else None

    # 各型態分組
    pattern_groups: dict[str, list[float]] = defaultdict(list)
    for r, ret in zip(ticker_rows, returns):
        pattern_groups[r.get("pattern_type", "none")].append(ret)

    # 組合輸出
    alpha_str = f"，均 Alpha {avg_alpha:+.2f}%" if avg_alpha is not None else ""
    lines = [
        "【個股歷史型態表現（量化回測統計，截止日前資料）】",
        f"整體：{total_n} 次訊號，勝率 {overall_wr*100:.0f}%，均漲 {avg_ret:+.2f}%{alpha_str}",
        "",
    ]

    for pat, rets in sorted(pattern_groups.items(), key=lambda x: -len(x[1])):
        n     = len(rets)
        avg_r = sum(rets) / n
        std_r = math.sqrt(sum((x - avg_r) ** 2 for x in rets) / n) if n > 1 else 0.0
        wr    = sum(1 for x in rets if x > 0.2) / n
        tag   = " ← 今日型態" if pat == current_pattern else ""

        # 樣本數警示
        if n < 5:
            confidence = "（樣本不足，僅供參考）"
        elif n < 10:
            confidence = "（弱參考）"
        else:
            confidence = ""

        label = ""
        if n >= 3:
            if wr < 0.4 or avg_r < -1.0:
                label = " [歷史偏空]"
            elif wr >= 0.65 and avg_r > 0.8:
                label = " [歷史偏多]"

        lines.append(
            f"  {pat}：{n} 次，勝率 {wr*100:.0f}%，"
            f"均 {avg_r:+.2f}%（σ={std_r:.2f}%）{label}{confidence}{tag}"
        )

    lines.append("")
    lines.append("※ 以上為統計傾向，非預測保證，請結合新聞與籌碼綜合判斷。")
    return "\n".join(lines)


# ── 主查詢（DB 版，live 模式用）───────────────────────────────────────────────

def get_pattern_profile(
    ticker: str,
    current_pattern: str = "",
    as_of_date: Optional[date] = None,
) -> str:
    """回傳個股歷史型態表現的 LLM 可讀字串。

    as_of_date 有值時（回測模式）：從 CSV 計算截止日前統計，避免 look-ahead bias。
    as_of_date 為 None（live 模式）：從預建 DB 查詢（較快）。

    若無資料，回傳空字串（不影響原有流程）。
    """
    if as_of_date is not None:
        return _profile_from_csv(ticker, current_pattern, as_of_date)

    # ── Live 模式：查預建 DB ─────────────────────────────────────────────────
    conn = _connect()
    if conn is None:
        return ""

    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT total_signals, overall_win_rate, avg_return, avg_alpha
            FROM ticker_overview WHERE ticker = ?
        """, (ticker,))
        row = cur.fetchone()
        if row is None:
            return ""

        total_n, overall_wr, avg_ret, avg_alpha = row

        cur.execute("""
            SELECT pattern_type, n, avg_return, std_return, win_rate, avg_alpha
            FROM pattern_stats WHERE ticker = ?
            ORDER BY n DESC
        """, (ticker,))
        patterns = cur.fetchall()

        if not patterns:
            return ""

        alpha_str = f"，均 Alpha {avg_alpha:+.2f}%" if avg_alpha is not None else ""
        lines = [
            "【個股歷史型態表現（量化回測統計）】",
            f"整體：{total_n} 次訊號，勝率 {overall_wr*100:.0f}%，均漲 {avg_ret:+.2f}%{alpha_str}",
            "",
        ]

        for pat, n, avg_r, std_r, wr, avg_a in patterns:
            tag   = " ← 今日型態" if pat == current_pattern else ""
            a_str = f"，Alpha {avg_a:+.2f}%" if avg_a is not None else ""

            label = ""
            if n >= 3:
                if wr < 0.4 or avg_r < -1.0:
                    label = " ⚠️ 歷史偏空"
                elif wr >= 0.65 and avg_r > 0.8:
                    label = " ✅ 歷史偏多"

            lines.append(
                f"  {pat}：{n} 次，勝率 {wr*100:.0f}%，"
                f"均 {avg_r:+.2f}%（σ={std_r:.2f}%）{a_str}{label}{tag}"
            )

        lines.append("")
        lines.append("※ 以上為統計傾向，非預測保證，請結合新聞與籌碼綜合判斷。")
        return "\n".join(lines)

    finally:
        conn.close()


def profile_exists(ticker: str) -> bool:
    """快速檢查該股票是否有歷史資料。"""
    conn = _connect()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM ticker_overview WHERE ticker = ?", (ticker,))
        return cur.fetchone() is not None
    finally:
        conn.close()
