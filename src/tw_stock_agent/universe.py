"""Point-in-time 掃描池:用「截止 as_of(含)」的流動性排名取前 N(0050 代理)。

零未來偏誤:流動性只用 ≤as_of 的資料算,所以同一支程式跑 6/9 跟跑 2022 都拿到
「當時」的前 N 大,不會用到未來才變大的股票。as_of=None → 用最新(live)。
候選池 = base_universe.json(長期大型/高流動性股)。
"""
from __future__ import annotations

import json

from tw_stock_agent.config import DATA_DIR


def _load_json(name: str) -> dict:
    f = DATA_DIR / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def universe_as_of(as_of: str | None, top_n: int = 0, lookback: int = 60) -> list[dict]:
    """回傳 as_of 當天的候選股 dict（與 pipeline candidates 同格式）。

    top_n <= 0 → 回傳「全部 base_universe」（對齊 112 檔回測:讓 Node4 量化篩選去淘汰）。
    top_n > 0  → point-in-time:用截止 as_of 最近 lookback 日的日均成交額排名取前 N
                 （0050 代理,零未來偏誤）。
    """
    base = _load_json("base_universe.json")
    idx = _load_json("tw_stock_index.json")

    turn_map: dict[str, float] = {}
    if top_n and top_n > 0:
        # point-in-time 流動性排名
        from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
        scored: list[tuple[str, float]] = []
        for code in base:
            try:
                oh = get_daily_ohlcv(code)
            except Exception:
                continue
            days = sorted(d for d in oh if (as_of is None or d <= as_of))[-lookback:]
            if len(days) < 20:
                continue
            turns = [(oh[d].get("amount") or oh[d].get("close", 0) * oh[d].get("volume", 0))
                     for d in days]
            scored.append((code, sum(turns) / len(turns) if turns else 0.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_n]
        turn_map = dict(scored)
        codes = [c for c, _ in scored]
    else:
        codes = list(base)   # 全部 base_universe

    out: list[dict] = []
    for code in codes:
        meta = base.get(code, {})
        info = idx.get(code, {})
        suffix = ".TW" if meta.get("market", "TWSE") == "TWSE" else ".TWO"
        out.append({
            "code": code,
            "name": meta.get("name") or info.get("name", code),
            "yf_ticker": info.get("yf_ticker") or f"{code}{suffix}",
            "nickname": meta.get("name", code),
            "bfs_depth": 0,
            "via_path": "",
            "source_news": "[base_universe]",
            "_avg_turnover": turn_map.get(code, 0.0),
        })
    return out