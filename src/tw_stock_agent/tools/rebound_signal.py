"""跌深反彈訊號(第一梯隊 confirmed edge,劑量分級)。

權重直接來自 edge_scanner v2 的『樣本外(2022-2026)實證』——扣成本淨報酬 + 勝率,
不是憑感覺調的。核心規律:大型/高流動性股跌越深 → 反彈越強(劑量反應)。

leak-safe:只用 ≤ 當日的收盤序列。
輸出 rebound_edge(期望淨5日報酬)可直接跟 LLM 的 edge(center%×conf)比較/相加當部位權重。

用法:
    from tw_stock_agent.tools.rebound_signal import rebound_signal
    sig = rebound_signal(closes, avg_turnover)   # closes 升冪,最後一筆=今日
"""
from __future__ import annotations

import numpy as np

# 流動性分層(日均成交額 TWD)
LARGE_TURN = 5e8
MID_TURN = 5e7

# 權重表 = edge_scanner v2 OOS 實證 (勝率, 扣成本淨5日報酬)
#   large = 大型股組;mid = 較弱(用「全部」條件數字打折,涵蓋中型)
_LARGE = {"d20": (0.78, 0.093), "d12": (0.64, 0.037), "bias10": (0.63, 0.021), "d8": (0.59, 0.017)}
_MID   = {"d20": (0.63, 0.058), "d12": (0.59, 0.025), "bias10": (0.58, 0.014), "d8": (0.56, 0.009)}

_LABEL = {"d20": "極深跌(≤-20%)", "d12": "深跌(≤-12%)", "d8": "中跌(≤-8%)", "bias10": "乖離過大(≤-10%)"}


def _tier(avg_turnover: float) -> str:
    if avg_turnover >= LARGE_TURN:
        return "large"
    if avg_turnover >= MID_TURN:
        return "mid"
    return "small"


def rebound_signal(closes: list[float], avg_turnover: float) -> dict:
    """closes 升冪(最後一筆=今日)。回傳跌深反彈訊號。

    回傳:
      fired: bool                是否觸發
      score: 0-1                 信心(由勝率換算,給排序/篩選)
      rebound_edge: float        期望淨5日報酬(給部位權重,與 LLM edge 同單位)
      win_rate: float            歷史勝率
      depth: str                 觸發的劑量等級
      tier: large/mid/small
      summary: str               LLM 可讀
    """
    out = {"fired": False, "score": 0.0, "rebound_edge": 0.0, "win_rate": 0.0,
           "depth": "", "tier": "", "summary": ""}
    if len(closes) < 25 or closes[-6] <= 0:
        return out
    tier = _tier(avg_turnover)
    out["tier"] = tier
    from tw_stock_agent.config import cfg
    if avg_turnover < cfg("trading.signals.rebound_min_turnover", 0):
        return out                # 成交額門檻(dose-response:反彈edge集中在高週轉大型股)
    if tier == "small":           # 小型股無 edge(會被主力操控/買不到)
        return out
    if tier == "mid":
        from tw_stock_agent.config import cfg
        if cfg("trading.signals.rebound_large_only", False):
            return out            # 設定為「只大型股」時,中型不觸發

    ret5 = closes[-1] / closes[-6] - 1
    ma20 = float(np.mean(closes[-20:]))
    bias = closes[-1] / ma20 - 1 if ma20 > 0 else 0.0

    table = _LARGE if tier == "large" else _MID
    from tw_stock_agent.config import cfg
    min_depth = cfg("trading.signals.rebound_min_depth", "d8")
    rank = {"d8": 1, "d12": 2, "d20": 3}
    drop = None
    if ret5 <= -0.20:
        drop = "d20"
    elif ret5 <= -0.12:
        drop = "d12"
    elif ret5 <= -0.08:
        drop = "d8"
    cands = []
    if drop and rank[drop] >= rank.get(min_depth, 1):   # 低於最淺門檻的跌幅不觸發
        cands.append(drop)
    if bias <= -0.10:
        cands.append("bias10")
    if not cands:
        return out
    # 用期望淨報酬最大的那個劑量
    depth = max(cands, key=lambda b: table[b][1])
    win, edge = table[depth]
    # 註:試過「新鮮瀑布(末日跌停/≤-5%)加權」,A/B(2年264筆)反而從 +35.9%→+25.8% 變差
    #     原因:新鮮瀑布的股還在猛殺,-6% 停損會先把它掃掉、錯過反彈。已撤回。

    out.update({
        "fired": True,
        "score": float(min(1.0, max(0.0, (win - 0.5) / 0.3))),  # 0.5→0, 0.8→1
        "rebound_edge": float(edge),
        "win_rate": float(win),
        "depth": depth,
        "summary": (f"【跌深反彈訊號】{tier}股、近5日{ret5*100:.1f}%、乖離{bias*100:.1f}%"
                    f" → {_LABEL[depth]}:歷史勝率{win*100:.0f}%、期望淨5日{edge*100:+.1f}%"
                    f"（樣本外實證;短線2-4天、隔日開盤進、跌停可能買不到要扛續跌）"),
    })
    return out


_TURN: dict | None = None


def avg_turnover_of(code: str) -> float:
    """從 base_universe.json 取該股日均成交額(沒有=0=不在基本盤)。"""
    global _TURN
    if _TURN is None:
        import json
        from tw_stock_agent.config import DATA_DIR
        f = DATA_DIR / "base_universe.json"
        _TURN = ({c: v.get("avg_turnover", 0) for c, v in
                  json.loads(f.read_text(encoding="utf-8")).items()} if f.exists() else {})
    return _TURN.get(code, 0.0)


def rebound_signal_for_code(code: str, avg_turnover: float, as_of: str | None = None) -> dict:
    """從 FinMind 快取讀某股收盤,算跌深反彈訊號(as_of 為 None=最新)。"""
    from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
    oh = get_daily_ohlcv(code)
    ds = sorted(d for d in oh if (as_of is None or d <= as_of))
    closes = [oh[d]["close"] for d in ds]
    return rebound_signal(closes, avg_turnover)