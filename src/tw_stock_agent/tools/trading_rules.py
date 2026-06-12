"""設定檔驅動的交易規則引擎。

所有規則寫在 configs/default.yaml 的 `trading:` 區塊,本引擎每次讀取 →
你改 yaml 就改行為(動態組合訊號、進場/部位/出場規則),不用碰程式。

供回測(run_backtest_60d)與實盤(live_portfolio)共用,確保兩邊規則一致。

核心函式:
  combine_edge(llm_edge, rebound_edge)         動態組合多訊號 → 單一 edge
  entry_decision(...)                          實際面進場:隔日開盤/追高上限/漲停買不到/滑價/剩餘edge
  position_size(equity, avg_turnover)          部位上限:佔權益% / 絕對 / 流動性%
  exit_decision(entry_price, days, now, edge)  出場:持有天數/停利/停損/訊號翻空
"""
from __future__ import annotations

from tw_stock_agent.config import cfg


def _t(key: str, default):
    return cfg(f"trading.{key}", default)


# ── 訊號動態組合 ──────────────────────────────────────────────────────────────
def combine_edge(llm_edge: float = 0.0, rebound_edge: float = 0.0) -> float:
    """依 yaml 設定把多個訊號的 edge 組成一個。回傳組合後 edge(期望%單位)。"""
    lw = float(_t("signals.llm_weight", 1.0))
    rw = float(_t("signals.rebound_weight", 1.0))
    mode = _t("signals.combine", "sum")
    le, re = llm_edge * lw, rebound_edge * rw
    if mode == "llm_only":
        return le
    if mode == "rebound_only":
        return re
    if mode == "max":
        return max(le, re)
    return le + re                       # sum(預設)


def passes_min_edge(edge: float) -> bool:
    return edge >= float(_t("signals.min_edge", 0.0))


# ── 進場(實際面)──────────────────────────────────────────────────────────────
def entry_decision(ref_close: float, target_pct: float, conf: float,
                   next_open: float, prev_close: float, avg_turnover: float,
                   equity: float) -> dict:
    """決定『實際能不能買、買在哪、扣掉 gap/成本後還剩多少 edge』。

    ref_close   : 決策基準(訊號日收盤)
    target_pct  : 預測目標漲幅(小數,如 0.037)
    conf        : 信心 0-1
    next_open   : 隔日開盤價(timing=next_open 用)
    prev_close  : 進場日前一交易日收盤(判漲停用)
    avg_turnover: 日均成交額(流動性上限用)
    equity      : 目前權益(部位上限用)
    回傳 dict: buy, price, realistic_edge, size_cap, gap, reason
    """
    timing = _t("entry.timing", "next_open")
    price = next_open if timing == "next_open" else ref_close
    if price <= 0:
        return {"buy": False, "reason": "無進場價"}

    # 鎖漲停買不到
    if _t("entry.skip_limit_up", True) and prev_close > 0 and price / prev_close - 1 >= 0.094:
        return {"buy": False, "reason": "鎖漲停買不到"}

    gap = price / ref_close - 1 if ref_close > 0 else 0.0
    max_chase = float(_t("entry.max_chase_pct", 0.03))
    if gap > max_chase:
        return {"buy": False, "reason": f"追高 gap {gap*100:.1f}% 超過上限 {max_chase*100:.0f}%"}

    fill = price * (1 + float(_t("entry.slippage_pct", 0.001)))
    gap_after = fill / ref_close - 1 if ref_close > 0 else 0.0
    remaining = target_pct - gap_after                       # gap 吃掉的空間要扣
    realistic_edge = conf * remaining - float(_t("cost.round_trip_pct", 0.005))
    if realistic_edge <= 0:
        return {"buy": False, "reason": "扣 gap+成本後無 edge", "gap": gap_after}

    return {"buy": True, "price": fill, "realistic_edge": realistic_edge,
            "size_cap": position_size(equity, avg_turnover), "gap": gap_after, "reason": "ok"}


def position_size(equity: float, avg_turnover: float) -> float:
    """單一部位金額上限:min(佔權益%, 絕對, 流動性%)。0 = 該項不限。"""
    cap = equity
    fr = float(_t("sizing.pos_cap_frac", 0.0))
    ab = float(_t("sizing.pos_cap_abs", 0))
    liq = float(_t("sizing.max_pct_of_turnover", 0.0))
    if fr > 0:
        cap = min(cap, fr * equity)
    if ab > 0:
        cap = min(cap, ab)
    if liq > 0 and avg_turnover > 0:
        cap = min(cap, liq * avg_turnover)
    return cap


def daily_add_cap() -> float:
    c = float(_t("sizing.daily_add_cap", 0))
    return c if c > 0 else float("inf")


# ── 出場 ──────────────────────────────────────────────────────────────────────
def exit_decision(entry_price: float, days_held: int, current_price: float,
                  edge_now: float = 0.0) -> tuple[bool, str]:
    """回傳 (要不要賣, 原因)。"""
    pnl = current_price / entry_price - 1 if entry_price > 0 else 0.0
    if days_held >= int(_t("exit.max_hold_days", 5)):
        return True, "持有到期"
    tp = float(_t("exit.take_profit_pct", 0))
    sl = float(_t("exit.stop_loss_pct", 0))
    if tp > 0 and pnl >= tp:
        return True, f"停利 {pnl*100:+.1f}%"
    if sl < 0 and pnl <= sl:
        return True, f"停損 {pnl*100:+.1f}%"
    if _t("exit.exit_on_reverse", True) and edge_now <= 0:
        return True, "訊號翻空"
    return False, "續抱"


def capital_cfg() -> dict:
    return {
        "initial": float(_t("capital.initial", 15000)),
        "daily_budget": float(_t("capital.daily_budget", 1000)),
        "max_contribution": float(_t("capital.max_contribution", 50000)),
    }