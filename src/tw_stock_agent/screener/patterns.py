"""K 線型態辨識：縮量整理後放量突破 + 其他型態。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tw_stock_agent.config import cfg

try:
    import pandas_ta_classic as ta  # type: ignore
except ImportError:
    ta = None  # type: ignore


@dataclass
class PatternResult:
    code: str
    has_pattern: bool
    pattern_type: str        # "volume_breakout" | "golden_cross" | "none"
    pattern_detail: str      # 人讀說明
    rsi_14: float = 0.0
    macd_hist: float = 0.0


def _safe_ta(df: pd.DataFrame) -> dict:
    """安全地計算 RSI + MACD，若 pandas-ta 不可用則回傳空。"""
    if ta is None:
        return {}
    try:
        rsi = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"])
        rsi_val = float(rsi.iloc[-1]) if rsi is not None and len(rsi) > 0 else 0.0
        macd_hist = 0.0
        if macd is not None and not macd.empty:
            # MACDh_ 欄位名稱
            hist_col = [c for c in macd.columns if "MACDh" in c or "Histogram" in c]
            if hist_col:
                macd_hist = float(macd[hist_col[0]].iloc[-1])
        return {"rsi_14": rsi_val, "macd_hist": macd_hist}
    except Exception:
        return {}


def check_volume_breakout(df: pd.DataFrame) -> tuple[bool, str]:
    """縮量整理後放量突破：

    條件 1：近 3-5 日量縮（低於 20 日均量 70%）
    條件 2：今日放量（超過 20 日均量 150%）
    條件 3：今日收盤突破近 10 日最高收盤
    """
    vol = df["Volume"]
    close = df["Close"]
    ma20vol = vol.rolling(20).mean()

    if ma20vol.iloc[-1] == 0 or len(df) < 25:
        return False, "insufficient_data"

    consol_ratio = cfg("screener.consolidation_ratio", 0.7)
    consol_days = cfg("screener.consolidation_days", 3)
    vol_surge_ratio = cfg("screener.min_volume_ratio", 1.5)

    consolidating = (vol.iloc[-6:-1] < ma20vol.iloc[-6:-1] * consol_ratio).sum() >= consol_days
    vol_surge = vol.iloc[-1] > ma20vol.iloc[-1] * vol_surge_ratio
    price_break = close.iloc[-1] > close.iloc[-11:-1].max()

    if consolidating and vol_surge and price_break:
        detail = (f"縮量{consol_days}日後今日量比{vol.iloc[-1]/ma20vol.iloc[-1]:.1f}x，"
                  f"收盤{close.iloc[-1]:.1f}突破10日高點")
        return True, detail
    reasons = []
    if not consolidating:
        reasons.append("縮量不足")
    if not vol_surge:
        reasons.append(f"量比{vol.iloc[-1]/ma20vol.iloc[-1]:.1f}x未達{vol_surge_ratio}x")
    if not price_break:
        reasons.append("未突破10日高點")
    return False, " | ".join(reasons)


def check_golden_cross(df: pd.DataFrame) -> tuple[bool, str]:
    """MA5 剛黃金交叉 MA20（近 3 天內發生）。"""
    close = df["Close"]
    if len(close) < 22:
        return False, "insufficient_data"
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    # 今天 ma5 > ma20，3天前 ma5 < ma20
    if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-4] < ma20.iloc[-4]:
        return True, f"MA5 {ma5.iloc[-1]:.1f} 剛上穿 MA20 {ma20.iloc[-1]:.1f}"
    return False, "no_golden_cross"


def check_strong_daily_candle(df: pd.DataFrame) -> tuple[bool, str]:
    """今日單根強勢陽線：實體 > 收盤價 1.5%，收盤在日高的上半段。"""
    if len(df) < 2:
        return False, "insufficient_data"
    o, h, l, c = (df[col].iloc[-1] for col in ("Open", "High", "Low", "Close"))
    body = c - o
    candle_range = h - l
    if candle_range == 0:
        return False, "zero_range"
    body_pct = body / c
    close_pos = (c - l) / candle_range   # 收盤在日 range 的哪個位置（1=日高）
    vol = df["Volume"]
    vol_ratio = float(vol.iloc[-1] / vol.rolling(5).mean().iloc[-1]) if vol.rolling(5).mean().iloc[-1] > 0 else 1.0
    if body_pct >= 0.015 and close_pos >= 0.6 and body > 0:
        detail = f"陽線實體 {body_pct*100:.1f}%，收盤位於日 range {close_pos*100:.0f}%，量比 {vol_ratio:.1f}x"
        return True, detail
    return False, f"body={body_pct*100:.1f}%<1.5% or close_pos={close_pos:.2f}<0.6"


def check_gap_up(df: pd.DataFrame) -> tuple[bool, str]:
    """今日跳空開高：開盤 > 昨日收盤 0.8% 以上。"""
    if len(df) < 2:
        return False, "insufficient_data"
    prev_close = float(df["Close"].iloc[-2])
    today_open = float(df["Open"].iloc[-1])
    today_close = float(df["Close"].iloc[-1])
    gap_pct = (today_open / prev_close - 1)
    if gap_pct >= 0.008 and today_close >= today_open:  # 跳空且未回補
        return True, f"跳空開高 {gap_pct*100:.1f}%，收盤維持在開盤上方"
    return False, f"gap={gap_pct*100:.1f}%<0.8% or filled"


def analyze_pattern(candidate: dict, df: pd.DataFrame) -> PatternResult:
    """對給定 OHLCV 跑所有型態，回傳最強的一個。

    優先順序：放量突破 > 黃金交叉 > 跳空缺口 > 強勢陽線（單日訊號）
    """
    code = candidate.get("code", "")
    ta_vals = _safe_ta(df)

    # 1. 放量突破（最強訊號）
    ok_vb, detail_vb = check_volume_breakout(df)
    if ok_vb:
        return PatternResult(code=code, has_pattern=True,
                             pattern_type="volume_breakout", pattern_detail=detail_vb,
                             rsi_14=ta_vals.get("rsi_14", 0.0),
                             macd_hist=ta_vals.get("macd_hist", 0.0))

    # 2. 黃金交叉（中期趨勢轉強）
    ok_gc, detail_gc = check_golden_cross(df)
    if ok_gc:
        return PatternResult(code=code, has_pattern=True,
                             pattern_type="golden_cross", pattern_detail=detail_gc,
                             rsi_14=ta_vals.get("rsi_14", 0.0),
                             macd_hist=ta_vals.get("macd_hist", 0.0))

    # 3. 跳空缺口（單日訊號）
    ok_gap, detail_gap = check_gap_up(df)
    if ok_gap:
        return PatternResult(code=code, has_pattern=True,
                             pattern_type="gap_up", pattern_detail=detail_gap,
                             rsi_14=ta_vals.get("rsi_14", 0.0),
                             macd_hist=ta_vals.get("macd_hist", 0.0))

    # 4. 強勢陽線（單日訊號）
    ok_sd, detail_sd = check_strong_daily_candle(df)
    if ok_sd:
        return PatternResult(code=code, has_pattern=True,
                             pattern_type="strong_candle", pattern_detail=detail_sd,
                             rsi_14=ta_vals.get("rsi_14", 0.0),
                             macd_hist=ta_vals.get("macd_hist", 0.0))

    return PatternResult(code=code, has_pattern=False,
                         pattern_type="none", pattern_detail="no pattern today",
                         rsi_14=ta_vals.get("rsi_14", 0.0),
                         macd_hist=ta_vals.get("macd_hist", 0.0))
