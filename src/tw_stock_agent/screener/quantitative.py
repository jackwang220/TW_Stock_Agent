"""量化篩選：成交量突破 + 均線 + 相對強度。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from tw_stock_agent.config import cfg
from tw_stock_agent.data.price_data import get_index_return, get_ohlcv, validate_price_data


@dataclass
class ScreenResult:
    code: str
    name: str
    yf_ticker: str
    passed: bool                 # True = PASS or WARN（進入後續節點）
    pass_level: str = "REJECT"   # "PASS" | "WARN" | "REJECT"
    reason: str = ""             # 通過或被淘汰的原因
    close_price: float = 0.0
    volume_ratio: float = 0.0    # 今日量 / 20日均量
    ma5_gt_ma20: bool = False
    rs_20d: float = 0.0          # (股票20日漲幅) / (大盤20日漲幅)
    weeks_52_warn: bool = False  # 接近52週高點時標 WARN
    extra: dict = field(default_factory=dict)


def _volume_ratio(df: pd.DataFrame) -> float:
    vol = df["Volume"]
    ma20 = vol.rolling(20).mean()
    if ma20.iloc[-1] == 0:
        return 0.0
    return float(vol.iloc[-1] / ma20.iloc[-1])


def _ma5_gt_ma20(df: pd.DataFrame) -> bool:
    close = df["Close"]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    return bool(ma5 > ma20)


def _rs_20d(df: pd.DataFrame, index_return_20d: float) -> float:
    """相對強度 = (1 + 股票20日報酬) / (1 + 大盤20日報酬)。
    >1 代表跑贏大盤，<1 代表跑輸，無論漲跌都有意義。
    """
    if len(df) < 21:
        return 1.0
    stock_ret = float(df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1)
    denom = 1.0 + index_return_20d
    if abs(denom) < 1e-6:
        return 1.0
    return (1.0 + stock_ret) / denom


def _weeks_52_position(df: pd.DataFrame) -> float:
    """收盤價在52週區間的相對位置（1.0 = 52週高點）。"""
    if len(df) < 252:
        lookback = df
    else:
        lookback = df.iloc[-252:]
    high52 = lookback["High"].max()
    if high52 == 0:
        return 0.0
    return float(df["Close"].iloc[-1] / high52)


def screen_one(
    candidate: dict,
    today: date | None = None,
    as_of_date: str | None = None,
    _index_return: float | None = None,
) -> ScreenResult:
    """對單支股票跑完整量化篩選。

    Args:
        candidate: 含 code, name, yf_ticker 的 dict
        today: 預期最新資料日期（None 則跳過 freshness check）
        as_of_date: 回測模式下的截止日期（ISO string）
        _index_return: 預先計算的大盤 20 日報酬（避免每支股票都抓一次）
    """
    code = candidate["code"]
    name = candidate.get("name", code)
    yf_ticker = candidate.get("yf_ticker", f"{code}.TW")

    df = get_ohlcv(yf_ticker, as_of_date=as_of_date)

    if df.empty:
        return ScreenResult(code=code, name=name, yf_ticker=yf_ticker,
                            passed=False, pass_level="REJECT", reason="no_price_data")

    if today:
        ok, msg = validate_price_data(df, today)
        if not ok:
            return ScreenResult(code=code, name=name, yf_ticker=yf_ticker,
                                passed=False, pass_level="REJECT", reason=f"stale:{msg}")

    min_hist = cfg("screener.min_history_days", 60)
    if len(df) < min_hist:
        return ScreenResult(code=code, name=name, yf_ticker=yf_ticker,
                            passed=False, pass_level="REJECT",
                            reason=f"history<{min_hist}d:{len(df)}")

    # ── 指標計算 ───────────────────────────────────────────────
    vol_ratio = _volume_ratio(df)
    ma5_ma20 = _ma5_gt_ma20(df)
    idx_ret = _index_return if _index_return is not None else get_index_return(20)
    rs = _rs_20d(df, idx_ret)
    pos52 = _weeks_52_position(df)
    close = float(df["Close"].iloc[-1])

    # ── 門檻 ───────────────────────────────────────────────────
    min_vr = cfg("screener.min_volume_ratio", 1.2)
    min_rs = cfg("screener.rs_min", 0.9)
    w52_warn_thresh = cfg("screener.weeks_52_warn", 0.95)
    require_ma = cfg("screener.ma5_above_ma20", True)

    hits = 0
    reasons: list[str] = []
    if vol_ratio >= min_vr:
        hits += 1
    else:
        reasons.append(f"vol={vol_ratio:.2f}<{min_vr}")
    if not require_ma or ma5_ma20:
        hits += 1
    else:
        reasons.append("ma5<ma20")
    if rs >= min_rs:
        hits += 1
    else:
        reasons.append(f"rs={rs:.2f}<{min_rs}")

    # PASS=全中，WARN=2/3，REJECT=0-1
    if hits == 3:
        pass_level = "PASS"
    elif hits >= 2:
        pass_level = "WARN"
    else:
        pass_level = "REJECT"

    passed = pass_level in ("PASS", "WARN")
    warn_52 = pos52 >= w52_warn_thresh

    return ScreenResult(
        code=code, name=name, yf_ticker=yf_ticker,
        passed=passed,
        pass_level=pass_level,
        reason="pass" if hits == 3 else "|".join(reasons),
        close_price=close,
        volume_ratio=vol_ratio,
        ma5_gt_ma20=ma5_ma20,
        rs_20d=rs,
        weeks_52_warn=warn_52,
    )


def screen_batch(candidates: list[dict], today: date | None = None,
                 as_of_date: str | None = None) -> list[ScreenResult]:
    """批次篩選，預先抓一次大盤報酬。"""
    idx_ret = get_index_return(20)
    results = []
    for c in candidates:
        r = screen_one(c, today=today, as_of_date=as_of_date, _index_return=idx_ret)
        results.append(r)
    return results
