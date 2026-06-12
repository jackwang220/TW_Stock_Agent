"""yfinance 包裝：下載價量資料 + freshness check + 重試。"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from tw_stock_agent.config import cfg, DATA_DIR

_OHLCV_CACHE = DATA_DIR / "ohlcv_cache"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       reraise=True)
def _download(ticker: str, period: str = "3mo", end: str | None = None) -> pd.DataFrame:
    kwargs: dict = {"progress": False, "auto_adjust": True}
    if end:
        kwargs["end"] = end
        kwargs["start"] = (date.fromisoformat(end) - timedelta(days=120)).isoformat()
    else:
        kwargs["period"] = period
    df = yf.download(ticker, **kwargs)
    # yfinance 有時回傳 multi-level columns（當只下載一支時也可能）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_ohlcv(ticker: str, period: str = "3mo", as_of_date: str | None = None) -> pd.DataFrame:
    """下載日 K OHLCV，回傳單層 column 的 DataFrame。

    Args:
        ticker: e.g. "2059.TW"
        period: yfinance period string，as_of_date 為 None 時使用
        as_of_date: ISO date string，設定後只抓到這天為止（回測用）

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume；index 為 DatetimeIndex
        失敗時回傳空 DataFrame
    """
    # 磁碟快取:live(as_of=None)用今天日期當 key → 每天重抓一次;
    # 回測(as_of 有值)用 as_of 當 key。大 universe 反覆篩選時避免重打 yfinance 被限流。
    key = as_of_date or f"live-{date.today().isoformat()}"
    cache_file = _OHLCV_CACHE / f"{ticker}_{key}.pkl"
    if cache_file.exists():
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass
    try:
        df = _download(ticker, period=period, end=as_of_date)
        df = df.ffill().fillna(0)
        if not df.empty:
            _OHLCV_CACHE.mkdir(parents=True, exist_ok=True)
            try:
                df.to_pickle(cache_file)
            except Exception:
                pass
        return df
    except Exception:
        return pd.DataFrame()


def validate_price_data(df: pd.DataFrame, expected_date: date) -> tuple[bool, str]:
    """檢查資料是否是最新的。

    Returns:
        (ok: bool, reason: str)
    """
    min_history = cfg("screener.min_history_days", 60)
    if df.empty:
        return False, "empty dataframe"
    last_date = df.index[-1].date()
    # 允許 T-1（週末後的週一等情況）
    delta = (expected_date - last_date).days
    if delta > 3:
        return False, f"stale: last={last_date}, expected={expected_date} (delta={delta}d)"
    if len(df) < min_history:
        return False, f"not enough history ({len(df)} < {min_history} days)"
    return True, "ok"


def get_index_return(days: int = 20) -> float:
    """抓加權指數近 N 日報酬率（用於相對強度計算）。"""
    try:
        df = _download("^TWII", period="3mo")
        if len(df) < days:
            return 0.0
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-days] - 1)
    except Exception:
        return 0.0
