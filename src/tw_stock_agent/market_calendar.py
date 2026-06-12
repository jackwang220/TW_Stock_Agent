"""交易日曆 + 盤後/盤前 session 解析。

解決「你幾點跑」造成的日期混亂:
  - 盤後 11pm(6/9 週二)跑   → 決策日=6/9、交易日=6/10
  - 半夜 1am(6/10 週三)跑   → 決策日=6/9(市場還沒開、最新收盤仍是6/9)、交易日=6/10
  - 開盤前 8am(6/10)跑      → 同上(決策6/9、交易6/10)
  - 盤中/收盤前跑            → 決策日=前一交易日(當日收盤還沒底定)

慣例(使用者定的盤後制):
  決策日 D = 最近一個「收盤已底定」的交易日(用 D 收盤+盤後法人資料做決策)。
  交易日 = D 的下一個交易日(隔日開盤/收盤進場、再隔一交易日結算的標的日)。
  報告檔以「決策日」命名;signal_log / 帳本以「交易日」命名。
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

# 台股 13:30 收盤;盤後資料(收盤價、三大法人)約 14:00 後底定 → 之後才算「當日決策可用」
SETTLE_AFTER = time(14, 0)


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5   # 0-4 = 一~五(國定假日忽略;缺資料時下游會自動跳過)


def _prev_weekday(d: date) -> date:
    while not _is_weekday(d):
        d -= timedelta(days=1)
    return d


def _next_weekday(d: date) -> date:
    while not _is_weekday(d):
        d += timedelta(days=1)
    return d


def last_close_date(now: datetime | None = None) -> str:
    """now 時點『最近一個收盤已底定』的交易日(ISO)。盤後算今天,否則往前找。"""
    if now is None:
        now = datetime.now()
    d = now.date()
    # 今天若是交易日且已過盤後底定時間 → 用今天;否則從昨天起往前找
    if not (_is_weekday(d) and now.time() >= SETTLE_AFTER):
        d -= timedelta(days=1)
    return _prev_weekday(d).isoformat()


def next_trading_day(d: str) -> str:
    """d(ISO)的下一個交易日(ISO),跨過週末。"""
    return _next_weekday(date.fromisoformat(d) + timedelta(days=1)).isoformat()


def resolve_session(as_of: str | None = None, now: datetime | None = None) -> tuple[str, str]:
    """回傳 (決策日, 交易日)。as_of 有給=手動指定決策日;否則用 now 自動解析。"""
    decision = as_of or last_close_date(now)
    return decision, next_trading_day(decision)