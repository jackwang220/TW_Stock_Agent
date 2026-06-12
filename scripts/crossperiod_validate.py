"""跨期間 LLM 驗證(取樣):2022熊/2023復甦/2024多頭/2025 各抓一段,
用統一系統的 as-of 邏輯(FinMind 新聞+反彈+法人+營收,全 ≤決策日,零洩漏)跑 LLM 預測,
量各 regime 的方向準確率 + 平均報酬,並同步記錄純反彈訊號 → 比較「LLM 是真強還是窗運氣」。

全程 FinMind(10年快取),不碰 yfinance(避免歷史 as-of 撞限流)。
輸出 data/crossperiod_results.csv。可重複執行(resume:跳過已完成 regime+as_of+code)。
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.universe import universe_as_of
from tw_stock_agent.debate.bear import run_debate
from tw_stock_agent.market_calendar import next_trading_day
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.market_status import get_stock_market_status
from tw_stock_agent.tools.rebound_signal import rebound_signal_for_code, avg_turnover_of

OUT = DATA_DIR / "crossperiod_results.csv"
TOP_N = 12          # 每決策日 debate 的流動性前 N(point-in-time)
DAYS_PER_WINDOW = 20  # 每regime取樣天數(拉寬→更多方向單,解n太小)
WORKERS = 8         # 同一天內並發檔數(LLM 並發,大幅加速)
MKT = "0050"        # 市場基準(算 alpha=個股報酬-大盤報酬)

# 各 regime 取樣窗(會用 FinMind 交易日曆過濾出實際交易日,取前 DAYS_PER_WINDOW 天)
WINDOWS = {
    "2022熊市":   ("2022-06-13", "2022-07-22", 20),   # (start, end, 取幾個交易日)
    "2023復甦":   ("2023-05-15", "2023-06-23", 20),
    "2024多頭":   ("2024-06-10", "2024-07-19", 20),
    "2025關稅崩": ("2025-04-07", "2025-05-16", 20),
    "2026最近":   ("2026-03-17", "2026-06-09", 60),   # 最近最重要 → 取整段 ~60 交易日
}

FIELDS = ["regime", "as_of", "trade_date", "code", "name", "predicted_direction",
          "predicted_center_pct", "prediction_confidence", "verdict", "bear_score",
          "bull_score", "rebound_fired", "rebound_edge", "as_of_close", "next_close",
          "ret_1d", "index_ret_1d", "alpha_1d", "dir_correct"]


def _trading_days(start: str, end: str, n: int) -> list[str]:
    oh = get_daily_ohlcv("2330")
    days = sorted(d for d in oh if start <= d <= end)
    return days[:n]


def _finmind_stock(code: str, name: str, as_of: str) -> dict | None:
    oh = get_daily_ohlcv(code)
    days = sorted(d for d in oh if d <= as_of)
    if len(days) < 25:
        return None
    closes = [oh[d]["close"] for d in days]
    vols = [oh[d].get("volume", 0) for d in days]
    last = days[-1]
    avg20v = sum(vols[-20:]) / 20 if len(vols) >= 20 else (sum(vols) / len(vols) or 1)
    bars = []
    for i, d in enumerate(days[-5:]):
        prev_c = oh[days[-6 + i]]["close"] if len(days) >= 6 else oh[d]["close"]
        r = oh[d]
        bars.append({"date": d, "open": r.get("open", 0), "high": r.get("high", 0),
                     "low": r.get("low", 0), "close": r.get("close", 0),
                     "vol_ratio": (r.get("volume", 0) / avg20v) if avg20v else 0,
                     "chg_pct": (r["close"] / prev_c - 1) * 100 if prev_c else 0})
    ma5 = sum(closes[-5:]) / 5
    ma20 = sum(closes[-20:]) / 20
    stock = {
        "code": code, "name": name, "yf_ticker": f"{code}.TW",
        "close_price": closes[-1], "volume_ratio": vols[-1] / avg20v if avg20v else 0,
        "ma5_gt_ma20": ma5 > ma20, "rs_20d": 1.0, "pattern_type": "none",
        "pass_level": "WARN", "recent_bars": bars,
    }
    try:
        from datetime import date as _d
        prev_close = closes[-2] if len(closes) >= 2 else closes[-1]
        stock["market_status"] = get_stock_market_status(
            code, name, close=closes[-1], prev_close=prev_close, as_of=_d.fromisoformat(as_of))
    except Exception:
        pass
    return stock


def _done_keys() -> set:
    if not OUT.exists():
        return set()
    with OUT.open(encoding="utf-8") as f:
        return {f"{r['regime']}|{r['as_of']}|{r['code']}" for r in csv.DictReader(f)}


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | {message}")

    done = _done_keys()
    write_header = not OUT.exists()
    f = OUT.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if write_header:
        w.writeheader()

    IDX = get_daily_ohlcv(MKT, start="2021-01-01", force_refresh=True)  # 市場基準(0050)→ alpha
    logger.info(f"市場基準 {MKT}:{len(IDX)} 天")

    def _process(regime, as_of, trade_date, c):
        """單檔:FinMind 建 stock + run_debate(≤as_of)。回傳 row dict 或 None。執行緒安全(不寫檔)。"""
        from datetime import date as _d
        code, name = c["code"], c["name"]
        stock = _finmind_stock(code, name, as_of)
        if stock is None:
            return None
        oh = get_daily_ohlcv(code)
        as_close = stock["close_price"]
        fdays = sorted(d for d in oh if d > as_of)
        next_close = oh[fdays[0]]["close"] if fdays else None
        ret = (next_close / as_close - 1) if next_close else None
        # alpha = 個股報酬 - 大盤(0050)同期報酬(扣 beta,才知道是不是真選股)
        idx_le = [d for d in IDX if d <= as_of]; idx_gt = [d for d in IDX if d > as_of]
        idx_ret = (IDX[min(idx_gt)]["close"] / IDX[max(idx_le)]["close"] - 1) if (idx_le and idx_gt) else None
        alpha = (ret - idx_ret) if (ret is not None and idx_ret is not None) else None
        turn = c.get("_avg_turnover") or avg_turnover_of(code)
        rb = rebound_signal_for_code(code, turn, as_of=as_of)
        try:
            res = run_debate(stock, [], historical_date=_d.fromisoformat(as_of))
        except Exception as e:
            logger.error(f"  {as_of} {code} 失敗:{str(e)[:80]}")
            return None
        dc = ""
        if ret is not None and res.predicted_direction in ("up", "down"):
            dc = "1" if ((res.predicted_direction == "up" and ret > 0) or
                         (res.predicted_direction == "down" and ret < 0)) else "0"
        return {
            "regime": regime, "as_of": as_of, "trade_date": trade_date, "code": code, "name": name,
            "predicted_direction": res.predicted_direction,
            "predicted_center_pct": f"{res.predicted_center_pct:.2f}",
            "prediction_confidence": f"{res.prediction_confidence:.2f}",
            "verdict": res.verdict, "bear_score": res.bear_score, "bull_score": res.bull_score,
            "rebound_fired": int(rb.get("fired", False)),
            "rebound_edge": f"{rb.get('rebound_edge', 0):.4f}",
            "as_of_close": f"{as_close:.2f}",
            "next_close": f"{next_close:.2f}" if next_close else "",
            "ret_1d": f"{ret*100:.2f}" if ret is not None else "",
            "index_ret_1d": f"{idx_ret*100:.2f}" if idx_ret is not None else "",
            "alpha_1d": f"{alpha*100:.2f}" if alpha is not None else "",
            "dir_correct": dc,
        }

    total_done = 0
    for regime, (start, end, ndays) in WINDOWS.items():
        tdays = _trading_days(start, end, ndays)
        if not tdays:
            logger.warning(f"=== {regime}:無交易日資料({start}~{end}),跳過(快取需深抓)===")
            continue
        logger.info(f"=== {regime}:{len(tdays)} 決策日 {tdays[0]}~{tdays[-1]} ===")
        for as_of in tdays:
            trade_date = next_trading_day(as_of)
            cands = universe_as_of(as_of, top_n=TOP_N)
            todo = [c for c in cands if f"{regime}|{as_of}|{c['code']}" not in done]
            if not todo:
                continue
            # 同一天內各檔並發(不同 ticker→不同快取檔,執行緒安全;只在主緒寫 CSV)
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                rows = list(ex.map(lambda c: _process(regime, as_of, trade_date, c), todo))
            ok = 0
            for row in rows:
                if row:
                    w.writerow(row); ok += 1; total_done += 1
            f.flush()
            logger.info(f"  {as_of} 完成 {ok}/{len(todo)}(累計新增 {total_done})")
    f.close()
    logger.info(f"全部完成,新增 {total_done} 筆 → {OUT}")


if __name__ == "__main__":
    main()