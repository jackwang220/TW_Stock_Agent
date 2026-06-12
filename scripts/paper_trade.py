"""永豐金 Shioaji 模擬下單系統（simulation=True）。

使用方式：
    uv run python scripts/paper_trade.py           # 執行今日掃描並模擬下單
    uv run python scripts/paper_trade.py --reconcile   # 只對帳（不下單）
    uv run python scripts/paper_trade.py --status      # 顯示持倉 + 損益

需要 .env 設定（或環境變數）:
    SINOPAC_APIKEY / SINOPAC_SECRETKEY
    SINOPAC_CA_PATH / SINOPAC_CA_PASSWORD
    SINOPAC_PERSON_ID

注意：
    - simulation=True 為強制硬編碼，永遠不會真實下單
    - 14:30 盤後零股固定時間送出（通過 schedule.every().day.at）
    - Graduation check：60 天方向準確率 > 55% 且最大回撤 > -10%
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from tw_stock_agent.config import DATA_DIR, SIGNAL_LOG, cfg
from tw_stock_agent.signal_log import backfill_returns

try:
    import shioaji as sj
    from shioaji.constant import Action, StockPriceType
    HAS_SHIOAJI = True
except ImportError:
    HAS_SHIOAJI = False

# ── 持倉帳本 ────────────────────────────────────────────────────────────────
POSITIONS_FILE = DATA_DIR / "paper_positions.json"
TRADE_LOG_FILE = DATA_DIR / "paper_trades.csv"
TRADE_LOG_FIELDS = [
    "date", "ticker", "name", "action",   # BUY / SELL
    "price", "shares", "amount", "reason",
]

# ── 分層倉位（與 screener 對應的三段設定）─────────────────────────────────
VOL_TIER = {
    "low":  0.04,  # 年化波動 < 20%
    "med":  0.025, # 年化波動 20-35%
    "high": 0.015, # 年化波動 > 35%
}
CAPITAL = 200_000          # 模擬資金（TWD）
MAX_POSITIONS = 5          # 最大同時持倉數
ANNUAL_VOL_THRESHOLDS = (0.20, 0.35)


def _vol_tier(annual_vol: float) -> str:
    low, high = ANNUAL_VOL_THRESHOLDS
    if annual_vol < low:
        return "low"
    if annual_vol < high:
        return "med"
    return "high"


def vol_tier_shares(price: float, annual_vol: float) -> int:
    """計算下單零股張數（最少 1 股、上限 CAPITAL * tier_pct）。"""
    tier = _vol_tier(annual_vol)
    budget = CAPITAL * VOL_TIER[tier]
    shares = int(budget / price)
    return max(1, shares)


# ── 持倉帳本 I/O ─────────────────────────────────────────────────────────
def load_positions() -> dict[str, dict]:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
    return {}


def save_positions(pos: dict[str, dict]) -> None:
    POSITIONS_FILE.write_text(
        json.dumps(pos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_trade(row: dict) -> None:
    is_new = not TRADE_LOG_FILE.exists()
    with TRADE_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)


# ── Shioaji 連線 ─────────────────────────────────────────────────────────
def connect_api() -> "sj.Shioaji | None":
    if not HAS_SHIOAJI:
        logger.warning("shioaji not installed. pip install shioaji (optional)")
        return None

    import os
    api_key    = os.getenv("SINOPAC_APIKEY", "")
    secret_key = os.getenv("SINOPAC_SECRETKEY", "")
    ca_path    = os.getenv("SINOPAC_CA_PATH", "")
    ca_pw      = os.getenv("SINOPAC_CA_PASSWORD", "")
    person_id  = os.getenv("SINOPAC_PERSON_ID", "")

    if not api_key or not secret_key:
        logger.error("Missing SINOPAC_APIKEY / SINOPAC_SECRETKEY in env.")
        return None

    api = sj.Shioaji(simulation=True)  # 強制模擬模式
    api.login(api_key=api_key, secret_key=secret_key, fetch_contract=True)

    if ca_path and Path(ca_path).exists():
        try:
            api.activate_ca(
                ca_path=ca_path,
                ca_passwd=ca_pw,
                person_id=person_id,
            )
            logger.info("CA activated.")
        except Exception as e:
            logger.warning(f"CA activation failed (simulation OK without CA): {e}")

    logger.info("Shioaji simulation=True connected.")
    return api


def place_paper_order(api: "sj.Shioaji", ticker: str, shares: int, price: float) -> bool:
    """盤後零股下單（模擬）。"""
    try:
        contract = api.Contracts.Stocks[ticker]
        order = api.Order(
            price=price,
            quantity=shares,
            action=Action.Buy,
            price_type=StockPriceType.MKT,  # 市價
            order_type=sj.constant.OrderType.ROD,
            order_lot=sj.constant.StockOrderLot.IntradayOdd,  # 零股
        )
        trade = api.place_order(contract, order)
        logger.info(f"  [SIM] 下單 {ticker} {shares}股 ≈${price:.1f}  status={trade.status.status}")
        return True
    except Exception as e:
        logger.error(f"  place_order failed {ticker}: {e}")
        return False


# ── 訊號讀取 ─────────────────────────────────────────────────────────────
def read_today_signals(today: str) -> list[dict]:
    """從 signal_log.csv 讀取今天的 BUY 訊號。"""
    if not SIGNAL_LOG.exists():
        return []
    with SIGNAL_LOG.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("date") == today and r.get("verdict") == "BUY"]


# ── 損益計算 ─────────────────────────────────────────────────────────────
def calc_pnl(positions: dict[str, dict]) -> None:
    import yfinance as yf
    print(f"\n{'='*55}")
    print(f"  持倉損益  ({date.today()})")
    print(f"{'='*55}")
    total_pnl = 0.0
    for ticker, pos in positions.items():
        try:
            yf_tk = pos.get("yf_ticker", f"{ticker}.TW")
            data = yf.download(yf_tk, period="2d", progress=False, auto_adjust=True)
            if data.empty:
                continue
            curr = float(data["Close"].iloc[-1])
            cost = float(pos["avg_price"])
            shares = int(pos["shares"])
            pnl = (curr - cost) * shares
            pct = (curr / cost - 1) * 100
            total_pnl += pnl
            print(f"  {ticker:6s} {pos['name']:12s}  "
                  f"均價={cost:.1f}  現價={curr:.1f}  "
                  f"損益={pnl:+.0f}  ({pct:+.1f}%)")
        except Exception as e:
            logger.debug(f"  {ticker} pnl error: {e}")
    print(f"{'─'*55}")
    print(f"  合計損益: {total_pnl:+,.0f} TWD")
    print(f"{'='*55}\n")


# ── Graduation check ────────────────────────────────────────────────────
def graduation_check() -> None:
    """讀最近 60 天 signal_log，檢查方向準確率和最大回撤。"""
    if not SIGNAL_LOG.exists():
        logger.warning("signal_log.csv not found, skip graduation check.")
        return

    cutoff = (date.today() - timedelta(days=60)).isoformat()
    rows = []
    with SIGNAL_LOG.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("date", "") >= cutoff and r.get("return_10d"):
                try:
                    rows.append(float(r["return_10d"]) / 100)
                except ValueError:
                    pass

    if len(rows) < 10:
        logger.info(f"Graduation: not enough signals ({len(rows)}/10 minimum), skip.")
        return

    import numpy as np
    from tw_stock_agent.backtest.metrics import direction_accuracy, max_drawdown

    returns = np.array(rows)
    da = direction_accuracy(returns)
    mdd = max_drawdown(returns)
    da_thr = cfg("graduation.min_direction_accuracy")
    mdd_thr = cfg("graduation.max_drawdown_threshold")

    status = "PASS" if (da >= da_thr and mdd >= mdd_thr) else "FAIL"
    logger.info(
        f"Graduation ({len(returns)} signals, 60d): "
        f"DirAcc={da:.1%} (≥{da_thr:.0%}?{'✓' if da>=da_thr else '✗'}), "
        f"MaxDD={mdd*100:.1f}% (≥{mdd_thr*100:.0f}%?{'✓' if mdd>=mdd_thr else '✗'}) "
        f"→ {status}"
    )
    if status == "PASS":
        logger.success("Graduation condition met! Consider switching to live trading.")


# ── 對帳 ──────────────────────────────────────────────────────────────────
def reconcile(positions: dict[str, dict]) -> dict[str, dict]:
    """
    移除已持倉 > HOLD_DAYS 的部位（模擬賣出），更新持倉帳本。
    預設持倉 10 個交易日（約 14 天曆日）。
    """
    HOLD_DAYS = 14
    cutoff = (date.today() - timedelta(days=HOLD_DAYS)).isoformat()
    to_sell = [code for code, pos in positions.items() if pos.get("date", "") <= cutoff]

    for code in to_sell:
        pos = positions.pop(code)
        logger.info(f"  [RECONCILE] 平倉 {code} {pos['name']}  (entered {pos['date']})")
        append_trade({
            "date": date.today().isoformat(),
            "ticker": code,
            "name": pos.get("name", ""),
            "action": "SELL",
            "price": "",
            "shares": pos.get("shares", 0),
            "amount": "",
            "reason": f"hold_days>{HOLD_DAYS}d",
        })

    return positions


# ── 主流程 ────────────────────────────────────────────────────────────────
def run_paper_trade(api: "sj.Shioaji | None") -> None:
    today = date.today().isoformat()
    positions = load_positions()

    # 對帳：平掉過期部位
    positions = reconcile(positions)

    # 讀今日訊號
    signals = read_today_signals(today)
    if not signals:
        logger.info("No BUY signals today.")
    else:
        logger.info(f"{len(signals)} BUY signal(s) today.")

    # 下單（模擬）
    slot_available = MAX_POSITIONS - len(positions)
    placed = 0
    for sig in signals:
        if placed >= slot_available:
            logger.warning(f"  Reached max positions ({MAX_POSITIONS}), skipping rest.")
            break

        ticker = sig.get("ticker", "")
        name   = sig.get("name", ticker)
        if ticker in positions:
            logger.debug(f"  {ticker} already in positions, skip.")
            continue

        # 取現價和波動率
        try:
            import yfinance as yf
            yf_tk = f"{ticker}.TW"
            data = yf.download(yf_tk, period="60d", progress=False, auto_adjust=True)
            if data.empty or len(data) < 20:
                continue
            price = float(data["Close"].iloc[-1])
            vol = float(data["Close"].pct_change().dropna().std() * (252 ** 0.5))
        except Exception as e:
            logger.warning(f"  {ticker} price fetch error: {e}")
            continue

        shares = vol_tier_shares(price, vol)
        amount = price * shares
        logger.info(f"  下單 {ticker} {name}  {shares}股 × ${price:.1f} = ${amount:.0f}")

        ok = True
        if api is not None:
            ok = place_paper_order(api, ticker, shares, price)

        if ok:
            positions[ticker] = {
                "name": name,
                "date": today,
                "avg_price": price,
                "shares": shares,
                "yf_ticker": yf_tk,
                "vol": round(vol, 4),
            }
            append_trade({
                "date": today,
                "ticker": ticker,
                "name": name,
                "action": "BUY",
                "price": f"{price:.2f}",
                "shares": shares,
                "amount": f"{amount:.0f}",
                "reason": "signal",
            })
            placed += 1

    save_positions(positions)
    logger.info(f"Positions saved ({len(positions)} open).")

    # 回填報酬
    updated = backfill_returns()
    if updated:
        logger.info(f"Backfilled {updated} return cells in signal_log.")

    # Graduation check
    graduation_check()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shioaji paper trading")
    parser.add_argument("--reconcile", action="store_true", help="只對帳，不下單")
    parser.add_argument("--status",    action="store_true", help="顯示持倉損益")
    parser.add_argument("--no-shioaji", action="store_true", help="跳過 Shioaji 連線（pure Python）")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    positions = load_positions()

    if args.status:
        calc_pnl(positions)
        return

    if args.reconcile:
        positions = reconcile(positions)
        save_positions(positions)
        logger.success("Reconcile done.")
        graduation_check()
        return

    api = None
    if not args.no_shioaji:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        api = connect_api()

    run_paper_trade(api)

    if api is not None:
        try:
            api.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
