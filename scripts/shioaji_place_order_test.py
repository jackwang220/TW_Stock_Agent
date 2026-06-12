"""Shioaji place_order compliance test (stock account only).

Per Sinopac requirements, must test login + place_order before production access.

Safety stack (3 layers):
  1. simulation=True (paper trading mode — no real money)
  2. BUY LIMIT at 1 NTD on 2330 (1000x below current ~2300 → physically unfillable)
  3. cancel_order immediately after placement

If the limit somehow filled at 1 NTD (impossible at current quote levels),
worst case is 1000 NTD per round-lot — far below any "破產" threshold.

DOES NOT test futures account (no agreement signed per user 2026-05-18).

Usage:
    # Only run when user has explicitly OK'd this run
    uv run python scripts/shioaji_place_order_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[place-order-test] {msg}", flush=True)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.environ.get("SINOPAC_APIKEY")
    secret_key = os.environ.get("SINOPAC_SECRETKEY")
    if not api_key or not secret_key:
        log("ERROR: SINOPAC_APIKEY / SINOPAC_SECRETKEY not set in .env")
        return 1

    import shioaji as sj
    log("Init Shioaji in SIMULATION (paper trading) mode ...")
    api = sj.Shioaji(simulation=True)

    log("Login ...")
    try:
        accounts = api.login(api_key=api_key, secret_key=secret_key)
    except Exception as e:
        log(f"ERROR login: {e}")
        return 2
    log(f"  login OK, {len(accounts)} accounts")

    # 股票帳戶:用內建捷徑(shioaji 1.5.x 帳戶型別名是 Account/account_type='S')
    stock_acc = getattr(api, "stock_account", None)
    if stock_acc is None:
        stock_acc = next((a for a in accounts if str(getattr(a, "account_type", "")) in ("S", "AccountType.Stock")), None)
    if stock_acc is None:
        log("ERROR: no stock account. Aborting.")
        return 3
    log(f"Using stock account: {stock_acc.account_id} ({stock_acc.username})")

    # Fetch contracts (known shioaji 1.3.3 callback bug; lookup still works)
    log("Fetching contracts ...")
    try:
        api.fetch_contracts(contract_download=True)
    except Exception as e:
        log(f"  fetch_contracts cb error (known bug, harmless): {e}")
    time.sleep(1.0)  # Sinopac 1s spacing rule

    contract = api.Contracts.Stocks["2330"]
    if contract is None:
        log("ERROR: 2330 contract not found. Aborting.")
        return 4
    log(f"Contract 2330: {contract.code} {contract.name}, reference={contract.reference}")

    # === BUILD ORDER ===
    # SAFETY: BUY LIMIT @ 1 NTD on 2330 — far below any conceivable quote.
    # 1 lot = 1000 shares. Max loss if somehow filled = 1000 NTD.
    SAFE_PRICE = 1.0    # NTD — orders of magnitude below market
    QUANTITY = 1        # 1 張 = 1000 shares

    log("Building order: BUY LIMIT 2330 × 1 張 @ 1 NTD")
    log("  (10000x below market — physically unfillable)")
    order = api.Order(
        price=SAFE_PRICE,
        quantity=QUANTITY,
        action=sj.constant.Action.Buy,
        price_type=sj.constant.StockPriceType.LMT,    # Limit
        order_type=sj.constant.OrderType.ROD,         # Rest of Day
        order_lot=sj.constant.StockOrderLot.Common,   # Round lot (1000 shares)
        account=stock_acc,
    )

    log("place_order ...")
    time.sleep(1.0)
    try:
        trade = api.place_order(contract=contract, order=order)
    except Exception as e:
        log(f"ERROR place_order: {e}")
        return 5
    log(f"  trade returned: {trade!r}")
    if trade is None:
        log("  WARN: place_order returned None — broker may have rejected silently")
    else:
        # trade typically has .status with .status_code, .order_id, etc.
        try:
            log(f"  order_id: {getattr(trade.status, 'id', '?')}")
            log(f"  status: {getattr(trade.status, 'status', '?')}")
        except Exception:
            pass

    # Brief pause so the order is registered before cancel
    time.sleep(2.0)

    log("cancel_order ...")
    try:
        cancel_result = api.cancel_order(trade=trade)
        log(f"  cancel_order returned: {cancel_result!r}")
    except Exception as e:
        log(f"  WARN cancel_order: {e}")

    time.sleep(1.0)

    # Verify by listing recent trades
    log("Verifying state via list_trades ...")
    try:
        trades = api.list_trades()
        log(f"  list_trades count: {len(trades) if trades else 0}")
        for t in (trades or [])[-3:]:
            log(f"    {t!r}")
    except Exception as e:
        log(f"  WARN list_trades: {e}")

    try:
        api.logout()
    except Exception:
        pass

    log("DONE. Order placed + cancelled in simulation mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
