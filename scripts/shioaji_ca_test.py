"""Shioaji CA (certificate) activation test — isolated from order placement.

Reads:
  SINOPAC_APIKEY, SINOPAC_SECRETKEY → login
  SINOPAC_CA_PATH (e.g. ./Sinopac.pfx), SINOPAC_CA_PASSWORD → activate_ca

Does NOT place any orders. Pure cert validation.

Usage:
    # 1. Fill SINOPAC_CA_PASSWORD in .env
    # 2. Run:
    uv run python scripts/shioaji_ca_test.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[ca-test] {msg}", flush=True)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.environ.get("SINOPAC_APIKEY")
    secret_key = os.environ.get("SINOPAC_SECRETKEY")
    ca_path = os.environ.get("SINOPAC_CA_PATH")
    ca_passwd = os.environ.get("SINOPAC_CA_PASSWORD")

    if not api_key or not secret_key:
        log("ERROR: missing SINOPAC_APIKEY / SINOPAC_SECRETKEY")
        return 1
    if not ca_path or not ca_passwd:
        log("ERROR: missing SINOPAC_CA_PATH / SINOPAC_CA_PASSWORD")
        log("  Fill them in .env then re-run.")
        return 1

    # Resolve CA path (allow relative to project root)
    ca_path_full = Path(ca_path)
    if not ca_path_full.is_absolute():
        ca_path_full = PROJECT_ROOT / ca_path
    if not ca_path_full.exists():
        log(f"ERROR: CA file not found at {ca_path_full}")
        return 2
    log(f"CA file: {ca_path_full} ({ca_path_full.stat().st_size} bytes)")

    import shioaji as sj
    log("Init Shioaji simulation=True ...")
    api = sj.Shioaji(simulation=True)

    log("Login ...")
    try:
        accounts = api.login(api_key=api_key, secret_key=secret_key)
    except Exception as e:
        log(f"ERROR login: {e}")
        return 3
    person_id = accounts[0].person_id if accounts else None
    log(f"  login OK, person_id={person_id}")

    log("activate_ca ...")
    try:
        ok = api.activate_ca(ca_path=str(ca_path_full), ca_passwd=ca_passwd)
    except Exception as e:
        log(f"  ERROR activate_ca: {e}")
        return 4
    log(f"  activate_ca returned: {ok}")

    if ok:
        log("  ✅ CA activated successfully — would be usable for production trading")
    else:
        log("  ❌ CA activation returned falsy. Check password / file format.")

    # Check expiry — newer shioaji requires person_id
    try:
        expiry = api.get_ca_expiretime(person_id=person_id)
        log(f"CA expiry: {expiry}")
        if isinstance(expiry, datetime):
            days_left = (expiry - datetime.now()).days
            log(f"  Days remaining: {days_left}")
    except Exception as e:
        log(f"  WARN get_ca_expiretime: {e}")

    try:
        api.logout()
    except Exception:
        pass
    log("DONE.")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
