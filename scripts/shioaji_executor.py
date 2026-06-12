"""Shioaji 執行器:目標持倉 → 比對現有 → 算買賣單 → 安全閘 → 下單。

把策略(雙引擎/live_portfolio)算出的「目標持倉」變成真實委託。
和 live_portfolio 的「多退少補」一樣,只是把紙上再平衡換成真的送單。

三段式安全(務必依序驗證):
  --dry-run (預設):登入+讀現有持倉+算單,但「不送任何委託」,只印出來給你看。
  --sim          :simulation=True 模擬下單(不花真錢)。
  --live         :simulation=False 真實下單。需 env SHIOAJI_LIVE_CONFIRM=YES 才放行。

安全閘:單股上限、單次總買入上限、最大委託數、漲停不買、最小單、KILL_SWITCH 檔、限價單(非市價)。

目標格式 (JSON):
  {"date": "2026-06-12", "targets": {"2330": 15000, "2454": 10000}}   # 代號 → 目標金額TWD

用法:
  python scripts/shioaji_executor.py --targets targets.json            # dry-run
  python scripts/shioaji_executor.py --targets targets.json --sim
  SHIOAJI_LIVE_CONFIRM=YES python scripts/shioaji_executor.py --targets targets.json --live
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# ── 安全參數(上線前依你的資金調)─────────────────────────────────────────────
MAX_POSITION_TWD   = 20_000     # 單一個股「持有」金額上限
MAX_TOTAL_BUY_TWD  = 50_000     # 本次「總買入」金額上限
MAX_ORDERS         = 12         # 本次最多送幾筆委託
MIN_ORDER_TWD      = 2_000      # 小於此金額的調整不送(避免零碎)
LOT_SHARES         = 1_000      # 整股 1 張 = 1000 股
LIMIT_BUFFER       = 0.015      # 限價:買 +1.5% / 賣 -1.5%(求成交,但不追市價)
LIMIT_UP_GUARD     = 0.094      # 漲幅 ≥9.4% 視為鎖漲停 → 不買
KILL_SWITCH        = ROOT / "KILL_SWITCH"   # 此檔存在 → 立即中止(緊急開關)
SPACING_SEC        = 1.0        # 永豐 API 呼叫間隔規則


def log(msg: str) -> None:
    print(f"[executor] {msg}", flush=True)


def round_lots(shares: float) -> int:
    """無條件捨去到整張(1000股)。"""
    return int(shares // LOT_SHARES) * LOT_SHARES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="目標持倉 JSON 路徑")
    ap.add_argument("--sim", action="store_true", help="模擬下單(simulation=True)")
    ap.add_argument("--live", action="store_true", help="真實下單(simulation=False,需 SHIOAJI_LIVE_CONFIRM=YES)")
    args = ap.parse_args()

    mode = "live" if args.live else ("sim" if args.sim else "dry-run")
    place_orders = mode in ("sim", "live")           # dry-run 不送單
    simulation = (mode != "live")

    # ── 安全:KILL_SWITCH ──
    if KILL_SWITCH.exists():
        log(f"🛑 KILL_SWITCH 存在 ({KILL_SWITCH}) → 中止。刪掉它才會執行。")
        return 10
    # ── 安全:live 二次確認 ──
    if mode == "live" and os.environ.get("SHIOAJI_LIVE_CONFIRM") != "YES":
        log("🛑 --live 需要環境變數 SHIOAJI_LIVE_CONFIRM=YES 才放行(防誤觸真錢)。中止。")
        return 11

    load_dotenv(ROOT / ".env", override=False)
    api_key = os.environ.get("SINOPAC_APIKEY")
    secret_key = os.environ.get("SINOPAC_SECRETKEY")
    ca_path = os.environ.get("SINOPAC_CA_PATH")
    ca_passwd = os.environ.get("SINOPAC_CA_PASSWORD")
    if not api_key or not secret_key:
        log("ERROR: 缺 SINOPAC_APIKEY / SINOPAC_SECRETKEY"); return 1

    targets_raw = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    targets: dict[str, float] = {str(k): float(v) for k, v in targets_raw.get("targets", {}).items()}
    log(f"模式={mode} (送單={place_orders}, simulation={simulation})｜目標 {len(targets)} 檔｜資料日 {targets_raw.get('date','?')}")

    import shioaji as sj
    api = sj.Shioaji(simulation=simulation)
    log("登入 ...")
    accounts = api.login(api_key=api_key, secret_key=secret_key)
    stock_acc = getattr(api, "stock_account", None) or \
        next((a for a in accounts if str(getattr(a, "account_type", "")) in ("S", "AccountType.Stock")), None)
    if stock_acc is None:
        log("ERROR: 找不到證券帳戶"); return 2
    log(f"帳戶 {stock_acc.account_id} ({stock_acc.username})")

    # ── 憑證(送單必須)──
    if place_orders:
        if not ca_path or not ca_passwd:
            log("ERROR: 送單需 SINOPAC_CA_PATH / SINOPAC_CA_PASSWORD"); return 3
        cap = Path(ca_path); cap = cap if cap.is_absolute() else ROOT / cap
        time.sleep(SPACING_SEC)
        if not api.activate_ca(ca_path=str(cap), ca_passwd=ca_passwd):
            log("ERROR: 憑證啟用失敗"); return 4
        log("憑證 OK")

    log("下載商品檔 ...")
    try:
        api.fetch_contracts(contract_download=True)
    except Exception as e:
        log(f"  fetch_contracts cb(已知無害bug): {e}")
    time.sleep(SPACING_SEC)

    # ── 現有持倉(股數)──
    cur_shares: dict[str, int] = {}
    try:
        for p in (api.list_positions(stock_acc) or []):
            cur_shares[str(p.code)] = int(getattr(p, "quantity", 0))   # 註:整股部位 quantity 多為「股」
    except Exception as e:
        log(f"  WARN list_positions: {e}(當作空倉)")
    log(f"現有持倉 {len(cur_shares)} 檔: {cur_shares}")

    # ── 取價(快照)──
    codes = sorted(set(targets) | set(cur_shares))
    px: dict[str, float] = {}
    upref: dict[str, float] = {}
    for code in codes:
        ct = api.Contracts.Stocks.get(code) if hasattr(api.Contracts.Stocks, "get") else api.Contracts.Stocks[code]
        if ct is None:
            continue
        try:
            snap = api.snapshots([ct])
            if snap:
                px[code] = float(snap[0].close or snap[0].sell_price or ct.reference)
        except Exception:
            pass
        if not px.get(code):
            px[code] = float(getattr(ct, "reference", 0) or 0)
        upref[code] = float(getattr(ct, "reference", 0) or 0)   # 昨收(判漲停)
        if not px.get(code):                                    # 即時價抓不到 → 用清洗後歷史收盤後備
            try:
                from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
                o = get_daily_ohlcv(code)
                if o:
                    last = o[max(o)]; px[code] = float(last["close"])
                    if not upref.get(code):
                        ds = sorted(o); upref[code] = float(o[ds[-2]]["close"]) if len(ds) >= 2 else px[code]
                    log(f"  {code} 用歷史收盤後備價 {px[code]}")
            except Exception:
                pass
    time.sleep(SPACING_SEC)

    # ── 算單(diff,整張)──
    plan = []   # (code, action, lots, limit_price, reason)
    total_buy = 0.0
    for code in codes:
        p = px.get(code, 0.0)
        if p <= 0:
            continue
        tgt_val = min(targets.get(code, 0.0), MAX_POSITION_TWD)      # 單股上限
        tgt_sh = round_lots(tgt_val / p)
        cur_sh = cur_shares.get(code, 0)
        delta_sh = tgt_sh - cur_sh
        if abs(delta_sh) < LOT_SHARES:                               # 不足一張不動
            continue
        if abs(delta_sh * p) < MIN_ORDER_TWD:
            continue
        lots = abs(delta_sh) // LOT_SHARES
        if delta_sh > 0:   # 買
            ref = upref.get(code, 0.0)
            if ref > 0 and p / ref - 1 >= LIMIT_UP_GUARD:
                log(f"  跳過 {code}:接近漲停({p/ref-1:+.1%})買不到"); continue
            limit_px = round(p * (1 + LIMIT_BUFFER), 2)
            if total_buy + lots * LOT_SHARES * limit_px > MAX_TOTAL_BUY_TWD:
                log(f"  跳過 {code}:超過單次總買入上限 {MAX_TOTAL_BUY_TWD}"); continue
            total_buy += lots * LOT_SHARES * limit_px
            plan.append((code, "Buy", lots, limit_px, f"目標{tgt_val:.0f} 現{cur_sh*p:.0f}"))
        else:              # 賣
            limit_px = round(p * (1 - LIMIT_BUFFER), 2)
            plan.append((code, "Sell", lots, limit_px, f"減/出 {cur_sh}→{tgt_sh}股"))
    plan = plan[:MAX_ORDERS]

    # ── 印出計畫 ──
    log(f"{'='*60}")
    log(f"下單計畫({len(plan)} 筆;總買入 ~{total_buy:,.0f} TWD):")
    for code, act, lots, lp, why in plan:
        ct = api.Contracts.Stocks[code]
        log(f"  {act:<4} {code} {getattr(ct,'name','')[:5]:<6} {lots}張 @ 限價{lp}  ({why})")
    if not plan:
        log("  無需調整。"); api.logout(); return 0

    # ── 送單(dry-run 不送)──
    if not place_orders:
        log("🟡 dry-run:以上「不會送出」。確認無誤後加 --sim 模擬、或 --live 真實。")
        api.logout(); return 0

    log(f"🔴 開始送單(mode={mode})...")
    logf = ROOT / "reports" / f"shioaji_orders_{datetime.now():%Y%m%d_%H%M%S}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for code, act, lots, lp, why in plan:
        ct = api.Contracts.Stocks[code]
        order = api.Order(
            price=lp, quantity=lots,
            action=sj.Action.Buy if act == "Buy" else sj.Action.Sell,
            price_type=sj.StockPriceType.LMT,
            order_type=sj.OrderType.ROD,
            order_lot=sj.StockOrderLot.Common,
            account=stock_acc,
        )
        time.sleep(SPACING_SEC)
        try:
            trade = api.place_order(contract=ct, order=order)
            sid = getattr(getattr(trade, "status", None), "id", "?")
            st = getattr(getattr(trade, "status", None), "status", "?")
            log(f"  ✅ {act} {code} {lots}張 → id={sid} status={st}")
            records.append({"code": code, "action": act, "lots": lots, "price": lp, "id": str(sid), "status": str(st)})
        except Exception as e:
            log(f"  ❌ {act} {code} 失敗: {e}")
            records.append({"code": code, "action": act, "lots": lots, "price": lp, "error": str(e)})
    logf.write_text(json.dumps({"mode": mode, "time": datetime.now().isoformat(), "orders": records},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"委託紀錄 → {logf}")
    api.logout()
    log("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())