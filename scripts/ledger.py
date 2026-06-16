"""手動帳本(元大手動交易用)。記錄你實際買賣,讓 live_dual_v5_trade.py --ledger 算持有/損益。
資料檔:data/positions.json  格式:{ "2330": {"qty": 18, "cost": 1010.0}, ... }(cost=每股平均成本)

用法:
  uv run python scripts/ledger.py buy  2330 18 1010     # 買18股@1010(自動算新均價)
  uv run python scripts/ledger.py sell 2330 5  1050     # 賣5股@1050(印已實現損益;歸零自動移除)
  uv run python scripts/ledger.py set  2330 18 1010     # 直接覆寫成 18股、均價1010
  uv run python scripts/ledger.py rm   2330             # 刪掉一檔
  uv run python scripts/ledger.py show                  # 看目前帳本
  uv run python scripts/ledger.py cash 30000            # 設定現金池=30000(所有執行/排程都用這個金額)
  uv run python scripts/ledger.py cash auto             # 取消現金池,恢復 DCA 自動(15000+1000/日封頂5萬)
  uv run python scripts/ledger.py cash                  # 看目前現金池設定
  uv run python scripts/ledger.py weights 1,0.3,0,1.5   # 設雙引擎權重(多頭H,多頭reb,空頭H,空頭reb)
  uv run python scripts/ledger.py weights auto          # 恢復預設 B純切 1,0,0,1.5
  uv run python scripts/ledger.py weights               # 看目前權重設定
名稱會自動帶入(查 base_universe);價格單位=元/股。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "positions.json"
CAP_FILE = ROOT / "data" / "capital.json"
WEIGHTS_FILE = ROOT / "data" / "weights.json"
UNIV = ROOT / "data" / "base_universe.json"


def _names() -> dict:
    try:
        u = json.loads(UNIV.read_text(encoding="utf-8"))
        return {c: u[c].get("name", c) for c in u}
    except Exception:
        return {}


def load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {}


def save(d: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def show(d: dict, names: dict) -> None:
    if not d:
        print("帳本空。"); return
    tot = 0.0
    print(f"{'代號':<6}{'名稱':<8}{'股數':>7}{'均價':>10}{'成本市值':>12}")
    for code, v in sorted(d.items()):
        qty, cost = int(v.get("qty", 0)), float(v.get("cost", 0))
        val = qty * cost; tot += val
        print(f"{code:<6}{names.get(code, code):<8}{qty:>7}{cost:>10.2f}{val:>12,.0f}")
    print(f"{'總成本':<6}{'':<8}{'':>7}{'':>10}{tot:>12,.0f}")


def main() -> int:
    names = _names()
    a = sys.argv[1:]
    if not a or a[0] == "show":
        show(load(), names); return 0
    cmd = a[0]

    if cmd == "weights":
        if len(a) == 1:                      # 查詢
            cur = None
            if WEIGHTS_FILE.exists():
                try: cur = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8")).get("weights")
                except Exception: cur = None
            print(f"雙引擎權重 = {cur} (多頭H,多頭reb,空頭H,空頭reb)" if cur else "雙引擎權重 = 預設 B純切 1,0,0,1.5(未設)")
        elif a[1] == "auto":
            WEIGHTS_FILE.unlink(missing_ok=True)
            print("已取消自訂權重 → 恢復預設 B純切(1,0,0,1.5)")
        else:
            w = [float(x) for x in a[1].split(",")]
            if len(w) != 4:
                print("格式:weights 多頭H,多頭reb,空頭H,空頭reb  例 weights 1,0.3,0,1.5"); return 2
            WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            WEIGHTS_FILE.write_text(json.dumps({"weights": w}, ensure_ascii=False), encoding="utf-8")
            print(f"雙引擎權重設為 {w}(之後所有執行/排程都用;weights auto 可恢復)")
        return 0

    if cmd == "cash":
        if len(a) == 1:                      # 查詢
            cur = None
            if CAP_FILE.exists():
                try: cur = json.loads(CAP_FILE.read_text(encoding="utf-8")).get("capital")
                except Exception: cur = None
            print(f"現金池 = {cur:,.0f} 元" if cur else "現金池 = 自動 DCA(未設手動值)")
        elif a[1] == "auto":                 # 恢復 DCA
            CAP_FILE.unlink(missing_ok=True)
            print("已取消手動現金池 → 恢復 DCA 自動(15000+1000/日,封頂5萬)")
        else:                                # 設定
            amt = float(a[1])
            CAP_FILE.parent.mkdir(parents=True, exist_ok=True)
            CAP_FILE.write_text(json.dumps({"capital": amt}, ensure_ascii=False), encoding="utf-8")
            print(f"現金池設為 {amt:,.0f} 元(之後所有執行/排程都用這個金額,cash auto 可恢復)")
        return 0

    d = load()

    if cmd == "rm" and len(a) == 2:
        code = a[1]
        if d.pop(code, None) is not None:
            save(d); print(f"已刪 {code}")
        else:
            print(f"帳本沒有 {code}")
        return 0

    if cmd in ("buy", "sell", "set") and len(a) >= 3:
        code = a[1]; qty = int(a[2])
        px = float(a[3]) if len(a) >= 4 else 0.0
        cur = d.get(code, {"qty": 0, "cost": 0.0})
        cq, cc = int(cur.get("qty", 0)), float(cur.get("cost", 0.0))

        if cmd == "set":
            if qty <= 0:
                d.pop(code, None); print(f"已清 {code}")
            else:
                d[code] = {"qty": qty, "cost": px}
                print(f"設定 {code} {names.get(code,code)} = {qty}股 @均價{px}")
        elif cmd == "buy":
            new_qty = cq + qty
            new_cost = (cq * cc + qty * px) / new_qty if new_qty else 0.0   # 加權平均成本
            d[code] = {"qty": new_qty, "cost": round(new_cost, 4)}
            print(f"買 {code} {names.get(code,code)} +{qty}股@{px} → 共{new_qty}股 均價{new_cost:.2f}")
        elif cmd == "sell":
            if qty > cq:
                print(f"⚠️ 賣 {qty} 股 > 持有 {cq} 股,只賣 {cq}"); qty = cq
            realized = (px - cc) * qty if px else 0.0
            left = cq - qty
            if left <= 0:
                d.pop(code, None)
            else:
                d[code] = {"qty": left, "cost": cc}   # 均價不變
            print(f"賣 {code} {names.get(code,code)} -{qty}股@{px} → 剩{left}股"
                  + (f"｜本筆已實現 {realized:+,.0f}元" if px else ""))
        save(d); return 0

    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main())
