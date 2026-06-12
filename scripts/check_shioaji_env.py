"""驗證永豐 Shioaji 設定是否就位(不印出密鑰內容,只顯示有沒有填 + 憑證檔在不在)。"""
import os, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

def mask(v): return f"已填(長度 {len(v)})" if v else "❌ 沒填"

ak = os.environ.get("SINOPAC_APIKEY", "")
sk = os.environ.get("SINOPAC_SECRETKEY", "")
cp = os.environ.get("SINOPAC_CA_PATH", "")
pw = os.environ.get("SINOPAC_CA_PASSWORD", "")

print("=== 永豐 Shioaji .env 檢查 ===")
print(f"  SINOPAC_APIKEY     : {mask(ak)}")
print(f"  SINOPAC_SECRETKEY  : {mask(sk)}")
print(f"  SINOPAC_CA_PATH    : {cp or '❌ 沒填'}")
print(f"  SINOPAC_CA_PASSWORD: {mask(pw)}")

if cp:
    p = Path(cp)
    if not p.is_absolute():
        p = ROOT / cp
    if p.exists():
        print(f"  憑證檔             : ✅ 找到 ({p}, {p.stat().st_size} bytes)")
    else:
        print(f"  憑證檔             : ❌ 找不到 → {p}")

ok = all([ak, sk, cp, pw]) and (Path(cp) if Path(cp).is_absolute() else ROOT / cp).exists()
print("\n" + ("✅ 全部就位,可以跑 shioaji_ca_test.py 了" if ok else "⚠️ 還有缺,補齊上面 ❌ 的項目"))