"""#4 Ablation:近期反彈觸發股,各拿掉一個 LLM 輸入(新聞/籌碼/營收),
看 PASS-vs-REJECT 的 alpha 價差(=filter 力)會不會崩。崩最多的=該輸入是 filter 關鍵。
temp=0(已設)確保差異來自輸入不是隨機。"""
import sys, csv, importlib.util
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import numpy as np
sys.path.insert(0, "src"); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
from tw_stock_agent.config import DATA_DIR, get_yaml_cfg
from tw_stock_agent.debate.bear import run_debate

spec = importlib.util.spec_from_file_location("cpv", "scripts/crossperiod_validate.py")
cpv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cpv)
_finmind_stock = cpv._finmind_stock

cfg = get_yaml_cfg()
RECENT = {"2023復甦", "2024多頭", "2025關稅崩", "2026最近"}
rows = [r for r in csv.DictReader(open(DATA_DIR / "crossperiod_results.csv", encoding="utf-8"))
        if r["rebound_fired"] == "1" and r["regime"] in RECENT and r.get("alpha_1d", "") != ""]
print(f"近期反彈觸發 {len(rows)} 檔做 ablation\n")

def f(x):
    try: return float(x)
    except: return None

def verdict_for(r):
    stock = _finmind_stock(r["code"], r["name"], r["as_of"])
    if stock is None: return None
    try:
        return run_debate(stock, [], historical_date=date.fromisoformat(r["as_of"])).verdict
    except Exception:
        return None

def run_config(name, news, inst, rev):
    cfg["debate"]["ablate_news"] = news
    cfg["debate"]["ablate_inst"] = inst
    cfg["debate"]["ablate_rev"] = rev
    with ThreadPoolExecutor(max_workers=8) as ex:
        vs = list(ex.map(verdict_for, rows))
    P = [f(r["alpha_1d"]) for r, v in zip(rows, vs) if v == "PASS"]
    R = [f(r["alpha_1d"]) for r, v in zip(rows, vs) if v == "REJECT"]
    sp = (np.mean(P) - np.mean(R)) if P and R else 0
    print(f"{name:<12} PASS n={len(P):>3} α{(np.mean(P) if P else 0):+.2f}%  "
          f"REJECT n={len(R):>3} α{(np.mean(R) if R else 0):+.2f}%  → 價差 {sp:+.2f}%")
    return sp

print(f"{'設定':<12}{'PASS/REJECT alpha → 價差(filter力)'}")
print("-" * 70)
base = run_config("全輸入基準", False, False, False)
run_config("關新聞", True, False, False)
run_config("關籌碼", False, True, False)
run_config("關營收", False, False, True)
print(f"\n基準價差 {base:+.2f}%。哪個『關掉』後價差掉最多 = 那個輸入是 LLM 守門的關鍵。")