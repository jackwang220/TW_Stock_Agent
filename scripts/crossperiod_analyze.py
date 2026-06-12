"""跨期間 LLM 裁決分析:各 regime 的方向準確率(raw + alpha)、up選股的均報酬/均alpha/勝大盤%、反彈均alpha。
alpha = 個股報酬 - 大盤(0050)同期 → 扣 beta,才知道是真選股還是只是順勢。"""
import sys, csv
from collections import defaultdict
sys.path.insert(0, "src"); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR

rows = list(csv.DictReader(open(DATA_DIR / "crossperiod_results.csv", encoding="utf-8")))
by = defaultdict(list)
for r in rows:
    by[r["regime"]].append(r)

def f(x):
    try: return float(x)
    except: return None

print(f"{'regime':<11}{'方向n':>6}{'準確raw':>8}{'準確α':>7}{'up數':>5}{'up均報酬':>9}{'up均α':>8}{'up勝盤%':>8}{'反彈n':>6}{'反彈均α':>9}")
print("-" * 92)
for reg, rs in by.items():
    dp = [r for r in rs if r["predicted_direction"] in ("up", "down") and r["dir_correct"] != ""]
    acc = sum(int(r["dir_correct"]) for r in dp) / len(dp) if dp else 0
    dpa = [r for r in dp if f(r["alpha_1d"]) is not None]
    acca = (sum(1 for r in dpa if (r["predicted_direction"] == "up" and f(r["alpha_1d"]) > 0)
                or (r["predicted_direction"] == "down" and f(r["alpha_1d"]) < 0)) / len(dpa)) if dpa else 0
    ups = [r for r in rs if r["predicted_direction"] == "up" and f(r["ret_1d"]) is not None]
    upret = sum(f(r["ret_1d"]) for r in ups) / len(ups) if ups else 0
    upa = [r for r in ups if f(r["alpha_1d"]) is not None]
    upalpha = sum(f(r["alpha_1d"]) for r in upa) / len(upa) if upa else 0
    upbeat = sum(1 for r in upa if f(r["alpha_1d"]) > 0) / len(upa) * 100 if upa else 0
    rebs = [r for r in rs if r["rebound_fired"] == "1" and f(r["alpha_1d"]) is not None]
    reba = sum(f(r["alpha_1d"]) for r in rebs) / len(rebs) if rebs else 0
    print(f"{reg:<11}{len(dp):>6}{acc*100:>7.0f}%{acca*100:>6.0f}%{len(ups):>5}"
          f"{upret:>8.2f}%{upalpha:>7.2f}%{upbeat:>7.0f}%{len(rebs):>6}{reba:>8.2f}%")
print("\n判讀:")
print("  準確α>50% = 真選股(贏大盤);≈50% = 只是順勢(beta)。")
print("  up均α>0 = LLM 看漲的股隔日真的跑贏大盤=真edge;≈0或負 = 沒選股能力。")
print("  反彈均α = 純價格反彈訊號扣大盤後的隔日表現(跟 LLM 對照)。")