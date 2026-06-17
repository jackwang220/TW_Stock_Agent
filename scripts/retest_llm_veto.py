"""重測 llm_veto:用 60d 快取的辯論 verdict join 真實 5日/隔日報酬,量測 LLM 否決的真實分辨力。
不打網路,純讀 data/_60dllm_debate.json + ohlcv_cache。"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

d = json.loads((ROOT/"data/_60dllm_debate.json").read_text(encoding="utf-8"))
rows = []  # (date, code, verdict, bull, bear, fwd1_openopen, fwd5)
for k, v in d.items():
    dt, c = k.split("_")
    vd, bull, bear = v
    oh = get_daily_ohlcv(c)
    ds = sorted(x for x in oh if x >= dt)
    if len(ds) < 7: continue
    # 隔日開盤買→D+1開盤後賣 不現實;用 close@d → close@d+5 (跟 veto 原本 5日 horizon 一致)
    fwd5 = oh[ds[5]]["close"]/oh[ds[0]]["close"]-1 if len(ds)>5 else None
    fwd1 = oh[ds[1]]["close"]/oh[ds[0]]["close"]-1 if len(ds)>1 else None
    rows.append((dt,c,vd,bull,bear,fwd1,fwd5))

def stats(label, rs):
    n=len(rs)
    if not n: return
    f5=[r[6] for r in rs if r[6] is not None]
    avg5=sum(f5)/len(f5)*100 if f5 else 0
    win=sum(1 for x in f5 if x>0)/len(f5)*100 if f5 else 0
    print(f"{label:22} n={n:3} 均5日={avg5:+6.2f}% 勝率={win:4.0f}%")

print("=== 全部候選 ===")
stats("ALL", rows)
print("\n=== 依 verdict 分組(關鍵:REJECT 的票真的比較爛嗎?)===")
for vd in ["PASS","WARN","REJECT"]:
    stats(vd, [r for r in rows if r[2]==vd])
print("\n=== 二分:KEEP(非REJECT) vs REJECT ===")
stats("KEEP(PASS+WARN)", [r for r in rows if r[2]!="REJECT"])
stats("REJECT", [r for r in rows if r[2]=="REJECT"])

# 分辨力:REJECT 在 winners/losers 各占多少
WIN_TH=0.08
winners=[r for r in rows if r[6] is not None and r[6]>WIN_TH]
losers=[r for r in rows if r[6] is not None and r[6]<-WIN_TH]
def rejrate(rs): 
    return sum(1 for r in rs if r[2]=="REJECT")/len(rs)*100 if rs else 0
print(f"\n=== 分辨力(>+8% winner / <-8% loser)===")
print(f"winners n={len(winners)} REJECT率={rejrate(winners):.0f}%")
print(f"losers  n={len(losers)} REJECT率={rejrate(losers):.0f}%")
print(f"分辨力差(loser-winner REJECT率)= {rejrate(losers)-rejrate(winners):+.1f}pp")

# bear_score 作為連續訊號的 AUC-ish:bear高 是否對應 fwd5低?
import statistics
pairs=[(r[4],r[6]) for r in rows if r[6] is not None]
# spearman-ish: correlation of bear vs fwd5
xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]
n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
cov=sum((x-mx)*(y-my) for x,y in pairs)/n
sx=statistics.pstdev(xs); sy=statistics.pstdev(ys)
print(f"\nbear_score vs fwd5 相關係數 = {cov/(sx*sy):+.3f}  (負=bear高→報酬低=有效;~0=沒訊號)")
bull_pairs=[(r[3],r[6]) for r in rows if r[6] is not None]
xs2=[p[0] for p in bull_pairs]; mx2=sum(xs2)/n; sx2=statistics.pstdev(xs2)
cov2=sum((x-mx2)*(y-my) for x,y in bull_pairs)/n
print(f"bull_score vs fwd5 相關係數 = {cov2/(sx2*sy):+.3f}  (正=bull高→報酬高=有效)")
