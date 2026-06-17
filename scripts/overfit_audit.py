"""現行H雙引擎 overfitting 稽核:它真的最好,還是搜出來的in-sample贏家?
1) 各regime alpha → edge是否集中在1個regime(集中=脆/可能運氣)
2) 去掉最強regime後 平均alpha 還剩多少
3) 參數穩健性:INC/MAX_SIGNALS/EDGE_DECAY 擾動 → 現行是尖峰(overfit)還是平原(robust)
4) vs naive(每日成交值top-N等權,不用H評分)→ H的edge厚不厚
全史、扣0050、扣成本。
"""
import sys, json, math
import pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove(); logger.add(sys.stderr,level="ERROR")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
import importlib.util as U
def load(n,p):
    s=U.spec_from_file_location(n,p); m=U.module_from_spec(s); s.loader.exec_module(m); return m
nls=load("nls",ROOT/"scripts/news_llm_select.py")
features,_factors,oh=nls.features,nls._factors,nls.oh
build_candidates,sim5,v6=nls.build_candidates,nls.sim5,nls.v6; END=nls.END; REGIMES=nls.REGIMES; ec=nls.ec; r60=ec.r60
u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
codes=list(u.keys()); names={c:u[c].get("name",c) for c in codes}; turns={c:u[c].get("avg_turnover",0.0) for c in codes}
twii=features("0050"); feats={c:features(c) for c in codes}
opens,closes={},{}
for c in codes+["0050"]:
    o=oh(c); opens[c]={d:o[d]["open"] for d in o}; closes[c]={d:o[d]["close"] for d in o}
sig=sorted({d for c in codes for d in closes.get(c,{}) if d<=END})
reb,limitup={},{}
for c in codes:
    o=oh(c); ds=sorted(d for d in o if d<=END); cl=[]; m={}; s=set()
    for j,d in enumerate(ds):
        cl.append(o[d]["close"])
        if len(cl)>=25:
            try:
                gg=rebound_signal(cl,turns.get(c,0.0))
                if gg.get("fired"): m[d]=gg["score"]*100
            except Exception: pass
        if j>0 and o[ds[j-1]]["close"]>0 and o[d]["close"]/o[ds[j-1]]["close"]-1>=0.095: s.add(d)
    reb[c]=m; limitup[c]=s
turn_pct={}
for d in sig:
    vals=sorted(((c,feats[c][d]["turn"]) for c in codes if d in feats.get(c,{}) and feats[c][d]["turn"]>0),key=lambda x:x[1])
    turn_pct[d]={c:(i+1)/len(vals) for i,(c,_) in enumerate(vals)} if vals else {}
cands,_=build_candidates(codes,names,feats,twii,reb,turn_pct,sig)
def bench(ds): dd=sorted(x for x in ds if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
REG=list(REGIMES.items()); rdays={rn:set(d for d in sig if s<=d<=e) for rn,(s,e) in REG}; breg={rn:bench(dd) for rn,dd in rdays.items()}
def Hrows(dayset): return [(d,c,v/100.0) for d in sig if d in dayset for v,c in cands[d]]
def naive_rows(dayset,N=3):  # 每日成交值top-N等權(不用H評分)
    rw=[]
    for d in sig:
        if d not in dayset: continue
        tv=sorted(((feats[c][d]["turn"],c) for c in codes if d in feats.get(c,{}) and feats[c][d].get("turn",0)>0),reverse=True)[:N]
        for _,c in tv: rw.append((d,c,0.5))
    return rw
def alpha(rowfn,dayset,bv):
    r=sim5(rowfn(dayset),opens,closes,limitup,switch_cost_mult=1.0); return r["ret"]-bv if r else None

print("=== 1) 現行H雙引擎 各regime alpha(看是否集中在1個regime)===")
ra={}
for rn,_ in REG:
    a=alpha(Hrows,rdays[rn],breg[rn]); ra[rn]=a
    print(f"  {rn}: {a:+.0f}")
vals=[v for v in ra.values() if v is not None]
print(f"  平均 {np.mean(vals):+.0f} / 最差 {min(vals):+.0f}")
best=max(ra,key=lambda k:ra[k])
print(f"  >>> 最強regime = {best}({ra[best]:+.0f});**去掉它後** 其餘平均 = {np.mean([v for k,v in ra.items() if k!=best]):+.0f}")

print("\n=== 2) vs naive(每日成交值top3等權,不用H評分)各regime ===")
for rn,_ in REG:
    na=alpha(lambda ds:naive_rows(ds),rdays[rn],breg[rn])
    print(f"  {rn}: H {ra[rn]:+.0f}  naive {na:+.0f}  H贏naive {ra[rn]-na:+.0f}")

print("\n=== 3) 參數穩健性(全史平均/最差 regime alpha):現行=尖峰還是平原? ===")
DEF=dict(INC=1.5,MS=r60.MAX_SIGNALS,ED=r60.EDGE_DECAY)
def run_params(inc,ms,ed):
    o_ms,o_ed=r60.MAX_SIGNALS,r60.EDGE_DECAY; r60.MAX_SIGNALS=ms; r60.EDGE_DECAY=ed
    aa=[]
    for rn,_ in REG:
        r=sim5([(d,c,v/100.0) for d in sig if d in rdays[rn] for v,c in cands[d]],opens,closes,limitup,incumbent=inc,switch_cost_mult=1.0)
        if r: aa.append(r["ret"]-breg[rn])
    r60.MAX_SIGNALS=o_ms; r60.EDGE_DECAY=o_ed
    return np.mean(aa),min(aa)
print(f"  {'設定':<22}{'平均α':>7}{'最差α':>7}")
print(f"  {'現行 INC1.5/MS3/ED0.8':<22}{run_params(1.5,3,0.8)[0]:>+7.0f}{run_params(1.5,3,0.8)[1]:>+7.0f}")
for inc in (1.0,2.0,2.5): m,w=run_params(inc,3,0.8); print(f"  {'INC='+str(inc):<22}{m:>+7.0f}{w:>+7.0f}")
for ms in (2,4): m,w=run_params(1.5,ms,0.8); print(f"  {'MAX_SIGNALS='+str(ms):<22}{m:>+7.0f}{w:>+7.0f}")
for ed in (0.7,0.9): m,w=run_params(1.5,3,ed); print(f"  {'EDGE_DECAY='+str(ed):<22}{m:>+7.0f}{w:>+7.0f}")
print("\n判讀:1)若alpha集中在1個regime、去掉就沒了→脆/可能運氣。2)若H≈naive→edge薄。3)若現行是尖峰、鄰近參數掉很多→overfit;平原→robust。")
