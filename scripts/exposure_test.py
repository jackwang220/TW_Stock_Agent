"""regime 連續調曝險 vs 簡單規則 vs 固定降曝險(null)。回答:用訊號調曝險會不會比較好?
overlay:protected_pnl = pnl × scale_t(曝險倍數∈[0,1]);連續切換成本=|Δscale|×0.3%×曝險×權益。
關鍵對照:'固定0.7×'(只是de-lever不timing)——若regime法贏不過它=timing白工。
leak-safe:scale用前一交易日訊號。
"""
import sys, json, math
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove(); logger.add(sys.stderr,level="ERROR")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
import importlib.util as U
def load(n,p):
    s=U.spec_from_file_location(n,p); m=U.module_from_spec(s); s.loader.exec_module(m); return m
nine=load("nine",ROOT/"scripts/exp_step1_v9.py")
is_bull,h_score,_factors=nine.is_bull,nine.h_score,nine._factors
v5,v6,ec=nine.v5,nine.v6,nine.ec; sim5=ec.sim_buyclose_sellopen
features=v5.features; get_ohlcv=nine.get_daily_ohlcv; START=nine.START
u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
codes=list(u.keys()); turns={c:u[c].get("avg_turnover",0.0) for c in codes}
print("載入...",flush=True)
OH={c:get_ohlcv(c,start=START) for c in codes}; OH["0050"]=get_ohlcv("0050",start=START); v5._OH=OH
twii=features("0050"); feats={c:features(c) for c in codes}
alld=sorted({d for c in codes for d in OH[c]})
opens={c:{d:OH[c][d]["open"] for d in OH[c]} for c in codes+["0050"]}
closes={c:{d:OH[c][d]["close"] for d in OH[c]} for c in codes+["0050"]}
reb,limitup={},{}
for c in codes:
    ds=sorted(OH[c]); cl=[]; m={}; s=set()
    for j,d in enumerate(ds):
        cl.append(OH[c][d]["close"])
        if len(cl)>=25:
            try:
                g=rebound_signal(cl,turns.get(c,0.0))
                if g.get("fired"): m[d]=g["score"]*100
            except Exception: pass
        if j>0 and OH[c][ds[j-1]]["close"]>0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1>=0.095: s.add(d)
    reb[c]=m; limitup[c]=s
turn_pct={}
for d in alld:
    vals=sorted(((c,feats[c][d]["turn"]) for c in codes if d in feats.get(c,{}) and feats[c][d]["turn"]>0),key=lambda x:x[1])
    turn_pct[d]={c:(i+1)/len(vals) for i,(c,_) in enumerate(vals)} if vals else {}
rows=[]
for d in alld:
    ir=twii.get(d,{}).get("ret20"); bull=is_bull(twii.get(d,{}),"原版c>20MA")
    for c in codes:
        f=feats.get(c,{})
        if d not in f or math.isnan(f[d].get("ma20",float("nan"))): continue
        sc=h_score(_factors(f[d],ir),turn_pct.get(d,{}).get(c,0.5)) if bull else reb.get(c,{}).get(d,0.0)*1.5
        if sc>0: rows.append((d,c,sc/100))
base=sim5(rows,opens,closes,limitup,incumbent=1.5,sell_mode="exit_only")
cal=base["cal"]; pnl=np.array(base["day_pnl"]); contributed=base["contributed"]; AVGEXPO=base["avg_expo"]
oc=closes["0050"]; od=sorted(oc); prevd={od[i]:od[i-1] for i in range(1,len(od))}
def ma(d,n):
    i=od.index(d); return np.mean([oc[od[k]] for k in range(i-n+1,i+1)]) if i>=n-1 else None
def vol20(d):
    i=od.index(d)
    if i<20: return None
    rr=[oc[od[k]]/oc[od[k-1]]-1 for k in range(i-19,i+1)]; return np.std(rr)
volmed=np.median([v for v in (vol20(d) for d in od) if v])
def sig(d,fn):
    pdd=prevd.get(d); return fn(pdd) if pdd in oc else 1.0
def clip(x,a=0.0,b=1.0): return max(a,min(b,x))
SCALES={
 "基準(全曝險)":      lambda d:1.0,
 "固定0.7×(null)":   lambda d:0.7,
 "固定0.5×(null)":   lambda d:0.5,
 "A2 binary<MA60":  lambda d:sig(d,lambda x:0.0 if oc[x]<(ma(x,60) or 1e9) else 1.0),
 "half<MA60":       lambda d:sig(d,lambda x:0.5 if oc[x]<(ma(x,60) or 1e9) else 1.0),
 "趨勢連續(對MA60)":  lambda d:sig(d,lambda x:clip(0.5+8*((oc[x]/(ma(x,60) or oc[x]))-1))),
 "反波動(target/vol)":lambda d:sig(d,lambda x:clip((volmed/(vol20(x) or volmed)),0.3,1.0)),
 "趨勢×反波動":       lambda d:sig(d,lambda x:clip(0.5+8*((oc[x]/(ma(x,60) or oc[x]))-1))*clip((volmed/(vol20(x) or volmed)),0.3,1.2)),
}
SW=0.003
def run(scfn,s=None,e=None):
    idx=[i for i in range(len(cal)) if (s is None or s<=cal[i]<=e)]
    eq=contributed; prev=1.0; p=[]
    for i in idx:
        sc=scfn(cal[i])
        c=SW*AVGEXPO*eq*abs(sc-prev) if scfn else 0.0
        day=pnl[i]*sc - c; eq+=day; p.append(day); prev=sc
    p=np.array(p); tot=sum(p)/contributed*100
    cum=np.cumsum(p); mdd=(np.maximum.accumulate(cum)-cum).max()/contributed*100 if len(p) else 0
    return tot,mdd
def per(scfn,s,e): return run(scfn,s,e)[0]
print(f"\n{'方法':<20}{'報酬%':>8}{'MDD%':>7}{'2022':>7}{'2025-04':>9}{'牛爆衝':>8}{'平均scale':>10}")
for name,fn in SCALES.items():
    t,m=run(fn)
    b22=per(fn,"2022-01-01","2022-12-31"); b25=per(fn,"2025-03-01","2025-04-30")
    bull=per(fn,"2025-10-20","2025-11-30")+per(fn,"2026-04-01","2026-05-31")
    avgs=np.mean([fn(cal[i]) for i in range(len(cal))])
    print(f"{name:<20}{t:>+8.0f}{m:>7.0f}{b22:>+7.0f}{b25:>+9.0f}{bull:>+8.0f}{avgs:>10.2f}")
print("\n判讀:regime調曝險法 要在『同樣平均scale』下 比固定降曝險(null) 報酬高/2022跌少 才算timing有用;")
print("     否則=只是de-lever,沒必要搞複雜。反波動會砍高波動的牛爆衝→預期傷報酬。")
