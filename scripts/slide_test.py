"""滑動倉位 vs 固定(binary/half)。倉位隨「跌破MA60的深度」滑:小回檔幾乎不減、深空頭才大砍。
slide-below kN: 0050在MA60上=滿倉;跌破後 scale=clip(1+k*(ratio-1),0,1),k=斜率(越大越快砍到0)。
slide-dd: 隨0050距60日高點回撤深度滑。
比 binary/half,看「只在深跌才砍」是否兩全(保留報酬+空頭保護)。含切換成本。
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
def dd60(d):
    i=od.index(d); hi=max(oc[od[k]] for k in range(max(0,i-59),i+1)); return (oc[d]/hi-1)
def clip(x,a=0.0,b=1.0): return max(a,min(b,x))
def sig(d,fn):
    pdd=prevd.get(d); return fn(pdd) if pdd in oc else 1.0
def slidebelow(k):
    def f(d):
        return sig(d,lambda x:(1.0 if oc[x]>= (ma(x,60) or oc[x]) else clip(1+k*(oc[x]/(ma(x,60) or oc[x])-1))))
    return f
def slidedd(div):  # 隨60日回撤深度:dd=0→1, dd=-div→0
    def f(d): return sig(d,lambda x:clip(1+dd60(x)/div))
    return f
SCALES={
 "基準(全)":         lambda d:1.0,
 "binary<MA60":     lambda d:sig(d,lambda x:0.0 if oc[x]<(ma(x,60) or 1e9) else 1.0),
 "half<MA60":       lambda d:sig(d,lambda x:0.5 if oc[x]<(ma(x,60) or 1e9) else 1.0),
 "滑動below k5":     slidebelow(5),
 "滑動below k8":     slidebelow(8),
 "滑動below k12":    slidebelow(12),
 "滑動回撤/15%":     slidedd(0.15),
 "滑動回撤/10%":     slidedd(0.10),
}
SW=0.003
def run(scfn,s=None,e=None):
    idx=[i for i in range(len(cal)) if (s is None or s<=cal[i]<=e)]
    eq=contributed; prev=1.0; p=[]
    for i in idx:
        sc=scfn(cal[i]); c=SW*AVGEXPO*eq*abs(sc-prev); day=pnl[i]*sc-c; eq+=day; p.append(day); prev=sc
    p=np.array(p); tot=sum(p)/contributed*100
    cum=np.cumsum(p); mdd=(np.maximum.accumulate(cum)-cum).max()/contributed*100 if len(p) else 0
    return tot,mdd
def per(fn,s,e): return run(fn,s,e)[0]
print(f"\n{'方法':<16}{'報酬%':>8}{'MDD%':>7}{'2022':>7}{'2025-04':>9}{'牛爆衝':>8}{'平均scale':>10}")
for name,fn in SCALES.items():
    t,m=run(fn)
    b22=per(fn,"2022-01-01","2022-12-31"); b25=per(fn,"2025-03-01","2025-04-30")
    bull=per(fn,"2025-10-20","2025-11-30")+per(fn,"2026-04-01","2026-05-31")
    avgs=np.mean([fn(cal[i]) for i in range(len(cal))])
    print(f"{name:<16}{t:>+8.0f}{m:>7.0f}{b22:>+7.0f}{b25:>+9.0f}{bull:>+8.0f}{avgs:>10.2f}")
print("\n判讀:滑動法若能『同平均scale下 報酬≥half且2022更少』=兩全;若只是介於binary/half之間=沒新增價值、徒增參數(overfit風險)。")
