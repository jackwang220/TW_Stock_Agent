"""驗證 A2(0050<MA→空手)是不是穩健、扣成本後還在。
1) MA長度 plateau:MA40/60/90/120 都測,若都明顯幫→不是MA60 curve-fit
2) 扣切換成本:每次進出場扣 ~0.3%×曝險×當下權益
3) OOS:用2021-2023(含2022空頭)挑MA → 套2024-2026(含2025-04跌)看generalize
leak-safe:濾網用前一交易日。
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
def mask_ma(N):  # 前一交易日 0050<MAN → 0(現金) 否則1
    out={}
    for d in cal:
        pdd=prevd.get(d); mn=ma(pdd,N) if pdd in oc else None
        out[d]= 0.0 if (mn is not None and oc[pdd]<mn) else 1.0
    return out
SWITCH=0.003  # 每次切換單向成本 ~0.3%(賣0.44/買0.14+滑;取保守單向均值)
def metrics(maskmap, s=None, e=None, cost=True):
    idx=[i for i in range(len(cal)) if (s is None or s<=cal[i]<=e)]
    eq=contributed; tog=0; p=[]
    prev_m=1.0
    for i in idx:
        m=maskmap.get(cal[i],1.0) if maskmap else 1.0
        c=0.0
        if cost and maskmap and m!=prev_m:  # 切換:扣 單向成本 × 曝險 × 當下權益
            c=SWITCH*AVGEXPO*eq; tog+=1
        day=pnl[i]*m - c
        eq+=day; p.append(day); prev_m=m
    p=np.array(p)
    tot=sum(p)/contributed*100
    cum=np.cumsum(p); mdd=(np.maximum.accumulate(cum)-cum).max()/contributed*100 if len(p) else 0
    return tot,mdd,tog
def period(maskmap,s,e,cost=True): return metrics(maskmap,s,e,cost)[0]

print(f"基準曝險~{AVGEXPO:.0%};切換成本 單向{SWITCH:.1%}×曝險×權益")
print("\n=== (1) MA長度 plateau(全期,含切換成本)===")
print(f"{'濾網':<12}{'報酬%':>8}{'MDD%':>7}{'2022':>7}{'2025-04':>9}{'牛爆衝':>8}{'切換次':>7}")
none_t,none_m,_=metrics(None);
nb22=period(None,"2022-01-01","2022-12-31"); nb25=period(None,"2025-03-01","2025-04-30")
nbull=period(None,"2025-10-20","2025-11-30")+period(None,"2026-04-01","2026-05-31")
print(f"{'無防禦':<12}{none_t:>+8.0f}{none_m:>7.0f}{nb22:>+7.0f}{nb25:>+9.0f}{nbull:>+8.0f}{0:>7}")
for N in (40,60,90,120):
    mm=mask_ma(N); t,m,tog=metrics(mm)
    b22=period(mm,"2022-01-01","2022-12-31"); b25=period(mm,"2025-03-01","2025-04-30")
    bull=period(mm,"2025-10-20","2025-11-30")+period(mm,"2026-04-01","2026-05-31")
    print(f"{'MA'+str(N):<12}{t:>+8.0f}{m:>7.0f}{b22:>+7.0f}{b25:>+9.0f}{bull:>+8.0f}{tog:>7}")

print("\n=== (2) OOS:2021-2023挑MA(看哪個train最佳)→ 套2024-2026(含成本)===")
tr=("2021-01-01","2023-12-31"); te=("2024-01-01","2026-12-31")
print(f"{'濾網':<10}{'train報酬':>10}{'train MDD':>10}{'test報酬':>10}{'test MDD':>10}")
base_te_t,base_te_m,_=metrics(None,*te)
print(f"{'無防禦':<10}{metrics(None,*tr)[0]:>+10.0f}{metrics(None,*tr)[1]:>10.0f}{base_te_t:>+10.0f}{base_te_m:>10.0f}")
tr_scores={}
for N in (40,60,90,120):
    mm=mask_ma(N); tt,tm,_=metrics(mm,*tr); et,em,_=metrics(mm,*te)
    tr_scores[N]=(tt,tm,et,em)
    print(f"{'MA'+str(N):<10}{tt:>+10.0f}{tm:>10.0f}{et:>+10.0f}{em:>10.0f}")
bestN=max(tr_scores,key=lambda N: tr_scores[N][0]/max(tr_scores[N][1],1))  # train 報酬/MDD 最佳
print(f"\n→ train(2021-23,含2022空頭)以報酬/MDD選出 MA{bestN};它在test(2024-26,含2025-04跌)報酬{tr_scores[bestN][2]:+.0f}/MDD{tr_scores[bestN][3]:.0f} vs 無防禦{base_te_t:+.0f}/{base_te_m:.0f}")
print("\n判讀:若MA40~120全幫(plateau)→非curve-fit;若train選的MA在test也優於無防禦→OOS generalize、可上線。")
