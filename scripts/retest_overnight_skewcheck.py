"""Skew/robustness stress test on the pure-leg overnight decomposition.
Reuses retest_overnight_fix machinery indirectly by recomputing the daily
weighted overnight vs intraday series, then reports: mean, median, % positive
days, and winsorized mean (drop top/bottom 1% daily values). Read-only, offline.
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v5=_load("v5",ROOT/"scripts/exp_step1_v5.py"); v6=_load("v6",ROOT/"scripts/exp_step1_v6.py")
ec=_load("ec",ROOT/"scripts/exp_60d_entry_compare.py"); r60=_load("r60",ROOT/"scripts/run_backtest_60d.py")
features,_factors,oh=v5.features,v5._factors,v5.oh
END="2026-06-08"
u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
codes=list(u.keys()); turns={c:u[c].get("avg_turnover",0.0) for c in codes}
twii=features("0050"); feats={c:features(c) for c in codes}
opens,closes={},{}
for c in codes:
    o=oh(c); ds=sorted(d for d in o if d<=END)
    closes[c]={d:o[d]["close"] for d in ds}; opens[c]={d:o[d]["open"] for d in ds}
reb={}; lim={}
for c in codes:
    o=oh(c); ds=sorted(d for d in o if d<=END); cll=[]; m={}; s=set()
    for j,d in enumerate(ds):
        cll.append(o[d]["close"])
        if len(cll)>=25:
            try:
                g=rebound_signal(cll,turns.get(c,0.0))
                if g.get("fired"): m[d]=g["score"]*100
            except Exception: pass
        if j>0 and o[ds[j-1]]["close"]>0 and o[d]["close"]/o[ds[j-1]]["close"]-1>=0.095: s.add(d)
    reb[c]=m; lim[c]=s
alld=sorted({d for c in codes for d in closes.get(c,{}) if d<=END})
full=alld[-504:]
turn_pct={}
for d in full:
    vals=sorted(((c,feats[c][d]["turn"]) for c in codes if d in feats.get(c,{}) and feats[c][d]["turn"]>0),key=lambda x:x[1])
    turn_pct[d]={c:(i+1)/len(vals) for i,(c,_) in enumerate(vals)} if vals else {}
rb={d:bool(twii.get(d,{}).get("close") and twii[d].get("ma20") and twii[d]["close"]>twii[d]["ma20"]) for d in alld}
rows=[]
for d in sorted(set(full)):
    ir=twii.get(d,{}).get("ret20"); bull=rb.get(d); sc=[]
    for c in codes:
        f=feats.get(c,{})
        if d not in f or math.isnan(f[d].get("ma20",float("nan"))): continue
        v=ec.h_score(_factors(f[d],ir),turn_pct.get(d,{}).get(c,0.5)) if bull else reb.get(c,{}).get(d,0.0)
        if v>0: sc.append((v,c))
    sc.sort(reverse=True)
    for v,c in sc[:ec.TOPN]: rows.append((d,c,v/100))
sig=defaultdict(dict); tickers=set()
for d,tk,e in rows: tickers.add(tk); sig[d][tk]=e
cal=sorted(set(full))
on_series=[]; id_series=[]
for i in range(len(cal)-1):
    d,e=cal[i],cal[i+1]; todays=sig.get(d,{})
    edges={tk:ed for tk,ed in todays.items()}
    if not edges: continue
    ranked=sorted([tk for tk in edges if edges[tk]>0],key=lambda t:edges[t],reverse=True)
    sel=ranked[:r60.MAX_SIGNALS]
    if len(ranked)>r60.MAX_SIGNALS and edges[ranked[r60.MAX_SIGNALS]]>=edges[ranked[r60.MAX_SIGNALS-1]]*r60.TIE_RATIO:
        sel=ranked[:r60.MAX_SIGNALS+1]
    confs=[todays[tk] for tk in sel if tk in todays]; avg=sum(confs)/len(confs) if confs else 0.0
    expo=min(r60.EXPOSURE_CAP,max(r60.EXPOSURE_FLOOR,avg)) if sel else 0.0
    wsum=sum(edges[tk] for tk in sel)
    if wsum<=0 or expo<=0: continue
    on_d=id_d=0.0; ok=False
    for tk in sel:
        w=expo*edges[tk]/wsum; cd=closes.get(tk,{}).get(d); oe=opens.get(tk,{}).get(e); ce=closes.get(tk,{}).get(e)
        if not(cd and oe and ce) or cd<=0 or oe<=0: continue
        on_d+=w*(oe/cd-1); id_d+=w*(ce/oe-1); ok=True
    if ok: on_series.append(on_d); id_series.append(id_d)
def stats(x):
    n=len(x); s=sorted(x); mean=sum(x)/n
    med=s[n//2]; pos=sum(1 for v in x if v>0)/n
    k=max(1,int(n*0.01)); w=s[k:n-k]; wmean=sum(w)/len(w)
    return mean*100,med*100,pos*100,wmean*100,n
om,omd,op,owm,n=stats(on_series); im,imd,ip,iwm,_=stats(id_series)
print(f"OVERNIGHT seg: mean {om:+.4f}% median {omd:+.4f}% %pos {op:.0f}% winsor1% {owm:+.4f}% n={n}")
print(f"INTRADAY seg:  mean {im:+.4f}% median {imd:+.4f}% %pos {ip:.0f}% winsor1% {iwm:+.4f}%")
print(f"ON-ID diff mean {om-im:+.4f}%  winsorized {owm-iwm:+.4f}%")
