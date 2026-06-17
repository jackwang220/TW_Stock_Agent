"""B2 在「真實資金=只押~3檔」限制下能不能穩定用 + 分散版科學診斷。
全部報 前段α/後段α(兩段都正才算穩)。
A baseline(3檔,無B2) B tilt-pool4(3檔,原版) C 選股-pool8(3檔,B2從更寬H池挑)
D 純B2選-pool8(3檔,取CV最高3檔) E 分散sleeve(10檔,純診斷不可執行)
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
sim5,v6=nls.sim5,nls.v6; END=nls.END; ec=nls.ec; r60=ec.r60

df=pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv",encoding="utf-8")
df["Date"]=pd.to_datetime(df["Date"]); df["Stock_Code"]=df["Stock_Code"].astype(str)
ag=df.groupby(["Date","Stock_Code"]).agg(wt_std=("Weight(%)",lambda s:s.std(ddof=0)),wt_mean=("Weight(%)","mean")).reset_index()
ag["wt_cv"]=ag["wt_std"]/ag["wt_mean"].replace(0,np.nan); ag["d"]=ag["Date"].dt.strftime("%Y-%m-%d")
ag["zcv"]=ag.groupby("d")["wt_cv"].transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0.0).fillna(0.0)
ZCV={(r.d,r.Stock_Code):r.zcv for r in ag.itertuples()}

u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
codes=list(u.keys()); names={c:u[c].get("name",c) for c in codes}; turns={c:u[c].get("avg_turnover",0.0) for c in codes}
twii=features("0050"); feats={c:features(c) for c in codes}
opens,closes={},{}
for c in codes+["0050"]:
    o=oh(c); opens[c]={d:o[d]["open"] for d in o}; closes[c]={d:o[d]["close"] for d in o}
alld=sorted({d for c in codes for d in closes.get(c,{}) if d<=END}); sig=alld[-504:]
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
# 自建候選池(可調寬度 P)
def build(P):
    bull={d:bool(twii.get(d,{}).get("close") and twii[d].get("ma20") and twii[d]["close"]>twii[d]["ma20"]) for d in sig}
    cд={}
    for d in sig:
        ir=twii.get(d,{}).get("ret20"); sc=[]
        for c in codes:
            f=feats.get(c,{})
            if d not in f or math.isnan(f[d].get("ma20",float("nan"))): continue
            val=(ec.h_score(_factors(f[d],ir),turn_pct.get(d,{}).get(c,0.5)) if bull[d] else reb.get(c,{}).get(d,0.0))
            if val>0: sc.append((val,c))
        sc.sort(reverse=True); cд[d]=sc[:P]
    return cд
c4=build(4); c8=build(8)
b2d=sorted({d for d in sig if d>="2025-06-17"}); cut=b2d[int(len(b2d)*0.6)]
SEG={"前段":set(d for d in b2d if d<cut),"後段":set(d for d in b2d if d>=cut)}

def rows(mode,dayset,k=0.3):
    rw=[]
    for d in sig:
        if d not in dayset: continue
        if mode=="base":   # 3檔=pool4給H edge,sim取top3
            for v,c in c4[d]: rw.append((d,c,v/100.0))
        elif mode=="tilt4":
            for v,c in c4[d]: rw.append((d,c,(v/100.0)*(1+k*math.tanh(ZCV.get((d,c),0.0)))))
        elif mode=="tilt8":   # 更寬池,B2可把H排名5-8的拉進top3
            for v,c in c8[d]: rw.append((d,c,(v/100.0)*(1+k*math.tanh(ZCV.get((d,c),0.0)))))
        elif mode=="select8": # pool8裡取CV最高3檔,給H-based sizing
            cs=sorted(c8[d],key=lambda x:ZCV.get((d,x[1]),0.0),reverse=True)[:3]
            for v,c in cs: rw.append((d,c,v/100.0))
        elif mode=="sleeve10": # 分散診斷:CV最高10檔(等權sizing)
            cs=sorted(c8[d],key=lambda x:ZCV.get((d,x[1]),0.0),reverse=True)
            # pool8最多8檔,改用全universe CV top10(僅診斷)
            pool=[(ZCV.get((d,c),-9),c) for v,c in [(0,c) for c in codes] if d in feats.get(c,{}) and (d,c) in ZCV]
            for z,c in sorted(pool,reverse=True)[:10]: rw.append((d,c,0.5))
    return rw
def run(rw): return sim5(rw,opens,closes,limitup,switch_cost_mult=1.0)
def bench(ds): dd=sorted(x for x in ds if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
bseg={s:bench(SEG[s]) for s in SEG}
CFG=[("A baseline 3檔(無B2)","base"),("B tilt pool4 3檔(原版)","tilt4"),
     ("C tilt pool8 3檔(B2選股)","tilt8"),("D 純B2選 pool8 3檔","select8"),
     ("E 分散sleeve 10檔(診斷不可執行)","sleeve10")]
print(f"切分 {cut};0050: 前段{bseg['前段']:+.0f}% 後段{bseg['後段']:+.0f}%\n")
print(f"{'配置':<26}{'前段α':>7}{'後段α':>7}{'持股':>6}{'曝險':>6}  兩段都正?")
base={}
for lab,mode in CFG:
    if mode=="sleeve10": r60.MAX_SIGNALS=10
    a={}; r_={}
    for s in SEG:
        r=run(rows(mode,SEG[s])); a[s]=r["ret"]-bseg[s]; r_[s]=r
    r60.MAX_SIGNALS=3
    if mode=="base": base=a
    both="✅" if a["前段"]>base.get("前段",a["前段"])-0.1 and a["後段"]>base.get("後段",a["後段"])-0.1 and (mode=="base" or (a["前段"]>base["前段"] and a["後段"]>base["後段"])) else ("⚠️只一段贏baseline" if mode!="base" else "—")
    if mode!="base":
        win1=a["前段"]>base["前段"]; win2=a["後段"]>base["後段"]
        both="✅兩段都贏baseline" if win1 and win2 else ("❌方向相反" if win1!=win2 else "兩段都輸")
    print(f"{lab:<26}{a['前段']:>+7.0f}{a['後段']:>+7.0f}{r_['後段'].get('avg_pos',0):>6.1f}{r_['後段'].get('avg_expo',0)*100:>5.0f}%  {both}")
print("\n判讀:C/D是3檔可執行版,要『兩段都贏baseline』才算B2在真實資金下穩定可用;E是分散診斷(確認訊號本身收得到)。")
