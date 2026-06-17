"""B2 regime gate 第2輪:先用作弊(oracle)確認『關對就兩段都贏』,再試真實可用的非大盤方向訊號。
gate候選:
  oracle      = 完美後見(只在後段開B2)→上界,確認機制
  factor_pay  = 動能因子『實際報酬』trailing20d(top1/3動能股-bot1/3 的實現報酬);低=因子冷→開B2
  Hself       = baseline H 自己trailing20d vs 0050(自身近況差→開B2)
全史校準閾值。報 前段α/後段α,看誰能『兩段都贏baseline』。
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
build_candidates,sim5,v6=nls.build_candidates,nls.sim5,nls.v6; END=nls.END

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
alld=sorted({d for c in codes for d in closes.get(c,{}) if d<=END})

# 每日股票報酬 & 60日動能(算實現動能因子報酬)
dret={}  # (c,d)->當日報酬
mom60={} # (c,d)->60日報酬
for c in codes:
    o=oh(c); ds=sorted(x for x in o if x<=END)
    for j,d in enumerate(ds):
        if j>0 and o[ds[j-1]]["close"]>0: dret[(c,d)]=o[d]["close"]/o[ds[j-1]]["close"]-1
        if j>=60 and o[ds[j-60]]["close"]>0: mom60[(c,d)]=o[d]["close"]/o[ds[j-60]]["close"]-1
# 每日動能因子實現報酬 = top1/3 mom60 籃 - bot1/3 籃 的當日equal-weight報酬
fac_daily={}
for d in alld:
    mm=[(mom60[(c,d)],c) for c in codes if (c,d) in mom60 and (c,d) in dret]
    if len(mm)<15: continue
    mm.sort(); k=len(mm)//3
    top=[c for _,c in mm[-k:]]; bot=[c for _,c in mm[:k]]
    rt=np.mean([dret[(c,d)] for c in top]); rb=np.mean([dret[(c,d)] for c in bot])
    fac_daily[d]=(rt-rb)*100
def roll(dic,d,n):
    ds=[x for x in alld if x<=d][-n:]; vv=[dic[x] for x in ds if x in dic]; return np.mean(vv) if vv else np.nan
FAC={d:roll(fac_daily,d,20) for d in alld}   # 動能因子近20日實現payoff
TH_FAC=np.median([v for v in FAC.values() if not pd.isna(v)])
print(f"動能因子payoff全史中位數={TH_FAC:+.3f}")

sig=alld[-504:]
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
b2d=sorted({d for d in sig if d>="2025-06-17"}); cut=b2d[int(len(b2d)*0.6)]
FRONT=set(d for d in b2d if d<cut); BACK=set(d for d in b2d if d>=cut)
print(f"動能因子payoff: 前段{np.mean([FAC[d] for d in FRONT if not pd.isna(FAC.get(d))]):+.3f} 後段{np.mean([FAC[d] for d in BACK if not pd.isna(FAC.get(d))]):+.3f} (閾值{TH_FAC:+.3f})")

def use_b2(mode,d):
    if mode=="always": return True
    if mode=="oracle": return d in BACK           # 作弊:只在後段開
    if mode=="factor": return FAC.get(d) is not None and not pd.isna(FAC.get(d)) and FAC[d]<TH_FAC  # 因子冷→開
    return False
def rows(mode,dayset,k=0.3):
    rw=[]
    for d in sig:
        if d not in dayset: continue
        ub=use_b2(mode,d)
        for v,c in cands[d]:
            e=v/100.0
            if ub and mode not in ("base",): e*=(1+k*math.tanh(ZCV.get((d,c),0.0)))
            rw.append((d,c,e))
    return rw
def run(rw): return sim5(rw,opens,closes,limitup,switch_cost_mult=1.0)
def bench(ds): dd=sorted(x for x in ds if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
bF,bB=bench(FRONT),bench(BACK)
print(f"\n0050: 前段{bF:+.0f}% 後段{bB:+.0f}%")
print(f"\n{'策略':<22}{'前段α':>7}{'後段α':>7}  兩段都贏baseline?")
baseF=run(rows('base',FRONT))['ret']-bF; baseB=run(rows('base',BACK))['ret']-bB
print(f"{'baseline 無B2':<22}{baseF:>+7.0f}{baseB:>+7.0f}  —")
for mode,lab in [("always","B2 always"),("oracle","B2 oracle(作弊上界)"),("factor","B2 gate=動能因子冷")]:
    f=run(rows(mode,FRONT))['ret']-bF; b=run(rows(mode,BACK))['ret']-bB
    ok='✅兩段都贏' if f>=baseF-0.5 and b>baseB else ('❌'+('前段輸' if f<baseF-0.5 else '')+('後段沒贏' if b<=baseB else ''))
    print(f"{lab:<22}{f:>+7.0f}{b:>+7.0f}  {ok}")
print("\n判讀:oracle『兩段都贏』=機制成立(關對就行)→值得找真實detector;若連oracle都救不了前段=機制本身有問題。")
print("     factor gate 若接近oracle=找到可用的非大盤方向訊號。")
