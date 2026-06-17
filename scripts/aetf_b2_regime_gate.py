"""regime 條件式 B2:動能強→關B2,動能弱→開B2。
regime定義用『全部歷年股價』校準閾值(非只B2年),動態每日切換。
測幾種動能強弱定義,看哪個能分開前/後段,再gate B2,看gated版是否『兩段都贏baseline』。
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

# B2訊號
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
# === regime 訊號(用全史校準)===
oc=closes["0050"]; od=sorted(oc)
ret=lambda d,n: (oc[d]/oc[od[od.index(d)-n]]-1)*100 if d in oc and od.index(d)>=n else np.nan
# G1 0050動能(60日報酬) G2 0050動能(20日) G3 動能因子payoff(高60d動能籃-低籃,近20日)
# G3:每日跨股60日動能,top1/3 vs bot1/3 當日報酬,近20日累積
def stock_ret(c,d,n):
    o=oh(c); ds=sorted(x for x in o if x<=d)
    if len(ds)<=n or d not in o: return np.nan
    p0=o[ds[-n-1]]["close"]; return (o[d]["close"]/p0-1) if p0>0 else np.nan
# 預算 G3 太重→用 feats 的 ret20 近似動能因子;G3=每日(top quintile ret20 mean - bottom)的近20日均
alld_full=[d for d in alld]
g3raw={}
for d in alld_full:
    vals=[(feats[c][d].get("ret20"),c) for c in codes if d in feats.get(c,{}) and feats[c][d].get("ret20") is not None and not math.isnan(feats[c][d].get("ret20",float("nan")))]
    if len(vals)<10: continue
    vals.sort(); k=len(vals)//5
    # 用「動能因子當日是否在賺」近似:高動能股近期報酬 vs 低 → 這裡用 ret20 分佈的 top-bot 差當 proxy
    g3raw[d]=np.mean([v for v,_ in vals[-k:]])-np.mean([v for v,_ in vals[:k]])
def roll(dic,d,n):
    ds=[x for x in alld_full if x<=d][-n:]; vv=[dic[x] for x in ds if x in dic]
    return np.mean(vv) if vv else np.nan
G={}
for d in alld_full:
    G[d]={"G1_0050_60d":ret(d,60),"G2_0050_20d":ret(d,20),"G3_momfactor":roll(g3raw,d,20)}
# 全史閾值=中位數
import numpy as _np
def thresh(key):
    vv=[G[d][key] for d in alld_full if not pd.isna(G[d].get(key))]; return _np.median(vv)
TH={k:thresh(k) for k in ("G1_0050_60d","G2_0050_20d","G3_momfactor")}
print("全史中位數閾值:",{k:round(v,2) for k,v in TH.items()})

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

# regime 能不能分開兩段?
print("\n=== 各 regime 訊號在 前段 vs 後段 的均值(能分開才能 gate)===")
for k in ("G1_0050_60d","G2_0050_20d","G3_momfactor"):
    f=_np.mean([G[d][k] for d in FRONT if not pd.isna(G[d].get(k))])
    b=_np.mean([G[d][k] for d in BACK if not pd.isna(G[d].get(k))])
    print(f"  {k}: 前段 {f:+.2f}  後段 {b:+.2f}  (閾值{TH[k]:+.2f}) → 前段{'強' if f>TH[k] else '弱'}/後段{'強' if b>TH[k] else '弱'}")

def rows(mode,dayset,k=0.3,gate=None):
    rw=[]
    for d in sig:
        if d not in dayset: continue
        strong = (G[d].get(gate) is not None and not pd.isna(G[d].get(gate)) and G[d][gate]>TH[gate]) if gate else False
        use_b2 = (mode=="always") or (mode=="gated" and not strong)   # gated:只在動能弱時開B2
        for v,c in cands[d]:
            e=v/100.0
            if use_b2 and mode!="base": e*= (1+k*math.tanh(ZCV.get((d,c),0.0)))
            rw.append((d,c,e))
    return rw
def run(rw): return sim5(rw,opens,closes,limitup,switch_cost_mult=1.0)
def bench(ds): dd=sorted(x for x in ds if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
bF,bB=bench(FRONT),bench(BACK)
print(f"\n0050: 前段{bF:+.0f}% 後段{bB:+.0f}%")
print(f"\n{'策略':<26}{'前段α':>7}{'後段α':>7}  兩段都贏baseline?")
baseF=run(rows('base',FRONT))['ret']-bF; baseB=run(rows('base',BACK))['ret']-bB
print(f"{'baseline 無B2':<26}{baseF:>+7.0f}{baseB:>+7.0f}  —")
aF=run(rows('always',FRONT))['ret']-bF; aB=run(rows('always',BACK))['ret']-bB
print(f"{'B2 always(原版)':<26}{aF:>+7.0f}{aB:>+7.0f}  {'✅' if aF>baseF and aB>baseB else '❌ '+('前段輸' if aF<=baseF else '後段輸')}")
for gate in ("G1_0050_60d","G2_0050_20d","G3_momfactor"):
    gF=run(rows('gated',FRONT,gate=gate))['ret']-bF; gB=run(rows('gated',BACK,gate=gate))['ret']-bB
    ok='✅兩段都贏' if gF>baseF and gB>baseB else ('❌'+('前段輸' if gF<=baseF else '')+('後段輸' if gB<=baseB else ''))
    print(f"{'B2 gated by '+gate:<26}{gF:>+7.0f}{gB:>+7.0f}  {ok}")
print("\n判讀:gated版若『兩段都贏baseline』→ regime條件式修好了(動態、閾值用全史校準);")
print("     若gate後前段還是輸 → 兩段差異不是這個動能定義能抓的(回到『因子alpha』而非市場方向)。")
