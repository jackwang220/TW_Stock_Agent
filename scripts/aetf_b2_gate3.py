"""B2 gate 第3輪:用『市場廣度/分散』類訊號(非大盤方向、非K線動能)當detector。
文獻:動能在『窄、少數股領軍』時失靈→該開B2(分散)。
gate候選(全史校準閾值,低廣度→開B2):
  breadth_ma20 = universe 站上MA20比例
  breadth_pos20= ret20>0 比例
  breadth_win  = 近20日贏0050的股票比例
  disp_ret20   = ret20 橫斷面標準差(離散度高=分歧)
對照 oracle。報 前段/後段,看哪個能正確分段且兩段都贏。
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
oc=closes["0050"]; od=sorted(oc)
def b50(d,n):
    i=od.index(d) if d in oc else -1
    return (oc[d]/oc[od[i-n]]-1) if i>=n else np.nan
# 廣度訊號
B={}
for d in alld:
    above=[1 for c in codes if d in feats.get(c,{}) and feats[c][d].get("close") and feats[c][d].get("ma20") and not math.isnan(feats[c][d]["ma20"]) and feats[c][d]["close"]>feats[c][d]["ma20"]]
    tot=[1 for c in codes if d in feats.get(c,{}) and feats[c][d].get("ma20") and not math.isnan(feats[c][d]["ma20"])]
    r20=[feats[c][d].get("ret20") for c in codes if d in feats.get(c,{}) and feats[c][d].get("ret20") is not None and not math.isnan(feats[c][d].get("ret20",float('nan')))]
    m50=b50(d,20)
    win=[1 for c in codes if d in feats.get(c,{}) and feats[c][d].get("ret20") is not None and not math.isnan(feats[c][d].get("ret20",float('nan'))) and m50 is not None and not math.isnan(m50) and feats[c][d]["ret20"]>m50]
    B[d]={"breadth_ma20":sum(above)/len(tot) if tot else np.nan,
          "breadth_pos20":sum(1 for x in r20 if x>0)/len(r20) if r20 else np.nan,
          "breadth_win":sum(win)/len(r20) if r20 else np.nan,
          "disp_ret20":np.std(r20) if len(r20)>5 else np.nan}
KEYS=["breadth_ma20","breadth_pos20","breadth_win","disp_ret20"]
TH={k:np.median([B[d][k] for d in alld if not pd.isna(B[d].get(k))]) for k in KEYS}
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
print("=== 廣度訊號 前段 vs 後段(低廣度=窄=該開B2;看能否分開:前高後低才對)===")
for k in KEYS:
    f=np.mean([B[d][k] for d in FRONT if not pd.isna(B[d].get(k))]); b=np.mean([B[d][k] for d in BACK if not pd.isna(B[d].get(k))])
    print(f"  {k}: 前段{f:+.3f} 後段{b:+.3f} 閾值{TH[k]:+.3f} → 前{'高' if f>TH[k] else '低'}/後{'高' if b>TH[k] else '低'}  {'✅前高後低(可gate)' if f>b and f>TH[k]>=b else ''}")
def use_b2(mode,d,gate=None):
    if mode=="always": return True
    if mode=="oracle": return d in BACK
    if mode=="gate": return B[d].get(gate) is not None and not pd.isna(B[d].get(gate)) and B[d][gate]<TH[gate]  # 低廣度→開
    return False
def rows(mode,dayset,gate=None,k=0.3):
    rw=[]
    for d in sig:
        if d not in dayset: continue
        ub=use_b2(mode,d,gate)
        for v,c in cands[d]:
            e=v/100.0
            if ub and mode!="base": e*=(1+k*math.tanh(ZCV.get((d,c),0.0)))
            rw.append((d,c,e))
    return rw
def run(rw): return sim5(rw,opens,closes,limitup,switch_cost_mult=1.0)
def bench(ds): dd=sorted(x for x in ds if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
bF,bB=bench(FRONT),bench(BACK)
baseF=run(rows('base',FRONT))['ret']-bF; baseB=run(rows('base',BACK))['ret']-bB
print(f"\n0050 前段{bF:+.0f}% 後段{bB:+.0f}%")
print(f"\n{'策略':<24}{'前段α':>7}{'後段α':>7}  兩段都贏?")
print(f"{'baseline 無B2':<24}{baseF:>+7.0f}{baseB:>+7.0f}  —")
oF=run(rows('oracle',FRONT))['ret']-bF; oB=run(rows('oracle',BACK))['ret']-bB
print(f"{'oracle作弊上界':<24}{oF:>+7.0f}{oB:>+7.0f}  ✅")
for gate in KEYS:
    gF=run(rows('gate',FRONT,gate=gate))['ret']-bF; gB=run(rows('gate',BACK,gate=gate))['ret']-bB
    ok='✅兩段都贏' if gF>=baseF-0.5 and gB>baseB else ('❌'+('前段輸' if gF<baseF-0.5 else '')+('後段沒贏' if gB<=baseB else ''))
    print(f"{'gate='+gate:<24}{gF:>+7.0f}{gB:>+7.0f}  {ok}")
print("\n註:1個regime轉折→就算某gate兩段都贏,仍是fit這一個轉折點,需更多regime才敢信(誠實caveat)。")
