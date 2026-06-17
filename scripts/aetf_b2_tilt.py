"""B2(經理人權重變異係數 wt_cv)疊到 H 雙引擎當 tilt — 決定性可用性測試。
對照:baseline(純H) vs B2-tilt(k掃) vs placebo(隨機tilt) vs A+B2組合。
多窗 alpha vs 同資金DCA0050、扣成本、曝險中性。leak-safe:wt_cv@D→H候選@D edge→⑤隔日執行。
重用 news_llm_select 的 H 候選/sim5/基準機制。
"""
import sys, json, math
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove(); logger.add(sys.stderr, level="WARNING")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
import importlib.util as U
def load(n,p):
    s=U.spec_from_file_location(n,p); m=U.module_from_spec(s); s.loader.exec_module(m); return m
nls=load("nls",ROOT/"scripts/news_llm_select.py")
features,_factors,oh=nls.features,nls._factors,nls.oh
build_candidates,sim5,v6=nls.build_candidates,nls.sim5,nls.v6
WINDOWS,REGIMES,END=nls.WINDOWS,nls.REGIMES,nls.END

# --- B2 訊號:每(date,stock)跨ETF權重變異係數,再做每日橫斷面 z ---
df=pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv",encoding="utf-8")
df["Date"]=pd.to_datetime(df["Date"]); df["Stock_Code"]=df["Stock_Code"].astype(str)
df=df.sort_values(["ETF_Code","Stock_Code","Date"])
df["dwt"]=df.groupby(["ETF_Code","Stock_Code"])["Weight(%)"].diff()
agg=df.groupby(["Date","Stock_Code"]).agg(wt_std=("Weight(%)",lambda s:s.std(ddof=0)),
        wt_mean=("Weight(%)","mean"), n_add=("dwt",lambda s:(s>0.01).sum())).reset_index()
agg["wt_cv"]=agg["wt_std"]/agg["wt_mean"].replace(0,np.nan)
agg["d"]=agg["Date"].dt.strftime("%Y-%m-%d")
# 每日橫斷面 z-score
def zby(col):
    return agg.groupby("d")[col].transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0.0)
agg["z_cv"]=zby("wt_cv").fillna(0.0); agg["z_add"]=zby("n_add").fillna(0.0)
CV={(r.d,r.Stock_Code):r.z_cv for r in agg.itertuples()}
ADD={(r.d,r.Stock_Code):r.z_add for r in agg.itertuples()}

# --- H 設定(複製 news_llm_select.main 的前置)---
u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
codes=list(u.keys()); names={c:u[c].get("name",c) for c in codes}; turns={c:u[c].get("avg_turnover",0.0) for c in codes}
twii=features("0050"); feats={c:features(c) for c in codes}
opens,closes={},{}
for c in codes+["0050"]:
    o=oh(c); opens[c]={d:o[d]["open"] for d in o}; closes[c]={d:o[d]["close"] for d in o}
alld=sorted({d for c in codes for d in closes.get(c,{}) if d<=END}); sig=alld[-max(n for _,n in WINDOWS):]
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

rng=np.random.RandomState(7)
def rows(k,mode):
    rw=[]
    for d in sig:
        for v,c in cands[d]:
            e=v/100.0
            if k and mode=="b2": e*= (1+k*math.tanh(CV.get((d,c),0.0)))
            elif k and mode=="add": e*=(1+k*math.tanh(ADD.get((d,c),0.0)))
            elif k and mode=="combo": e*=(1+k*math.tanh((CV.get((d,c),0.0)+ADD.get((d,c),0.0))/2))
            elif k and mode=="placebo": e*=(1+k*math.tanh(rng.randn()))
            rw.append((d,c,e))
    return rw
def run(rw): return sim5(rw,opens,closes,limitup,switch_cost_mult=1.0)
def bench(day):
    dd=sorted(x for x in day if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
bwin={wl:bench(set(sig[-n:])) for wl,n in WINDOWS}
rdays={rn:set(d for d in sig if s<=d<=e) for rn,(s,e) in REGIMES.items()}
breg={rn:bench(dd) for rn,dd in rdays.items()}
cols=[wl for wl,_ in WINDOWS]+[rn for rn in REGIMES]
configs=[("baseline純H",0,None)]+[(f"B2tilt×{k}",k,"b2") for k in (0.3,0.6,1.0)]+\
        [("placebo×0.6",0.6,"placebo"),("A加碼×0.6",0.6,"add"),("A+B2×0.6",0.6,"combo")]
res={}
for lab,k,mode in configs:
    for wl,n in WINDOWS: res[(lab,wl)]=run(rows(k,mode) if mode else [(d,c,v/100.0) for d in sig for v,c in cands[d]])  # baseline
    for rn,dd in rdays.items():
        rw=[(d,c,v/100.0) for d in sig if d in dd for v,c in cands[d]] if mode is None else [r for r in rows(k,mode) if r[0] in dd]
        res[(lab,rn)]=run(rw)
def A(lab,col):
    r=res.get((lab,col)); b=bwin.get(col,breg.get(col)); return (r["ret"]-b) if (r and b is not None) else None
print("\n=== B2 tilt 疊到 H:多窗 ALPHA(扣0050)+ uplift vs baseline ===")
print("變體            "+" ".join(f"{c:>7}" for c in ["60天","半年","1年","2年","平均","曝險","換手"]))
base={col:A("baseline純H",col) for col in cols}
for lab,_,_ in configs:
    av=[A(lab,c) for c in cols if A(lab,c) is not None]
    r2=res.get((lab,"2年"))
    a60,ah,a1,a2=A(lab,"60天"),A(lab,"半年"),A(lab,"1年"),A(lab,"2年")
    def f(x): return f"{x:+.0f}" if x is not None else "  —"
    print(f"{lab:<14} {f(a60):>7} {f(ah):>7} {f(a1):>7} {f(a2):>7} {sum(av)/len(av):>+7.0f} {r2.get('avg_expo',0)*100:>6.0f}% {r2.get('turn',0):>6.0f}x")
print("\nuplift vs baseline(2年/平均):")
for lab,_,m in configs:
    if m is None: continue
    u2=A(lab,"2年")-base["2年"] if A(lab,"2年") is not None else None
    avs=[A(lab,c)-base[c] for c in cols if A(lab,c) is not None and base[c] is not None]
    print(f"  {lab:<14} 2年uplift {u2:+.0f}pp  平均uplift {sum(avs)/len(avs):+.0f}pp")
print("\n判讀:B2tilt 要明顯 > placebo 才算真;A/combo 看有沒有比純B2更好。曝險要跟baseline接近。")
