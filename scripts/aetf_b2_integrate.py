"""B2(經理人權重分歧度 CV)+ 擁擠度反向(sum_wt fade)整進 H 雙引擎 — 多種整合法。
整合法:①純H(原始) ②單純加分(boost) ③只加分中分位Q2-Q3 ④保留1股空間(inject slot)
        ⑤B2+擁擠反向combo ⑥兩套系統混合(80%H+20%B2 sleeve,近似) ⑦placebo(隨機,對照)
輸出:每個變體的「原始報酬%(未扣大盤)」與「alpha%(扣同資金DCA0050)」多窗對照 + 曝險/換手。
leak-safe:訊號@D→H候選@D edge→⑤隔日執行;扣真實成本。
"""
import sys, json, math
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove(); logger.add(sys.stderr,level="WARNING")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
import importlib.util as U
def load(n,p):
    s=U.spec_from_file_location(n,p); m=U.module_from_spec(s); s.loader.exec_module(m); return m
nls=load("nls",ROOT/"scripts/news_llm_select.py")
features,_factors,oh=nls.features,nls._factors,nls.oh
build_candidates,sim5,v6=nls.build_candidates,nls.sim5,nls.v6
WINDOWS,REGIMES,END=nls.WINDOWS,nls.REGIMES,nls.END

# ---- 訊號:wt_cv(分歧) + sum_wt(擁擠),每日橫斷面 z ----
df=pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv",encoding="utf-8")
df["Date"]=pd.to_datetime(df["Date"]); df["Stock_Code"]=df["Stock_Code"].astype(str)
ag=df.groupby(["Date","Stock_Code"]).agg(wt_std=("Weight(%)",lambda s:s.std(ddof=0)),
        wt_mean=("Weight(%)","mean"), sum_wt=("Weight(%)","sum")).reset_index()
ag["wt_cv"]=ag["wt_std"]/ag["wt_mean"].replace(0,np.nan)
ag["d"]=ag["Date"].dt.strftime("%Y-%m-%d")
def zby(c): return ag.groupby("d")[c].transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0.0)
ag["zcv"]=zby("wt_cv").fillna(0.0); ag["zcrowd"]=zby("sum_wt").fillna(0.0)
# CV 當日五分位(0..4),用於 Q2-Q3 filter
ag["q"]=ag.groupby("d")["wt_cv"].transform(lambda s:pd.qcut(s.rank(method="first"),5,labels=False) if s.nunique()>=5 else 2)
ZCV={(r.d,r.Stock_Code):r.zcv for r in ag.itertuples()}
ZCR={(r.d,r.Stock_Code):r.zcrowd for r in ag.itertuples()}
QCV={(r.d,r.Stock_Code):r.q for r in ag.itertuples()}
CVval={(r.d,r.Stock_Code):(r.wt_cv if pd.notna(r.wt_cv) else 0.0) for r in ag.itertuples()}

# ---- H 設定 ----
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
                gg=rebound_signal(cl,turns.get(c,0.0));
                if gg.get("fired"): m[d]=gg["score"]*100
            except Exception: pass
        if j>0 and o[ds[j-1]]["close"]>0 and o[d]["close"]/o[ds[j-1]]["close"]-1>=0.095: s.add(d)
    reb[c]=m; limitup[c]=s
turn_pct={}
for d in sig:
    vals=sorted(((c,feats[c][d]["turn"]) for c in codes if d in feats.get(c,{}) and feats[c][d]["turn"]>0),key=lambda x:x[1])
    turn_pct[d]={c:(i+1)/len(vals) for i,(c,_) in enumerate(vals)} if vals else {}
cands,_=build_candidates(codes,names,feats,twii,reb,turn_pct,sig)
poolcodes=set(ag["Stock_Code"].unique())&set(codes)   # 池股∩可交易
rng=np.random.RandomState(11); K=0.3
def base_rows():
    return [(d,c,v/100.0) for d in sig for v,c in cands[d]]
def make_rows(mode):
    rw=[]
    for d in sig:
        lst=cands[d]
        if mode=="inject":   # 保留1股:當日池內CV最高且有價的股,給它一個slot(edge=當日H最高分)
            avail=[(CVval.get((d,c),0),c) for c in poolcodes if d in closes.get(c,{}) and CVval.get((d,c),0)>0]
            top=max(avail)[1] if avail else None
            emax=max((v for v,_ in lst),default=50.0)/100.0
            base=[(d,c,v/100.0) for v,c in lst]
            if top is not None and top not in [c for _,c in lst]:
                base.append((d,top,emax))   # 注入,擠掉H最弱的(sim取top3)
            rw+=base; continue
        for v,c in lst:
            e=v/100.0
            if mode=="boost":   e*= (1+K*math.tanh(ZCV.get((d,c),0.0)))
            elif mode=="boostQ":  # 只加分中分位Q2-Q3
                if QCV.get((d,c),2) in (2,3): e*= (1+K)
            elif mode=="combo": e*= (1+K*math.tanh(ZCV.get((d,c),0.0)-0.5*ZCR.get((d,c),0.0)))
            elif mode=="placebo": e*= (1+K*math.tanh(rng.randn()))
            rw.append((d,c,e))
    return rw
def b2_sleeve_rows():  # B2-only sleeve:每日池內CV top-3,edge=0.4+0.4*分位
    rw=[]
    for d in sig:
        avail=[(CVval.get((d,c),0),c) for c in poolcodes if d in closes.get(c,{}) and CVval.get((d,c),0)>0]
        avail.sort(reverse=True)
        for rank,(cv,c) in enumerate(avail[:3]):
            rw.append((d,c,0.6-0.1*rank))
    return rw
def run(rw): return sim5(rw,opens,closes,limitup,switch_cost_mult=1.0)
def bench(day):
    dd=sorted(x for x in day if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)

sims={"①純H(原始)":run(base_rows()),"②加分boost":run(make_rows("boost")),
      "③加分Q2-Q3":run(make_rows("boostQ")),"④保留1股inject":run(make_rows("inject")),
      "⑤B2+擁擠反向combo":run(make_rows("combo")),"⑦placebo隨機":run(make_rows("placebo"))}
# ⑥ 兩套系統混合(近似:窗報酬加權 0.8H+0.2 B2sleeve)
b2s=run(b2_sleeve_rows())
WL=[("60天",60),("半年",126),("1年",252),("2年",504)]
def winret(simdict,wl,n):
    # 重跑該窗(sim5吃全rows,但ret是全期;為多窗需逐窗跑)→ 改用逐窗
    return None
# 逐窗跑(每個變體×每窗)
def rows_for(mode,n):
    dayset=set(sig[-n:])
    if mode=="base": return [(d,c,v/100.0) for d in sig if d in dayset for v,c in cands[d]]
    if mode=="b2sleeve": return [r for r in b2_sleeve_rows() if r[0] in dayset]
    return [r for r in make_rows(mode) if r[0] in dayset]
VAR={"①純H(原始)":"base","②加分boost":"boost","③加分Q2-Q3":"boostQ","④保留1股inject":"inject",
     "⑤B2+擁擠反向combo":"combo","⑦placebo隨機":"placebo"}
print("\n=== 整合 B2 進 H:原始報酬% / alpha%(扣0050),多窗 ===")
hdr="變體".ljust(18)+"".join(f"{w:>16}" for w,_ in WL)+"  曝險  換手"
print(hdr)
bwin={w:bench(set(sig[-n:])) for w,n in WL}
print("0050基準".ljust(18)+"".join(f"{bwin[w]:>+8.0f}%(raw) " for w,_ in WL))
rows_out={}
for lab,mode in VAR.items():
    cells=""; r2=None
    for w,n in WL:
        r=run(rows_for(mode,n));
        if w=="2年": r2=r
        a=r["ret"]-bwin[w]
        cells+=f" {r['ret']:>+6.0f}/{a:>+5.0f}"
        cells+=" "*1
    print(lab.ljust(18)+cells+f"  {r2.get('avg_expo',0)*100:>3.0f}% {r2.get('turn',0):>4.0f}x")
# ⑥ 混合
print("\n⑥ 兩套系統混合(近似 0.8純H + 0.2 B2sleeve,窗報酬加權):")
for w,n in WL:
    h=run(rows_for("base",n))["ret"]; b=run(rows_for("b2sleeve",n))["ret"]
    mix=0.8*h+0.2*b; print(f"  {w}: 純H raw {h:+.0f}% | B2sleeve raw {b:+.0f}% | 混合 raw {mix:+.0f}% (alpha {mix-bwin[w]:+.0f})")
print("\n格式: 報酬raw% / alpha%。判讀:整合版要 alpha 明顯 > ①純H 且 > ⑦placebo 才算真加分。")
