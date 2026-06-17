"""H雙引擎+B2加分 的參數調整 + overfitting 檢查。
B2訊號只有~1年(2025-06-17起)→ 切 TRAIN(前60%)/OOS(後40%)。
每組報 TRAIN alpha / OOS alpha / 換手 / 曝險:in-sample好但OOS崩=過配。
掃:B2強度k、曝險cap/floor、incumbent、換股門檻switch_cost、MAX_SIGNALS、EDGE_DECAY、資金規模。
"""
import sys, json, math
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove(); logger.add(sys.stderr,level="ERROR")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
import importlib.util as U
def load(n,p):
    s=U.spec_from_file_location(n,p); m=U.module_from_spec(s); s.loader.exec_module(m); return m
nls=load("nls",ROOT/"scripts/news_llm_select.py")
features,_factors,oh=nls.features,nls._factors,nls.oh
build_candidates,sim5,v6=nls.build_candidates,nls.sim5,nls.v6
END=nls.END; ec=nls.ec; r60=ec.r60
DEF=dict(EXPOSURE_CAP=r60.EXPOSURE_CAP,EXPOSURE_FLOOR=r60.EXPOSURE_FLOOR,MAX_SIGNALS=r60.MAX_SIGNALS,
         EDGE_DECAY=r60.EDGE_DECAY,INITIAL_CAPITAL=r60.INITIAL_CAPITAL,DAILY_BUDGET=r60.DAILY_BUDGET,
         MAX_CONTRIBUTION=r60.MAX_CONTRIBUTION,DAILY_ADD_CAP=r60.DAILY_ADD_CAP,TIE_RATIO=r60.TIE_RATIO)

# B2 訊號
df=pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv",encoding="utf-8")
df["Date"]=pd.to_datetime(df["Date"]); df["Stock_Code"]=df["Stock_Code"].astype(str)
ag=df.groupby(["Date","Stock_Code"]).agg(wt_std=("Weight(%)",lambda s:s.std(ddof=0)),wt_mean=("Weight(%)","mean")).reset_index()
ag["wt_cv"]=ag["wt_std"]/ag["wt_mean"].replace(0,np.nan); ag["d"]=ag["Date"].dt.strftime("%Y-%m-%d")
ag["zcv"]=ag.groupby("d")["wt_cv"].transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0.0).fillna(0.0)
ZCV={(r.d,r.Stock_Code):r.zcv for r in ag.itertuples()}

# H 設定
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
cands,_=build_candidates(codes,names,feats,twii,reb,turn_pct,sig)

# B2 期間 + TRAIN/OOS 切分
b2dates=[d for d in sig if d>="2025-06-17"]
cut=b2dates[int(len(b2dates)*0.6)]
TRAIN=set(d for d in b2dates if d<cut); OOS=set(d for d in b2dates if d>=cut)
print(f"B2期間 {b2dates[0]}~{b2dates[-1]} ({len(b2dates)}日);TRAIN {len(TRAIN)}日(<{cut}) / OOS {len(OOS)}日(>={cut})")

def rows_for(k, dayset):
    rw=[]
    for d in sig:
        if d not in dayset: continue
        for v,c in cands[d]:
            e=v/100.0
            if k: e*= (1+k*math.tanh(ZCV.get((d,c),0.0)))
            rw.append((d,c,e))
    return rw
def bench(dayset):
    dd=sorted(x for x in dayset if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
btr,boo=bench(TRAIN),bench(OOS)
def runcfg(cfg, dayset):
    for kk,vv in DEF.items(): setattr(r60,kk,vv)             # reset
    for kk in ("EXPOSURE_CAP","EXPOSURE_FLOOR","MAX_SIGNALS","EDGE_DECAY","INITIAL_CAPITAL","DAILY_BUDGET","MAX_CONTRIBUTION","DAILY_ADD_CAP"):
        if kk in cfg: setattr(r60,kk,cfg[kk])
    r=sim5(rows_for(cfg["k"],dayset),opens,closes,limitup,incumbent=cfg.get("inc",1.5),switch_cost_mult=cfg.get("scm",1.0))
    for kk,vv in DEF.items(): setattr(r60,kk,vv)             # restore
    return r

CFG=[
 ("C0 無B2(現行)",        dict(k=0)),
 ("C1 +B2 預設k0.3",      dict(k=0.3)),
 ("C2 B2 k0.2",          dict(k=0.2)),
 ("C3 B2 k0.5",          dict(k=0.5)),
 ("C4 曝險cap0.8",        dict(k=0.3,EXPOSURE_CAP=0.8)),
 ("C5 曝險cap1.0",        dict(k=0.3,EXPOSURE_CAP=1.0)),
 ("C6 曝險floor0.4",      dict(k=0.3,EXPOSURE_FLOOR=0.4)),
 ("C7 incumbent2.0",     dict(k=0.3,inc=2.0)),
 ("C8 換股門檻x2(降換手)", dict(k=0.3,scm=2.0)),
 ("C9 持股MAX2(更集中)",   dict(k=0.3,MAX_SIGNALS=2)),
 ("C10 持股MAX4(更分散)",  dict(k=0.3,MAX_SIGNALS=4)),
 ("C11 EDGE_DECAY0.9",   dict(k=0.3,EDGE_DECAY=0.9)),
 ("C12 資金x2",           dict(k=0.3,INITIAL_CAPITAL=30000,DAILY_BUDGET=2000,MAX_CONTRIBUTION=100000,DAILY_ADD_CAP=30000)),
 ("C13 B2 k0.3+inc2+scm2", dict(k=0.3,inc=2.0,scm=2.0)),
]
print(f"\n0050基準: 前段(142日) {btr:+.0f}% / 後段(96日) {boo:+.0f}%")
print("交換考訓練 = 看 B2 在「前段當考卷」與「後段當考卷」是否一致")
res={}
for lab,cfg in CFG:
    rt=runcfg(cfg,TRAIN); ro=runcfg(cfg,OOS)
    res[lab]=(rt["ret"]-btr, ro["ret"]-boo, rt, ro)
c0_tr,c0_oo=res["C0 無B2(現行)"][0],res["C0 無B2(現行)"][1]
print(f"\n{'組別':<22}{'前段α':>7}{'後段α':>7}  {'前段vs現行':>9}{'後段vs現行':>9}  一致?")
for lab,cfg in CFG:
    at,ao,rt,ro=res[lab]
    u1,u2=at-c0_tr,ao-c0_oo
    consist="✅兩段都贏" if (u1>3 and u2>3) else ("❌方向相反" if u1*u2<0 and abs(u1)>5 and abs(u2)>5 else ("〜中性" if abs(u1)<=5 and abs(u2)<=5 else "⚠️只一段"))
    print(f"{lab:<22}{at:>+7.0f}{ao:>+7.0f}  {u1:>+9.0f}{u2:>+9.0f}  {consist}")
print("\n判讀:'前段vs現行'與'後段vs現行'都>0=兩段一致有效(穩);一正一負=效果只在某半年、換段就反(脆)。")
