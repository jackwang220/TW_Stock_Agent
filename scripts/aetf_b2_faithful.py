"""B2 疊 H 重驗 — 用忠實 live 引擎(v9 rows_for: 多頭h_score/空頭reb×1.5, exit_only, INC可選)。
取代之前的簡化代理。標準:前段/後段都要看 + placebo 對照,看結論翻不翻。
"""
import sys, json, math
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove(); logger.add(sys.stderr,level="ERROR")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
import pandas as pd, importlib.util as U
def load(n,p):
    s=U.spec_from_file_location(n,p); m=U.module_from_spec(s); s.loader.exec_module(m); return m
nine=load("nine",ROOT/"scripts/exp_step1_v9.py")
is_bull,h_score,_factors=nine.is_bull,nine.h_score,nine._factors
v5,v6,ec=nine.v5,nine.v6,nine.ec; sim5=ec.sim_buyclose_sellopen
features=v5.features; get_ohlcv=nine.get_daily_ohlcv; START=nine.START
u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
codes=list(u.keys()); turns={c:u[c].get("avg_turnover",0.0) for c in codes}
# B2 訊號
df=pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv",encoding="utf-8")
df["Date"]=pd.to_datetime(df["Date"]); df["Stock_Code"]=df["Stock_Code"].astype(str)
ag=df.groupby(["Date","Stock_Code"]).agg(wt_std=("Weight(%)",lambda s:s.std(ddof=0)),wt_mean=("Weight(%)","mean")).reset_index()
ag["wt_cv"]=ag["wt_std"]/ag["wt_mean"].replace(0,np.nan); ag["d"]=ag["Date"].dt.strftime("%Y-%m-%d")
ag["zcv"]=ag.groupby("d")["wt_cv"].transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0.0).fillna(0.0)
ZCV={(r.d,r.Stock_Code):r.zcv for r in ag.itertuples()}
print("載入(v9引擎)...",flush=True)
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
                g=rebound_signal(cl,turns.get(c,0.0));
                if g.get("fired"): m[d]=g["score"]*100
            except Exception: pass
        if j>0 and OH[c][ds[j-1]]["close"]>0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1>=0.095: s.add(d)
    reb[c]=m; limitup[c]=s
turn_pct={}
for d in alld:
    vals=sorted(((c,feats[c][d]["turn"]) for c in codes if d in feats.get(c,{}) and feats[c][d]["turn"]>0),key=lambda x:x[1])
    turn_pct[d]={c:(i+1)/len(vals) for i,(c,_) in enumerate(vals)} if vals else {}
rng=np.random.RandomState(7)
def rows(mode_tilt,k=0.3):  # mode_tilt: base/b2/placebo
    out=[]
    for d in alld:
        ir=twii.get(d,{}).get("ret20"); bull=is_bull(twii.get(d,{}),"原版c>20MA")
        for c in codes:
            f=feats.get(c,{})
            if d not in f or math.isnan(f[d].get("ma20",float("nan"))): continue
            sc=h_score(_factors(f[d],ir),turn_pct.get(d,{}).get(c,0.5)) if bull else reb.get(c,{}).get(d,0.0)*1.5
            if sc<=0: continue
            e=sc/100.0
            if mode_tilt=="b2": e*= (1+k*math.tanh(ZCV.get((d,c),0.0)))
            elif mode_tilt=="placebo": e*= (1+k*math.tanh(rng.randn()))
            out.append((d,c,e))
    return out
ROWS={m:rows(m) for m in ("base","b2","placebo")}
def bench(ds): dd=sorted(x for x in ds if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
b2d=sorted({d for d in alld if d>="2025-06-17" and d<="2026-06-08"}); cut=b2d[int(len(b2d)*0.6)]
SEG={"前段":[d for d in b2d if d<cut],"後段":[d for d in b2d if d>=cut]}
print(f"B2期間 {b2d[0]}~{b2d[-1]};前段{len(SEG['前段'])}日/後段{len(SEG['後段'])}日")
def A(rws,seg,inc):
    r=sim5([x for x in rws if x[0] in set(seg)],opens,closes,limitup,incumbent=inc,sell_mode="exit_only")
    return r["ret"]-bench(seg) if r else None
for inc in (1.5,2.0):
    print(f"\n=== 忠實live引擎(exit_only, INC{inc}) B2重驗 ===")
    print(f"{'策略':<16}{'前段α':>8}{'後段α':>8}  兩段都贏baseline?")
    bF=A(ROWS['base'],SEG['前段'],inc); bB=A(ROWS['base'],SEG['後段'],inc)
    print(f"{'baseline 無B2':<16}{bF:>+8.0f}{bB:>+8.0f}  —")
    for m,lab in [("b2","B2 tilt k0.3"),("placebo","placebo隨機")]:
        f=A(ROWS[m],SEG['前段'],inc); b=A(ROWS[m],SEG['後段'],inc)
        ok='✅兩段都贏' if f>bF and b>bB else ('❌'+('前段輸 ' if f<=bF else '')+('後段輸' if b<=bB else ''))
        print(f"{lab:<16}{f:>+8.0f}{b:>+8.0f}  {ok}")
print("\n判讀:B2若在正確baseline上『兩段都贏且勝過placebo』→結論翻盤(B2可用);若仍前段輸/翻號→結論不變(B2不穩/相對訊號)。")
