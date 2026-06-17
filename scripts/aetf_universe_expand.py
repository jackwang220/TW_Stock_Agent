"""用『主動式ETF經理人持有』當基準擴充 universe 有沒有用。
比較 H 雙引擎在:(A)現行112檔 (B)112+經理人持有但池外的股 (C)經理人50檔池 三種universe。
多窗 alpha vs 同資金DCA0050、扣成本、曝險。擴充股價歷史完整(僅成員資格來自近期ETF=hypothesis來源)。
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
build_candidates,sim5,v6=nls.build_candidates,nls.sim5,nls.v6; END=nls.END; WINDOWS,REGIMES=nls.WINDOWS,nls.REGIMES

df=pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv",encoding="utf-8")
df["Stock_Code"]=df["Stock_Code"].astype(str)
pool=sorted(df["Stock_Code"].unique())          # 經理人50檔池
u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
base=list(u.keys())
outside=[c for c in pool if c not in u]          # 經理人持有但112池外
print(f"經理人池 {len(pool)} 檔;池外(可擴充) {len(outside)}: {outside}")

# 三種 universe
UNIV={"A 現行112":base, "B 112+池外擴充":base+outside, "C 經理人50池":pool}
# 各universe要的所有code
allc=sorted(set(base+pool))
names={c:u.get(c,{}).get("name",c) for c in allc}
turns={c:u.get(c,{}).get("avg_turnover",0.0) for c in allc}
print("載入特徵(含擴充股)...",flush=True)
twii=features("0050")
feats={}; opens={}; closes={}
for c in allc:
    try:
        f=features(c); o=oh(c)
        if f and o: feats[c]=f; opens[c]={d:o[d]["open"] for d in o}; closes[c]={d:o[d]["close"] for d in o}
    except Exception: pass
o0=oh("0050"); opens["0050"]={d:o0[d]["open"] for d in o0}; closes["0050"]={d:o0[d]["close"] for d in o0}
ok=set(feats);
for nm,lst in UNIV.items(): UNIV[nm]=[c for c in lst if c in ok]
print("各universe有效檔數:",{k:len(v) for k,v in UNIV.items()},"  池外取得:",[c for c in outside if c in ok])
alld=sorted({d for c in ok for d in closes.get(c,{}) if d<=END}); sig=alld   # 全史,讓2022空頭regime測得到

def setup(codes):
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
    tp={}
    for d in sig:
        vals=sorted(((c,feats[c][d]["turn"]) for c in codes if d in feats.get(c,{}) and feats[c][d]["turn"]>0),key=lambda x:x[1])
        tp[d]={c:(i+1)/len(vals) for i,(c,_) in enumerate(vals)} if vals else {}
    return reb,limitup,tp
def run_univ(codes):
    reb,limitup,tp=setup(codes)
    cands,_=build_candidates(codes,{c:names[c] for c in codes},feats,twii,reb,tp,sig)
    res={}
    def rows(dayset): return [(d,c,v/100.0) for d in sig if d in dayset for v,c in cands[d]]
    return cands,reb,limitup,rows
def bench(ds): dd=sorted(x for x in ds if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
WL=[("60天",60),("半年",126),("1年",252),("2年",504)]
REG=list(REGIMES.items())
bwin={w:bench(set(sig[-n:])) for w,n in WL}
breg={rn:bench(set(d for d in sig if s<=d<=e)) for rn,(s,e) in REG}
print(f"\n0050: "+" ".join(f"{w}{bwin[w]:+.0f}%" for w,_ in WL))
print(f"\n{'universe':<16}"+"".join(f"{w:>8}" for w,_ in WL)+f"{'2022空頭':>9}{'2024-25':>9}{'換手':>7}{'曝險':>6}")
for nm,codes in UNIV.items():
    cands,reb,limitup,rows=run_univ(codes)
    def A(dayset,bv):
        rw=rows(dayset)
        if not rw or bv is None: return None,None
        r=sim5(rw,opens,closes,limitup,switch_cost_mult=1.0)
        if not r: return None,None
        return r,r["ret"]-bv
    def fmt(x,w=8): return (f"{x:>+{w}.0f}" if x is not None else f"{'—':>{w}}")
    cells=""
    for w,n in WL:
        _,a=A(set(sig[-n:]),bwin[w]); cells+=fmt(a)
    r2,_=A(set(sig[-504:]),bwin["2年"])
    rd22=set(d for d in sig if REGIMES["2022空頭"][0]<=d<=REGIMES["2022空頭"][1])
    rd24=set(d for d in sig if REGIMES["2024-25多頭"][0]<=d<=REGIMES["2024-25多頭"][1])
    _,a22=A(rd22,breg["2022空頭"]); _,a24=A(rd24,breg["2024-25多頭"])
    tr=f"{r2.get('turn',0):>6.0f}x{r2.get('avg_expo',0)*100:>5.0f}%" if r2 else f"{'—':>6}{'—':>6}"
    print(f"{nm:<16}{cells}{fmt(a22,9)}{fmt(a24,9)}{tr}")
print("\n判讀:B(擴充)若多窗alpha≥A(現行)且2022空頭沒變差→經理人選股當擴充基準有用;若變差=擴充稀釋。")
print("註:擴充股的成員資格來自近期ETF持股(hypothesis來源),股價歷史完整;2022空頭那欄是真OOS壓力測試。")
