"""B2(wt_cv 經理人權重變異係數)產業中性化驗證:
是真的「同產業內」分歧度 alpha,還是只是吃「金融/中型補漲」的 sector 行情?
- 原始 IC vs 產業中性 IC(把訊號與報酬都在每日每產業內 demean,扣掉 sector rotation)
- 產業內 long-short(每日每產業內 long高CV/short低CV)扣成本
- 逐產業 IC(看是不是全靠金融)
- Q2-Q3 中分位 filter(避開非單調的極端 Q4)
leak-safe:wt_cv@D → forward 超額@D+1 起,扣同期0050。
"""
import sys, json
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.stdout.reconfigure(encoding="utf-8")
import yfinance as yf

df = pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv", encoding="utf-8")
df["Date"]=pd.to_datetime(df["Date"]); df["Stock_Code"]=df["Stock_Code"].astype(str)
agg = df.groupby(["Date","Stock_Code"]).agg(wt_std=("Weight(%)",lambda s:s.std(ddof=0)),
        wt_mean=("Weight(%)","mean")).reset_index()
agg["wt_cv"]=agg["wt_std"]/agg["wt_mean"].replace(0,np.nan)
codes=sorted(agg["Stock_Code"].unique())
names=df.groupby("Stock_Code")["Stock_Name"].first().to_dict()

# 產業標籤:base_universe + tw_stock_index 補
u=json.loads((ROOT/"data/base_universe.json").read_text(encoding="utf-8"))
ti=json.loads((ROOT/"data/tw_stock_index.json").read_text(encoding="utf-8"))
def ind(c):
    if c in u: return u[c].get("industry","?")
    return ti.get(c,{}).get("industry","?")
agg["ind"]=agg["Stock_Code"].map(ind)
print("產業分布(50檔池):")
for s,n in agg.groupby("ind")["Stock_Code"].nunique().sort_values(ascending=False).items():
    print(f"  {s}: {n}檔")

# 價格 + forward 超額(D+1)
def yft(c): return ti.get(c,{}).get("yf_ticker",f"{c}.TW")
def close(t):
    try:
        d=yf.download(t,start="2025-05-01",auto_adjust=True,progress=False)
        if d is None or d.empty: return None
        c=d["Close"]; c=c.iloc[:,0] if isinstance(c,pd.DataFrame) else c
        return c.dropna()
    except Exception: return None
px={}
for c in codes:
    s=close(yft(c)); s=close(f"{c}.TWO") if s is None else s
    if s is not None and len(s)>20: px[c]=s
bench=close("0050.TW")
alldates=sorted(set().union(*[set(s.index) for s in px.values()],set(bench.index)))
pxm=pd.DataFrame({c:px[c].reindex(alldates) for c in px}); bm=bench.reindex(alldates)
pos={d:i for i,d in enumerate(alldates)}
def fexc(n,code,d):
    i=pos.get(d)
    if i is None or i+1>=len(alldates) or code not in pxm.columns: return np.nan
    d1=alldates[i+1]; j=pos[d1]
    if j+n>=len(alldates): return np.nan
    r=pxm[code].iloc[j+n]/pxm[code].iloc[j]-1; b=bm.iloc[j+n]/bm.iloc[j]-1
    return (r-b)*100 if pd.notna(r) and pd.notna(b) else np.nan
for n in (10,20):
    agg[f"f{n}"]=[fexc(n,c,d) for c,d in zip(agg["Stock_Code"],agg["Date"])]

def ic(a,b,d):
    s=d.dropna(subset=[a,b]); return s[a].corr(s[b],method="spearman") if len(s)>50 else np.nan

print("\n=== 原始 vs 產業中性 IC(wt_cv → forward超額)===")
for n in (10,20):
    raw=ic("wt_cv",f"f{n}",agg)
    # 產業中性:訊號與報酬都在(date,ind)內 demean
    a=agg.dropna(subset=["wt_cv",f"f{n}"]).copy()
    a["cv_sn"]=a.groupby(["Date","ind"])["wt_cv"].transform(lambda s:s-s.mean())
    a["ret_sn"]=a.groupby(["Date","ind"])[f"f{n}"].transform(lambda s:s-s.mean())
    sn=a["cv_sn"].corr(a["ret_sn"],method="spearman")
    print(f"  fwd{n}: 原始IC {raw:+.3f}  →  產業中性IC {sn:+.3f}  ({'存活' if abs(sn)>0.03 and np.sign(sn)==np.sign(raw) else '大幅衰減/消失'})")

print("\n=== 產業內 long-short(每日每產業 long高CV-short低CV,fwd10,扣0.6%)===")
a=agg.dropna(subset=["wt_cv","f10"]).copy(); ls=[]
for (d,sec),grp in a.groupby(["Date","ind"]):
    if len(grp)<4: continue
    k=max(1,len(grp)//2)
    ls.append(grp.nlargest(k,"wt_cv")["f10"].mean()-grp.nsmallest(k,"wt_cv")["f10"].mean())
print(f"  產業內 LS = {np.mean(ls)-0.6:+.2f}pp/筆(n={len(ls)});對照:全池 LS 之前 +1.68")

print("\n=== 逐產業 IC(fwd20,看是不是全靠金融)===")
for sec,grp in agg.groupby("ind"):
    g2=grp.dropna(subset=["wt_cv","f20"])
    if g2["Stock_Code"].nunique()<3 or len(g2)<100: continue
    print(f"  {sec:<10}({g2['Stock_Code'].nunique()}檔): IC {ic('wt_cv','f20',g2):+.3f}  n={len(g2)}")

print("\n=== Q2-Q3 中分位 filter vs 極端(避開非單調Q4)===")
a=agg.dropna(subset=["wt_cv","f10"]).copy()
a["q"]=a.groupby("Date")["wt_cv"].transform(lambda s:pd.qcut(s.rank(method="first"),5,labels=False) if s.nunique()>=5 else np.nan)
for q in range(5):
    print(f"  Q{q}(CV分位) fwd10超額 {a[a['q']==q]['f10'].mean():+.2f}%")
print(f"  → 中分位Q2-Q3平均 {a[a['q'].isin([2,3])]['f10'].mean():+.2f}% vs 極端Q4 {a[a['q']==4]['f10'].mean():+.2f}%")
