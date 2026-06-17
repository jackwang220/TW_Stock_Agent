"""B2(wt_cv)訊號穩定性:cross-sectional IC/LS 分上下半年(統計力強、無fitting、無train/test方向問題)。
回答:策略級的「半年翻號」是訊號真的只在後半年,還是只是3檔組合的雜訊?
"""
import sys, json
import pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.stdout.reconfigure(encoding="utf-8")
import yfinance as yf
df=pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv",encoding="utf-8")
df["Date"]=pd.to_datetime(df["Date"]); df["Stock_Code"]=df["Stock_Code"].astype(str)
ag=df.groupby(["Date","Stock_Code"]).agg(wt_std=("Weight(%)",lambda s:s.std(ddof=0)),wt_mean=("Weight(%)","mean")).reset_index()
ag["wt_cv"]=ag["wt_std"]/ag["wt_mean"].replace(0,np.nan)
codes=sorted(ag["Stock_Code"].unique())
ti=json.loads((ROOT/"data/tw_stock_index.json").read_text(encoding="utf-8"))
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
pxm=pd.DataFrame({c:px[c].reindex(alldates) for c in px}); bm=bench.reindex(alldates); pos={d:i for i,d in enumerate(alldates)}
def fexc(n,code,d):
    i=pos.get(d)
    if i is None or i+1>=len(alldates) or code not in pxm.columns: return np.nan
    j=pos[alldates[i+1]]
    if j+n>=len(alldates): return np.nan
    r=pxm[code].iloc[j+n]/pxm[code].iloc[j]-1; b=bm.iloc[j+n]/bm.iloc[j]-1
    return (r-b)*100 if pd.notna(r) and pd.notna(b) else np.nan
for n in (5,10,20): ag[f"f{n}"]=[fexc(n,c,d) for c,d in zip(ag["Stock_Code"],ag["Date"])]
# 同樣的切分點
b2d=sorted(ag["Date"].dt.strftime("%Y-%m-%d").unique()); cut=b2d[int(len(b2d)*0.6)]
ag["seg"]=np.where(ag["Date"].dt.strftime("%Y-%m-%d")<cut,"前段","後段")
print(f"切分點 {cut};前段 {(ag['seg']=='前段').sum()} 列 / 後段 {(ag['seg']=='後段').sum()} 列")
def ic(d,a,b):
    s=d.dropna(subset=[a,b]); return s[a].corr(s[b],method="spearman") if len(s)>50 else np.nan
def ls(d,col,n,hold=10):
    out=[]
    for dt,day in d.dropna(subset=[col,f"f{n}"]).groupby("Date"):
        if len(day)<10: continue
        k=max(1,len(day)//5); out.append(day.nlargest(k,col)[f"f{n}"].mean()-day.nsmallest(k,col)[f"f{n}"].mean())
    return (np.mean(out)-0.6) if out else np.nan
print("\n=== wt_cv rank-IC 分段(統計力強、無fitting)===")
print(f"{'段':<6}{'IC5':>8}{'IC10':>8}{'IC20':>8}{'LS10(扣成本)':>14}{'n':>7}")
for seg in ["全期","前段","後段"]:
    d=ag if seg=="全期" else ag[ag["seg"]==seg]
    n=len(d.dropna(subset=['wt_cv','f10']))
    print(f"{seg:<6}{ic(d,'wt_cv','f5'):>+8.3f}{ic(d,'wt_cv','f10'):>+8.3f}{ic(d,'wt_cv','f20'):>+8.3f}{ls(d,'wt_cv',10):>+14.2f}{n:>7}")
print("\n判讀:若前段IC也明顯>0 → 訊號兩段都在,策略級翻號=3檔組合雜訊(B2其實穩);")
print("     若前段IC≈0或負 → 訊號真的後段才出現(regime相依),B2降級成立。")