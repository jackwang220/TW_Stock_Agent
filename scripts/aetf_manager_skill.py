"""跟強經理人:28檔ETF裡是不是有些經理人「超配/加碼」真的能預測報酬(異質性)?
OOS防作弊:前段(142日)算每個經理人skill並排名 → 後段(96日)只跟top-K強經理人的獨家超配,
對照 bottom-K / 全部。executable:每日取分數top-3買進(⑤),符合3檔資金。
訊號=經理人e對股s的『主動超配』= w(e,s) − 跨ETF均w(s);及其日變化(領先加碼)。
"""
import sys, json
import pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
import yfinance as yf
from loguru import logger; logger.remove(); logger.add(sys.stderr,level="ERROR")

df=pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv",encoding="utf-8")
df["Date"]=pd.to_datetime(df["Date"]); df["Stock_Code"]=df["Stock_Code"].astype(str)
# 跨ETF均權重(每date,stock)= 共識權重
cons=df.groupby(["Date","Stock_Code"])["Weight(%)"].mean().rename("cons_w").reset_index()
df=df.merge(cons,on=["Date","Stock_Code"])
df["active"]=df["Weight(%)"]-df["cons_w"]               # 主動超配:e比共識多持多少
df=df.sort_values(["ETF_Code","Stock_Code","Date"])
df["active_chg"]=df.groupby(["ETF_Code","Stock_Code"])["active"].diff()  # 領先加碼(主動超配上升)
codes=sorted(df["Stock_Code"].unique()); etfs=sorted(df["ETF_Code"].unique())

# 價格+forward超額(D+1)
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
key=list(zip(df["Stock_Code"],df["Date"]))
f10cache={}
def f10(c,d):
    k=(c,d)
    if k not in f10cache: f10cache[k]=fexc(10,c,d)
    return f10cache[k]
df["f10"]=[f10(c,d) for c,d in zip(df["Stock_Code"],df["Date"])]

# 切分
dts=sorted(df["Date"].unique()); cut=dts[int(len(dts)*0.6)]
TR=df[df["Date"]<cut]; OO=df[df["Date"]>=cut]
print(f"前段 {TR['Date'].nunique()}日 / 後段 {OO['Date'].nunique()}日 (cut {pd.Timestamp(cut).date()})")

# 前段:每個經理人 skill = IC(主動超配 active, forward10) + IC(領先加碼 active_chg)
print("\n=== 前段:各經理人『主動超配』預測力 IC(排名)===")
sk={}
for e in etfs:
    g=TR[TR["ETF_Code"]==e].dropna(subset=["active","f10"])
    g2=TR[TR["ETF_Code"]==e].dropna(subset=["active_chg","f10"])
    ic_a=g["active"].corr(g["f10"],method="spearman") if len(g)>100 else np.nan
    ic_c=g2["active_chg"].corr(g2["f10"],method="spearman") if len(g2)>100 else np.nan
    sk[e]=(ic_a,ic_c)
rank=sorted(sk.items(),key=lambda x:(x[1][0] if pd.notna(x[1][0]) else -9),reverse=True)
for e,(ia,ic) in rank[:6]: print(f"  強 {e}: 超配IC {ia:+.3f}  加碼IC {ic:+.3f}")
print("  ...")
for e,(ia,ic) in rank[-4:]: print(f"  弱 {e}: 超配IC {ia:+.3f}  加碼IC {ic:+.3f}")
topK=[e for e,_ in rank[:5]]; botK=[e for e,_ in rank[-5:]]
print(f"\n前段選出 強經理人top5={topK}  弱bottom5={botK}")

# 後段OOS:跟強/弱/全部 的『主動超配』分數,看誰預測得準 + 組合alpha
print("\n=== 後段OOS:跟不同經理人組的 主動超配分數 → forward10 IC ===")
def group_score(sub, members):
    s=sub[sub["ETF_Code"].isin(members)].groupby(["Date","Stock_Code"])["active"].mean().rename("score").reset_index()
    return s
for name,mem in [("跟強top5",topK),("跟弱bottom5",botK),("跟全部28",etfs)]:
    gs=group_score(OO,mem)
    m=gs.merge(OO[["Date","Stock_Code","f10"]].drop_duplicates(),on=["Date","Stock_Code"])
    m=m.dropna(subset=["score","f10"])
    ic=m["score"].corr(m["f10"],method="spearman")
    # 組合:每日score top3 買進,⑤近似用 forward10 當報酬(扣0.6%)
    ls=[]
    for d,day in m.groupby("Date"):
        if day["Stock_Code"].nunique()<5: continue
        ls.append(day.nlargest(3,"score")["f10"].mean())
    print(f"  {name}: IC {ic:+.3f}  每日top3超額(扣成本) {np.mean(ls)-0.6:+.2f}pp/筆  n={len(m)}")
print("\n判讀:若『跟強top5』OOS的IC與top3超額 明顯 > 跟全部/跟弱 → 經理人skill有異質性、可OOS跟隨(你的點子成立)。")
print("若三組差不多 → 沒有可OOS辨識的強經理人,跟誰都一樣(點子在此資料不成立)。")
