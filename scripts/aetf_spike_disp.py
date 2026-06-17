"""主動式ETF:A=突然廣泛加碼(co-add spike) + B=經理人分歧度(dispersion),各多種定義×多種測法。

A(突然加碼/spike):當天多檔ETF同時加碼同一股→可能是共識訊號。
B(分歧度):多檔ETF對同一股權重差異大→經理人意見分歧。
weight變化自己從 Weight 欄日差算(CSV 的 Daily_Change 欄疑似壞,不採用)。
leak-safe:訊號用決策日D(盤後揭露),forward報酬從D+1起算,扣同期0050。
測法(每個signal):rank-IC fwd5/10/20 + 分位long-short(D+1,持10日,扣來回0.6%)。
"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import yfinance as yf

df = pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv", encoding="utf-8")
df["Date"] = pd.to_datetime(df["Date"]); df["Stock_Code"] = df["Stock_Code"].astype(str)
df = df.sort_values(["ETF_Code","Stock_Code","Date"])
# 每(ETF,股)的權重日差(自己算,不信壞掉的 Daily_Change)
df["dwt"] = df.groupby(["ETF_Code","Stock_Code"])["Weight(%)"].diff()
codes = sorted(df["Stock_Code"].unique())
names = df.groupby("Stock_Code")["Stock_Name"].first().to_dict()

# 每(date,stock)跨28檔ETF聚合
EPS = 0.01  # 權重變化>0.01%才算「加碼」
def aggf(s):
    return pd.Series({
        "n_hold": s["Weight(%)"].gt(0).sum(),
        "sum_wt": s["Weight(%)"].sum(),
        "wt_std": s["Weight(%)"].std(ddof=0),
        "wt_mean": s["Weight(%)"].mean(),
        "wt_max": s["Weight(%)"].max(),
        "wt_min": s["Weight(%)"].min(),
        "n_add": (s["dwt"] > EPS).sum(),
        "net_dwt": s["dwt"].sum(),
    })
g = df.groupby(["Date","Stock_Code"]).apply(aggf).reset_index()
g = g.sort_values(["Stock_Code","Date"])
# 衍生
g["wt_cv"] = g["wt_std"]/g["wt_mean"].replace(0,np.nan)
g["wt_range"] = g["wt_max"]-g["wt_min"]
# spike:當天加碼檔數 相對該股近20日常態的 z-score
def zscore(x):
    m=x.rolling(20,min_periods=5).mean(); sd=x.rolling(20,min_periods=5).std(ddof=0)
    return (x-m)/sd.replace(0,np.nan)
g["breadth_z"] = g.groupby("Stock_Code")["n_add"].transform(zscore)
g["sumwt_jump"] = g.groupby("Stock_Code")["sum_wt"].diff()        # 聚合權重單日跳升
g["nhold_jump"] = g.groupby("Stock_Code")["n_hold"].diff()        # 持有檔數單日跳升
g["disp_chg"] = g.groupby("Stock_Code")["wt_std"].diff()          # 分歧度上升
g["disp_x_nhold"] = g["wt_std"]*g["n_hold"]

# 價格:用 tw_stock_index 的正確 yf_ticker(含 .TWO)
ti = json.loads((ROOT/"data/tw_stock_index.json").read_text(encoding="utf-8"))
def yft(c): return ti.get(c,{}).get("yf_ticker", f"{c}.TW")
def close(t):
    try:
        d=yf.download(t,start="2025-05-01",auto_adjust=True,progress=False)
        if d is None or d.empty: return None
        c=d["Close"]; c=c.iloc[:,0] if isinstance(c,pd.DataFrame) else c
        return c.dropna()
    except Exception: return None
px={}
for c in codes:
    s=close(yft(c))
    if s is None: s=close(f"{c}.TWO")
    if s is not None and len(s)>20: px[c]=s
bench=close("0050.TW")
if bench is None: bench=close("0050.TWO")
print(f"有價股票 {len(px)}/{len(codes)};缺:{[c for c in codes if c not in px]}", flush=True)

# 對齊日期、forward 超額(D+1起)
alldates=sorted(set().union(*[set(s.index) for s in px.values()], set(bench.index)))
pxm=pd.DataFrame({c:px[c].reindex(alldates) for c in px})
bm=bench.reindex(alldates)
def fexc_col(n):
    fwd=pxm.shift(-n)/pxm-1
    bfwd=bm.shift(-n)/bm-1
    return fwd.sub(bfwd,axis=0)*100   # 超額 % ,index=date col=stock
FEXC={n:fexc_col(n) for n in (5,10,20)}
# 把 forward 超額(從D+1起 = shift多一格)貼回 g:訊號在D→賺D+1..D+1+n,用 D 的下一交易日對齊
date_pos={d:i for i,d in enumerate(alldates)}
def lookup_fwd(n,code,d):
    i=date_pos.get(d)
    if i is None or i+1>=len(alldates) or code not in pxm.columns: return np.nan
    d1=alldates[i+1]
    return FEXC[n].at[d1,code] if d1 in FEXC[n].index else np.nan
for n in (5,10,20):
    g[f"fexc{n}"]=[lookup_fwd(n,c,d) for c,d in zip(g["Stock_Code"],g["Date"])]

SIGNALS = {
  # A:突然廣泛加碼 / spike
  "A1 加碼檔數n_add":"n_add", "A2 加碼spike(z)":"breadth_z", "A3 淨流向net_dwt":"net_dwt",
  "A4 聚合權重跳升":"sumwt_jump", "A5 持有檔數跳升":"nhold_jump",
  # B:分歧度
  "B1 權重std(分歧)":"wt_std", "B2 變異係數cv":"wt_cv", "B3 權重range":"wt_range",
  "B4 分歧上升disp_chg":"disp_chg", "B5 分歧×檔數":"disp_x_nhold",
}
# combo
g["combo_AxB"]=zscore_safe = (g["n_add"].rank(pct=True)+g["wt_std"].rank(pct=True))  # A1+B1 百分位相加
SIGNALS["COMBO A1+B1"]="combo_AxB"

print("\n=== rank-IC(訊號@D vs forward超額@D+1起,扣0050)+ 分位long-short(10日,扣0.6%)===")
print(f"{'signal':<18} {'IC5':>7} {'IC10':>7} {'IC20':>7}   {'LS10(pp/筆)':>10}")
for label,col in SIGNALS.items():
    row=[]
    for n in (5,10,20):
        sub=g.dropna(subset=[col,f"fexc{n}"])
        ic=sub[col].corr(sub[f"fexc{n}"],method="spearman") if len(sub)>50 else np.nan
        row.append(ic)
    # 分位 LS 10日:每日依signal分位,long top20%-short bottom20%
    ls=[]
    for d,day in g.dropna(subset=[col,"fexc10"]).groupby("Date"):
        if len(day)<10: continue
        k=max(1,len(day)//5)
        ls.append(day.nlargest(k,col)["fexc10"].mean()-day.nsmallest(k,col)["fexc10"].mean())
    lsv=np.mean(ls)-0.6 if ls else np.nan
    print(f"{label:<18} {row[0]:>+7.3f} {row[1]:>+7.3f} {row[2]:>+7.3f}   {lsv:>+10.2f}")
print("\n註:IC≈0或LS≤0(扣成本)=無預測力;IC>+0.05且LS明顯>0才值得進tilt+placebo回測。leak-safe:forward從D+1起。")

# ============ B2(wt_cv)deep-dive:placebo + 分位單調 + 它在選什麼 ============
print("\n\n############ B2 變異係數(wt_cv)DEEP-DIVE ############")
sub=g.dropna(subset=["wt_cv","fexc10"]).copy()
# 1) 分位單調(Q1低CV..Q5高CV 的 fwd10 超額)
sub["q"]=sub.groupby("Date")["wt_cv"].transform(lambda s: pd.qcut(s.rank(method="first"),5,labels=False) if s.nunique()>=5 else np.nan)
qm=sub.groupby("q")["fexc10"].mean()
print("分位 fwd10 超額%(Q0=最低CV..Q4=最高CV):", " ".join(f"Q{int(q)}={v:+.2f}" for q,v in qm.items()))
print(f"  Q4−Q0 = {qm.get(4,np.nan)-qm.get(0,np.nan):+.2f}pp;單調?", list(np.round(qm.values,2)))
# 2) placebo:每日把 wt_cv 隨機洗牌,重算 LS10,看 +1.68 在隨機分布的哪
def ls_of(colvals):
    tmp=sub.assign(_s=colvals); out=[]
    for d,day in tmp.groupby("Date"):
        if len(day)<10: continue
        k=max(1,len(day)//5)
        out.append(day.nlargest(k,"_s")["fexc10"].mean()-day.nsmallest(k,"_s")["fexc10"].mean())
    return np.mean(out)-0.6
real_ls=ls_of(sub["wt_cv"].values)
rng=np.random.RandomState(0); placebo=[]
for _ in range(200):
    shuffled=sub.groupby("Date")["wt_cv"].transform(lambda s: s.sample(frac=1,random_state=rng.randint(1e9)).values)
    placebo.append(ls_of(shuffled.values))
placebo=np.array(placebo)
print(f"\nplacebo洗牌(200次)LS10分布: mean={placebo.mean():+.2f} std={placebo.std():.2f} 95%上界={np.percentile(placebo,95):+.2f}")
print(f"真實 B2 LS10 = {real_ls:+.2f} → {'超出95%隨機(可能真訊號)' if real_ls>np.percentile(placebo,95) else '落在隨機範圍內(疑似雜訊)'};百分位={(placebo<real_ls).mean()*100:.0f}%")
# 3) 高CV vs 低CV 選到什麼股(平均權重、持有檔數、是不是池外中型)
u=set(json.loads((ROOT/"data/base_universe.json").read_text(encoding="utf-8")).keys())
hi=sub[sub["q"]==4]; lo=sub[sub["q"]==0]
print(f"\n高CV組(Q4): 平均wt_mean={hi['wt_mean'].mean():.2f}% 平均n_hold={hi['n_hold'].mean():.1f} 池外股佔比={hi['Stock_Code'].apply(lambda c:c not in u).mean()*100:.0f}%")
print(f"低CV組(Q0): 平均wt_mean={lo['wt_mean'].mean():.2f}% 平均n_hold={lo['n_hold'].mean():.1f} 池外股佔比={lo['Stock_Code'].apply(lambda c:c not in u).mean()*100:.0f}%")
print("高CV組最常入選股:", [f"{c}{names.get(c,c)}" for c in hi["Stock_Code"].value_counts().head(8).index.tolist()])
# 4) wt_cv 跟 wt_mean 的關係(是不是只是在抓小部位)
print(f"\ncorr(wt_cv, wt_mean) = {sub['wt_cv'].corr(sub['wt_mean']):+.3f}(很負=CV只是抓小部位股的artifact)")
# 5) 更保守 D+2 LS(再延一天,洩漏更不可能)
sub2=g.dropna(subset=["wt_cv"]).copy()
def lookup_fwd2(n,code,d):  # D+2 起
    i=date_pos.get(d)
    if i is None or i+2>=len(alldates) or code not in pxm.columns: return np.nan
    d2=alldates[i+2]
    return FEXC[n].at[d2,code] if d2 in FEXC[n].index else np.nan
sub2["f10_d2"]=[lookup_fwd2(10,c,d) for c,d in zip(sub2["Stock_Code"],sub2["Date"])]
sub2=sub2.dropna(subset=["f10_d2"]); out=[]
for d,day in sub2.groupby("Date"):
    if len(day)<10: continue
    k=max(1,len(day)//5); out.append(day.nlargest(k,"wt_cv")["f10_d2"].mean()-day.nsmallest(k,"wt_cv")["f10_d2"].mean())
print(f"D+2(更保守leak-free)LS10 = {np.mean(out)-0.6:+.2f}(跟D+1的{real_ls:+.2f}比,差很多=洩漏放大)")
