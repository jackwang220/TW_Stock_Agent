"""主動式ETF共識持股訊號驗證(28檔×1年完整持股)。

測:跨28檔ETF的「共識流向」(多少檔在加碼/淨權重變化)能不能預測個股前向超額報酬?
- 訊號(每 date×stock,跨ETF聚合):n_hold(幾檔持有)、sum_wt(總權重)、net_chg(日權重變化總和=聚合經理人流向)、n_add/n_cut。
- forward 從 D+1 起(主動式ETF盤後揭露當日持股→隔日才能行動,避洩漏)。
- 評估:rank-IC、event study(共識加碼vs減碼)、分位 long-short tilt 回測 vs 0050(扣成本)。
資料:data/Active_ETF_1Y_Daily_28ETFs.csv。價格:yfinance 還原(含息)。
"""
import sys, json
import pandas as pd, numpy as np, yfinance as yf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv(ROOT/"data/Active_ETF_1Y_Daily_28ETFs.csv", encoding="utf-8")
df["Date"] = pd.to_datetime(df["Date"]); df["Stock_Code"] = df["Stock_Code"].astype(str)
codes = sorted(df["Stock_Code"].unique())
names = df.groupby("Stock_Code")["Stock_Name"].first().to_dict()

# 共識訊號:每 (date, stock) 跨 ETF 聚合
g = df.groupby(["Date", "Stock_Code"]).agg(
    n_hold=("ETF_Code", "nunique"),
    sum_wt=("Weight(%)", "sum"),
    net_chg=("Daily_Change(%)", "sum"),
    n_add=("Daily_Change(%)", lambda s: (s > 0).sum()),
    n_cut=("Daily_Change(%)", lambda s: (s < 0).sum()),
).reset_index()
g["net_breadth"] = g["n_add"] - g["n_cut"]      # 加碼檔數 - 減碼檔數

# 價格(還原含息)
print("下載價格...", flush=True)
px = {}
for c in codes + ["0050"]:
    try:
        s = yf.download(f"{c}.TW", start="2025-05-01", auto_adjust=True, progress=False)["Close"]
        if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
        s = s.dropna()
        if len(s) > 20: px[c] = s
    except Exception:
        pass
print(f"有價股票 {len(px)-1}/{len(codes)}", flush=True)
bench = px["0050"]

def fwd_excess(code, d, n):
    s = px.get(code)
    if s is None: return None
    si = s.index[s.index > d]; bi = bench.index[bench.index > d]
    if len(si) <= n or len(bi) <= n: return None
    return (float(s.loc[si[n]]/s.loc[si[0]]-1) - float(bench.loc[bi[n]]/bench.loc[bi[0]]-1)) * 100

for n in (1, 5, 10, 20):
    g[f"fwd{n}"] = [fwd_excess(c, d, n) for c, d in zip(g["Stock_Code"], g["Date"])]

print("\n=== rank-IC:共識訊號 vs 前向超額報酬(扣0050,forward從D+1) ===")
print(f"{'signal':<12} " + " ".join(f"fwd{n:<5}" for n in (1,5,10,20)))
for sigcol in ["net_chg", "net_breadth", "sum_wt", "n_hold"]:
    cells = []
    for n in (1, 5, 10, 20):
        sub = g.dropna(subset=[f"fwd{n}", sigcol])
        ic = sub[sigcol].corr(sub[f"fwd{n}"], method="spearman")
        cells.append(f"{ic:+.3f}")
    print(f"{sigcol:<12} " + "   ".join(cells))

print("\n=== event study:強共識加碼 vs 強共識減碼 的平均前向超額報酬% ===")
for n in (1, 5, 10, 20):
    sub = g.dropna(subset=[f"fwd{n}"])
    q_hi = sub[sub["net_breadth"] >= 3]   # ≥3檔淨加碼
    q_lo = sub[sub["net_breadth"] <= -3]  # ≥3檔淨減碼
    print(f"  fwd{n}: 強加碼 {q_hi[f'fwd{n}'].mean():+.2f}%(n={len(q_hi)})  "
          f"強減碼 {q_lo[f'fwd{n}'].mean():+.2f}%(n={len(q_lo)})  "
          f"差 {q_hi[f'fwd{n}'].mean()-q_lo[f'fwd{n}'].mean():+.2f}")

# 分位 long-short tilt 回測(每日依 net_chg 排序,多空各取 top/bottom 五分位,持有5日,扣成本)
print("\n=== 分位回測:每日依 net_chg 排序,long top20% / short bottom20%,持有n日(扣來回0.6%) ===")
COST = 0.6
for n in (5, 10):
    longs, shorts = [], []
    for d, day in g.groupby("Date"):
        day = day.dropna(subset=[f"fwd{n}"])
        if len(day) < 10: continue
        k = max(1, len(day)//5)
        top = day.nlargest(k, "net_chg"); bot = day.nsmallest(k, "net_chg")
        longs.append(top[f"fwd{n}"].mean()); shorts.append(bot[f"fwd{n}"].mean())
    L = np.mean(longs) - COST; S = np.mean(shorts) - COST
    print(f"  持有{n}日: long超額 {L:+.2f}%/筆  short超額 {S:+.2f}%  long-short {L-S:+.2f}pp  (天數{len(longs)})")

# 6 檔池外股(H 漏掉的)
u = set(json.loads((ROOT/"data/base_universe.json").read_text(encoding="utf-8")).keys())
outside = [c for c in codes if c not in u]
print(f"\n池外股({len(outside)}):", {c: names[c] for c in outside})
RPT = ROOT/"reports/etf_consensus_probe.md"
print(f"\n(數據已印出,可整理進 {RPT})")
