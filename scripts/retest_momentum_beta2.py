"""retest momentum_beta (part 2) — 直接用產生「42/48/45/50/49」那份 crossperiod_results.csv,
但改用正確的問法重新切:
  原否決證據 = up勝盤% = (predicted_direction=='up' 的股票, 隔日1d alpha>0 的比例) → 42/48/45/50/49。
  這是 LLM-as-selector 的 1 日二元命中率,粗糙:
    (a) 1 日 horizon 雜訊極大;(b) 二元化丟掉劑量;(c) top12 大型股 universe(高 beta)。
  正確問法(用同一份 CSV 內已有的欄位):
    - dose-response: 把 LLM 連續訊號 predicted_center_pct 分桶 → 看 forward alpha 是否單調。
    - IC: corr(predicted_center_pct, alpha_1d)  以及 corr(bull-bear分差, alpha)。
    - regime-conditional 已天然分好(CSV 有 regime 欄)。
  並對照「同一批股票若用 raw ret(不扣 beta)」vs「alpha(扣 beta)」命中率差多少 → 量化『beta 灌水』。
"""
import sys, csv, math
from collections import defaultdict
sys.path.insert(0,"src"); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR

rows=list(csv.DictReader(open(DATA_DIR/"crossperiod_results.csv",encoding="utf-8")))
def f(x):
    try: return float(x)
    except: return None

def pearson(xs,ys):
    n=len(xs)
    if n<10: return None
    mx=sum(xs)/n;my=sum(ys)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    sxx=sum((a-mx)**2 for a in xs);syy=sum((b-my)**2 for b in ys)
    if sxx<=0 or syy<=0: return None
    return sxy/math.sqrt(sxx*syy)

def spearman(xs,ys):
    n=len(xs)
    if n<10: return None
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]);r=[0.0]*len(v);i=0
        while i<len(v):
            j=i
            while j+1<len(v) and v[o[j+1]]==v[o[i]]: j+=1
            a=(i+j)/2.0+1
            for k in range(i,j+1): r[o[k]]=a
            i=j+1
        return r
    return pearson(rk(xs),rk(ys))

by=defaultdict(list)
for r in rows: by[r["regime"]].append(r)

L=["# retest momentum_beta part2 — 用產生 42/48/45/50/49 那份 CSV 的正確切法\n",
   "說明: 原證據是『LLM up 預測股, 隔日1d alpha>0 比例』(二元命中率)。下面改用劑量(predicted_center_pct分桶)+IC。\n"]

# 1) beta 灌水量化: 同一批 up 股, 用 raw ret 命中 vs 用 alpha 命中
L.append("\n## 1) 量化 beta 灌水: LLM up 股的『隔日漲(raw)』比例 vs 『贏0050(alpha)』比例\n")
L.append("| regime | up數 | raw勝率(漲) | alpha勝率(贏盤) | 差(=beta灌水) |")
L.append("|---|---|---|---|---|")
for reg,rs in by.items():
    ups=[r for r in rs if r["predicted_direction"]=="up" and f(r["ret_1d"]) is not None and f(r["alpha_1d"]) is not None]
    if not ups: continue
    raww=sum(1 for r in ups if f(r["ret_1d"])>0)/len(ups)*100
    alw=sum(1 for r in ups if f(r["alpha_1d"])>0)/len(ups)*100
    L.append(f"| {reg} | {len(ups)} | {raww:.0f}% | {alw:.0f}% | {raww-alw:+.0f}pp |")

# 2) dose-response on predicted_center_pct -> forward alpha (regime-conditional)
L.append("\n## 2) Dose-response: LLM predicted_center_pct(看多強度)分桶 → 隔日 alpha 均值\n")
L.append("| regime | n | IC_spear(center,alpha) | 低桶alpha | 中桶 | 高桶 | 高-低 |")
L.append("|---|---|---|---|---|---|---|")
for reg,rs in by.items():
    pts=[(f(r["predicted_center_pct"]),f(r["alpha_1d"])) for r in rs
         if f(r["predicted_center_pct"]) is not None and f(r["alpha_1d"]) is not None]
    if len(pts)<30:
        L.append(f"| {reg} | {len(pts)} | n太小 | | | | |"); continue
    ic=spearman([p[0] for p in pts],[p[1] for p in pts])
    pts.sort(key=lambda x:x[0]); n=len(pts)
    def seg(a,b): s=[p[1] for p in pts[a:b]]; return sum(s)/len(s)*100 if s else 0
    lo=seg(0,n//3); mid=seg(n//3,2*n//3); hi=seg(2*n//3,n)
    L.append(f"| {reg} | {n} | {ic:+.3f} | {lo/100:+.2f}% | {mid/100:+.2f}% | {hi/100:+.2f}% | {(hi-lo)/100:+.2f}% |")

# 3) pooled IC across all regimes for the continuous signal
allpts=[(f(r["predicted_center_pct"]),f(r["alpha_1d"])) for r in rows
        if f(r["predicted_center_pct"]) is not None and f(r["alpha_1d"]) is not None]
ic_all=spearman([p[0] for p in allpts],[p[1] for p in allpts])
L.append(f"\n## 3) 全期 pooled IC(center vs 隔日alpha) = {ic_all:+.3f} (n={len(allpts)})")
L.append("> |IC| 一般 >0.03~0.05 才算有微弱訊號;接近 0 = 無方向預測力。")

# 4) up vs down vs neutral 的 alpha 均值(regime內) — LLM 方向分組的真實前瞻 alpha
L.append("\n## 4) LLM 方向分組的隔日平均 alpha(regime-conditional)\n")
L.append("| regime | up_α | neutral_α | down_α | up-down價差 |")
L.append("|---|---|---|---|---|")
for reg,rs in by.items():
    def grp(dirn):
        v=[f(r["alpha_1d"]) for r in rs if r["predicted_direction"]==dirn and f(r["alpha_1d"]) is not None]
        return sum(v)/len(v) if v else None
    u_,n_,d_=grp("up"),grp("neutral"),grp("down")
    spread=(u_-d_) if (u_ is not None and d_ is not None) else None
    L.append(f"| {reg} | {u_:+.2f}% | {n_ if n_ is None else f'{n_:+.2f}%'} | {d_ if d_ is None else f'{d_:+.2f}%'} | {spread if spread is None else f'{spread:+.2f}%'} |")

(__import__('pathlib').Path("reports/retest_momentum_beta2.md")).write_text("\n".join(L),encoding="utf-8")
print("\n".join(L))
