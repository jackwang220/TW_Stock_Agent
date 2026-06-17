"""retest momentum_beta:
被否決結論「動能=beta非alpha(方向命中≤50%)」其證據 42/48/45/50/49 來自 crossperiod_analyze
的 up勝盤% 欄 = LLM predicted_direction=='up' 個股「隔日(1d)alpha>0」的比例,且 universe 只有 top12 大型股。
那是 LLM-as-selector 的 1 日命中率,並非任何「動能訊號變體」的測試。

本腳本做 T2 真正該問的事(全 112 檔 universe、point-in-time、扣 0050 beta):
  1. regime-conditional:把動能訊號在每個 regime 內分別測 forward alpha(不要把多空混算)。
  2. dose-response:把訊號分位數(quintile)→ forward alpha 分桶,看單調性。
  3. IC:Spearman/Pearson(signal, forward_alpha) 而非粗命中率。
  4. 同時測「絕對動能 ret20」(疑似 beta)與「殘差/相對動能 rs20(扣大盤)」(疑似 alpha)。
  5. forward horizon 取 5d 與 20d(動能本就非 1 日現象)。

全程 FinMind 快取,離線。
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2021-01-01"
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
HORIZONS = [5, 20]
MKT = "0050"

def spearman(xs, ys):
    n=len(xs)
    if n<10: return None
    def rank(v):
        order=sorted(range(len(v)), key=lambda i:v[i]); r=[0.0]*len(v); i=0
        while i<len(v):
            j=i
            while j+1<len(v) and v[order[j+1]]==v[order[i]]: j+=1
            avg=(i+j)/2.0+1
            for k in range(i,j+1): r[order[k]]=avg
            i=j+1
        return r
    rx,ry=rank(xs),rank(ys)
    return pearson(rx,ry)

def pearson(xs,ys):
    n=len(xs)
    if n<10: return None
    mx=sum(xs)/n; my=sum(ys)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    sxx=sum((a-mx)**2 for a in xs); syy=sum((b-my)**2 for b in ys)
    if sxx<=0 or syy<=0: return None
    return sxy/math.sqrt(sxx*syy)

def main():
    u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes=list(u.keys())
    OH={c:get_daily_ohlcv(c,start=START) for c in codes}
    OH[MKT]=get_daily_ohlcv(MKT,start=START)
    # precompute per-code: sorted days, close series, ret20 (abs momentum), and index for fwd
    idx_days=sorted(OH[MKT]); idx_close={d:OH[MKT][d]["close"] for d in idx_days}
    # index 20d trailing ret per day (for relative strength)
    series={}
    for c in list(OH.keys()):
        ds=sorted(OH[c]); cl=[OH[c][d]["close"] for d in ds]
        series[c]=(ds,cl,{d:i for i,d in enumerate(ds)})

    def fwd_ret(c,d,h):
        ds,cl,pos=series[c]; i=pos.get(d)
        if i is None or i+h>=len(ds): return None
        if cl[i]<=0: return None
        return cl[i+h]/cl[i]-1
    def trail_ret(c,d,k):
        ds,cl,pos=series[c]; i=pos.get(d)
        if i is None or i-k<0 or cl[i-k]<=0: return None
        return cl[i]/cl[i-k]-1

    # build observations: for each trading day (sampled, step to reduce overlap), each stock
    # signals: abs_mom = trailing 20d ret; rel_mom = (1+stock20)/(1+idx20)-1 ; trend = ma5/ma20-1
    out_lines=["# retest momentum_beta — regime-conditional dose-response + IC\n",
               "> 訊號: abs_mom=過去20d報酬(絕對動能/疑似beta); rel_mom=扣大盤的相對動能(疑似alpha); 全112檔universe, point-in-time, fwd alpha=個股fwd報酬-0050同期\n"]

    # sample dates: every 5 trading days to reduce return overlap autocorrelation
    for h in HORIZONS:
        out_lines.append(f"\n## Forward {h}d ── alpha = stock_fwd{h} - 0050_fwd{h}\n")
        for signame in ["abs_mom","rel_mom"]:
            out_lines.append(f"\n### 訊號={signame}, fwd={h}d\n")
            out_lines.append("| regime | n | IC(spearman) | Q1低 | Q2 | Q3 | Q4 | Q5高 | Q5-Q1 | 單調? |")
            out_lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for rlab,rs_,re_ in REGIMES:
                # collect (signal, fwd_alpha)
                obs=[]
                sample_days=[d for d in idx_days if rs_<=d<=re_]
                sample_days=sample_days[::5]
                for d in sample_days:
                    # index fwd ret for this day,h
                    ii=series[MKT][2].get(d)
                    if ii is None or ii+h>=len(series[MKT][0]): continue
                    idx_fwd=series[MKT][1][ii+h]/series[MKT][1][ii]-1 if series[MKT][1][ii]>0 else None
                    if idx_fwd is None: continue
                    idx20=trail_ret(MKT,d,20)
                    for c in codes:
                        s20=trail_ret(c,d,20)
                        if s20 is None: continue
                        if signame=="abs_mom":
                            sig=s20
                        else:
                            if idx20 is None or abs(1+idx20)<1e-6: continue
                            sig=(1+s20)/(1+idx20)-1
                        fr=fwd_ret(c,d,h)
                        if fr is None: continue
                        alpha=fr-idx_fwd
                        obs.append((sig,alpha))
                n=len(obs)
                if n<50:
                    out_lines.append(f"| {rlab} | {n} | n太小 | | | | | | | |")
                    continue
                ic=spearman([o[0] for o in obs],[o[1] for o in obs])
                # quintiles by signal
                obs.sort(key=lambda x:x[0])
                qs=[]
                for k in range(5):
                    a=n*k//5; b=n*(k+1)//5
                    seg=[o[1] for o in obs[a:b]]
                    qs.append(sum(seg)/len(seg)*100 if seg else 0)
                mono = all(qs[i]<=qs[i+1] for i in range(4)) or all(qs[i]>=qs[i+1] for i in range(4))
                q5q1=qs[4]-qs[0]
                out_lines.append(f"| {rlab} | {n} | {ic:+.3f} | {qs[0]:+.2f}% | {qs[1]:+.2f}% | {qs[2]:+.2f}% | {qs[3]:+.2f}% | {qs[4]:+.2f}% | {q5q1:+.2f}% | {'是' if mono else '否'} |")

    (ROOT/"reports"/"retest_momentum_beta.md").write_text("\n".join(out_lines),encoding="utf-8")
    print("\n".join(out_lines))
    print("\n→ reports/retest_momentum_beta.md")

if __name__=="__main__":
    main()
