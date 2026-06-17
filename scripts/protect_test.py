"""下檔防禦 — 多組方法發散測試(overlay 在策略逐日P&L上)。
基準=現行live(原版/exit_only/INC1.5)跑一次拿逐日P&L;再套防禦遮罩:
家族A(切現金 mask=0): MA20/MA60/MA120濾網、回撤觸發、TS動能、波動、廣度
家族B(降曝險 ×0.5/連續): 趨勢半倉、趨勢連續縮
比較:全期報酬 / 最大回撤MDD / 報酬÷MDD / 2022空頭 / 2025-04跌窗 / 牛市爆衝(2025-11+2026-05保留多少) / 空手天數%。
leak-safe:遮罩用前一交易日資訊。
"""
import sys, json, math
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove(); logger.add(sys.stderr,level="ERROR")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
import importlib.util as U
def load(n,p):
    s=U.spec_from_file_location(n,p); m=U.module_from_spec(s); s.loader.exec_module(m); return m
nine=load("nine",ROOT/"scripts/exp_step1_v9.py")
is_bull,h_score,_factors=nine.is_bull,nine.h_score,nine._factors
v5,v6,ec=nine.v5,nine.v6,nine.ec; sim5=ec.sim_buyclose_sellopen
features=v5.features; get_ohlcv=nine.get_daily_ohlcv; START=nine.START
u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
codes=list(u.keys()); turns={c:u[c].get("avg_turnover",0.0) for c in codes}
print("載入...",flush=True)
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
                g=rebound_signal(cl,turns.get(c,0.0))
                if g.get("fired"): m[d]=g["score"]*100
            except Exception: pass
        if j>0 and OH[c][ds[j-1]]["close"]>0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1>=0.095: s.add(d)
    reb[c]=m; limitup[c]=s
turn_pct={}
for d in alld:
    vals=sorted(((c,feats[c][d]["turn"]) for c in codes if d in feats.get(c,{}) and feats[c][d]["turn"]>0),key=lambda x:x[1])
    turn_pct[d]={c:(i+1)/len(vals) for i,(c,_) in enumerate(vals)} if vals else {}
rows=[]
for d in alld:
    ir=twii.get(d,{}).get("ret20"); bull=is_bull(twii.get(d,{}),"原版c>20MA")
    for c in codes:
        f=feats.get(c,{})
        if d not in f or math.isnan(f[d].get("ma20",float("nan"))): continue
        sc=h_score(_factors(f[d],ir),turn_pct.get(d,{}).get(c,0.5)) if bull else reb.get(c,{}).get(d,0.0)*1.5
        if sc>0: rows.append((d,c,sc/100))
base=sim5(rows,opens,closes,limitup,incumbent=1.5,sell_mode="exit_only")
cal=base["cal"]; pnl=np.array(base["day_pnl"]); contributed=base["contributed"]
print(f"基準: 全期報酬{sum(pnl)/contributed*100:+.0f}% MDD {base['mdd']/contributed*100:.0f}% (contributed {contributed:.0f})")

# ── 0050 訊號序列(leak-safe:用前一交易日)──
oc=closes["0050"]; od=sorted(oc)
def ma(d,n):
    i=od.index(d); return np.mean([oc[od[k]] for k in range(i-n+1,i+1)]) if i>=n-1 else None
def ret(d,n):
    i=od.index(d); return (oc[d]/oc[od[i-n]]-1)*100 if i>=n else None
def dd60(d):  # 距60日高點回撤%
    i=od.index(d); hi=max(oc[od[k]] for k in range(max(0,i-59),i+1)); return (oc[d]/hi-1)*100
def vol20(d):
    i=od.index(d)
    if i<20: return None
    rr=[oc[od[k]]/oc[od[k-1]]-1 for k in range(i-19,i+1)]; return np.std(rr)*100
def breadth(d):
    ab=[1 for c in codes if d in feats.get(c,{}) and feats[c][d].get("close") and feats[c][d].get("ma20") and not math.isnan(feats[c][d]["ma20"]) and feats[c][d]["close"]>feats[c][d]["ma20"]]
    tot=[1 for c in codes if d in feats.get(c,{}) and feats[c][d].get("ma20") and not math.isnan(feats[c][d]["ma20"])]
    return len(ab)/len(tot) if tot else None
# 前一交易日對映
prevd={od[i]:od[i-1] for i in range(1,len(od))}
volhist=[vol20(d) for d in od if vol20(d) is not None]; VOLP80=np.percentile(volhist,80)
def sig_at(d,fn):  # 用前一交易日的值(leak-safe)
    pd_=prevd.get(d); return fn(pd_) if pd_ in oc else None
METHODS={
 "A1 0050<MA20→現金": lambda d:0.0 if (sig_at(d,lambda x:(oc[x]<(ma(x,20) or 1e9)))) else 1.0,
 "A2 0050<MA60→現金": lambda d:0.0 if (sig_at(d,lambda x:(oc[x]<(ma(x,60) or 1e9)))) else 1.0,
 "A3 0050<MA120→現金":lambda d:0.0 if (sig_at(d,lambda x:(oc[x]<(ma(x,120) or 1e9)))) else 1.0,
 "A4 回撤>8%→現金":   lambda d:0.0 if (sig_at(d,lambda x:(dd60(x)<-8))) else 1.0,
 "A5 0050 20日<0→現金":lambda d:0.0 if (sig_at(d,lambda x:((ret(x,20) or 0)<0))) else 1.0,
 "A6 高波動→現金":     lambda d:0.0 if (sig_at(d,lambda x:((vol20(x) or 0)>VOLP80))) else 1.0,
 "A7 廣度<40%→現金":  lambda d:0.0 if (sig_at(d,lambda x:((breadth(x) or 1)<0.40))) else 1.0,
 "B1 MA60→半倉":      lambda d:0.5 if (sig_at(d,lambda x:(oc[x]<(ma(x,60) or 1e9)))) else 1.0,
 "B2 趨勢連續縮":      lambda d:(lambda v: max(0.3,min(1.0,1.0+ (v/10 if v else 0)))) (sig_at(d,lambda x:(dd60(x)))),
}
def metrics(mask):
    p=pnl*np.array([mask(cal[i]) if mask else 1.0 for i in range(len(cal))])
    tot=sum(p)/contributed*100
    cum=np.cumsum(p); peak=np.maximum.accumulate(cum); mdd=(peak-cum).max()/contributed*100
    def psum(s,e): return sum(p[i] for i in range(len(cal)) if s<=cal[i]<=e)/contributed*100
    bull=psum("2025-10-20","2025-11-30")+psum("2026-04-01","2026-05-31")
    cashpct=np.mean([1 for i in range(len(cal)) if (mask(cal[i]) if mask else 1)<0.999]) if mask else 0
    cashpct=sum(1 for i in range(len(cal)) if (mask(cal[i]) if mask else 1)<0.999)/len(cal)*100 if mask else 0
    return tot,mdd,psum("2022-01-01","2022-12-31"),psum("2025-03-01","2025-04-30"),bull,cashpct
print(f"\n{'方法':<20}{'報酬%':>7}{'MDD%':>7}{'報酬/MDD':>9}{'2022':>7}{'2025-04跌':>9}{'牛爆衝':>8}{'空手%':>7}")
bt,bm,b22,b25,bb,_=metrics(None)
print(f"{'基準(無防禦)':<20}{bt:>+7.0f}{bm:>7.0f}{bt/bm:>9.2f}{b22:>+7.0f}{b25:>+9.0f}{bb:>+8.0f}{0:>6.0f}%")
for name,mk in METHODS.items():
    t,m,a22,a25,bull,cash=metrics(mk)
    print(f"{name:<20}{t:>+7.0f}{m:>7.0f}{t/m if m>0 else 0:>9.2f}{a22:>+7.0f}{a25:>+9.0f}{bull:>+8.0f}{cash:>6.0f}%")
print("\n判讀:好的防禦=MDD與2022/2025-04跌幅明顯縮小、報酬/MDD變高,但牛爆衝與總報酬別犧牲太多。")
print("注意:這是overlay近似(防禦日P&L歸零=當天空手免成本),且空頭樣本少(2022+2025-04),別overfit門檻。")
