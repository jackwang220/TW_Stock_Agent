"""現行H雙引擎 walk-forward overfit 稽核(用v9正式引擎,空頭腿rebound×1.5)。
1) 先驗證重現v9正式regime數字(原版c>20MA, INC1.5)
2) walk-forward:每個regime當測試期,只用『該期之前』資料挑最佳(bull門檻×INC),套到該期=真OOS
3) 對照:fixed現行(原版/INC1.5) vs walk-forward選 vs full-sample最佳(=overfit上界)
搜尋空間=v9的5種bull偵測 × INC{1.5,2.0}。選參準則=train期複合報酬最高。
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
features=v5.features; get_ohlcv=nine.get_daily_ohlcv
DETECT=nine.DETECT; REGIMES=nine.REGIMES; START=nine.START

u=json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
codes=list(u.keys()); turns={c:u[c].get("avg_turnover",0.0) for c in codes}
print("載入(v9引擎,全史)...",flush=True)
OH={c:get_ohlcv(c,start=START) for c in codes}; OH["0050"]=get_ohlcv("0050",start=START)
v5._OH=OH
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

def rows_for(mode):  # v9 正式:多頭純H / 空頭 rebound×1.5
    out=[]
    for d in alld:
        ir=twii.get(d,{}).get("ret20"); bull=is_bull(twii.get(d,{}),mode)
        for c in codes:
            f=feats.get(c,{})
            if d not in f or math.isnan(f[d].get("ma20",float("nan"))): continue
            sc=h_score(_factors(f[d],ir),turn_pct.get(d,{}).get(c,0.5)) if bull else reb.get(c,{}).get(d,0.0)*1.5
            if sc>0: out.append((d,c,sc/100))
    return out
ROWS={mode:rows_for(mode) for mode in DETECT}
bench={lab:v6.bench_0050(opens["0050"],closes["0050"],[d for d in alld if s<=d<=e]) for lab,s,e in REGIMES}
def alpha(mode,inc,s,e):
    r=sim5([x for x in ROWS[mode] if s<=x[0]<=e],opens,closes,limitup,incumbent=inc,sell_mode="exit_only")
    lab=[L for L,a,b in REGIMES if a==s][0] if any(a==s for L,a,b in REGIMES) else None
    return (r["ret"]-bench[lab]) if (r and lab) else (r["ret"] if r else None)

# 1) 驗證重現 v9(原版c>20MA, INC1.5)
print("\n=== 驗證:重現v9正式數字(原版c>20MA, INC1.5)===")
print("  regime alpha:",{lab:round(alpha("原版c>20MA",1.5,s,e)) for lab,s,e in REGIMES})
print("  (STEP1記錄: 2021+62/2022-3/2023+13/2024-25-38/2025下-26+214)")

# 2) walk-forward:regime依序當測試期,只用之前資料挑最佳(mode×INC),選參準則=train複合報酬
REGS=list(REGIMES)  # [(lab,s,e),...] 已時序
INCS=[1.5,2.0]
def regime_ret(mode,inc,s,e):
    r=sim5([x for x in ROWS[mode] if s<=x[0]<=e],opens,closes,limitup,incumbent=inc,sell_mode="exit_only"); return r["ret"] if r else None
def compound_over(modes_inc, regs):  # 在給定regimes上的複合報酬
    p=1.0
    for lab,s,e in regs:
        rr=regime_ret(modes_inc[0],modes_inc[1],s,e)
        if rr is not None: p*=(1+rr/100)
    return p-1
print("\n=== walk-forward(每regime只用之前資料選參)===")
print(f"{'測試期(OOS)':<14}{'WF選的參數':<22}{'WF OOS α':>9}{'現行固定α':>9}{'全史最佳α':>9}")
wf_chain=[]; fix_chain=[]; orac_chain=[]
for i,(lab,s,e) in enumerate(REGS):
    prior=REGS[:i]   # 嚴格之前的regimes
    cur_a=alpha("原版c>20MA",1.5,s,e)
    # full-sample 最佳(用所有regime含未來→overfit上界):挑在全部regime複合最高的combo
    combos=[(m,inc) for m in DETECT for inc in INCS]
    if i==0:
        pick=("原版c>20MA",1.5); note="(無先前資料→用預設)"
    else:
        best=max(combos,key=lambda mi: compound_over(mi,prior)); pick=best; note=""
    wf_a=alpha(pick[0],pick[1],s,e)
    orac=max(combos,key=lambda mi: compound_over(mi,REGS))   # 用全史(含未來)挑
    orac_a=alpha(orac[0],orac[1],s,e)
    wf_chain.append(wf_a); fix_chain.append(cur_a); orac_chain.append(orac_a)
    print(f"{lab:<14}{(pick[0]+'/INC'+str(pick[1])):<22}{wf_a:>+9.0f}{cur_a:>+9.0f}{orac_a:>+9.0f}  {note}")
def avg(x): x=[v for v in x if v is not None]; return sum(x)/len(x) if x else 0
print(f"\n平均OOS α: walk-forward {avg(wf_chain):+.0f} | 現行固定 {avg(fix_chain):+.0f} | 全史最佳(overfit上界) {avg(orac_chain):+.0f}")
print("haircut = 全史最佳 − walk-forward =",f"{avg(orac_chain)-avg(wf_chain):+.0f}pp(這就是overfit灌水的量)")
print("\n判讀:全史最佳 >> walk-forward → 用全史挑參數是灌水(overfit);現行固定 vs walk-forward 看現行算不算穩健選擇。")

# ── 滾動60日窗(step30) × INC{1.5,1.75,2.0}:看alpha是某窗暴衝還是連續穩定 ──
INCS=[1.5,1.75,2.0]
win=60; step=30; lookback=560
windays=alld[-lookback:] if len(alld)>=lookback else alld
ends=list(range(win-1,len(windays),step))
def bench_fn(ds): dd=sorted(x for x in ds if x in closes["0050"]); return v6.bench_0050(opens["0050"],closes["0050"],dd)
print(f"\n=== 滾動{win}日窗(每{step}日一格)× INC{INCS}(原版/exit_only)===")
print("欄: 0050原始% | 策略原始%(INC1.5) | 之後三欄=各INC的alpha(扣0050)")
print(f"{'窗結束日':<12}{'0050%':>7}{'策略1.5%':>9}"+"".join(f"{'α'+str(i):>7}" for i in INCS)+"  贏家")
acc={i:[] for i in INCS}
for ei in ends:
    wdays=set(windays[ei-win+1:ei+1]); bv=bench_fn(wdays); vals={}; raw15=None
    rawrow=""
    for inc in INCS:
        r=sim5([x for x in ROWS["原版c>20MA"] if x[0] in wdays],opens,closes,limitup,incumbent=inc,sell_mode="exit_only")
        a=(r["ret"]-bv) if (r and bv is not None) else None; vals[inc]=a; acc[inc].append(a)
        if inc==1.5 and r: raw15=r["ret"]
        rawrow+=f"{a:>+7.0f}" if a is not None else f"{'—':>7}"
    good={k:v for k,v in vals.items() if v is not None}
    win_tag=f"  INC{max(good,key=good.get)}" if good else ""
    b=f"{bv:>+7.0f}" if bv is not None else f"{'—':>7}"; rr=f"{raw15:>+9.0f}" if raw15 is not None else f"{'—':>9}"
    print(f"{windays[ei]:<12}{b}{rr}{rawrow}{win_tag}")
print(f"\n{'各INC窗平均':<12}"+"".join(f"{avg(acc[i]):>+8.0f}" for i in INCS))
print(f"{'最差窗':<12}"+"".join(f"{min(v for v in acc[i] if v is not None):>+8.0f}" for i in INCS))
for i in INCS:
    wins=sum(1 for j,e in enumerate(ends) if acc[i][j] is not None and acc[i][j]==max((acc[k][j] for k in INCS if acc[k][j] is not None),default=None))
    print(f"  INC{i}: 當贏家 {wins}/{len(ends)} 窗")
print("\n判讀:若某INC只在1-2個窗暴衝、其餘輸→那是單一時段運氣;若連續多窗(一季/半年)都贏→才是穩定優勢。")
