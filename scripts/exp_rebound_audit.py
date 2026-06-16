"""反彈引擎行為稽核:反彈訊號有沒有觸發?觸發後策略有沒有買進+抱住?
B純切(空頭才打反彈)、112檔、⑤收盤買開盤賣(exit_only)、incumbent1.5、DCA。
追蹤每個部位:進場日/原因(空頭=反彈 or 多頭=H)/持有天數/報酬。分類:
  ① 反彈觸發 + 進場 + 抱住(≥HOLD_MIN日)
  ② 反彈觸發 + 進場 + 沒抱住(<HOLD_MIN日)
  ③ 反彈觸發(空頭日) 但 沒進場(被擠掉)
  ④ 反彈沒觸發
也看「抱住 vs 沒抱住」的報酬,判斷反彈到底有沒有被有效執行。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

_v5s = importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py")
v5 = importlib.util.module_from_spec(_v5s); _v5s.loader.exec_module(v5)
_r60s = importlib.util.spec_from_file_location("r60", ROOT/"scripts/run_backtest_60d.py")
r60 = importlib.util.module_from_spec(_r60s); _r60s.loader.exec_module(r60)
features, _factors = v5.features, v5._factors

START, END = "2021-01-01", "2026-06-08"
INC, MAX_SIG, EXPO_CAP, EXPO_FLOOR, TIE, DECAY = 1.5, 3, 0.90, 0.30, 0.90, 0.80
HOLD_MIN = 3   # 抱住門檻(交易日)


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    print("載入全史 ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    twii = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    cl = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes}
    op = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes}

    reb, limitup = {}, {}
    for c in codes:
        ds = sorted(OH[c]); s=[]; m={}; lu=set()
        for j, d in enumerate(ds):
            s.append(OH[c][d]["close"])
            if len(s) >= 25:
                try:
                    g = rebound_signal(s, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
            if j>0 and OH[c][ds[j-1]]["close"]>0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1>=0.095: lu.add(d)
        reb[c]=m; limitup[c]=lu

    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c,{}) and feats[c][d]["turn"]>0), key=lambda x:x[1])
        turn_pct[d] = {c:(i+1)/len(vals) for i,(c,_) in enumerate(vals)} if vals else {}
    bull = {d: bool(twii.get(d,{}).get("close") and twii[d].get("ma20") and twii[d]["close"]>twii[d]["ma20"]) for d in alld}

    # ── ⑤ exit_only + incumbent 追蹤部位 ──
    cash=contrib=0.0; shares={}; entry={}; last_edge={}
    positions=[]   # (code, ent_d, reason, hold_days, pnl_pct)
    bear_fire_days=0; bear_fire_selected=set()  # 空頭觸發但有沒有進場
    def price(c,d):
        cc=cl[c]; ds=[x for x in cc if x<=d]; return cc[max(ds)] if ds else None

    for i,d in enumerate(alld):
        add = r60.INITIAL_CAPITAL if i==0 else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contrib) if contrib<r60.MAX_CONTRIBUTION else 0)
        cash+=add; contrib+=add
        if i+1>=len(alld): break
        e=alld[i+1]
        port = cash + sum(shares[c]*(price(c,d) or 0) for c in shares)
        # edges(B純切)
        edges={}
        for c in codes:
            f=feats.get(c,{})
            if d not in f or math.isnan(f[d].get("ma20",float("nan"))): continue
            if bull[d]:
                sc=h_score(_factors(f[d], twii.get(d,{}).get("ret20")), turn_pct.get(d,{}).get(c,0.5))
            else:
                sc=reb[c].get(d,0.0)*1.5
            if sc>0: edges[c]=sc/100
        # 空頭觸發統計
        if not bull[d]:
            for c in codes:
                if reb[c].get(d,0)>0: bear_fire_days+=1
        for c in shares:
            if c not in edges:
                ed=last_edge.get(c,0)*DECAY; edges[c]=ed
        for c in edges: last_edge[c]=edges[c] if c in edges else last_edge.get(c,0)
        rk=lambda c: edges[c]*(INC if c in shares else 1.0)
        ranked=sorted([c for c in edges if edges[c]>0], key=rk, reverse=True)
        sel=ranked[:MAX_SIG]
        if len(ranked)>MAX_SIG and rk(ranked[MAX_SIG])>=rk(ranked[MAX_SIG-1])*TIE: sel=ranked[:MAX_SIG+1]
        confs=[edges[c]*100 for c in sel]; avg=sum(confs)/len(confs)/100 if confs else 0
        expo=min(EXPO_CAP,max(EXPO_FLOOR,avg)) if sel else 0
        wsum=sum(edges[c] for c in sel); targets={c: port*expo*(edges[c]/wsum) for c in sel} if wsum>0 and expo>0 else {}
        if not bull[d]:
            for c in sel:
                if reb[c].get(d,0)>0: bear_fire_selected.add((c,d))
        # A 收盤買
        for c in sel:
            cp=cl[c].get(d)
            if not cp or cp<=0 or d in limitup[c]: continue
            cur=shares.get(c,0)*cp; tgt=targets.get(c,0); buy=tgt-cur
            if buy < 1000: continue
            if c not in shares or shares[c]<=1e-6:
                entry[c]=(d, "反彈" if not bull[d] else "H")   # 進場原因=當日regime
            shares[c]=shares.get(c,0)+buy/cp
        # B 開盤賣(掉出名單 exit_only)
        keep=set(targets)
        for c in list(shares):
            if c in keep: continue
            o=op[c].get(e)
            if not o or o<=0: continue
            ent_d, reason = entry.get(c,(d,"?"))
            hold=sum(1 for x in alld if ent_d<=x<=d)
            entry_px=cl[c].get(ent_d) or o
            pnl=(o/entry_px-1)*100 if entry_px else 0
            positions.append((c, ent_d, reason, hold, pnl))
            shares.pop(c,None); entry.pop(c,None)
    # 出場剩餘
    for c in list(shares):
        ent_d, reason = entry.get(c,(alld[-1],"?")); hold=sum(1 for x in alld if ent_d<=x<=alld[-1])
        entry_px=cl[c].get(ent_d) or 1; pnl=((price(c,alld[-1]) or entry_px)/entry_px-1)*100
        positions.append((c, ent_d, reason, hold, pnl))

    reb_pos=[p for p in positions if p[2]=="反彈"]; h_pos=[p for p in positions if p[2]=="H"]
    held=[p for p in reb_pos if p[3]>=HOLD_MIN]; nothold=[p for p in reb_pos if p[3]<HOLD_MIN]
    def avg(x,i): return sum(p[i] for p in x)/len(x) if x else float("nan")
    tot_fire=sum(len(m) for m in reb.values())

    print(f"\n===== 反彈引擎稽核(抱住門檻={HOLD_MIN}交易日)=====")
    print(f"反彈訊號觸發總次數(全期,所有股×日): {tot_fire:,}")
    print(f"  其中『空頭日』觸發(反彈引擎實際生效): {bear_fire_days:,} 檔次")
    print(f"  空頭觸發中『有被選進場』的(去重(股,日)): {len(bear_fire_selected):,}")
    print(f"\n進場部位總數 {len(positions)}:反彈進場 {len(reb_pos)}｜H進場 {len(h_pos)}")
    print(f"\n【反彈進場部位】平均持有 {avg(reb_pos,3):.1f} 日,平均報酬 {avg(reb_pos,4):+.1f}%")
    print(f"  ① 觸發+抱住(≥{HOLD_MIN}日): {len(held):>4} 筆,平均持有 {avg(held,3):.1f}日,報酬 {avg(held,4):+.1f}%")
    print(f"  ② 觸發+沒抱住(<{HOLD_MIN}日): {len(nothold):>4} 筆,平均持有 {avg(nothold,3):.1f}日,報酬 {avg(nothold,4):+.1f}%")
    print(f"\n【對照 H進場部位】平均持有 {avg(h_pos,3):.1f} 日,平均報酬 {avg(h_pos,4):+.1f}%")
    print(f"\n抱住比例(反彈): {len(held)/len(reb_pos)*100:.0f}% 抱住 / {len(nothold)/len(reb_pos)*100:.0f}% 沒抱住" if reb_pos else "無反彈進場")


if __name__ == "__main__":
    main()
