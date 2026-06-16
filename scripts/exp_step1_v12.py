"""Step1 v12 — 每季獨立(期初一次投5萬,每季重跑)報告 + 兩架構對決:
  版本1(單一策略,不切換):純 A~H + 反彈only + H雙引擎A/B/C,每季都用同一支。
  版本2(0050 頂層換策略,每日判斷):多頭→打 H,空頭→直接換成另一支(反彈/G/C/空手)。
                                    多空開關用 0050 的 MA20>MA60(v11 最佳判斷)。
比較:純 H+ABC 這類單一策略 vs 0050 判斷後直接切策略,哪個好。
引擎=⑤收盤買+開盤賣買;成本買0.14/賣0.44+滑價0.1+漲停買不到;全史2021~還原價;112檔。
資本=每季期初一次 50000,季末估值,當季報酬;ALPHA=策略當季 − 0050 當季(lump 5萬買收盤持有)。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

v5 = _load("v5", "scripts/exp_step1_v5.py")
ec = _load("ec", "scripts/exp_60d_entry_compare.py")
features, _factors = v5.features, v5._factors
sim5 = ec.sim_buyclose_sellopen
SLIP = ec.SLIP
# ── 資本模型:每季期初一次投 5 萬,無每日加碼(每季 sim 各自從 5 萬起跑) ──
ec.r60.INITIAL_CAPITAL = 50000.0
ec.r60.DAILY_BUDGET = 0.0
ec.r60.MAX_CONTRIBUTION = 50000.0

START, END = "2021-01-01", "2026-06-08"
PURE = ["A", "B", "C", "D", "E", "F", "G", "H"]          # v5.VAR 的純單因子
DUAL = {"H雙引擎A": (1.0,0.4,0.2,1.4), "H雙引擎B純切": (1.0,0.0,0.0,1.5), "H雙引擎C": (1.0,0.6,0.3,1.3)}
# 版本2:每日判斷,多頭一律 H,空頭換成 →
V2_BEAR = {"V2日_H→反彈": "反彈only", "V2日_H→G": "G", "V2日_H→C": "C", "V2日_H→空手": None}
# 版本3:每季初判一次(整季鎖定),多頭整季 H,空頭整季換成 →
V3_BEAR = {"V3季_H→反彈": "反彈only", "V3季_H→G": "G", "V3季_H→C": "C", "V3季_H→空手": None}


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info("載入全史還原 OHLCV 2021~ ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    logger.info("特徵 ...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c]})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes + ["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes + ["0050"]}

    logger.info("反彈/漲停 ...")
    reb_cache, limitup = {}, {}
    for c in codes:
        ds = sorted(OH[c]); cl = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl.append(OH[c][d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
            if j > 0 and OH[c][ds[j-1]]["close"] > 0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1 >= 0.095: s.add(d)
        reb_cache[c] = m; limitup[c] = s

    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    # ── 0050 趨勢:V2 用 MA20>MA60(每日);V1 雙引擎內部仍用原版 c>MA20 ──
    ds0 = sorted(OH["0050"]); c0 = [OH["0050"][d]["close"] for d in ds0]
    def ma(i, k): return sum(c0[i-k+1:i+1])/k if i >= k-1 else float("nan")
    raw_bull = {}
    for i, d in enumerate(ds0):
        m20, m60 = ma(i,20), ma(i,60)
        raw_bull[d] = (not math.isnan(m20)) and (not math.isnan(m60)) and m20 > m60
    bull_v2, last = {}, False
    for d in alld:
        if d in raw_bull: last = raw_bull[d]
        bull_v2[d] = last
    bull_days = {d for d in alld if bull_v2[d]}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    # ── 各「純策略」rows(全史) ──
    base_rows = {}
    for vn in PURE:
        base_rows[vn] = v5.build_rows(codes, names, feats, twii_feat, reb_cache, turn_pct, v5.VAR[vn], alld)
    base_rows["反彈only"] = [(d, c, reb_cache[c][d]/100) for d in alld for c in codes if d in reb_cache[c]]

    def rows_dual(w):
        whb, wrb, whs, wrs = w; out = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                rb = reb_cache.get(c, {}).get(d, 0.0)
                sc = max(hh*whb, rb*wrb) if bull else max(hh*whs, rb*wrs)
                if sc > 0: out.append((d, c, sc/100))
        return out

    # ── 組裝完整策略 rows 字典(V1 單一 + V2 切換) ──
    strat_rows = {}
    for vn in PURE: strat_rows[vn] = base_rows[vn]
    strat_rows["反彈only"] = base_rows["反彈only"]
    for vn, w in DUAL.items(): strat_rows[vn] = rows_dual(w)
    for v2, bear in V2_BEAR.items():
        bull_part = [r for r in base_rows["H"] if r[0] in bull_days]
        bear_part = [r for r in base_rows[bear] if r[0] not in bull_days] if bear else []
        strat_rows[v2] = bull_part + bear_part

    V1 = PURE + ["反彈only"] + list(DUAL)
    V2 = list(V2_BEAR)
    V3 = list(V3_BEAR)
    ALL = V1 + V2 + V3

    # ── 每季(期初一次5萬,每季獨立) ──
    def qkey(d):
        y, mo = int(d[:4]), int(d[5:7]); return (y, (mo-1)//3 + 1)
    quarters = {}
    for d in alld:
        if d < "2021-04-01" or d > END: continue   # 2021Q1 warmup 略過
        quarters.setdefault(qkey(d), []).append(d)
    qlabels = [f"{y}Q{q}" for (y, q) in sorted(quarters)]
    qdates = {f"{y}Q{q}": sorted(quarters[(y, q)]) for (y, q) in sorted(quarters)}

    def q_bench(ds):
        if len(ds) < 2: return 0.0
        f, l = ds[0], ds[-1]; cf = closes["0050"].get(f); cl = closes["0050"].get(l)
        if not cf or cf <= 0: return 0.0
        return (cl/(cf*(1+SLIP)) - 1) * 100      # 期初收盤買 0050、季末收盤估值(含滑價)
    bench_q = {ql: q_bench(qdates[ql]) for ql in qlabels}
    # V3:每季初(第一個交易日)看一次 0050 MA20>MA60,整季鎖定
    q_bull = {ql: bull_v2[qdates[ql][0]] for ql in qlabels}

    def sim_q(rows, ds):
        r = sim5([x for x in rows if x[0] in ds], opens, closes, limitup)
        return r["ret"] if r else 0.0

    # ── 跑:每策略 × 每季 ──
    res = {}   # (strat, qlabel) -> ret%
    for vn in V1 + V2:
        for ql in qlabels:
            res[(vn, ql)] = sim_q(strat_rows[vn], set(qdates[ql]))
        logger.info(f"{vn} 完成")
    for v3, bear in V3_BEAR.items():
        for ql in qlabels:
            chosen = "H" if q_bull[ql] else bear      # 整季鎖定:多頭H / 空頭換策略(None=空手)
            res[(v3, ql)] = sim_q(base_rows[chosen], set(qdates[ql])) if chosen else 0.0
        logger.info(f"{v3} 完成")

    def agg(vn):
        rets = [res[(vn, ql)] for ql in qlabels]
        alphas = [res[(vn, ql)] - bench_q[ql] for ql in qlabels]
        comp = (math.prod(1 + x/100 for x in rets) - 1) * 100
        return {"comp": comp, "avg": sum(rets)/len(rets), "avg_a": sum(alphas)/len(alphas),
                "worst_a": min(alphas), "win": sum(1 for a in alphas if a > 0)/len(alphas)*100}
    bench_comp = (math.prod(1 + bench_q[ql]/100 for ql in qlabels) - 1) * 100
    bench_avg = sum(bench_q.values())/len(bench_q)

    # ── 報告 ──
    nbull = sum(1 for ql in qlabels if q_bull[ql])
    L = ["# Step1 v12 — 每季獨立(期初5萬/季)報告 × 三架構對決:單一 / 0050每日切 / 0050每季切\n",
         f"> ⑤執行｜{qlabels[0]}~{qlabels[-1]}({len(qlabels)}季)｜還原價(全史2021~)｜買0.14/賣0.44+滑價0.1+漲停買不到｜112檔\n",
         "> 每季期初一次投 5 萬、季末結算、各季獨立重跑｜ALPHA=策略當季 − 0050當季(lump 5萬持有)\n",
         f"> 多空開關=0050 的 **MA20>MA60**｜V2日=每日判斷切換, V3季=每季初判一次整季鎖定(判多{nbull}/{len(qlabels)}季)｜V1雙引擎內部用原版 c>MA20\n",
         f"> 0050 本身:全期複合 **{bench_comp:+.0f}%**,平均每季 {bench_avg:+.1f}%\n",
         "## 彙總(排序=全期複合報酬;V1=單一, V2日=每日切, V3季=每季切)\n",
         "| 策略 | 類 | 全期複合% | 平均季報酬% | 平均季alpha | 最差季alpha | 勝率vs0050 |",
         "|---|---|---|---|---|---|---|",
         f"| 0050(基準) | — | {bench_comp:+.0f} | {bench_avg:+.1f} | +0.0 | +0.0 | — |"]
    rank = sorted(ALL, key=lambda vn: agg(vn)["comp"], reverse=True)
    for vn in rank:
        a = agg(vn); cls = "V3季" if vn in V3 else ("V2日" if vn in V2 else "V1")
        L.append(f"| {vn} | {cls} | {a['comp']:+.0f} | {a['avg']:+.1f} | {a['avg_a']:+.1f} | **{a['worst_a']:+.1f}** | {a['win']:.0f}% |")

    # 每季明細(精選:0050 + 代表V1 + 每日切vs每季切 各取反彈/空手對照)
    show = ["H雙引擎C", "H", "V2日_H→反彈", "V3季_H→反彈", "V2日_H→空手", "V3季_H→空手"]
    L += ["", "## 每季報酬 %(精選;0050=當季大盤)\n",
          "| 季 | 多頭日% | 0050 | " + " | ".join(show) + " |",
          "|" + "---|" * (len(show) + 3)]
    for ql in qlabels:
        bp = sum(1 for d in qdates[ql] if bull_v2[d]) / len(qdates[ql]) * 100
        cells = " | ".join(f"{res[(vn,ql)]:+.0f}" for vn in show)
        L.append(f"| {ql} | {bp:.0f}% | {bench_q[ql]:+.0f} | {cells} |")

    def best(group): return max(group, key=lambda vn: agg(vn)["comp"])
    b1, b2, b3 = best(V1), best(V2), best(V3)
    L += ["", "> **對決**(全期複合 / 最差季alpha):"
          f"最佳V1 `{b1}` {agg(b1)['comp']:+.0f}% / {agg(b1)['worst_a']:+.1f}"
          f" ｜ 最佳V2日 `{b2}` {agg(b2)['comp']:+.0f}% / {agg(b2)['worst_a']:+.1f}"
          f" ｜ 最佳V3季 `{b3}` {agg(b3)['comp']:+.0f}% / {agg(b3)['worst_a']:+.1f}"]
    REPORT = ROOT / "reports" / "exp_step1_v12.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")


if __name__ == "__main__":
    main()
