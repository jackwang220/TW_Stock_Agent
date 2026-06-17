"""retest: rebound_offregime — 反彈引擎 off-regime 是否「主動倒賠」還是 gating 問題。

被否決結論:純反彈策略 held-out 2021~2024H1、甚至 2021 多頭都 -16.7% → 判定
「regime 賭注、離開對的 regime 會倒賠」。

本實驗:把「反彈引擎何時開」的 gate 抽成可掃描變數,在固定反彈訊號 + 固定 ⑤執行引擎
(收盤買+開盤賣)下,掃 gate 定義,逐 regime 看:
  (a) always-on(無 gate)是否重現 off-regime 倒賠(特別是 2021 多頭)?
  (b) 加上 gate(只在大盤弱時才放反彈訊號)後,多頭 regime 是否變成「空手/低曝險」
      而非「主動倒賠」? alpha 是否改善?
  (c) 報告每個 regime 的 avg_expo,區分「曝險低→少賺」 vs 「曝險高→真倒賠」。

固定:反彈訊號=rebound_signal(原參數)、引擎=sim_buyclose_sellopen、universe=base_universe。
gate 只擋「反彈訊號要不要在當天放出來」(bull 日不打反彈)。
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
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

v5 = _load("v5", "scripts/exp_step1_v5.py")
v6 = _load("v6", "scripts/exp_step1_v6.py")
ec = _load("ec", "scripts/exp_60d_entry_compare.py")
sim5 = ec.sim_buyclose_sellopen
bench_0050 = v6.bench_0050
features, _factors = v5.features, v5._factors

START, END = "2021-01-01", "2026-06-08"
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]

# gate:給定當天大盤特徵,回傳「今天可以打反彈嗎」(True=放反彈訊號)
def gate_allows(tf, mode):
    c, m20, m5, m60, r20 = tf.get("close"), tf.get("ma20"), tf.get("ma5"), tf.get("ma60"), tf.get("ret20")
    if mode == "always":          # 無 gate(=被否決的純反彈)
        return True
    if not c:
        return False
    def ok(x): return x is not None and not (isinstance(x, float) and math.isnan(x))
    if mode == "bear_c<ma20":     # 大盤跌破月線才開反彈
        return ok(m20) and c < m20
    if mode == "bear_c<ma60":     # 跌破季線才開
        return ok(m60) and c < m60
    if mode == "bear_ma20<ma60":  # 死亡交叉(月線<季線)才開
        return ok(m20) and ok(m60) and m20 < m60
    if mode == "bear_ret20<0":    # 月動能轉負才開
        return ok(r20) and r20 < 0
    if mode == "bear_c<ma20_or_ret20<0":
        return (ok(m20) and c < m20) or (ok(r20) and r20 < 0)
    return True

GATES = ["always", "bear_c<ma20", "bear_c<ma60", "bear_ma20<ma60", "bear_ret20<0", "bear_c<ma20_or_ret20<0"]

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入全史({len(codes)}支)...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    v5._OH = OH
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in list(codes)+["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in list(codes)+["0050"]}

    logger.info("反彈/漲停...")
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

    # gate 開啟天數佔比(整體)
    gate_days = {g: 0 for g in GATES}
    for d in alld:
        tf = twii_feat.get(d, {})
        for g in GATES:
            if gate_allows(tf, g): gate_days[g] += 1

    def rows_for(gmode):
        out = []
        for d in alld:
            if not gate_allows(twii_feat.get(d, {}), gmode):
                continue
            for c in codes:
                sc = reb_cache.get(c, {}).get(d, 0.0) * 1.5   # 與 v9 反彈腿同尺度
                if sc > 0: out.append((d, c, sc/100))
        return out

    bench = {lab: bench_0050(opens["0050"], closes["0050"], [d for d in alld if s <= d <= e]) for lab, s, e in REGIMES}
    res = {}
    for g in GATES:
        rows = rows_for(g)
        for lab, s, e in REGIMES:
            res[(g, lab)] = sim5([r for r in rows if s <= r[0] <= e], opens, closes, limitup)
        logger.info(f"gate={g} 完成(gate開啟日佔比 {gate_days[g]/len(alld)*100:.0f}%)")

    def cell(g, lab):
        r = res.get((g, lab)); return r

    L = ["# retest: rebound_offregime — 反彈引擎 gating 掃描\n",
         "> 固定反彈訊號 + ⑤執行(收盤買/開盤賣)、還原價、真實成本。只改『何時放反彈訊號』(gate)。\n",
         "> always = 被否決的純反彈(無 gate);bear_* = 只在大盤弱時才打反彈。\n",
         f"> 0050各regime:" + " ".join(f"{lab}{bench[lab]:+.0f}%" for lab,_,_ in REGIMES) + "\n",
         "## ALPHA %(策略−0050)\n",
         "| gate | 開啟日% | 2021復甦 | 2022空頭 | 2023復甦 | 2024-25多頭 | 2025下-26 | 最差 |",
         "|---|---|---|---|---|---|---|---|"]
    for g in GATES:
        vals = []
        for lab,_,_ in REGIMES:
            r = cell(g, lab)
            vals.append((r["ret"]-bench[lab]) if r else None)
        valid = [v for v in vals if v is not None]
        c = [f"{v:+.0f}" if v is not None else "—" for v in vals]
        worst = f"{min(valid):+.0f}" if valid else "—"
        L.append(f"| {g} | {gate_days[g]/len(alld)*100:.0f}% | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | **{worst}** |")

    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| gate | 2021復甦 | 2022空頭 | 2023復甦 | 2024-25多頭 | 2025下-26 |", "|---|---|---|---|---|---|"]
    for g in GATES:
        cc = [f"{cell(g,lab)['ret']:+.0f}" if cell(g,lab) else "—" for lab,_,_ in REGIMES]
        L.append(f"| {g} | {cc[0]} | {cc[1]} | {cc[2]} | {cc[3]} | {cc[4]} |")

    L += ["", "## 平均曝險 %(區分『低曝險少賺』vs『高曝險真倒賠』)\n",
          "| gate | 2021復甦 | 2022空頭 | 2023復甦 | 2024-25多頭 | 2025下-26 |", "|---|---|---|---|---|---|"]
    for g in GATES:
        cc = [f"{cell(g,lab)['avg_expo']*100:.0f}" if cell(g,lab) else "—" for lab,_,_ in REGIMES]
        L.append(f"| {g} | {cc[0]} | {cc[1]} | {cc[2]} | {cc[3]} | {cc[4]} |")

    (ROOT/"reports"/"retest_rebound_offregime.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/retest_rebound_offregime.md")
    # console 摘要
    print("\n".join(L))

if __name__ == "__main__":
    main()
