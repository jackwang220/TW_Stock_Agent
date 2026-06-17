"""retest part 2: 反彈 off-regime 倒賠的真兇 = 出場機制(SL=-12/max_hold=5)還是訊號?

被否決結論用的是 backtest_realistic 引擎:等權、TP=0、conditional SL=-12(反彈)、max_hold=5,
進場=反彈觸發隔日開盤,出場=碰停損價(含跳空穿價用開盤)或抱滿 5 天到期。
本實驗在『同一份反彈訊號 + 同一個等權/隔日開盤進場』框架下,只掃出場參數:
  - SL: none / -12% / -8%
  - max_hold: 5 / 20 / 60(交易日)
看 2021 多頭(被否決說 -16.7%)是否由 SL/短抱造成,放寬出場後是否不再倒賠。
逐 regime 報酬 vs 0050 buy&hold(同期、同等權無腦持有當基準對照不公平,故報原始報酬+0050)。
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
features = v5.features
bench_0050 = v6.bench_0050

START, END = "2021-01-01", "2026-06-08"
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
FEE = 0.005; SLIP = 0.001; MAX_POS = 3
INIT, DAILY, MAXC = 15000.0, 1000.0, 50000.0

# 出場情境
EXITS = [
    ("SL-12 / hold5", -0.12, 5),     # 被否決原版
    ("SL-8 / hold5",  -0.08, 5),
    ("noSL / hold5",  None,  5),
    ("SL-12 / hold20",-0.12, 20),
    ("noSL / hold20", None,  20),
    ("noSL / hold60", None,  60),
]

def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入全史({len(codes)}支)...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in list(codes)+["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in list(codes)+["0050"]}

    logger.info("反彈觸發日...")
    fire = {}  # code -> set(d) 反彈觸發(隔日開盤進場)
    for c in codes:
        ds = sorted(OH[c]); cl = []; s = set()
        for d in ds:
            cl.append(OH[c][d]["close"])
            if len(cl) >= 25:
                try:
                    if rebound_signal(cl, turns.get(c, 0.0)).get("fired"): s.add(d)
                except Exception: pass
        fire[c] = s

    def sim(s_date, e_date, sl, max_hold):
        cal = [d for d in alld if s_date <= d <= e_date]
        if len(cal) < 2: return None
        cash = INIT; contributed = INIT
        pos = []  # {tk, eidx, eprice, shares}
        for di, d in enumerate(cal):
            add = (min(DAILY, MAXC-contributed) if contributed < MAXC else 0.0) if di>0 else 0.0
            cash += add; contributed += add
            # 出場(用當日 bar)
            for p in list(pos):
                bar = OH[p["tk"]].get(d)
                if not bar: continue
                held = di - p["eidx"]; ep = p["eprice"]; xp = None
                if sl is not None and bar["low"] <= ep*(1+sl):
                    xp = min(ep*(1+sl), bar["open"])
                elif held >= max_hold:
                    xp = bar["close"]
                if xp is not None:
                    xp *= (1-SLIP); fee = FEE*p["shares"]*ep
                    cash += p["shares"]*xp - fee; pos.remove(p)
            # 進場:前一日觸發 → 今日開盤
            if di > 0:
                prev = cal[di-1]; held_tks = {p["tk"] for p in pos}
                eq = cash + sum(p["shares"]*(OH[p["tk"]].get(d,{}).get("close",0)) for p in pos)
                cands = [c for c in codes if prev in fire[c] and c not in held_tks and OH[c].get(d)]
                # 等權:每檔目標 = eq*0.9/MAX_POS
                for c in cands:
                    if len(pos) >= MAX_POS: break
                    bar = OH[c].get(d); op = bar["open"]
                    if not op or op <= 0: continue
                    if di>0 and OH[c].get(prev) and OH[c][prev]["close"]>0 and op/OH[c][prev]["close"]-1>=0.095:
                        continue  # 開盤已漲停買不到
                    budget = min(eq*0.9/MAX_POS, cash)
                    if budget < 1000: continue
                    px = op*(1+SLIP); sh = budget/px
                    cash -= sh*px; pos.append({"tk": c, "eidx": di, "eprice": op, "shares": sh})
            # 估值
        final = cash + sum(p["shares"]*(OH[p["tk"]].get(cal[-1],{}).get("close",0)) for p in pos)
        return (final-contributed)/contributed*100 if contributed else 0.0

    bench = {lab: bench_0050(opens["0050"], closes["0050"], [d for d in alld if s <= d <= e]) for lab, s, e in REGIMES}
    res = {}
    for name, sl, mh in EXITS:
        for lab, s, e in REGIMES:
            res[(name, lab)] = sim(s, e, sl, mh)
        logger.info(f"{name} 完成")

    L = ["# retest part2: 反彈 off-regime 倒賠是出場機制造成的嗎\n",
         "> 同反彈訊號 + 等權隔日開盤進場(MAX_POS=3),只改 SL / max_hold。重點看 2021 多頭。\n",
         f"> 0050各regime:" + " ".join(f"{lab}{bench[lab]:+.0f}%" for lab,_,_ in REGIMES) + "\n",
         "## 原始報酬 %\n",
         "| 出場 | 2021復甦 | 2022空頭 | 2023復甦 | 2024-25多頭 | 2025下-26 |",
         "|---|---|---|---|---|---|"]
    for name, _, _ in EXITS:
        cc = [f"{res[(name,lab)]:+.0f}" if res.get((name,lab)) is not None else "—" for lab,_,_ in REGIMES]
        L.append(f"| {name} | {cc[0]} | {cc[1]} | {cc[2]} | {cc[3]} | {cc[4]} |")
    L += ["", "## ALPHA %(−0050)\n",
          "| 出場 | 2021復甦 | 2022空頭 | 2023復甦 | 2024-25多頭 | 2025下-26 | 最差 |",
          "|---|---|---|---|---|---|---|"]
    for name, _, _ in EXITS:
        vals = [(res[(name,lab)]-bench[lab]) if res.get((name,lab)) is not None else None for lab,_,_ in REGIMES]
        valid=[v for v in vals if v is not None]
        cc = [f"{v:+.0f}" if v is not None else "—" for v in vals]
        L.append(f"| {name} | {cc[0]} | {cc[1]} | {cc[2]} | {cc[3]} | {cc[4]} | **{min(valid):+.0f}** |")
    (ROOT/"reports"/"retest_rebound_exits.md").write_text("\n".join(L), encoding="utf-8")
    logger.success("報告 → reports/retest_rebound_exits.md")
    print("\n".join(L))

if __name__ == "__main__":
    main()
