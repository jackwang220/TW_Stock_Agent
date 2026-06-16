"""Step1 v14 — 兩個權重實驗(執行=⑤收盤買+開盤只賣 exit_only,對齊實盤):
  實驗A:持股加權(incumbent)比例 — 固定 B純切,變 incumbent ∈ {1.0,1.2,1.5,2.0,3.0}。看 churn↓ 換手↓ 對 alpha 影響。
  實驗B:反彈引擎權重 — 固定 incumbent=1.5,變雙引擎權重(多頭H,多頭reb,空頭H,空頭reb)。
112檔;全史2021~還原價;成本買0.14/賣0.44+滑價0.1+漲停買不到;DCA(15000+1000/日上限5萬)。
ALPHA = 策略 − 同資金DCA進0050;換手 = 看 2年窗口的 traded/contributed(越低越省手續費)。
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
v6 = _load("v6", "scripts/exp_step1_v6.py")
ec = _load("ec", "scripts/exp_60d_entry_compare.py")
features, _factors = v5.features, v5._factors
sim5 = ec.sim_buyclose_sellopen
bench_0050 = v6.bench_0050

START, END = "2021-01-01", "2026-06-08"
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
INCUMBENTS = [1.0, 1.2, 1.5, 2.0, 3.0]
DUALS = {
    "B純切(現行) 1,0,0,1.5":   (1.0, 0.0, 0.0, 1.5),
    "空頭反彈更重 1,0,0,2.0":   (1.0, 0.0, 0.0, 2.0),
    "空頭反彈輕 1,0,0,1.0":     (1.0, 0.0, 0.0, 1.0),
    "多頭混反彈 1,0.3,0,1.5":   (1.0, 0.3, 0.0, 1.5),
    "空頭留點H 1,0,0.3,1.5":    (1.0, 0.0, 0.3, 1.5),
    "H雙C 1,0.6,0.3,1.3":      (1.0, 0.6, 0.3, 1.3),
    "H雙A 1,0.4,0.2,1.4":      (1.0, 0.4, 0.2, 1.4),
}
EXEC = dict(sell_mode="exit_only", open_buy="none")   # 對齊實盤:收盤買 + 開盤只賣


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
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

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

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    def run(rows, inc):
        out = {}
        for col in COLS:
            ds = col_dates[col]
            out[col] = sim5([x for x in rows if x[0] in ds], opens, closes, limitup, incumbent=inc, **EXEC)
        return out

    def table(title, items):
        # items: list of (label, res_dict)
        L = [f"## {title}\n",
             "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) |",
             "|" + "---|" * (len(COLS) + 4)]
        ranked = sorted(items, key=lambda it: min((it[1][c]["ret"]-bench[c]) for c in COLS if it[1].get(c)), reverse=True)
        for lab, res in ranked:
            a = [(res[c]["ret"]-bench[c]) if res.get(c) else None for c in COLS]
            valid = [x for x in a if x is not None]
            cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in a)
            turn = res["2年"]["turn"] if res.get("2年") else 0
            L.append(f"| {lab} | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {turn:.1f}x |")
        return L

    # 實驗A:incumbent(固定 B純切)
    logger.info("=== 實驗A:持股加權 incumbent ===")
    rowsB = rows_dual((1.0, 0.0, 0.0, 1.5))
    itemsA = []
    for inc in INCUMBENTS:
        itemsA.append((f"INC={inc}" + ("(回測值)" if inc == 1.5 else ""), run(rowsB, inc)))
        logger.info(f"  INC={inc} 完成")

    # 實驗B:雙引擎權重(固定 incumbent=1.5)
    logger.info("=== 實驗B:反彈/引擎權重 ===")
    itemsB = []
    for lab, w in DUALS.items():
        itemsB.append((lab, run(rows_dual(w), 1.5)))
        logger.info(f"  {lab} 完成")

    L = ["# Step1 v14 — 持股加權(incumbent)× 反彈引擎權重 實驗\n",
         f"> ⑤收盤買+開盤只賣(exit_only,對齊實盤)｜結束{END}｜還原價｜112檔｜DCA(15000+1000/日上限5萬)\n",
         "> ALPHA=策略−同資金DCA0050;換手=2年窗口 traded/contributed(越低越省手續費)\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n"]
    L += table("實驗A — 持股加權 incumbent(固定 B純切;越大持股越黏、換手越低)", itemsA)
    L += [""]
    L += table("實驗B — 反彈/引擎權重(固定 incumbent=1.5)", itemsB)
    REPORT = ROOT / "reports" / "exp_step1_v14.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")


if __name__ == "__main__":
    main()
