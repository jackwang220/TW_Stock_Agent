"""Step1 v13 — 固定 B純切策略,比較『開盤腿執行法』:收盤買 + 開盤(只賣 / 視情況買賣)。
開盤買多組對比(都用真實日線開盤價成交):
  開盤只賣            : 只賣掉出名單(=目前實盤行為)
  開盤買-全部         : 賣 + 把沒買夠的全部開盤補買
  開盤買-僅漲停沒買到 : 賣 + 只補「收盤鎖漲停沒買到」的
  開盤買-僅開低        : 賣 + 只在開盤≤昨收(開低/開平)時補買
  開盤買-跳空>3%不追   : 賣 + 開盤跳空向上>3%就不追
⚠️ 資料=日線;開盤價=當天官方開盤價(09:00集合競價)。看不到「開盤後幾分鐘」盤中走勢(無分鐘資料)。
引擎=⑤;成本買0.14/賣0.44+滑價0.1+漲停買不到;全史2021~還原價;112檔;資金DCA(15000+1000/日上限5萬)。
ALPHA = 策略 − 同資金DCA進0050(每欄各自區間)。
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
WEIGHTS = (1.0, 0.0, 0.0, 1.5)   # B純切:多頭純H、空頭純反彈
# 開盤腿執行法(label, sell_mode, open_buy)
EXEC = [
    ("開盤只賣(現行實盤)", "exit_only", "none"),
    ("開盤買-全部",         "exit_only", "full"),
    ("開盤買-僅漲停沒買到", "exit_only", "limitup"),
    ("開盤買-僅開低",       "exit_only", "dip"),
    ("開盤買-跳空>3%不追",  "exit_only", "gapcap"),
]


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

    # B純切 rows(全史)
    whb, wrb, whs, wrs = WEIGHTS
    rows = []
    for d in alld:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            rb = reb_cache.get(c, {}).get(d, 0.0)
            sc = max(hh*whb, rb*wrb) if bull else max(hh*whs, rb*wrs)
            if sc > 0: rows.append((d, c, sc/100))

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    res, stats = {}, {}
    for label, sm, ob in EXEC:
        nbuy_open = nblock = 0
        for col in COLS:
            ds = col_dates[col]
            r = sim5([x for x in rows if x[0] in ds], opens, closes, limitup, sell_mode=sm, open_buy=ob)
            res[(label, col)] = r
            if r: nbuy_open += r.get("buy_open", 0); nblock += r.get("blocked", 0)
        stats[label] = (nbuy_open, nblock)
        logger.info(f"{label} 完成(開盤買單數合計≈{nbuy_open})")

    def a(label, col):
        r = res.get((label, col)); return (r["ret"] - bench[col]) if r else None

    L = ["# Step1 v13 — B純切:開盤腿執行法對比(收盤買 + 開盤只賣 / 視情況買賣)\n",
         f"> ⑤執行｜結束{END}｜還原價(全史2021~)｜買0.14/賣0.44+滑價0.1+漲停買不到｜112檔｜DCA(15000+1000/日上限5萬)\n",
         "> ⚠️ 日線回測:開盤價=當天官方開盤價(09:00集合競價);**無分鐘資料,看不到開盤後幾分鐘走勢**\n",
         "> ALPHA = B純切 − 同資金DCA進0050｜開盤買條件用『開盤 vs 昨收跳空』判斷\n",
         "> 0050 各欄基準: " + " ".join(f"{col}{bench[col]:+.0f}%" for col in COLS) + "\n",
         "## ALPHA %(排序=11欄最差;比較不同開盤腿做法)\n",
         "| 開盤腿做法 | " + " | ".join(COLS) + " | 最差 | 平均 |",
         "|" + "---|" * (len(COLS) + 3)]
    rank = sorted([e[0] for e in EXEC],
                  key=lambda lb: min((a(lb, c) for c in COLS if a(lb, c) is not None), default=-999), reverse=True)
    for lb in rank:
        vals = [a(lb, c) for c in COLS]; valid = [v for v in vals if v is not None]
        cells = " | ".join(f"{v:+.0f}" if v is not None else "—" for v in vals)
        w = f"{min(valid):+.0f}" if valid else "—"; av = f"{sum(valid)/len(valid):+.0f}" if valid else "—"
        L.append(f"| {lb} | {cells} | **{w}** | {av} |")
    L += ["", "### 各做法的開盤補買筆數(合計;看條件嚴不嚴)\n", "| 開盤腿做法 | 開盤補買筆數 | 漲停擋下 |", "|---|---|---|"]
    for lb in [e[0] for e in EXEC]:
        nb, blk = stats[lb]; L.append(f"| {lb} | {nb} | {blk} |")
    REPORT = ROOT / "reports" / "exp_step1_v13.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")


if __name__ == "__main__":
    main()
