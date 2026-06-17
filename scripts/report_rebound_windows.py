"""反彈引擎多窗損益表 —— 格式照 reports/exp_step1_v15.md。

比較兩個反彈配置(純反彈訊號、leak-safe ≤ 當日、112 檔、還原價、結束 2026-06-08):

  配置A 反彈-被否決版: 純反彈、等權、硬停損 -12%、max_hold=5、無 EDGE_DECAY。
                       (= retest_rebound_offregime_fix.baseline_eqweight)
  配置B 反彈-救援版  : 純反彈走 ⑤(收盤買/隔日開盤賣)、EDGE_DECAY 0.8 自然老化、
                       無硬停損、incumbent×1.5。
                       (= retest_rebound_offregime_fix.sim5_instrumented)

額外參考列(對照用):
  純H            : 只用 H 動能分(全期 always-on), 走 ⑤c(exit_only/open_buy=none)。
  H雙引擎B純切   : bull→H / bear→反彈×1.5 切換(v15 現行實盤訊號), 走 ⑤c。

欄位 = 6 個多窗(60天/90天/半年/1年/1年半/2年, 從結束日往回切)
      + 5 個 regime(2021復甦/2022空頭/2023復甦/2024-25多頭/2025下-26)。
每格 = ALPHA = 策略報酬 − 同資金 DCA 0050(同窗口/regime)。
換手 / 持股數 取「2年」窗口。

防洩漏/扣基準/扣成本/曝險中性: 全部沿用既有引擎(本檔不改既有檔, 只 import 重用)。
  - rebound_signal 只吃 ≤ 當日收盤序列; 進場用隔日開盤或當日收盤, 無未來 bar。
  - alpha = ret − 0050 同期同資金 DCA buy&hold(隔日開盤買, 滑價)。
  - 成本 買0.1425%/賣0.4425%/滑價0.1%; 漲停買不到。
  - 配置A/B 曝險不同(B 有 EXPOSURE_FLOOR/CAP + 老化, A 等權滿格 max3), 解讀 alpha 要扣這層。
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

# 重用既有引擎(不改既有檔)
v5  = _load("v5",  "scripts/exp_step1_v5.py")
v6  = _load("v6",  "scripts/exp_step1_v6.py")
ec  = _load("ec",  "scripts/exp_60d_entry_compare.py")
fix = _load("fix", "scripts/retest_rebound_offregime_fix.py")  # 配置A/B 的引擎在這

features, _factors = v5.features, v5._factors
sim5         = ec.sim_buyclose_sellopen
bench_0050   = v6.bench_0050
baseline_A   = fix.baseline_eqweight        # 配置A: 等權 SL-12/hold5/無老化
sim5_instr_B = fix.sim5_instrumented        # 配置B: ⑤ EDGE_DECAY 無硬停損 incumbent1.5

START, END = "2021-01-01", "2026-06-08"
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入全史還原 OHLCV({len(codes)}支)2021~ ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH

    logger.info("特徵 ...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    opens  = {c: {d: OH[c][d]["open"]  for d in OH[c]} for c in list(codes)+["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in list(codes)+["0050"]}

    logger.info("反彈訊號(score)+觸發日+漲停日 ...")
    reb_cache = {}; fire = {}; limitup = {}
    for c in codes:
        ds = sorted(OH[c]); cl = []; m = {}; s = set(); lu = set()
        for j, d in enumerate(ds):
            cl.append(OH[c][d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100; s.add(d)
                except Exception: pass
            if j > 0 and OH[c][ds[j-1]]["close"] > 0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1 >= 0.095:
                lu.add(d)
        reb_cache[c] = m; fire[c] = s; limitup[c] = lu

    # 成交值百分位(供 h_score 的 turn 維度)
    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    # ── 訊號 rows ──
    # 配置B(純反彈, 走⑤): 反彈分 ×1.5 always-on
    def rows_B(s_date, e_date):
        out = []
        for d in alld:
            if not (s_date <= d <= e_date): continue
            for c in codes:
                sc = reb_cache.get(c, {}).get(d, 0.0)*1.5
                if sc > 0: out.append((d, c, sc/100))
        return out

    # 參考: 純H(全期 H 動能 always-on)
    def rows_H(ds_set):
        out = []
        for d in alld:
            if d not in ds_set: continue
            ir = twii_feat.get(d, {}).get("ret20")
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                if hh > 0: out.append((d, c, hh/100))
        return out

    # 參考: H雙引擎B純切(bull→H / bear→反彈×1.5; v15 現行實盤訊號)
    def rows_dual(ds_set):
        out = []
        for d in alld:
            if d not in ds_set: continue
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                rb = reb_cache.get(c, {}).get(d, 0.0)
                sc = hh if bull else rb*1.5
                if sc > 0: out.append((d, c, sc/100))
        return out

    # ── 欄位日期集合 ──
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(alld[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]

    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    # ── 跑各列 ──
    # 配置A: baseline_eqweight 需要 (OH, codes, fire, s_date, e_date, alld)
    def run_A(col):
        ds = sorted(col_dates[col])
        if len(ds) < 2: return None
        return baseline_A(OH, codes, fire, ds[0], ds[-1], alld)

    # 配置B: sim5_instrumented(rows, opens, closes, limitup) —— ⑤ 反彈救援版
    def run_B(col):
        ds = sorted(col_dates[col])
        if len(ds) < 2: return None
        return sim5_instr_B(rows_B(ds[0], ds[-1]), opens, closes, limitup)

    # 參考列走 ⑤c(收盤買 + 開盤只賣出場, open_buy=none) —— 與 v15 現行實盤一致
    def run_ref(rows_fn, col):
        ds = col_dates[col]
        rows = rows_fn(ds)
        return sim5(rows, opens, closes, limitup, incumbent=1.5,
                    sell_mode="exit_only", open_buy="none")

    ROWS = [
        ("配置A 反彈-被否決(等權/SL-12/hold5)", run_A),
        ("配置B 反彈-救援(⑤/EDGE_DECAY/inc1.5)", run_B),
        ("參考·純H(⑤c)",                       lambda col: run_ref(rows_H, col)),
        ("參考·H雙引擎B純切(⑤c)",              lambda col: run_ref(rows_dual, col)),
    ]

    res = {}
    for lab, fn in ROWS:
        for col in COLS:
            res[(lab, col)] = fn(col)
        logger.info(f"{lab} 完成")

    def a(lab, col):
        r = res.get((lab, col)); return (r["ret"]-bench[col]) if r else None

    # ── 報告 ──
    L = ["# 反彈引擎多窗損益表 — 配置A(被否決) vs 配置B(⑤救援)\n",
         f"> 純反彈訊號(leak-safe ≤當日)｜結束{END}｜還原價｜112檔｜DCA(15000+1000/日上限5萬, 單股加碼≤15000)\n",
         "> 配置A=等權/隔日開盤進場/硬SL-12/max_hold=5/無老化(被否決引擎);"
         "配置B=⑤收盤買+隔日開盤賣/EDGE_DECAY0.8老化/無硬停損/incumbent×1.5\n",
         "> 參考列(⑤c 收盤買+開盤只賣出場): 純H=全期H動能; H雙引擎B純切=bull→H/bear→反彈×1.5(現行實盤訊號)\n",
         "> 成本 買0.1425%/賣0.4425%/滑價0.1%; 漲停買不到。ALPHA=策略−同資金DCA0050; 換手/持股數取2年窗口\n",
         "> ⚠ 曝險中性: A 等權滿格(最多3檔), B 有 EXPOSURE_FLOOR/CAP+老化, 兩者 avg_expo 不同, alpha 不可逐格直比大小\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) | 持股數(2年) |",
         "|" + "---|" * (len(COLS) + 5)]

    for lab, _ in ROWS:
        vals = [a(lab, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res.get((lab, "2年"))
        turn = r2["turn"] if r2 else 0.0
        # 配置A 沒有 avg_pos(等權引擎);用 max_pos 不可得, 標 —
        npos = r2.get("avg_pos") if r2 and ("avg_pos" in r2) else None
        npos_s = f"{npos:.1f}" if npos is not None else "≤3*"
        wst = f"**{min(valid):+.0f}**" if valid else "—"
        avg = f"{sum(valid)/len(valid):+.0f}" if valid else "—"
        L.append(f"| {lab} | {cells} | {wst} | {avg} | {turn:.1f}x | {npos_s} |")

    L += ["", "> *配置A 為等權引擎(無 avg_pos 紀錄), 上限固定 max_pos=3, 故標 ≤3。",
          "",
          "## A vs B 差在哪\n",
          f"- **配置A(被否決)** 硬 SL-12 會在跳空時用更差的開盤穿價砍, 加上 max_hold=5 強制到期出場, "
          f"在多頭把反彈倉位一再砍在低點再追回 → off-regime 倒賠。",
          f"- **配置B(救援)** 改用 EDGE_DECAY 自然老化 + 無硬停損 + incumbent×1.5 黏住持股, "
          f"降低被洗出場的次數, 最差 regime alpha 由 A 的大幅負值改善。",
          "- **救援版在多頭是否仍負 alpha**: 看『2024-25多頭』『2025下-26』兩欄 —— "
          "B 在大多頭相對同資金 DCA 0050 多半仍是負 alpha(純反彈本質是逆勢接刀, 多頭跑輸買大盤), "
          "救援只是把『主動倒賠』收斂成『少賺』, 不是把多頭翻正。",
          "- 解讀 alpha 須扣曝險: B 的 EXPOSURE_FLOOR/CAP 使其平均曝險與 A 不同, 不可逐格直接比大小。"]

    out = ROOT/"reports"/"report_rebound_windows.md"
    out.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {out}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
