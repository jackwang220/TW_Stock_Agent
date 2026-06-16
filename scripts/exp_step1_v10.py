"""Step1 v10 整合:v6 的 6 個 END 往回窗口 + v7~v9 的 5 個真實 regime,同一張表。
策略 = v6 的 16 變體(A~L+組合+反彈only) + v9 的 H 變體(H雙引擎A/B/C、H純)。
引擎統一 = ⑤ 收盤買 + 開盤賣 + 開盤補買(sim_buyclose_sellopen),真實成交。
成本 = 買0.1425%/賣0.4425% + 滑價0.1% + 漲停買不到;全史 2021~ 還原價(get_daily_ohlcv 除息還原)。
ALPHA = 策略報酬% − 同資金 DCA 進 0050 報酬%(每欄各自用該欄日期區間的 0050 基準)。
兩個 universe:112檔 全 base_universe;20檔 = avg_turnover 成交值前20大。
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
sim5 = ec.sim_buyclose_sellopen          # ⑤:收盤買 + 開盤賣 + 開盤補買
bench_0050 = v6.bench_0050               # 同資金 DCA 進 0050(隔日開盤、滑價、持有到底)

START = "2021-01-01"
END = "2026-06-08"
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
# v9 committed 的雙引擎權重:(多頭 H, 多頭 reb, 空頭 H, 空頭 reb)
DUAL = {"H雙引擎A": (1.0,0.4,0.2,1.4), "H雙引擎B純切": (1.0,0.0,0.0,1.5), "H雙引擎C": (1.0,0.6,0.3,1.3)}


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes_all = list(u.keys())
    names = {c: u[c].get("name", c) for c in codes_all}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes_all}

    logger.info("載入全史還原 OHLCV 2021~ ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes_all}
    OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH       # 讓 v3.features/oh 共用全史(同 v7 作法)

    logger.info("特徵 ...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes_all}
    alld = sorted({d for c in codes_all for d in OH[c]})
    allset = set(alld)
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes_all + ["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes_all + ["0050"]}

    logger.info("反彈/漲停 ...")
    reb_cache, limitup = {}, {}
    for c in codes_all:
        ds = sorted(OH[c]); cl = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl.append(OH[c][d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
            if j > 0 and OH[c][ds[j-1]]["close"] > 0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s

    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    # ── 欄位(6 窗口 + 5 regime)各自的日期集合 + 0050 同資金 DCA 基準 ──
    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    ORDER = list(v5.VAR) + ["反彈only"] + list(DUAL) + ["H純"]

    def turn_pct_for(codes, dates):
        tp = {}
        for d in dates:
            vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                           if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
            tp[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
        return tp

    def rows_for(vn, codes, dates, tp):
        """指定 universe + 日期 + turn_pct,產生該策略的 (d, ticker, edge) rows。"""
        if vn == "反彈only":
            return [(d, c, reb_cache[c][d]/100) for d in dates for c in codes if d in reb_cache[c]]
        if vn in v5.VAR:
            return v5.build_rows(codes, names, feats, twii_feat, reb_cache, tp, v5.VAR[vn], dates)
        out = []
        for d in dates:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                hh = h_score(_factors(f[d], ir), tp.get(d, {}).get(c, 0.5))
                if vn == "H純":
                    sc = hh
                else:
                    whb, wrb, whs, wrs = DUAL[vn]; rb = reb_cache.get(c, {}).get(d, 0.0)
                    sc = max(hh*whb, rb*wrb) if bull else max(hh*whs, rb*wrs)
                if sc > 0: out.append((d, c, sc/100))
        return out

    def run_grid(uni_of_col):
        """uni_of_col: 每欄各自的 universe(codes list)。turn_pct 在該欄 universe 內計算。"""
        res = {}
        for col in COLS:
            codes = uni_of_col[col]; dates = sorted(col_dates[col])
            tp = turn_pct_for(codes, dates)
            for vn in ORDER:
                res[(vn, col)] = sim5(rows_for(vn, codes, dates, tp), opens, closes, limitup)
            logger.info(f"  {col} 完成({len(codes)}檔)")
        return ORDER, res

    def top_by_turnover_asof(d0, k):
        """以 d0 為界,取『往前 60 交易日平均成交值』前 k 名(point-in-time,不看未來)。"""
        scored = []
        for c in codes_all:
            f = feats.get(c, {})
            ts = [f[d]["turn"] for d in alld if d <= d0 and d in f and not math.isnan(f[d]["turn"]) and f[d]["turn"] > 0]
            if ts: scored.append((c, sum(ts[-60:]) / len(ts[-60:])))
        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored[:k]]

    def render(title, order, res):
        def a(vn, col):
            r = res.get((vn, col)); return (r["ret"] - bench[col]) if r else None
        def worst_avg(vn):
            vals = [a(vn, col) for col in COLS]; valid = [v for v in vals if v is not None]
            return (min(valid) if valid else None, sum(valid)/len(valid) if valid else None)
        ordered = sorted(order, key=lambda vn: (worst_avg(vn)[0] if worst_avg(vn)[0] is not None else -999), reverse=True)
        head = "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 |"
        sep = "|" + "---|" * (len(COLS) + 3)
        L = [f"## {title} — ALPHA %(策略 − 同資金DCA 0050;排序=11欄最差)\n", head, sep]
        for vn in ordered:
            cells = " | ".join(f"{a(vn,col):+.0f}" if a(vn,col) is not None else "—" for col in COLS)
            w, av = worst_avg(vn)
            L.append(f"| {vn} | {cells} | **{w:+.0f}** | {av:+.0f} |")
        L += ["", f"### {title} — 參考:原始報酬 %(未扣大盤)\n", head.replace(" | 最差 | 平均 ", ""),
              "|" + "---|" * (len(COLS) + 1)]
        for vn in ordered:
            cells = " | ".join(f"{res[(vn,col)]['ret']:+.0f}" if res.get((vn,col)) else "—" for col in COLS)
            L.append(f"| {vn} | {cells} |")
        return L

    logger.info("=== 跑 112 檔 ===")
    o112, r112 = run_grid({col: codes_all for col in COLS})
    logger.info("=== 跑 20 檔(point-in-time:各欄起始日往前60日成交值前20) ===")
    uni20 = {col: top_by_turnover_asof(min(col_dates[col]), 20) for col in COLS}
    for col in COLS:
        logger.info(f"  {col} 當時前20: {', '.join(names[c] for c in uni20[col])}")
    o20, r20 = run_grid(uni20)

    bench_line = "> 0050 各欄基準: " + " ".join(f"{col}{bench[col]:+.0f}%" for col in COLS)
    L = ["# Step1 v10 整合 — ⑤執行 × (6 END窗口 + 5 真實regime) × {112檔, 當時前20檔}\n",
         f"> 結束{END}｜還原價(全史2021~)｜手續費買0.14%/賣0.44%+滑價0.1%+漲停買不到｜資金15000+1000/日上限5萬\n",
         f"> ALPHA = 策略報酬 − 同資金 DCA 進 0050｜策略=v6的16變體 + v9的H雙引擎A/B/C+H純(反彈純≡反彈only已併)\n",
         bench_line + "\n"]
    L += render("112 檔", o112, r112)
    L += [""]
    L += render("20 檔(point-in-time:各欄起始日往前60日成交值前20,無 look-ahead)", o20, r20)
    L += ["", "### 各欄『當時前20』名單(point-in-time)\n"]
    for col in COLS:
        L.append(f"- **{col}**(起{min(col_dates[col])}): " + ", ".join(names[c] for c in uni20[col]))
    REPORT = ROOT / "reports" / "exp_step1_v10.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")


if __name__ == "__main__":
    main()
