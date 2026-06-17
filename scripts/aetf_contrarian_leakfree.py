"""#6 反向擁擠 — 對抗式 leak-free 驗證。

把第一輪 aetf_contrarian.py 的作弊捷徑改嚴格,重跑看 uplift 還剩多少:
  (1) 時點洩漏修正: tilt 用「決策日 d 前一個交易日(d-1)的擁擠度」。
      ETF 持股盤後揭露 → d 收盤決策時,可用的最新已揭露持股是 d-1 的(d-1 盤後揭露)。
      原版用 crowd_pct[d](d 當日,d 收盤時還沒揭露)= 時點洩漏。
  (2) in-sample 選指標: 鎖定 sum_wt(唯一探針 D+1 IC<0 支持的),不採信 max_wt(IC>0,curve-fit)。
  (3) PLACEBO(關鍵): 用「同分布隨機 tilt」(把每日 crowd_pct 在候選間隨機洗牌)當對照。
      若隨機 tilt 的 uplift 與真 contrarian 相近 → uplift 不是訊號,是「擾動 sizing 在多頭年的雜訊/曝險效果」。
  (4) 曝險中性: 報每變體實際 avg_expo + 用 EXPOSURE_CAP 還原是否 tilt 把曝險頂到上限。
  (5) 集中度: 報 avg_pos。

baseline = 純引擎(k=0)。⑤買收賣開、真實成本、ALPHA = 策略 − 同資金 DCA0050。
不改既有檔。用法: uv run python scripts/aetf_contrarian_leakfree.py
"""
from __future__ import annotations
import sys, json, importlib.util, math, random
from collections import defaultdict
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

v5 = _load("v5", ROOT / "scripts/exp_step1_v5.py")
v6 = _load("v6", ROOT / "scripts/exp_step1_v6.py")
ec = _load("ec", ROOT / "scripts/exp_60d_entry_compare.py")
features, _factors, oh = v5.features, v5._factors, v5.oh
h_score = ec.h_score
sim5 = ec.sim_buyclose_sellopen

END = "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]
REGIMES = {"2025下-26": ("2025-07-01", "2026-06-08")}
TOPN = 4
CSV = ROOT / "data/Active_ETF_1Y_Daily_28ETFs.csv"


def build_crowd():
    df = pd.read_csv(CSV, encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Stock_Code"] = df["Stock_Code"].astype(str)
    g = df.groupby(["Date", "Stock_Code"]).agg(
        n_hold=("ETF_Code", "nunique"),
        sum_wt=("Weight(%)", "sum"),
        max_wt=("Weight(%)", "max"),
    ).reset_index()
    crowd = defaultdict(dict)
    for _, r in g.iterrows():
        crowd[r["Date"]][r["Stock_Code"]] = {
            "sum_wt": r["sum_wt"], "n_hold": r["n_hold"], "max_wt": r["max_wt"]}
    return crowd, sorted(df["Stock_Code"].unique()), sorted(crowd.keys())


def pctile_map(crowd, metric, dates, codes):
    out = {}
    for d in dates:
        day = crowd.get(d, {})
        vals = [(c, day[c][metric]) for c in codes if c in day]
        vals.sort(key=lambda x: x[1])
        n = len(vals)
        out[d] = {c: (i + 1) / n for i, (c, _) in enumerate(vals)} if n else {}
    return out


def build_candidates(codes, feats, twii_feat, reb_cache, turn_pct, sig_days):
    regime_bull = {
        d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
        for d in sig_days}
    cands = {}
    for d in sig_days:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        sc = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            v = (h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                 if bull else reb_cache.get(c, {}).get(d, 0.0))
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        cands[d] = sc[:TOPN]
    return cands


def main():
    crowd, etf_codes, crowd_dates = build_crowd()
    # 前一交易日映射(用 crowd 揭露日序列)
    prev_disclosed = {crowd_dates[i]: crowd_dates[i-1] for i in range(1, len(crowd_dates))}
    logger.info(f"擁擠度:{len(crowd)} 日 × ~{len(etf_codes)} 池股")

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info(f"載入特徵({len(codes)} 支)...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    opens, closes = {}, {}
    for c in codes + ["0050"]:
        o = oh(c); opens[c] = {d: o[d]["open"] for d in o}; closes[c] = {d: o[d]["close"] for d in o}
    alld = sorted({d for c in codes for d in closes.get(c, {}) if d <= END})
    sig_days = alld[-max(n for _, n in WINDOWS):]
    logger.info(f"訊號範圍 {sig_days[0]} ~ {sig_days[-1]} ({len(sig_days)} 日)")

    logger.info("反彈訊號 + 漲停日...")
    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); cl_list = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    cands = build_candidates(codes, feats, twii_feat, reb_cache, turn_pct, sig_days)

    # pctile maps
    pm_sum = pctile_map(crowd, "sum_wt", crowd_dates, etf_codes)

    # leak-free 取值: 決策日 d -> 用 d 之前最後一個已揭露日的擁擠度
    def crowd_at(metric_pm, d, c, lag):
        """lag=False(D0, 洩漏): 用 d 當日; lag=True(leak-free): 用 d-1 已揭露。"""
        dd = d if not lag else prev_disclosed.get(d)
        if dd is None:
            return None
        return metric_pm.get(dd, {}).get(c)

    def run_sim(rw):
        return sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)

    def bench_for(day_set):
        dd = sorted(d for d in day_set if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    bench_win = {wl: bench_for(set(sig_days[-n:])) for wl, n in WINDOWS}
    regime_days = {rn: set(d for d in sig_days if s <= d <= e) for rn, (s, e) in REGIMES.items()}
    bench_reg = {rn: bench_for(dd) for rn, dd in regime_days.items()}

    # ── tilt rows builders ───────────────────────────────────────────────────
    def rows_real(k, day_set, lag):
        rw = []
        for d in sig_days:
            if d not in day_set:
                continue
            for v, c in cands[d]:
                e = v / 100.0
                if k == 0.0:
                    rw.append((d, c, e)); continue
                p = crowd_at(pm_sum, d, c, lag)
                if p is None:
                    rw.append((d, c, e)); continue
                rw.append((d, c, e * max(0.0, 1.0 + k * (0.5 - p))))
        return rw

    def rows_placebo(k, day_set, seed):
        """PLACEBO: 每日把候選股的 crowd pctile 隨機洗牌(同分布,破壞 stock↔crowd 對應)。
        若 uplift 仍出現 → 證明 uplift 來自 tilt 擾動 sizing,而非真擁擠訊號。"""
        rng = random.Random(seed)
        rw = []
        for d in sig_days:
            if d not in day_set:
                continue
            cl = cands[d]
            # 取候選股真實 pctile (D+1 leak-free),隨機重新分配給候選
            ps = []
            for v, c in cl:
                p = crowd_at(pm_sum, d, c, True)
                ps.append(p if p is not None else 0.5)
            shuffled = ps[:]; rng.shuffle(shuffled)
            for (v, c), p in zip(cl, shuffled):
                e = v / 100.0
                rw.append((d, c, e * max(0.0, 1.0 + k * (0.5 - p))))
        return rw

    KS = [0.3, 0.6, 1.0]
    cols_w = [wl for wl, _ in WINDOWS]
    cols_r = list(REGIMES.keys())
    allcols = cols_w + cols_r
    daysets = {wl: set(sig_days[-n:]) for wl, n in WINDOWS}
    daysets.update(regime_days)

    def alpha_of(rw_fn, col):
        ds = daysets[col]
        r = run_sim(rw_fn(ds))
        bench = bench_win.get(col, bench_reg.get(col))
        if r is None or bench is None:
            return None, None, None
        return r["ret"] - bench, r.get("avg_expo", 0)*100, r.get("avg_pos", 0)

    # baseline
    base = {}
    base_meta = {}
    for col in allcols:
        a, ex, po = alpha_of(lambda ds: rows_real(0.0, ds, True), col)
        base[col] = a; base_meta[col] = (ex, po)

    results = []  # (label, {col: (alpha, expo, pos)}, {col: uplift})
    def add_variant(label, rw_fn):
        row = {}
        for col in allcols:
            a, ex, po = alpha_of(rw_fn, col)
            row[col] = (a, ex, po)
        results.append((label, row))
        logger.info(f"{label} 完成")

    add_variant("baseline(k0)", lambda ds: rows_real(0.0, ds, True))
    for k in KS:
        add_variant(f"sum_wt D0洩漏×{k}", lambda ds, k=k: rows_real(k, ds, False))
    for k in KS:
        add_variant(f"sum_wt LEAKFREE×{k}", lambda ds, k=k: rows_real(k, ds, True))
    # placebo: 多 seed 取平均
    for k in KS:
        def pl(ds, k=k):
            return rows_placebo(k, ds, 0)
        # 用單一 builder 但跑多 seed 求 alpha 平均
        def alpha_placebo(col, k=k):
            ds = daysets[col]
            bench = bench_win.get(col, bench_reg.get(col))
            aa = []
            for seed in range(8):
                r = run_sim(rows_placebo(k, ds, seed))
                if r and bench is not None:
                    aa.append(r["ret"] - bench)
            return (np.mean(aa) if aa else None), (np.std(aa) if aa else None)
        row = {}
        for col in allcols:
            m_, s_ = alpha_placebo(col)
            row[col] = (m_, s_, None)
        results.append((f"PLACEBO 隨機×{k} (8seed均±std)", row))
        logger.info(f"placebo k={k} 完成")

    # ── report ──
    L = [
        "# #6 反向擁擠 — 對抗式 leak-free 驗證\n",
        f"> baseline=純引擎k0(leak-free)｜⑤買收賣開｜結束 {END}｜{len(codes)} 檔池｜還原含息價(同原腳本快取)\n",
        "> 修正: (1)tilt 用 d-1 已揭露擁擠度(leak-free) (2)鎖 sum_wt(探針唯一支持) (3)PLACEBO 隨機 tilt 對照\n",
        "> 0050 基準: " + " ".join(f"{c}{bench_win[c]:+.0f}%" for c in cols_w)
        + " | " + " ".join(f"{c}{bench_reg[c]:+.0f}%" for c in cols_r) + "\n",
        "## ALPHA %(扣0050) | 曝險% | 持股\n",
        "| 變體 | " + " | ".join(allcols) + " | 平均 | 曝險(1年) | 持股(1年) |",
        "|---|" + "|".join(["---"]*(len(allcols)+3)) + "|",
    ]
    for label, row in results:
        cells = []; av = []
        for col in allcols:
            a = row[col][0]
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None: av.append(a)
        mean = f"{np.mean(av):+.0f}" if av else "—"
        ex, po = row["1年"][1], row["1年"][2]
        exs = f"{ex:.0f}%" if ex is not None else "—"
        pos = f"{po:.1f}" if po is not None else "—"
        L.append(f"| {label} | " + " | ".join(cells) + f" | {mean} | {exs} | {pos} |")

    # uplift vs baseline
    L += ["", "## Uplift = 變體 alpha − baseline alpha (pp)\n",
          "| 變體 | " + " | ".join(allcols) + " | Σ |",
          "|---|" + "|".join(["---"]*(len(allcols)+1)) + "|"]
    for label, row in results:
        if label.startswith("baseline"):
            continue
        cells = []; tot = 0; ok = True
        for col in allcols:
            a = row[col][0]; b = base[col]
            if a is None or b is None:
                cells.append("—"); continue
            cells.append(f"{a-b:+.0f}"); tot += (a-b)
        L.append(f"| {label} | " + " | ".join(cells) + f" | {tot:+.0f} |")

    L += ["", "## 判讀\n",
          "- LEAKFREE vs D0洩漏: 差多少 = 時點洩漏貢獻。",
          "- 關鍵: 真 contrarian LEAKFREE 的 uplift 若 ≤ PLACEBO 隨機 tilt → uplift 不是擁擠訊號,是 sizing 擾動雜訊。",
          "- 曝險欄: tilt 拉高曝險 → 多頭年虛假贏(非中性)。"]
    RPT = ROOT / "reports" / "aetf_contrarian_leakfree.md"
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    print("\n### 摘要 ###")
    for label, row in results:
        cells = " ".join(f"{c}{row[c][0]:+.0f}" if row[c][0] is not None else f"{c}—" for c in allcols)
        up = ""
        if not label.startswith("baseline"):
            ups = [row[c][0]-base[c] for c in allcols if row[c][0] is not None and base[c] is not None]
            up = f" | Σup{sum(ups):+.0f}"
        ex = row["1年"][1]
        print(f"{label:<26} {cells} | expo{ex:.0f}%{up}" if ex is not None else f"{label:<26} {cells}{up}")


if __name__ == "__main__":
    main()
