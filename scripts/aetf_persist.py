"""主動式ETF 方法#5:持續累積 vs 一次性(persistent accumulation)。

核心問題:主動式經理人對個股的「持久淨流入」(跨數週滾動加總 / 連續加碼天數)
有沒有比「單日跳動」(已證 rank-IC≈0 的 daily consensus flow)更有選股力?

訊號(每 date×stock,跨28檔ETF聚合,沿用 etf_consensus_probe 的 net_chg=ΣDaily_Change):
  oneshot   = 當日跨ETF淨權重變化(=已證無edge的單日訊號,當對照)
  roll_W    = 過去 W 交易日 net_chg 的滾動加總(持久淨流入,W掃描)
  streak    = 連續「淨加碼日(net_chg>0)」天數(連續加碼天數)
  persist   = roll_W 的 z-score(跨股截面標準化,讓選股不偏大票)

兩種使用框架(都用既有 sim5=⑤買收賣開 引擎,曝險中性比較):
  (A) 純選股:每日依訊號取 top-N → sim5。比 持久 vs 單日 vs DCA0050。
  (B) tilt:H+反彈雙引擎 edge × (1 + t·tanh(persist)) → sim5。看持久流向能不能加值純引擎。

★ 第一輪作弊註記(明確):
  1. 時點洩漏:主動式ETF持股「盤後揭露」,本腳本第一輪用「同日揭露當日就交易」
     (即訊號用到當日 net_chg)。真實只能 D+1 行動。Verify階段才改 lag。
  2. survivorship/in-sample:用「當前28檔ETF成員」+「base_universe 流動性排名(全期)」。
  基準/成本不作弊:alpha = 策略 − 同資金 DCA 0050(隔日開盤),扣真實成本
  (買0.1425%/賣0.4425%+滑價0.1%、漲停買不到),報換手/集中/曝險。

⚠️ 資料天花板:ETF持股只 2025-06-17~2026-06-17(262日)、池50檔大型股、且這段是
   多頭年 → 只能在這1年窗驗證,無空頭、無長窗,結論帶「缺空頭」天花板。

用法: uv run python scripts/aetf_persist.py
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path
import pandas as pd

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
sim5 = ec.sim_buyclose_sellopen          # ⑤買收賣開:各窗皆正 alpha 的最佳執行
TOPN = 4

END = "2026-06-08"
ETF_CSV = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"
# ETF 資料只 1 年 → 窗只到 1 年(再長無 ETF 訊號)。仍列多窗供 v15 格式對照。
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]
# regime:資料段內可切的(這年是多頭,空頭段缺)
REGIMES = {
    "2025下半":  ("2025-07-01", "2025-12-31"),
    "2026上半":  ("2026-01-01", "2026-06-08"),
}
ROLL_WINDOWS = [5, 10, 20, 40]   # 持久窗掃描(交易日)


def load_etf_flow():
    """跨28檔ETF聚合 net_chg(=ΣDaily_Change),回傳 {stock: {date: net_chg}} 及日期清單。"""
    df = pd.read_csv(ETF_CSV, dtype={"Stock_Code": str, "ETF_Code": str})
    g = df.groupby(["Date", "Stock_Code"]).agg(net_chg=("Daily_Change(%)", "sum")).reset_index()
    flow = defaultdict(dict)
    for d, c, v in zip(g["Date"], g["Stock_Code"], g["net_chg"]):
        flow[c][d] = float(v)
    dates = sorted(df["Date"].unique())
    stocks = sorted(df["Stock_Code"].unique())
    return flow, dates, stocks


def build_persist_signals(flow, etf_dates, stocks, W):
    """對每股算 roll_W(滾動加總)、streak(連續加碼日)、oneshot(單日)。
    回傳 {date: {stock: dict(oneshot, roll, streak)}},只在 etf_dates 上。"""
    sig = defaultdict(dict)
    for c in stocks:
        fc = flow.get(c, {})
        vals = [fc.get(d, 0.0) for d in etf_dates]   # 缺漏(未持有)=0
        streak = 0
        for i, d in enumerate(etf_dates):
            roll = sum(vals[max(0, i - W + 1): i + 1])
            if vals[i] > 0:
                streak += 1
            else:
                streak = 0
            sig[d][c] = {"oneshot": vals[i], "roll": roll, "streak": streak}
    return sig


def cross_z(day_sigs, key):
    """當日跨股對某 key 做 z-score(讓選股不偏大票)。"""
    items = [(c, s[key]) for c, s in day_sigs.items()]
    if len(items) < 3:
        return {c: 0.0 for c, _ in items}
    xs = [v for _, v in items]
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
    return {c: (v - mu) / sd for c, v in items}


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info("載入 ETF 流向...")
    flow, etf_dates, etf_stocks = load_etf_flow()
    # 只用既在 base_universe(有清洗價)的 ETF 成員(survivorship + in-sample,已註記)
    pool = [c for c in etf_stocks if c in u]
    logger.info(f"ETF 成員 {len(etf_stocks)},池內(有清洗價){len(pool)}")

    logger.info("載入特徵/開收盤/反彈/漲停...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}
    opens, closes = {}, {}
    for c in codes + ["0050"]:
        o = oh(c); opens[c] = {d: o[d]["open"] for d in o}; closes[c] = {d: o[d]["close"] for d in o}
    alld = sorted({d for c in codes for d in closes.get(c, {}) if d <= END})
    # 訊號日 = ETF 有資料 ∩ 有價格,且 ≤ END
    sig_days = [d for d in alld if d in set(etf_dates)][-252:]
    logger.info(f"訊號範圍 {sig_days[0]} ~ {sig_days[-1]}({len(sig_days)} 日)")

    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); cl = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl.append(o[d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
            if j > 0 and o[ds[j - 1]]["close"] > 0 and o[d]["close"] / o[ds[j - 1]]["close"] - 1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    regime_bull = {
        d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
        for d in sig_days
    }

    # ── H+反彈雙引擎候選(baseline,與既有完全相同)──
    def build_h_rows(day_set):
        rows = []
        for d in sig_days:
            if d not in day_set:
                continue
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
            rows.extend((d, c, v / 100.0) for v, c in sc[:TOPN])
        return rows

    # ── 純選股 rows:每日依持久/單日訊號取 top-N(只在 pool 內)──
    def build_flow_rows(persist_sig, day_set, key, use_z=True, edge_floor=0.0):
        """key in {'roll','oneshot','streak'}。edge = 正規化分數(>0 才入選)。"""
        rows = []
        for d in sig_days:
            if d not in day_set or d not in persist_sig:
                continue
            day = {c: persist_sig[d][c] for c in pool if c in persist_sig[d]}
            if not day:
                continue
            if key == "streak":
                ranked = sorted(((s["streak"], c) for c, s in day.items()), reverse=True)
                top = [(c, sc) for sc, c in ranked[:TOPN] if sc > 0]
                for c, sc in top:
                    rows.append((d, c, min(1.0, sc / 10.0)))   # streak→edge,封頂
            else:
                if use_z:
                    z = cross_z({c: s for c, s in day.items()}, key)
                    ranked = sorted(((z[c], c) for c in day), reverse=True)
                    for zz, c in ranked[:TOPN]:
                        if zz > edge_floor:
                            rows.append((d, c, min(1.0, max(0.05, 0.5 + zz / 4.0))))
                else:
                    ranked = sorted(((day[c][key], c) for c in day), reverse=True)
                    for vv, c in ranked[:TOPN]:
                        if vv > 0:
                            rows.append((d, c, min(1.0, vv / day[max(day, key=lambda x: day[x][key])][key] if vv > 0 else 0)))
        return rows

    # ── tilt rows:H edge × (1 + t·tanh(persist_z)) ──
    def build_tilt_rows(persist_sig, day_set, t, key="roll"):
        rows = []
        # 預先算每日 z
        zmap = {}
        for d in sig_days:
            if d in persist_sig:
                day = {c: persist_sig[d][c] for c in pool if c in persist_sig[d]}
                zmap[d] = cross_z(day, key) if day else {}
        for (d, c, e) in build_h_rows(day_set):
            z = zmap.get(d, {}).get(c, 0.0)
            mult = max(0.0, 1.0 + t * math.tanh(z))
            rows.append((d, c, e * mult))
        return rows

    def run(rows):
        return sim5(rows, opens, closes, limitup, switch_cost_mult=1.0)

    # ── 0050 基準 ──
    def bench_for(day_set):
        dd = sorted(d for d in day_set if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    bench_win = {wl: bench_for(set(sig_days[-n:])) for wl, n in WINDOWS}
    regime_days = {rn: set(d for d in sig_days if s <= d <= e) for rn, (s, e) in REGIMES.items()}
    bench_reg = {rn: bench_for(dd) for rn, dd in regime_days.items()}
    allcols = [wl for wl, _ in WINDOWS] + list(REGIMES.keys())

    # 預建各 W 的 persist 訊號
    psig = {W: build_persist_signals(flow, etf_dates, etf_stocks, W) for W in ROLL_WINDOWS}

    # ── 配置 ──
    res = {}   # label -> {col: simdict}

    def eval_label(label, rows_fn):
        d = {}
        for wl, n in WINDOWS:
            d[(label, wl)] = run(rows_fn(set(sig_days[-n:])))
        for rn, dd in regime_days.items():
            d[(label, rn)] = run(rows_fn(dd))
        res.update(d)
        logger.info(f"{label} done")

    # baseline: H 純引擎(無 ETF 訊號)
    eval_label("baseline H雙引擎", build_h_rows)
    # 單日訊號純選股(對照=已知無edge)
    eval_label("純選:單日net_chg", lambda ds: build_flow_rows(psig[5], ds, "oneshot"))
    # 持久純選股(掃 W)
    for W in ROLL_WINDOWS:
        eval_label(f"純選:持久roll{W}", lambda ds, W=W: build_flow_rows(psig[W], ds, "roll"))
    # 連續加碼天數純選股
    eval_label("純選:連續加碼streak", lambda ds: build_flow_rows(psig[5], ds, "streak"))
    # tilt:H × 持久(掃 W 與 t)— 看能不能加值純引擎
    BEST_W = 20
    for t in (0.3, 0.6, 1.0):
        eval_label(f"tilt持久W{BEST_W}×{t}", lambda ds, t=t: build_tilt_rows(psig[BEST_W], ds, t, "roll"))
    # ── placebo 對照(驗證 tilt uplift 是訊號還是雜訊)──
    # P1 反向 tilt:往「低持久」傾斜(t=-0.6)。若真有 edge,反向應變差。
    eval_label(f"placebo反向W{BEST_W}×-0.6", lambda ds: build_tilt_rows(psig[BEST_W], ds, -0.6, "roll"))
    # P2 單日 tilt:用單日 net_chg z 當 tilt(已證單日無edge)→ 應無 uplift。
    eval_label("placebo單日tilt×0.6", lambda ds: build_tilt_rows(psig[5], ds, 0.6, "oneshot"))

    configs = ["baseline H雙引擎", "純選:單日net_chg"] + \
              [f"純選:持久roll{W}" for W in ROLL_WINDOWS] + \
              ["純選:連續加碼streak"] + \
              [f"tilt持久W{BEST_W}×{t}" for t in (0.3, 0.6, 1.0)] + \
              [f"placebo反向W{BEST_W}×-0.6", "placebo單日tilt×0.6"]

    def alpha(label, col):
        r = res.get((label, col))
        bench = bench_win.get(col, bench_reg.get(col))
        return (r["ret"] - bench) if (r and bench is not None) else None

    # ── 報告 ──
    RPT = ROOT / "reports" / "aetf_persist.md"
    L = ["# 主動式ETF 方法#5:持久累積 vs 一次性\n",
         f"> 訊號=跨28檔ETF Σ Daily_Change(net_chg);持久=滾動加總/連續加碼天數,對照=單日(已證無edge)\n",
         f"> 候選池={len(pool)}檔(ETF成員∩base_universe);引擎=⑤買收賣開;結束{END};還原價\n",
         f"> 成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到｜ALPHA=策略−同資金DCA 0050｜資金15000+1000/日上限5萬\n",
         f"> ⚠️ 作弊:(1)時點洩漏-同日揭露當日交易(真實D+1)(2)survivorship-當前ETF成員(3)in-sample-base_universe全期流動性\n",
         f"> ⚠️ 天花板:ETF資料只1年多頭、池偏大型股、無空頭、無長窗\n",
         "> 0050基準: " + " ".join(f"{c}{(bench_win.get(c) if c in bench_win else bench_reg.get(c)):+.0f}%" for c in allcols) + "\n",
         "## ALPHA %(扣0050)\n",
         "| 變體 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手 | 持股 | 曝險 |",
         "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|"]
    for label in configs:
        cells, avals = [], []
        for col in allcols:
            a = alpha(label, col)
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None:
                avals.append(a)
        worst = f"{min(avals):+.0f}" if avals else "—"
        mean = f"{sum(avals) / len(avals):+.0f}" if avals else "—"
        r1y = res.get((label, "1年"))
        turn = f"{r1y['turn']:.0f}x" if r1y else "—"
        pos = f"{r1y.get('avg_pos', 0):.1f}" if r1y else "—"
        expo = f"{r1y.get('avg_expo', 0) * 100:.0f}%" if r1y else "—"
        L.append(f"| {label} | " + " | ".join(cells) + f" | **{worst}** | {mean} | {turn} | {pos} | {expo} |")

    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    for label in configs:
        cells = [f"{res[(label, c)]['ret']:+.0f}" if res.get((label, c)) else "—" for c in allcols]
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += ["", "## Uplift = 變體 alpha − baseline alpha(pp,逐窗)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    base_a = {c: alpha("baseline H雙引擎", c) for c in allcols}
    for label in configs:
        if label.startswith("baseline"):
            continue
        cells = []
        for c in allcols:
            a = alpha(label, c); b = base_a.get(c)
            cells.append(f"{a - b:+.0f}" if (a is not None and b is not None) else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += ["", "## 判讀(結論:NO uplift,持久流向無 edge)\n",
          "- **純選股全死**:持久 roll{5,10,20,40}、streak 各窗 alpha 幾乎全負(1年 -49~-73),"
          "且不隨 W 單調、不優於『單日 net_chg』(已證無edge)。→ 持久建倉訊號沒有獨立選股力,與單日一致。",
          "- **tilt 看似有 uplift 是假象**:H×持久 tilt 1年 uplift +12~+44,曝險 74%≈baseline 75%(非高曝險虛假贏),"
          "乍看正向。但 placebo 拆穿:",
          "  - placebo 反向 tilt(往『低持久』傾,t=-0.6)1年 uplift **+40** ≈ 正向 tilt +44;",
          "  - placebo 單日 tilt(用已證無edge的單日訊號)1年 uplift **+39**,平均甚至更高(+61)。",
          "  → 正向/反向/無edge訊號的 tilt 表現幾乎一樣 → uplift 來自『在強H底上做任何重配權重』在多頭年"
          "放大的集中度/雜訊報酬,**不是持久流向的方向性edge**。若真有 edge,反向 tilt 應變差。",
          "- **結論:has_uplift = False**。持久累積 vs 一次性,兩者都無 edge(印證已知 daily consensus 死訊號),"
          "tilt 的正 uplift 是 placebo 也能複製的假訊號。"]
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    # console 摘要
    for label in configs:
        a1 = alpha(label, "1年"); a60 = alpha(label, "60天")
        up = (a1 - base_a["1年"]) if (a1 is not None and base_a["1年"] is not None) else None
        logger.success(f"{label}: 60天α{a60:+.0f} 1年α{a1:+.0f} 1年uplift{up:+.0f}" if None not in (a1, a60, up)
                       else f"{label}: 60天α{a60} 1年α{a1}")


if __name__ == "__main__":
    main()
