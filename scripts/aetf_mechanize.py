"""方法 #1：策略原型機械化複製 — 動能/趨勢型主動式ETF的「講白規則」機械化

哲學:主動式ETF的母策略最明確的原型就是「動能/趨勢」(追強勢股、汰弱留強)。
把這套講白規則機械化成純技術選股:每日在「主動式ETF的持股池」內,挑近 L 日動能
最強且站上均線的 top-N,套用與 baseline 完全相同的 ⑤買收賣開 執行引擎回測。

vs baseline:
  • baseline-H(同池)  : H+反彈雙引擎,但宇宙限縮成同一個 ETF 池(公平對照,池一樣)
  • baseline-H(全112)  : 既有 H 雙引擎(全 base_universe),供對照天花板
  • 動能機械化          : 本方法,各種 lookback L / TOPN 變體

═══ 本方法用到的「第一輪小作弊」(明確註記) ═══
  1. [選股池作弊] 用「全期 28 檔主動式ETF 的持股聯集 = 50 檔池」當選股宇宙。
     這 50 檔是用整年(含未來)的 ETF 成員定義 → survivorship + 時點洩漏(用了當前/全期
     的 ETF 成員資格)。Verify 階段要改成「point-in-time:只用 as_of 當天 ETF 揭露成員」。
  2. [in-sample 微調] 掃 lookback/TOPN 找最佳是在同一段資料上選的 → 有過配風險。
  ※ 基準與成本「不」作弊:alpha = 策略 − 同資金 DCA 0050(扣真實成本);⑤引擎漲停買不到;
    曝險用 ⑤ 內生(以 avg conf 定 expo),報 avg_expo 供曝險中性檢查。

資料現實警告(承接 Step0):ETF CSV 的 ETF_Code 標籤對不上真實世界該檔 ETF 的實際
持股、Daily_Change 欄壞掉、美股型 ETF 在 CSV 內 100% 持台股。所以本方法【不】用 CSV 的
ETF 標籤,也【不】用 CSV 的報酬欄;只取「50 檔池」這個事實(=經理人實際在挑的大型股池),
價格一律走 yfinance/FinMind 還原價(與既有引擎同一份)。

用法: uv run python scripts/aetf_mechanize.py
"""
from __future__ import annotations
import sys, json, csv, importlib.util, math
from pathlib import Path

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
bench_0050 = v6.bench_0050

END = "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252), ("1年半", 378), ("2年", 504)]
REGIMES = {
    "2021復甦":    ("2021-06-01", "2021-12-31"),
    "2022空頭":    ("2022-01-01", "2022-12-31"),
    "2023復甦":    ("2023-01-01", "2023-12-31"),
    "2024-25多頭": ("2024-01-01", "2025-06-30"),
    "2025下-26":   ("2025-07-01", "2026-06-08"),
}


def load_etf_pool():
    """[作弊#1] 全期 28 檔主動式ETF 的持股聯集 = 經理人實際在挑的大型股池(50 檔)。"""
    path = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"
    pool = set()
    for r in csv.DictReader(path.open(encoding="utf-8")):
        pool.add(r["Stock_Code"].strip())
    return sorted(pool)


def momentum_score(closes_list, L):
    """動能/趨勢型講白規則機械化:
       近 L 日報酬 (主動能) × 站上 MA(L) 加成 × 趨勢平滑(近 L 報酬 > 近 2L 報酬 加成)。
       回傳 >0 才算入選(只做多、汰弱)。"""
    n = len(closes_list)
    if n < 2 * L + 1:
        return 0.0
    c0 = closes_list[-1]
    cL = closes_list[-1 - L]
    if cL <= 0 or c0 <= 0:
        return 0.0
    ret_L = c0 / cL - 1.0                      # 近 L 日動能
    ma_L = sum(closes_list[-L:]) / L
    above = 1.0 if c0 > ma_L else 0.0          # 趨勢確認:站上均線才追
    c2L = closes_list[-1 - 2 * L]
    ret_2L = c0 / c2L - 1.0 if c2L > 0 else 0.0
    accel = 1.0 if ret_L > ret_2L / 2 else 0.0  # 動能加速(近段斜率 > 全段平均)
    if ret_L <= 0 or above == 0.0:
        return 0.0
    score = ret_L * (1.0 + 0.5 * accel)
    return score * 100.0


def build_mom_rows(pool, closes, sig_days, L, topn):
    """動能機械化訊號:每日池內挑 top-N 動能股。rows=[(date,code,edge)]。"""
    # 每檔的收盤序列(到 END)
    series = {}
    for c in pool:
        cc = closes.get(c, {})
        ds = sorted(d for d in cc if d <= END)
        series[c] = (ds, [cc[d] for d in ds])
    # 為每個 sig_day 找索引
    rows = []
    for d in sig_days:
        sc = []
        for c in pool:
            ds, cl = series[c]
            # 找 d 在該股序列的位置
            import bisect
            j = bisect.bisect_right(ds, d) - 1
            if j < 0 or ds[j] != d:
                continue
            window = cl[:j + 1]
            v = momentum_score(window, L)
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:topn]:
            rows.append((d, c, v / 100.0))   # edge 尺度對齊 h_score/100
    return rows


def build_h_rows(codes, feats, twii_feat, reb_cache, turn_pct, sig_days, topn):
    """baseline H+反彈雙引擎(限縮 codes 宇宙)。複製 ec.main 的訊號生成。"""
    regime_bull = {
        d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
        for d in sig_days
    }
    rows = []
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
        for v, c in sc[:topn]:
            rows.append((d, c, v / 100.0))
    return rows


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes_full = list(u.keys())
    names = {c: u[c].get("name", c) for c in codes_full}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes_full}

    pool = load_etf_pool()
    pool_in_bu = [c for c in pool if c in u]
    logger.info(f"ETF 池 {len(pool)} 檔,其中 {len(pool_in_bu)} 在 base_universe(有特徵/快取)")

    # 需要載入特徵的代號 = 全 base_universe ∪ 池 ∪ 0050
    need = sorted(set(codes_full) | set(pool) | {"0050"})
    logger.info(f"載入特徵({len(need)} 支,首次跑會抓 FinMind)...")
    twii_feat = features("0050")
    feats = {}
    opens, closes = {}, {}
    ok_pool = []
    for c in need:
        try:
            o = oh(c)
            if not o:
                continue
            opens[c] = {d: o[d]["open"] for d in o}
            closes[c] = {d: o[d]["close"] for d in o}
            if c in codes_full or c == "0050":
                feats[c] = features(c)
            if c in pool:
                ok_pool.append(c)
        except Exception as e:
            logger.warning(f"  {c} 載入失敗 {str(e)[:40]}")
    logger.info(f"池內有價格的 {len(ok_pool)}/{len(pool)} 檔")

    alld = sorted({d for c in closes for d in closes[c] if d <= END})
    sig_days = alld[-max(n for _, n in WINDOWS):]
    logger.info(f"訊號範圍 {sig_days[0]} ~ {sig_days[-1]}({len(sig_days)} 日)")

    # H baseline 需要的反彈訊號 + 漲停 + turn_pct(全 base_universe)
    logger.info("計算反彈訊號 + 漲停日(供 H baseline)...")
    reb_cache, limitup = {}, {}
    for c in need:
        o = oh(c) if c in opens else {}
        ds = sorted(d for d in o if d <= END); cl_list = []; m = {}; s = set()
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
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes_full
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    pool_priced = [c for c in ok_pool]

    # ── 變體定義 ──
    # baseline 兩條: H 全112、H 同池
    base_full = build_h_rows(codes_full, feats, twii_feat, reb_cache, turn_pct, sig_days, topn=4)
    base_pool = build_h_rows([c for c in pool_priced if c in feats],
                             feats, twii_feat, reb_cache, turn_pct, sig_days, topn=4)

    # 動能機械化:掃 lookback L 與 TOPN
    LOOKBACKS = [10, 20, 40, 60, 120]
    TOPNS = [3, 4, 6]
    mom_variants = {}
    for L in LOOKBACKS:
        for tn in TOPNS:
            mom_variants[(L, tn)] = build_mom_rows(pool_priced, closes, sig_days, L, tn)

    # ── 模擬 ──
    def run(rows, day_set):
        rw = [r for r in rows if r[0] in day_set]
        return sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)

    # 基準 0050(每窗/每regime)
    def bench_for(day_set):
        dd = sorted(d for d in day_set if d in closes.get("0050", {}))
        return bench_0050(opens["0050"], closes["0050"], dd)
    bench_win = {wl: bench_for(set(sig_days[-n:])) for wl, n in WINDOWS}
    regime_days = {rn: set(d for d in sig_days if s <= d <= e) for rn, (s, e) in REGIMES.items()}
    bench_reg = {rn: bench_for(dd) for rn, dd in regime_days.items()}

    configs = [("H全112", base_full), ("H同池", base_pool)]
    for (L, tn), rows in mom_variants.items():
        configs.append((f"動能L{L}N{tn}", rows))

    res = {}
    for label, rows in configs:
        for wl, n in WINDOWS:
            res[(label, wl)] = run(rows, set(sig_days[-n:]))
        for rn, dd in regime_days.items():
            res[(label, rn)] = run(rows, dd)
        logger.info(f"{label} 完成")

    # ── 報告(v15 多窗格式)──
    wlabels = [wl for wl, _ in WINDOWS]
    rlabels = list(REGIMES.keys())
    allcols = wlabels + rlabels

    def alpha(label, col):
        r = res.get((label, col))
        bench = bench_win.get(col, bench_reg.get(col))
        return (r["ret"] - bench) if (r and bench is not None) else None

    L = [
        "# 方法#1 策略原型機械化複製 — 動能/趨勢型ETF講白規則機械化\n",
        f"> 池=28檔主動式ETF全期持股聯集{len(pool)}檔(有價{len(pool_priced)})｜⑤買收賣開執行｜結束{END}｜還原價\n",
        "> 動能規則: 近L日報酬>0 且 站上MA(L);分數=retL×(1+0.5×加速旗);每日池內top-N\n",
        "> baseline: H全112=既有H雙引擎(全base_universe)｜H同池=H限縮到同一ETF池(公平對照)\n",
        "> 成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到｜資金 DCA 15000+1000/日上限5萬｜ALPHA=策略−同資金DCA0050\n",
        "> ⚠️ 作弊#1[選股池]: 用全期/當前ETF成員定義的50檔池=survivorship+時點洩漏。作弊#2[in-sample]: L/N在同段資料掃最佳。基準與成本不作弊。\n",
        "> ⚠️ 資料天花板: 僅1個多頭年的ETF持股(池偏大型股);價格回測雖含2021-2026但結論缺真正空頭壓力測試。\n",
        "> 0050 基準: " + " ".join(f"{wl}{bench_win[wl]:+.0f}%" for wl in wlabels) + " | "
        + " ".join(f"{rn}{bench_reg[rn]:+.0f}%" for rn in rlabels) + "\n",
        "## ALPHA %(扣 0050 同資金DCA)\n",
        "| 變體 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手(2年) | 持股(2年) | 曝險(2年) |",
        "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|",
    ]
    rows_out = []
    for label, _ in configs:
        cells = []; avals = []
        for col in allcols:
            a = alpha(label, col)
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None:
                avals.append(a)
        worst = f"{min(avals):+.0f}" if avals else "—"
        mean = f"{sum(avals)/len(avals):+.0f}" if avals else "—"
        r2y = res.get((label, "2年"))
        turn = f"{r2y['turn']:.0f}x" if r2y else "—"
        pos = f"{r2y.get('avg_pos', 0):.1f}" if r2y else "—"
        expo = f"{r2y.get('avg_expo', 0)*100:.0f}%" if r2y else "—"
        rows_out.append((label, cells, worst, mean, turn, pos, expo,
                         (sum(avals)/len(avals)) if avals else -1e9,
                         (min(avals)) if avals else -1e9))
        L.append(f"| {label} | " + " | ".join(cells) + f" | **{worst}** | {mean} | {turn} | {pos} | {expo} |")

    # 原始報酬
    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    for label, _ in configs:
        cells = [f"{res[(label,col)]['ret']:+.0f}" if res.get((label, col)) else "—" for col in allcols]
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    # uplift vs H同池(公平對照基準)
    L += ["", "## Uplift = 動能 alpha − H同池 alpha(pp,逐窗;對照同池H baseline)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    base_a = {col: alpha("H同池", col) for col in allcols}
    for label, _ in configs:
        if label in ("H全112", "H同池"):
            continue
        cells = []
        for col in allcols:
            a = alpha(label, col); b = base_a.get(col)
            cells.append(f"{a-b:+.0f}" if (a is not None and b is not None) else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    # 單調性檢查(平均alpha vs L, 固定N=4)
    L += ["", "## 單調性: 平均alpha vs lookback L (固定 N=4)\n", "| L | " + " | ".join(str(x) for x in LOOKBACKS) + " |",
          "|---|" + "|".join(["---"] * len(LOOKBACKS)) + "|"]
    rowm = []
    for Lk in LOOKBACKS:
        a = []
        for col in allcols:
            v = alpha(f"動能L{Lk}N4", col)
            if v is not None:
                a.append(v)
        rowm.append(f"{sum(a)/len(a):+.0f}" if a else "—")
    L.append("| 平均α | " + " | ".join(rowm) + " |")

    L += ["", "## 判讀\n",
          "- 看『動能各變體 vs H同池』的 uplift:全正且隨參數單調 → 動能原型機械化有加值。",
          "- 若 uplift 多為負/混亂 → 機械化動能在此大型股池無加值(H雙引擎已抓到趨勢)。",
          "- 曝險欄需與 H 同池接近才公平;最差欄看空頭/弱段保命力。"]

    RPT = ROOT / "reports" / "aetf_mechanize.md"
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    # console 摘要 + 找最佳動能變體
    best = max((r for r in rows_out if r[0].startswith("動能")), key=lambda r: r[7], default=None)
    if best:
        logger.success(f"最佳動能變體(平均α): {best[0]} 平均{best[3]} 最差{best[2]}")
    for label, _ in configs:
        a2 = alpha(label, "2年"); a90 = alpha(label, "90天"); a22 = alpha(label, "2022空頭")
        logger.success(f"{label}: 90天α{a90:+.0f} 2年α{a2:+.0f} 2022空頭α{a22:+.0f}"
                       if None not in (a2, a90, a22) else f"{label}: 部分窗無資料")

    # 機讀摘要回傳
    summary = {
        "bench_win": bench_win, "bench_reg": bench_reg,
        "rows": {label: {"cells": dict(zip(allcols, cells)), "worst": worst, "mean": mean,
                         "turn": turn, "pos": pos, "expo": expo}
                 for (label, cells, worst, mean, turn, pos, expo, _, _) in rows_out
                 for _ in [0]},
    }
    (ROOT / "reports" / "aetf_mechanize_summary.json").write_text(
        json.dumps({"best": best[0] if best else None,
                    "H同池_2年α": alpha("H同池", "2年"),
                    "H全112_2年α": alpha("H全112", "2年"),
                    "best_2年α": alpha(best[0], "2年") if best else None,
                    "best_mean": best[3] if best else None,
                    "best_worst": best[2] if best else None,
                    "uplift_2年": (alpha(best[0], "2年") - alpha("H同池", "2年")) if best else None,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
