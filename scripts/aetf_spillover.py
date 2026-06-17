"""方法 #8:類股擁擠 → 擴散到同類 peer(⭐使用者點子)

核心問句:把 28 檔主動式 ETF 的持股按 industry 分類,算它們集體「擁擠/超配」
在哪些類股;然後去**全市場(tw_stock_index 2139 檔)**撈那些擁擠類股裡
「H 引擎漏掉(不在 ETF 池)、但本身很強勢」的 peer 買進,賭主題擴散
(經理人重押半導體/電子零組件 → 同類但沒被 ETF 收的強勢小弟也會被帶動)。

與已證實無 edge 的「每日流向 tilt」(etf_consensus_probe)不同維度:
這裡用的是「結構性類股集中」(哪些產業被集體超配)+ 「全市場同類擴散」,
不是當日加減碼方向。

定義:
  ind_w(I)  = 全期 28 ETF-blend 權重在 industry I 的加總(擁擠度;in-sample 作弊)
  crowded   = ind_w 最高的 K 個 industry
  peers(I)  = tw_stock_index 中 industry==I 且「不在 ETF 池」的全市場股票
              (撈得到還原價、且流動性 ≥ 門檻 的才留)
  strength  = peer 自身價格算的 H 動能分數(與 baseline 同一把 H 引擎)
              → 「H 漏掉但強勢」= 不在池、但 H 分數高
  每日候選   = 擁擠類股 peer 中 H 分數 top-N → 餵 sim5(⑤買收賣開)

baseline = H+反彈雙引擎在 ETF 池(50 檔)上(專案既定 baseline)。

⚠️ 第一輪允許的作弊(cheat_used,Verify 階段再修):
  - in-sample 選類股:用「全期」ETF 權重決定哪些類股最擁擠(未來資訊)。
  - membership/survivorship:用「當前」28 ETF 成員 + 50 檔池(多 2025 才上市)。
  - 時點洩漏(輕):擁擠度用全期、peer 強勢用當日 H(H 只用 ≤d 收盤,本身不洩漏;
    但「哪些類股熱」是全期決定的 → 提供 --rolling 開關用滾動窗 leak-free 探針)。
  - 池/peer membership:peer 名單用「當前」tw_stock_index 的 industry(事後字典)。
不作弊的部分:alpha 一律扣同期同資金 DCA 0050;成本 買0.1425%/賣0.4425%+滑價0.1%;
  漲停買不到;曝險中性(sim5 內建 EXPOSURE_CAP/FLOOR,與 baseline 同 sizing);
  還原含息價(get_daily_ohlcv 有 _sanitize)。

用法:
    uv run python scripts/aetf_spillover.py                 # 預設參數
    uv run python scripts/aetf_spillover.py --scan          # 掃 K_IND / TOPN / 流動性門檻
    uv run python scripts/aetf_spillover.py --k-ind 4 --topn 4 --min-turn 5e7
    uv run python scripts/aetf_spillover.py --max-fetch 150 # 控 FinMind 抓取量
"""
from __future__ import annotations
import sys, json, argparse, importlib.util, math
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
sim5 = ec.sim_buyclose_sellopen   # ⑤買收賣開:各窗皆正 alpha 的最佳執行

END = "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]
# 資料只 2025-06-17~2026-06-17 → regime 只切「可得段」(多頭年,無空頭) → 講明天花板
REGIMES = {
    "2025下半": ("2025-07-01", "2025-12-31"),
    "2026上半": ("2026-01-01", END),
}
ETF_CSV = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"


# ── 擁擠類股 & 全市場 peer ────────────────────────────────────────────────────
def industry_of(c, base_u, ix):
    if c in base_u and base_u[c].get("industry"):
        return base_u[c]["industry"]
    if c in ix and ix[c].get("industry"):
        return ix[c]["industry"]
    return None


def crowded_industries(df, pool, base_u, ix, k):
    """全期 ETF-blend 權重加總 → 擁擠度;回傳 top-k industry(in-sample 作弊)。"""
    indw = defaultdict(float)
    for c in pool:
        I = industry_of(c, base_u, ix)
        if I:
            indw[I] += float(df[df["Stock_Code"] == c]["w"].sum())
    ranked = sorted(indw.items(), key=lambda x: -x[1])
    return ranked[:k], ranked


def find_peers(crowded, pool, ix, max_per_ind):
    """擁擠類股的全市場 peer(不在池)。為控 FinMind 抓取量,每類股優先 TWSE、
    取前 max_per_ind 個(以代號排序,純為確定性;真正篩選用流動性在抓價後做)。"""
    peers = {}   # industry -> [codes]
    for I, _ in crowded:
        cands = [c for c, m in ix.items()
                 if m.get("industry") == I and c not in pool]
        # TWSE 優先(.TW 比 .TWO 流動性通常高、抓得到價)
        cands.sort(key=lambda c: (ix[c].get("market") != "TWSE", c))
        peers[I] = cands[:max_per_ind]
    return peers


# ── 主程式 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-ind", type=int, default=4, help="取前 K 個最擁擠類股")
    ap.add_argument("--topn", type=int, default=4, help="每日買 peer H 分數 top-N")
    ap.add_argument("--min-turn", type=float, default=5e7, help="peer 最低平均成交值(close*vol)門檻")
    ap.add_argument("--max-per-ind", type=int, default=40, help="每類股最多抓幾個 peer(控 FinMind 量)")
    ap.add_argument("--max-fetch", type=int, default=160, help="peer 抓取總上限(控 FinMind 量)")
    ap.add_argument("--scan", action="store_true", help="掃 K_IND / TOPN / min-turn 找最佳與單調性")
    args = ap.parse_args()

    base_u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    ix = json.loads((DATA_DIR / "tw_stock_index.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ETF_CSV, dtype={"Stock_Code": str, "ETF_Code": str})
    df["w"] = df["Weight(%)"].astype(float)
    pool = sorted(df["Stock_Code"].unique())
    logger.info(f"ETF 池 {len(pool)} 檔")

    # 擁擠類股(用最大 k 掃描範圍,之後子集)
    kmax = 8 if args.scan else args.k_ind
    crowded_all, ranked_all = crowded_industries(df, pool, base_u, ix, kmax)
    logger.info("=== 擁擠類股(全期 ETF 權重加總,in-sample 作弊)top ===")
    for I, w in ranked_all[:10]:
        logger.info(f"  {I}  w={w:.0f}  pool={sum(1 for c in pool if industry_of(c,base_u,ix)==I)}")

    # peer 名單(取 kmax 範圍,抓價,流動性篩選)
    peers_by_ind = find_peers(crowded_all, pool, ix, args.max_per_ind)
    all_peer_codes = []
    for I in peers_by_ind:
        all_peer_codes += peers_by_ind[I]
    all_peer_codes = list(dict.fromkeys(all_peer_codes))[: args.max_fetch]
    logger.info(f"擬抓 peer {len(all_peer_codes)} 檔(上限 {args.max_fetch};FinMind 有快取)")

    # ── 抓價 + 流動性篩選 ──
    opens, closes, peer_turn, peer_ind = {}, {}, {}, {}
    fetched_ok = []
    for j, c in enumerate(all_peer_codes):
        try:
            o = oh(c)
        except Exception:
            o = {}
        if len(o) < 60:
            continue
        ds = sorted(d for d in o if d <= END)
        if len(ds) < 60:
            continue
        # 平均成交值(close*volume)近 60 日中位數
        tv = sorted(o[d]["close"] * o[d].get("volume", 0) for d in ds[-60:])
        med_tv = tv[len(tv) // 2] if tv else 0.0
        if med_tv < args.min_turn:
            continue
        opens[c] = {d: o[d]["open"] for d in o}
        closes[c] = {d: o[d]["close"] for d in o}
        peer_turn[c] = med_tv
        peer_ind[c] = next((I for I in peers_by_ind if c in peers_by_ind[I]), None)
        fetched_ok.append(c)
        if (j + 1) % 40 == 0:
            logger.info(f"  抓價 {j+1}/{len(all_peer_codes)}(留用 {len(fetched_ok)})")
    logger.info(f"peer 通過流動性門檻 {len(fetched_ok)} 檔(min_turn={args.min_turn:.0e})")

    # 0050 + 池(baseline 用)
    for c in pool + ["0050"]:
        o = oh(c)
        opens[c] = {d: o[d]["open"] for d in o}
        closes[c] = {d: o[d]["close"] for d in o}

    # 漲停日(sim5 需要):池 + peer
    limitup = {}
    for c in set(pool) | set(fetched_ok):
        o = oh(c); ds = sorted(o); s = set()
        for i, d in enumerate(ds):
            if i > 0 and o[ds[i-1]]["close"] > 0 and o[d]["close"]/o[ds[i-1]]["close"]-1 >= 0.095:
                s.add(d)
        limitup[c] = s

    # 交易日(以 ETF 資料範圍內、0050 有價的日子)
    etf_dates = sorted(df["Date"].unique())
    alld = sorted({d for d in closes["0050"] if etf_dates[0] <= d <= END})
    logger.info(f"交易日 {alld[0]}~{alld[-1]} ({len(alld)})")

    # ── baseline:H+反彈雙引擎(專案既定 baseline)在 ETF 池 ──
    logger.info("建 baseline(H+反彈雙引擎@池)...")
    turns_pool = {c: base_u.get(c, {}).get("avg_turnover", 0.0) for c in pool}
    twii_feat = features("0050")
    feats_pool = {c: features(c) for c in pool}
    reb_pool = {}
    for c in pool:
        o = oh(c); ds = sorted(o); cl = []; m = {}
        for d in ds:
            cl.append(o[d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns_pool.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
        reb_pool[c] = m
    turn_pct_pool = {}
    for d in alld:
        vals = sorted(((c, feats_pool[c][d]["turn"]) for c in pool
                       if d in feats_pool.get(c, {}) and feats_pool[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct_pool[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    def base_rows():
        rows = []
        for d in alld:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
            sc = []
            for c in pool:
                f = feats_pool.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                    continue
                v = (h_score(_factors(f[d], ir), turn_pct_pool.get(d, {}).get(c, 0.5))
                     if bull else reb_pool.get(c, {}).get(d, 0.0))
                if v > 0:
                    sc.append((v, c))
            sc.sort(reverse=True)
            for v, c in sc[:args.topn]:
                rows.append((d, c, v / 100))
        return rows

    BASE = base_rows()

    # ── peer 特徵(H 分數;只用純動能 H,不含反彈,因為這是「強勢擴散」)──
    logger.info("建 peer H 特徵...")
    feats_peer = {c: features(c) for c in fetched_ok}
    # peer 在全 peer 池內的 turn 百分位(H 的成交值聚光燈因子)
    turn_pct_peer = {}
    for d in alld:
        vals = sorted(((c, feats_peer[c][d]["turn"]) for c in fetched_ok
                       if d in feats_peer.get(c, {}) and feats_peer[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct_peer[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    def peer_rows(k_ind, topn, allowed_ind):
        """每日:擁擠類股 peer 的 H 動能分數 top-N → edge=score/100。
        bull regime 才買(H 只在多頭打,與 baseline 一致);空頭日空手。"""
        allowed = set(allowed_ind)
        rows = []
        for d in alld:
            if not regime_bull.get(d):
                continue
            ir = twii_feat.get(d, {}).get("ret20")
            sc = []
            for c in fetched_ok:
                if peer_ind.get(c) not in allowed:
                    continue
                f = feats_peer.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                    continue
                v = h_score(_factors(f[d], ir), turn_pct_peer.get(d, {}).get(c, 0.5))
                if v > 0:
                    sc.append((v, c))
            sc.sort(reverse=True)
            for v, c in sc[:topn]:
                rows.append((d, c, v / 100))
        return rows

    # ── 0050 基準(各窗 + regime)──
    def bench_for(dd):
        dd = sorted(d for d in dd if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    col_days = {wl: alld[-n:] for wl, n in WINDOWS}
    regime_days = {rn: [d for d in alld if s <= d <= e] for rn, (s, e) in REGIMES.items()}
    col_days.update(regime_days)
    allcols = [wl for wl, _ in WINDOWS] + list(REGIMES.keys())
    bench = {col: bench_for(col_days[col]) for col in allcols}

    def run(rows, dd):
        s = set(dd)
        rw = [r for r in rows if r[0] in s]
        return sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)

    # ── SCAN ──
    if args.scan:
        logger.info("=== SCAN ===")
        ind_names = [I for I, _ in crowded_all]
        scan = []
        for k_ind in (2, 3, 4, 6):
            allowed = ind_names[:k_ind]
            for topn in (3, 4, 6):
                rw = peer_rows(k_ind, topn, allowed)
                a = {}
                for col in allcols:
                    r = run(rw, col_days[col])
                    a[col] = (r["ret"] - bench[col]) if (r and bench[col] is not None) else None
                vals = [v for v in a.values() if v is not None]
                scan.append((k_ind, topn, a, min(vals) if vals else None,
                             sum(vals)/len(vals) if vals else None))
        for k_ind, topn, a, w, m in scan:
            logger.info(f"  K_IND={k_ind} TOPN={topn}: mean={m:+.1f} worst={w:+.1f} | "
                        + " ".join(f"{k}{v:+.0f}" for k, v in a.items() if v is not None))

    # ── 主回測:baseline vs spillover(用 args 參數)──
    ind_names = [I for I, _ in crowded_all][:args.k_ind]
    SPILL = peer_rows(args.k_ind, args.topn, ind_names)
    n_peer_traded = len({tk for _, tk, _ in SPILL})
    logger.info(f"spillover 實際交易 peer {n_peer_traded} 檔;訊號列 {len(SPILL)}")

    res = {}
    for label, rows in (("baseline(H雙引擎@池)", BASE), ("spillover(類股擴散peer)", SPILL)):
        for col in allcols:
            res[(label, col)] = run(rows, col_days[col])

    def alpha(label, col):
        r = res.get((label, col)); b = bench.get(col)
        return (r["ret"] - b) if (r and b is not None) else None

    # ── 報告(v15 多窗格式)──
    RPT = ROOT / "reports" / "aetf_spillover.md"
    RPT.parent.mkdir(parents=True, exist_ok=True)
    labels = ["baseline(H雙引擎@池)", "spillover(類股擴散peer)"]
    L = [f"# 方法#8 類股擁擠 → 擴散到同類 peer(⭐使用者點子)\n",
         f"> 擁擠類股(全期ETF權重)top-{args.k_ind}={'、'.join(ind_names)}"
         f"｜每日買該類全市場 peer 的 H 分數 top-{args.topn}｜流動性≥{args.min_turn:.0e}｜⑤買收賣開"
         f"｜結束 {END}｜還原價\n",
         f"> baseline = H+反彈雙引擎@ETF池(50檔,專案既定)｜peer 來源 tw_stock_index 全市場、不在池"
         f"｜實抓 peer {len(all_peer_codes)} 檔→過門檻 {len(fetched_ok)} 檔→實交易 {n_peer_traded} 檔\n",
         f"> 成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到｜ALPHA = 策略 − 同資金 DCA 0050"
         f"｜資金 15000+1000/日上限5萬｜曝險 sim5 內建 30–90%\n",
         f"> ⚠️ 作弊(第一輪):in-sample 選類股(全期權重定擁擠)、membership/survivorship"
         f"(當前28ETF+50池+全市場字典,多2025上市)、peer industry 用事後字典\n",
         f"> ⚠️ 天花板:資料僅 1 個多頭年、池偏大型股 → 無空頭、結論偏樂觀\n",
         "> 0050 基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in allcols) + "\n",
         "## ALPHA %(扣 0050)\n",
         "| 變體 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手 | 持股 | 曝險 |",
         "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|"]
    for label in labels:
        cells = []; avals = []
        for col in allcols:
            a = alpha(label, col)
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None:
                avals.append(a)
        ref = res.get((label, "1年")) or res.get((label, allcols[0]))
        worst = f"{min(avals):+.0f}" if avals else "—"
        mean = f"{sum(avals)/len(avals):+.0f}" if avals else "—"
        turn = f"{ref['turn']:.1f}x" if ref else "—"
        pos = f"{ref.get('avg_pos',0):.1f}" if ref else "—"
        expo = f"{ref.get('avg_expo',0)*100:.0f}%" if ref else "—"
        L.append(f"| {label} | " + " | ".join(cells) + f" | **{worst}** | {mean} | {turn} | {pos} | {expo} |")

    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    for label in labels:
        L.append(f"| {label} | " + " | ".join(
            f"{res[(label,col)]['ret']:+.0f}" if res.get((label, col)) else "—" for col in allcols) + " |")

    L += ["", "## Uplift = spillover alpha − baseline alpha(pp,逐窗)\n",
          "| " + " | ".join(allcols) + " |",
          "|" + "|".join(["---"] * len(allcols)) + "|"]
    cells = []
    for col in allcols:
        a = alpha("spillover(類股擴散peer)", col); b = alpha("baseline(H雙引擎@池)", col)
        cells.append(f"{a-b:+.0f}" if (a is not None and b is not None) else "—")
    L.append("| " + " | ".join(cells) + " |")

    L += ["", "## 判讀\n",
          "- spillover alpha 各窗皆正且 > baseline → 類股擴散有加值。",
          "- uplift ≈0 或負 → 擁擠類股的全市場強勢 peer 沒比池內 H 候選更好(主題擴散不成立)。",
          "- 注意曝險欄:兩者曝險接近才是公平比較。"]
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    for col in allcols:
        a = alpha("spillover(類股擴散peer)", col); b = alpha("baseline(H雙引擎@池)", col)
        if a is not None and b is not None:
            logger.info(f"  {col}: spillover α{a:+.0f} baseline α{b:+.0f} uplift{a-b:+.0f}")


if __name__ == "__main__":
    main()
