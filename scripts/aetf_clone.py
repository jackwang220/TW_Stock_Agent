"""#2 複製最強經理人 (cloning 影子組合)。

方法:
  1. 算 28 檔主動式 ETF 各自 vs 0050(還原含息)的風險調整後績效,挑最強 1-3 檔。
  2. 鏡像那 1-3 檔的「每日揭露持股」(照權重),每日或每週 rebalance。
  3. 扣真實成本回測 vs DCA-0050 與 H 引擎 baseline。

資料: data/Active_ETF_1Y_Daily_28ETFs.csv(28 ETF、262 交易日、每檔~39 部位、池 50 大型股)。
價格: oh()=finmind get_daily_ohlcv(有 _sanitize 還原)。資金/成本模型重用 r60 常數。

⚠️ 第一輪小作弊(明確註記):
  (A) 時點洩漏: 用「同日揭露持股」當日就交易 —— ETF 持股實際盤後才揭露,
      鏡像時應 D+1 才知道。這裡 D 當天 close 就照 D 的揭露權重建倉。
  (B) in-sample 選股: 用「全期」資料挑風險調整後最強的 ETF(survivorship / look-ahead 選擇)。
  (C) 用當前 ETF 成員池(這批 ETF 本身是 2025 年新上市,存活者偏多頭)。
基準(DCA-0050)與成本(買0.1425%/賣0.4425%/滑價0.1%/漲停買不到)不作弊。

資料只 2025-06-17 ~ 2026-06-17 一年,且是多頭年、池偏大型股 → 缺空頭、結論有天花板。
多窗用全期切尾(60/90/126/252),無 2021-2024 regime 可切。

用法: uv run python scripts/aetf_clone.py
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

v3  = _load("v3",  ROOT / "scripts/exp_step1_v3.py")
ec  = _load("ec",  ROOT / "scripts/exp_60d_entry_compare.py")
v6  = _load("v6",  ROOT / "scripts/exp_step1_v6.py")
oh = v3.oh
bench_0050 = v6.bench_0050

# 重用 r60 資金模型 + ec 成本常數
INITIAL_CAPITAL = ec.r60.INITIAL_CAPITAL
DAILY_BUDGET    = ec.r60.DAILY_BUDGET
MAX_CONTRIBUTION= ec.r60.MAX_CONTRIBUTION
EXPOSURE_CAP    = ec.r60.EXPOSURE_CAP          # 0.90,留 ≥10% 現金(與 H 公平)
FEE_BUY, FEE_SELL, SLIP = ec.FEE_BUY, ec.FEE_SELL, ec.SLIP
ROUNDTRIP = FEE_BUY + FEE_SELL + 2 * SLIP      # ≈0.0078

CSV = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"
END = "2026-06-16"                              # 留 d+1 open 給賣出腿
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252)]


# ── 1. 讀 ETF 每日持股 ────────────────────────────────────────────────────────
def load_etf_holdings():
    df = pd.read_csv(CSV, dtype={"Stock_Code": str, "ETF_Code": str})
    df = df[df["Date"] <= END]
    hold = {}   # etf -> {date: {stock: weight_frac}}
    for et, g in df.groupby("ETF_Code"):
        byd = {}
        for d, gd in g.groupby("Date"):
            w = {r.Stock_Code: float(r._5) for r in gd.itertuples()}  # _5 = Weight(%)
            s = sum(w.values())
            if s > 0:
                byd[d] = {k: v / s for k, v in w.items()}   # 重新歸一(剩餘為現金/未揭露)
        hold[et] = byd
    stocks = sorted(df["Stock_Code"].unique())
    return hold, stocks


# ── 2. 載入價格 ───────────────────────────────────────────────────────────────
def load_prices(stocks):
    opens, closes = {}, {}
    for tk in stocks + ["0050"]:
        o = oh(tk)
        ds = sorted(d for d in o if d <= END)
        opens[tk] = {d: o[d]["open"] for d in ds}
        closes[tk] = {d: o[d]["close"] for d in ds}
    # 漲停集合(隔日開盤仍漲停 → 買不到)
    limitup = {}
    for tk in stocks + ["0050"]:
        cl = closes[tk]; ds = sorted(cl); s = set()
        for j in range(1, len(ds)):
            p = cl[ds[j-1]]
            if p > 0 and cl[ds[j]] / p - 1 >= 0.095:
                s.add(ds[j])
        limitup[tk] = s
    return opens, closes, limitup


# ── 3. ETF 強弱排名 (cheat B: in-sample 全期) ────────────────────────────────
def rank_etfs(hold, stocks, opens, closes, cal):
    """用每檔 ETF 自己揭露持股組成的「紙上組合日報酬」算 Sharpe vs 0050。
    這是評估『經理人選股能力』,不含交易成本(純訊號強弱),用來挑要 clone 誰。"""
    # 0050 日報酬(還原含息 close-to-close)
    c50 = closes["0050"]
    def daily_rets(weight_series):
        rets = []
        for i in range(1, len(cal)):
            d0, d1 = cal[i-1], cal[i]
            w = weight_series(d0)
            if not w:
                rets.append(0.0); continue
            r = 0.0
            for tk, wt in w.items():
                p0 = closes.get(tk, {}).get(d0); p1 = closes.get(tk, {}).get(d1)
                if p0 and p1 and p0 > 0:
                    r += wt * (p1 / p0 - 1)
            rets.append(r)
        return rets
    bench_rets = [c50[cal[i]] / c50[cal[i-1]] - 1 for i in range(1, len(cal))
                  if cal[i] in c50 and cal[i-1] in c50]
    import statistics
    res = []
    for et, byd in hold.items():
        # forward-fill 持股
        last = {}
        def ws(d, _byd=byd):
            nonlocal last
            return _byd.get(d, last)
        rets = []
        last = {}
        for i in range(1, len(cal)):
            d0 = cal[i-1]
            if d0 in byd:
                last = byd[d0]
            w = last
            d1 = cal[i]; r = 0.0
            for tk, wt in w.items():
                p0 = closes.get(tk, {}).get(d0); p1 = closes.get(tk, {}).get(d1)
                if p0 and p1 and p0 > 0:
                    r += wt * (p1 / p0 - 1)
            rets.append(r)
        if len(rets) < 30:
            continue
        n = min(len(rets), len(bench_rets))
        excess = [rets[i] - bench_rets[i] for i in range(n)]
        mu = sum(excess) / n
        sd = statistics.pstdev(excess) or 1e-9
        ir = mu / sd * math.sqrt(252)                 # 資訊比率 (年化, vs 0050)
        tot = 1.0
        for r in rets:
            tot *= (1 + r)
        res.append((et, ir, (tot - 1) * 100, mu * 252 * 100))
    res.sort(key=lambda x: x[1], reverse=True)
    return res


# ── 4. clone 加權組合模擬器(DCA 資金 + 全成本 + 曝險上限 + rebalance 頻率)──
def sim_clone(target_weights_by_day, opens, closes, limitup, cal,
              rebal_days=1, expo_cap=EXPOSURE_CAP, band=0.0, switch_mult=1.0):
    """每日(或每 rebal_days 日)把持倉 rebalance 到 ETF 揭露權重 × 曝險上限。
    執行: D 收盤決策(cheat A: 用 D 當天揭露權重),賣腿/買腿都用 D+1 開盤成交。
      - 賣: D+1 open 賣出超配/掉出名單(現金當天可用)
      - 買: D+1 open 補到目標(隔日開盤仍漲停 → 買不到)
    band: 只有 |目標-現值|/port > band 才動(降換手); switch_mult×ROUNDTRIP 為最低換手門檻。
    """
    cash = contributed = traded = fees = 0.0
    shares = defaultdict(float)
    prev_eq = 0.0
    day_pnl = []
    expo_track = []
    pos_track = []
    last_target = {}
    thresh = max(band, switch_mult * ROUNDTRIP)

    def cl(tk, d):
        c = closes.get(tk, {}); ds = [x for x in c if x <= d]
        return c[max(ds)] if ds else None

    for i, d in enumerate(cal):
        add = (INITIAL_CAPITAL if i == 0
               else (min(DAILY_BUDGET, MAX_CONTRIBUTION - contributed)
                     if contributed < MAX_CONTRIBUTION else 0.0))
        cash += add; contributed += add

        if i + 1 >= len(cal):
            eq = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)
            day_pnl.append(eq - prev_eq - add)
            break

        e = cal[i + 1]
        port = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)

        # rebalance 頻率: 非 rebalance 日沿用上次目標
        if i % rebal_days == 0 and d in target_weights_by_day:
            last_target = target_weights_by_day[d]
        tw = last_target
        # 目標金額 = port × expo_cap × 權重
        targets = {tk: port * expo_cap * w for tk, w in tw.items()} if tw else {}

        # ── 賣腿 @ D+1 open(超配 / 掉出名單) ──
        for tk in list(shares):
            op = opens.get(tk, {}).get(e)
            if not op or op <= 0:
                continue
            cur = shares[tk] * op
            tgt = targets.get(tk, 0.0)
            if cur - tgt > thresh * port:
                sell_amt = cur - tgt
                shares[tk] -= sell_amt / op
                cash += sell_amt - sell_amt * (FEE_SELL + SLIP)
                traded += sell_amt; fees += sell_amt * (FEE_SELL + SLIP)
                if shares[tk] < 1e-9:
                    del shares[tk]

        # ── 買腿 @ D+1 open(補到目標;漲停買不到) ──
        for tk, tgt in targets.items():
            op = opens.get(tk, {}).get(e)
            if not op or op <= 0:
                continue
            if e in limitup.get(tk, set()):     # 隔日開盤仍漲停 → 買不到
                continue
            cur = shares[tk] * op
            if tgt - cur > thresh * port:
                buy_amt = min(tgt - cur, cash / (1 + FEE_BUY + SLIP))
                if buy_amt <= 0:
                    continue
                shares[tk] += buy_amt / op
                cash -= buy_amt + buy_amt * (FEE_BUY + SLIP)
                traded += buy_amt; fees += buy_amt * (FEE_BUY + SLIP)

        eq = cash + sum(shares[tk] * (cl(tk, e) or 0) for tk in shares)
        day_pnl.append(eq - prev_eq - add)
        prev_eq = eq
        invested = sum(shares[tk] * (cl(tk, e) or 0) for tk in shares)
        expo_track.append(invested / eq if eq > 0 else 0.0)
        pos_track.append(len(shares))

    final_eq = cash + sum(shares[tk] * (cl(tk, cal[-1]) or 0) for tk in shares)
    ret = (final_eq - contributed) / contributed * 100 if contributed else 0.0
    return {
        "ret": ret,
        "turn": traded / contributed if contributed else 0.0,
        "fees": fees,
        "avg_expo": sum(expo_track) / len(expo_track) if expo_track else 0.0,
        "avg_pos": sum(pos_track) / len(pos_track) if pos_track else 0.0,
    }


def build_clone_targets(hold, etf_codes, cal):
    """合併選中的 1-N 檔 ETF 的每日揭露權重(等權平均),forward-fill。"""
    tw = {}
    last = {}
    for d in cal:
        merged = defaultdict(float)
        present = 0
        for et in etf_codes:
            byd = hold[et]
            if d in byd:
                for tk, w in byd[d].items():
                    merged[tk] += w
                present += 1
        if present:
            s = sum(merged.values())
            last = {tk: w / s for tk, w in merged.items()} if s > 0 else last
        if last:
            tw[d] = dict(last)
    return tw


def main():
    logger.info("讀 ETF 持股 + 價格...")
    hold, stocks = load_etf_holdings()
    opens, closes, limitup = load_prices(stocks)
    cal = sorted(set(closes["0050"]) & {d for d in
                 set().union(*[set(closes[s]) for s in stocks])})
    cal = [d for d in cal if d <= END]
    # 用 ETF 資料涵蓋的交易日(以 0050 為基準日曆,限制在 ETF 期間)
    etf_days = sorted({d for byd in hold.values() for d in byd})
    cal = [d for d in cal if etf_days[0] <= d <= etf_days[-1]]
    logger.info(f"日曆 {cal[0]} ~ {cal[-1]}({len(cal)} 日)")

    # ── 排名 (cheat B) ──
    ranking = rank_etfs(hold, stocks, opens, closes, cal)
    logger.info("ETF 風險調整排名 (IR vs 0050, 全期 in-sample):")
    for et, ir, tot, ann in ranking[:8]:
        logger.info(f"  {et}: IR={ir:+.2f} 總報酬{tot:+.1f}% 年化超額{ann:+.1f}%")
    top1 = [ranking[0][0]]
    top3 = [r[0] for r in ranking[:3]]

    # ── 0050 基準(各窗) ──
    bench_win = {}
    for wl, n in WINDOWS:
        dd = cal[-n:]
        bench_win[wl] = bench_0050(opens["0050"], closes["0050"], dd)

    # ── H 引擎 baseline(重用 news_llm 的候選+sim5,純引擎不加 LLM) ──
    h_alpha = compute_h_baseline(cal, bench_win)

    # ── clone 各設定 ──
    configs = [
        ("clone-top1 每日", top1, 1, 0.0),
        ("clone-top1 每週", top1, 5, 0.0),
        ("clone-top3 每日", top3, 1, 0.0),
        ("clone-top3 每週", top3, 5, 0.0),
        ("clone-top3 每週+2%帶", top3, 5, 0.02),
        ("clone-top1 每月", top1, 21, 0.0),
    ]
    res = {}   # (label, wl) -> sim
    tw_cache = {}
    for label, etfs, rebal, band in configs:
        key = (tuple(etfs), rebal, band)
        if key not in tw_cache:
            tw_cache[key] = build_clone_targets(hold, etfs, cal)
        tw = tw_cache[key]
        for wl, n in WINDOWS:
            dd = cal[-n:]
            twd = {d: tw[d] for d in dd if d in tw}
            res[(label, wl)] = sim_clone(twd, opens, closes, limitup, dd,
                                         rebal_days=rebal, band=band)
        logger.info(f"{label} 完成 1年α="
                    f"{res[(label,'1年')]['ret']-bench_win['1年']:+.0f}")

    write_report(ranking, top1, top3, bench_win, h_alpha, configs, res)


def compute_h_baseline(cal, bench_win):
    """H+反彈雙引擎(純,無 LLM)alpha,各窗。重用 news_llm 的 build_candidates + sim5。"""
    try:
        nl = _load("nl", ROOT / "scripts/news_llm_select.py")
    except Exception as ex:
        logger.warning(f"H baseline 載入失敗,跳過: {ex}")
        return {wl: None for wl, _ in WINDOWS}
    from tw_stock_agent.tools.rebound_signal import rebound_signal
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    feats = {c: ec.features(c) for c in codes}
    twii_feat = ec.features("0050")
    o_open, o_close = {}, {}
    for c in codes + ["0050"]:
        o = oh(c)
        o_open[c] = {d: o[d]["open"] for d in o if d <= END}
        o_close[c] = {d: o[d]["close"] for d in o if d <= END}
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
    sig_days = [d for d in sorted(twii_feat) if d <= END][-252:]
    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    cands, _ = nl.build_candidates(codes, names, feats, twii_feat, reb_cache, turn_pct, sig_days)
    out = {}
    for wl, n in WINDOWS:
        dd = set(sig_days[-n:])
        rows = [(d, c, v/100.0) for d in sig_days if d in dd for v, c in cands[d]]
        sim = ec.sim_buyclose_sellopen(rows, o_open, o_close, limitup, switch_cost_mult=1.0)
        out[wl] = (sim["ret"] - bench_win[wl]) if sim else None
    return out


def write_report(ranking, top1, top3, bench_win, h_alpha, configs, res):
    wlabels = [wl for wl, _ in WINDOWS]
    L = ["# #2 複製最強經理人 (cloning 影子組合)\n",
         f"> 28 主動式 ETF 每日揭露持股｜結束 {END}｜還原含息價(finmind _sanitize)\n",
         "> 資金 DCA 15000+1000/日上限5萬｜成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到\n",
         "> ALPHA = clone − 同資金 DCA 0050；曝險上限 90%(與 H 引擎同,公平)\n",
         "> ⚠️ 作弊: (A)時點洩漏=同日揭露持股當日決策 (B)in-sample 全期挑最強 ETF "
         "(C)當前 ETF 成員池(2025新上市,存活偏多頭)\n",
         "> 資料僅 1 年多頭、池偏大型股 → 缺空頭、無 2021-24 regime,結論有天花板\n",
         "> 0050 基準: " + " ".join(f"{wl}{bench_win[wl]:+.0f}%" for wl in wlabels) + "\n",
         "## ETF 風險調整排名 (IR vs 0050, 全期 in-sample, 紙上不含成本)\n",
         "| 名次 | ETF | IR(年化) | 全期總報酬% | 年化超額% |",
         "|---|---|---|---|---|"]
    for i, (et, ir, tot, ann) in enumerate(ranking[:8], 1):
        L.append(f"| {i} | {et} | {ir:+.2f} | {tot:+.1f} | {ann:+.1f} |")
    L += [f"\n選中: top1={top1}  top3={top3}\n",
          "## ALPHA %(扣 0050 beta)\n",
          "| 變體 | " + " | ".join(wlabels) + " | 最差 | 平均 | 換手 | 持股 | 曝險 |",
          "|---|" + "|".join(["---"] * (len(wlabels) + 5)) + "|"]
    # H baseline 列
    havals = [h_alpha[wl] for wl in wlabels if h_alpha[wl] is not None]
    hcells = [f"{h_alpha[wl]:+.0f}" if h_alpha[wl] is not None else "—" for wl in wlabels]
    L.append(f"| **H雙引擎 baseline** | " + " | ".join(hcells)
             + f" | **{min(havals):+.0f}** | {sum(havals)/len(havals):+.0f} | — | — | — |"
             if havals else f"| **H雙引擎 baseline** | " + " | ".join(hcells) + " | — | — | — | — | — |")
    for label, _, _, _ in configs:
        avals = []; cells = []
        for wl in wlabels:
            a = res[(label, wl)]["ret"] - bench_win[wl]
            cells.append(f"{a:+.0f}"); avals.append(a)
        r1 = res[(label, "1年")]
        L.append(f"| {label} | " + " | ".join(cells)
                 + f" | **{min(avals):+.0f}** | {sum(avals)/len(avals):+.0f} |"
                 + f" {r1['turn']:.1f}x | {r1['avg_pos']:.0f} | {r1['avg_expo']*100:.0f}% |")

    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 變體 | " + " | ".join(wlabels) + " |",
          "|---|" + "|".join(["---"] * len(wlabels)) + "|"]
    for wl in wlabels:
        pass
    L.append(f"| 0050 DCA | " + " | ".join(f"{bench_win[wl]:+.0f}" for wl in wlabels) + " |")
    for label, _, _, _ in configs:
        L.append(f"| {label} | " + " | ".join(f"{res[(label,wl)]['ret']:+.0f}" for wl in wlabels) + " |")

    # uplift vs H baseline + vs 0050
    L += ["", "## Uplift = clone alpha − H baseline alpha (pp, 逐窗)\n",
          "| 變體 | " + " | ".join(wlabels) + " |",
          "|---|" + "|".join(["---"] * len(wlabels)) + "|"]
    for label, _, _, _ in configs:
        cells = []
        for wl in wlabels:
            a = res[(label, wl)]["ret"] - bench_win[wl]
            cells.append(f"{a - h_alpha[wl]:+.0f}" if h_alpha[wl] is not None else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += ["", "## 判讀",
          "- clone alpha vs 0050 為正且隨頻率單調 → 經理人選股有 edge(扣成本後)。",
          "- vs H baseline uplift: clone > H 才代表『抄經理人』勝過自建技術引擎。",
          "- 換手過高(每日 rebalance)會被成本吃掉 → 看每週/每月是否更優(單調性)。",
          "- ⚠️ 此 alpha 含時點洩漏(A)+in-sample 選股(B);Verify 階段需改 D+1 揭露 + walk-forward 選 ETF。"]
    RPT = ROOT / "reports" / "aetf_clone.md"
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
