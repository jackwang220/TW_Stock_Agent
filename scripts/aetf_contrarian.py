"""#6 反向擁擠 (Contrarian / Crowding-fade) — 主動式ETF擁擠度反向訊號驗證。

已知:總權重越高的股「前向略輸」(IC ≈ −0.04, etf_consensus_probe)。本腳本「認真測 contrarian」:
  (A) 探針: 跨多種擁擠度指標的 rank-IC + 分位 long-short(空最擁擠/多最不擁擠)。
      擁擠度指標(每 date×stock 跨28檔ETF聚合):
        - sum_wt    : 總持有權重(已知 IC<0)
        - n_hold    : 幾檔ETF持有(廣度擁擠)
        - max_wt    : 單檔ETF最大押注(最重押)
        - wt_disp   : 權重離散(經理人分歧度,std/mean)→ 分歧高=機會?
      forward 從 D+1(leak-free 版) 與 D0(時點洩漏作弊版) 都跑,看洩漏放大多少。
  (B) 策略: 在 H+反彈雙引擎候選上做 contrarian tilt:
        edge_contra = edge × (1 + k·(0.5 − crowd_pctile))
      crowd 高(pctile→1) → 乘數<1 縮;crowd 低 → 放大。掃 k 找最佳與單調性。
      baseline = 不加 contrarian 的純引擎(⑤買收賣開)。多窗 alpha vs DCA0050、扣成本。

執行/成本/基準(不作弊):
  ⑤買收賣開引擎、買0.1425%/賣0.4425%+滑價0.1%、漲停買不到。
  ALPHA = 策略 − 同資金 DCA 0050(15000+1000/日,上限5萬)。曝險中性(tilt 以1為中心)。

⚠️ 作弊註記(cheat_used):
  - in-sample/全期:擁擠度 pctile 用「當日橫斷面」排名(不是未來資訊,OK),
    但分位 long-short 與 tilt 的「哪個指標有效」是看全期結果挑的(in-sample 選指標)。
  - survivorship + current members:用「當前28檔ETF成員 + 其每日持股」(成員是現存ETF)。
  - 時點洩漏:ETF持股當日盤後才揭露。探針(A)同時跑 D+1(leak-free)與 D0(洩漏)兩版對照;
    策略(B)的 tilt 用「決策日 d 的擁擠度」→ d 收盤決策時其實還沒揭露 → 算輕微洩漏(第一輪允許)。

用法: uv run python scripts/aetf_contrarian.py
"""
from __future__ import annotations
import sys, json, importlib.util, math
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
# 資料只 2025-06-17~2026-06-17 → 只有 2025下-26 一段 regime 可得;仍切出來標明
REGIMES = {"2025下-26": ("2025-07-01", "2026-06-08")}
TOPN = 4
CSV = ROOT / "data/Active_ETF_1Y_Daily_28ETFs.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 擁擠度指標 (每 date×stock 跨ETF聚合)
# ─────────────────────────────────────────────────────────────────────────────
def build_crowd():
    df = pd.read_csv(CSV, encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Stock_Code"] = df["Stock_Code"].astype(str)

    def _disp(s):
        if len(s) < 2 or s.mean() == 0:
            return 0.0
        return float(s.std() / s.mean())   # 變異係數=經理人押注分歧度

    g = df.groupby(["Date", "Stock_Code"]).agg(
        n_hold=("ETF_Code", "nunique"),
        sum_wt=("Weight(%)", "sum"),
        max_wt=("Weight(%)", "max"),
        wt_disp=("Weight(%)", _disp),
        net_chg=("Daily_Change(%)", "sum"),
    ).reset_index()
    # crowd: {date: {code: {metric: val}}}
    crowd = defaultdict(dict)
    for _, r in g.iterrows():
        crowd[r["Date"]][r["Stock_Code"]] = {
            "sum_wt": r["sum_wt"], "n_hold": r["n_hold"],
            "max_wt": r["max_wt"], "wt_disp": r["wt_disp"], "net_chg": r["net_chg"],
        }
    return crowd, sorted(df["Stock_Code"].unique())


def pctile_map(crowd, metric, dates, codes):
    """{date: {code: 橫斷面百分位 0~1}}。pct 高=該指標數值高(越擁擠)。"""
    out = {}
    for d in dates:
        day = crowd.get(d, {})
        vals = [(c, day[c][metric]) for c in codes if c in day]
        vals.sort(key=lambda x: x[1])
        n = len(vals)
        out[d] = {c: (i + 1) / n for i, (c, _) in enumerate(vals)} if n else {}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# (A) 探針: rank-IC + 分位 long-short (yfinance 還原價,獨立於策略引擎)
# ─────────────────────────────────────────────────────────────────────────────
def probe(crowd, codes):
    import yfinance as yf
    logger.info("探針:下載還原含息價(yfinance auto_adjust)...")
    px = {}
    for c in codes + ["0050"]:
        try:
            s = yf.download(f"{c}.TW", start="2025-05-01", auto_adjust=True, progress=False)["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.dropna()
            if len(s) > 20:
                px[c] = s
        except Exception:
            pass
    bench = px.get("0050")
    logger.info(f"  有價 {len(px)-1}/{len(codes)}")

    rows = []
    for d, day in crowd.items():
        ts = pd.Timestamp(d)
        for c, m in day.items():
            rows.append({"Date": ts, "code": c, **m})
    g = pd.DataFrame(rows)

    def fwd_excess(code, ts, n, leak):
        """leak=True: 含當日(D0買進=時點洩漏);leak=False: D+1起(leak-free)。"""
        s = px.get(code)
        if s is None or bench is None:
            return None
        if leak:
            si = s.index[s.index >= ts]; bi = bench.index[bench.index >= ts]
        else:
            si = s.index[s.index > ts]; bi = bench.index[bench.index > ts]
        if len(si) <= n or len(bi) <= n:
            return None
        return (float(s.loc[si[n]] / s.loc[si[0]] - 1) - float(bench.loc[bi[n]] / bench.loc[bi[0]] - 1)) * 100

    horizons = (5, 10, 20)
    lines = []
    for leak in (False, True):
        tag = "D0洩漏(作弊)" if leak else "D+1(leak-free)"
        for n in horizons:
            g[f"fwd{n}_{leak}"] = [fwd_excess(c, d, n, leak) for c, d in zip(g["code"], g["Date"])]
        lines.append(f"\n=== rank-IC ({tag}) :擁擠度 vs 前向超額(扣0050) ===")
        lines.append(f"{'metric':<10} " + " ".join(f"fwd{n:<4}" for n in horizons))
        ic_tbl = {}
        for sig in ["sum_wt", "n_hold", "max_wt", "wt_disp", "net_chg"]:
            cells = []
            for n in horizons:
                sub = g.dropna(subset=[f"fwd{n}_{leak}", sig])
                ic = sub[sig].corr(sub[f"fwd{n}_{leak}"], method="spearman")
                cells.append(ic); ic_tbl[(sig, n)] = ic
            lines.append(f"{sig:<10} " + "  ".join(f"{c:+.3f}" for c in cells))
        # 分位 long-short: contrarian = long 最不擁擠 / short 最擁擠
        lines.append(f"--- 分位 long-short ({tag}) contrarian: long bottom20%擁擠 − short top20%擁擠 ---")
        COST = 0.6
        for sig in ["sum_wt", "n_hold", "max_wt"]:
            for n in (5, 10):
                lo_rets, hi_rets = [], []
                for ts, day in g.groupby("Date"):
                    day = day.dropna(subset=[f"fwd{n}_{leak}"])
                    if len(day) < 10:
                        continue
                    k = max(1, len(day) // 5)
                    hi = day.nlargest(k, sig)    # 最擁擠
                    lo = day.nsmallest(k, sig)   # 最不擁擠
                    hi_rets.append(hi[f"fwd{n}_{leak}"].mean())
                    lo_rets.append(lo[f"fwd{n}_{leak}"].mean())
                Lf = np.mean(lo_rets) - COST   # 多最不擁擠
                Sf = np.mean(hi_rets) - COST   # 空最擁擠(報酬=-Sf)
                ls = Lf - Sf
                lines.append(f"  {sig:<8} 持{n}日: long不擁擠 {Lf:+.2f}%  short擁擠端報酬 {-Sf:+.2f}%  "
                             f"contrarian LS {ls:+.2f}pp (天{len(lo_rets)})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# (B) 策略: H+反彈雙引擎候選 + contrarian tilt
# ─────────────────────────────────────────────────────────────────────────────
def build_candidates(codes, feats, twii_feat, reb_cache, turn_pct, sig_days):
    regime_bull = {
        d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
        for d in sig_days
    }
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
    crowd, etf_codes = build_crowd()
    logger.info(f"擁擠度:{len(crowd)} 日 × ~{len(etf_codes)} 池股")

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    names = {c: u[c].get("name", c) for c in codes}

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
    n_cand = sum(len(v) for v in cands.values())
    logger.info(f"候選 {n_cand} 筆(每日 top-{TOPN})")

    # 擁擠度百分位(用於 tilt)。預設用 sum_wt(已知 IC<0,最該被 fade)
    crowd_pct = {m: pctile_map(crowd, m, sig_days, etf_codes)
                 for m in ["sum_wt", "n_hold", "max_wt"]}

    # (A) 探針
    probe_txt = probe(crowd, etf_codes)
    logger.info("探針完成")

    # 候選股擁擠度覆蓋率(候選股有幾個在 ETF 池內 → tilt 才生效)
    pairs = [(d, c) for d in sig_days for _, c in cands[d]]
    cover = sum(1 for d, c in pairs if c in crowd_pct["sum_wt"].get(d, {}))
    logger.info(f"候選擁擠度覆蓋 {cover}/{len(pairs)} = {cover/max(1,len(pairs))*100:.0f}% "
                f"(沒覆蓋的 tilt=1 中性)")

    # (B) tilt 回測
    def rows_for(metric, k, day_set):
        """contrarian: edge × (1 + k·(0.5 − crowd_pctile))。crowd 高→縮,crowd 低→放大。"""
        pm = crowd_pct[metric]
        rw = []
        for d in sig_days:
            if d not in day_set:
                continue
            for v, c in cands[d]:
                e = v / 100.0
                if k == 0.0:
                    rw.append((d, c, e)); continue
                p = pm.get(d, {}).get(c)
                if p is None:
                    rw.append((d, c, e)); continue   # 池外→中性
                mult = max(0.0, 1.0 + k * (0.5 - p))
                rw.append((d, c, e * mult))
        return rw

    def run_sim(rw):
        return sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)

    KS = [0.3, 0.6, 1.0, 1.5]
    configs = [("baseline(無tilt)", "sum_wt", 0.0)]
    for m in ["sum_wt", "n_hold", "max_wt"]:
        for k in KS:
            configs.append((f"fade_{m}×{k}", m, k))

    # 0050 基準
    def bench_for(day_set):
        dd = sorted(d for d in day_set if d in closes["0050"])
        return v6.bench_0050(opens["0050"], closes["0050"], dd)
    bench_win = {wl: bench_for(set(sig_days[-n:])) for wl, n in WINDOWS}
    regime_days = {rn: set(d for d in sig_days if s <= d <= e) for rn, (s, e) in REGIMES.items()}
    bench_reg = {rn: bench_for(dd) for rn, dd in regime_days.items()}

    res = {}
    for label, metric, k in configs:
        for wl, n in WINDOWS:
            res[(label, wl)] = run_sim(rows_for(metric, k, set(sig_days[-n:])))
        for rn, dd in regime_days.items():
            res[(label, rn)] = run_sim(rows_for(metric, k, dd))
        logger.info(f"{label} 完成")

    # ── 報告 ──
    wlabels = [wl for wl, _ in WINDOWS]
    rlabels = list(REGIMES.keys())
    allcols = wlabels + rlabels

    def alpha(label, col):
        r = res.get((label, col))
        bench = bench_win.get(col, bench_reg.get(col))
        return (r["ret"] - bench) if (r and bench is not None) else None

    longest = "1年"
    L = [
        "# #6 反向擁擠 (Contrarian / Crowding-fade) — 主動式ETF\n",
        f"> 候選 H+反彈雙引擎每日 top-{TOPN}｜⑤買收賣開執行｜結束 {END}｜{len(codes)} 檔池｜還原含息價\n",
        f"> contrarian tilt = edge × (1 + k·(0.5 − 擁擠度橫斷面百分位));baseline=k0 純引擎\n",
        f"> 成本 買0.1425%/賣0.4425%+滑價0.1%+漲停買不到｜ALPHA = 策略 − 同資金 DCA 0050\n",
        f"> 候選擁擠度覆蓋 {cover}/{len(pairs)} ({cover/max(1,len(pairs))*100:.0f}%);資料僅 1 個多頭年→無空頭,結論有天花板\n",
        "> ⚠️ 作弊: in-sample 選指標 + 當前ETF成員(survivorship) + tilt用決策日擁擠度(輕微時點洩漏)\n",
        "> 0050 基準: " + " ".join(f"{wl}{bench_win[wl]:+.0f}%" for wl in wlabels)
        + " | " + " ".join(f"{rn}{bench_reg[rn]:+.0f}%" for rn in rlabels) + "\n",
        "## ALPHA %(扣 0050)\n",
        "| 變體 | " + " | ".join(allcols) + " | 最差 | 平均 | 換手 | 持股 | 曝險 |",
        "|---|" + "|".join(["---"] * (len(allcols) + 5)) + "|",
    ]
    for label, _, _ in configs:
        cells, avals = [], []
        for col in allcols:
            a = alpha(label, col)
            cells.append(f"{a:+.0f}" if a is not None else "—")
            if a is not None:
                avals.append(a)
        worst = f"{min(avals):+.0f}" if avals else "—"
        mean = f"{sum(avals)/len(avals):+.0f}" if avals else "—"
        rL = res.get((label, longest))
        turn = f"{rL['turn']:.0f}x" if rL else "—"
        pos = f"{rL.get('avg_pos', 0):.1f}" if rL else "—"
        expo = f"{rL.get('avg_expo', 0)*100:.0f}%" if rL else "—"
        L.append(f"| {label} | " + " | ".join(cells) + f" | **{worst}** | {mean} | {turn} | {pos} | {expo} |")

    L += ["", "## 原始報酬 %(未扣大盤)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    for label, _, _ in configs:
        cells = [f"{res[(label,col)]['ret']:+.0f}" if res.get((label, col)) else "—" for col in allcols]
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += ["", "## Uplift = tilt alpha − baseline alpha (pp, 逐窗)\n",
          "| 變體 | " + " | ".join(allcols) + " |",
          "|---|" + "|".join(["---"] * len(allcols)) + "|"]
    base_a = {col: alpha("baseline(無tilt)", col) for col in allcols}
    for label, _, _ in configs:
        if label.startswith("baseline"):
            continue
        cells = []
        for col in allcols:
            a = alpha(label, col); b = base_a.get(col)
            cells.append(f"{a-b:+.0f}" if (a is not None and b is not None) else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += ["", "## (A) 探針: 擁擠度 rank-IC + 分位 long-short\n", "```", probe_txt, "```\n",
          "## 判讀\n",
          "- 探針若 fade 擁擠 IC>0 / contrarian LS>0 且 D+1 仍成立 → 有 edge 苗頭。",
          "- tilt uplift 全正且隨 k 單調 → contrarian 在引擎上加值;否則無 edge。",
          "- 曝險欄需與 baseline 接近才是公平比較。"]
    RPT = ROOT / "reports" / "aetf_contrarian.md"
    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    # console 摘要(供結構化輸出)
    print("\n### 摘要(供回報) ###")
    for label, _, _ in configs:
        cells = []
        for col in allcols:
            a = alpha(label, col)
            cells.append(f"{col}{a:+.0f}" if a is not None else f"{col}—")
        up = ""
        if not label.startswith("baseline"):
            ups = [alpha(label, c) - base_a[c] for c in allcols
                   if alpha(label, c) is not None and base_a[c] is not None]
            up = f" | upliftΣ{sum(ups):+.0f}" if ups else ""
        print(f"{label:<18} " + " ".join(cells) + up)
    print(probe_txt)


if __name__ == "__main__":
    main()
