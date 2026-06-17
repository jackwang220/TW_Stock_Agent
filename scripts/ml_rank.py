"""ML 選股 baseline (leak-safe walk-forward gradient boosting).

軌道: ML/RL 第1步。建一個誠實、防洩漏的機器學習選股 baseline,跟現有 H 雙引擎比。

設計:
  特徵 : 重用 v3.features / v3._factors 的技術特徵(全部 rolling,只用當日及之前的 bar)。
         每檔每日一個特徵向量 + 大盤 regime 欄位。
  目標 : forward N 日「收盤→收盤」報酬,做成「橫截面排序」的回歸目標(每日 z-score)。
  切分 : walk-forward expanding window。訓練只用「目標已完全實現且實現日 < 測試折起點」
         的樣本(purge gap = N 個交易日),嚴禁未來資料訓練過去。
  模型 : sklearn HistGradientBoostingRegressor。
  評估 : (a) OOS 每日 rank-IC(Spearman, 模型分數 vs 真實 forward 報酬)
         (b) OOS 分桶(decile)單調性
         (c) 把模型分數當 edge 餵進 ⑤(sim_buyclose_sellopen)回測框架,扣成本後 alpha vs 0050
誠實對照: H 雙引擎(exp_60d_entry_compare 的訊號)在同窗口、同 ⑤ 引擎下的 alpha。

用法: uv run python scripts/ml_rank.py
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal

from sklearn.ensemble import HistGradientBoostingRegressor


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v3  = _load("v3",  ROOT / "scripts/exp_step1_v3.py")
v6  = _load("v6",  ROOT / "scripts/exp_step1_v6.py")
ec  = _load("ec",  ROOT / "scripts/exp_60d_entry_compare.py")
r60 = _load("r60", ROOT / "scripts/run_backtest_60d.py")

features, _factors, oh = v3.features, v3._factors, v3.oh

END = "2026-06-08"
HORIZONS = [5, 20]                 # forward N 日目標
N_FOLDS = 5                        # walk-forward 折數(在可用區間上等分測試段)
MIN_TRAIN = 120                    # 第一折前至少要有這麼多訓練日
FEATURE_NAMES = [
    # _factors 的 7 維(部分)
    "f_trend", "f_rs", "f_vol", "f_rsi", "f_macd", "f_break", "bias",
    # 原始/衍生
    "ret20", "volr", "rsi", "macdh", "gap", "vcp",
    "dist_ma20", "dist_ma60", "ma20_slope", "turn_pct",
    # 大盤 regime(對所有股票同日相同,讓模型學 regime 交互)
    "idx_ret20", "idx_bull",
]


def build_panel(codes, feats, twii_feat, turn_pct, cal):
    """組出 long-format 特徵表: 一列 = (date, ticker, 特徵...)。全部 point-in-time。"""
    rows = []
    # 預先算 0050 的 MA20 斜率代理 (idx)
    for d in cal:
        tf = twii_feat.get(d, {})
        ir = tf.get("ret20")
        idx_ret20 = ir if (ir is not None and not (isinstance(ir, float) and math.isnan(ir))) else 0.0
        idx_bull = 1.0 if (tf.get("close") and tf.get("ma20") and tf["close"] > tf["ma20"]) else 0.0
        for c in codes:
            f = feats.get(c, {})
            if d not in f:
                continue
            fd = f[d]
            ma20 = fd.get("ma20", float("nan"))
            if ma20 is None or (isinstance(ma20, float) and math.isnan(ma20)):
                continue
            ff = _factors(fd, ir)  # 可能 None(MA5<=MA20 或 rsi nan)
            if ff is None:
                f_trend = f_rs = f_vol = f_rsi = f_macd = f_break = 0.0
                bias = fd["close"] / ma20 - 1 if ma20 else 0.0
            else:
                f_trend, f_rs, f_vol, f_rsi, f_macd, f_break, bias = ff
            close = fd["close"]
            ma60 = fd.get("ma60", float("nan"))
            ma5 = fd.get("ma5", float("nan"))
            def _safe(x, default=0.0):
                return default if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)
            rows.append({
                "date": d, "ticker": c,
                "f_trend": _safe(f_trend), "f_rs": _safe(f_rs), "f_vol": _safe(f_vol),
                "f_rsi": _safe(f_rsi), "f_macd": _safe(f_macd), "f_break": _safe(f_break),
                "bias": _safe(bias),
                "ret20": _safe(fd.get("ret20")),
                "volr": _safe(fd.get("volr")),
                "rsi": _safe(fd.get("rsi")),
                "macdh": _safe(fd.get("macdh")),
                "gap": _safe(fd.get("gap")),
                "vcp": 1.0 if fd.get("vcp") else 0.0,
                "dist_ma20": _safe(close / ma20 - 1) if ma20 else 0.0,
                "dist_ma60": _safe(close / ma60 - 1) if (ma60 and not math.isnan(ma60)) else 0.0,
                "ma20_slope": _safe(ma5 / ma20 - 1) if (ma20 and ma5 and not math.isnan(ma5)) else 0.0,
                "turn_pct": turn_pct.get(d, {}).get(c, 0.5),
                "idx_ret20": idx_ret20,
                "idx_bull": idx_bull,
            })
    return pd.DataFrame(rows)


def add_forward_target(df, closes, horizon):
    """forward N 日 close-to-close 報酬。只用 d 與 d+N 的收盤,不洩漏。
    target_date(實現日)記錄下來,供 purge 用。"""
    # 建每檔的交易日序列與索引
    fwd = []
    real_date = []
    by_tk_dates = {tk: sorted(closes.get(tk, {})) for tk in df["ticker"].unique()}
    pos = {tk: {d: i for i, d in enumerate(ds)} for tk, ds in by_tk_dates.items()}
    for r in df.itertuples(index=False):
        tk, d = r.ticker, r.date
        ds = by_tk_dates.get(tk, [])
        i = pos.get(tk, {}).get(d)
        if i is None or i + horizon >= len(ds):
            fwd.append(np.nan); real_date.append(None); continue
        d2 = ds[i + horizon]
        c0 = closes[tk].get(d); c1 = closes[tk].get(d2)
        if not c0 or not c1 or c0 <= 0:
            fwd.append(np.nan); real_date.append(None); continue
        fwd.append(c1 / c0 - 1); real_date.append(d2)
    out = df.copy()
    out["fwd_ret"] = fwd
    out["real_date"] = real_date
    return out


def xsec_zscore(df, col="fwd_ret"):
    """橫截面(每日)z-score 當回歸目標,讓模型學的是『當天相對排序』而非市場 beta。"""
    z = df.groupby("date")[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    return z


def spearman(a, b):
    a = pd.Series(a); b = pd.Series(b)
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return a.rank().corr(b.rank())


def walk_forward(df, horizon):
    """expanding-window walk-forward。回傳:
       preds_df: OOS 預測(date,ticker,pred,fwd_ret), daily_ic list, decile means。"""
    df = df.dropna(subset=["fwd_ret", "real_date"]).copy()
    df["y"] = xsec_zscore(df, "fwd_ret")
    dates = sorted(df["date"].unique())
    if len(dates) < MIN_TRAIN + N_FOLDS * 5:
        logger.warning(f"H{horizon}: 樣本日數 {len(dates)} 偏少")
    # 測試段: 把 [MIN_TRAIN:] 等分成 N_FOLDS 段
    test_dates = dates[MIN_TRAIN:]
    fold_bounds = np.array_split(test_dates, N_FOLDS)
    preds = []
    feat_cols = FEATURE_NAMES
    for fi, fold in enumerate(fold_bounds):
        if len(fold) == 0:
            continue
        test_start = fold[0]
        # purge: 訓練樣本的『實現日』必須 < test_start(嚴格小於 → 標籤不與測試期重疊)
        train = df[df["real_date"] < test_start]
        test = df[df["date"].isin(set(fold))]
        if len(train) < 200 or len(test) == 0:
            continue
        Xtr = train[feat_cols].to_numpy(); ytr = train["y"].to_numpy()
        Xte = test[feat_cols].to_numpy()
        model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=42,
        )
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        t = test[["date", "ticker", "fwd_ret"]].copy()
        t["pred"] = p
        t["fold"] = fi
        preds.append(t)
    if not preds:
        return None
    pdf = pd.concat(preds, ignore_index=True)
    # 每日 rank-IC
    ic_by_day = pdf.groupby("date").apply(
        lambda g: spearman(g["pred"], g["fwd_ret"]), include_groups=False
    ).dropna()
    # decile 單調性(整體 OOS pooled,用 pred 分 10 桶看 fwd_ret 均值)
    pdf2 = pdf.dropna(subset=["fwd_ret"]).copy()
    try:
        pdf2["bucket"] = pd.qcut(pdf2["pred"].rank(method="first"), 10, labels=False)
    except Exception:
        pdf2["bucket"] = pd.qcut(pdf2["pred"], 10, labels=False, duplicates="drop")
    decile = pdf2.groupby("bucket")["fwd_ret"].mean()
    return {"preds": pdf, "ic": ic_by_day, "decile": decile}


def build_ml_rows(pred_df, topn=4):
    """把 OOS 模型分數轉成 ⑤ 回測引擎吃的 rows=[(date,ticker,edge)]。
       與 H baseline 對齊: 每日只取分數最高的 topn 檔(⑤引擎自己再壓到最多3+1檔)。
       edge = 當日橫截面 min-max 正規化到 (0,1](僅在被選的 topn 內正規化)。"""
    rows = []
    for d, g in pred_df.groupby("date"):
        g = g.sort_values("pred", ascending=False).head(topn)
        lo, hi = g["pred"].min(), g["pred"].max()
        rng = hi - lo
        for r in g.itertuples(index=False):
            e = 0.5 + 0.5 * ((r.pred - lo) / rng) if rng > 1e-9 else 0.7   # 0.5..1.0
            rows.append((d, r.ticker, e))
    return rows


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info(f"載入特徵({len(codes)} 檔)...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}

    opens, closes = {}, {}
    for c in codes + ["0050"]:
        o = oh(c)
        opens[c] = {d: o[d]["open"] for d in o}
        closes[c] = {d: o[d]["close"] for d in o}

    cal = [d for d in sorted(twii_feat) if d <= END][-504:]   # 全史 2 年
    logger.info(f"回測日曆: {cal[0]} ~ {cal[-1]} ({len(cal)} 日)")

    # 成交值百分位 + 漲停 + 反彈(供 H baseline)
    turn_pct = {}
    for d in cal:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    limitup, reb_cache = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END)
        s = set(); cl_list = []; m = {}
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
            if j > 0 and o[ds[j - 1]]["close"] > 0 and o[d]["close"] / o[ds[j - 1]]["close"] - 1 >= 0.095:
                s.add(d)
        limitup[c] = s; reb_cache[c] = m

    logger.info("組特徵 panel...")
    panel = build_panel(codes, feats, twii_feat, turn_pct, cal)
    logger.info(f"panel 列數 {len(panel)}")

    # ── H 雙引擎 baseline 訊號(同 exp_60d_entry_compare,用於同窗對照)──
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in cal}
    h_rows_all = []
    for d in cal:
        ir = twii_feat.get(d, {}).get("ret20")
        bull = regime_bull.get(d)
        sc = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            if bull:
                v = ec.h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            else:
                v = reb_cache.get(c, {}).get(d, 0.0)
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:4]:
            h_rows_all.append((d, c, v / 100))

    results = {}
    L = ["# ML 選股 baseline (leak-safe walk-forward GBT) vs H 雙引擎\n",
         f"> 結束 {END}｜{len(codes)} 檔｜2年全史｜walk-forward {N_FOLDS} 折(expanding, purge=H天)\n",
         "> 目標=forward N 日報酬的橫截面 z-score｜模型=HistGradientBoostingRegressor\n",
         "> 回測引擎=⑤(買收盤/賣開盤),手續費買0.14%/賣0.44%+滑價0.1%,漲停買不到\n",
         "> alpha = 策略本金報酬 − 0050 同期 DCA(開盤)\n"]

    for H in HORIZONS:
        logger.info(f"=== Horizon {H} 日 ===")
        dfH = add_forward_target(panel, closes, H)
        wf = walk_forward(dfH, H)
        if wf is None:
            logger.warning(f"H{H}: walk-forward 無結果")
            continue
        ic = wf["ic"]
        ic_mean = ic.mean(); ic_std = ic.std(ddof=0)
        icir = ic_mean / ic_std * math.sqrt(252 / H) if ic_std > 0 else 0.0
        hit = (ic > 0).mean()
        decile = wf["decile"]
        # decile 單調性: Spearman(bucket_index, bucket_mean)
        mono = spearman(decile.index.to_numpy(), decile.to_numpy())
        top_bot = decile.iloc[-1] - decile.iloc[0]

        # OOS 期間(用於對照 H baseline,只在 OOS 日上回測)
        oos_dates = sorted(wf["preds"]["date"].unique())
        oos_set = set(oos_dates)
        oos_cal = [d for d in cal if d in oos_set]

        # ML edge → ⑤ 回測
        ml_rows = build_ml_rows(wf["preds"])
        h_rows = [r for r in h_rows_all if r[0] in oos_set]
        bench = v6.bench_0050(opens["0050"], closes["0050"], oos_cal)

        ml_sim = ec.sim_buyclose_sellopen(ml_rows, opens, closes, limitup, switch_cost_mult=1.0)
        h_sim = ec.sim_buyclose_sellopen(h_rows, opens, closes, limitup, switch_cost_mult=1.0)

        # 換手紅旗檢查: 拉高換股門檻(switch_cost_mult=4)強迫低換手,看 alpha 是否仍在
        #   → 若高換手版 alpha 暴跌而低換手版輸 H, 代表原 alpha 是換手偷拉的(假象)。
        ml_sim_lo = ec.sim_buyclose_sellopen(ml_rows, opens, closes, limitup, switch_cost_mult=4.0)
        h_sim_lo = ec.sim_buyclose_sellopen(h_rows, opens, closes, limitup, switch_cost_mult=4.0)

        ml_ret = ml_sim["ret"] if ml_sim else float("nan")
        h_ret = h_sim["ret"] if h_sim else float("nan")
        ml_alpha = ml_ret - bench
        h_alpha = h_ret - bench
        ml_alpha_lo = (ml_sim_lo["ret"] - bench) if ml_sim_lo else float("nan")
        h_alpha_lo = (h_sim_lo["ret"] - bench) if h_sim_lo else float("nan")

        results[H] = dict(ic_mean=ic_mean, icir=icir, hit=hit, mono=mono, top_bot=top_bot,
                          ml_ret=ml_ret, h_ret=h_ret, bench=bench,
                          ml_alpha=ml_alpha, h_alpha=h_alpha,
                          ml_alpha_lo=ml_alpha_lo, h_alpha_lo=h_alpha_lo,
                          ml_turn=ml_sim["turn"] if ml_sim else float("nan"),
                          h_turn=h_sim["turn"] if h_sim else float("nan"),
                          ml_turn_lo=ml_sim_lo["turn"] if ml_sim_lo else float("nan"),
                          h_turn_lo=h_sim_lo["turn"] if h_sim_lo else float("nan"),
                          n_days=len(oos_cal), n_ml_rows=len(ml_rows))

        logger.info(f"H{H}: IC={ic_mean:+.4f} ICIR={icir:+.2f} hit={hit:.0%} mono={mono:+.2f} "
                    f"top-bot={top_bot*100:+.2f}% | ML alpha={ml_alpha:+.1f}%(turn{ml_sim['turn']:.0f}) "
                    f"H alpha={h_alpha:+.1f}%(turn{h_sim['turn']:.0f}) | "
                    f"低換手: ML{ml_alpha_lo:+.1f}%(turn{ml_sim_lo['turn']:.0f}) H{h_alpha_lo:+.1f}%(turn{h_sim_lo['turn']:.0f})")

        L += [f"## Horizon {H} 日\n",
              "### 排序預測力 (OOS)\n",
              "| 指標 | 值 |", "|---|---|",
              f"| 每日 rank-IC 均值 | {ic_mean:+.4f} |",
              f"| IC 年化 IR (ICIR) | {icir:+.2f} |",
              f"| IC>0 命中率 | {hit:.1%} |",
              f"| decile 單調性 (Spearman) | {mono:+.3f} |",
              f"| 第10桶−第1桶 fwd報酬 | {top_bot*100:+.2f}% |",
              "",
              "#### decile 平均 forward 報酬(由低分到高分桶,單調遞增=好)\n",
              "| 桶 | " + " | ".join(str(int(i)) for i in decile.index) + " |",
              "|---|" + "|".join(["---"] * len(decile)) + "|",
              "| fwd% | " + " | ".join(f"{x*100:+.2f}" for x in decile.values) + " |",
              "",
              "### 套進 ⑤ 回測(OOS 期間,扣成本)\n",
              f"> OOS 期間 {oos_cal[0]}~{oos_cal[-1]}({len(oos_cal)} 日)｜0050 DCA 同期 {bench:+.1f}%\n",
              "| 策略 | 本金報酬 | alpha vs 0050 | 換手x | 低換手版 alpha(換股門檻×4) | 低換手 換手x |",
              "|---|---|---|---|---|---|",
              f"| ML(GBT, H{H}) | {ml_ret:+.1f}% | {ml_alpha:+.1f}% | {results[H]['ml_turn']:.0f} | {ml_alpha_lo:+.1f}% | {results[H]['ml_turn_lo']:.0f} |",
              f"| H 雙引擎(同期) | {h_ret:+.1f}% | {h_alpha:+.1f}% | {results[H]['h_turn']:.0f} | {h_alpha_lo:+.1f}% | {results[H]['h_turn_lo']:.0f} |",
              "",
              "> ⚠️ 換手紅旗: 若高換手版 alpha 很高但低換手版崩掉,代表報酬靠高換手(+⑤隔夜溢價)偷拉,非真 edge。",
              ""]

    # 結論
    L += ["## 誠實結論\n"]
    win_any = False
    for H in HORIZONS:
        if H not in results:
            continue
        r = results[H]
        verdict = "勝過" if r["ml_alpha"] > r["h_alpha"] else "輸給"
        if r["ml_alpha"] > r["h_alpha"]:
            win_any = True
        L.append(f"- **H{H}**: ML alpha {r['ml_alpha']:+.1f}% vs H 雙引擎 {r['h_alpha']:+.1f}% "
                 f"→ ML {verdict} baseline。IC={r['ic_mean']:+.4f}(ICIR {r['icir']:+.2f}), "
                 f"decile 單調 {r['mono']:+.2f}。")
    L.append("")
    L.append(f"- **總評**: {'有至少一個 horizon ML 勝過 H' if win_any else 'ML 全 horizon 未勝過 H 雙引擎'}。")

    REPORT = ROOT / "reports" / "ml_rank.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    # 機器可讀摘要(最後一行 stdout)
    print("RESULT_JSON " + json.dumps({
        str(H): {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                 for k, v in results[H].items()} for H in results}))


if __name__ == "__main__":
    main()
