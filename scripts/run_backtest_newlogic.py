"""60D 回測 —— 用『這版新邏輯』全部重跑(不重用舊 498,不同邏輯),理想版(無手續費)。

  換腦不換錢包:
    第一層(訊號)= 新邏輯:每個決策日 FinMind 篩 base_universe 112 (_quant_signals)
                  → PASS → run_debate(stock, [], historical_date=決策日)(現行 bear.py)
    第二層(資金)= 沿用舊 60D 的 _paper_trade:期初 15000、每日 +1000、投入上限 50000、
                  edge 加權再平衡、最多 3 檔、曝險 30-90%(與舊報告同一段程式碼 → 資金模型一致)

  全部新跑、快取到 data/_btnew_llm.csv(可續跑),輸出 reports/llm_backtest_60d_NEW.md。

用法: python scripts/run_backtest_newlogic.py            # 近 60 交易日(對齊舊 60D 窗口)
      python scripts/run_backtest_newlogic.py --days 90  # 90 天
"""
from __future__ import annotations
import argparse, csv, sys, importlib.util, json, math
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.debate.bear import run_debate
from tw_stock_agent.screener.patterns import analyze_pattern

def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
_bom = _load("bom", "scripts/backtest_one_month.py")
_bwl = _load("bwl", "scripts/backtest_with_llm.py")
_r60 = _load("r60", "scripts/run_backtest_60d.py")
_quant_signals = _bom._quant_signals
_stock_from_row = _bwl._stock_from_row
_paper_trade = _r60._paper_trade

END_DEFAULT = "2026-06-08"
CACHE_CSV = DATA_DIR / "_btnew_llm.csv"
REPORT = ROOT / "reports" / "llm_backtest_60d_NEW.md"
CACHE_HDR = ["date","ticker","name","pass_level","pattern_type","vol_ratio","rs_20d","ma5_gt_ma20",
             "close_price","bear_score","bull_score","evidence_level","llm_verdict",
             "predicted_direction","predicted_low_pct","predicted_high_pct","predicted_center_pct",
             "prediction_confidence","return_1d"]

_OH: dict[str, dict] = {}
def oh(tk):
    if tk not in _OH:
        try: _OH[tk] = get_daily_ohlcv(tk) or {}
        except Exception: _OH[tk] = {}
    return _OH[tk]

def build_df(tk):
    o = oh(tk)
    if not o: return None
    ds = sorted(o)
    return pd.DataFrame({
        "Open":[o[d]["open"] for d in ds], "High":[o[d]["high"] for d in ds],
        "Low":[o[d]["low"] for d in ds], "Close":[o[d]["close"] for d in ds],
        "Volume":[o[d].get("volume",0) for d in ds],
    }, index=pd.to_datetime(ds))

def fwd_ret1(tk, d_str):
    o = oh(tk); ds = sorted(o)
    later = [x for x in ds if x > d_str]; le = [x for x in ds if x <= d_str]
    if not later or not le: return ""
    return f"{(o[later[0]]['close']/o[le[-1]]['close']-1)*100:.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--end", default=END_DEFAULT)
    args = ap.parse_args()

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()) if isinstance(u, dict) else list(u)
    names = {c: (u[c].get("name", c) if isinstance(u, dict) else c) for c in codes}
    logger.info(f"股票池 {len(codes)} 檔(base_universe)｜窗口 末 {args.days} 交易日(≤{args.end})")

    twii = build_df("0050")
    cal = [d.strftime("%Y-%m-%d") for d in twii.index if d.strftime("%Y-%m-%d") <= args.end]
    window = cal[-args.days:]
    logger.info(f"決策日 {window[0]} ~ {window[-1]}({len(window)} 天)")

    # ── 篩選(全 112 × 每日)──
    dfs = {c: build_df(c) for c in codes}
    cand = []   # 每筆 = (date, code, row_for_stock)
    for d in window:
        as_of = date.fromisoformat(d)
        for c in codes:
            df = dfs.get(c)
            if df is None: continue
            q = _quant_signals(df, twii, as_of)
            if not q or q["pass_level"] != "PASS": continue
            df_slice = q.pop("_df")
            pat = analyze_pattern({"code": c, "name": names[c], "yf_ticker": f"{c}.TW"}, df_slice)
            cand.append({"date": d, "ticker": c, "name": names[c], "pass_level": "PASS",
                         "pattern_type": pat.pattern_type if pat else "none",
                         "vol_ratio": f"{q['vol_ratio']:.2f}", "rs_20d": f"{q['rs_20d']:.3f}",
                         "ma5_gt_ma20": str(q["ma5_gt_ma20"]), "close_price": f"{q['close_price']:.1f}"})
    logger.info(f"PASS 候選 {len(cand)} 筆(平均 {len(cand)/len(window):.1f}/天)→ 進辯論")

    # ── 載入 cache(可續跑)──
    done = {}
    if CACHE_CSV.exists():
        for r in csv.DictReader(CACHE_CSV.open(encoding="utf-8")):
            done[f"{r['date']}_{r['ticker']}"] = r
    todo = [c for c in cand if f"{c['date']}_{c['ticker']}" not in done]
    logger.info(f"cache 已有 {len(done)}｜本次需辯論 {len(todo)}")

    # ── 辯論(並行 + 定期 flush)──
    new_rows = []
    def one(row):
        try:
            res = run_debate(_stock_from_row(row), [], historical_date=date.fromisoformat(row["date"]))
        except Exception as e:
            logger.warning(f"  {row['date']} {row['ticker']} 失敗: {str(e)[:50]}"); return None
        return {**row, "bear_score": res.bear_score, "bull_score": res.bull_score,
                "evidence_level": res.evidence_level, "llm_verdict": res.verdict,
                "predicted_direction": res.predicted_direction,
                "predicted_low_pct": f"{res.predicted_low_pct:.2f}",
                "predicted_high_pct": f"{res.predicted_high_pct:.2f}",
                "predicted_center_pct": f"{res.predicted_center_pct:.2f}",
                "prediction_confidence": f"{res.prediction_confidence:.2f}",
                "return_1d": fwd_ret1(row["ticker"], row["date"])}
    win_set = set(window)
    def write_report():
        rows = ([r for r in csv.DictReader(CACHE_CSV.open(encoding="utf-8")) if r["date"] in win_set]
                if CACHE_CSV.exists() else [])
        if not rows:
            return None
        pt = _paper_trade(rows)
        def dc(r):
            try: r1 = float(r.get("return_1d", "") or "x")
            except (ValueError, TypeError): return None
            dirn = r.get("predicted_direction", "")
            if dirn == "up": return r1 > 0
            if dirn == "down": return r1 < 0
            return None
        judged = [x for x in (dc(r) for r in rows) if x is not None]
        acc = sum(judged) / len(judged) * 100 if judged else 0.0
        ndays = len({r["date"] for r in rows})
        L = [f"# LLM {args.days} 天回測報告（新邏輯·理想版無手續費）\n",
             f"> 窗口 {window[0]}~{window[-1]}｜股票池 base_universe {len(codes)} 檔｜"
             f"新邏輯(FinMind 篩 + 現行辯論)｜資金模型同舊 60D(_paper_trade)｜"
             f"**邊跑邊更新:已產 {len(rows)} 筆訊號 / {ndays} 天**\n",
             "## 對照舊 60D(不同邏輯,僅參考)\n",
             "| 指標 | 新邏輯 | 舊 60D |", "|------|--------|--------|",
             f"| 訊號數 | {len(rows)} | 498 |",
             f"| 方向準確率 | **{acc:.1f}%**（{sum(judged)}/{len(judged)}）| 65.3% |",
             f"| 本金報酬率 | **{pt['return_pct']:+.2f}%** | +41.84% |",
             f"| 期末權益 | {pt['final_equity']:,.0f} | 70,922 |",
             f"| 淨損益 | {pt['total_pnl']:+,.0f} | +20,922 |",
             f"| 獲利日勝率 | {pt['win_days']}/{pt['total_days']} | 33/56 |",
             f"| 最大回撤 | -{pt['max_dd']:,.0f} | -1,886 |",
             f"| 年化 Sharpe | {pt['sharpe']:.2f} | 5.43 |", "",
             "## 資金(期初 15000 + 每日 1000,投入上限 50000)\n", "| 指標 | 數值 |", "|------|------|",
             f"| 累計投入 | {pt['total_contributed']:,.0f} TWD |",
             f"| 期末權益 | {pt['final_equity']:,.0f} TWD |",
             f"| **淨損益 / 報酬率** | **{pt['total_pnl']:+,.0f} / {pt['return_pct']:+.2f}%** |",
             f"| 交易日數 | {pt['calendar_days']} |", ""]
        L += ["<details><summary>每日持倉明細</summary>\n",
              "| 日期 | 持股 | 投入 | 曝險 | 現金 | 當日P&L | 權益 |",
              "|------|------|------|------|------|---------|------|"]
        for t in pt["ledger"]:
            L.append(f"| {t['date']} | {t['holdings']} | {t['invested']:,.0f} | {t['exposure']:.0%} "
                     f"| {t['cash']:,.0f} | {t['pnl']:+,.0f} | {t['equity']:,.0f} |")
        L.append("</details>\n")
        REPORT.write_text("\n".join(L), encoding="utf-8")
        return pt, acc

    def flush():
        exists = CACHE_CSV.exists()
        with CACHE_CSV.open("a" if exists else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CACHE_HDR, extrasaction="ignore")
            if not exists: w.writeheader()
            w.writerows(new_rows)
        new_rows.clear()
        write_report()                      # 邊跑邊寫 MD(同舊版行為)

    write_report()                          # 開跑前先用既有 cache 寫一版
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, r in enumerate(ex.map(one, todo), 1):
                if r: new_rows.append(r)
                if i % 25 == 0:
                    logger.info(f"  辯論 {i}/{len(todo)} → 已更新 MD"); flush()
        flush()

    res = write_report()
    if res:
        pt, acc = res
        logger.success(f"報告 → {REPORT}")
        logger.success(f"新邏輯 {args.days}D:報酬 {pt['return_pct']:+.2f}% 方向準 {acc:.1f}% "
                       f"權益 {pt['final_equity']:,.0f}(舊 60D:+41.84% / 65.3% / 70,922)")

if __name__ == "__main__":
    main()