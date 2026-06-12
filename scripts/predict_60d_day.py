"""用『60D 正確邏輯』(= run_backtest_60d)對單一交易日產生股價預測。

  決策日 6/10 收盤 → 篩選 base_universe 112(_quant_signals,同 backtest_one_month)
  → PASS 的用 _stock_from_row + run_debate(stock, [], historical_date=6/10)
     (空 live 新聞 → 純 FinMind 歷史新聞/籌碼/營收/反彈注入,leak-safe)
  → 寫 signal_log 標交易日 6/11 → 之後跑 live_portfolio.py 更新帳本。

  跟 live 管線(run_daily_scan)不同:不抓即時新聞、不設 Top-25 上限。

用法: python scripts/predict_60d_day.py            # 決策 6/10 / 交易 6/11
      python scripts/predict_60d_day.py 2026-06-10 2026-06-11
"""
from __future__ import annotations
import sys, csv, importlib.util, json
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR, SIGNAL_LOG
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.debate.bear import run_debate
from tw_stock_agent.screener.patterns import analyze_pattern

def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, ROOT / relpath)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
_bom = _load("bom", "scripts/backtest_one_month.py")
_bwl = _load("bwl", "scripts/backtest_with_llm.py")
_quant_signals = _bom._quant_signals
_stock_from_row = _bwl._stock_from_row

DECISION = sys.argv[1] if len(sys.argv) > 1 else "2026-06-10"
TRADE    = sys.argv[2] if len(sys.argv) > 2 else "2026-06-11"
SIG_HEADER = ["date","ticker","name","source_news","bfs_depth","volume_ratio","ma5_gt_ma20",
              "rs_20d","pattern_type","rsi_14","bear_score","bull_score","evidence_level",
              "verdict","close_price","predicted_direction","predicted_low_pct","predicted_high_pct",
              "predicted_center_pct","prediction_confidence","return_1d","return_5d","return_10d",
              "return_20d","max_drawdown_10d"]

def build_df(code: str) -> pd.DataFrame | None:
    o = get_daily_ohlcv(code)
    if not o: return None
    ds = sorted(o)
    idx = pd.to_datetime(ds)
    return pd.DataFrame({
        "Open":   [o[d]["open"]  for d in ds],
        "High":   [o[d]["high"]  for d in ds],
        "Low":    [o[d]["low"]   for d in ds],
        "Close":  [o[d]["close"] for d in ds],
        "Volume": [o[d].get("volume", 0) for d in ds],
    }, index=idx)

def main():
    as_of = date.fromisoformat(DECISION)
    # 名稱對照(用 backtest_one_month.csv 的 name)
    name_map = {}
    bom_csv = DATA_DIR / "backtest_one_month.csv"
    if bom_csv.exists():
        for r in csv.DictReader(bom_csv.open(encoding="utf-8")):
            name_map.setdefault(r["ticker"], r.get("name", r["ticker"]))
    # base_universe 112
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()) if isinstance(u, dict) else list(u)
    twii = build_df("0050")   # RS 基準(FinMind,沿用 crossperiod 慣例)
    if twii is None:
        logger.error("0050 基準抓不到"); return

    # ── 篩選(完整 112)──
    logger.info(f"決策日 {DECISION} → 篩選 {len(codes)} 檔(base_universe)")
    passed = []
    for code in codes:
        df = build_df(code)
        if df is None: continue
        q = _quant_signals(df, twii, as_of)
        if not q or q["pass_level"] != "PASS":
            continue
        df_slice = q.pop("_df")
        nm = name_map.get(code, code)
        pat = analyze_pattern({"code": code, "name": nm, "yf_ticker": f"{code}.TW"}, df_slice)
        passed.append({
            "date": DECISION, "ticker": code, "name": nm, "pass_level": "PASS",
            "pattern_type": pat.pattern_type if pat else "none",
            "vol_ratio": f"{q['vol_ratio']:.2f}", "rs_20d": f"{q['rs_20d']:.3f}",
            "ma5_gt_ma20": str(q["ma5_gt_ma20"]), "close_price": f"{q['close_price']:.1f}",
        })
    logger.info(f"PASS {len(passed)} 檔 → 進辯論(60D 路徑:空 live 新聞 + historical_date)")

    # ── 辯論(並行,60D 呼叫)──
    def one(row):
        try:
            stock = _stock_from_row(row)
            res = run_debate(stock, [], historical_date=as_of)
        except Exception as e:
            logger.warning(f"  {row['ticker']} 辯論失敗: {str(e)[:60]}"); return None
        logger.info(f"  {row['ticker']} {row['name']} {res.verdict} "
                    f"{res.predicted_direction}({res.predicted_center_pct:+.1f}%) bear={res.bear_score}")
        return {
            "date": TRADE, "ticker": row["ticker"], "name": row["name"],
            "source_news": "", "bfs_depth": "", "volume_ratio": row["vol_ratio"],
            "ma5_gt_ma20": row["ma5_gt_ma20"], "rs_20d": row["rs_20d"],
            "pattern_type": row["pattern_type"], "rsi_14": "",
            "bear_score": res.bear_score, "bull_score": res.bull_score,
            "evidence_level": res.evidence_level, "verdict": res.verdict,
            "close_price": row["close_price"], "predicted_direction": res.predicted_direction,
            "predicted_low_pct": f"{res.predicted_low_pct:.2f}",
            "predicted_high_pct": f"{res.predicted_high_pct:.2f}",
            "predicted_center_pct": f"{res.predicted_center_pct:.2f}",
            "prediction_confidence": f"{res.prediction_confidence:.2f}",
            "return_1d": "", "return_5d": "", "return_10d": "", "return_20d": "",
            "max_drawdown_10d": "",
        }
    with ThreadPoolExecutor(max_workers=8) as ex:
        out = [r for r in ex.map(one, passed) if r]

    # ── 寫 signal_log(先清掉該交易日舊紀錄,冪等)──
    existing = []
    if SIGNAL_LOG.exists():
        existing = [r for r in csv.DictReader(SIGNAL_LOG.open(encoding="utf-8"))
                    if r.get("date") != TRADE]
    with SIGNAL_LOG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIG_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in existing: w.writerow(r)
        for r in out: w.writerow(r)
    ups = [r for r in out if r["predicted_direction"] == "up" and r["verdict"] != "REJECT"]
    logger.success(f"已寫 signal_log:{TRADE} 共 {len(out)} 檔(PASS+up 可進場 {len(ups)} 檔)")
    for r in sorted(ups, key=lambda x: -float(x["predicted_center_pct"])):
        logger.info(f"    ★ {r['ticker']} {r['name']} {r['verdict']} "
                    f"預測{r['predicted_direction']} {r['predicted_center_pct']}% conf={r['prediction_confidence']}")

if __name__ == "__main__":
    main()