"""實驗:Step1 純技術 Tech_Score 定大小(無 LLM、純 Python),比對舊 60D 的 +41%。

  換腦不換錢包:訊號來源 = Tech_Score(0-100),其餘(抱/不抱/再平衡/資金)= 舊 _paper_trade 原封不動。
  Tech_Score = max(動能分, 反彈分):
    動能分(僅 MA5>MA20):趨勢gap / RS / 量能 / RSI健康 / MACD柱 → 5項等權×100
    反彈分:rebound_signal.score × 100(大型股深跌,走 config)
  v1 等權樸素版,權重後續 A/B 調。

  資金:期初15000 + 每日1000(上限50000)、最多3檔(+tie)、曝險30-90% —— 全部沿用 _paper_trade。
  同時跑 112(base_universe)與 35(舊科技池)兩組,跟 +41% 對比。

用法: python scripts/exp_step1_techscore.py
"""
from __future__ import annotations
import sys, csv, json, importlib.util, math
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.tools.rebound_signal import rebound_signal

_spec = importlib.util.spec_from_file_location("r60", ROOT / "scripts/run_backtest_60d.py")
_r60 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r60)
_paper_trade = _r60._paper_trade

END = "2026-06-08"; DAYS = 60
REPORT = ROOT / "reports" / "exp_step1_techscore.md"

def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))

_OH = {}
def oh(tk):
    if tk not in _OH:
        try: _OH[tk] = get_daily_ohlcv(tk) or {}
        except Exception: _OH[tk] = {}
    return _OH[tk]

def features(tk):
    """一次算好所有滾動指標(只看過去 → leak-safe),回傳 date_str -> feat dict。"""
    o = oh(tk)
    if len(o) < 30: return {}
    ds = sorted(o)
    c = pd.Series([o[d]["close"] for d in ds], index=ds)
    v = pd.Series([o[d].get("volume", 0) for d in ds], index=ds)
    ma5 = c.rolling(5).mean(); ma20 = c.rolling(20).mean()
    volr = v / v.rolling(20).mean()
    d_ = c.diff(); gain = d_.clip(lower=0).rolling(14).mean(); loss = (-d_.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26; macdh = macd - macd.ewm(span=9, adjust=False).mean()
    ret20 = c / c.shift(20) - 1
    out = {}
    for i, dd in enumerate(ds):
        out[dd] = {"close": c.iloc[i], "ma5": ma5.iloc[i], "ma20": ma20.iloc[i],
                   "volr": volr.iloc[i], "rsi": rsi.iloc[i], "macdh": macdh.iloc[i],
                   "ret20": ret20.iloc[i]}
    return out

def tech_score(feat, idx_ret20, closes_upto, avg_turn):
    """回傳 (score 0-100, thesis)。"""
    # 動能分(僅多頭趨勢)
    mom = 0.0
    if feat["ma20"] and feat["ma5"] > feat["ma20"] and not math.isnan(feat.get("rsi", float("nan"))):
        rs = (1 + feat["ret20"]) / (1 + idx_ret20) if idx_ret20 is not None and abs(1+idx_ret20) > 1e-6 else 1.0
        f_trend = clamp((feat["ma5"] / feat["ma20"] - 1) / 0.05)
        f_rs    = clamp((rs - 0.9) / 0.2)
        f_vol   = clamp((feat["volr"] - 1.2) / 0.8) if not math.isnan(feat["volr"]) else 0.0
        r = feat["rsi"]
        f_rsi = clamp((r - 40) / 30) if r <= 70 else clamp(1 - (r - 70) / 20)   # 50-70健康、>80過熱扣
        f_macd = 1.0 if feat["macdh"] > 0 else 0.4
        mom = 100 * (f_trend + f_rs + f_vol + f_rsi + f_macd) / 5
    # 反彈分(大型股深跌,走 config)
    reb = 0.0
    try:
        sig = rebound_signal(closes_upto, avg_turn)
        if sig.get("fired"): reb = sig["score"] * 100
    except Exception:
        pass
    return (mom, "momentum") if mom >= reb else (reb, "rebound")

def run_exp(codes, names, turns, twii_feat, label):
    cal = [d for d in sorted(twii_feat) if d <= END]
    window = cal[-DAYS:]
    feats = {c: features(c) for c in codes}
    rows = []
    for d in window:
        ir = twii_feat.get(d, {}).get("ret20")
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            o = oh(c); closes = [o[x]["close"] for x in sorted(o) if x <= d]
            sc, thesis = tech_score(f[d], ir, closes, turns.get(c, 0.0))
            if sc <= 0: continue
            rows.append({"date": d, "ticker": c, "name": names.get(c, c),
                         "predicted_center_pct": "100", "prediction_confidence": f"{sc/100:.4f}",
                         "predicted_direction": "up", "llm_verdict": "PASS",
                         "pattern_type": thesis, "bull_score": "", "bear_score": ""})
    pt = _paper_trade(rows)
    n_reb = sum(1 for r in rows if r["pattern_type"] == "rebound")
    return {"label": label, "n_codes": len(codes), "n_sig": len(rows), "n_reb": n_reb,
            "window": (window[0], window[-1]), "pt": pt}

def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes112 = list(u.keys())
    names = {c: u[c].get("name", c) for c in codes112}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes112}
    codes35 = sorted({r["ticker"] for r in csv.DictReader(
        (DATA_DIR / "backtest_llm_results.csv").open(encoding="utf-8"))})
    logger.info("算 0050 基準特徵...")
    twii_feat = features("0050")

    results = []
    for codes, lab in [(codes112, "112 全 base_universe"), (codes35, "35 舊科技池")]:
        logger.info(f"跑 {lab} ({len(codes)} 檔)...")
        results.append(run_exp(codes, names, turns, twii_feat, lab))

    # 報告
    OLD = {"ret": 41.84, "eq": 70922, "pnl": 20922, "win": "33/56", "mdd": 1886, "sharpe": 5.43}
    L = ["# 實驗:Step1 純技術 Tech_Score 定大小(無 LLM)vs 舊 60D +41.84%\n",
         f"> 窗口 末{DAYS}交易日(≤{END})｜資金 15000+1000/日(上限5萬)｜最多3檔(+tie)｜"
         "抱/不抱/再平衡全沿用舊 _paper_trade｜訊號=Tech_Score(動能∪反彈)、純Python無LLM、無手續費\n",
         "## 結果對比\n",
         "| 版本 | 訊號數(反彈) | 本金報酬率 | 期末權益 | 獲利日勝率 | MDD | Sharpe |",
         "|------|------|------|------|------|------|------|"]
    for r in results:
        p = r["pt"]
        L.append(f"| **{r['label']}** | {r['n_sig']}({r['n_reb']}) | **{p['return_pct']:+.2f}%** "
                 f"| {p['final_equity']:,.0f} | {p['win_days']}/{p['total_days']} "
                 f"| -{p['max_dd']:,.0f} | {p['sharpe']:.2f} |")
    L.append(f"| 舊 60D(LLM,35科技,可能無新聞) | 498 | {OLD['ret']:+.2f}% | {OLD['eq']:,} "
             f"| {OLD['win']} | -{OLD['mdd']:,} | {OLD['sharpe']} |")
    L += ["", "## 判讀",
          "- **35 舊科技池 vs 舊 41%**:apples-to-apples。Tech_Score(無LLM)能 ≈ 或贏 → 代表 41% 主要靠技術選股,LLM(無新聞)沒加多少。",
          "- **112 vs 35**:看擴池(含傳產金融+反彈)是賺更多還是被稀釋。",
          "- Tech_Score 權重是 v1 等權樸素版,還沒調。"]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    for r in results:
        p = r["pt"]
        logger.success(f"{r['label']}: 報酬 {p['return_pct']:+.2f}% 勝率 {p['win_days']}/{p['total_days']} "
                       f"訊號{r['n_sig']}(反彈{r['n_reb']}) [舊 +41.84%]")

if __name__ == "__main__":
    main()