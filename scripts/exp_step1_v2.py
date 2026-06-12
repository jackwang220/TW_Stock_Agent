"""Step1 Tech_Score 規則實驗 v2:5 變體 × 5 時間窗口(60/90/半年/1年/2年)。
純 Python 無 LLM 無手續費,沿用舊 _paper_trade。重點看「跨窗口一致性」(防 regime overfit)。

變體:
  A 基準      = 等權 5 因子(趨勢/RS/量/RSI/MACD)
  B 趨勢RS主導 = 趨勢.35 RS.35 量.15 RSI.10 MACD.05
  C 防追高    = A × 乖離過熱懲罰(>12% 開始扣到 0.5@25%)
  D 突破      = A 再加「接近60日高點」因子(6 因子等權)
  E 反彈加重  = 反彈分 ×1.2(上限100)
全部 Tech = max(動能, 反彈);僅 MA5>MA20 算動能。
"""
from __future__ import annotations
import sys, csv, json, importlib.util, math
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

END = "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252), ("2年", 504)]
REPORT = ROOT / "reports" / "exp_step1_v2.md"

def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))

_OH = {}
def oh(tk):
    if tk not in _OH:
        try: _OH[tk] = get_daily_ohlcv(tk) or {}
        except Exception: _OH[tk] = {}
    return _OH[tk]

def features(tk):
    o = oh(tk)
    if len(o) < 30: return {}
    ds = sorted(o)
    c = pd.Series([o[d]["close"] for d in ds], index=ds)
    v = pd.Series([o[d].get("volume", 0) for d in ds], index=ds)
    ma5 = c.rolling(5).mean(); ma20 = c.rolling(20).mean()
    volr = v / v.rolling(20).mean()
    d_ = c.diff(); g = d_.clip(lower=0).rolling(14).mean(); l = (-d_.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + g / l.replace(0, 1e-9))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26; macdh = macd - macd.ewm(span=9, adjust=False).mean()
    ret20 = c / c.shift(20) - 1
    high60 = c.rolling(60).max()
    out = {}
    for i, dd in enumerate(ds):
        out[dd] = {"close": c.iloc[i], "ma5": ma5.iloc[i], "ma20": ma20.iloc[i], "volr": volr.iloc[i],
                   "rsi": rsi.iloc[i], "macdh": macdh.iloc[i], "ret20": ret20.iloc[i], "high60": high60.iloc[i]}
    return out

def _factors(feat, idx_ret20):
    """回傳 (f_trend, f_rs, f_vol, f_rsi, f_macd, f_break, bias) 或 None(非多頭)。"""
    if not feat["ma20"] or math.isnan(feat["ma20"]) or feat["ma5"] <= feat["ma20"]:
        return None
    if math.isnan(feat.get("rsi", float("nan"))): return None
    rs = (1 + feat["ret20"]) / (1 + idx_ret20) if idx_ret20 is not None and abs(1+idx_ret20) > 1e-6 else 1.0
    f_trend = clamp((feat["ma5"] / feat["ma20"] - 1) / 0.05)
    f_rs    = clamp((rs - 0.9) / 0.2)
    f_vol   = clamp((feat["volr"] - 1.2) / 0.8) if not math.isnan(feat["volr"]) else 0.0
    r = feat["rsi"]; f_rsi = clamp((r - 40) / 30) if r <= 70 else clamp(1 - (r - 70) / 20)
    f_macd = 1.0 if feat["macdh"] > 0 else 0.4
    f_break = clamp((feat["close"] / feat["high60"] - 0.90) / 0.10) if feat["high60"] else 0.0
    bias = feat["close"] / feat["ma20"] - 1
    return (f_trend, f_rs, f_vol, f_rsi, f_macd, f_break, bias)

def momentum(variant, ff):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    if variant == "A": s = (t + rs + vo + ri + ma) / 5
    elif variant == "B": s = 0.35*t + 0.35*rs + 0.15*vo + 0.10*ri + 0.05*ma
    elif variant == "C":
        s = (t + rs + vo + ri + ma) / 5
        pen = 1.0 if bias <= 0.12 else clamp(1 - (bias - 0.12) / 0.13, 0.5, 1.0)
        s *= pen
    elif variant == "D": s = (t + rs + vo + ri + ma + br) / 6
    elif variant == "E": s = (t + rs + vo + ri + ma) / 5
    else: s = 0.0
    return s * 100

def build_rows(codes, names, turns, twii_feat, feats, reb_cache, variant, dates):
    rows = []
    for d in dates:
        ir = twii_feat.get(d, {}).get("ret20")
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            mom = momentum(variant, _factors(f[d], ir))
            reb = reb_cache.get(c, {}).get(d, 0.0)
            if variant == "E": reb = min(100.0, reb * 1.2)
            sc = max(mom, reb); thesis = "rebound" if reb >= mom else "momentum"
            if sc <= 0: continue
            rows.append({"date": d, "ticker": c, "name": names.get(c, c),
                         "predicted_center_pct": "100", "prediction_confidence": f"{sc/100:.4f}",
                         "predicted_direction": "up", "llm_verdict": "PASS",
                         "pattern_type": thesis, "bull_score": "", "bear_score": ""})
    return rows

def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info("算特徵 + 0050 基準...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}
    cal = [d for d in sorted(twii_feat) if d <= END]

    logger.info("預算反彈分(各股各日)...")
    reb_cache = {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END)
        closes = []; m = {}
        for d in ds:
            closes.append(o[d]["close"])
            if len(closes) >= 25:
                try:
                    sig = rebound_signal(closes, turns.get(c, 0.0))
                    if sig.get("fired"): m[d] = sig["score"] * 100
                except Exception: pass
        reb_cache[c] = m

    variants = ["A", "B", "C", "D", "E"]
    vname = {"A": "A基準", "B": "B趨勢RS主導", "C": "C防追高", "D": "D突破", "E": "E反彈加重"}
    grid = {}   # (variant, win_label) -> pt
    for v in variants:
        full_dates = cal[-WINDOWS[-1][1]:]
        rows_full = build_rows(codes, names, turns, twii_feat, feats, reb_cache, v, full_dates)
        by_date = {}
        for r in rows_full: by_date.setdefault(r["date"], []).append(r)
        for wl, n in WINDOWS:
            wd = set(cal[-n:])
            rows = [r for r in rows_full if r["date"] in wd]
            grid[(v, wl)] = _paper_trade(rows)
        logger.info(f"變體 {v} 完成")

    # 報告
    L = ["# Step1 Tech_Score 規則實驗 v2(5 變體 × 5 窗口,純技術無 LLM)\n",
         f"> 結束日 {END}｜112 檔｜資金 15000+1000/日(上限5萬)｜最多3檔｜沿用舊 _paper_trade｜無手續費\n",
         "> 重點看**跨窗口一致性**,不是挑單一窗口最高(那是 regime overfit)。舊 60D(含LLM,無新聞)= +41.84%\n",
         "## 本金報酬率 %(列=變體,欄=窗口)\n",
         "| 變體 | 60天 | 90天 | 半年 | 1年 | 2年 |", "|------|------|------|------|------|------|"]
    for v in variants:
        cells = " | ".join(f"{grid[(v,wl)]['return_pct']:+.1f}" for wl,_ in WINDOWS)
        L.append(f"| {vname[v]} | {cells} |")
    L += ["", "## 年化 Sharpe(列=變體,欄=窗口)\n",
          "| 變體 | 60天 | 90天 | 半年 | 1年 | 2年 |", "|------|------|------|------|------|------|"]
    for v in variants:
        cells = " | ".join(f"{grid[(v,wl)]['sharpe']:.2f}" for wl,_ in WINDOWS)
        L.append(f"| {vname[v]} | {cells} |")
    L += ["", "## 最大回撤 TWD(越小越好)\n",
          "| 變體 | 60天 | 90天 | 半年 | 1年 | 2年 |", "|------|------|------|------|------|------|"]
    for v in variants:
        cells = " | ".join(f"-{grid[(v,wl)]['max_dd']:,.0f}" for wl,_ in WINDOWS)
        L.append(f"| {vname[v]} | {cells} |")
    L += ["", "## 判讀提示",
          "- 找**跨 5 窗口都穩**的變體(尤其 2年/1年別爛),不是只看 60/90 天。",
          "- 若所有變體在 2年都差不多 → 規則差異不大,維持最簡單的 A。",
          "- 權重都是假設值,還沒最佳化;這步只看『哪個方向值得續調』。"]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    for v in variants:
        logger.success(f"{vname[v]}: " + " ".join(f"{wl}{grid[(v,wl)]['return_pct']:+.0f}%" for wl,_ in WINDOWS))

if __name__ == "__main__":
    main()