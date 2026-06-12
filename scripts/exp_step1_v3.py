"""Step1 Tech_Score 規則實驗 v3:12 變體 × 5 窗口(60/90/半年/1年/2年),純技術無 LLM。
重點:跨窗口一致性(防 regime overfit)+ 多重檢定要小心。

單因子:
  A 基準(等權5)  B 趨勢RS主導  C 防追高  D 突破  E 反彈加重
  F VCP波動收斂(=D×收斂)  G 大盤濾網(動能×regime,熊市7折)
  H 成交值聚光燈(=B×112內成交值排名,代理全市場)  I 恐慌缺口(=E,缺口→反彈封頂;註:已被舊研究否決,測來反證)
組合/衝突:
  J 防禦動能=B+C+G   K 品質突破=D+F+H   L 衝突=C+D(防追高 vs 買近高)
全部 Tech=max(動能,反彈);僅 MA5>MA20 算動能;沿用舊 _paper_trade;無手續費。
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
REPORT = ROOT / "reports" / "exp_step1_v3.md"
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
    op = pd.Series([o[d]["open"] for d in ds], index=ds)
    ma5 = c.rolling(5).mean(); ma20 = c.rolling(20).mean()
    volr = v / v.rolling(20).mean()
    d_ = c.diff(); g = d_.clip(lower=0).rolling(14).mean(); l = (-d_.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + g / l.replace(0, 1e-9))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26; macdh = macd - macd.ewm(span=9, adjust=False).mean()
    ret = c.pct_change()
    vol10 = ret.rolling(10).std(); vol10med60 = vol10.rolling(60).median()
    out = {}
    prev_c = c.shift(1)
    for i, dd in enumerate(ds):
        out[dd] = {"close": c.iloc[i], "ma5": ma5.iloc[i], "ma20": ma20.iloc[i], "volr": volr.iloc[i],
                   "rsi": rsi.iloc[i], "macdh": macdh.iloc[i], "ret20": (c.iloc[i]/c.iloc[i-20]-1) if i>=20 else float("nan"),
                   "high60": c.iloc[max(0,i-59):i+1].max(),
                   "vcp": (vol10.iloc[i] <= 0.6*vol10med60.iloc[i]) if (i>=60 and not math.isnan(vol10med60.iloc[i]) and vol10med60.iloc[i]>0) else False,
                   "gap": (op.iloc[i]/prev_c.iloc[i]-1) if (i>0 and prev_c.iloc[i]>0) else 0.0,
                   "turn": c.iloc[i]*v.iloc[i]}
    return out

def _factors(feat, idx_ret20):
    if not feat["ma20"] or math.isnan(feat["ma20"]) or feat["ma5"] <= feat["ma20"]: return None
    if math.isnan(feat.get("rsi", float("nan"))): return None
    rs = (1+feat["ret20"])/(1+idx_ret20) if (idx_ret20 is not None and abs(1+idx_ret20)>1e-6 and not math.isnan(feat["ret20"])) else 1.0
    f_trend = clamp((feat["ma5"]/feat["ma20"]-1)/0.05)
    f_rs = clamp((rs-0.9)/0.2)
    f_vol = clamp((feat["volr"]-1.2)/0.8) if not math.isnan(feat["volr"]) else 0.0
    r = feat["rsi"]; f_rsi = clamp((r-40)/30) if r <= 70 else clamp(1-(r-70)/20)
    f_macd = 1.0 if feat["macdh"] > 0 else 0.4
    f_break = clamp((feat["close"]/feat["high60"]-0.90)/0.10) if feat["high60"] else 0.0
    bias = feat["close"]/feat["ma20"]-1
    return (f_trend, f_rs, f_vol, f_rsi, f_macd, f_break, bias)

def score(variant, ff, ex):
    """ff=factors or None; ex=dict(regime_mult, turn_pct, vcp, reb, panic)。回傳(分數,thesis)。"""
    reb = ex["reb"]
    if variant == "I" and ex["panic"] and reb > 0:
        reb = 100.0
    if variant in ("E", "I", "J"):  # E/I/J 反彈加重
        reb = min(100.0, reb*1.2)
    mom = 0.0
    if ff is not None:
        t, rs, vo, ri, ma, br, bias = ff
        eq = (t+rs+vo+ri+ma)/5
        anti = 1.0 if bias <= 0.12 else clamp(1-(bias-0.12)/0.13, 0.5, 1.0)
        if variant in ("A", "C", "G", "E", "I"):
            mom = eq*100
        elif variant in ("B", "H"):
            mom = (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100
        elif variant in ("D", "F", "K", "L"):
            mom = (t+rs+vo+ri+ma+br)/6*100
        elif variant == "J":
            mom = (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100
        # 修飾
        if variant in ("C", "L", "J"): mom *= anti
        if variant in ("G", "J"): mom *= ex["regime_mult"]
        if variant in ("F", "K"): mom *= (1.0 if ex["vcp"] else 0.5)
        if variant in ("H", "K"): mom *= (0.8 + 0.4*ex["turn_pct"])
    sc = max(mom, reb)
    return sc, ("rebound" if reb >= mom else "momentum")

def build_rows(codes, names, feats, twii_feat, reb_cache, turn_pct, variant, dates):
    rows = []
    for d in dates:
        tf = twii_feat.get(d, {})
        ir = tf.get("ret20")
        regime_mult = 1.0 if (tf.get("close") and tf.get("ma20") and tf["close"] > tf["ma20"]) else 0.7
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            fd = f[d]
            ex = {"regime_mult": regime_mult, "turn_pct": turn_pct.get(d, {}).get(c, 0.5),
                  "vcp": fd["vcp"], "reb": reb_cache.get(c, {}).get(d, 0.0),
                  "panic": fd["gap"] <= -0.03 and (not math.isnan(fd["volr"]) and fd["volr"] >= 1.5)}
            sc, thesis = score(variant, _factors(fd, ir), ex)
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
    logger.info("算特徵...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}
    cal = [d for d in sorted(twii_feat) if d <= END][-WINDOWS[-1][1]:]

    logger.info("預算反彈分 + 成交值排名...")
    reb_cache = {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END); closes = []; m = {}
        for d in ds:
            closes.append(o[d]["close"])
            if len(closes) >= 25:
                try:
                    sig = rebound_signal(closes, turns.get(c, 0.0))
                    if sig.get("fired"): m[d] = sig["score"]*100
                except Exception: pass
        reb_cache[c] = m
    turn_pct = {}   # date -> {tk: percentile 0-1}
    for d in cal:
        vals = [(c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0]
        vals.sort(key=lambda x: x[1]); n = len(vals)
        turn_pct[d] = {c: (i+1)/n for i, (c, _) in enumerate(vals)}

    variants = ["A","B","C","D","E","F","G","H","I","J","K","L"]
    vname = {"A":"A基準","B":"B趨勢RS","C":"C防追高","D":"D突破","E":"E反彈加重","F":"F VCP收斂",
             "G":"G大盤濾網","H":"H成交值","I":"I恐慌缺口⚠","J":"J防禦動能(B+C+G)","K":"K品質突破(D+F+H)","L":"L衝突(C+D)"}
    grid = {}
    for v in variants:
        rows_full = build_rows(codes, names, feats, twii_feat, reb_cache, turn_pct, v, cal)
        for wl, n in WINDOWS:
            wd = set(cal[-n:])
            grid[(v, wl)] = _paper_trade([r for r in rows_full if r["date"] in wd])
        logger.info(f"變體 {v} 完成")

    def tbl(key, fmt, title):
        L = [f"## {title}\n", "| 變體 | 60天 | 90天 | 半年 | 1年 | 2年 |", "|---|---|---|---|---|---|"]
        for v in variants:
            L.append(f"| {vname[v]} | " + " | ".join(fmt(grid[(v,wl)][key]) for wl,_ in WINDOWS) + " |")
        return L
    L = ["# Step1 Tech_Score 規則實驗 v3(12 變體 × 5 窗口,純技術無 LLM)\n",
         f"> 結束日 {END}｜112 檔｜資金 15000+1000/日(上限5萬)｜最多3檔｜沿用舊 _paper_trade｜無手續費｜舊60D(含LLM無新聞)=+41.84%\n",
         "> ⚠️ 多重檢定:12 變體易有運氣假象。**主看 2年、要跨窗口一致**;只在 60/90 天強 = regime overfit。\n",
         "> ⚠️ H = 僅 112 檔內成交值排名(非全市場);I = 你舊研究已否決(新鮮瀑布 A/B 變差),測來反證。\n"]
    L += tbl("return_pct", lambda x: f"{x:+.1f}", "本金報酬率 %")
    L += [""] + tbl("sharpe", lambda x: f"{x:.2f}", "年化 Sharpe")
    L += [""] + tbl("max_dd", lambda x: f"-{x:,.0f}", "最大回撤 TWD(越小越好)")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    for v in variants:
        logger.success(f"{vname[v]}: " + " ".join(f"{wl}{grid[(v,wl)]['return_pct']:+.0f}" for wl,_ in WINDOWS)
                       + f" | Sharpe2y {grid[(v,'2年')]['sharpe']:.2f}")

if __name__ == "__main__":
    main()