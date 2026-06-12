"""純雙引擎 live 帳本:從 START 起每個交易日用 H+反彈 regime 雙引擎選股,
寫進 signal_log → 重建 live_portfolio.md(延續資金)。最新一天=今日 pending 預測。純技術、無 LLM。
用法: python scripts/dual_ledger.py [--refresh] [--capital 50000]
"""
from __future__ import annotations
import sys, csv, json, importlib.util, math, argparse, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR, SIGNAL_LOG
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

v5 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py"))
importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py").loader.exec_module(v5)
features, _factors = v5.features, v5._factors

START = "2026-06-10"   # live 帳本起始日(對齊 live_portfolio.START_DATE)
MAX_SIG, TIE = 3, 0.90
SIG_HEADER = ["date","ticker","name","source_news","bfs_depth","volume_ratio","ma5_gt_ma20","rs_20d",
              "pattern_type","rsi_14","bear_score","bull_score","evidence_level","verdict","close_price",
              "predicted_direction","predicted_low_pct","predicted_high_pct","predicted_center_pct",
              "prediction_confidence","return_1d","return_5d","return_10d","return_20d","max_drawdown_10d"]

def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)

def next_bday(dstr):
    dt = datetime.date.fromisoformat(dstr) + datetime.timedelta(days=1)
    while dt.weekday() >= 5: dt += datetime.timedelta(days=1)
    return dt.isoformat()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true"); ap.add_argument("--capital", type=float, default=50000)
    args = ap.parse_args()
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    print(f"載入行情{'(強制更新)' if args.refresh else ''}...")
    OH = {c: get_daily_ohlcv(c, force_refresh=args.refresh) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050", force_refresh=args.refresh)
    v5._OH = OH
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    data_days = sorted(OH["0050"]); idx = {d: i for i, d in enumerate(data_days)}

    def dual_pick(dec):
        """決策日 dec 的雙引擎選股 → [(score, code)]。"""
        if dec not in twii_feat: return None, []
        bull = bool(twii_feat[dec].get("close") and twii_feat[dec].get("ma20") and twii_feat[dec]["close"] > twii_feat[dec]["ma20"])
        ir = twii_feat[dec].get("ret20")
        vals = sorted(((c, feats[c][dec]["turn"]) for c in codes if dec in feats.get(c, {}) and feats[c][dec].get("turn", 0) > 0), key=lambda x: x[1])
        tp = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
        scored = []
        for c in codes:
            f = feats.get(c, {})
            if dec not in f or math.isnan(f[dec].get("ma20", float("nan"))): continue
            if bull:
                sc = h_score(_factors(f[dec], ir), tp.get(c, 0.5))
            else:
                o = OH[c]; cl = [o[x]["close"] for x in sorted(o) if x <= dec]
                sig = rebound_signal(cl, turns.get(c, 0.0)); sc = sig["score"]*100 if sig.get("fired") else 0.0
            if sc > 0: scored.append((sc, c))
        scored.sort(reverse=True)
        sel = scored[:MAX_SIG]
        if len(scored) > MAX_SIG and scored[MAX_SIG][0] >= scored[MAX_SIG-1][0]*TIE:
            sel = scored[:MAX_SIG+1]
        return bull, sel

    # 交易日 = 資料中 ≥START 的日 + 一個 pending(最新+1 營業日)
    trade_days = [d for d in data_days if d >= START] + [next_bday(data_days[-1])]
    rows = []
    for T in trade_days:
        dec = data_days[idx[T]-1] if (T in idx and idx[T] > 0) else (data_days[-1] if T not in idx else None)
        if dec is None: continue
        bull, sel = dual_pick(dec)
        for s, c in sel:
            price = OH[c][dec]["close"]
            rows.append({"date": T, "ticker": c, "name": names.get(c, c), "source_news": "", "bfs_depth": "",
                         "volume_ratio": "", "ma5_gt_ma20": "", "rs_20d": "", "pattern_type": "rebound" if not bull else "momentum",
                         "rsi_14": "", "bear_score": "", "bull_score": "", "evidence_level": "", "verdict": "PASS",
                         "close_price": f"{price}", "predicted_direction": "up", "predicted_low_pct": "",
                         "predicted_high_pct": "", "predicted_center_pct": "100", "prediction_confidence": f"{s/100:.4f}",
                         "return_1d": "", "return_5d": "", "return_10d": "", "return_20d": "", "max_drawdown_10d": ""})
        print(f"  交易日 {T}(決策 {dec},{'多頭H' if bull else '空頭反彈'}): " + ", ".join(f"{c}{names.get(c,c)[:3]}({s:.0f})" for s, c in sel))

    # 寫 signal_log(< START 的保留,≥START 全換成純雙引擎)
    keep = [r for r in csv.DictReader(SIGNAL_LOG.open(encoding="utf-8")) if r.get("date", "") < START] if SIGNAL_LOG.exists() else []
    with SIGNAL_LOG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIG_HEADER, extrasaction="ignore"); w.writeheader()
        for r in keep: w.writerow(r)
        for r in rows: w.writerow(r)
    print(f"已寫 signal_log:{len(rows)} 列(≥{START} 全為純雙引擎)")

    lp = importlib.util.module_from_spec(importlib.util.spec_from_file_location("lp", ROOT/"scripts/live_portfolio.py"))
    importlib.util.spec_from_file_location("lp", ROOT/"scripts/live_portfolio.py").loader.exec_module(lp)
    lp.run()
    print("\n✅ live_portfolio.md 已更新(純雙引擎、延續資金;最新一天=今日 pending 預測)")

if __name__ == "__main__":
    main()