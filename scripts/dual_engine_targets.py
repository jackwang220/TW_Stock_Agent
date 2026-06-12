"""雙引擎 → targets.json:跑 H+反彈 regime 雙引擎,對「最新交易日」產出目標持倉,
給 shioaji_executor.py 下單。(策略 = 已驗證的 H雙引擎B純切:多頭打H、空頭打反彈)

用法:
  python scripts/dual_engine_targets.py --capital 50000               # 用最新一天
  python scripts/dual_engine_targets.py --capital 50000 --refresh     # 先更新行情
  python scripts/dual_engine_targets.py --date 2026-06-08             # 指定決策日
輸出: targets.json  → 再跑 stock-analysis/...executor 或 scripts/shioaji_executor.py --targets targets.json
"""
from __future__ import annotations
import sys, json, importlib.util, math, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

v5 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py"))
importlib.util.spec_from_file_location("v5", ROOT/"scripts/exp_step1_v5.py").loader.exec_module(v5)
features, _factors = v5.features, v5._factors

MAX_SIG, EXPO_CAP, EXPO_FLOOR, TIE = 3, 0.90, 0.30, 0.90

def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50000, help="可部署總資金 TWD")
    ap.add_argument("--date", default=None, help="決策日(預設=資料最新一天)")
    ap.add_argument("--refresh", action="store_true", help="先更新行情(force_refresh)")
    ap.add_argument("--out", default=str(ROOT / "targets.json"))
    args = ap.parse_args()

    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    print(f"載入行情({len(codes)} 檔{'+強制更新' if args.refresh else ''})...")
    OH = {c: get_daily_ohlcv(c, force_refresh=args.refresh) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050", force_refresh=args.refresh)
    v5._OH = OH
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}

    d = args.date or max(OH["0050"])
    if d not in twii_feat:
        print(f"❌ 決策日 {d} 無 0050 資料"); return 1
    bull = bool(twii_feat[d].get("close") and twii_feat[d].get("ma20") and twii_feat[d]["close"] > twii_feat[d]["ma20"])
    ir = twii_feat[d].get("ret20")
    print(f"決策日 {d}｜大盤 {'多頭(站上20MA)→ 打H動能' if bull else '空頭(跌破20MA)→ 打反彈'}")

    # 成交值排名(當日)
    vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d].get("turn", 0) > 0), key=lambda x: x[1])
    tp = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    # 反彈分(各股到 d)
    scored = []
    for c in codes:
        f = feats.get(c, {})
        if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
        if bull:
            sc = h_score(_factors(f[d], ir), tp.get(c, 0.5))
        else:
            o = OH[c]; closes = [o[x]["close"] for x in sorted(o) if x <= d]
            sig = rebound_signal(closes, turns.get(c, 0.0))
            sc = sig["score"]*100 if sig.get("fired") else 0.0
        if sc > 0: scored.append((sc, c))
    scored.sort(reverse=True)

    sel = scored[:MAX_SIG]
    if len(scored) > MAX_SIG and scored[MAX_SIG][0] >= scored[MAX_SIG-1][0]*TIE:
        sel = scored[:MAX_SIG+1]
    if not sel:
        print("今日無訊號 → 空手(targets 為空)。")
        Path(args.out).write_text(json.dumps({"date": d, "targets": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    avg = sum(s for s, _ in sel)/len(sel)/100
    expo = min(EXPO_CAP, max(EXPO_FLOOR, avg))
    ssum = sum(s for s, _ in sel)
    targets = {}
    rows = []
    print(f"\n選股(曝險 {expo:.0%},總資金 {args.capital:,.0f}):")
    for s, c in sel:
        tgt = round(args.capital * expo * (s/ssum))
        targets[c] = tgt
        price = OH[c][max(x for x in OH[c] if x <= d)]["close"]
        lots = int((tgt / price) // 1000)
        rows.append((c, names.get(c, c), s, tgt, price, lots))
        print(f"  {c} {names.get(c,c)[:6]:<7} 分數{s:.0f} → 目標 {tgt:,} TWD (約 {lots} 張 @ {price})")

    out = {"date": d, "regime": "bull" if bull else "bear", "capital": args.capital,
           "exposure": round(expo, 3), "targets": targets}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 今日預測報告 .md ──
    eng = "H 動能引擎(多頭)" if bull else "反彈引擎(空頭)"
    L = [f"# 今日選股預測 — H+反彈 regime 雙引擎(純技術,無 LLM)\n",
         f"> 決策日(資料截止){d}｜大盤 0050 {'站上' if bull else '跌破'} 20MA → **{eng}**"
         f"｜總資金 {args.capital:,.0f}｜曝險 {expo:.0%}｜進場=隔日開盤\n",
         "## 今日目標持倉\n",
         "| 代號 | 名稱 | 分數 | 目標金額TWD | 參考價 | 約張數 |",
         "|------|------|------|------|------|------|"]
    for c, nm, s, tgt, price, lots in rows:
        L.append(f"| {c} | {nm} | {s:.0f} | {tgt:,} | {price} | {lots} |")
    L += ["",
          f"**策略**:大盤站上20MA→打強勢動能(H 成交值聚光燈);跌破→打跌深反彈。最多 {MAX_SIG}(+1) 檔、曝險 {EXPO_FLOOR:.0%}–{EXPO_CAP:.0%}。",
          "**注意**:此為純技術引擎(已驗證跨5regime、扣真實成本);LLM 那層暫不接(歷史標題新聞下淨負,待 live 全文再驗)。",
          f"**下單**:`python scripts/shioaji_executor.py --targets {Path(args.out).name}`(預設 dry-run 不送單)。"]
    rpt = ROOT / "reports" / "today_picks.md"
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✅ 已寫 {args.out}")
    print(f"✅ 報告 → {rpt}")
    print(f"→ 下一步: python scripts/shioaji_executor.py --targets {args.out}   (預設 dry-run,先看不送)")

if __name__ == "__main__":
    sys.exit(main() or 0)