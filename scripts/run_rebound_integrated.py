"""反彈整合回測(你原本 60D 的 edge 邏輯 + 4 項改進)

  #1 反彈股「一律進辯論」:每日候選 = 動能篩選 ∪ 反彈觸發 ∪ 目前持股
  #2 反彈加減分旋鈕(--bias δ):對反彈股 bear_score - δ(δ↑ 越容易 PASS),
     套在辯論 cache「之後」→ 掃 δ 不用重跑 LLM(回溯哪個好)
  #4 持股「每日重辯」:持股一律進候選 → 每天重新辯論 → REJECT 即時砍(不再只靠 edge 衰減)
  #3 真實成本:隔日開盤成交(非收盤)、鎖漲停買不到、手續費+證交稅、滑價(trading_rules)

窗口 2026-03-17~06-08。動能骨幹=已辯論的 35 檔大型半導體(cache 命中);
反彈在 base_universe(112)上掃,反彈/持股缺的辯論「補跑並寫入 cache」(可 resume/掃描重用)。

用法:
    python scripts/run_rebound_integrated.py --bias 0        # 基準
    python scripts/run_rebound_integrated.py --bias 1        # 反彈 bear-1(更容易 PASS)
    python scripts/run_rebound_integrated.py --sweep         # 掃 -1,0,1,2(重用 cache,只比結果)
"""
from __future__ import annotations
import argparse, csv, sys, math, json, importlib.util
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR, cfg
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools import trading_rules as TR
from tw_stock_agent.debate.bear import run_debate

# 借用 crossperiod 的 _finmind_stock(leak-safe 組 stock dict)
_spec = importlib.util.spec_from_file_location("cpv", ROOT / "scripts" / "crossperiod_validate.py")
_cpv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cpv)
_finmind_stock = _cpv._finmind_stock

WINDOW_START, WINDOW_END = "2026-03-17", "2026-06-08"
INITIAL, DAILY, MAXC = 15000.0, 1000.0, 50000.0
MAX_SIG, EXPO_CAP, EXPO_FLOOR = 3, 0.90, 0.30
INCUMBENT, TIE = 1.20, 0.90
REJECT_T, WARN_T = 7, 5
FEE = float(cfg("trading.cost.round_trip_pct", 0.005))
SLIP = float(cfg("trading.entry.slippage_pct", 0.001))
QUANT_CSV = DATA_DIR / "backtest_one_month.csv"
SEED_CSV  = DATA_DIR / "backtest_llm_results.csv"
CACHE_CSV = DATA_DIR / "_reb_debate_cache.csv"
REPORT    = ROOT / "reports" / "rebound_integrated.md"

# ── 股票池 ──
def _load_base() -> list[str]:
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    return list(u.keys()) if isinstance(u, dict) else list(u)
BASE = _load_base()

# ── 價量(本地快取)+ 自算日均成交額(不依賴 json 的 turnover,避免 0 坑)──
_OH: dict[str, dict] = {}
def oh(tk: str) -> dict:
    if tk not in _OH:
        try: _OH[tk] = get_daily_ohlcv(tk) or {}
        except Exception: _OH[tk] = {}
    return _OH[tk]

def avg_turn(tk: str, as_of: str) -> float:
    o = oh(tk); ds = [d for d in sorted(o) if d <= as_of][-60:]
    vals = [o[d]["close"] * o[d].get("volume", 0) for d in ds if o[d].get("volume")]
    return sum(vals) / len(vals) if vals else 0.0

# ── 辯論 cache(seed=既有498 + 持久檔;新跑的補進去)──
CACHE: dict[str, dict] = {}
def _seed_cache():
    for src in (SEED_CSV, CACHE_CSV):
        if not src.exists(): continue
        for r in csv.DictReader(src.open(encoding="utf-8")):
            if r.get("llm_verdict", "") in ("", "ERROR"): continue
            k = f"{r.get('date')}_{r.get('ticker')}"
            try:
                CACHE[k] = {
                    "bear": float(r.get("bear_score") or 0), "bull": float(r.get("bull_score") or 0),
                    "evidence": r.get("evidence_level", ""), "verdict": r.get("llm_verdict", ""),
                    "dir": r.get("predicted_direction", ""),
                    "center": float(r.get("predicted_center_pct") or 0),
                    "conf": float(r.get("prediction_confidence") or 0),
                    "name": r.get("name", r.get("ticker")),
                }
            except (ValueError, TypeError): pass
    logger.info(f"cache 種子 {len(CACHE)} 筆")

_NEW: list[dict] = []
def _flush_new():
    if not _NEW: return
    hdr = ["date", "ticker", "name", "bear_score", "bull_score", "evidence_level",
           "llm_verdict", "predicted_direction", "predicted_center_pct", "prediction_confidence"]
    exists = CACHE_CSV.exists()
    with CACHE_CSV.open("a" if exists else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if not exists: w.writeheader()
        w.writerows(_NEW)
    _NEW.clear()

def get_debate(tk: str, name: str, d: str) -> dict | None:
    k = f"{d}_{tk}"
    if k in CACHE: return CACHE[k]
    stock = None
    try: stock = _finmind_stock(tk, name, d)
    except Exception: stock = None
    if stock is None: return None
    try:
        res = run_debate(stock, [], historical_date=date.fromisoformat(d))
    except Exception as e:
        logger.warning(f"  辯論失敗 {k}: {str(e)[:60]}"); return None
    rec = {"bear": res.bear_score, "bull": res.bull_score, "evidence": res.evidence_level,
           "verdict": res.verdict, "dir": res.predicted_direction,
           "center": res.predicted_center_pct, "conf": res.prediction_confidence, "name": name}
    CACHE[k] = rec
    _NEW.append({"date": d, "ticker": tk, "name": name, "bear_score": res.bear_score,
                 "bull_score": res.bull_score, "evidence_level": res.evidence_level,
                 "llm_verdict": res.verdict, "predicted_direction": res.predicted_direction,
                 "predicted_center_pct": f"{res.predicted_center_pct:.2f}",
                 "prediction_confidence": f"{res.prediction_confidence:.2f}"})
    return rec

# ── 候選來源 ──
def _momentum_by_date(allowed: set[str]) -> dict[str, list[tuple[str, str]]]:
    """動能候選:backtest_one_month PASS,且在 allowed(=完整 base_universe 112)裡。"""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if not QUANT_CSV.exists(): return out
    for r in csv.DictReader(QUANT_CSV.open(encoding="utf-8")):
        if r.get("pass_level") != "PASS": continue
        d, tk = r.get("date", ""), r.get("ticker", "")
        if WINDOW_START <= d <= WINDOW_END and tk in allowed:
            out[d].append((tk, r.get("name", tk)))
    return out

def _rebound_at(d: str) -> dict[str, dict]:
    """當日 base_universe 反彈觸發股(用自算 turnover)。"""
    res = {}
    for tk in BASE:
        o = oh(tk); ds = [x for x in sorted(o) if x <= d]
        if len(ds) < 25: continue
        closes = [o[x]["close"] for x in ds]
        sig = rebound_signal(closes, avg_turn(tk, d))
        if sig.get("fired"):
            res[tk] = sig
    return res

# ── 一個交易日的決策(算 targets$)──
def _decide(d: str, mom: list[tuple[str, str]], reb: dict[str, dict],
            holdings: set[str], names: dict[str, str], equity: float, bias: float) -> dict:
    cand: dict[str, str] = {}
    for tk, nm in mom: cand[tk] = nm
    for tk in reb: cand[tk] = reb[tk].get("name") or names.get(tk, tk)
    for tk in holdings: cand.setdefault(tk, names.get(tk, tk))

    # 並行辯論(cache 命中的瞬回)
    def _one(item):
        tk, nm = item
        return tk, get_debate(tk, nm, d)
    with ThreadPoolExecutor(max_workers=8) as ex:
        debated = dict(ex.map(_one, cand.items()))

    sig: dict[str, dict] = {}
    for tk, rec in debated.items():
        if rec is None: continue
        is_reb = tk in reb
        bear = rec["bear"] - (bias if is_reb else 0)           # #2 旋鈕
        verdict = "REJECT" if bear >= REJECT_T else ("WARN" if bear >= WARN_T else "PASS")
        is_up = rec["dir"] == "up" and verdict != "REJECT"
        edge = max(0.0, rec["center"]) / 100.0 * max(0.0, rec["conf"]) if is_up else 0.0
        names[tk] = rec.get("name", tk)
        sig[tk] = {"edge": edge, "conf": rec["conf"], "center": rec["center"],
                   "verdict": verdict, "is_reb": is_reb,
                   "reb_edge": reb[tk]["rebound_edge"] if is_reb else 0.0,
                   "reb_score": reb[tk]["score"] if is_reb else 0.0}

    def rank_key(tk): return sig[tk]["edge"] * (INCUMBENT if tk in holdings else 1.0)
    ranked = sorted([tk for tk in sig if sig[tk]["edge"] > 0], key=rank_key, reverse=True)
    selected = ranked[:MAX_SIG]
    if len(ranked) > MAX_SIG and rank_key(ranked[MAX_SIG]) >= rank_key(ranked[MAX_SIG - 1]) * TIE:
        selected = ranked[:MAX_SIG + 1]

    confs = [sig[tk]["conf"] for tk in selected]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    expo = min(EXPO_CAP, max(EXPO_FLOOR, avg_conf)) if selected else 0.0
    wsum = sum(sig[tk]["edge"] for tk in selected)
    targets = {}
    if wsum > 0 and expo > 0:
        for tk in selected:
            targets[tk] = equity * expo * (sig[tk]["edge"] / wsum)
    return {"sig": sig, "selected": selected, "targets": targets,
            "rejected_holds": [tk for tk in holdings if tk in sig and sig[tk]["verdict"] == "REJECT"]}

# ── 主回測 ──
def run(bias: float, verbose: bool = True) -> dict:
    mom_by_date = _momentum_by_date(set(BASE))   # 完整 112 支,不縮範圍

    # 交易日曆(窗口內 + 後一天供成交)
    cal_all = sorted({d for tk in set(BASE) for d in oh(tk)})
    sig_days = [d for d in cal_all if WINDOW_START <= d <= WINDOW_END]
    if not sig_days:
        logger.error("窗口內無交易日"); return {}
    nd_of = {}                                   # 訊號日 -> 隔一交易日(成交日)
    for d in sig_days:
        later = [x for x in cal_all if x > d]
        if later: nd_of[d] = later[0]

    cash = 0.0; contributed = 0.0
    shares: dict[str, float] = {}
    names: dict[str, str] = {}
    eq_curve: list[float] = []; day_pnl: list[float] = []
    ledger: list[dict] = []; trades: list[dict] = []
    costs = {"fees": 0.0, "slip": 0.0, "limit_up_skip": 0, "chase_skip": 0, "noedge_skip": 0,
             "reject_sell": 0, "n_debates_total": 0}
    prev_eq = 0.0

    for i, d in enumerate(sig_days):
        nd = nd_of.get(d)
        if nd is None: break
        # 加碼
        add = INITIAL if i == 0 else (min(DAILY, MAXC - contributed) if contributed < MAXC else 0.0)
        cash += add; contributed += add

        def close_on(tk, day):
            o = oh(tk); ds = [x for x in sorted(o) if x <= day]
            return o[ds[-1]]["close"] if ds else None

        equity_d = cash + sum(shares[tk] * (close_on(tk, d) or 0) for tk in shares)
        reb = _rebound_at(d)
        dec = _decide(d, mom_by_date.get(d, []), reb, set(shares), names, equity_d, bias)
        sig, targets = dec["sig"], dec["targets"]
        costs["reject_sell"] += len(dec["rejected_holds"])

        # ── 隔日(nd)開盤成交,真實成本 ──
        # 1) 賣出/減碼:目標< 現值 或 不在 targets(含 REJECT、edge=0)
        for tk in list(shares):
            ndbar = oh(tk).get(nd)
            if not ndbar: continue
            p_open = ndbar["open"]
            cur_val = shares[tk] * p_open
            tgt = targets.get(tk, 0.0)
            if tgt < cur_val - 1e-6:
                sell_val = (cur_val - tgt)
                sell_sh = sell_val / p_open
                fill = p_open * (1 - SLIP)
                fee = FEE * sell_sh * p_open
                costs["slip"] += sell_sh * p_open * SLIP; costs["fees"] += fee
                cash += sell_sh * fill - fee
                shares[tk] -= sell_sh
                ep = next((t["eprice"] for t in reversed(trades) if t["tk"] == tk and t.get("open")), p_open)
                if shares[tk] <= 1e-6: shares.pop(tk, None)
        # 2) 買進/加碼:trading_rules 真實進場(隔日開盤/漲停/追高/滑價/剩餘edge)
        order = sorted(dec["selected"], key=lambda t: -sig[t]["edge"])
        for tk in order:
            ndbar = oh(tk).get(nd)
            if not ndbar: continue
            ref_close = close_on(tk, d) or ndbar["open"]
            cur_val = shares.get(tk, 0.0) * ndbar["open"]
            tgt = targets.get(tk, 0.0)
            if tgt <= cur_val + 1e-6: continue
            s = sig[tk]
            target_pct = max(s["center"] / 100.0, s["reb_edge"])
            conf = max(s["conf"], s["reb_score"])
            d2 = TR.entry_decision(ref_close=ref_close, target_pct=target_pct, conf=conf,
                                   next_open=ndbar["open"], prev_close=ref_close,
                                   avg_turnover=avg_turn(tk, d), equity=equity_d)
            if not d2.get("buy"):
                rsn = d2.get("reason", "")
                if "漲停" in rsn: costs["limit_up_skip"] += 1
                elif "追高" in rsn: costs["chase_skip"] += 1
                elif "edge" in rsn: costs["noedge_skip"] += 1
                continue
            budget = min(tgt - cur_val, cash, d2.get("size_cap", cash))
            if budget < 1000: continue
            buy_sh = budget / d2["price"]
            costs["slip"] += buy_sh * ndbar["open"] * SLIP
            cash -= buy_sh * d2["price"]
            shares[tk] = shares.get(tk, 0.0) + buy_sh
            trades.append({"tk": tk, "name": names.get(tk, tk), "sigdate": d, "open": True,
                           "eprice": d2["price"], "shares": buy_sh,
                           "thesis": "rebound" if s["is_reb"] else "momentum"})

        # ── nd 收盤估值 ──
        invested = sum(shares[tk] * (close_on(tk, nd) or 0) for tk in shares)
        equity = cash + invested
        pnl = equity - prev_eq - add; prev_eq = equity
        day_pnl.append(pnl); eq_curve.append(equity)
        held = sorted(shares, key=lambda x: -(shares[x] * (close_on(x, nd) or 0)))
        ledger.append({"date": d, "n_reb": len(reb), "n_cand": len(sig),
                       "holdings": ", ".join(f"{tk}({names.get(tk,'')[:4]})" for tk in held) or "—",
                       "invested": invested, "cash": cash,
                       "expo": invested / equity if equity > 0 else 0.0,
                       "pnl": pnl, "equity": equity})
        if verbose and i % 5 == 0:
            logger.info(f"  {d} 反彈{len(reb)} 候選{len(sig)} 持{len(shares)} 權益{equity:,.0f}")
        _flush_new()

    # ── 統計 ──
    final_eq = eq_curve[-1] if eq_curve else 0.0
    total_pnl = sum(day_pnl)
    active = [x for x in day_pnl if abs(x) > 1e-9]
    win = sum(1 for x in active if x > 0)
    cum = peak = mdd = 0.0
    for x in day_pnl:
        cum += x; peak = max(peak, cum); mdd = max(mdd, peak - cum)
    if len(active) > 1:
        m = sum(active) / len(active)
        sd = math.sqrt(sum((x - m) ** 2 for x in active) / len(active))
        sharpe = (m / sd * math.sqrt(252)) if sd > 0 else 0.0
    else: sharpe = 0.0
    n_reb_trades = sum(1 for t in trades if t["thesis"] == "rebound")
    return {"bias": bias, "contributed": contributed, "final_eq": final_eq, "total_pnl": total_pnl,
            "return_pct": total_pnl / contributed * 100 if contributed else 0.0,
            "win": win, "active": len(active), "mdd": mdd, "sharpe": sharpe,
            "costs": costs, "ledger": ledger, "trades": trades,
            "n_trades": len(trades), "n_reb_trades": n_reb_trades, "days": len(sig_days)}

# ── 報告 ──
def write_report(results: list[dict]):
    L = ["# 反彈整合回測報告(動能∪反彈∪持股 每日辯論 · 真實成本)\n",
         f"> 窗口 {WINDOW_START}~{WINDOW_END}｜期初 {INITIAL:,.0f}+每日 {DAILY:,.0f}(上限 {MAXC:,.0f})"
         f"｜最多 {MAX_SIG} 檔｜手續費{FEE*100:.1f}%+滑價{SLIP*100:.1f}%\n",
         "> 改進:①反彈一律進辯論 ②反彈加減分旋鈕(bias) ③真實成本(漲停買不到/手續費/隔日開盤) ④持股每日重辯\n"]
    L += ["## bias 掃描比較(δ=反彈 bear_score 減項,越大越易 PASS)\n",
          "| bias δ | 本金報酬率 | 期末權益 | 交易數(反彈) | 漲停買不到 | 持股被REJECT砍 | MDD | Sharpe |",
          "|--------|-----------|---------|-------------|-----------|---------------|-----|--------|"]
    for r in results:
        c = r["costs"]
        L.append(f"| {r['bias']:+.0f} | **{r['return_pct']:+.2f}%** | {r['final_eq']:,.0f} "
                 f"| {r['n_trades']}({r['n_reb_trades']}) | {c['limit_up_skip']} | {c['reject_sell']} "
                 f"| -{r['mdd']:,.0f} | {r['sharpe']:.2f} |")
    L.append("")
    base = results[0]; c = base["costs"]
    L += [f"## 真實成本明細(bias={base['bias']:+.0f})\n", "| 項目 | 數值 |", "|------|------|",
          f"| 累計投入本金 | {base['contributed']:,.0f} TWD |",
          f"| 期末權益 | {base['final_eq']:,.0f} TWD |",
          f"| **淨損益 / 報酬率** | **{base['total_pnl']:+,.0f} / {base['return_pct']:+.2f}%** |",
          f"| 獲利日勝率 | {base['win']}/{base['active']} |",
          f"| 總手續費+證交稅 | -{c['fees']:,.0f} TWD |",
          f"| 總滑價 | -{c['slip']:,.0f} TWD |",
          f"| 鎖漲停買不到(跳過) | {c['limit_up_skip']} 次 |",
          f"| 追高超上限(跳過) | {c['chase_skip']} 次 |",
          f"| 扣 gap+成本後無 edge(跳過) | {c['noedge_skip']} 次 |",
          f"| 持股重辯後被 REJECT 砍 | {c['reject_sell']} 次 |", ""]
    L += ["<details><summary>每日持倉(反彈數/候選數/持股)</summary>\n",
          "| 訊號日 | 反彈 | 候選 | 持股 | 投入 | 曝險 | 當日P&L | 權益 |",
          "|------|------|------|------|------|------|---------|------|"]
    for t in base["ledger"]:
        L.append(f"| {t['date']} | {t['n_reb']} | {t['n_cand']} | {t['holdings']} "
                 f"| {t['invested']:,.0f} | {t['expo']:.0%} | {t['pnl']:+,.0f} | {t['equity']:,.0f} |")
    L.append("</details>\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true", help="掃 -1,0,1,2(重用 cache)")
    args = ap.parse_args()
    _seed_cache()
    biases = [-1, 0, 1, 2] if args.sweep else [args.bias]
    results = []
    for b in biases:
        logger.info(f"===== 跑 bias δ={b:+.0f} =====")
        r = run(b, verbose=True)
        if r: results.append(r); _flush_new()
        logger.info(f"  δ={b:+.0f} → 報酬 {r.get('return_pct',0):+.2f}% 交易 {r.get('n_trades',0)} "
                    f"反彈 {r.get('n_reb_trades',0)}")
    _flush_new()
    if results: write_report(results)
    logger.success("完成")

if __name__ == "__main__":
    main()