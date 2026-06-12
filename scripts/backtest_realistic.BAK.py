"""真實模擬回測:新策略(LLM + 跌深反彈,動態組合) + 真實成交規則。

跟舊的 llm_backtest_60d 不同:這版是『部位制 + 真實成交』——
  進場 = 訊號隔日「開盤」(非收盤),套追高上限/漲停買不到/滑價(trading_rules)
  出場 = 停利/停損(用當日高低)/持有天數到期(trading_rules)
  資金 = 期初15000+每日1000(上限50000);部位上限含流動性%
全部規則讀 configs/default.yaml 的 trading: 區塊(改 yaml 就改行為)。

資料只用 FinMind 日K 快取(不需任何新 API / 不需券商)。

用法:python scripts/backtest_realistic.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from loguru import logger
logger.remove()

from tw_stock_agent.config import DATA_DIR, REPORTS_DIR, TW_STOCK_INDEX, cfg
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.tools.rebound_signal import rebound_signal, avg_turnover_of
from tw_stock_agent.tools import trading_rules as TR

LLM_CSV = DATA_DIR / "backtest_llm_results.csv"
OUT = REPORTS_DIR / "backtest_realistic.md"


def load_signals() -> dict[str, list[dict]]:
    """讀 LLM 回測訊號,依『訊號日』分組(該日決策、隔日進場)。"""
    by_date: dict[str, list[dict]] = {}
    if not LLM_CSV.exists():
        return by_date
    for r in csv.DictReader(LLM_CSV.open(encoding="utf-8")):
        d, tk = r.get("date", ""), r.get("ticker", "")
        if not d or not tk:
            continue
        try:
            center = float(r.get("predicted_center_pct", "") or 0)
            conf = float(r.get("prediction_confidence", "") or 0)
        except (ValueError, TypeError):
            center, conf = 0.0, 0.0
        up = r.get("predicted_direction", "") == "up" and r.get("llm_verdict", "") != "REJECT"
        by_date.setdefault(d, []).append({
            "ticker": tk, "center": center, "conf": conf, "up": up,
            "llm_edge": (max(0.0, center) / 100.0 * max(0.0, conf)) if up else 0.0,
        })
    return by_date


def main():
    import argparse
    import json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebound-only", action="store_true",
                    help="用基本盤全部 + 純跌深反彈訊號(不需 LLM)")
    ap.add_argument("--days", type=int, default=60, help="rebound-only:模擬最近 N 個交易日")
    ap.add_argument("--start", default="", help="rebound-only:起始日(held-out 用,蓋過 --days)")
    ap.add_argument("--end", default="", help="rebound-only:結束日")
    args = ap.parse_args()
    names = {c: v.get("name", c) for c, v in
             _json.loads(TW_STOCK_INDEX.read_text(encoding="utf-8")).items()}

    if args.rebound_only:
        base = _json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
        min_turn = float(cfg("trading.signals.rebound_min_turnover", 0))   # 成交額門檻
        excl_fin = cfg("trading.signals.rebound_exclude_financial", False) # 排除金融
        FIN_KW = ("金融", "保險", "證券", "銀行", "金控")
        tickers = sorted(c for c, v in base.items()
                         if v.get("avg_turnover", 0) >= min_turn
                         and not (excl_fin and any(k in v.get("industry", "") for k in FIN_KW)))
        sigs = {}
    else:
        sigs = load_signals()
        if not sigs:
            print("找不到 backtest_llm_results.csv");  return
        tickers = sorted({s["ticker"] for day in sigs.values() for s in day})

    # 載入日K(開高低收)
    panel, closes_seq = {}, {}
    for tk in tickers:
        oh = get_daily_ohlcv(tk)
        if oh:
            panel[tk] = oh
            closes_seq[tk] = sorted(oh)
    cal = sorted({d for tk in panel for d in panel[tk]})

    if args.rebound_only:
        if args.start or args.end:                 # held-out:指定日期範圍(蓋過 --days)
            cal = [d for d in cal if (not args.start or d >= args.start)
                   and (not args.end or d <= args.end)]
        else:
            cal = cal[-args.days:]                  # 最近 N 個交易日
        # 每天所有基本盤股都是候選,由 rebound_signal 決定誰進場
        sigs = {d: [{"ticker": tk, "center": 0.0, "conf": 0.0, "up": False, "llm_edge": 0.0}
                    for tk in panel] for d in cal}
    else:
        first_sig = min(sigs)
        cal = [d for d in cal if d >= first_sig]
    cal_idx = {d: i for i, d in enumerate(cal)}

    cap = TR.capital_cfg()
    max_sig = int(cfg("trading.sizing.max_signals", 3))
    exp_cap = float(cfg("trading.sizing.exposure_cap", 0.90))
    add_cap = TR.daily_add_cap()

    cash = cap["initial"]; contributed = cap["initial"]; prev_equity = cap["initial"]
    positions = []          # {tk, edate, eidx, eprice, shares, target}
    trades = []             # 已結束
    ledger = []
    fee_pct = float(cfg("trading.cost.round_trip_pct", 0.005))   # 來回手續費+證交稅
    slip_pct = float(cfg("trading.entry.slippage_pct", 0.001))
    costs = {"fees": 0.0, "slip": 0.0, "limit_up_skip": 0, "chase_skip": 0, "noedge_skip": 0}

    def closes_upto(tk, d):
        return [panel[tk][x]["close"] for x in closes_seq[tk] if x <= d]

    for di, d in enumerate(cal):
        # 1) 每日加碼
        add = min(cap["daily_budget"], cap["max_contribution"] - contributed) if contributed < cap["max_contribution"] else 0.0
        cash += add; contributed += add

        # 2) 出場檢查(用當日高低判停利停損)
        for pos in list(positions):
            bar = panel[pos["tk"]].get(d)
            if not bar:
                continue
            days_held = di - pos["eidx"]
            tp = float(cfg("trading.exit.take_profit_pct", 0))
            if cfg("trading.exit.conditional_stop", False):     # 依進場論點分檔停損
                sl = float(cfg("trading.exit.stop_loss_rebound", -0.12)) if pos.get("thesis") == "rebound" \
                    else float(cfg("trading.exit.stop_loss_momentum", -0.08))
            else:
                sl = float(cfg("trading.exit.stop_loss_pct", 0))
            ep = pos["eprice"]; exit_price = None; reason = ""
            stop_lvl = ep * (1 + sl)
            if sl < 0 and bar["low"] <= stop_lvl:
                # 跳空/跌停穿價:當日開盤已低於停損價 → 只能用開盤成交(更差);否則用停損價
                exit_price = min(stop_lvl, bar["open"])
                reason = "停損"
            elif tp > 0 and bar["high"] / ep - 1 >= tp:
                exit_price, reason = ep * (1 + tp), "停利"
            elif days_held >= int(cfg("trading.exit.max_hold_days", 5)):
                exit_price, reason = bar["close"], "到期"
            if exit_price is not None:
                costs["slip"] += pos["shares"] * exit_price * slip_pct   # 出場滑價
                exit_price *= (1 - slip_pct)
                fee = fee_pct * pos["shares"] * ep                       # 來回手續費(以進場市值計)
                costs["fees"] += fee
                cash += pos["shares"] * exit_price - fee
                trades.append({"tk": pos["tk"], "edate": pos["edate"], "xdate": d,
                               "ret": (exit_price / ep - 1) - fee_pct, "reason": reason})
                positions.remove(pos)

        # 3) 進場:用「前一交易日」的訊號,在今日開盤進場
        prev_d = cal[di - 1] if di > 0 else None
        held_tks = {p["tk"] for p in positions}
        equity_now = cash + sum(p["shares"] * (panel[p["tk"]].get(d, {}).get("close", 0)) for p in positions)
        if prev_d and prev_d in sigs:
            cands = []
            for s in sigs[prev_d]:
                tk = s["ticker"]
                if tk in held_tks or tk not in panel:
                    continue
                cl = closes_upto(tk, prev_d)
                if len(cl) < 25:
                    continue
                rb = rebound_signal(cl, avg_turnover_of(tk))
                edge = TR.combine_edge(s["llm_edge"], rb.get("rebound_edge", 0.0))
                if not TR.passes_min_edge(edge):
                    continue
                target = max(s["center"] / 100.0, rb.get("rebound_edge", 0.0))
                cands.append((edge, tk, s, rb, target, cl[-1]))
            cands.sort(key=lambda x: -x[0])
            for edge, tk, s, rb, target, ref_close in cands:
                if len(positions) >= max_sig:
                    break
                bar = panel[tk].get(d)
                if not bar:
                    continue
                conf = max(s["conf"], rb.get("score", 0.0))
                dec = TR.entry_decision(ref_close=ref_close, target_pct=target, conf=conf,
                                        next_open=bar["open"], prev_close=ref_close,
                                        avg_turnover=avg_turnover_of(tk), equity=equity_now)
                if not dec["buy"]:
                    rsn = dec.get("reason", "")
                    if "漲停" in rsn:
                        costs["limit_up_skip"] += 1
                    elif "追高" in rsn:
                        costs["chase_skip"] += 1
                    elif "edge" in rsn:
                        costs["noedge_skip"] += 1
                    continue
                if cfg("trading.sizing.equal_weight", False):   # 純等權(測 sizing 值不值得)
                    budget = min(add_cap, cash, equity_now * exp_cap / max_sig)
                else:
                    budget = min(dec["size_cap"], add_cap, cash, equity_now * exp_cap / max_sig)
                if budget < 1000:
                    continue
                shares = budget / dec["price"]
                costs["slip"] += shares * bar["open"] * slip_pct          # 進場滑價
                cash -= shares * dec["price"]
                thesis = "rebound" if (rb.get("fired") and rb.get("rebound_edge", 0) >= s["llm_edge"]) else "momentum"
                positions.append({"tk": tk, "edate": d, "eidx": di, "eprice": dec["price"],
                                  "shares": shares, "target": target, "thesis": thesis})

        # 4) 估值
        invested = sum(p["shares"] * (panel[p["tk"]].get(d, {}).get("close", 0)) for p in positions)
        equity = cash + invested
        pnl = equity - prev_equity - add
        prev_equity = equity
        ledger.append({"date": d, "equity": equity, "cash": cash, "invested": invested,
                       "pnl": pnl, "npos": len(positions)})

    return _report(ledger, trades, contributed, names, costs)


def _report(ledger, trades, contributed, names, costs=None):
    if not ledger:
        print("無資料"); return
    final = ledger[-1]["equity"]
    total_pnl = final - contributed
    day_pnls = [x["pnl"] for x in ledger]
    active = [x for x in day_pnls if abs(x) > 1e-9]
    win_days = sum(1 for x in active if x > 0)
    cum = peak = mdd = 0.0
    for x in day_pnls:
        cum += x; peak = max(peak, cum); mdd = max(mdd, peak - cum)
    if len(active) > 1:
        m = sum(active) / len(active); sd = math.sqrt(sum((x - m) ** 2 for x in active) / len(active))
        sharpe = m / sd * math.sqrt(252) if sd > 0 else 0.0
    else:
        sharpe = 0.0
    wtrades = [t for t in trades if t["ret"] > 0]
    L = ["# 真實模擬回測(新策略 + 真實成交規則)\n",
         "> 進場=訊號隔日開盤｜含追高上限/漲停買不到/滑價/停利停損/持有天數｜規則讀 configs/default.yaml `trading:`\n",
         f"> 資料:FinMind 日K(無新 API);訊號:LLM+跌深反彈動態組合\n",
         "## 績效\n", "| 指標 | 數值 |", "|------|------|",
         f"| 累計投入 | {contributed:,.0f} |",
         f"| 期末權益 | {final:,.0f} |",
         f"| **淨損益** | **{total_pnl:+,.0f}** |",
         f"| **本金報酬率** | **{total_pnl/contributed*100:+.2f}%** |",
         f"| 完成交易數 | {len(trades)}(勝 {len(wtrades)},勝率 {len(wtrades)/len(trades)*100 if trades else 0:.0f}%)|",
         f"| 每筆平均報酬 | {sum(t['ret'] for t in trades)/len(trades)*100 if trades else 0:+.2f}% |",
         f"| 獲利日勝率 | {win_days}/{len(active)} = {win_days/len(active)*100 if active else 0:.0f}% |",
         f"| 最大回撤 | -{mdd:,.0f} |",
         f"| 年化 Sharpe | {sharpe:.2f} |", ""]
    # 出場原因分布
    from collections import Counter
    rc = Counter(t["reason"] for t in trades)
    L.append(f"出場原因:{dict(rc)}\n")
    # 真實成交成本明細
    if costs:
        gross = total_pnl + costs["fees"] + costs["slip"]   # 加回成本 ≈ 無摩擦損益
        L += ["## 真實成交成本明細\n", "| 項目 | 數值 |", "|------|------|",
              f"| 漲停買不到(跳過) | {costs['limit_up_skip']} 次 |",
              f"| 追高超上限(放棄) | {costs['chase_skip']} 次 |",
              f"| 扣成本後無 edge(跳過) | {costs['noedge_skip']} 次 |",
              f"| 總手續費+證交稅 | -{costs['fees']:,.0f} |",
              f"| 總滑價成本 | -{costs['slip']:,.0f} |",
              f"| 摩擦成本合計 | -{costs['fees']+costs['slip']:,.0f} |",
              f"| 無摩擦損益(理想) | {gross:+,.0f} |",
              f"| **真實損益(扣全部成本)** | **{total_pnl:+,.0f}** |", ""]
    # 交易明細(前30)
    L.append("<details><summary>交易明細</summary>\n")
    L.append("| 進場 | 出場 | 代號 | 名稱 | 報酬% | 原因 |")
    L.append("|------|------|------|------|------|------|")
    for t in sorted(trades, key=lambda x: x["edate"]):
        L.append(f"| {t['edate']} | {t['xdate']} | {t['tk']} | {names.get(t['tk'],'')[:5]} | {t['ret']*100:+.1f} | {t['reason']} |")
    L.append("</details>")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"完成 → {OUT}")
    print(f"  報酬率 {total_pnl/contributed*100:+.2f}%　交易{len(trades)}筆 勝率{len(wtrades)/len(trades)*100 if trades else 0:.0f}%　MDD -{mdd:,.0f}　Sharpe {sharpe:.2f}")
    return {"ret_pct": total_pnl / contributed * 100, "pnl": total_pnl,
            "trades": len(trades), "win": len(wtrades) / len(trades) * 100 if trades else 0,
            "fees": costs["fees"] if costs else 0, "slip": costs["slip"] if costs else 0,
            "mdd": mdd, "sharpe": sharpe,
            "trade_rets": [t["ret"] for t in trades]}


if __name__ == "__main__":
    main()