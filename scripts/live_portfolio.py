"""Live 真實投資帳本：把每天的掃描預測（signal_log）累積成一張會長大的帳本，
格式與回測的每日持倉明細相同。

時間軸（重要）：
  - 某日掃描（例如 signal_log 日期 6/10）是用『前一交易日 6/9 收盤』做的決策。
  - 進場/再平衡價 = 前一交易日收盤（6/9）；結算 = 當日收盤（6/10）。
  - 所以 6/10 那一列的『當日P&L』就是 6/10 當天的真實賺賠。
  - 當日若還沒收盤（FinMind 還沒有當日收盤價）→ P&L 顯示「pending」。

每日用法：
  1. 早上（開盤前）：python scripts/daily_stock_scan.py      # 產生當日預測
  2. 接著：       python scripts/live_portfolio.py           # 更新帳本、列出今天該怎麼投
  3. 收盤後再跑一次 live_portfolio.py（會 force-refresh 當日收盤）→ 填上今天的 P&L
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tw_stock_agent.config import SIGNAL_LOG, REPORTS_DIR
from tw_stock_agent.tools.finmind_client import get_daily_prices

# 資金規劃
INITIAL_CAPITAL  = 15000.0   # 初始資金（第一天注入）
DAILY_BUDGET     = 1000.0    # 每日加碼
MAX_CONTRIBUTION = 50000.0   # 總投入上限：達 50,000 後停止加碼，只用既有資金操作

# 與 run_backtest_60d.py 一致的風控旋鈕（請保持同步）
MAX_SIGNALS     = 3
DAILY_ADD_CAP   = 15000.0    # 單支股票「當天加碼」上限（賣出不限）；已取消持有上限
EXPOSURE_CAP    = 0.90
EXPOSURE_FLOOR  = 0.30
TIE_RATIO       = 0.90
EDGE_DECAY      = 0.80
INCUMBENT_BONUS = 1.20

START_DATE = "2026-06-10"   # 帳本起始日（第一份乾淨的 Step1 預測）
LEDGER_MD  = REPORTS_DIR / "live_portfolio.md"


def _load_signals() -> dict[str, dict[str, dict]]:
    """讀 signal_log，回傳 {date: {ticker: sig}}，只取 START_DATE 之後。"""
    sig_by_date: dict[str, dict[str, dict]] = defaultdict(dict)
    if not SIGNAL_LOG.exists():
        return sig_by_date
    for r in csv.DictReader(open(SIGNAL_LOG, encoding="utf-8")):
        d = r.get("date", "")
        if d < START_DATE:
            continue
        tk = r.get("ticker", "")
        if not tk:
            continue
        try:
            center = float(r.get("predicted_center_pct", "") or 0)
            conf = float(r.get("prediction_confidence", "") or 0)
        except (ValueError, TypeError):
            center, conf = 0.0, 0.0
        is_up = r.get("predicted_direction", "") == "up" and r.get("verdict", "") != "REJECT"
        sig_by_date[d][tk] = {
            "name": r.get("name", tk), "conf": conf, "center": center,
            "bull": r.get("bull_score", ""), "bear": r.get("bear_score", ""),
            "edge": max(0.0, center) / 100.0 * max(0.0, conf) if is_up else 0.0,
        }
    return sig_by_date


def run() -> None:
    sig_by_date = _load_signals()
    if not sig_by_date:
        print(f"signal_log 裡沒有 >= {START_DATE} 的預測。先跑 daily_stock_scan.py。")
        return

    tickers = {tk for day in sig_by_date.values() for tk in day}
    name_map = {tk: s["name"] for day in sig_by_date.values() for tk, s in day.items()}
    # force-refresh 一次，確保拿到最新收盤（含當日，若已收盤）
    panel = {tk: get_daily_prices(tk, force_refresh=True) for tk in tickers}

    def prior_close(tk: str, d: str) -> float | None:
        pr = panel.get(tk, {})
        ds = [x for x in pr if x < d]
        return pr[max(ds)] if ds else None

    def day_close(tk: str, d: str) -> float | None:
        return panel.get(tk, {}).get(d)   # 嚴格當日，沒有就是還沒收盤 → None(pending)

    pred_dates = sorted(sig_by_date)
    cash = 0.0
    shares: dict[str, float] = {}
    last_edge: dict[str, float] = {}
    prev_equity = 0.0
    total_contributed = 0.0
    ledger: list[dict] = []

    for i, d in enumerate(pred_dates):
        # 1) 加碼：第一天注入初始資金，其後每天 +1000；總投入達 MAX_CONTRIBUTION 即停
        if i == 0:
            contribution = INITIAL_CAPITAL
        elif total_contributed < MAX_CONTRIBUTION:
            contribution = min(DAILY_BUDGET, MAX_CONTRIBUTION - total_contributed)
        else:
            contribution = 0.0
        cash += contribution
        total_contributed += contribution
        # 用『前一交易日收盤』估現有持股
        port = cash + sum(sh * (prior_close(tk, d) or 0) for tk, sh in shares.items())

        # 2) edge：當日訊號 + 持股衰減
        todays = sig_by_date[d]
        edges, conf_of = {}, {}
        for tk, s in todays.items():
            edges[tk] = s["edge"]; conf_of[tk] = s["conf"]; last_edge[tk] = s["edge"]
        for tk in shares:
            if tk not in edges:
                e = last_edge.get(tk, 0.0) * EDGE_DECAY
                edges[tk] = e; last_edge[tk] = e

        def rank_key(tk: str) -> float:
            return edges[tk] * (INCUMBENT_BONUS if tk in shares else 1.0)

        ranked = sorted([tk for tk in edges if edges[tk] > 0], key=rank_key, reverse=True)
        selected = ranked[:MAX_SIGNALS]
        if len(ranked) > MAX_SIGNALS and rank_key(ranked[MAX_SIGNALS]) >= rank_key(ranked[MAX_SIGNALS - 1]) * TIE_RATIO:
            selected = ranked[:MAX_SIGNALS + 1]

        confs = [conf_of[tk] for tk in selected if tk in conf_of]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        exposure = min(EXPOSURE_CAP, max(EXPOSURE_FLOOR, avg_conf)) if selected else 0.0

        wsum = sum(edges[tk] for tk in selected)
        targets = {}
        if wsum > 0 and exposure > 0:
            for tk in selected:
                targets[tk] = port * exposure * (edges[tk] / wsum)

        # 3) 用前一交易日收盤價成交（多退少補）；買入時當天加碼 ≤ DAILY_ADD_CAP（賣出不限）
        for tk in set(shares) | set(targets):
            p = prior_close(tk, d)
            if not p:
                continue
            cur = shares.get(tk, 0.0) * p
            tgt = targets.get(tk, 0.0)
            delta = tgt - cur
            if delta > DAILY_ADD_CAP:
                delta = DAILY_ADD_CAP
                tgt = cur + delta
            cash -= delta
            if tgt <= 1e-6:
                shares.pop(tk, None)
            else:
                shares[tk] = tgt / p

        # 4) 用『當日收盤』結算 → 當日 P&L（沒收盤價=pending）
        marks = {tk: day_close(tk, d) for tk in shares}
        pending = any(v is None for v in marks.values())
        invested_entry = sum(shares[tk] * (prior_close(tk, d) or 0) for tk in shares)
        if pending:
            equity = cash + invested_entry
            pnl = None
        else:
            invested_mark = sum(shares[tk] * marks[tk] for tk in shares)
            equity = cash + invested_mark
            pnl = equity - prev_equity - contribution   # 扣掉當天加碼才是真實獲利
        prev_equity = equity

        held = sorted(shares, key=lambda x: -(shares[x] * (prior_close(x, d) or 0)))
        holdings_str = ", ".join(
            f"{tk}({name_map.get(tk,'')[:4]}) {shares[tk]*(prior_close(tk,d) or 0):.0f}"
            for tk in held) or "—"
        ledger.append({
            "date": d, "holdings": holdings_str, "invested": invested_entry,
            "exposure": invested_entry / equity if equity > 0 else 0.0,
            "cash": cash, "pnl": pnl, "equity": equity, "pending": pending,
            "contributed": total_contributed,
        })

    _write_ledger(ledger)
    _print_today(ledger[-1], sig_by_date[pred_dates[-1]], name_map)


def _write_ledger(ledger: list[dict]) -> None:
    lines = [f"# Live 真實投資帳本（初始 {INITIAL_CAPITAL:,.0f}，每日 +{DAILY_BUDGET:,.0f}，總投入上限 {MAX_CONTRIBUTION:,.0f} · 期望值加權再平衡）\n",
             f"> 進場價=前一交易日收盤、結算=當日收盤｜單股當天加碼 ≤{DAILY_ADD_CAP:,.0f}、無持有上限"
             f"｜曝險 {EXPOSURE_FLOOR:.0%}-{EXPOSURE_CAP:.0%}｜最多 {MAX_SIGNALS} 檔\n",
             "| 預測日 | 持股（代號 金額TWD） | 投入 | 曝險 | 現金 | 當日P&L | 權益 | 累計投入 |",
             "|--------|------|------|------|------|---------|------|---------|"]
    for t in ledger:
        pnl = "pending" if t["pending"] else f"{t['pnl']:+,.1f}"
        lines.append(f"| {t['date']} | {t['holdings']} | {t['invested']:,.0f} "
                     f"| {t['exposure']:.0%} | {t['cash']:,.0f} | {pnl} | {t['equity']:,.0f} "
                     f"| {t['contributed']:,.0f} |")
    realized = [t for t in ledger if not t["pending"]]
    contributed = ledger[-1]["contributed"]
    capped = f"（已達 {MAX_CONTRIBUTION:,.0f} 上限，停止加碼）" if contributed >= MAX_CONTRIBUTION else ""
    if realized:
        total = sum(t["pnl"] for t in realized)
        ret = total / contributed * 100 if contributed else 0
        lines += ["", f"**累計已結算 P&L：{total:+,.1f} TWD（本金報酬率 {ret:+.2f}%）**　"
                      f"累計投入 {contributed:,.0f}{capped}　期末權益 {ledger[-1]['equity']:,.0f}"]
    else:
        lines += ["", f"累計投入 {contributed:,.0f}{capped}　權益 {ledger[-1]['equity']:,.0f}（尚無已結算 P&L）"]
    LEDGER_MD.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_today(row: dict, sigs: dict, name_map: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  今日帳本（預測日 {row['date']}）")
    print(f"{'='*60}")
    print(f"  持股：{row['holdings']}")
    print(f"  投入 {row['invested']:,.0f}｜曝險 {row['exposure']:.0%}｜現金 {row['cash']:,.0f}")
    print(f"  當日P&L：{'pending（當日尚未收盤）' if row['pending'] else format(row['pnl'],'+,.1f')}"
          f"｜權益 {row['equity']:,.0f}")
    print(f"\n  帳本已更新 → {LEDGER_MD}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        START_DATE = sys.argv[1]
    run()
