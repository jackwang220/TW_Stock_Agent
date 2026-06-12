"""60 天 LLM 回測批次執行器

每批 5 個交易日，共 12 批。每批跑完立即更新 reports/llm_backtest_60d.md。
模擬交易：累積資金池，每個交易日加碼 1000 TWD，對現有持股做期望值加權
          （edge = 預測幅度 × 信心）再平衡；含單股上限、總曝險上下限等風控。

使用方式：
    python scripts/run_backtest_60d.py --plan          # 顯示批次規劃
    python scripts/run_backtest_60d.py --batch 1       # 只跑第 1 批
    python scripts/run_backtest_60d.py                 # 從第 1 批跑到第 12 批（支援 resume）
    python scripts/run_backtest_60d.py --report-only   # 只重新產生報告
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger
from tw_stock_agent.config import DATA_DIR

QUANT_CSV    = DATA_DIR / "backtest_one_month.csv"
LLM_CSV      = DATA_DIR / "backtest_llm_results.csv"
REPORT_PATH  = ROOT / "reports" / "llm_backtest_60d.md"
INITIAL_CAPITAL  = 15000.0  # TWD：期初一次性投入
DAILY_BUDGET     = 1000.0   # TWD：每個交易日加碼進資金池
MAX_CONTRIBUTION = 50000.0  # TWD：總投入上限（期初+每日加碼），達到後停止再投錢
MAX_SIGNALS  = 3        # 同時最多持有幾支（分數太接近時放行第 4 支）

# ── 部位管理 / 風控旋鈕 ──────────────────────────────────────────────────────
DAILY_ADD_CAP   = 15000.0  # 單支股票「當天加碼」上限（單日買入增量；賣出不限）
# 註：已取消單支股票「持有」上限，贏家可長期累積到任意大小
EXPOSURE_CAP    = 0.90     # 總曝險上限：最多投 90%，永遠留 ≥10% 現金
EXPOSURE_FLOOR  = 0.30     # 總曝險下限：當天有 up 訊號時至少投 30%（避免過度空手；無訊號則空手）
INCUMBENT_BONUS = 1.20     # 持股排名加成：挑戰者 edge 要高過持股 20% 才換手（防無謂換手）
EDGE_DECAY      = 0.80     # 持股當天無訊號時 edge 衰減係數（自然老化出場）
TIE_RATIO       = 0.90     # 第 4 名 edge ≥ 第 3 名的 90% 就一起納入（分數太接近）

# 與舊版 backtest_llm_results.csv 相同的股票池（18 支主要半導體 / AI 伺服器）
DEFAULT_TICKERS = {
    "2059","2303","2308","2317","2330","2337",
    "2356","2357","2376","2382","2454","3017",
    "3037","3324","3533","3711","6669","8046",
}


# ── 批次計算 ──────────────────────────────────────────────────────────────────

def _get_trading_dates() -> list[str]:
    """從 quant CSV 取出所有有 PASS 訊號的交易日（升冪）。"""
    if not QUANT_CSV.exists():
        logger.error(f"找不到 {QUANT_CSV}，請先執行 backtest_one_month.py")
        sys.exit(1)
    dates: set[str] = set()
    with QUANT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("pass_level") == "PASS":
                dates.add(r["date"])
    return sorted(dates)


def _compute_batches(n_days: int = 60) -> list[tuple[str, str, int]]:
    """計算最近 n_days 交易日的 12 批次，每批 5 天。
    回傳 [(from_date, to_date, batch_no), ...]，從最舊到最新。
    """
    all_dates = _get_trading_dates()
    # 取最後 n_days 個交易日
    window = all_dates[-n_days:] if len(all_dates) >= n_days else all_dates
    if not window:
        return []

    batches = []
    batch_size = 5
    for i in range(0, len(window), batch_size):
        chunk = window[i: i + batch_size]
        batches.append((chunk[0], chunk[-1], len(batches) + 1))
    return batches


# ── LLM 回測執行（引入 backtest_with_llm 邏輯）──────────────────────────────

def _run_batch(from_date: str, to_date: str, batch_no: int,
               delay: float = 2.0, tickers: set | None = None,
               batches: list | None = None) -> int:
    """跑一批 LLM 分析，回傳新完成的筆數。"""
    import importlib.util

    if tickers is None:
        tickers = DEFAULT_TICKERS

    # 動態載入 backtest_with_llm（避免 import 時跑 main）
    spec = importlib.util.spec_from_file_location(
        "bwl", ROOT / "scripts" / "backtest_with_llm.py"
    )
    bwl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bwl)

    rows = bwl._load_quant_results("ab", from_date, to_date, levels=["PASS"])
    # 篩選股票池
    rows = [r for r in rows if r.get("ticker","") in tickers]
    done = bwl._load_done_keys(resume=True)
    todo = [r for r in rows if f"{r['date']}_{r['ticker']}" not in done]

    if not todo:
        logger.info(f"  批次 #{batch_no} 已全部完成，跳過。")
        return 0

    logger.info(f"  批次 #{batch_no}：{from_date}～{to_date}，待跑 {len(todo)} 筆")

    from datetime import date as date_cls
    from tw_stock_agent.debate.bear import run_debate
    from tw_stock_agent.news.scanner import search_news_around_date
    from tw_stock_agent.market_status import get_stock_market_status
    from tw_stock_agent.tools.finmind_client import get_close_on_or_before, get_prev_close

    LLM_CSV.parent.mkdir(parents=True, exist_ok=True)
    append_mode  = LLM_CSV.exists()
    write_header = not append_mode

    done_count = 0
    with LLM_CSV.open("a" if append_mode else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bwl.FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, row in enumerate(todo, 1):
            target_date = date_cls.fromisoformat(row["date"])
            stock = bwl._stock_from_row(row)
            # 注入特殊市場狀態：漲跌停用 FinMind 日線歷史重建（零洩漏），
            # 處置股用 Google News 但只採 published <= 該日的新聞（as_of 過濾）
            try:
                cob  = get_close_on_or_before(row["ticker"], target_date)
                prev = get_prev_close(row["ticker"], target_date)
                if cob and prev:
                    stock["market_status"] = get_stock_market_status(
                        row["ticker"], row["name"],
                        close=cob[1], prev_close=prev, as_of=target_date)
            except Exception as e:
                logger.warning(f"      market_status 注入失敗：{e}")
            logger.info(f"    [{i}/{len(todo)}] {row['date']} {row['ticker']} {row['name']} "
                        f"({row['pattern_type']})")
            try:
                result = run_debate(stock, [], historical_date=target_date)

                comp_news  = search_news_around_date(row["name"], target_date, window_days=1, max_articles=12)
                broad_news = search_news_around_date(
                    "台灣科技股 OR 半導體 OR AI伺服器", target_date, window_days=1, max_articles=5)
                comp_n  = sum(1 for a in comp_news  if "[搜尋失敗]" not in a.get("title",""))
                broad_n = sum(1 for a in broad_news if "[搜尋失敗]" not in a.get("title",""))
                news_mode = "historical" if (comp_n + broad_n) >= 3 else "technical_only"

                out = {
                    "date": row["date"], "ticker": row["ticker"], "name": row["name"],
                    "pass_level": row["pass_level"], "pattern_type": row["pattern_type"],
                    "vol_ratio": row.get("vol_ratio",""), "rs_20d": row.get("rs_20d",""),
                    "ma5_gt_ma20": row.get("ma5_gt_ma20",""), "close_price": row.get("close_price",""),
                    "bear_score": result.bear_score, "bull_score": result.bull_score,
                    "evidence_level": result.evidence_level, "llm_verdict": result.verdict,
                    "bear_reason": result.bear_reason[:120], "bull_reason": result.bull_reason[:120],
                    "predicted_direction": result.predicted_direction,
                    "predicted_low_pct":    f"{result.predicted_low_pct:.2f}",
                    "predicted_high_pct":   f"{result.predicted_high_pct:.2f}",
                    "predicted_center_pct": f"{result.predicted_center_pct:.2f}",
                    "prediction_confidence": f"{result.prediction_confidence:.2f}",
                    "prediction_key_factor": result.prediction_key_factor[:80],
                    "news_mode": news_mode, "company_news_n": comp_n, "broad_news_n": broad_n,
                    "return_1d": row.get("return_1d",""), "return_5d": row.get("return_5d",""),
                    "twii_1d":   row.get("twii_1d",""),
                    "alpha_1d":  bwl._calc_alpha(row.get("return_1d",""), row.get("twii_1d","")),
                    "direction_correct": bwl._direction_correct(
                        result.predicted_direction, row.get("return_1d","")),
                }
                writer.writerow(out)
                f.flush()
                done_count += 1
                logger.info(f"      bear={result.bear_score} bull={result.bull_score} "
                            f"pred={result.predicted_direction}({result.predicted_center_pct:+.1f}%)")
                if batches:
                    _generate_report(batches)
            except Exception as e:
                logger.error(f"      LLM 失敗：{e}")
                writer.writerow({
                    "date": row["date"], "ticker": row["ticker"], "name": row["name"],
                    "pass_level": row["pass_level"], "pattern_type": row["pattern_type"],
                    "bear_reason": f"[ERROR] {str(e)[:100]}", "news_mode": "error",
                    "return_1d": row.get("return_1d",""), "return_5d": row.get("return_5d",""),
                    "twii_1d": row.get("twii_1d",""),
                    "alpha_1d": bwl._calc_alpha(row.get("return_1d",""), row.get("twii_1d","")),
                })
                f.flush()
                if batches:
                    _generate_report(batches)

            time.sleep(delay)

    return done_count


# ── 模擬交易計算 ──────────────────────────────────────────────────────────────

def _paper_trade(llm_rows: list[dict]) -> dict:
    """累積資金池 + 每日加碼 + 期望值加權再平衡 + 風控。

    每個交易日：
      1. 持股以 FinMind 當日收盤估值；首日注入 INITIAL_CAPITAL，其後每日 +DAILY_BUDGET
         （總投入達 MAX_CONTRIBUTION 後停止加碼）。
      2. 每支 edge = 期望報酬(center%) × 信心(confidence)；down/REJECT → edge 0（強制賣）。
         持股當天無訊號 → edge 用上次 ×EDGE_DECAY 衰減（自然老化）。
      3. 選股：edge 排名（持股有 INCUMBENT_BONUS 加成防換手），最多 MAX_SIGNALS
         （第 4 名 ≥ 第 3 名 ×TIE_RATIO 放行）。
      4. 總曝險 = min(EXPOSURE_CAP, 選中股平均信心)（弱訊號自動留現金）。
         權重 ∝ edge；單支股票「當天加碼」≤ DAILY_ADD_CAP（賣出不限、無持有上限）。
      5. 多退少補再平衡到目標金額。
    """
    from tw_stock_agent.tools.finmind_client import get_daily_prices

    # ── 整理每日訊號 ──
    sig_by_date: dict[str, dict[str, dict]] = defaultdict(dict)
    tickers: set[str] = set()
    for r in llm_rows:
        if r.get("llm_verdict","") == "ERROR":
            continue
        d, tk = r.get("date",""), r.get("ticker","")
        if not d or not tk:
            continue
        tickers.add(tk)
        try:
            center = float(r.get("predicted_center_pct","") or 0)
            conf   = float(r.get("prediction_confidence","") or 0)
        except (ValueError, TypeError):
            center, conf = 0.0, 0.0
        is_up = r.get("predicted_direction","") == "up" and r.get("llm_verdict","") != "REJECT"
        sig_by_date[d][tk] = {
            "name": r.get("name", tk), "pattern": r.get("pattern_type",""),
            "bull": r.get("bull_score",""), "bear": r.get("bear_score",""),
            "center": center, "conf": conf,
            "edge": max(0.0, center) / 100.0 * max(0.0, conf) if is_up else 0.0,
        }

    if not sig_by_date:
        return _empty_trade_result()

    # 全域代號→名稱對照（持股當天無訊號時仍能顯示名稱）
    name_map: dict[str, str] = {}
    for day_sig in sig_by_date.values():
        for tk, s in day_sig.items():
            name_map[tk] = s["name"]

    # ── 價格面板（FinMind，本地快取）+ forward-fill 取價 ──
    panel = {tk: get_daily_prices(tk) for tk in tickers}

    def price(tk: str, d: str) -> float | None:
        pr = panel.get(tk, {})
        ds = [x for x in pr if x <= d]
        return pr[max(ds)] if ds else None

    first, last = min(sig_by_date), max(sig_by_date)
    cal = sorted({d for pr in panel.values() for d in pr if first <= d <= last})
    if not cal:
        return _empty_trade_result()

    # ── 逐日模擬 ──
    cash = 0.0
    contributed = 0.0               # 累計總投入
    shares: dict[str, float] = {}
    last_edge: dict[str, float] = {}
    day_pnl: list[float] = []
    equity_curve: list[float] = []
    ledger: list[dict] = []
    prev_equity = 0.0

    for i, d in enumerate(cal):
        # 加碼：第一天注入初始資金，其後每天 +DAILY_BUDGET；總投入達上限即停
        if i == 0:
            add = INITIAL_CAPITAL
        elif contributed < MAX_CONTRIBUTION:
            add = min(DAILY_BUDGET, MAX_CONTRIBUTION - contributed)
        else:
            add = 0.0
        cash += add
        contributed += add
        port = cash + sum(shares[tk] * (price(tk, d) or 0) for tk in shares)

        # 候選 edge：今日訊號 + 持股衰減
        todays = sig_by_date.get(d, {})
        edges: dict[str, float] = {}
        conf_of: dict[str, float] = {}
        for tk, s in todays.items():
            edges[tk] = s["edge"]
            conf_of[tk] = s["conf"]
            last_edge[tk] = s["edge"]
        for tk in shares:
            if tk not in edges:
                e = last_edge.get(tk, 0.0) * EDGE_DECAY
                edges[tk] = e
                last_edge[tk] = e

        # 選股：持股加成排名
        def rank_key(tk: str) -> float:
            return edges[tk] * (INCUMBENT_BONUS if tk in shares else 1.0)

        ranked = sorted([tk for tk in edges if edges[tk] > 0], key=rank_key, reverse=True)
        selected = ranked[:MAX_SIGNALS]
        if len(ranked) > MAX_SIGNALS and rank_key(ranked[MAX_SIGNALS]) >= \
                rank_key(ranked[MAX_SIGNALS - 1]) * TIE_RATIO:
            selected = ranked[:MAX_SIGNALS + 1]

        # 總曝險：信心驅動（強訊號多投、弱訊號留現金），但有標的時至少投下限
        confs = [conf_of[tk] for tk in selected if tk in conf_of]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        if selected:
            target_exposure = min(EXPOSURE_CAP, max(EXPOSURE_FLOOR, avg_conf))
        else:
            target_exposure = 0.0    # 當天無 up 訊號（含持股都衰減完）→ 空手

        # 目標金額：權重 ∝ edge（不設單支持有上限）
        wsum = sum(edges[tk] for tk in selected)
        targets: dict[str, float] = {}
        if wsum > 0 and target_exposure > 0:
            for tk in selected:
                targets[tk] = port * target_exposure * (edges[tk] / wsum)

        # 再平衡：多退少補（含全數出場）；買入時「當天加碼」≤ DAILY_ADD_CAP（賣出不限）
        for tk in set(shares) | set(targets):
            p = price(tk, d)
            if not p:
                continue
            cur = shares.get(tk, 0.0) * p
            tgt = targets.get(tk, 0.0)
            delta = tgt - cur
            if delta > DAILY_ADD_CAP:             # 單日加碼增量超過上限 → 砍到上限
                delta = DAILY_ADD_CAP
                tgt = cur + delta
            cash -= delta                         # 買花現金、賣收現金
            if tgt <= 1e-6:
                shares.pop(tk, None)
            else:
                shares[tk] = tgt / p

        # 收盤估值 + 當日損益（扣掉當天加碼才是真實獲利）
        mv = {tk: shares[tk] * (price(tk, d) or 0) for tk in shares}
        invested = sum(mv.values())
        equity = cash + invested
        pnl = equity - prev_equity - add          # 扣掉當天加碼才是真實獲利
        prev_equity = equity
        day_pnl.append(pnl)
        equity_curve.append(equity)
        held = sorted(mv, key=lambda x: -mv[x])
        ledger.append({
            "date": d,
            "holdings": ", ".join(f"{tk}({name_map.get(tk,'')[:4]}) {mv[tk]:.0f}" for tk in held) or "—",
            "invested": invested, "cash": cash,
            "exposure": invested / equity if equity > 0 else 0.0,
            "pnl": pnl, "equity": equity,
        })

    # ── 統計 ──
    total_contributed = contributed
    final_equity = equity_curve[-1] if equity_curve else 0.0
    total_pnl = sum(day_pnl)
    active = [x for x in day_pnl if abs(x) > 1e-9]
    win_days = sum(1 for x in active if x > 0)
    total_days = len(active)

    cum, peak, max_dd = 0.0, 0.0, 0.0
    for x in day_pnl:
        cum += x
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    if len(active) > 1:
        mean_d = sum(active) / len(active)
        std_d = math.sqrt(sum((x - mean_d) ** 2 for x in active) / len(active))
        sharpe = (mean_d / std_d * math.sqrt(252)) if std_d > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_pnl": total_pnl, "win_days": win_days, "total_days": total_days,
        "max_dd": max_dd, "sharpe": sharpe, "ledger": ledger,
        "total_contributed": total_contributed, "final_equity": final_equity,
        "return_pct": (total_pnl / total_contributed * 100) if total_contributed else 0.0,
        "calendar_days": len(cal),
    }


def _empty_trade_result() -> dict:
    return {"total_pnl": 0, "win_days": 0, "total_days": 0, "max_dd": 0,
            "sharpe": 0, "ledger": [], "total_contributed": 0,
            "final_equity": 0, "return_pct": 0, "calendar_days": 0}


# ── 報告產生 ──────────────────────────────────────────────────────────────────

def _load_llm_rows() -> list[dict]:
    if not LLM_CSV.exists():
        return []
    with LLM_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _batch_stats(rows: list[dict], from_date: str, to_date: str) -> dict:
    batch_rows = [r for r in rows if from_date <= r.get("date","") <= to_date]
    valid  = [r for r in batch_rows if r.get("direction_correct","") in ("0","1")]
    if not valid:
        return {"n": 0, "correct": 0, "total": 0, "a_correct": 0, "a_total": 0,
                "b_correct": 0, "b_total": 0, "a_alpha": None, "b_alpha": None, "twii": None}

    correct = sum(1 for r in valid if r["direction_correct"] == "1")
    a_rows  = [r for r in valid if r.get("pattern_type","none") != "none"]
    b_rows  = [r for r in valid if r.get("pattern_type","none") == "none"]

    def _avg_alpha(rr):
        vals = []
        for r in rr:
            try:
                vals.append(float(r.get("alpha_1d","")))
            except (ValueError, TypeError):
                pass
        return sum(vals)/len(vals) if vals else None

    def _avg_twii(rr):
        vals = []
        for r in rr:
            try:
                vals.append(float(r.get("twii_1d","")))
            except (ValueError, TypeError):
                pass
        return sum(vals)/len(vals) if vals else None

    return {
        "n": len(batch_rows), "correct": correct, "total": len(valid),
        "a_correct": sum(1 for r in a_rows if r["direction_correct"]=="1"),
        "a_total": len(a_rows),
        "b_correct": sum(1 for r in b_rows if r["direction_correct"]=="1"),
        "b_total": len(b_rows),
        "a_alpha": _avg_alpha(a_rows),
        "b_alpha": _avg_alpha(b_rows),
        "twii": _avg_twii(valid),
    }


def _fmt_pct(v, denom) -> str:
    if denom == 0:
        return "—"
    return f"{v}/{denom} = {v/denom:.1%}"


def _generate_report(batches: list[tuple[str, str, int]]) -> None:
    rows = _load_llm_rows()
    pt   = _paper_trade(rows)
    today_str = date.today().isoformat()

    done_batches = [b for b in batches if any(
        b[0] <= r.get("date","") <= b[1] for r in rows)]

    lines: list[str] = []
    lines.append("# LLM 60 天回測報告（含模擬交易）\n")
    lines.append(f"> 更新時間：{today_str}　"
                 f"已完成批次：{', '.join('#'+str(b[2]) for b in done_batches)}\n")

    # ── 總體準確率 ───────────────────────────────────────────────────────────
    valid  = [r for r in rows if r.get("direction_correct","") in ("0","1")]
    corr   = sum(1 for r in valid if r["direction_correct"]=="1")
    a_rows = [r for r in valid if r.get("pattern_type","none") != "none"]
    b_rows = [r for r in valid if r.get("pattern_type","none") == "none"]

    def _avg(lst, key):
        vals = []
        for r in lst:
            try:
                vals.append(float(r.get(key,"")))
            except (ValueError, TypeError):
                pass
        return sum(vals)/len(vals) if vals else None

    a_alpha = _avg(a_rows, "alpha_1d")
    b_alpha = _avg(b_rows, "alpha_1d")
    twii    = _avg(valid, "twii_1d")

    a_corr = sum(1 for r in a_rows if r["direction_correct"]=="1")
    b_corr = sum(1 for r in b_rows if r["direction_correct"]=="1")

    errs = []
    for r in rows:
        try:
            errs.append(abs(float(r.get("predicted_center_pct","")) - float(r.get("return_1d",""))))
        except (ValueError, TypeError):
            pass
    mae_str = f"{sum(errs)/len(errs):.2f}%" if errs else "—"

    all_ret = _avg(rows, "return_1d")
    all_al  = _avg(rows, "alpha_1d")

    lines.append("## 總體準確率（累計）\n")
    lines.append("| 指標 | 數值 |")
    lines.append("|------|------|")
    lines.append(f"| 總訊號數 | {len(rows)} 筆 |")
    lines.append(f"| 有方向預測（非 neutral）| {len(valid)} 筆 |")
    lines.append(f"| **方向準確率** | **{_fmt_pct(corr, len(valid))}** |")
    lines.append(f"| 幅度 MAE | {mae_str} |")
    lines.append(f"| 全部均報酬 | {f'{all_ret:+.2f}%' if all_ret is not None else '—'} |")
    lines.append(f"| **全部均 Alpha** | **{f'{all_al:+.2f}%' if all_al is not None else '—'}** |")
    lines.append(f"| 大盤均日報酬（TWII）| {f'{twii:+.2f}%' if twii is not None else '—'} |")
    lines.append("")

    # ── A 組 vs B 組 ─────────────────────────────────────────────────────────
    a_ret = _avg(a_rows, "return_1d")
    b_ret = _avg(b_rows, "return_1d")
    lines.append("## A 組 vs B 組\n")
    lines.append("| 組別 | 方向準確率 | 均報酬 | **均 Alpha** |")
    lines.append("|------|-----------|--------|------------|")
    lines.append(f"| **A（PASS + 有型態）** | **{_fmt_pct(a_corr, len(a_rows))}** "
                 f"| {f'{a_ret:+.2f}%' if a_ret is not None else '—'} "
                 f"| **{f'{a_alpha:+.2f}%' if a_alpha is not None else '—'}** |")
    lines.append(f"| B（PASS + 無型態） | {_fmt_pct(b_corr, len(b_rows))} "
                 f"| {f'{b_ret:+.2f}%' if b_ret is not None else '—'} "
                 f"| {f'{b_alpha:+.2f}%' if b_alpha is not None else '—'} |")
    lines.append(f"| 大盤（買進持有） | — | "
                 f"{f'{twii:+.2f}%' if twii is not None else '—'} | +0.00% |")
    lines.append("")

    # ── 模擬交易（累積資金池 + 每日加碼 + 再平衡）──────────────────────────────
    lines.append("## 模擬交易（期初 15000＋每日加碼 1000 · 期望值加權再平衡）\n")
    lines.append(f"> 期初 {INITIAL_CAPITAL:,.0f}＋每日 {DAILY_BUDGET:,.0f}（總投入上限 {MAX_CONTRIBUTION:,.0f}）"
                 f"｜單股當天加碼 ≤{DAILY_ADD_CAP:,.0f}、無持有上限"
                 f"｜總曝險 {EXPOSURE_FLOOR:.0%}–{EXPOSURE_CAP:.0%}｜最多 {MAX_SIGNALS} 檔\n")
    lines.append("| 指標 | 數值 |")
    lines.append("|------|------|")
    lines.append(f"| 累計投入本金 | {pt['total_contributed']:,.0f} TWD（期初 {INITIAL_CAPITAL:,.0f}＋每日 {DAILY_BUDGET:,.0f}，上限 {MAX_CONTRIBUTION:,.0f}）|")
    lines.append(f"| 期末權益 | {pt['final_equity']:,.0f} TWD |")
    lines.append(f"| **淨損益** | **{pt['total_pnl']:+,.1f} TWD** |")
    lines.append(f"| **本金報酬率** | **{pt['return_pct']:+.2f}%** |")
    lines.append(f"| 獲利日勝率 | {_fmt_pct(pt['win_days'], pt['total_days'])} |")
    lines.append(f"| 最大回撤 | -{pt['max_dd']:,.1f} TWD |")
    lines.append(f"| 年化 Sharpe | {pt['sharpe']:.2f} |")
    lines.append("")

    # 每日持倉明細（再平衡後快照）
    if pt.get("ledger"):
        lines.append("<details><summary>每日持倉明細（再平衡後快照）</summary>\n")
        lines.append("| 日期 | 持股（代號 金額TWD） | 投入 | 曝險 | 現金 | 當日P&L | 權益 |")
        lines.append("|------|------|------|------|------|---------|------|")
        for t in pt["ledger"]:
            lines.append(f"| {t['date']} | {t['holdings']} "
                         f"| {t['invested']:,.0f} | {t['exposure']:.0%} | {t['cash']:,.0f} "
                         f"| {t['pnl']:+,.1f} | {t['equity']:,.0f} |")
        lines.append("</details>\n")

    # ── 各批次摘要 ────────────────────────────────────────────────────────────
    lines.append("## 各批次摘要\n")
    lines.append("| 批次 | 日期範圍 | 訊號 | 方向準 | A準 | B準 | A均Alpha | B均Alpha | 大盤均 |")
    lines.append("|------|----------|------|--------|-----|-----|----------|----------|--------|")
    for from_d, to_d, bno in batches:
        s = _batch_stats(rows, from_d, to_d)
        if s["n"] == 0:
            lines.append(f"| #{bno} | {from_d}～{to_d} | — | — | — | — | — | — | — |")
        else:
            a_al_str = f"{s['a_alpha']:+.2f}%" if s["a_alpha"] is not None else "—"
            b_al_str = f"{s['b_alpha']:+.2f}%" if s["b_alpha"] is not None else "—"
            tw_str   = f"{s['twii']:+.2f}%"    if s["twii"]    is not None else "—"
            lines.append(
                f"| #{bno} | {from_d}～{to_d} | {s['n']} "
                f"| {_fmt_pct(s['correct'], s['total'])} "
                f"| {_fmt_pct(s['a_correct'], s['a_total'])} "
                f"| {_fmt_pct(s['b_correct'], s['b_total'])} "
                f"| {a_al_str} | {b_al_str} | {tw_str} |"
            )
    lines.append("")

    # ── 各批次明細表 ──────────────────────────────────────────────────────────
    for from_d, to_d, bno in batches:
        batch_rows = [r for r in rows if from_d <= r.get("date","") <= to_d]
        if not batch_rows:
            continue
        s = _batch_stats(rows, from_d, to_d)
        lines.append(f"## 第 {bno} 批（{from_d} ～ {to_d}）\n")
        lines.append(f"- 方向準確率：{_fmt_pct(s['correct'], s['total'])}")
        a_al_str = f"{s['a_alpha']:+.2f}%" if s["a_alpha"] is not None else "—"
        b_al_str = f"{s['b_alpha']:+.2f}%" if s["b_alpha"] is not None else "—"
        lines.append(f"- A 組（有型態）：{_fmt_pct(s['a_correct'], s['a_total'])}"
                     f"，均 Alpha {a_al_str}")
        lines.append(f"- B 組（無型態）：{_fmt_pct(s['b_correct'], s['b_total'])}"
                     f"，均 Alpha {b_al_str}")
        if s["twii"] is not None:
            lines.append(f"- 大盤均日報酬：{s['twii']:+.2f}%")
        lines.append("")
        lines.append("| 日期 | 代號 | 名稱 | 型態 | pred | 預測% | 實際% | Alpha | 結果 |")
        lines.append("|------|------|------|------|------|-------|-------|-------|------|")
        for r in sorted(batch_rows, key=lambda x: (x.get("date",""), x.get("ticker",""))):
            pred  = r.get("predicted_direction","—")
            pred_pct = r.get("predicted_center_pct","?")
            if pred_pct not in ("","?"):
                try:
                    pred_pct = f"{float(pred_pct):+.1f}%"
                except Exception:
                    pred_pct = "?"
            actual = r.get("return_1d","?")
            if actual not in ("","?"):
                try:
                    actual = f"{float(actual):+.2f}%"
                except Exception:
                    actual = "?"
            alpha_v = r.get("alpha_1d","?")
            if alpha_v not in ("","?"):
                try:
                    alpha_v = f"{float(alpha_v):+.2f}%"
                except Exception:
                    alpha_v = "?"
            dc = r.get("direction_correct","")
            result_str = "✓" if dc=="1" else ("✗" if dc=="0" else "—")
            lines.append(
                f"| {r.get('date','')} | {r.get('ticker','')} | {r.get('name','')} "
                f"| {r.get('pattern_type','none')} | {pred} | {pred_pct} "
                f"| {actual} | {alpha_v} | {result_str} |"
            )
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"報告已更新 → {REPORT_PATH}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    global REPORT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan",        action="store_true", help="只顯示批次規劃")
    parser.add_argument("--report-only", action="store_true", help="只重新產生報告")
    parser.add_argument("--batch", type=int, default=0, help="只跑第 N 批（0=全部）")
    parser.add_argument("--days", type=int, default=60, help="回測交易日數（預設 60，可設 90）")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--tickers", default="",
                        help="覆蓋股票池（逗號分隔），預設用 18 支主要股")
    parser.add_argument("--base", action="store_true",
                        help="用 base_universe.json 全部（~112 檔）")
    args = parser.parse_args()

    tickers = DEFAULT_TICKERS
    if args.base:
        import json as _j
        tickers = set(_j.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8")))
    elif args.tickers:
        tickers = set(t.strip() for t in args.tickers.split(",") if t.strip())

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    if args.days != 60:
        REPORT_PATH = ROOT / "reports" / f"llm_backtest_{args.days}d.md"
    batches = _compute_batches(n_days=args.days)
    if not batches:
        logger.error("無法計算批次，請確認 backtest_one_month.csv 存在")
        return

    if args.plan:
        print(f"\n{'='*70}")
        print(f"  60 天 LLM 回測批次規劃（{len(batches)} 批 × 5 交易日，{len(tickers)} 支股票）")
        print(f"{'='*70}")
        rows_all = _load_llm_rows()
        total_todo = 0
        for from_d, to_d, bno in batches:
            done_n = sum(1 for r in rows_all
                         if from_d <= r.get("date","") <= to_d
                         and r.get("ticker","") in tickers)
            qrows = []
            if QUANT_CSV.exists():
                with QUANT_CSV.open(encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if (r.get("pass_level")=="PASS"
                                and r.get("ticker","") in tickers
                                and from_d <= r["date"] <= to_d):
                            qrows.append(r)
            a_n     = sum(1 for r in qrows if r.get("pattern_type","none") != "none")
            total_q = len(qrows)
            todo_n  = max(0, total_q - done_n)
            total_todo += todo_n
            est_min = todo_n * 50 // 60
            status  = "完成" if done_n >= total_q and total_q > 0 else f"待跑 {todo_n}"
            print(f"  #{bno:2d}  {from_d}~{to_d}  PASS={total_q}(A={a_n})  "
                  f"已完={done_n}  {status}  ~{est_min}分")
        print(f"\n  總待跑：{total_todo} 筆，預估 ~{total_todo*50//60} 分鐘")
        print()
        return

    if args.report_only:
        _generate_report(batches)
        return

    target_batches = ([b for b in batches if b[2] == args.batch]
                      if args.batch else batches)

    # 預抓 FinMind 資料（三大法人 + 月營收）並快取，避免 LLM loop 中打 API
    try:
        from tw_stock_agent.tools.finmind_client import prefetch_all
        logger.info("預抓 FinMind 三大法人 & 月營收資料...")
        prefetch_all(sorted(tickers), delay=0.3)
    except Exception as e:
        logger.warning(f"FinMind 預抓失敗（繼續回測，籌碼面資料將即時補抓）：{e}")

    for from_d, to_d, bno in target_batches:
        logger.info(f"{'='*60}")
        logger.info(f"批次 #{bno}：{from_d} ～ {to_d}")
        n = _run_batch(from_d, to_d, bno, delay=args.delay, tickers=tickers, batches=batches)
        logger.info(f"批次 #{bno} 完成 {n} 筆")

    logger.success("全部批次完成！")


if __name__ == "__main__":
    main()
