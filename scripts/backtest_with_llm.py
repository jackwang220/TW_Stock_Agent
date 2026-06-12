"""歷史 LLM 回測：對過去每個 PASS/WARN 量化訊號，用 Google News 歷史日期搜尋
還原當天新聞，再跑 Bear/Bull + 隔日漲跌預測，事後對比實際結果。

【說明】
- 費 token：每支股票每天 3 次 LLM 呼叫（Bear + Bull + Predict）
- 退一步：若 Google News 搜不到足夠新聞（< 3 篇），改為純技術面分析
- 前置條件：先跑 backtest_one_month.py 產出 data/backtest_one_month.csv

使用方式：
    uv run python scripts/backtest_with_llm.py
    uv run python scripts/backtest_with_llm.py --pass-only       # 只跑 PASS 訊號
    uv run python scripts/backtest_with_llm.py --resume          # 跳過已做完的
    uv run python scripts/backtest_with_llm.py --limit 20        # 只跑前 20 筆（測試用）
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.debate.bear import run_debate

QUANT_CSV = DATA_DIR / "backtest_one_month.csv"
OUTPUT    = DATA_DIR / "backtest_llm_results.csv"

FIELDS = [
    # 量化資料（從 quant CSV 帶過來）
    "date", "ticker", "name", "pass_level", "pattern_type",
    "vol_ratio", "rs_20d", "ma5_gt_ma20", "close_price",
    # LLM 分析結果
    "bear_score", "bull_score", "evidence_level", "llm_verdict",
    "bear_reason", "bull_reason",
    # AI 隔日預測
    "predicted_direction", "predicted_low_pct", "predicted_high_pct",
    "predicted_center_pct", "prediction_confidence", "prediction_key_factor",
    # 新聞品質（回測模式）
    "news_mode",        # "historical" or "technical_only"
    "company_news_n",   # 搜到幾篇個股新聞
    "broad_news_n",     # 搜到幾篇廣域新聞
    # 實際報酬（從 quant CSV 帶過來）
    "return_1d", "return_5d",
    # 準確度（自動計算）
    "direction_correct",  # 1=正確 0=錯誤 空=無資料
    # Alpha（與大盤對比）
    "twii_1d",    # 大盤同日報酬
    "alpha_1d",   # return_1d - twii_1d
]


def _load_quant_results(group: str, from_date: str = "", to_date: str = "",
                        levels: list | None = None) -> list[dict]:
    """載入量化回測結果，依 group 篩選。

    group:  "a" = 有型態, "b" = 無型態, "ab" = 兩組
    levels: 允許的 pass_level 清單，預設 ["PASS"]
            傳入 ["PASS", "WARN"] 可補入 WARN 訊號
    """
    if levels is None:
        levels = ["PASS"]
    if not QUANT_CSV.exists():
        logger.error(f"找不到 {QUANT_CSV}，請先執行 backtest_one_month.py")
        sys.exit(1)
    rows = []
    with QUANT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("pass_level") not in levels:
                continue
            has_pattern = r.get("pattern_type", "none") != "none"
            if group == "a" and not has_pattern:
                continue
            if group == "b" and has_pattern:
                continue
            # 日期篩選
            if from_date and r["date"] < from_date:
                continue
            if to_date and r["date"] > to_date:
                continue
            rows.append(r)
    return rows


def _load_done_keys(resume: bool) -> set[str]:
    if not resume or not OUTPUT.exists():
        return set()
    done = set()
    with OUTPUT.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add(f"{r['date']}_{r['ticker']}")
    logger.info(f"Resume: {len(done)} already done, skipping.")
    return done


def _calc_alpha(ret_str: str, twii_str: str) -> str:
    try:
        return f"{float(ret_str) - float(twii_str):.2f}"
    except (ValueError, TypeError):
        return ""


def _direction_correct(predicted: str, actual_ret_str: str) -> str:
    if not actual_ret_str or not predicted or predicted == "neutral":
        return ""
    try:
        ret = float(actual_ret_str)
    except ValueError:
        return ""
    actual_dir = "up" if ret > 0.2 else ("down" if ret < -0.2 else "neutral")
    if actual_dir == "neutral":
        return ""
    return "1" if predicted == actual_dir else "0"


def _stock_from_row(row: dict) -> dict:
    """把 quant CSV 的一行還原成 run_debate 需要的 stock dict。"""
    return {
        "code": row["ticker"],
        "name": row["name"],
        "yf_ticker": f"{row['ticker']}.TW",
        "pass_level": row["pass_level"],
        "pattern_type": row["pattern_type"],
        "pattern_detail": row.get("pattern_type", ""),
        "close_price": float(row.get("close_price") or 0),
        "volume_ratio": float(row.get("vol_ratio") or 0),
        "ma5_gt_ma20": row.get("ma5_gt_ma20", "False") == "True",
        "rs_20d": float(row.get("rs_20d") or 1.0),
        "weeks_52_warn": False,
        "rsi_14": 0.0,
        "macd_hist": 0.0,
    }


def _print_plan() -> None:
    """顯示所有 5 天批次的訊號數與預估時間。"""
    if not QUANT_CSV.exists():
        print("找不到 backtest_one_month.csv，請先執行 backtest_one_month.py")
        return

    all_rows = []
    with QUANT_CSV.open(encoding="utf-8") as f:
        all_rows = [r for r in csv.DictReader(f) if r.get("pass_level") == "PASS"]

    if not all_rows:
        print("沒有 PASS 訊號")
        return

    dates = sorted(set(r["date"] for r in all_rows))
    from datetime import date as date_cls, timedelta

    # 5 天一批（日曆天）
    start = date_cls.fromisoformat(dates[0])
    end   = date_cls.fromisoformat(dates[-1])

    # 已完成的
    done = _load_done_keys(resume=True)

    print(f"\n{'='*70}")
    print(f"  LLM 回測批次規劃 — PASS 訊號分組")
    print(f"  A組=PASS+有型態  B組=PASS+無型態  每筆約 50 秒")
    print(f"{'='*70}")
    print(f"  {'批次':4s}  {'日期範圍':22s}  {'A組':4s}  {'B組':4s}  {'已完成':5s}  {'預估時間':8s}  指令")
    print(f"  {'-'*65}")

    batch = 1
    cur = start
    while cur <= end:
        w_start = cur.isoformat()
        w_end   = (cur + timedelta(days=4)).isoformat()
        a_n = sum(1 for r in all_rows
                  if w_start <= r["date"] <= w_end and r.get("pattern_type","none") != "none")
        b_n = sum(1 for r in all_rows
                  if w_start <= r["date"] <= w_end and r.get("pattern_type","none") == "none")
        total = a_n + b_n
        done_n = sum(1 for r in all_rows
                     if w_start <= r["date"] <= w_end
                     and f"{r['date']}_{r['ticker']}" in done)
        est_s = (total - done_n) * 50
        est_str = f"~{est_s//60}分{est_s%60:02d}秒" if total > 0 else "（空）"
        cmd = f"--from-date {w_start} --to-date {w_end}"
        print(f"  #{batch:2d}    {w_start}~{w_end}  {a_n:4d}  {b_n:4d}  {done_n:5d}  {est_str:8s}  {cmd}")
        cur += timedelta(days=5)
        batch += 1

    total_a = sum(1 for r in all_rows if r.get("pattern_type","none") != "none")
    total_b = sum(1 for r in all_rows if r.get("pattern_type","none") == "none")
    total_done = len(done)
    total_est = (total_a + total_b - total_done) * 50
    print(f"  {'─'*65}")
    print(f"  合計：A={total_a} + B={total_b} = {total_a+total_b} 筆，"
          f"已完成={total_done}，剩餘預估 ~{total_est//60} 分鐘")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="ab",
                        help="跑哪個組：a=有型態, b=無型態, ab=兩組都跑（預設 ab）")
    parser.add_argument("--from-date", default="", help="起始日期 YYYY-MM-DD（含）")
    parser.add_argument("--to-date",   default="", help="結束日期 YYYY-MM-DD（含）")
    parser.add_argument("--include-warn", action="store_true",
                        help="也納入 WARN 訊號（預設只跑 PASS）")
    parser.add_argument("--min-pass", type=int, default=0,
                        help="若 PASS 訊號數低於此值，自動補入 WARN（例如 --min-pass 3）")
    parser.add_argument("--plan", action="store_true",
                        help="只顯示各批次訊號數，不實際跑 LLM")
    parser.add_argument("--resume", action="store_true", help="跳過已完成的")
    parser.add_argument("--limit", type=int, default=0, help="最多跑幾筆（0=全部）")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="每次 LLM 呼叫後等待秒數（避免 rate limit，預設 2s）")
    parser.add_argument("--tickers", default="",
                        help="只跑指定股票（逗號分隔，例如 2330,2454,3017）")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    # ── --plan 模式：只顯示批次規劃，不跑 LLM ────────────────────────────
    if args.plan:
        _print_plan()
        return

    # 決定要跑哪些 pass_level
    levels = ["PASS"]
    if args.include_warn:
        levels = ["PASS", "WARN"]
    elif args.min_pass > 0:
        pass_only = _load_quant_results(args.group, args.from_date, args.to_date, levels=["PASS"])
        if len(pass_only) < args.min_pass:
            logger.info(f"PASS 訊號數 {len(pass_only)} < --min-pass {args.min_pass}，自動補入 WARN")
            levels = ["PASS", "WARN"]

    rows = _load_quant_results(args.group, args.from_date, args.to_date, levels=levels)

    # 股票篩選
    if args.tickers:
        allowed = set(t.strip() for t in args.tickers.split(",") if t.strip())
        rows = [r for r in rows if r["ticker"] in allowed]
        logger.info(f"--tickers 篩選後：{len(rows)} 筆（{len(allowed)} 支股票）")

    done = _load_done_keys(args.resume)
    todo = [r for r in rows if f"{r['date']}_{r['ticker']}" not in done]
    if args.limit:
        todo = todo[:args.limit]

    pass_n = sum(1 for r in todo if r.get("pass_level") == "PASS")
    warn_n = sum(1 for r in todo if r.get("pass_level") == "WARN")
    lvl_label = "+".join(levels)
    logger.info(f"Level={lvl_label} Group={args.group.upper()}：PASS={pass_n} WARN={warn_n} 共 {len(todo)} 筆")
    if not todo:
        logger.warning("沒有待分析的資料（已全部完成或日期範圍無訊號）")
        return

    # 估計費用
    approx_calls = len(todo) * 3
    logger.info(f"預估 LLM 呼叫次數：~{approx_calls} 次（每筆 3 次：Bear + Bull + Predict）")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # 有指定日期範圍時預設 append（避免覆蓋其他批次資料），只有全量重跑才清檔
    append_mode = args.resume or bool(args.from_date or args.to_date)
    write_header = not OUTPUT.exists() or not append_mode

    with OUTPUT.open("a" if append_mode else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, row in enumerate(todo, 1):
            target_date = date.fromisoformat(row["date"])
            stock = _stock_from_row(row)
            key = f"{row['date']}_{row['ticker']}"

            logger.info(f"[{i}/{len(todo)}] {row['date']} {row['ticker']} {row['name']} "
                        f"({row['pass_level']} / {row['pattern_type']})")

            try:
                result = run_debate(stock, [], historical_date=target_date)

                # 判斷新聞模式（搜到多少篇）
                from tw_stock_agent.news.scanner import search_news_around_date
                comp_news = search_news_around_date(row["name"], target_date,
                                                    window_days=1, max_articles=12)
                broad_news = search_news_around_date(
                    "台灣科技股 OR 半導體 OR AI伺服器", target_date,
                    window_days=1, max_articles=5,
                )
                comp_n = sum(1 for a in comp_news if "[搜尋失敗]" not in a.get("title",""))
                broad_n = sum(1 for a in broad_news if "[搜尋失敗]" not in a.get("title",""))
                news_mode = "historical" if (comp_n + broad_n) >= 3 else "technical_only"

                out = {
                    "date": row["date"],
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "pass_level": row["pass_level"],
                    "pattern_type": row["pattern_type"],
                    "vol_ratio": row.get("vol_ratio", ""),
                    "rs_20d": row.get("rs_20d", ""),
                    "ma5_gt_ma20": row.get("ma5_gt_ma20", ""),
                    "close_price": row.get("close_price", ""),
                    "bear_score": result.bear_score,
                    "bull_score": result.bull_score,
                    "evidence_level": result.evidence_level,
                    "llm_verdict": result.verdict,
                    "bear_reason": result.bear_reason[:120],
                    "bull_reason": result.bull_reason[:120],
                    "predicted_direction": result.predicted_direction,
                    "predicted_low_pct": f"{result.predicted_low_pct:.2f}",
                    "predicted_high_pct": f"{result.predicted_high_pct:.2f}",
                    "predicted_center_pct": f"{result.predicted_center_pct:.2f}",
                    "prediction_confidence": f"{result.prediction_confidence:.2f}",
                    "prediction_key_factor": result.prediction_key_factor[:80],
                    "news_mode": news_mode,
                    "company_news_n": comp_n,
                    "broad_news_n": broad_n,
                    "return_1d": row.get("return_1d", ""),
                    "return_5d": row.get("return_5d", ""),
                    "twii_1d":   row.get("twii_1d", ""),
                    "alpha_1d":  _calc_alpha(row.get("return_1d", ""), row.get("twii_1d", "")),
                    "direction_correct": _direction_correct(
                        result.predicted_direction, row.get("return_1d", "")
                    ),
                }
                writer.writerow(out)
                f.flush()

                lvl_str = f"bear={result.bear_score} bull={result.bull_score} " \
                          f"pred={result.predicted_direction}({result.predicted_center_pct:+.1f}%) " \
                          f"news={comp_n}+{broad_n}"
                logger.info(f"  -> {lvl_str}")

            except Exception as e:
                logger.error(f"  LLM 失敗：{e}")
                writer.writerow({
                    "date": row["date"], "ticker": row["ticker"], "name": row["name"],
                    "pass_level": row["pass_level"], "pattern_type": row["pattern_type"],
                    "bear_reason": f"[ERROR] {str(e)[:100]}",
                    "news_mode": "error",
                    "return_1d": row.get("return_1d", ""),
                    "return_5d": row.get("return_5d", ""),
                    "twii_1d":  row.get("twii_1d", ""),
                    "alpha_1d": _calc_alpha(row.get("return_1d", ""), row.get("twii_1d", "")),
                })
                f.flush()

            time.sleep(args.delay)

    logger.success(f"完成 → {OUTPUT}")
    _print_summary()


def _print_summary() -> None:
    if not OUTPUT.exists():
        return
    rows = []
    with OUTPUT.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if r.get("direction_correct") in ("0", "1")]
    if not valid:
        print("\n尚無可計算準確率的資料（return_1d 未回填或 predicted_direction=neutral）")
        return

    correct = sum(1 for r in valid if r["direction_correct"] == "1")
    total = len(valid)
    print(f"\n{'='*60}")
    print(f"  AI 隔日預測準確率（歷史回測）")
    print(f"{'='*60}")
    print(f"  方向預測：{correct}/{total} = {correct/total:.1%}")

    import numpy as np
    errs = []
    for r in rows:
        try:
            pred = float(r.get("predicted_center_pct", ""))
            actual = float(r.get("return_1d", ""))
            errs.append(abs(pred - actual))
        except (ValueError, TypeError):
            pass
    if errs:
        print(f"  幅度 MAE：{np.mean(errs):.2f}%（{len(errs)} 筆）")

    # 按有沒有新聞分組
    for mode in ["historical", "technical_only"]:
        grp = [r for r in valid if r.get("news_mode") == mode]
        if grp:
            g_c = sum(1 for r in grp if r["direction_correct"] == "1")
            print(f"  {mode:15s}：{g_c}/{len(grp)} = {g_c/len(grp):.1%}")

    print(f"\n  結果儲存 -> {OUTPUT}\n")


if __name__ == "__main__":
    main()
