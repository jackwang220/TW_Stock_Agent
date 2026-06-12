"""預測準確率驗證：比較 AI 隔日漲跌預測 vs 實際漲跌。

使用方式：
    uv run python scripts/check_predictions.py
    uv run python scripts/check_predictions.py --backfill   # 先回填再分析
    uv run python scripts/check_predictions.py --min-conf 0.5  # 只看高把握度預測
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger
from tw_stock_agent.signal_log import backfill_returns, load_signal_log


def _direction_from_return(ret: float) -> str:
    if ret > 0.2:
        return "up"
    if ret < -0.2:
        return "down"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description="預測準確率分析")
    parser.add_argument("--backfill", action="store_true", help="執行前先回填 return_1d")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="只分析 prediction_confidence >= 此值的預測（預設 0.0 = 全部）")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    if args.backfill:
        updated = backfill_returns()
        print(f"回填了 {updated} 個欄位\n")

    rows = load_signal_log()
    if not rows:
        print("signal_log.csv 是空的，請先執行 daily_stock_scan.py")
        return

    # 篩選有預測且有 return_1d 的列
    valid = []
    for r in rows:
        pred_dir = r.get("predicted_direction", "")
        ret_str = r.get("return_1d", "")
        conf_str = r.get("prediction_confidence", "0")
        if not pred_dir or not ret_str:
            continue
        try:
            ret = float(ret_str)
            conf = float(conf_str) if conf_str else 0.0
        except ValueError:
            continue
        if conf < args.min_conf:
            continue
        valid.append({**r, "_ret": ret, "_conf": conf, "_pred_dir": pred_dir})

    if not valid:
        print(f"沒有可分析的資料（有預測且已回填 return_1d）。\n"
              f"總筆數：{len(rows)}，含預測欄：{sum(1 for r in rows if r.get('predicted_direction'))}\n"
              f"請執行：uv run python scripts/check_predictions.py --backfill")
        return

    # ── 整體統計 ──────────────────────────────────────────────────────
    total = len(valid)
    dir_correct = sum(
        1 for r in valid
        if r["_pred_dir"] != "neutral"
        and r["_pred_dir"] == _direction_from_return(r["_ret"])
    )
    dir_predicted = sum(1 for r in valid if r["_pred_dir"] != "neutral")
    neutral_cnt = total - dir_predicted

    center_errors = [
        abs(float(r.get("predicted_center_pct", 0)) - r["_ret"])
        for r in valid
        if r.get("predicted_center_pct", "")
    ]
    mae = sum(center_errors) / len(center_errors) if center_errors else 0.0
    biases = [
        float(r.get("predicted_center_pct", 0)) - r["_ret"]
        for r in valid
        if r.get("predicted_center_pct", "")
    ]
    mean_bias = sum(biases) / len(biases) if biases else 0.0

    print("=" * 60)
    print(f"  AI 隔日漲跌預測準確率報告")
    if args.min_conf > 0:
        print(f"  過濾：把握度 >= {args.min_conf:.0%}")
    print("=" * 60)
    print(f"\n【樣本數】 {total} 筆（含方向預測 {dir_predicted}，中性 {neutral_cnt}）")
    print(f"\n【方向準確率】")
    if dir_predicted:
        acc = dir_correct / dir_predicted
        print(f"  預測有方向（up/down）共 {dir_predicted} 次")
        print(f"  方向正確 {dir_correct} 次　→　{acc:.1%}")
    else:
        print("  無方向性預測（全部 neutral）")

    print(f"\n【幅度誤差（vs predicted_center）】")
    print(f"  平均絕對誤差（MAE）：{mae:.2f}%")
    print(f"  平均偏差（正=預測過高）：{mean_bias:+.2f}%")

    # ── 按 verdict 分組 ───────────────────────────────────────────────
    print(f"\n【按 Verdict 分組】")
    for verdict in ["PASS", "WARN", "REJECT"]:
        grp = [r for r in valid if r.get("verdict") == verdict]
        if not grp:
            continue
        g_dir = sum(1 for r in grp
                    if r["_pred_dir"] != "neutral"
                    and r["_pred_dir"] == _direction_from_return(r["_ret"]))
        g_dp = sum(1 for r in grp if r["_pred_dir"] != "neutral")
        g_errs = [abs(float(r.get("predicted_center_pct", 0)) - r["_ret"]) for r in grp
                  if r.get("predicted_center_pct", "")]
        g_mae = sum(g_errs) / len(g_errs) if g_errs else 0.0
        dir_str = f"{g_dir}/{g_dp} ({g_dir/g_dp:.0%})" if g_dp else "—"
        print(f"  {verdict:6s}  n={len(grp):3d}  方向準:{dir_str:12s}  MAE={g_mae:.2f}%")

    # ── 高把握度 vs 低把握度 ──────────────────────────────────────────
    high_conf = [r for r in valid if r["_conf"] >= 0.6]
    low_conf  = [r for r in valid if r["_conf"] < 0.6]
    print(f"\n【把握度分組（≥0.6 vs <0.6）】")
    for label, grp in [("高把握 ≥0.6", high_conf), ("低把握 <0.6", low_conf)]:
        if not grp:
            continue
        g_dp = sum(1 for r in grp if r["_pred_dir"] != "neutral")
        g_dir = sum(1 for r in grp
                    if r["_pred_dir"] != "neutral"
                    and r["_pred_dir"] == _direction_from_return(r["_ret"]))
        dir_str = f"{g_dir}/{g_dp} ({g_dir/g_dp:.0%})" if g_dp else "—"
        print(f"  {label}  n={len(grp):3d}  方向準:{dir_str}")

    # ── 最近 10 筆明細 ────────────────────────────────────────────────
    print(f"\n【最近 10 筆明細】")
    print(f"  {'日期':10s}  {'代號':6s}  {'預測':8s}  {'中心值':7s}  {'實際':7s}  {'方向':4s}  {'把握':5s}")
    print("  " + "-" * 60)
    for r in valid[-10:]:
        pred_str = f"{float(r.get('predicted_center_pct',0)):+.1f}%"
        actual_str = f"{r['_ret']:+.1f}%"
        is_correct = (r["_pred_dir"] != "neutral"
                      and r["_pred_dir"] == _direction_from_return(r["_ret"]))
        ok = "✓" if is_correct else ("—" if r["_pred_dir"] == "neutral" else "✗")
        print(f"  {r['date']:10s}  {r['ticker']:6s}  "
              f"{r['_pred_dir']:8s}  {pred_str:7s}  {actual_str:7s}  {ok:4s}  "
              f"{r['_conf']:.0%}")

    print()


if __name__ == "__main__":
    main()
