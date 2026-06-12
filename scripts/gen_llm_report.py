"""從 backtest_llm_results.csv 產生 Markdown 回測報告。

使用方式：
    uv run python scripts/gen_llm_report.py
    uv run python scripts/gen_llm_report.py --batch 1      # 只看第 1 批
    uv run python scripts/gen_llm_report.py --out reports/llm_backtest.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tw_stock_agent.config import DATA_DIR

INPUT_CSV = DATA_DIR / "backtest_llm_results.csv"
REPORT_DIR = ROOT / "reports"

BATCH_RANGES = [
    ("2026-04-16", "2026-04-20"),
    ("2026-04-21", "2026-04-25"),
    ("2026-04-26", "2026-04-30"),
    ("2026-05-01", "2026-05-05"),
    ("2026-05-06", "2026-05-10"),
    ("2026-05-11", "2026-05-15"),
    ("2026-05-16", "2026-05-20"),
    ("2026-05-21", "2026-05-25"),
    ("2026-05-26", "2026-05-30"),
    ("2026-05-31", "2026-06-04"),
    ("2026-06-05", "2026-06-09"),
]


def _load(from_date: str | None = None, to_date: str | None = None) -> list[dict]:
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if from_date:
        rows = [r for r in rows if r["date"] >= from_date]
    if to_date:
        rows = [r for r in rows if r["date"] <= to_date]
    return rows


def _grp(rows: list[dict]) -> dict:
    directional, correct, wrong = [], [], []
    a_dir_c = a_dir_t = 0
    b_dir_c = b_dir_t = 0
    a_rets, b_rets, a_alphas, b_alphas = [], [], [], []
    all_rets, all_alphas, all_twii = [], [], []

    for r in rows:
        has_pattern = r.get("pattern_type", "none") != "none"
        ret = r.get("return_1d", "")
        alpha = r.get("alpha_1d", "")
        twii = r.get("twii_1d", "")
        dc = r.get("direction_correct", "")

        if ret:
            all_rets.append(float(ret))
            if has_pattern:
                a_rets.append(float(ret))
            else:
                b_rets.append(float(ret))
        if alpha:
            all_alphas.append(float(alpha))
            if has_pattern:
                a_alphas.append(float(alpha))
            else:
                b_alphas.append(float(alpha))
        if twii:
            all_twii.append(float(twii))

        if dc in ("0", "1"):
            directional.append(r)
            if dc == "1":
                correct.append(r)
            else:
                wrong.append(r)
            if has_pattern:
                a_dir_t += 1
                if dc == "1":
                    a_dir_c += 1
            else:
                b_dir_t += 1
                if dc == "1":
                    b_dir_c += 1

    return {
        "total": len(rows),
        "directional": len(directional),
        "correct": len(correct),
        "wrong": len(wrong),
        "all_rets": all_rets, "all_alphas": all_alphas, "all_twii": all_twii,
        "a_dir_c": a_dir_c, "a_dir_t": a_dir_t, "a_rets": a_rets, "a_alphas": a_alphas,
        "b_dir_c": b_dir_c, "b_dir_t": b_dir_t, "b_rets": b_rets, "b_alphas": b_alphas,
    }


def _pct(c, t):
    return f"{c}/{t} = {c/t*100:.1f}%" if t else "—"


def _avg(lst):
    return f"{np.mean(lst):+.2f}%" if lst else "—"


def _mad(rows):
    errs = []
    for r in rows:
        ret = r.get("return_1d", "")
        pc = r.get("predicted_center_pct", "")
        if ret and pc:
            errs.append(abs(float(ret) - float(pc)))
    return f"{np.mean(errs):.2f}%" if errs else "—"


def _batch_num(d: str) -> int:
    for i, (s, e) in enumerate(BATCH_RANGES, 1):
        if s <= d <= e:
            return i
    return 0


def _fmt(val: str, plus: bool = True) -> str:
    try:
        f = float(val)
        return f"{f:+.2f}%" if plus else f"{f:.2f}%"
    except (ValueError, TypeError):
        return "?"


def _detail_table(rows: list[dict]) -> str:
    lines = []
    lines.append("| 日期 | 代號 | 名稱 | 型態 | pred | 預測% | 實際% | Alpha | 結果 |")
    lines.append("|------|------|------|------|------|-------|-------|-------|------|")
    for r in rows:
        pred_c = r.get("predicted_center_pct", "")
        ret = r.get("return_1d", "")
        alpha = r.get("alpha_1d", "")
        dc = r.get("direction_correct", "")
        icon = {"1": "✓", "0": "✗"}.get(dc, "—")
        pred_str = f"+{float(pred_c):.1f}%" if pred_c and float(pred_c) >= 0 else (f"{float(pred_c):.1f}%" if pred_c else "?")
        lines.append(
            f"| {r['date']} | {r['ticker']} | {r['name']} | {r.get('pattern_type','none')} "
            f"| {r.get('predicted_direction','')} | {pred_str} | {_fmt(ret)} | {_fmt(alpha)} | {icon} |"
        )
    return "\n".join(lines)


def generate(rows: list[dict], batches_done: list[int], today: str) -> str:
    g = _grp(rows)
    lines = []

    lines.append(f"# LLM 隔日預測回測報告")
    lines.append(f"")
    lines.append(f"> 更新時間：{today}　已完成批次：{', '.join(f'#{b}' for b in batches_done)}")
    lines.append(f"")

    twii_avg = _avg(g['all_twii'])

    # 總體摘要
    lines.append(f"## 總體準確率（累計）")
    lines.append(f"")
    lines.append(f"| 指標 | 數值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 總訊號數 | {g['total']} 筆 |")
    lines.append(f"| 有方向預測（非 neutral）| {g['directional']} 筆 |")
    lines.append(f"| **方向準確率** | **{_pct(g['correct'], g['directional'])}** |")
    lines.append(f"| 幅度 MAE | {_mad(rows)} |")
    lines.append(f"| 全部均報酬 | {_avg(g['all_rets'])} |")
    lines.append(f"| **全部均 Alpha** | **{_avg(g['all_alphas'])}** |")
    lines.append(f"| 大盤均日報酬（TWII）| {twii_avg} |")
    lines.append(f"")

    # A vs B
    lines.append(f"## A 組 vs B 組")
    lines.append(f"")
    lines.append(f"| 組別 | 方向準確率 | 均報酬 | **均 Alpha** |")
    lines.append(f"|------|-----------|--------|------------|")
    lines.append(f"| **A（PASS + 有型態）** | **{_pct(g['a_dir_c'], g['a_dir_t'])}** | {_avg(g['a_rets'])} | **{_avg(g['a_alphas'])}** |")
    lines.append(f"| B（PASS + 無型態） | {_pct(g['b_dir_c'], g['b_dir_t'])} | {_avg(g['b_rets'])} | {_avg(g['b_alphas'])} |")
    lines.append(f"| 大盤（買進持有） | — | {twii_avg} | +0.00% |")
    lines.append(f"")

    # 按批次
    lines.append(f"## 各批次摘要")
    lines.append(f"")
    lines.append(f"| 批次 | 日期範圍 | 訊號 | 方向準 | A準 | B準 | A均Alpha | B均Alpha | 大盤均 |")
    lines.append(f"|------|----------|------|--------|-----|-----|----------|----------|--------|")
    for bn in batches_done:
        s, e = BATCH_RANGES[bn - 1]
        br = [r for r in rows if s <= r["date"] <= e]
        bg = _grp(br)
        lines.append(
            f"| #{bn} | {s}～{e} | {bg['total']} | {_pct(bg['correct'], bg['directional'])} "
            f"| {_pct(bg['a_dir_c'], bg['a_dir_t'])} | {_pct(bg['b_dir_c'], bg['b_dir_t'])} "
            f"| {_avg(bg['a_alphas'])} | {_avg(bg['b_alphas'])} | {_avg(bg['all_twii'])} |"
        )
    lines.append(f"")

    # 各批次逐筆
    for bn in batches_done:
        s, e = BATCH_RANGES[bn - 1]
        br = [r for r in rows if s <= r["date"] <= e]
        bg = _grp(br)
        lines.append(f"## 第 {bn} 批（{s} ～ {e}）")
        lines.append(f"")
        lines.append(f"- 方向準確率：{_pct(bg['correct'], bg['directional'])}")
        lines.append(f"- MAE：{_mad(br)}")
        lines.append(f"- A 組（有型態）：{_pct(bg['a_dir_c'], bg['a_dir_t'])}，均 {_avg(bg['a_rets'])}，Alpha {_avg(bg['a_alphas'])}")
        lines.append(f"- B 組（無型態）：{_pct(bg['b_dir_c'], bg['b_dir_t'])}，均 {_avg(bg['b_rets'])}，Alpha {_avg(bg['b_alphas'])}")
        lines.append(f"- 大盤均日報酬：{_avg(bg['all_twii'])}")
        lines.append(f"")
        lines.append(_detail_table(br))
        lines.append(f"")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPORT_DIR / "llm_backtest.md"))
    args = parser.parse_args()

    rows = _load()
    if not rows:
        print("backtest_llm_results.csv 是空的，請先跑回測。")
        return

    batches_done = sorted(set(_batch_num(r["date"]) for r in rows if _batch_num(r["date"]) > 0))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = generate(rows, batches_done, date.today().isoformat())
    out_path.write_text(md, encoding="utf-8")
    print(f"報告已儲存 -> {out_path}")
    print(f"批次：{batches_done}，總 {len(rows)} 筆")


if __name__ == "__main__":
    main()
