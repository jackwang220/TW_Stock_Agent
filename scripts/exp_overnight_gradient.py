"""今日收盤強度 → 隔夜溢價梯度。
buy 在收盤時,觀察得到「今天漲跌幅」;測:今天越強 → 隔天開盤跳空(隔夜)越大嗎?
重點桶:+5~9.4%(強但未鎖漲停=買得到) 的隔夜溢價,值不值得在 ⑤ 買腿加「今日強度」傾斜。
259檔｜還原日線｜2021~。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2021-01-01"
# (label, lo, hi) 今日 close-to-close 漲跌幅分桶
BINS = [("≤ -5%", -1.0, -0.05), ("-5~-2%", -0.05, -0.02), ("-2~0%", -0.02, 0.0),
        ("0~2%", 0.0, 0.02), ("2~5%", 0.02, 0.05), ("5~7%", 0.05, 0.07),
        ("7~9.4%(強可買)", 0.07, 0.094), ("≥9.4%(鎖漲停)", 0.094, 1.0)]


def main():
    u = json.loads((DATA_DIR / "base_universe_v2.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    print(f"載入 {len(codes)} 檔 ...")
    bins_on = {b[0]: [] for b in BINS}    # 隔夜
    bins_id = {b[0]: [] for b in BINS}    # 隔日日間
    for c in codes:
        o = get_daily_ohlcv(c, start=START)
        if not o or len(o) < 60: continue
        ds = sorted(o)
        op = np.array([o[d]["open"] for d in ds], float)
        cl = np.array([o[d]["close"] for d in ds], float)
        for t in range(1, len(ds) - 1):
            if cl[t-1] <= 0 or cl[t] <= 0 or op[t+1] <= 0: continue
            today = cl[t] / cl[t-1] - 1
            n_on = op[t+1] / cl[t] - 1
            n_id = cl[t+1] / op[t+1] - 1
            if not (np.isfinite(n_on) and np.isfinite(n_id)): continue
            for lab, lo, hi in BINS:
                if lo <= today < hi:
                    bins_on[lab].append(n_on); bins_id[lab].append(n_id); break

    def m(x): return np.mean(x) * 100 if x else float("nan")
    print(f"\n{'今日漲跌幅':<16}{'→隔夜均%':>10}{'隔日日間均%':>12}{'n':>10}")
    out = ["# 今日收盤強度 → 隔夜溢價梯度(259檔,2021~)\n",
           "> buy在收盤觀察『今日漲跌幅』→ 隔天開盤跳空(隔夜)+ 隔日盤中(日間)\n",
           "| 今日漲跌幅 | →隔夜均% | 隔日日間均% | n |", "|---|---|---|---|"]
    for lab, _, _ in BINS:
        on, idr = bins_on[lab], bins_id[lab]
        print(f"{lab:<16}{m(on):>+10.3f}{m(idr):>+12.3f}{len(on):>10,}")
        out.append(f"| {lab} | {m(on):+.3f} | {m(idr):+.3f} | {len(on):,} |")
    out += ["", "> 看:隔夜溢價是否隨『今日漲跌幅』單調上升;『7~9.4%(強可買)』桶若明顯高於平均(+0.27%),"
            "代表 ⑤ 買腿傾斜『今日強勢』可多吃隔夜跳空。"]
    (ROOT / "reports" / "overnight_gradient.md").write_text("\n".join(out), encoding="utf-8")
    print("\n報告 → reports/overnight_gradient.md")


if __name__ == "__main__":
    main()
