"""RETEST robustness: overnight premium 的穩健性護欄。
  - 防skew: winsorize 1%/99% 後隔夜溢價還在嗎? 排除鎖漲停隔日後還在嗎?
  - 逐年(含OOS 2024-2026)隔夜 vs 日間是否一致為正/負
  - 0050(指數本身)的隔夜 vs 日間,直接對應 ⑤ 執行腿
  - 散戶可執行性: 「市價隔日開盤賣」是真的能拿到 open 嗎(滑價已知,僅做幅度感)
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2021-01-01"
CACHE = DATA_DIR / "finmind_cache"


def main():
    codes = sorted({re.match(r"TaiwanStockPrice_(.+)\.json", p.name).group(1)
                    for p in CACHE.glob("TaiwanStockPrice_*.json")})
    rows_on, rows_id, rows_yr, rows_lu = [], [], [], []
    for c in codes:
        o = get_daily_ohlcv(c, start=START)
        if not o or len(o) < 60:
            continue
        ds = sorted(d for d in o if d >= START)
        op = np.array([o[d]["open"] for d in ds], float)
        cl = np.array([o[d]["close"] for d in ds], float)
        for t in range(1, len(ds)):
            if cl[t-1] <= 0 or op[t] <= 0 or cl[t] <= 0:
                continue
            on = op[t] / cl[t-1] - 1
            idr = cl[t] / op[t] - 1
            if not (np.isfinite(on) and np.isfinite(idr)):
                continue
            prev = cl[t-1] / cl[t-2] - 1 if t >= 2 and cl[t-2] > 0 else 0.0
            rows_on.append(on); rows_id.append(idr); rows_yr.append(ds[t][:4])
            rows_lu.append(prev >= 0.095)
    on = np.array(rows_on); idr = np.array(rows_id)
    yr = np.array(rows_yr); lu = np.array(rows_lu)

    def stat(name, a):
        print(f"  {name:<28} 均 {np.mean(a)*100:+.4f}%  中位 {np.median(a)*100:+.4f}%  n={len(a):,}")

    print("===== raw =====")
    stat("隔夜(all)", on); stat("日間(all)", idr)

    lo, hi = np.quantile(on, [0.01, 0.99])
    onw = np.clip(on, lo, hi)
    lo2, hi2 = np.quantile(idr, [0.01, 0.99])
    idw = np.clip(idr, lo2, hi2)
    print("===== winsorize 1/99% (防大跳空主導) =====")
    stat("隔夜(wins)", onw); stat("日間(wins)", idw)

    print("===== 排除鎖漲停隔日(前日c2c>=9.5%) =====")
    mask = ~lu
    stat("隔夜(no LU-next)", on[mask]); stat("日間(no LU-next)", idr[mask])

    print("===== 逐年(隔夜 / 日間 均%) =====")
    for y in sorted(set(rows_yr)):
        ym = yr == y
        print(f"  {y}: 隔夜 {np.mean(on[ym])*100:+.4f}%  日間 {np.mean(idr[ym])*100:+.4f}%  n={ym.sum():,}")

    # 0050 本身
    print("===== 0050 (⑤指數腿) =====")
    o = get_daily_ohlcv("0050", start=START)
    ds = sorted(d for d in o if d >= START)
    op = np.array([o[d]["open"] for d in ds], float)
    cl = np.array([o[d]["close"] for d in ds], float)
    on5 = op[1:] / cl[:-1] - 1
    id5 = cl[1:] / op[1:] - 1
    c2c = cl[1:] / cl[:-1] - 1
    stat("0050 隔夜", on5); stat("0050 日間", id5); stat("0050 c2c(全)", c2c)
    print(f"  0050 隔夜累積(複利) {(np.prod(1+on5)-1)*100:+.1f}%  日間累積 {(np.prod(1+id5)-1)*100:+.1f}%  c2c累積 {(np.prod(1+c2c)-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
