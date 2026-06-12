"""近一個月量化訊號回測（純股價資料，不含新聞/LLM）。

【說明】
- [OK] 可以回測：vol_ratio、MA5>MA20、RS20、K 線型態
- [NA] 無法回測：Bear/Bull LLM 辯論（歷史新聞 RSS 拿不到）
- [NA] 無法回測：AI 隔日漲跌預測（需要當天新聞才能分析）

使用方式：
    uv run python scripts/backtest_one_month.py
    uv run python scripts/backtest_one_month.py --days 30
    uv run python scripts/backtest_one_month.py --days 60 --all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from tw_stock_agent.config import COMPANIES_JSON, DATA_DIR, TW_STOCK_INDEX
from tw_stock_agent.screener.patterns import analyze_pattern

OUTPUT = DATA_DIR / "backtest_one_month.csv"

FIELDS = [
    "date", "ticker", "name",
    "pass_level", "pattern_type",
    "vol_ratio", "rs_20d", "ma5_gt_ma20",
    "close_price",
    "return_1d", "return_5d",
    "twii_1d",        # 大盤同日報酬（Alpha 用）
    "alpha_1d",       # return_1d - twii_1d（超額報酬）
    "net_return_1d",  # return_1d - 0.4（扣手續費）
    "hit_direction",
]


# ── 股票清單 ──────────────────────────────────────────────────────────────────

def _load_watchlist(source: str = "watchlist") -> list[dict]:
    """股票清單載入。

    source:
      "watchlist"  - companies.json 核心供應鏈 (~30 支，預設)
      "tw-tech"    - tw_tech_stocks.json 全台電子/科技類股 (~300-400 支)
      "all"        - tw_stock_index.json 全上市櫃 (~900 支)
    """
    if source == "tw-tech":
        path = DATA_DIR / "tw_tech_stocks.json"
        if not path.exists():
            print(f"[WARN] 找不到 {path}，請先執行 fetch_tw_tech_stocks.py，改用 watchlist")
            source = "watchlist"
        else:
            index = json.loads(path.read_text(encoding="utf-8"))
            return [
                {"code": code, "name": info["name"], "yf_ticker": info["yf_ticker"]}
                for code, info in index.items()
                if code.isdigit()
            ]

    if source == "all":
        path = TW_STOCK_INDEX
        if not path.exists():
            print(f"[WARN] 找不到 {path}，改用 watchlist")
            source = "watchlist"
        else:
            index = json.loads(path.read_text(encoding="utf-8"))
            return [
                {"code": code, "name": info["name"], "yf_ticker": info["yf_ticker"]}
                for code, info in index.items()
                if code.isdigit()
            ]

    # 預設：companies.json
    blob = json.loads(COMPANIES_JSON.read_text(encoding="utf-8"))
    out = []
    for c in blob.get("companies", []):
        code = c.get("code", "")
        if not code or not code.isdigit():
            continue
        if c.get("country", "TW") != "TW":
            continue
        suffix = ".TW" if c.get("country", "TW") == "TW" else ".TWO"
        out.append({"code": code, "name": c.get("name_zh", code), "yf_ticker": f"{code}{suffix}"})
    return out


# ── 量化指標計算 ──────────────────────────────────────────────────────────────

def _quant_signals(df: pd.DataFrame, twii_df: pd.DataFrame, as_of: date) -> dict | None:
    """從完整歷史資料中截取到 as_of 日，計算量化指標。"""
    d = df[df.index.date <= as_of]
    t = twii_df[twii_df.index.date <= as_of]

    if len(d) < 60:
        return None
    # 若最後一筆距離 as_of 超過 5 天 → 不是交易日附近，跳過
    if (as_of - d.index[-1].date()).days > 5:
        return None

    vol = d["Volume"]
    close = d["Close"]
    ma20vol = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / ma20vol.iloc[-1]) if ma20vol.iloc[-1] > 0 else 0.0
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20c = close.rolling(20).mean().iloc[-1]
    ma5_gt_ma20 = bool(ma5 > ma20c)

    if len(d) >= 21 and len(t) >= 21:
        stock_ret = float(close.iloc[-1] / close.iloc[-21] - 1)
        idx_ret = float(t["Close"].iloc[-1] / t["Close"].iloc[-21] - 1)
        rs = (1 + stock_ret) / (1 + idx_ret) if abs(1 + idx_ret) > 1e-6 else 1.0
    else:
        rs = 1.0

    hits = sum([vol_ratio >= 1.2, ma5_gt_ma20, rs >= 0.9])
    pass_level = "PASS" if hits == 3 else ("WARN" if hits >= 2 else "REJECT")

    return {
        "pass_level": pass_level,
        "vol_ratio": vol_ratio,
        "rs_20d": rs,
        "ma5_gt_ma20": ma5_gt_ma20,
        "close_price": float(close.iloc[-1]),
        "_df": d,   # 給型態辨識用，跑完後丟棄
    }


def _fwd_return(full_df: pd.DataFrame, as_of: date, n: int) -> str:
    """計算 as_of 日後第 n 個交易日的報酬率（%）。"""
    try:
        dates = full_df.index.date
        positions = [i for i, d in enumerate(dates) if d >= as_of]
        if not positions or positions[0] + n >= len(full_df):
            return ""
        s = positions[0]
        entry = float(full_df["Close"].iloc[s])
        exit_p = float(full_df["Close"].iloc[s + n])
        return f"{(exit_p / entry - 1) * 100:.2f}"
    except Exception:
        return ""


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="回測日曆天數（預設 30）")
    parser.add_argument("--all", action="store_true",
                        help="也記錄 REJECT（預設只記 PASS/WARN）")
    parser.add_argument("--source", default="watchlist",
                        choices=["watchlist", "tw-tech", "all"],
                        help="股票清單來源（預設 watchlist=companies.json；tw-tech=全台電子類股；all=全上市櫃）")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    watchlist = _load_watchlist(args.source)
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=args.days)
    # 下載需要往前多抓 90 天（計算 MA20、RS20 需要足夠歷史）
    fetch_start = start_date - timedelta(days=90)
    # 往後多抓 10 天算 5 日報酬
    fetch_end = end_date + timedelta(days=14)

    logger.info(f"回測範圍：{start_date} ～ {end_date}（{args.days} 日曆天）")
    logger.info(f"標的數：{len(watchlist)}")

    # ── 一次性下載所有資料（避免 N×M API calls）──────────────────────────
    logger.info("下載股票資料（一次性）...")
    frames: dict[str, pd.DataFrame] = {}
    for t in watchlist:
        logger.info(f"  {t['yf_ticker']}")
        try:
            df = yf.download(
                t["yf_ticker"],
                start=fetch_start.isoformat(),
                end=fetch_end.isoformat(),
                progress=False, auto_adjust=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames[t["yf_ticker"]] = df
        except Exception as e:
            logger.warning(f"  {t['yf_ticker']} 下載失敗: {e}")

    logger.info("下載加權指數...")
    try:
        twii = yf.download(
            "^TWII",
            start=fetch_start.isoformat(),
            end=fetch_end.isoformat(),
            progress=False, auto_adjust=True,
        )
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = twii.columns.get_level_values(0)
    except Exception:
        twii = pd.DataFrame()
        logger.warning("加權指數下載失敗，RS 計算會固定為 1.0")

    # ── 逐日掃描 ─────────────────────────────────────────────────────────
    logger.info("開始回測...")
    rows: list[dict] = []
    current = start_date

    while current <= end_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        # 大盤當日報酬（用於 Alpha 計算）
        twii_r1 = _fwd_return(twii, current, 1) if not twii.empty else ""

        for t in watchlist:
            df = frames.get(t["yf_ticker"])
            if df is None or df.empty:
                continue

            q = _quant_signals(df, twii, current)
            if q is None:
                continue
            if not args.all and q["pass_level"] == "REJECT":
                continue

            df_slice = q.pop("_df")
            candidate = {"code": t["code"], "name": t["name"], "yf_ticker": t["yf_ticker"]}
            pat = analyze_pattern(candidate, df_slice)
            pattern_type = pat.pattern_type if pat else "none"

            r1 = _fwd_return(df, current, 1)
            r5 = _fwd_return(df, current, 5)

            # Alpha 和淨報酬
            alpha = ""
            net = ""
            if r1 != "" and twii_r1 != "":
                alpha = f"{float(r1) - float(twii_r1):.2f}"
            if r1 != "":
                net = f"{float(r1) - 0.4:.2f}"  # 扣手續費 0.4%

            hit = ""
            if r1 != "":
                hit = "1" if float(r1) > 0.2 else ("-1" if float(r1) < -0.2 else "0")

            rows.append({
                "date": current.isoformat(),
                "ticker": t["code"],
                "name": t["name"],
                "pass_level": q["pass_level"],
                "pattern_type": pattern_type,
                "vol_ratio": f"{q['vol_ratio']:.2f}",
                "rs_20d": f"{q['rs_20d']:.3f}",
                "ma5_gt_ma20": q["ma5_gt_ma20"],
                "close_price": f"{q['close_price']:.1f}",
                "return_1d": r1,
                "return_5d": r5,
                "twii_1d": twii_r1,
                "alpha_1d": alpha,
                "net_return_1d": net,
                "hit_direction": hit,
            })

        current += timedelta(days=1)

    # ── 存檔 ──────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.success(f"已存 {len(rows)} 筆 → {OUTPUT}")

    if not rows:
        print("\n沒有訊號。嘗試加 --all 看所有 REJECT 股。")
        return

    # ── 摘要報告 ──────────────────────────────────────────────────────────
    FRICTION = 0.4  # 手續費 %

    def _grp_stats(grp: list[dict], label: str) -> None:
        rets, alphas, nets = [], [], []
        for r in grp:
            try:
                rets.append(float(r["return_1d"]))
            except (ValueError, TypeError):
                pass
            try:
                alphas.append(float(r["alpha_1d"]))
            except (ValueError, TypeError):
                pass
            try:
                nets.append(float(r["net_return_1d"]))
            except (ValueError, TypeError):
                pass
        if not rets:
            print(f"  {label:35s}  n={len(grp):3d}  （1日報酬尚未入帳）")
            return
        a = np.array(rets)
        wins = (a > 0.2).sum()
        alpha_str = f"Alpha={np.mean(alphas):+.2f}%  " if alphas else ""
        net_str   = f"淨={np.mean(nets):+.2f}%  " if nets else ""
        print(f"  {label:35s}  n={len(grp):3d}  "
              f"勝率={wins/len(a):.0%}  "
              f"均={a.mean():+.2f}%  "
              f"{alpha_str}"
              f"{net_str}"
              f"中位={np.median(a):+.2f}%")

    # 大盤同期報酬
    twii_rets = []
    for r in rows:
        try:
            twii_rets.append(float(r["twii_1d"]))
        except (ValueError, TypeError):
            pass
    twii_avg = f"{np.mean(twii_rets):+.2f}%" if twii_rets else "N/A"

    print("\n" + "=" * 85)
    print(f"  近 {args.days} 日量化回測結果 — {start_date} ～ {end_date}")
    print(f"  回測標的：{len(watchlist)} 支 | 大盤同期均日報酬：{twii_avg} | 手續費：{FRICTION}%")
    print(f"  [注意] 不含 LLM 分析（需要當天新聞，歷史無法重播）")
    print("=" * 85)

    print(f"\n【一、按量化評級（PASS/WARN/REJECT）的隔日勝率與 Alpha】")
    for lvl in ["PASS", "WARN", "REJECT"]:
        grp = [r for r in rows if r["pass_level"] == lvl]
        if grp:
            _grp_stats(grp, lvl)

    print(f"\n【二、按 K 線型態的隔日勝率與 Alpha】")
    for pt in sorted(set(r["pattern_type"] for r in rows)):
        grp = [r for r in rows if r["pattern_type"] == pt]
        if grp:
            _grp_stats(grp, pt)

    print(f"\n【三、組合訊號 Alpha 對比（扣手續費後）】")
    _grp_stats([r for r in rows if r["pass_level"] == "PASS" and r["pattern_type"] != "none"],
               "A: PASS + 有型態（實驗組）")
    _grp_stats([r for r in rows if r["pass_level"] == "PASS" and r["pattern_type"] == "none"],
               "B: PASS + 無型態（對照組1）")
    _grp_stats([r for r in rows if r["pass_level"] == "WARN" and r["pattern_type"] != "none"],
               "C: WARN + 有型態（對照組2）")
    print(f"  {'大盤（買進持有）':35s}  均={twii_avg}")

    print(f"\n【四、各股命中次數（PASS 訊號）— 含 Alpha】")
    from collections import Counter
    pass_rows = [r for r in rows if r["pass_level"] == "PASS"]
    cnt = Counter(f"{r['ticker']} {r['name']}" for r in pass_rows)
    for stock, n in cnt.most_common(15):
        code = stock.split()[0]
        sub = [r for r in pass_rows if r["ticker"] == code]
        rets   = [float(r["return_1d"]) for r in sub if r.get("return_1d")]
        alphas = [float(r["alpha_1d"])  for r in sub if r.get("alpha_1d")]
        avg   = f"{np.mean(rets):+.2f}%"   if rets   else "—"
        alpha = f"{np.mean(alphas):+.2f}%" if alphas else "—"
        print(f"  {stock:20s}  {n:2d}次  均={avg}  Alpha={alpha}")

    print()


if __name__ == "__main__":
    main()
