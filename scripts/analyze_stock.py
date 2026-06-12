"""單支股票即時 LLM 辯論分析。

用法：
    uv run python scripts/analyze_stock.py 2327
    uv run python scripts/analyze_stock.py 3026 --market TWO
    uv run python scripts/analyze_stock.py 2330 --report   # 另存 Markdown 報告
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tw_stock_agent.debate.bear import run_debate, DebateResult
from tw_stock_agent.market_status import get_stock_market_status


def fetch_indicators(code: str, market: str = "TW") -> dict:
    ticker = f"{code}.{market}"
    df = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"資料不足：{ticker}")

    # flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 過濾收盤價為 NaN 的列（盤中尚未收盤 / 非交易日）
    df = df.dropna(subset=["Close"])

    if len(df) < 21:
        raise ValueError(f"資料不足：{ticker}")

    close = df["Close"]
    volume = df["Volume"]

    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    vol20 = volume.rolling(20).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    last = df.iloc[-1]
    prev = df.iloc[-2]

    vol_ratio  = float(last["Volume"] / vol20.iloc[-1]) if vol20.iloc[-1] > 0 else 1.0
    ma5_gt_ma20 = float(ma5.iloc[-1]) > float(ma20.iloc[-1])
    rsi_val    = float(rsi.iloc[-1])
    close_price = float(last["Close"])
    prev_close  = float(prev["Close"])
    gap_pct     = (close_price - prev_close) / prev_close * 100

    # RS vs index（用 0050.TW 近 20 日）
    idx_df = yf.download("0050.TW", period="60d", interval="1d", progress=False, auto_adjust=True)
    if isinstance(idx_df.columns, pd.MultiIndex):
        idx_df.columns = idx_df.columns.get_level_values(0)
    idx_df = idx_df.dropna(subset=["Close"])
    if len(idx_df) >= 20:
        stock_ret = close.iloc[-1] / close.iloc[-20] - 1
        idx_ret   = float(idx_df["Close"].iloc[-1]) / float(idx_df["Close"].iloc[-20]) - 1
        rs_20d    = (1 + stock_ret) / (1 + idx_ret) if (1 + idx_ret) != 0 else 1.0
    else:
        rs_20d = 1.0

    # 簡易型態偵測
    cross_now  = ma5_gt_ma20
    cross_prev = float(ma5.iloc[-2]) > float(ma20.iloc[-2])
    golden_cross = cross_now and not cross_prev

    if golden_cross:
        pattern = "golden_cross"
    elif vol_ratio >= 2.0 and close_price > prev_close:
        pattern = "volume_breakout"
    elif gap_pct >= 2.0:
        pattern = "gap_up"
    elif abs(close_price - prev_close) / prev_close * 100 >= 2.0 and close_price > prev_close:
        pattern = "strong_candle"
    else:
        pattern = "none"

    # 近 5 日 K 線（給 LLM 看連續走勢）
    bars_df = df.tail(5).copy()
    recent_bars: list[dict] = []
    for i, (ts, row) in enumerate(bars_df.iterrows()):
        v20 = float(vol20.loc[ts]) if ts in vol20.index and vol20.loc[ts] > 0 else 0
        vr_i = float(row["Volume"] / v20) if v20 > 0 else 1.0
        if i == 0:
            chg_i = 0.0
        else:
            p = float(bars_df.iloc[i - 1]["Close"])
            chg_i = (float(row["Close"]) - p) / p * 100 if p > 0 else 0.0
        recent_bars.append({
            "date": ts.strftime("%Y-%m-%d"),
            "open":  round(float(row["Open"]),  1),
            "high":  round(float(row["High"]),  1),
            "low":   round(float(row["Low"]),   1),
            "close": round(float(row["Close"]), 1),
            "vol_ratio": round(vr_i, 2),
            "chg_pct":   round(chg_i, 2),
        })

    # 取名稱（含英文，供多語搜尋用）
    info = yf.Ticker(ticker).info
    name = info.get("longName") or info.get("shortName") or code
    name_en = info.get("longName", "") or ""

    return {
        "code": code,
        "name": name,
        "name_en": name_en,
        "yf_ticker": ticker,
        "pass_level": "PASS",
        "pattern_type": pattern,
        "pattern_detail": pattern,
        "close_price": close_price,
        "prev_close": prev_close,
        "volume_ratio": round(vol_ratio, 2),
        "ma5_gt_ma20": ma5_gt_ma20,
        "rs_20d": round(float(rs_20d), 3),
        "rsi_14": round(rsi_val, 1),
        "macd_hist": 0.0,
        "weeks_52_warn": False,
        "recent_bars": recent_bars,
    }


def write_report(stock: dict, result: DebateResult, ms: dict, output_path: Path) -> None:
    """把分析結果寫成 Markdown 報告。"""
    code = stock["code"]
    name = stock["name"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    bars = stock.get("recent_bars", [])

    verdict_emoji = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "REJECT": "❌ REJECT"}.get(result.verdict, result.verdict)

    lines = [
        f"# {name}（{code}）分析報告",
        f"",
        f"> 產生時間：{now}",
        f"",
        f"## 技術面快照",
        f"",
        f"| 項目 | 數值 |",
        f"|------|------|",
        f"| 收盤價 | {stock['close_price']:.1f} TWD |",
        f"| 前收盤 | {stock['prev_close']:.1f} TWD |",
        f"| 漲跌幅 | {(stock['close_price'] - stock['prev_close']) / stock['prev_close'] * 100:+.2f}% |",
        f"| 成交量比 | {stock['volume_ratio']:.2f}x |",
        f"| MA5 > MA20 | {stock['ma5_gt_ma20']} |",
        f"| RSI14 | {stock['rsi_14']:.1f} |",
        f"| RS/0050（20日） | {stock['rs_20d']:.3f} |",
        f"| 偵測型態 | {stock['pattern_type']} |",
        f"",
    ]

    # 特殊狀態
    if ms.get("summary"):
        lines += [f"## 特殊市場狀態", f"", f"**{ms['summary']}**", f""]
        if ms.get("is_disposal"):
            days = ms.get("disposal_days_left")
            days_str = ""
            if days is not None:
                if days == 0:
                    days_str = "（**今日出關**）"
                elif days > 0:
                    days_str = f"（**{days} 天後出關**）"
                else:
                    days_str = f"（{abs(days)} 天前已出關）"
            lines += [
                f"- 處置期間：至 {ms['disposal_until']}{days_str}",
                f"- 來源：{ms.get('disposal_source', '')}",
                f"- 交易限制：約每 20 分鐘撮合一次、需全額預繳、無法當沖",
                f"",
            ]

    # 近 5 日 K 線
    if bars:
        lines += [f"## 近 5 日 K 線", f""]
        lines += [f"| 日期 | 開 | 高 | 低 | 收 | 量比 | 漲跌 |"]
        lines += [f"|------|---:|---:|---:|---:|-----:|-----:|"]
        for i, bar in enumerate(bars):
            chg = f"{bar['chg_pct']:+.2f}%" if i > 0 else "（基準）"
            lines.append(
                f"| {bar['date']} | {bar['open']:.1f} | {bar['high']:.1f} |"
                f" {bar['low']:.1f} | {bar['close']:.1f} | {bar['vol_ratio']:.2f}x | {chg} |"
            )
        lines.append("")

    dir_map = {"up": "↑ 看漲", "down": "↓ 看跌", "neutral": "→ 中性"}

    # LLM 辯論結果
    lines += [
        f"## LLM 辯論結果",
        f"",
        f"| 項目 | 結果 |",
        f"|------|------|",
        f"| 裁決 | **{verdict_emoji}** |",
        f"| 多方分數 | {result.bull_score} / 10 |",
        f"| 空方分數 | {result.bear_score} / 10 |",
        f"",
        f"### 走勢預測",
        f"",
        f"| 期間 | 方向 | 保守 | 樂觀 | 中位 | 信心 |",
        f"|------|------|-----:|-----:|-----:|-----:|",
        f"| **D+1（明日）** | {dir_map.get(result.predicted_direction, result.predicted_direction)} | {result.predicted_low_pct:+.1f}% | {result.predicted_high_pct:+.1f}% | {result.predicted_center_pct:+.1f}% | {result.prediction_confidence:.0%} |",
        f"| **D+3（3日）** | {dir_map.get(result.d3_direction, result.d3_direction)} | {result.d3_low_pct:+.1f}% | {result.d3_high_pct:+.1f}% | {result.d3_center_pct:+.1f}% | {result.d3_confidence:.0%} |",
        f"| **D+5（5日）** | {dir_map.get(result.d5_direction, result.d5_direction)} | {result.d5_low_pct:+.1f}% | {result.d5_high_pct:+.1f}% | {result.d5_center_pct:+.1f}% | {result.d5_confidence:.0%} |",
        f"",
    ]
    if result.prediction_scenario:
        lines += [f"**情境描述：** {result.prediction_scenario}", f""]
    lines += [
        f"### 多方理由",
        f"",
        f"{result.bull_reason}",
        f"",
        f"### 空方理由",
        f"",
        f"{result.bear_reason}",
        f"",
        f"### 關鍵因素",
        f"",
        f"{result.prediction_key_factor}",
        f"",
    ]

    # 即將發生事件
    if result.upcoming_events:
        lines += [f"## ⏰ 即將發生事件（Catalyst Timelines）", f""]
        for ev in result.upcoming_events:
            status = "✅ 確定" if ev["is_confirmed"] else "⚠️ 傳聞"
            days = ev.get("days_until")
            if days is not None:
                if days == 0:
                    days_str = "（**今日**）"
                elif days > 0:
                    days_str = f"（**{days} 天後**）"
                else:
                    days_str = f"（{abs(days)} 天前）"
            else:
                days_str = ""
            lines.append(f"- [{status}] **{ev['event_name']}** ｜ {ev['date_mention']}{days_str}")
        lines.append("")

    # 相關新聞
    if result.positive_news:
        lines += [f"## 利多新聞", f""]
        for n in result.positive_news[:5]:
            link = n.get("link", "")
            title = n.get("title", "")
            lines.append(f"- [{title}]({link})" if link else f"- {title}")
        lines.append("")

    if result.negative_news:
        lines += [f"## 利空新聞", f""]
        for n in result.negative_news[:5]:
            link = n.get("link", "")
            title = n.get("title", "")
            lines.append(f"- [{title}]({link})" if link else f"- {title}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n報告已儲存：{output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("code", help="股票代號，例如 2327")
    parser.add_argument("--market", default="TW", choices=["TW", "TWO"],
                        help="TW=上市 TWO=上櫃（預設 TW）")
    parser.add_argument("--report", action="store_true",
                        help="另存 Markdown 報告到 reports/ 資料夾")
    args = parser.parse_args()

    print(f"正在下載 {args.code}.{args.market} 技術指標...")
    stock = fetch_indicators(args.code, args.market)

    # 查詢特殊市場狀態
    print("正在查詢處置股 / 漲跌停狀態...")
    ms = get_stock_market_status(
        code=args.code,
        name_zh=stock["name"],
        close=stock["close_price"],
        prev_close=stock["prev_close"],
        market=args.market,
    )
    stock["market_status"] = ms

    print(f"\n【技術面快照】{stock['name']} ({args.code})")
    print(f"  收盤價：{stock['close_price']:.1f}  前收：{stock['prev_close']:.1f}")
    chg = (stock['close_price'] - stock['prev_close']) / stock['prev_close'] * 100
    print(f"  漲跌幅：{chg:+.2f}%")
    print(f"  成交量比：{stock['volume_ratio']:.2f}x")
    print(f"  MA5>MA20：{stock['ma5_gt_ma20']}")
    print(f"  RSI14：{stock['rsi_14']:.1f}")
    print(f"  RS/0050：{stock['rs_20d']:.3f}")
    print(f"  偵測型態：{stock['pattern_type']}")

    # 近 5 日 K 線
    bars = stock.get("recent_bars", [])
    if bars:
        print(f"\n  近5日K線：")
        print(f"  {'日期':<12} {'開':>7} {'高':>7} {'低':>7} {'收':>7} {'量比':>6} {'漲跌':>8}")
        for i, bar in enumerate(bars):
            chg_str = f"{bar['chg_pct']:+.2f}%" if i > 0 else " (基準)"
            print(f"  {bar['date']:<12} {bar['open']:>7.1f} {bar['high']:>7.1f}"
                  f" {bar['low']:>7.1f} {bar['close']:>7.1f}"
                  f" {bar['vol_ratio']:>5.2f}x {chg_str:>8}")

    # 顯示特殊狀態
    if ms["summary"]:
        print(f"\n【特殊狀態】{ms['summary']}")
    else:
        print("  市場狀態：無異常（非處置股、非注意股）")

    print(f"\n開始 LLM 辯論分析（含法說/電話會議新聞搜尋）...\n")

    result = run_debate(stock, [])

    print("\n" + "="*60)
    print(f"  {stock['name']} ({args.code}) LLM 分析結果")
    print("="*60)
    print(f"  多方分數：{result.bull_score}")
    print(f"  空方分數：{result.bear_score}")
    print(f"  裁決：{result.verdict}")
    print(f"\n  【走勢預測】")
    print(f"  {'期間':<6} {'方向':<6} {'保守':>7} {'樂觀':>7} {'中位':>7} {'信心':>6}")
    print(f"  {'------':<6} {'------':<6} {'-------':>7} {'-------':>7} {'-------':>7} {'------':>6}")
    for label, d, lo, hi, ctr, conf in [
        ("D+1", result.predicted_direction, result.predicted_low_pct, result.predicted_high_pct, result.predicted_center_pct, result.prediction_confidence),
        ("D+3", result.d3_direction,        result.d3_low_pct,        result.d3_high_pct,        result.d3_center_pct,        result.d3_confidence),
        ("D+5", result.d5_direction,        result.d5_low_pct,        result.d5_high_pct,        result.d5_center_pct,        result.d5_confidence),
    ]:
        print(f"  {label:<6} {d:<6} {lo:>+6.1f}% {hi:>+6.1f}% {ctr:>+6.1f}% {conf:>5.0%}")
    if result.prediction_scenario:
        print(f"\n  情境：{result.prediction_scenario}")
    print(f"\n  多方理由：{result.bull_reason[:300]}")
    print(f"\n  空方理由：{result.bear_reason[:300]}")
    print(f"\n  關鍵因素：{result.prediction_key_factor}")

    # 即將發生事件
    if result.upcoming_events:
        print(f"\n  【即將發生事件】")
        for ev in result.upcoming_events:
            status = "[確定]" if ev["is_confirmed"] else "[傳聞]"
            days = ev.get("days_until")
            days_str = f"（{days:+d} 天）" if days is not None else ""
            print(f"    {status} {ev['event_name']} — {ev['date_mention']}{days_str}")

    if args.report:
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        report_dir = ROOT / "reports"
        report_dir.mkdir(exist_ok=True)
        out = report_dir / f"{args.code}_{date_str}.md"
        write_report(stock, result, ms, out)


if __name__ == "__main__":
    main()
