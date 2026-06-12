"""市場 regime guard:大盤(^TWII)跌破 200MA 時示警——只在觸發時報警。

依據(本專案實證,2021-2026 逐筆):
  - 大盤 > 200MA:反彈 edge gated 每筆 alpha +0.48%(扣 0050)。
  - 大盤 < 200MA:每筆 raw -0.11%、alpha +0.01% → 兩頭空,反彈訊號該關機。
因此:跌破 200MA = 暫停反彈/抄底訊號。

用法:
  python scripts/regime_guard.py          # 手動查當前 regime
  from regime_guard import check_regime    # 在 daily 流程裡擋單
回傳 exit code 1 代表「gate 關(暫停)」,可供自動化流程判斷。
"""
from __future__ import annotations
import sys
import pandas as pd
import yfinance as yf

MA = 200
RECENT_CROSS_DAYS = 5


def check_regime() -> dict:
    df = yf.download("^TWII", period="2y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    ma = close.rolling(MA).mean()
    if ma.dropna().empty:
        return {"ok": None, "reason": "資料不足以算 200MA"}
    last_close = float(close.iloc[-1])
    last_ma = float(ma.iloc[-1])
    on = last_close > last_ma
    # 近 N 日是否剛由上轉下(穿破)
    recent = (close.iloc[-RECENT_CROSS_DAYS - 1:] > ma.iloc[-RECENT_CROSS_DAYS - 1:]).tolist()
    just_crossed_down = (len(recent) >= 2 and recent[0] and not recent[-1])
    return {
        "ok": on,
        "date": close.index[-1].date().isoformat(),
        "close": last_close,
        "ma200": last_ma,
        "gap_pct": (last_close / last_ma - 1) * 100,
        "just_crossed_down": just_crossed_down,
    }


def main() -> int:
    r = check_regime()
    if r["ok"] is None:
        print(f"⚠️  regime guard:{r['reason']}")
        return 0
    if r["ok"]:
        # 正常:只報一行,不吵
        print(f"✅ regime ON | TWII {r['close']:.0f} > 200MA {r['ma200']:.0f} "
              f"(+{r['gap_pct']:.1f}%) | {r['date']} — 反彈策略可運作")
        return 0
    # 觸發:大盤跌破 200MA → 示警
    cross = "（今日剛穿破⚠️）" if r["just_crossed_down"] else ""
    print("=" * 60)
    print(f"🚨 REGIME 警訊 | {r['date']}{cross}")
    print(f"   大盤 TWII {r['close']:.0f} < 200MA {r['ma200']:.0f}（{r['gap_pct']:+.1f}%）")
    print("   → 下跌趨勢:反彈/抄底 edge 實證為『無 alpha、raw 為負』")
    print("   → 建議:暫停反彈訊號進場（接刀區），等站回 200MA 再開機")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())