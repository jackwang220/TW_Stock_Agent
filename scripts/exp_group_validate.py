"""驗證「族群超額動能 = 族群平均報酬 − 大盤報酬」能不能正確抓出最近的熱門族群。
正確性指標:這幾週記憶體/被動元件狂漲 → 超額動能版應把它們排前面。
對比:原始動能(未扣大盤)會被大盤稀釋、排名失真。
為了讓被動元件有成員,臨時補進不在 112 universe 的被動元件股(僅驗證方法)。
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2024-06-01"

GROUPS = {
    "記憶體":     ["2408", "2344", "2337", "8299"],                  # 南亞科/華邦電/旺宏/群聯
    "被動元件":   ["2327", "2492", "3026", "2478", "6173"],          # 國巨/華新科/禾伸堂/大毅/信昌電(後4補入)
    "航運":       ["2603", "2609", "2615"],                          # 長榮/陽明/萬海
    "金融":       ["2891", "2882", "2886", "2884", "2892"],          # 中信/國泰/兆豐/玉山/一銀
    "ABF載板":    ["3037", "8046", "3189"],                          # 欣興/南電/景碩
    "AI伺服器":   ["2376", "6669", "3231", "2356"],                  # 技嘉/緯穎/緯創/英業達
    "散熱":       ["3017", "3324"],                                  # 奇鋐/雙鴻
    "面板":       ["2409", "3481"],                                  # 友達/群創
    "IC設計":     ["2454", "3034", "2379"],                          # 聯發科/聯詠/瑞昱
}
NAMES = {"2408":"南亞科","2344":"華邦電","2337":"旺宏","8299":"群聯","2327":"國巨","2492":"華新科",
         "3026":"禾伸堂","2478":"大毅","6173":"信昌電","2603":"長榮","2609":"陽明","2615":"萬海"}


def main():
    allc = sorted({c for g in GROUPS.values() for c in g})
    print(f"載入 {len(allc)} 檔 + 0050 ...")
    close = {}
    for c in allc + ["0050"]:
        o = get_daily_ohlcv(c, start=START)
        if o: close[c] = pd.Series({d: o[d]["close"] for d in o})
    df = pd.DataFrame(close).sort_index()
    if "0050" not in df:
        print("缺 0050"); return
    d = df.index[-1]
    print(f"最新交易日 = {d}\n")

    def ret(code, n):
        s = df[code].dropna()
        return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100 if len(s) > n else float("nan")

    rows = []
    for win, label in [(20, "20日(這幾週)"), (60, "60日(這幾月)")]:
        mkt = ret("0050", win)
        recs = []
        for g, members in GROUPS.items():
            present = [c for c in members if c in df.columns]
            rs = [ret(c, win) for c in present]
            rs = [x for x in rs if x == x]
            if not rs: continue
            grp = sum(rs) / len(rs)
            recs.append((g, grp, grp - mkt, len(rs)))
        recs.sort(key=lambda x: -x[2])
        print(f"===== {label}｜大盤(0050) {mkt:+.1f}% =====")
        print(f"  {'族群':<10}{'原始動能%':>10}{'超額動能%':>10}{'成員':>5}")
        for g, raw, exc, n in recs:
            star = "  ⭐" if g in ("記憶體", "被動元件") else ""
            print(f"  {g:<10}{raw:>+10.1f}{exc:>+10.1f}{n:>5}{star}")
        print()
        rows.append((label, recs))

    # 排名驗證
    print("===== 驗證:記憶體/被動元件 在『超額動能』的排名 =====")
    for label, recs in rows:
        order = [g for g, *_ in recs]
        for tgt in ("記憶體", "被動元件"):
            if tgt in order:
                print(f"  {label}: {tgt} 超額動能排第 {order.index(tgt)+1} / {len(order)}")


if __name__ == "__main__":
    main()
