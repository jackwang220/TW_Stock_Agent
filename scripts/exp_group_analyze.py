"""剖析族群關係圖:為什麼相關鄰居的族群動能接進策略效果不佳?
假設:原始報酬相關被「大盤beta」汙染 → 大型電子股的鄰居 = 跟大盤一起動的權值blob,
非真主題(被動元件/記憶體) → 族群動能≈大盤動能 → 跟RS/大盤重複 → 增量小。
驗證:① 鄰居 in-degree 集中度(是不是少數權值股當大家的鄰居)
     ② 族群動能 vs 大盤動能 相關(汙染程度)
     ③ 原始相關 vs 去大盤(excess)相關 的鄰居名單對比(去beta後是否更像真主題)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2021-01-01"; K = 8


def peers_from_corr(cmat, codes):
    out = {}
    for c in codes:
        col = cmat[c].drop(c).dropna()
        if len(col) >= K:
            out[c] = list(col.nlargest(K).index)
    return out


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    print("載入行情...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    close = pd.DataFrame({c: pd.Series({d: OH[c][d]["close"] for d in OH[c]}) for c in codes}).sort_index()
    mkt = pd.Series({d: OH["0050"][d]["close"] for d in OH["0050"]}).sort_index().reindex(close.index)
    ret = close.pct_change()
    mret = mkt.pct_change()
    excess = ret.sub(mret, axis=0)        # 去大盤(簡單 beta=1 殘差)

    win = ret.tail(120); winx = excess.tail(120)     # 最近120日快照
    praw = peers_from_corr(win.corr(), codes)
    pexc = peers_from_corr(winx.corr(), codes)

    print("\n===== ① 鄰居 in-degree(誰最常被當別人的鄰居=市場代理)=====")
    deg_raw = Counter(p for ps in praw.values() for p in ps)
    deg_exc = Counter(p for ps in pexc.values() for p in ps)
    print("【原始相關】最常當鄰居 Top10(被多少檔選為鄰居 / 共112):")
    for c, n in deg_raw.most_common(10):
        print(f"  {names.get(c,c)}({c}): {n}")
    print("【去大盤後】最常當鄰居 Top10:")
    for c, n in deg_exc.most_common(10):
        print(f"  {names.get(c,c)}({c}): {n}")

    # 平均成對相關(看原始 vs 去大盤,整體相關水位)
    cr = win.corr().values; ce = winx.corr().values
    iu = np.triu_indices(len(codes), 1)
    print(f"\n平均成對相關:原始={np.nanmean(cr[iu]):.2f}  去大盤={np.nanmean(ce[iu]):.2f}"
          f"  (原始高=大家一起跟大盤動)")

    print("\n===== ② 族群動能 vs 大盤動能 相關(汙染程度)=====")
    ret60 = close.pct_change(60); mret60 = mkt.pct_change(60)
    idx = {c: i for i, c in enumerate(codes)}
    W = np.zeros((len(codes), len(codes)))
    for c, ps in praw.items():
        for p in ps: W[idx[c], idx[p]] = 1.0/K
    gm = pd.DataFrame(ret60.values @ W.T, index=close.index, columns=codes)
    # pool 不重疊抽樣
    dts = list(close.index)[::20]
    g = np.concatenate([gm.loc[d].values for d in dts])
    mm = np.concatenate([np.full(len(codes), mret60.get(d, np.nan)) for d in dts])
    m = ~(np.isnan(g) | np.isnan(mm))
    print(f"  corr(族群動能60, 大盤動能60) = {np.corrcoef(g[m], mm[m])[0,1]:+.2f}"
          f"  (越接近1 → 族群動能其實就是大盤動能,難怪跟RS重複)")

    print("\n===== ③ 原始 vs 去大盤 鄰居名單對比(去beta後是否更像真主題)=====")
    for c in ["2327", "2408", "2603", "2891", "2330", "3037"]:
        if c in praw:
            print(f"\n{names.get(c,c)}({c})")
            print(f"  原始相關鄰居 : {'、'.join(names.get(p,p) for p in praw[c])}")
            if c in pexc:
                print(f"  去大盤後鄰居 : {'、'.join(names.get(p,p) for p in pexc[c])}")


if __name__ == "__main__":
    main()
