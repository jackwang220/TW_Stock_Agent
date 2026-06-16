"""族群動能因子診斷(純數據,無LLM):判斷「同族群一起漲」是否能預測個股未來報酬,
且是否在「個股自身相對強度(RS)」之外還有增量(=不是 RS 的影子)。
① 印相關鄰居分群(看抓得準不準) ② 族群動能 IC + 分位數前後組差
③ 控制 RS 後的偏相關(增量) ④ 換定義(相關鄰居K8 / 產業;動能窗 20日 / 60日)看穩不穩。
point-in-time:分群用「過去」資料、族群動能排除自己、預測測試用不重疊(每20交易日)抽樣。
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
K = 8            # 相關鄰居數
MOM = 20         # 動能窗(交易日)
FWD = 20         # 未來報酬窗


def ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 30 else float("nan")


def quantile_spread(x, y, q=5):
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 50: return float("nan"), []
    ranks = pd.qcut(pd.Series(x).rank(method="first"), q, labels=False)
    means = [y[ranks.values == i].mean() * 100 for i in range(q)]
    return means[-1] - means[0], means


def resid(y, ctrl):
    m = ~(np.isnan(y) | np.isnan(ctrl));
    A = np.vstack([np.ones(m.sum()), ctrl[m]]).T
    beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    r = np.full_like(y, np.nan); r[m] = y[m] - A @ beta
    return r


def partial_ic(x, y, ctrl):
    return ic(resid(x, ctrl), resid(y, ctrl))


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    names = {c: u[c].get("name", c) for c in codes}
    industry = {c: u[c].get("industry", "?") for c in codes}

    print("載入行情...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050", start=START)
    close = pd.DataFrame({c: pd.Series({d: OH[c][d]["close"] for d in OH[c]}) for c in codes}).sort_index()
    mkt = pd.Series({d: OH["0050"][d]["close"] for d in OH["0050"]}).sort_index().reindex(close.index)
    dates = list(close.index)
    idx = {c: i for i, c in enumerate(codes)}

    ret1 = close.pct_change()
    retM = close.pct_change(MOM)
    ret60 = close.pct_change(60)
    fwd = close.shift(-FWD) / close - 1
    rs = retM.sub(mkt.pct_change(MOM), axis=0)          # 個股相對強度(報酬 − 大盤報酬)

    # ── 相關鄰居分群(每月重算,用過去120日日報酬;point-in-time)──
    print("分群(相關鄰居,逐月)...")
    gmM = pd.DataFrame(np.nan, index=close.index, columns=codes)     # 族群動能(MOM窗)
    gm60 = pd.DataFrame(np.nan, index=close.index, columns=codes)    # 族群動能(60窗)
    last_peers = {}
    month_first = {}
    for d in dates:
        month_first.setdefault(d[:7], d)
    for ym, rd in month_first.items():
        win = ret1.loc[:rd].iloc[:-1].tail(120)
        if len(win) < 60:
            continue
        cmat = win.corr()
        W = np.zeros((len(codes), len(codes)))
        for c in codes:
            col = cmat[c].drop(c).dropna()
            if len(col) < K:
                continue
            peers = col.nlargest(K).index
            last_peers[c] = list(peers)
            for p in peers:
                W[idx[c], idx[p]] = 1.0 / K
        md = [d for d in dates if d[:7] == ym]
        gmM.loc[md] = retM.loc[md, codes].values @ W.T
        gm60.loc[md] = ret60.loc[md, codes].values @ W.T

    # ── 產業分群(靜態)──
    ind_members = {}
    for c in codes:
        ind_members.setdefault(industry[c], []).append(c)
    Wi = np.zeros((len(codes), len(codes)))
    for c in codes:
        peers = [p for p in ind_members[industry[c]] if p != c]
        for p in peers:
            Wi[idx[c], idx[p]] = 1.0 / len(peers)
    gmI = pd.DataFrame(retM.values @ Wi.T, index=close.index, columns=codes)

    out = ["# 族群動能因子診斷(純數據;判斷是否有預測力 / 是否只是 RS 影子)\n",
           f"> 112檔｜全史2021~｜分群=相關鄰居K={K}(逐月、過去120日)或產業｜動能窗{MOM}日｜未來報酬窗{FWD}日\n",
           f"> point-in-time;族群動能排除自己;預測測試用不重疊抽樣(每{FWD}交易日)\n"]

    # ① 分群肉眼檢查
    out.append("## ① 相關鄰居分群抽查(看像不像真族群)\n")
    for c in ["2327", "2408", "2603", "2330", "2891", "1101"]:
        if c in last_peers:
            ps = "、".join(f"{names.get(p,p)}" for p in last_peers[c][:6])
            out.append(f"- **{names.get(c,c)}({c})** 的鄰居:{ps}")
    out.append("")

    # ②③④ 不重疊抽樣,合併觀測
    test_dates = [d for i, d in enumerate(dates) if i % FWD == 0 and not fwd.loc[d].isna().all()]
    def pool(df):
        xs, ys, rss = [], [], []
        for d in test_dates:
            xs.append(df.loc[d, codes].values.astype(float))
            ys.append(fwd.loc[d, codes].values.astype(float))
            rss.append(rs.loc[d, codes].values.astype(float))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(rss)

    out.append("## ②③④ 預測力(IC=與未來報酬相關;分位差=最強組−最弱組 未來報酬%;增量=控制RS後偏相關)\n")
    out.append("| 因子 | IC | 高分組−低分組(未來%) | 控制RS後偏相關(增量) | n |")
    out.append("|---|---|---|---|---|")
    # 基準:個股自身 RS(參考線)
    xrs, yrs, _ = pool(rs)
    out.append(f"| 個股RS(參考) | {ic(xrs,yrs):+.3f} | {quantile_spread(xrs,yrs)[0]:+.1f} | — | {(~np.isnan(xrs*yrs)).sum()} |")
    for lab, df in [("族群動能(相關鄰居,20日)", gmM),
                    ("族群動能(相關鄰居,60日)", gm60),
                    ("族群動能(產業,20日)", gmI)]:
        x, y, r = pool(df)
        sp, _ = quantile_spread(x, y)
        out.append(f"| {lab} | {ic(x,y):+.3f} | {sp:+.1f} | {partial_ic(x,y,r):+.3f} | {(~np.isnan(x*y)).sum()} |")

    out += ["",
            "## 怎麼讀",
            "- **IC**:>~0.03 算有點預測力;接近0=沒料。",
            "- **高分組−低分組**:正且大=族群強的股票未來確實較會漲。",
            "- **控制RS後偏相關(關鍵)**:若 ≈0 → 族群動能只是個股RS的影子(沒增量,不值得做);若仍明顯正 → 有RS沒有的新資訊,值得接進策略。",
            "- ④穩健:三種定義(相關鄰居20/60、產業20)若方向一致=真效果;只有一種才出現=雜訊。"]
    REPORT = ROOT / "reports" / "group_factor_diag.md"
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\n報告 → {REPORT}")


if __name__ == "__main__":
    main()
