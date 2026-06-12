"""回測 kNN 類比引擎『本身』的預測力(純資料,防洩漏)。

問題:當引擎說「類比偏多/高勝率」時,實際後續是不是真的比較會漲?
做法:對歷史上數千個(股票,日期)跑引擎預測 → 對比該股實際後續 5 日報酬。
指標:
  - 預測 vs 實際的等級相關(Spearman)
  - 方向命中率(引擎看多時實際漲的比例)
  - 十分位:照預測排序分10組,看實際報酬是否單調(top組 > bottom組 = 有edge)
  - 多空價差(top decile − bottom decile)
  - 分 regime 看

防洩漏:庫按「結果完成日」排序,查詢日 D 的合法庫=end<D 的前綴;排除同股鄰近窗口。

用法:python scripts/analog_backtest.py [--sample 4000]
"""
from __future__ import annotations

import argparse
import bisect
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import analog_engine as ae
from tw_stock_agent.config import REPORTS_DIR

OUT = REPORTS_DIR / "analog_backtest.md"
FWD = 5      # 預測/評估的前瞻天數(累積報酬)


def regime_tag(reg) -> str:
    if reg is None:
        return "n/a"
    tr, vo, dd = reg
    trend = "多頭" if tr > 0.02 else ("空頭" if tr < -0.02 else "盤整")
    vol = "高波" if vo > 0.18 else "低波"
    return f"{trend}/{vol}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--step", type=int, default=8, help="每股每隔幾日取一個查詢點")
    args = ap.parse_args()

    print("載入庫...", flush=True)
    panel = ae.load_panel()
    regime = ae.build_regime(panel)
    Vz, F, meta, scaler = ae.build_library(panel, regime)
    mu, sd = scaler
    print(f"庫:{len(panel)} 檔、{len(meta)} 窗口", flush=True)

    # 按結果完成日(meta[2])排序 → 查詢日的合法庫是前綴
    order = np.argsort([m[2] for m in meta])
    Vz = Vz[order]; F = F[order]
    meta = [meta[i] for i in order]
    end_sorted = [m[2] for m in meta]
    codes_arr = np.array([m[0] for m in meta])
    dord_arr = np.array([date.fromisoformat(m[1]).toordinal() for m in meta])
    libcum = np.prod(1 + F[:, :FWD], axis=1) - 1     # 每個庫窗口的實際前瞻5日累積
    libwin1 = (F[:, 0] > 0).astype(float)            # D+1 是否上漲

    W = ae.WIN
    w_arr = np.ones(Vz.shape[1]); w_arr[W:] = ae.CTX_W
    K = ae.K

    # 取樣查詢點
    pts = []
    for code, ser in panel.items():
        n = len(ser)
        for i in range(max(W, 60), n - FWD - 1, args.step):
            pts.append((code, i))
    random.seed(42)
    random.shuffle(pts)
    pts = pts[:args.sample]
    print(f"查詢點:{len(pts)}(每股每{args.step}日取樣)", flush=True)

    rows = []
    done = 0
    for code, i in pts:
        ser = panel[code]
        e = ae.encode(ser, i, regime)
        if e is None:
            continue
        qdate = ser[i][0]
        q = (e[0] - mu) / sd
        cutoff = bisect.bisect_left(end_sorted, qdate)   # end < qdate(防洩漏)
        if cutoff < 500:
            continue
        sV = Vz[:cutoff]
        dist = np.sqrt(((sV - q) ** 2 * w_arr).sum(1))
        # 排除同股鄰近窗口
        qord = date.fromisoformat(qdate).toordinal()
        mask = (codes_arr[:cutoff] == code) & (np.abs(dord_arr[:cutoff] - qord) < 30)
        dist[mask] = np.inf
        nn = np.argpartition(dist, K)[:K]
        pred = float(libcum[:cutoff][nn].mean())          # 引擎預測:類比平均前瞻5日
        pwin = float(libwin1[:cutoff][nn].mean())         # 類比 D+1 勝率
        closes = [c for _, c in ser]
        actual = closes[i + FWD] / closes[i] - 1          # 該股實際前瞻5日
        rows.append((qdate, code, regime_tag(regime[qdate]), pred, pwin, actual))
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(pts)} ...", flush=True)

    if len(rows) < 100:
        print("樣本太少,無法統計"); return

    pred = np.array([r[3] for r in rows])
    pwin = np.array([r[4] for r in rows])
    actual = np.array([r[5] for r in rows])
    n = len(rows)

    # Spearman(等級相關)
    from scipy.stats import spearmanr  # type: ignore
    try:
        rho, pval = spearmanr(pred, actual)
    except Exception:
        # 無 scipy 時手算等級相關
        def rank(a):
            o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a)); return r
        rp, ra = rank(pred), rank(actual)
        rho = float(np.corrcoef(rp, ra)[0, 1]); pval = float("nan")

    # 方向命中(引擎看多 vs 實際)
    up_mask = pred > 0
    hit_up = (actual[up_mask] > 0).mean() if up_mask.sum() else float("nan")
    down_mask = pred < 0
    hit_down = (actual[down_mask] < 0).mean() if down_mask.sum() else float("nan")

    # 十分位
    deciles = np.argsort(pred)
    bucket = np.array_split(deciles, 10)
    dec_actual = [actual[b].mean() for b in bucket]
    dec_pred = [pred[b].mean() for b in bucket]

    L = ["# 回測:kNN 類比引擎的預測力\n",
         f"> 樣本 {n} 個(股票,日期)查詢點｜前瞻 {FWD} 日累積報酬｜防洩漏(庫=end<查詢日)\n",
         "## 總體\n",
         f"- **Spearman 等級相關(預測 vs 實際):{rho:+.3f}**(p={pval:.1e})　← >0 且顯著 = 有預測力",
         f"- 引擎看多(pred>0)時實際上漲比例:**{hit_up*100:.1f}%**（n={int(up_mask.sum())}）",
         f"- 引擎看空(pred<0)時實際下跌比例:**{hit_down*100:.1f}%**（n={int(down_mask.sum())}）",
         f"- 全樣本實際平均5日報酬:{actual.mean()*100:+.2f}%（基準線）",
         "",
         "## 十分位（照『預測報酬』排序分10組,看實際報酬是否單調遞增）\n",
         "| 組(低→高預測) | 平均預測% | **實際平均%** | 樣本 |",
         "|------|------|------|------|"]
    for d in range(10):
        L.append(f"| D{d+1} | {dec_pred[d]*100:+.2f} | **{dec_actual[d]*100:+.2f}** | {len(bucket[d])} |")
    ls = dec_actual[-1] - dec_actual[0]
    L.append("")
    L.append(f"**多空價差(最高預測組 − 最低預測組)的實際報酬差:{ls*100:+.2f}%** ← 越大越有 edge")
    L.append("")

    # 分 regime
    L.append("## 分 regime（引擎在哪種盤有用）\n")
    L.append("| regime | 樣本 | Spearman | 看多命中% | 多空價差% |")
    L.append("|------|------|------|------|------|")
    tags = sorted(set(r[2] for r in rows))
    for tg in tags:
        idx = [j for j, r in enumerate(rows) if r[2] == tg]
        if len(idx) < 80:
            continue
        p = pred[idx]; a = actual[idx]
        try:
            rr = spearmanr(p, a)[0]
        except Exception:
            rr = float("nan")
        um = p > 0
        hu = (a[um] > 0).mean() if um.sum() else float("nan")
        dec2 = np.array_split(np.argsort(p), 5)
        lsr = a[dec2[-1]].mean() - a[dec2[0]].mean()
        L.append(f"| {tg} | {len(idx)} | {rr:+.3f} | {hu*100:.0f}% | {lsr*100:+.2f} |")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n完成 → {OUT}")
    print(f"Spearman={rho:+.3f}  看多命中={hit_up*100:.1f}%  多空價差={ls*100:+.2f}%")


if __name__ == "__main__":
    main()