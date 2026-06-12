"""類比引擎實驗矩陣:一次回答 A(設定閘門)、B(該走哪條路)、C(加特徵)。

同一個庫,跑多個變體,比較預測力:
  特徵:basic(波形+regime) vs rich(再加 量比/RSI/乖離/近5日報酬=回歸訊號)
  設定:all(每天) vs drop(只在深跌日,近5日≤-8%)
  流動性:all vs large(日均成交≥5億)

每個變體報:樣本數、Spearman(預測vs實際)、看多命中%、多空價差(五分位 top-bottom)。
防洩漏:庫按結果完成日排序,查詢日只用 end<D 的前綴。

用法:python scripts/analog_experiments.py
"""
from __future__ import annotations

import bisect
import glob
import json
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from tw_stock_agent.config import DATA_DIR, REPORTS_DIR

CACHE = DATA_DIR / "finmind_cache"
OUT = REPORTS_DIR / "analog_experiments.md"
WIN, FWD, K, CTX_W = 20, 5, 50, 3.0
NCTX = 9                     # context 特徵數
DROP_THR = -0.08            # 深跌設定:近5日 ≤ -8%
LARGE_TURN = 5e8           # 大型:日均成交 ≥ 5 億


def load_panel():
    base_f = DATA_DIR / "base_universe.json"
    only = set(json.loads(base_f.read_text(encoding="utf-8"))) if base_f.exists() else None
    turn = {}
    if base_f.exists():
        for c, v in json.loads(base_f.read_text(encoding="utf-8")).items():
            turn[c] = v.get("avg_turnover", 0)
    panel = {}
    for f in glob.glob(str(CACHE / "TaiwanStockPrice_*.json")):
        code = Path(f).stem.replace("TaiwanStockPrice_", "")
        if only is not None and code not in only:
            continue
        try:
            rows = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        ds, cl, vo = [], [], []
        for r in rows:
            c = r.get("close")
            try:
                c = float(c)
            except (TypeError, ValueError):
                continue
            if c > 0:
                ds.append(r["date"]); cl.append(c); vo.append(float(r.get("Trading_Volume") or 0))
        if len(cl) > WIN + FWD + 220:
            order = np.argsort(ds)
            panel[code] = (list(np.array(ds)[order]), list(np.array(cl)[order]), list(np.array(vo)[order]))
    return panel, turn


def build_regime(panel):
    from collections import defaultdict
    rb = defaultdict(list)
    for ds, cl, _ in panel.values():
        for i in range(1, len(cl)):
            if cl[i - 1] > 0:
                rb[ds[i]].append(cl[i] / cl[i - 1] - 1)
    dates = sorted(rb)
    med = np.array([np.median(rb[d]) for d in dates])
    idx = np.cumprod(1 + med)
    out = {}
    for k, d in enumerate(dates):
        if k < 200:
            out[d] = None
        else:
            out[d] = (float(idx[k] / idx[k - 200:k].mean() - 1),
                      float(np.std(med[k - 20:k]) * np.sqrt(252)),
                      float(idx[k] / idx[k - 60:k + 1].max() - 1))
    return out


def _rsi(closes, i, n=14):
    g = l = 0.0
    for j in range(i - n + 1, i + 1):
        ch = closes[j] - closes[j - 1]
        g += max(ch, 0); l += max(-ch, 0)
    if g + l == 0:
        return 50.0
    rs = g / l if l > 0 else 999
    return 100 - 100 / (1 + rs)


def encode(cl, vo, i, regime, d):
    """回傳 (vec[WIN+NCTX], ret5) 或 None。"""
    if i < max(WIN, 60) or regime.get(d) is None:
        return None
    shape = [cl[i - WIN + 1 + j] / cl[i - WIN + j] - 1 for j in range(WIN)]
    own_vol = float(np.std(shape) * np.sqrt(252))
    own_mom = cl[i] / cl[i - 60] - 1
    tr, vol, dd = regime[d]
    ma20v = np.mean(vo[i - 19:i + 1]) if np.mean(vo[i - 19:i + 1]) > 0 else 1
    vol_ratio = vo[i] / ma20v
    rsi = _rsi(cl, i)
    ma20 = np.mean(cl[i - 19:i + 1])
    bias = cl[i] / ma20 - 1 if ma20 > 0 else 0
    ret5 = cl[i] / cl[i - 5] - 1
    vec = np.array(shape + [own_vol, own_mom, tr, vol, dd, vol_ratio, rsi, bias, ret5], dtype=float)
    return vec, ret5


def build_library(panel, regime):
    V, fcum, fw1, meta, ret5s = [], [], [], [], []
    for code, (ds, cl, vo) in panel.items():
        for i in range(max(WIN, 60), len(cl) - FWD):
            e = encode(cl, vo, i, regime, ds[i])
            if e is None:
                continue
            cum = cl[i + FWD] / cl[i] - 1
            w1 = 1.0 if cl[i + 1] / cl[i] - 1 > 0 else 0.0
            V.append(e[0]); fcum.append(cum); fw1.append(w1)
            meta.append((code, ds[i], ds[i + FWD])); ret5s.append(e[1])
    V = np.array(V); fcum = np.array(fcum); fw1 = np.array(fw1); ret5s = np.array(ret5s)
    # ★ 修 leakage:庫必須按「結果完成日」排序,run_variant 的 bisect 才正確(否則切點錯=混進未來)
    order = np.argsort([m[2] for m in meta])
    V = V[order]; fcum = fcum[order]; fw1 = fw1[order]; ret5s = ret5s[order]
    meta = [meta[i] for i in order]
    mu, sd = V.mean(0), V.std(0); sd[sd == 0] = 1
    return (V - mu) / sd, fcum, fw1, meta, ret5s, (mu, sd)


def spearman(a, b):
    def rk(x):
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
    if len(a) < 10:
        return float("nan")
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


def run_variant(name, panel, regime, lib, turn, feature, setup, liq, sample=2500):
    Vz, fcum, fw1, meta, ret5s, (mu, sd) = lib
    end_sorted = [m[2] for m in meta]
    codes_arr = np.array([m[0] for m in meta])
    dord = np.array([date.fromisoformat(m[1]).toordinal() for m in meta])
    w = np.ones(Vz.shape[1]); w[WIN:] = CTX_W
    if feature == "basic":
        w[WIN + 5:] = 0.0      # 關掉 rich 特徵(vol_ratio/rsi/bias/ret5)

    pts = []
    for code, (ds, cl, vo) in panel.items():
        if liq == "large" and turn.get(code, 0) < LARGE_TURN:
            continue
        for i in range(max(WIN, 60), len(cl) - FWD):
            pts.append((code, i))
    random.seed(42); random.shuffle(pts)

    preds, acts = [], []
    used = 0
    for code, i in pts:
        if used >= sample:
            break
        ds, cl, vo = panel[code]
        e = encode(cl, vo, i, regime, ds[i])
        if e is None:
            continue
        if setup == "drop" and e[1] > DROP_THR:
            continue
        qdate = ds[i]
        cutoff = bisect.bisect_left(end_sorted, qdate)
        if cutoff < 500:
            continue
        q = (e[0] - mu) / sd
        dist = np.sqrt(((Vz[:cutoff] - q) ** 2 * w).sum(1))
        qord = date.fromisoformat(qdate).toordinal()
        m = (codes_arr[:cutoff] == code) & (np.abs(dord[:cutoff] - qord) < 30)
        dist[m] = np.inf
        nn = np.argpartition(dist, K)[:K]
        preds.append(float(fcum[:cutoff][nn].mean()))
        acts.append(cl[i + FWD] / cl[i] - 1)
        used += 1

    if len(preds) < 60:
        return name, len(preds), None, None, None
    p, a = np.array(preds), np.array(acts)
    rho = spearman(p, a)
    hit = (a[p > 0] > 0).mean() if (p > 0).sum() else float("nan")
    o = np.argsort(p); q5 = np.array_split(o, 5)
    ls = a[q5[-1]].mean() - a[q5[0]].mean()
    return name, len(preds), rho, hit, ls


def main():
    print("載入...", flush=True)
    panel, turn = load_panel()
    regime = build_regime(panel)
    print(f"庫:{len(panel)} 檔,建立中...", flush=True)
    lib = build_library(panel, regime)
    print(f"窗口:{len(lib[3])}\n", flush=True)

    variants = [
        ("基準:basic特徵+全部日子", "basic", "all", "all"),
        ("C:rich特徵+全部日子", "rich", "all", "all"),
        ("A:basic+深跌日", "basic", "drop", "all"),
        ("A+C:rich+深跌日", "rich", "drop", "all"),
        ("A+大型:basic+深跌+大型股", "basic", "drop", "large"),
        ("A+C+大型:rich+深跌+大型股", "rich", "drop", "large"),
    ]
    rows = []
    for nm, f, s, lq in variants:
        r = run_variant(nm, panel, regime, lib, turn, f, s, lq)
        rows.append(r)
        rho = f"{r[2]:+.3f}" if r[2] is not None else "—"
        hit = f"{r[3]*100:.0f}%" if r[3] is not None else "—"
        ls = f"{r[4]*100:+.2f}%" if r[4] is not None else "—"
        print(f"  {r[0]:32s} n={r[1]:5d} Spearman={rho} 看多命中={hit} 多空={ls}", flush=True)

    L = ["# 類比引擎實驗矩陣(A設定閘門 / B路線 / C特徵)\n",
         f"> 庫 {len(panel)} 檔深歷史｜前瞻{FWD}日｜防洩漏｜深跌=近5日≤{DROP_THR*100:.0f}%｜大型=日均成交≥{LARGE_TURN/1e8:.0f}億\n",
         "| 變體 | 樣本 | Spearman | 看多命中% | 多空價差% |",
         "|------|------|------|------|------|"]
    for r in rows:
        rho = f"{r[2]:+.3f}" if r[2] is not None else "—"
        hit = f"{r[3]*100:.0f}%" if r[3] is not None else "—"
        ls = f"{r[4]*100:+.2f}%" if r[4] is not None else "—"
        L.append(f"| {r[0]} | {r[1]} | {rho} | {hit} | {ls} |")
    L += ["", "判讀:Spearman>0 且 多空價差>0 且 看多命中>52% = 該組合有 edge。",
          "對照基準(通用版≈0)看哪個變體把 edge 拉出來,那就是該蓋的系統方向。"]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n完成 → {OUT}")


if __name__ == "__main__":
    main()