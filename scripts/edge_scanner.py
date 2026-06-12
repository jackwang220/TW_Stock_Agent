"""Edge 掃描器:系統化找「類深跌反彈」的條件式 edge,內建樣本外確認。

對液性深歷史庫,網格掃描【設定 × 條件】,每組算前瞻5日 Alpha/勝率,
並切成【樣本內 2016-2021】與【樣本外 2022-今】分別算 —— 只有兩段都成立的才採信。
這條樣本外濾網就是擋假陽性的核心(我們被 leak 騙過一次,這次靠 OOS 把關)。

leak-safe by design:設定只用 ≤當日;前瞻只用未來;每檔用自己的未來(無跨股庫、無 bisect)。
Alpha = 個股前瞻報酬 − 同期大盤(全市場中位數)前瞻報酬,長天期防 beta。

用法:python scripts/edge_scanner.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
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
OUT = REPORTS_DIR / "edge_scan.md"
FWD = 5
FRICTION = 0.005          # 來回成本 ~0.5%
OOS_DATE = "2022-01-01"   # 樣本內/外切分
LARGE_TURN = 5e8


def load():
    base_f = DATA_DIR / "base_universe.json"
    base = json.loads(base_f.read_text(encoding="utf-8")) if base_f.exists() else {}
    only = set(base) or None
    turn = {c: v.get("avg_turnover", 0) for c, v in base.items()}
    panel = {}
    for f in glob.glob(str(CACHE / "TaiwanStockPrice_*.json")):
        code = Path(f).stem.replace("TaiwanStockPrice_", "")
        if only and code not in only:
            continue
        try:
            rows = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        ds, o, h, l, c, v = [], [], [], [], [], []
        for r in rows:
            try:
                cc = float(r["close"])
            except (TypeError, ValueError, KeyError):
                continue
            if cc > 0:
                ds.append(r["date"]); c.append(cc)
                o.append(float(r.get("open") or cc)); h.append(float(r.get("max") or cc))
                l.append(float(r.get("min") or cc)); v.append(float(r.get("Trading_Volume") or 0))
        if len(c) > 300:
            idx = np.argsort(ds)
            panel[code] = {k: list(np.array(val)[idx]) for k, val in
                           zip("dohlcv", [ds, o, h, l, c, v])}
    return panel, turn


def market_index(panel):
    rb = defaultdict(list)
    for p in panel.values():
        d, c = p["d"], p["c"]
        for i in range(1, len(c)):
            if c[i - 1] > 0:
                rb[d[i]].append(c[i] / c[i - 1] - 1)
    dates = sorted(rb)
    med = np.array([np.median(rb[x]) for x in dates])
    idx = np.cumprod(1 + med)
    val = dict(zip(dates, idx))
    # regime(往回看):趨勢(對200MA)
    trend = {}
    arr = idx
    for k, x in enumerate(dates):
        trend[x] = (arr[k] / arr[max(0, k - 200):k].mean() - 1) if k >= 200 else None
    return val, trend


def rsi(c, i, n=14):
    g = l = 0.0
    for j in range(i - n + 1, i + 1):
        ch = c[j] - c[j - 1]
        g += max(ch, 0); l += max(-ch, 0)
    if l == 0:
        return 100.0
    return 100 - 100 / (1 + g / l)


# ── 設定(每個回傳 bool,只用 ≤i 資料)──────────────────────────────────────────
def setups(p, i):
    c, o, v, l = p["c"], p["o"], p["v"], p["l"]
    out = {}
    r1 = c[i] / c[i - 1] - 1
    r5 = c[i] / c[i - 5] - 1
    ma20 = np.mean(c[i - 19:i + 1])
    bias = c[i] / ma20 - 1 if ma20 > 0 else 0
    vma = np.mean(v[i - 19:i + 1]) or 1
    vr = v[i] / vma
    gap = o[i] / c[i - 1] - 1 if c[i - 1] > 0 else 0
    rs = rsi(c, i)
    cd = cu = 0
    for j in range(i, max(i - 8, 0), -1):
        ch = c[j] - c[j - 1]
        if ch < 0 and cu == 0:
            cd += 1
        elif ch > 0 and cd == 0:
            cu += 1
        else:
            break
    hi252 = max(c[max(0, i - 251):i + 1]); lo252 = min(c[max(0, i - 251):i + 1])
    out["深跌5日≤-8%"] = r5 <= -0.08
    out["深跌5日≤-12%"] = r5 <= -0.12
    out["深跌5日≤-20%"] = r5 <= -0.20
    out["單日跌停"] = r1 <= -0.095
    out["跳空跌≥3%"] = gap <= -0.03
    out["跳空漲≥3%"] = gap >= 0.03
    out["爆量≥2.5x"] = vr >= 2.5
    out["RSI超賣<30"] = rs < 30
    out["RSI超買>70"] = rs > 70
    out["連3黑"] = cd >= 3
    out["連3紅"] = cu >= 3
    out["創52週高"] = c[i] >= hi252
    out["創52週低"] = c[i] <= lo252
    out["乖離≤-10%"] = bias <= -0.10
    out["乖離≥+15%"] = bias >= 0.15
    # ── 借鏡幣圈:強制/恐慌賣壓竭盡(清算瀑布)→ 反彈更猛(價量當代理)──────────
    deep = r5 <= -0.12
    out["竭盡:深跌+爆量≥2x"] = deep and vr >= 2.0          # 恐慌climax
    out["對照:深跌+無爆量"] = deep and vr < 1.2            # 安靜陰跌(對照組)
    out["竭盡:深跌+末日跌停"] = deep and r1 <= -0.095       # 最後一根跌停=斷頭
    out["竭盡:深跌+末日≤-5%"] = deep and r1 <= -0.05        # 加速崩(瀑布還新鮮)
    out["竭盡:深跌+末日爆量黑"] = deep and r1 <= -0.03 and vr >= 2.0  # 末日放量殺
    # ── Liquidity sweep:盤中跌破近期低點(掃停損/斷頭)但收盤站回 = 假跌破反轉 ──
    lo20 = min(l[i - 20:i]); lo60 = min(l[i - 60:i])
    out["掃單reclaim20"] = l[i] < lo20 <= c[i]                 # 破20日低後收回
    out["掃單reclaim20+深跌"] = (l[i] < lo20 <= c[i]) and r5 <= -0.08
    out["掃單reclaim60大支撐"] = l[i] < lo60 <= c[i]           # 破60日低後收回
    return out


def main():
    print("載入液性深庫...", flush=True)
    panel, turn = load()
    idxval, trend = market_index(panel)
    print(f"{len(panel)} 檔,掃描中...", flush=True)

    # acc[(setup, cond)] = {'IS':[alpha], 'OOS':[alpha], 'IS_net':[net], 'OOS_net':[net]}
    acc = defaultdict(lambda: {"IS": [], "OOS": [], "ISn": [], "OOSn": []})

    def conds(code, dt):
        out = ["全部"]
        if turn.get(code, 0) >= LARGE_TURN:
            out.append("大型")
        tr = trend.get(dt)
        if tr is not None:
            out.append("空頭" if tr < 0 else "多頭")
            if turn.get(code, 0) >= LARGE_TURN:
                out.append("大型+空頭" if tr < 0 else "大型+多頭")
        return out

    for code, p in panel.items():
        c, d = p["c"], p["d"]
        n = len(c)
        for i in range(60, n - FWD):
            if c[i] <= 0 or c[i - 5] <= 0:
                continue
            dt = d[i]
            if dt not in idxval or d[i + FWD] not in idxval:
                continue
            fwd = c[i + FWD] / c[i] - 1
            mkt = idxval[d[i + FWD]] / idxval[dt] - 1
            alpha = fwd - mkt
            net = fwd - FRICTION
            seg = "IS" if dt < OOS_DATE else "OOS"
            cs = conds(code, dt)
            for cd in cs:                          # 基準:該條件「所有日子」
                b = acc[("__ALL__", cd)]
                b[seg].append(alpha); b[seg + "n"].append(net)
            fired = setups(p, i)
            for sname, ok in fired.items():
                if not ok:
                    continue
                for cd in cs:
                    a = acc[(sname, cd)]
                    a[seg].append(alpha); a[seg + "n"].append(net)

    def stat(lst):
        if not lst:
            return (0, 0.0, 0.0)
        a = np.array(lst)
        return (len(a), float(a.mean()), float((a > 0).mean()))

    # 各條件的「基準」= 該條件所有日子(扣掉右偏skew + 條件本身的漂移)
    base = {}
    for cd in ["全部", "大型", "空頭", "多頭", "大型+空頭", "大型+多頭"]:
        a = acc.get(("__ALL__", cd))
        if a:
            _, baI, bwI = stat(a["IS"]); _, baO, bwO = stat(a["OOS"])
            base[cd] = (bwI, baI, bwO, baO)

    rows = []
    for (sname, cd), a in acc.items():
        if sname == "__ALL__" or cd not in base:
            continue
        nI, aI, wI = stat(a["IS"]); nO, aO, wO = stat(a["OOS"])
        if nI < 40 or nO < 40:
            continue
        bwI, baI, bwO, baO = base[cd]
        exwI, exwO = wI - bwI, wO - bwO       # 超額勝率(對同條件基準)
        exaO = aO - baO                        # 超額 alpha
        netO = np.mean(a["OOSn"]) if a["OOSn"] else 0
        robust = (exwI >= 0.03 and exwO >= 0.03) or (exwI <= -0.03 and exwO <= -0.03)
        rows.append((sname, cd, nO, wO, bwO, exwO, exaO, netO, robust))

    rows.sort(key=lambda r: (not r[8], -r[5]))

    L = ["# Edge 掃描器 v2:設定 × 條件(扣基準 + 樣本外確認)\n",
         f"> 液性庫 {len(panel)} 檔｜前瞻{FWD}日｜IS<{OOS_DATE}, OOS>={OOS_DATE}｜成本{FRICTION*100:.1f}%\n",
         "> 勝率=贏過大盤中位數比例(基準理論~50%)。**超額勝=設定勝率−同條件基準勝率**(扣掉skew偏差才是真edge)。\n",
         "> ✅robust = IS/OOS 超額勝同向且都≥3pts。\n",
         "| 設定 | 條件 | OOS_n | OOS勝% | 基準勝% | **超額勝pts** | 超額α% | OOS淨% | robust |",
         "|------|------|------|------|------|------|------|------|------|"]
    for r in rows:
        sname, cd, nO, wO, bwO, exwO, exaO, netO, rob = r
        L.append(f"| {sname} | {cd} | {nO} | {wO*100:.0f}% | {bwO*100:.0f}% | "
                 f"**{exwO*100:+.0f}** | {exaO*100:+.2f} | {netO*100:+.2f} | {'✅' if rob else ''} |")
    nrob = sum(1 for r in rows if r[8])
    L += ["", f"**扣基準後仍 robust 的 edge:{nrob} 個。**",
          "判讀:超額勝 pts 越大越真。跌深反彈家族應遙遙領先;動能類若超額勝仍>0且robust=真,≈0=skew假象。"]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n完成 → {OUT}　robust: {nrob}")
    for r in rows[:15]:
        print(f"  {'✅' if r[8] else '  '} {r[0]:13s}{r[1]:8s} 超額勝{r[5]*100:+.0f}pts(勝{r[3]*100:.0f}% vs基準{r[4]*100:.0f}%, n={r[2]})", flush=True)


if __name__ == "__main__":
    main()