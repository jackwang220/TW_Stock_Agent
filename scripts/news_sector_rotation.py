"""消息面軌 — 族群輪動超配（純量化版主題面）。

假設:把 universe 按 industry 分群,排序各族群近 20/60 日「相對大盤超額報酬」,
超配最強 1-2 個族群、低配最弱族群 → 看族群輪動是否帶 alpha。

做法:
  baseline = H+反彈雙引擎(多頭打 H 動能/空頭打反彈),⑤執行(收盤買+開盤賣)。
            這是三方確認的現行冠軍,直接重用 exp_60d_entry_compare 的訊號生成。
  approach = 在 baseline edge 上乘一個「族群輪動 tilt」:
            tilt_mult(stock) = 1 + W * sector_rank_z(industry_of_stock, d)
            sector_rank_z ∈ [-1,+1]:該股所屬族群當日相對強度排名(扣大盤)的標準化分數。
            最強族群 → +1(超配)、最弱族群 → -1(低配)。W 為 tilt 強度,掃 0.1~2.0。

族群相對強度(point-in-time,逐日只用 ≤ d 的價):
  sector_RS(ind, d) = 該族群成分股 (20日報酬 與 60日報酬 各扣 0050 同窗報酬) 的等權平均,
  再混合 0.5*RS20 + 0.5*RS60。每日對所有族群排名 → 轉成 [-1,+1] 的 z(線性映射)。

洩漏註記:族群成員(industry 欄)取自 base_universe.json,為目前的靜態分類(survivorship/
  靜態膜成),非 point-in-time;但相對強度訊號本身只用 ≤ d 的還原價,無未來價洩漏。

ALPHA = 策略 - 同資金 DCA 0050(開盤),扣真實成本,曝險中性比較(報 avg_expo)。
輸出 v15 格式多窗表(60天/90天/半年/1年/1年半/2年 + 5 regime + 最差 + 平均 + 換手 + 持股)。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

v3 = _load("v3", ROOT / "scripts/exp_step1_v3.py")
v6 = _load("v6", ROOT / "scripts/exp_step1_v6.py")
ec = _load("ec", ROOT / "scripts/exp_60d_entry_compare.py")
r60 = _load("r60", ROOT / "scripts/run_backtest_60d.py")
features, _factors, oh = v3.features, v3._factors, v3.oh
h_score = ec.h_score
sim5 = ec.sim_buyclose_sellopen
bench_0050 = v6.bench_0050

END = "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252), ("1年半", 378), ("2年", 504)]
# regime 子區間(交易日索引,從 cal 尾端往回算;對齊 v15 的 5 regime 語意)
REGIMES = [
    ("2021復甦", 504, 420), ("2022空頭", 420, 290), ("2023復甦", 290, 200),
    ("2024-25多頭", 200, 60), ("2025下-26", 60, 0),
]
TOPN = ec.TOPN
W_SWEEP = [0.0, 0.1, 0.3, 0.6, 1.0, 2.0]   # 0.0 = baseline


def sector_rs_by_day(codes, ind_of, closes, cal):
    """逐日計算每檔的『族群相對強度 z 分數』∈[-1,+1]。
    sector_RS(ind,d) = 成分股 (0.5*RS20 + 0.5*RS60) 等權平均, RS = 個股報酬 - 0050 同窗報酬。
    每日對族群排名 → 線性映射到 [-1,+1]。回傳 z[d][code]。
    """
    # 0050 報酬序列
    c50 = closes.get("0050", {})
    def ret_n(c, d, n):
        s = closes.get(c, {})
        ds = [x for x in s if x <= d]
        if len(ds) <= n: return None
        ds = ds[-(n+1):]
        p0, p1 = s[ds[0]], s[ds[-1]]
        return (p1 / p0 - 1) if p0 > 0 else None
    z = {}
    for d in cal:
        r50_20 = ret_n("0050", d, 20); r50_60 = ret_n("0050", d, 60)
        if r50_20 is None or r50_60 is None:
            z[d] = {}; continue
        # 個股 RS
        per_ind = defaultdict(list)
        for c in codes:
            r20 = ret_n(c, d, 20); r60 = ret_n(c, d, 60)
            if r20 is None or r60 is None: continue
            rs = 0.5 * (r20 - r50_20) + 0.5 * (r60 - r50_60)
            per_ind[ind_of.get(c, "?")].append(rs)
        sec = {ind: sum(v) / len(v) for ind, v in per_ind.items() if v}
        if len(sec) < 2:
            z[d] = {}; continue
        lo, hi = min(sec.values()), max(sec.values())
        rng = hi - lo
        # 族群 z ∈[-1,+1]
        sec_z = {ind: (2 * (val - lo) / rng - 1) if rng > 0 else 0.0 for ind, val in sec.items()}
        z[d] = {c: sec_z.get(ind_of.get(c, "?"), 0.0) for c in codes}
    return z


def build_rows(codes, feats, twii_feat, reb_cache, turn_pct, regime_bull, sig_days, sec_z, W):
    """baseline 雙引擎訊號 × 族群 tilt。W=0 → baseline。tilt = 1 + W*sec_z(clamp≥0)。"""
    rows = []
    for d in sig_days:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        zd = sec_z.get(d, {})
        sc = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            if bull:
                v = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            else:
                v = reb_cache.get(c, {}).get(d, 0.0)
            if v <= 0: continue
            if W != 0.0:
                tilt = 1.0 + W * zd.get(c, 0.0)
                if tilt < 0.0: tilt = 0.0
                v *= tilt
            sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:TOPN]:
            rows.append((d, c, v / 100))
    return rows


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    ind_of = {c: u[c].get("industry", "?") for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入特徵({len(codes)} 檔)...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}

    opens, closes = {}, {}
    for c in codes + ["0050"]:
        o = oh(c)
        opens[c] = {d: o[d]["open"] for d in o}; closes[c] = {d: o[d]["close"] for d in o}

    alld = sorted({d for c in codes for d in closes.get(c, {}) if d <= END})
    cal = alld[-WINDOWS[-1][1]:]
    sig_days = cal

    logger.info("反彈訊號 + 漲停日...")
    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END)
        cl_list = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"] * 100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"] / o[ds[j-1]]["close"] - 1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s

    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in sig_days}

    logger.info("族群相對強度(point-in-time)...")
    sec_z = sector_rs_by_day(codes, ind_of, closes, cal)

    # 基準:同資金 DCA 0050(開盤),逐窗 + 逐 regime
    bench = {}
    for wl, n in WINDOWS:
        bench[wl] = bench_0050(opens["0050"], closes["0050"], cal[-n:])
    for rl, a, b in REGIMES:
        seg = cal[max(0, len(cal)-a):len(cal)-b] if b > 0 else cal[max(0, len(cal)-a):]
        bench[rl] = bench_0050(opens["0050"], closes["0050"], seg)

    # 跑各 W
    res = {}   # (W, label) -> sim dict
    for W in W_SWEEP:
        rows_full = build_rows(codes, feats, twii_feat, reb_cache, turn_pct, regime_bull, sig_days, sec_z, W)
        for wl, n in WINDOWS:
            wd = set(cal[-n:]); rw = [r for r in rows_full if r[0] in wd]
            res[(W, wl)] = sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)
        for rl, a, b in REGIMES:
            seg = cal[max(0, len(cal)-a):len(cal)-b] if b > 0 else cal[max(0, len(cal)-a):]
            wd = set(seg); rw = [r for r in rows_full if r[0] in wd]
            res[(W, rl)] = sim5(rw, opens, closes, limitup, switch_cost_mult=1.0) if len(seg) >= 2 else None
        logger.info(f"W={W} 完成")

    def alp(W, lbl):
        r = res.get((W, lbl))
        return (r["ret"] - bench[lbl]) if r else None

    labels = [wl for wl, _ in WINDOWS] + [rl for rl, _, _ in REGIMES]
    longw = ["1年", "1年半", "2年"]

    # 報表
    L = ["# 消息面軌 — 族群輪動超配（純量化主題面）｜v15 格式多窗\n",
         f"> baseline=H+反彈雙引擎(多頭H/空頭反彈)｜⑤收盤買+開盤賣｜{len(codes)}檔｜結束{END}｜還原價｜DCA(15000+1000/日上限5萬)\n",
         "> approach=baseline edge × 族群輪動 tilt(1+W·sec_z, sec_z∈[-1,+1]);W=族群輪動超配強度\n",
         "> 族群RS=成分股(0.5·RS20+0.5·RS60, 各扣0050同窗)等權均, point-in-time(≤d還原價);W=0即baseline\n",
         "> ALPHA=策略−同資金DCA0050(開盤)｜扣費 買0.14%/賣0.44%+滑價0.1%+漲停買不到\n",
         "> 洩漏註記:industry分類取自base_universe(靜態,非PIT);相對強度訊號本身無未來價洩漏\n",
         "> 0050基準: " + " ".join(f"{lbl}{bench[lbl]:+.0f}%" for lbl in labels) + "\n",
         "## 🎯 ALPHA %(扣大盤beta) — 各 tilt 強度 W × 6窗 + 5regime\n",
         "| W(tilt) | " + " | ".join(labels) + " | 最差 | 平均 | 換手(2年) | 持股(2年) |",
         "|---|" + "|".join(["---"] * len(labels)) + "|---|---|---|---|"]
    for W in W_SWEEP:
        cells = []
        for lbl in labels:
            v = alp(W, lbl)
            cells.append(f"{v:+.0f}" if v is not None else "—")
        allv = [alp(W, lbl) for lbl in labels if alp(W, lbl) is not None]
        worst = min(allv) if allv else None
        avg = sum(allv) / len(allv) if allv else None
        r2 = res.get((W, "2年"))
        turn = f"{r2['turn']:.0f}x" if r2 else "—"
        pos = f"{r2['avg_pos']:.1f}" if r2 else "—"
        tag = "(baseline)" if W == 0.0 else ""
        L.append(f"| {W}{tag} | " + " | ".join(cells) +
                 f" | **{worst:+.0f}** | {avg:+.0f} | {turn} | {pos} |")

    # 原始報酬 + 曝險(曝險中性檢查)
    L += ["", "## 參考:原始報酬%(未扣大盤) + 平均曝險(曝險中性檢查)\n",
          "| W | 60天 | 90天 | 半年 | 1年 | 1年半 | 2年 | 曝險(2年) |",
          "|---|---|---|---|---|---|---|---|"]
    for W in W_SWEEP:
        cells = " | ".join(f"{res[(W,wl)]['ret']:+.0f}" if res.get((W,wl)) else "—" for wl, _ in WINDOWS)
        r2 = res.get((W, "2年"))
        L.append(f"| {W} | {cells} | {r2['avg_expo']*100:.0f}% |" if r2 else f"| {W} | {cells} | — |")

    # uplift vs baseline(每窗)
    L += ["", "## uplift = approach − baseline 的 alpha 差(pp,逐窗)\n",
          "| W | " + " | ".join(labels) + " | 平均uplift |", "|---|" + "|".join(["---"]*len(labels)) + "|---|"]
    base_alp = {lbl: alp(0.0, lbl) for lbl in labels}
    for W in W_SWEEP:
        if W == 0.0: continue
        cells = []; ups = []
        for lbl in labels:
            a = alp(W, lbl); b = base_alp[lbl]
            if a is None or b is None: cells.append("—"); continue
            up = a - b; cells.append(f"{up:+.0f}"); ups.append(up)
        avgup = sum(ups)/len(ups) if ups else None
        L.append(f"| {W} | " + " | ".join(cells) + f" | **{avgup:+.0f}** |")

    REPORT = ROOT / "reports" / "news_sector_rotation.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")

    # 終端摘要
    print("\n=== ALPHA(扣大盤) 摘要 ===")
    print("W      " + "  ".join(f"{wl:>5}" for wl, _ in WINDOWS) + "   平均  最差")
    for W in W_SWEEP:
        cells = " ".join(f"{(alp(W,wl) or 0):+6.0f}" for wl, _ in WINDOWS)
        allv = [alp(W, lbl) for lbl in labels if alp(W, lbl) is not None]
        print(f"{W:<5} {cells}  {sum(allv)/len(allv):+6.0f} {min(allv):+6.0f}")
    print("\n=== uplift vs baseline(平均跨全窗+regime) ===")
    for W in W_SWEEP:
        if W == 0.0: continue
        ups = [alp(W, lbl) - base_alp[lbl] for lbl in labels
               if alp(W, lbl) is not None and base_alp[lbl] is not None]
        print(f"W={W}: 平均uplift {sum(ups)/len(ups):+.1f}pp  最佳窗 {max(ups):+.0f}  最差窗 {min(ups):+.0f}")


if __name__ == "__main__":
    main()
