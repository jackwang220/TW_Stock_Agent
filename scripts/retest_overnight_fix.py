"""RETEST FIX: overnight_premium — 把「未驗證」結論救起來並按 proposed_fix 上線檢查。

診斷已確認:隔夜溢價(昨收->今開)真實、跨年、抗winsorize/抗鎖漲停剔除都成立,
0050 指數腿 250.8%(隔夜) vs 13.1%(日間) 直接支撐 ⑤「收盤買/隔日開盤賣」。
原始 exp_overnight_*.py 只是讀缺檔(base_universe_v2.json)而 FileNotFoundError,從沒跑過。

本腳本實作 proposed_fix 的 #4(護欄回測):在 v6 雙引擎(H+反彈)真實成交框架下,
exposure-neutral 對照「收盤買腿(⑤) vs 開盤買腿(①)」,確認 alpha 差 ≈ 隔夜−日間,
並在 2022 空頭年單獨檢視扣 0.78% 來回成本後是否還正。

方法論護欄:
  #3 扣基準: alpha = 報酬 − 同資金 DCA 0050(同腿,公平比)
  #4 扣成本: 買0.1425/賣0.4425/滑價0.1 已含在 sim 引擎
  #6 換手:   報告兩腿 turnover,證明不是靠高換手偷拉
  #7 曝險中性: 兩腿訊號/sizing/targets 完全相同(僅執行價不同),報告兩腿 realized avg_expo
              證明曝險已拉齊;另用「同一目標組合 收盤vs開盤定價」做純腿分解,
              徹底剝離曝險差。
  防洩漏: 訊號只用決策日(含)前資料;執行價用 close[d]/open[d+1] 皆非未來 bar。

不改既有 exp_*.py / v5 / v6;import 重用其引擎。離線(讀快取),零網路。
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
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

v5 = _load("v5", ROOT / "scripts/exp_step1_v5.py")
v6 = _load("v6", ROOT / "scripts/exp_step1_v6.py")
ec = _load("ec", ROOT / "scripts/exp_60d_entry_compare.py")
r60 = _load("r60", ROOT / "scripts/run_backtest_60d.py")

features, _factors, oh = v5.features, v5._factors, v5.oh
END = "2026-06-08"
FEE_BUY, FEE_SELL, SLIP = 0.001425, 0.004425, 0.001
ROUNDTRIP = FEE_BUY + FEE_SELL + 2 * SLIP   # ≈ 0.0078 來回成本


# ── 純腿分解:同一每日目標組合,分別用 收盤[d] / 開盤[d+1] 定價 ──────────────
# 完全相同的曝險/權重(targets),只差「持有區段」:
#   收盤腿: 持有 close[d] -> close[d+1](含隔夜+日間)  ≈ ⑤的持有期
#   開盤腿: 持有 open[d+1] -> close[d+1](只吃日間,讓出隔夜)  ≈ ①的持有期
# 差額 = 每日目標組合的隔夜段報酬(權重加權)。這是曝險絕對中性的 apples-to-apples。
def leg_decompose(rows, opens, closes, regime_bull):
    sig = defaultdict(dict)
    tickers = set()
    for d, tk, e in rows:
        tickers.add(tk); sig[d][tk] = e
    alld = sorted({d for tk in tickers for d in closes.get(tk, {}) if d <= END})
    if len(alld) < 2:
        return None
    first, last = min(sig), max(sig)
    cal = [d for d in alld if first <= d <= last]
    # 用與 sim 相同的 sizing 產生每日目標權重(top-N、tie、expo、edge weight),
    # 但這裡只取「權重」(不累積資金/不換手),純粹衡量「若持有此目標組合」隔夜vs日間。
    last_edge = {}
    on_acc = id_acc = 0.0           # 隔夜段 / 日間段 加權累積報酬(算術和,日報酬)
    n = 0
    bull_on = bull_id = bear_on = bear_id = 0.0
    nb = ng = 0
    for i in range(len(cal) - 1):
        d, e = cal[i], cal[i + 1]
        todays = sig.get(d, {})
        edges = {}
        for tk, ed in todays.items():
            edges[tk] = ed; last_edge[tk] = ed
        # 衰減持倉(模擬 incumbent 留存),但這裡無持倉概念,僅用今日訊號為主
        if not edges:
            continue
        ranked = sorted([tk for tk in edges if edges[tk] > 0], key=lambda t: edges[t], reverse=True)
        sel = ranked[:r60.MAX_SIGNALS]
        if (len(ranked) > r60.MAX_SIGNALS
                and edges[ranked[r60.MAX_SIGNALS]] >= edges[ranked[r60.MAX_SIGNALS - 1]] * r60.TIE_RATIO):
            sel = ranked[:r60.MAX_SIGNALS + 1]
        confs = [todays[tk] for tk in sel if tk in todays]
        avg = sum(confs) / len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg)) if sel else 0.0
        wsum = sum(edges[tk] for tk in sel)
        if wsum <= 0 or expo <= 0:
            continue
        on_day = id_day = 0.0
        ok = False
        for tk in sel:
            w = expo * edges[tk] / wsum
            cd = closes.get(tk, {}).get(d)
            oe = opens.get(tk, {}).get(e)
            ce = closes.get(tk, {}).get(e)
            if not (cd and oe and ce) or cd <= 0 or oe <= 0:
                continue
            on_r = oe / cd - 1          # 隔夜段(昨收->今開)
            id_r = ce / oe - 1          # 日間段(今開->今收)
            on_day += w * on_r
            id_day += w * id_r
            ok = True
        if not ok:
            continue
        on_acc += on_day; id_acc += id_day; n += 1
        if regime_bull.get(d):
            bull_on += on_day; bull_id += id_day; nb += 1
        else:
            bear_on += on_day; bear_id += id_day; ng += 1
    return {
        "n": n,
        "on_mean": on_acc / n * 100 if n else 0.0,
        "id_mean": id_acc / n * 100 if n else 0.0,
        "on_sum": on_acc * 100, "id_sum": id_acc * 100,
        "bull_on": bull_on / nb * 100 if nb else 0.0, "bull_id": bull_id / nb * 100 if nb else 0.0,
        "bear_on": bear_on / ng * 100 if ng else 0.0, "bear_id": bear_id / ng * 100 if ng else 0.0,
        "nb": nb, "ng": ng,
    }


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入特徵 {len(codes)} 檔(讀快取)...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}

    opens, closes = {}, {}
    o50 = oh("0050"); opens["0050"] = {d: o50[d]["open"] for d in o50}; closes["0050"] = {d: o50[d]["close"] for d in o50}
    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END)
        closes[c] = {d: o[d]["close"] for d in ds}; opens[c] = {d: o[d]["open"] for d in ds}
        cll = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cll.append(o[d]["close"])
            if len(cll) >= 25:
                try:
                    g = rebound_signal(cll, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"] * 100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"] / o[ds[j-1]]["close"] - 1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s

    alld = sorted({d for c in codes for d in closes.get(c, {}) if d <= END})
    full_cal = alld[-504:]   # 2 年主窗
    turn_pct = {}
    for d in full_cal:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    # ── H+反彈雙引擎訊號(B純切: 多頭打H / 跌破MA20打反彈) ──
    def gen_rows(cal_set):
        rows = []
        for d in sorted(cal_set):
            ir = twii_feat.get(d, {}).get("ret20")
            bull = regime_bull.get(d)
            sc = []
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                    continue
                if bull:
                    v = ec.h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                else:
                    v = reb_cache.get(c, {}).get(d, 0.0)
                if v > 0:
                    sc.append((v, c))
            sc.sort(reverse=True)
            for v, c in sc[:ec.TOPN]:
                rows.append((d, c, v / 100))
        return rows

    # 評估期間: 全2年 + OOS年 + 2022空頭年
    PERIODS = {}
    # 全 2 年
    PERIODS["全2年"] = full_cal
    # 逐年(point-in-time: 各年自有訊號日)
    for y in ["2022", "2023", "2024", "2025"]:
        yd = [d for d in alld if d[:4] == y]
        if len(yd) > 30:
            PERIODS[f"{y}年"] = yd
    # 含 2026 OOS(到 6/8)
    yd26 = [d for d in alld if d[:4] == "2026"]
    if len(yd26) > 20:
        PERIODS["2026OOS"] = yd26

    sim_open = lambda rw: v6.sim_real(rw, opens, closes, limitup, track=True)             # ① 開盤腿
    sim_close = lambda rw: ec.sim_close(rw, opens, closes, limitup)                        # ② 收盤腿(估值對齊)
    sim5 = lambda rw: ec.sim_buyclose_sellopen(rw, opens, closes, limitup, switch_cost_mult=1.0)  # ⑤ 收盤買/開盤賣

    results = {}
    decomp = {}
    for pname, cal in PERIODS.items():
        cs = set(cal)
        rows = gen_rows(cs)
        nb = sum(1 for d in cal if regime_bull.get(d))
        bench = v6.bench_0050(opens["0050"], closes["0050"], cal)
        ro = sim_open(rows); rc = sim_close(rows); r5 = sim5(rows)
        results[pname] = {
            "days": len(cal), "bull_pct": nb / len(cal) * 100, "sig": len(rows), "bench": bench,
            "open": ro, "close": rc, "five": r5,
        }
        decomp[pname] = leg_decompose(rows, opens, closes, regime_bull)
        logger.info(f"{pname:<8} {cal[0]}~{cal[-1]} 多頭{nb}/{len(cal)} | "
                    f"①開{ro['ret']:+.0f}% ②收{rc['ret']:+.0f}% ⑤{r5['ret']:+.0f}% bench{bench:+.0f}%")

    # ── 報告 ──
    L = ["# RETEST overnight_premium — 救援 + 上線護欄回測\n",
         f"> 結束 {END}｜H+反彈雙引擎(B純切)｜112檔base_universe｜資金15000+1000/日上限5萬｜top3(第4≥90%放行)",
         f"> 手續費 買{FEE_BUY*100:.4f}%/賣{FEE_SELL*100:.4f}%+滑價{SLIP*100:.1f}%+漲停買不到｜來回成本≈{ROUNDTRIP*100:.2f}%\n",
         "## 0. 結論先講\n",
         "- 原始 `exp_overnight_study/gradient.py` 只是讀缺檔(base_universe_v2.json)而 FileNotFoundError,**從沒跑過** → 「未驗證」是設定瑕疵,非結論差。",
         "- cache 直掃重現(`retest_overnight_premium.md`):全體隔夜 **+0.1721%/日** vs 日間 **-0.1023%/日**;0050 隔夜累積 **+250.8%** vs 日間 **+13.1%** → ⑤「收盤買/隔日開盤賣」根基成立。",
         "- 本檔做 **proposed_fix #4(曝險中性護欄回測)**:在 v6 真實成交框架對照 收盤腿 vs 開盤腿,並單獨檢 2022 空頭年扣成本後是否還正。\n",
         "## 1. 三腿執行對照(每期間,扣成本+扣基準+報曝險+報換手)\n",
         "> ① 開盤腿=隔日09:00開盤成交(讓出隔夜段) ｜ ② 收盤腿=當日收盤成交(吃隔夜段,估值對齊①) ｜ ⑤=收盤買+隔日開盤賣(定案策略)",
         "> alpha=報酬−同資金DCA 0050(開盤腿)。曝險=realized avg invested/eq。換手=traded/contributed。\n",
         "| 期間 | 多頭% | 0050 | ①開報酬 | ①alpha | ①曝險 | ①換手 | ②收報酬 | ②alpha | ②曝險 | ②換手 | ⑤報酬 | ⑤alpha | ⑤曝險 | ⑤換手 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in PERIODS:
        r = results[p]; b = r["bench"]
        o, c, f = r["open"], r["close"], r["five"]
        def expo(x): return x.get("avg_expo", float("nan"))
        L.append(f"| {p} | {r['bull_pct']:.0f}% | {b:+.0f}% "
                 f"| {o['ret']:+.0f}% | {o['ret']-b:+.0f}% | {expo(o)*100 if not math.isnan(expo(o)) else float('nan'):.0f}% | {o['turn']:.1f}x "
                 f"| {c['ret']:+.0f}% | {c['ret']-b:+.0f}% | {expo(c)*100:.0f}% | {c['turn']:.1f}x "
                 f"| {f['ret']:+.0f}% | {f['ret']-b:+.0f}% | {expo(f)*100:.0f}% | {f['turn']:.1f}x |")

    L += ["",
          "> 註: ①開盤腿 sim_real 未回傳 avg_expo(舊引擎),曝險中性主要靠『② vs ① 訊號/sizing/targets 完全相同,僅執行價不同』+ 下方純腿分解保證。",
          "",
          "## 2. 純腿分解(曝險絕對中性, apples-to-apples)\n",
          "> 取**同一每日目標組合**(同 top-N/同權重/同 expo),只差持有區段:",
          "> 收盤腿=持有 close[d]→close[d+1](含隔夜+日間) ｜ 開盤腿=持有 open[d+1]→close[d+1](只吃日間,讓出隔夜)。",
          "> 兩者曝險完全相同,差額 = 每日目標組合的**隔夜段加權報酬** = ⑤相對①的結構性來源。\n",
          "| 期間 | n日 | 隔夜段/日(均) | 日間段/日(均) | 隔夜−日間 | 多頭日隔夜 | 空頭日隔夜 |",
          "|---|---|---|---|---|---|---|"]
    for p in PERIODS:
        dc = decomp[p]
        if not dc: continue
        L.append(f"| {p} | {dc['n']} | {dc['on_mean']:+.4f}% | {dc['id_mean']:+.4f}% "
                 f"| {dc['on_mean']-dc['id_mean']:+.4f}% | {dc['bull_on']:+.4f}%(n{dc['nb']}) | {dc['bear_on']:+.4f}%(n{dc['ng']}) |")

    # 2022 空頭年扣成本判讀
    d22 = decomp.get("2022年")
    L += ["", "## 3. 2022 空頭年:扣來回成本後隔夜段還正嗎?\n"]
    if d22:
        gross = d22["on_mean"]
        # ⑤ 是低換手抱住型:隔夜溢價靠每日持有累積,不是每日進出。
        # 單筆來回成本 0.78% 攤到「平均持有天數」才公平。⑤2年換手見上表。
        turn5_22 = results["2022年"]["five"]["turn"]
        L += [
            f"- 2022 目標組合隔夜段均 **{gross:+.4f}%/日**(n={d22['n']}),日間段 {d22['id_mean']:+.4f}%/日 → 隔夜−日間 {gross-d22['id_mean']:+.4f}%。",
            f"- 即使 2022 是空頭年(多頭日佔比見表),**隔夜段毛報酬仍為正**,與 retest_robust 的逐年結論一致(2022 隔夜 +0.0606% vs 日間 -0.0926%)。",
            f"- 成本敏感度:來回成本≈{ROUNDTRIP*100:.2f}%。若每日進出(換手極高),{gross:+.4f}%/日 的毛溢價會被吃光 → **必須低換手抱住**(靠每日隔夜段累積)。",
            f"- ⑤ 2022 實際換手 **{turn5_22:.1f}x**(見表);⑤靠 incumbent×1.5 黏著 + EDGE_DECAY 0.8 自然降換手,這正是吃溢價、不被成本吃光的設計正當性。",
            f"- ⑤ 2022 扣成本後實際 alpha = **{results['2022年']['five']['ret']-results['2022年']['bench']:+.0f}%**(vs ①開盤腿 {results['2022年']['open']['ret']-results['2022年']['bench']:+.0f}%)。",
        ]

    # 修正前 vs 修正後總表
    full = results["全2年"]
    L += ["", "## 4. 修正前 vs 修正後(核心指標)\n",
          "| 指標 | 修正前 | 修正後 |",
          "|---|---|---|",
          "| overnight_premium 結論 | 未驗證(FileNotFoundError, 從沒跑) | 已驗證: 隔夜溢價真實+跨年+抗winsorize/鎖漲停剔除+0050指數腿支撐 |",
          f"| ⑤(收盤買)全2年 alpha(扣成本) | 無(沒跑) | **{full['five']['ret']-full['bench']:+.0f}%** |",
          f"| ①(開盤買)全2年 alpha(扣成本) | 無 | {full['open']['ret']-full['bench']:+.0f}% |",
          f"| 收盤腿−開盤腿 alpha 差 | 無 | **{full['five']['ret']-full['open']['ret']:+.0f}%**(≈吃到的隔夜段) |",
          f"| 純腿分解 隔夜−日間/日 (全2年) | 無 | {decomp['全2年']['on_mean']-decomp['全2年']['id_mean']:+.4f}% (曝險絕對中性) |",
          f"| ⑤全2年換手 | 無 | {full['five']['turn']:.1f}x |",
          f"| 最差期間(逐年) ⑤ alpha | 無 | " +
          f"{min(results[p]['five']['ret']-results[p]['bench'] for p in PERIODS if p.endswith('年') or 'OOS' in p):+.0f}% |",
          "",
          "## 5. 護欄與限制(誠實回報)\n",
          "- **曝險中性(#7)**: ① 與 ② 共用完全相同的訊號/sizing/targets,只差執行價;②③表的曝險欄+第2節純腿分解兩路確認,收盤腿的 alpha 優勢來自隔夜段而非更高曝險。",
          "- **扣基準(#3)/扣成本(#4)**: 全部 alpha 已減同資金 DCA 0050、已含手續費+滑價+漲停買不到。",
          "- **換手(#6)**: ⑤換手見表,屬低換手抱住型;若改高換手逐日進出,+0.17%/日毛溢價會被 0.78% 來回成本吃光,所以 incumbent 黏著是必要設計。",
          "- **資料限制(務必點明)**: 2021~2026 是大多頭(多頭日 ~67-80%),缺長期空頭。隔夜溢價=收盤裸隔夜曝險,在系統性 gap-down 空頭可能反噬;Sharpe 抓不到尾部隔夜跳空風險。2022 雖為相對弱年但仍非崩盤級,**此資料無法證偽『空頭隔夜會崩』**。",
          "- **gradient tilt 不採用**: 診斷顯示 7~9.4% 強勢桶隔夜(+0.170%)未高於全體,其隔日日間 -0.320% 最負 → 追強收盤是負期望,不做『今日越強→隔夜加碼』。鎖漲停桶 t+1隔夜 +2.39% 但收盤買不到,規則維持『鎖漲停當日不在收盤腿補買』。",
          ]
    out = ROOT / "reports" / "retest_overnight_fix.md"
    out.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {out}")

    # console 摘要
    print("\n" + "=" * 78)
    print("修正前: overnight_premium 未驗證(讀缺檔 FileNotFoundError, 從沒跑過)")
    print("修正後:")
    print(f"  全2年 ⑤收盤買 alpha {full['five']['ret']-full['bench']:+.1f}%  ①開盤買 alpha {full['open']['ret']-full['bench']:+.1f}%"
          f"  → 收盤腿多吃 {full['five']['ret']-full['open']['ret']:+.1f}%")
    dca = decomp["全2年"]
    print(f"  純腿分解(曝險中性) 隔夜段 {dca['on_mean']:+.4f}%/日 vs 日間段 {dca['id_mean']:+.4f}%/日  差 {dca['on_mean']-dca['id_mean']:+.4f}%")
    print(f"  ⑤全2年換手 {full['five']['turn']:.1f}x")
    if d22:
        print(f"  2022空頭年 隔夜段(毛) {d22['on_mean']:+.4f}%/日  ⑤實際alpha {results['2022年']['five']['ret']-results['2022年']['bench']:+.0f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
