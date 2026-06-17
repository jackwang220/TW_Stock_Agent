"""對抗式 leak-free 驗證：把 aetf_mechanize.py 的作弊捷徑全部改掉，重跑看剩多少 uplift。

被驗證的第一輪聲稱：動能 L60N6 vs H同池，逐窗 uplift 60天-1 / 90天+35 / 半年+137 /
1年+307 / 1年半+287 / 2年+251 / 2025下-26 +279 pp。

═══ 原作弊 → leak-free 修法 ═══
  作弊#1[選股池 survivorship+時點洩漏]:
     原 = 28檔主動式ETF「全期持股聯集50檔」，用整年(含未來)成員定義池。
     ★致命：CSV 只涵蓋 2025-06-17~2026-06-17，但回測窗開到 2年(504交易日,回溯到~2024-06)。
            等於拿「2025-26 ETF 實際持有的 50 檔大型贏家」去回測 2024 的價格 = 純未來選股。
     leak-free 修法 = (a) point-in-time 成員：訊號日 d 只能用「揭露日 ≤ d 的 ETF 成員」；
                      (b) 揭露落後：ETF 持股盤後揭露→隔日才可知→成員資格用 D-1 揭露(forward)；
                      (c) 回測期只能落在 CSV 有揭露的區間(2025-06-17→END)，更早無 point-in-time 成員 → 不可回測。
  作弊#2[in-sample 調參]:
     原 = L/TOPN 在同一段資料掃最佳(L60N6)。
     leak-free 修法 = 時序切分：前半段(train)選 (L,N)，後半段(test=OOS)評估，只報 OOS。
  基準與成本：原本就不作弊，沿用(alpha=策略−同資金DCA0050、⑤漲停買不到、報曝險)。

用法: uv run python scripts/aetf_mechanize_leakfree.py
"""
from __future__ import annotations
import sys, json, csv, importlib.util, math, bisect
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
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
features, _factors, oh = v5.features, v5._factors, v5.oh
h_score = ec.h_score
sim5 = ec.sim_buyclose_sellopen
bench_0050 = v6.bench_0050

END = "2026-06-08"
DISCLOSURE_LAG = 1   # ETF 持股盤後揭露：成員資格用 D-LAG 才可知(forward-only)


def load_pit_membership():
    """[leak-free 修法#1a/1b] point-in-time ETF 成員 + 揭露落後。
    回傳: (sorted_disclosure_dates, {date: set(codes known as-of that date)})。
    某代號在揭露日 dd 出現 → 從 dd 之後(+LAG)的交易日才可被選。
    """
    path = DATA_DIR / "Active_ETF_1Y_Daily_28ETFs.csv"
    bydate = defaultdict(set)
    for r in csv.DictReader(path.open(encoding="utf-8")):
        bydate[r["Date"].strip()].add(r["Stock_Code"].strip())
    disc_dates = sorted(bydate)
    return disc_dates, bydate


def momentum_score(closes_list, L):
    """與原 aetf_mechanize 完全相同的動能規則(不動演算法，只動洩漏)。"""
    n = len(closes_list)
    if n < 2 * L + 1:
        return 0.0
    c0 = closes_list[-1]; cL = closes_list[-1 - L]
    if cL <= 0 or c0 <= 0:
        return 0.0
    ret_L = c0 / cL - 1.0
    ma_L = sum(closes_list[-L:]) / L
    above = 1.0 if c0 > ma_L else 0.0
    c2L = closes_list[-1 - 2 * L]
    ret_2L = c0 / c2L - 1.0 if c2L > 0 else 0.0
    accel = 1.0 if ret_L > ret_2L / 2 else 0.0
    if ret_L <= 0 or above == 0.0:
        return 0.0
    return ret_L * (1.0 + 0.5 * accel) * 100.0


def build_mom_rows_pit(closes, sig_days, disc_dates, bydate, eligible_on, L, topn):
    """leak-free 動能訊號：每日只在『該日 point-in-time ETF 池』內挑 top-N。"""
    series = {}
    allcodes = set()
    for s in bydate.values():
        allcodes |= s
    for c in allcodes:
        cc = closes.get(c, {})
        ds = sorted(d for d in cc if d <= END)
        series[c] = (ds, [cc[d] for d in ds])
    rows = []
    for d in sig_days:
        pool_d = eligible_on(d)         # 該訊號日可交易的 point-in-time 成員
        sc = []
        for c in pool_d:
            sd = series.get(c)
            if not sd:
                continue
            ds, cl = sd
            j = bisect.bisect_right(ds, d) - 1
            if j < 0 or ds[j] != d:
                continue
            v = momentum_score(cl[:j + 1], L)
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:topn]:
            rows.append((d, c, v / 100.0))
    return rows


def build_h_rows(codes_fn, feats, twii_feat, reb_cache, turn_pct, sig_days, topn):
    """baseline H+反彈雙引擎。codes_fn(d) 回傳該日可用宇宙(支援 point-in-time 同池)。"""
    regime_bull = {
        d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
        for d in sig_days
    }
    rows = []
    for d in sig_days:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
        codes = codes_fn(d)
        sc = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            v = (h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
                 if bull else reb_cache.get(c, {}).get(d, 0.0))
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:topn]:
            rows.append((d, c, v / 100.0))
    return rows


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes_full = list(u.keys())
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes_full}

    disc_dates, bydate = load_pit_membership()
    all_pool = sorted({c for s in bydate.values() for c in s})
    logger.info(f"ETF 揭露日 {len(disc_dates)} 天 {disc_dates[0]}~{disc_dates[-1]}｜全期池 {len(all_pool)} 檔")

    # 載入價格/特徵
    need = sorted(set(codes_full) | set(all_pool) | {"0050"})
    logger.info(f"載入特徵/價格 {len(need)} 支...")
    twii_feat = features("0050")
    feats, opens, closes = {}, {}, {}
    for c in need:
        try:
            o = oh(c)
            if not o:
                continue
            opens[c] = {d: o[d]["open"] for d in o}
            closes[c] = {d: o[d]["close"] for d in o}
            if c in codes_full or c == "0050":
                feats[c] = features(c)
        except Exception as e:
            logger.warning(f"  {c} 失敗 {str(e)[:40]}")

    # ★ leak-free 修法#1c：回測期限定在 ETF 有 point-in-time 揭露的區間
    #    (CSV 起 2025-06-17。更早沒有成員資格 → 不可回測，否則=用未來池選過去)
    PIT_START = disc_dates[0]
    alld = sorted({d for c in closes for d in closes[c] if d <= END})
    sig_days = [d for d in alld if d >= PIT_START]
    logger.info(f"★ leak-free 回測區間限定 {sig_days[0]}~{sig_days[-1]}({len(sig_days)} 交易日；"
                f"更早無 point-in-time 成員，禁止回測)")

    # point-in-time eligibility：訊號日 d 可交易的代號 = 揭露日 < d-LAG 內曾出現過的成員
    #  (用「揭露日 dd 加 LAG 個交易日後才生效」的累積成員；揭露盤後→forward-only)
    cal_idx = {d: i for i, d in enumerate(alld)}
    eligible_cache = {}
    # 對每個揭露日找其在交易日曆的位置，+LAG 後生效
    def disc_effective(dd):
        # dd 是揭露日；生效日 = 交易曆上 dd 之後第 LAG 個交易日
        i = bisect.bisect_left(alld, dd)
        j = i + DISCLOSURE_LAG
        return alld[j] if j < len(alld) else None
    # 預計算每個代號的最早生效日
    code_eff = {}
    for dd in disc_dates:
        eff = disc_effective(dd)
        if eff is None:
            continue
        for c in bydate[dd]:
            if c not in code_eff or eff < code_eff[c]:
                code_eff[c] = eff
    def eligible_on(d):
        if d in eligible_cache:
            return eligible_cache[d]
        s = [c for c, eff in code_eff.items() if eff <= d]
        eligible_cache[d] = s
        return s

    # 反彈訊號 + 漲停 + turn_pct (供 H baseline)
    logger.info("反彈訊號/漲停/成交值百分位...")
    reb_cache, limitup = {}, {}
    for c in need:
        if c not in opens:
            continue
        o = oh(c); ds = sorted(d for d in o if d <= END); cl_list = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s

    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes_full
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    # ── 時序切分 (leak-free 修法#2)：前半 train 選參，後半 test=OOS 評估 ──
    half = len(sig_days) // 2
    train_days = sig_days[:half]
    test_days = sig_days[half:]
    logger.info(f"train {train_days[0]}~{train_days[-1]}({len(train_days)})｜"
                f"OOS test {test_days[0]}~{test_days[-1]}({len(test_days)})")

    def run(rows, day_set):
        rw = [r for r in rows if r[0] in day_set]
        return sim5(rw, opens, closes, limitup, switch_cost_mult=1.0)

    def bench_for(day_list):
        dd = [d for d in day_list if d in closes.get("0050", {})]
        return bench_0050(opens["0050"], closes["0050"], dd)

    # baseline H 同池 (point-in-time 池) — 兩個 split 各自生成
    def hpool_codes(d):
        return [c for c in eligible_on(d) if c in feats]
    h_full_codes = lambda d: codes_full

    # ── 在 TRAIN 上掃 L/N (誠實 in-sample 只用於選參，不報結果) ──
    LOOKBACKS = [10, 20, 40, 60, 120]
    TOPNS = [3, 4, 6]
    train_alpha = {}
    bench_train = bench_for(train_days)
    for L in LOOKBACKS:
        for tn in TOPNS:
            rows = build_mom_rows_pit(closes, train_days, disc_dates, bydate, eligible_on, L, tn)
            r = run(rows, set(train_days))
            train_alpha[(L, tn)] = (r["ret"] - bench_train) if r else None
    best_LN = max((k for k in train_alpha if train_alpha[k] is not None),
                  key=lambda k: train_alpha[k], default=(60, 6))
    logger.info(f"TRAIN 選出最佳 (L,N)={best_LN} (train α={train_alpha[best_LN]:+.0f})")

    # ── 在 OOS test 上評估選定參數 vs baseline ──
    L_sel, N_sel = best_LN
    bench_test = bench_for(test_days)
    bench_full = bench_for(sig_days)

    def eval_block(day_list, bench):
        out = {}
        mom = build_mom_rows_pit(closes, day_list, disc_dates, bydate, eligible_on, L_sel, N_sel)
        hp = build_h_rows(hpool_codes, feats, twii_feat, reb_cache, turn_pct, day_list, topn=4)
        hf = build_h_rows(h_full_codes, feats, twii_feat, reb_cache, turn_pct, day_list, topn=4)
        for label, rows in [(f"動能L{L_sel}N{N_sel}", mom), ("H同池(PIT)", hp), ("H全112", hf)]:
            r = run(rows, set(day_list))
            out[label] = {
                "ret": r["ret"] if r else None,
                "alpha": (r["ret"] - bench) if r else None,
                "turn": r["turn"] if r else None,
                "pos": r.get("avg_pos") if r else None,
                "expo": r.get("avg_expo") if r else None,
            } if r else {"ret": None, "alpha": None, "turn": None, "pos": None, "expo": None}
        return out

    oos = eval_block(test_days, bench_test)
    full = eval_block(sig_days, bench_full)        # 全 PIT 區間 (參考)
    tr   = eval_block(train_days, bench_train)

    mlabel = f"動能L{L_sel}N{N_sel}"
    uplift_oos = (oos[mlabel]["alpha"] - oos["H同池(PIT)"]["alpha"]
                  if oos[mlabel]["alpha"] is not None and oos["H同池(PIT)"]["alpha"] is not None else None)
    uplift_full = (full[mlabel]["alpha"] - full["H同池(PIT)"]["alpha"]
                   if full[mlabel]["alpha"] is not None and full["H同池(PIT)"]["alpha"] is not None else None)

    # ── 報告 ──
    def fmt(x, s="%"):
        return f"{x:+.0f}{s}" if isinstance(x, (int, float)) else "—"

    Lr = [
        "# 方法#1 動能機械化 — 對抗式 LEAK-FREE 重跑\n",
        f"> 選股池=point-in-time ETF 成員(揭露 forward +{DISCLOSURE_LAG} 交易日生效)｜⑤買收賣開｜還原價\n",
        f"> ★回測期嚴格限定 {sig_days[0]}~{sig_days[-1]}({len(sig_days)}日)。"
        "原版開到『2年』回溯到 2024，但 ETF 揭露 2025-06 才起 → 那是拿未來池選過去=純洩漏，本版禁止。\n",
        f"> 參數選擇 OOS：train {train_days[0]}~{train_days[-1]} 選 (L,N)={best_LN}；"
        f"test {test_days[0]}~{test_days[-1]} 評估(不在 test 上調參)。\n",
        "> 基準與成本不作弊：alpha=策略−同資金DCA0050、漲停買不到、報曝險。\n",
        "",
        f"## 結論數字\n",
        f"- TRAIN 選出 (L,N) = **{best_LN}**\n",
        f"- **OOS(後半段) alpha**：動能 {fmt(oos[mlabel]['alpha'])}｜H同池(PIT) {fmt(oos['H同池(PIT)']['alpha'])}"
        f"｜H全112 {fmt(oos['H全112']['alpha'])}\n",
        f"- **OOS uplift(動能 − H同池PIT) = {fmt(uplift_oos,'pp')}**  ← 這才是 leak-free 的真實 uplift\n",
        f"- 全 PIT 區間 uplift(參考,含 train) = {fmt(uplift_full,'pp')}\n",
        f"- 0050 基準 DCA：OOS {fmt(bench_test)}｜全PIT {fmt(bench_full)}｜train {fmt(bench_train)}\n",
        "",
        "## 各 split 明細\n",
        "| split | 變體 | 報酬 | alpha | 換手 | 持股 | 曝險 |",
        "|---|---|---|---|---|---|---|",
    ]
    for split_name, block in [("TRAIN", tr), ("OOS", oos), ("全PIT", full)]:
        for label in [mlabel, "H同池(PIT)", "H全112"]:
            b = block[label]
            Lr.append(f"| {split_name} | {label} | {fmt(b['ret'])} | {fmt(b['alpha'])} | "
                      f"{fmt(b['turn'],'x') if b['turn'] is not None else '—'} | "
                      f"{b['pos']:.1f} | {b['expo']*100:.0f}% |" if b['ret'] is not None
                      else f"| {split_name} | {label} | — | — | — | — | — |")

    Lr += [
        "",
        "## TRAIN 上 L/N 掃描 (僅供選參,非結果)\n",
        "| L\\N | " + " | ".join(str(n) for n in TOPNS) + " |",
        "|---|" + "|".join(["---"]*len(TOPNS)) + "|",
    ]
    for L in LOOKBACKS:
        Lr.append(f"| {L} | " + " | ".join(fmt(train_alpha[(L, n)]) for n in TOPNS) + " |")

    Lr += [
        "",
        "## 原版作弊 vs 本版修法\n",
        "| 作弊 | 原版 | leak-free 修法 |",
        "|---|---|---|",
        "| 選股池 | 全期28ETF聯集50檔(未來成員) | point-in-time 揭露成員(forward+1日) |",
        "| 回測期 | 開到2年(2024,池定義前) | 限定 2025-06~ (有揭露才回測) |",
        "| 調參 | L/N同段掃最佳(in-sample) | train選參/OOS評估 |",
        "| 基準成本 | 不作弊 | 不作弊(沿用) |",
        "",
        "## 判讀\n",
        "- 原版聲稱 1年/2年 uplift +307/+251pp，主要來自『把 2025-26 ETF 持有的大型贏家池套到 2024 價格』的純未來選股。",
        "- 本版把回測期砍到只有 point-in-time 成員存在的區間後，長窗洩漏被切除；只剩 OOS uplift 為真實值。",
    ]

    RPT = ROOT / "reports" / "aetf_mechanize_leakfree.md"
    RPT.write_text("\n".join(Lr), encoding="utf-8")
    logger.success(f"報告 → {RPT}")

    summary = {
        "best_LN_from_train": list(best_LN),
        "oos": {k: oos[k] for k in oos},
        "full_pit": {k: full[k] for k in full},
        "train": {k: tr[k] for k in tr},
        "bench": {"oos": bench_test, "full_pit": bench_full, "train": bench_train},
        "uplift_oos_pp": uplift_oos,
        "uplift_fullpit_pp": uplift_full,
        "pit_range": [sig_days[0], sig_days[-1], len(sig_days)],
    }
    (ROOT / "reports" / "aetf_mechanize_leakfree_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"OOS uplift = {uplift_oos:+.1f}pp ; 全PIT uplift = {uplift_full:+.1f}pp"
                   if uplift_oos is not None else "OOS uplift = None")


if __name__ == "__main__":
    main()
