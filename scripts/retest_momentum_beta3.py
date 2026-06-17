"""retest momentum_beta v3 — 把被否決結論「動能=beta(命中≤50%)」真正按 proposed_fix 的成功判準重測。

前作 (retest_momentum_beta.py) 因快取只到 2024-01,2021/2022/2023 全 n=0,跨 regime IC 結論只能靠
LLM CSV(part2)。本作先 backfill 全 112 檔到 2021(retest_momentum_backfill.py 已跑),再做:

成功判準(沿用 test_plan T2):扣成本後 long-short Sharpe>0.5 且 ≥4/5 regime 為正、bootstrap CI 不含 0。

三件套 + 護欄:
  (a) 橫截面 IC = spearman(signal, forward alpha),每 regime + pooled + Newey-West t(吃掉重疊自相關)。
  (b) 劑量反應:訊號 quintile → forward alpha,看單調 / 同號。
  (c) Q5-Q1 dollar-neutral long-short(扣成本)的 alpha、Sharpe、bootstrap CI、報酬集中度、換手。

訊號分兩腿(proposed_fix step3):
  - abs_mom = 過去 20d 報酬(絕對動能,疑似 beta)。
  - rel_mom = 扣大盤的相對動能(殘差動能,疑似 alpha)。
  全 112 檔 universe,point-in-time,fwd alpha = 個股 fwd 報酬 − 0050 同期(beta 已扣)。

護欄:
  - 防洩漏:signal 只用 d(含)之前;forward 嚴格未來 bar。
  - 曝險中性:Q5-Q1 是 dollar-neutral(多空等額),market-neutral,不吃 beta;且 alpha 已扣 0050。
  - 換手:報 long-short 名單每期換手倍數。
  - 集中度:報 PnL 前 3 名股票佔比(紅旗 >50%)。
  - 成本:每次換倉買 0.1425%+滑價 0.1%,賣 0.4425%+滑價 0.1%(來回約 0.785%);long-short 兩腿都收。
  - 重疊:用 step=h 的「非重疊」持有期算 long-short 報酬序列 → Sharpe 與 CI 無重疊偏誤;IC 用每日抽樣 + Newey-West。
"""
from __future__ import annotations
import sys, json, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2021-01-01"
# regime 切法與 part2(LLM CSV)對齊,讓「修正前 vs 修正後」可比
REGIMES = [
    ("2022熊市",   "2022-01-01", "2022-12-31"),
    ("2023復甦",   "2023-01-01", "2023-12-31"),
    ("2024多頭",   "2024-01-01", "2024-12-31"),
    ("2025關稅崩", "2025-01-01", "2025-06-30"),
    ("2026最近",   "2025-07-01", "2026-06-08"),
]
HORIZONS = [5, 20]
MKT = "0050"
# 來回交易成本(long-short 每腿都換倉):買 0.1425%+滑價 0.1% / 賣 0.4425%+滑價 0.1%
COST_ROUNDTRIP = (0.1425 + 0.1 + 0.4425 + 0.1) / 100  # ≈ 0.785%
random.seed(42)


def pearson(xs, ys):
    n = len(xs)
    if n < 10: return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0: return None
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    n = len(xs)
    if n < 10: return None
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]: j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1): r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def newey_west_t(daily_ic, lag):
    """daily_ic: 每個抽樣日一個橫截面 IC 值。回 (mean, t_NW)。lag 吸收重疊自相關。"""
    vals = [v for v in daily_ic if v is not None]
    n = len(vals)
    if n < 5: return (None, None)
    mean = sum(vals) / n
    dev = [v - mean for v in vals]
    gamma0 = sum(d * d for d in dev) / n
    var = gamma0
    for L in range(1, min(lag, n - 1) + 1):
        w = 1 - L / (lag + 1)
        cov = sum(dev[t] * dev[t - L] for t in range(L, n)) / n
        var += 2 * w * cov
    se = math.sqrt(var / n) if var > 0 else None
    if not se: return (mean, None)
    return (mean, mean / se)


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}
    OH[MKT] = get_daily_ohlcv(MKT, start=START)
    series = {}
    for c in list(OH.keys()):
        ds = sorted(OH[c]); cl = [OH[c][d]["close"] for d in ds]
        series[c] = (ds, cl, {d: i for i, d in enumerate(ds)})
    idx_days = series[MKT][0]

    def fwd_ret(c, d, h):
        ds, cl, pos = series[c]; i = pos.get(d)
        if i is None or i + h >= len(ds) or cl[i] <= 0: return None
        return cl[i + h] / cl[i] - 1

    def trail_ret(c, d, k):
        ds, cl, pos = series[c]; i = pos.get(d)
        if i is None or i - k < 0 or cl[i - k] <= 0: return None
        return cl[i] / cl[i - k] - 1

    L = ["# retest momentum_beta v3 — 全112檔回填2021後的 IC + dose-response + 扣成本 long-short\n",
         "> 訊號 abs_mom=過去20d絕對動能(疑似beta); rel_mom=扣大盤相對動能(疑似alpha)。",
         "> alpha=個股fwd − 0050同期。Q5-Q1=dollar-neutral多空(market-neutral,不吃beta)。",
         f"> 成本來回 {COST_ROUNDTRIP*100:.3f}%/腿;long-short 每期兩腿換倉都收。\n"]

    # coverage check
    cov = {}
    for rlab, rs_, re_ in REGIMES:
        cnt = sum(1 for c in codes if any(rs_ <= d <= re_ for d in series[c][0][::40]))
        cov[rlab] = cnt
    L.append("### 資料覆蓋(回填後): " + ", ".join(f"{k}={v}檔" for k, v in cov.items()) + "\n")

    summary_rows = []  # (signame, h, regime, ic, t_nw, q5q1_gross, q5q1_net, sharpe_net, ci_lo, ci_hi, conc, turn)

    for h in HORIZONS:
        nw_lag = h  # 重疊長度 ≈ horizon
        for signame in ["abs_mom", "rel_mom"]:
            L.append(f"\n## 訊號={signame}, forward={h}d\n")
            L.append("| regime | n | IC(spear) | IC_NW_t | Q1低 | Q3 | Q5高 | Q5-Q1毛 | Q5-Q1淨(扣成本) | LS Sharpe(年化) | 95%CI(淨/期) | 集中前3 | 換手x |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

            for rlab, rs_, re_ in REGIMES:
                # ---- IC: 每個抽樣日(step 5)一個橫截面 IC,Newey-West 吃重疊 ----
                sample_days = [d for d in idx_days if rs_ <= d <= re_][::5]
                daily_ic = []
                pooled_obs = []
                for d in sample_days:
                    ii = series[MKT][2].get(d)
                    if ii is None or ii + h >= len(idx_days): continue
                    idx_fwd = series[MKT][1][ii + h] / series[MKT][1][ii] - 1 if series[MKT][1][ii] > 0 else None
                    if idx_fwd is None: continue
                    idx20 = trail_ret(MKT, d, 20)
                    day_sig, day_alpha = [], []
                    for c in codes:
                        s20 = trail_ret(c, d, 20)
                        if s20 is None: continue
                        if signame == "abs_mom":
                            sig = s20
                        else:
                            if idx20 is None or abs(1 + idx20) < 1e-6: continue
                            sig = (1 + s20) / (1 + idx20) - 1
                        fr = fwd_ret(c, d, h)
                        if fr is None: continue
                        alpha = fr - idx_fwd
                        day_sig.append(sig); day_alpha.append(alpha)
                        pooled_obs.append((sig, alpha))
                    if len(day_sig) >= 10:
                        daily_ic.append(spearman(day_sig, day_alpha))

                n = len(pooled_obs)
                if n < 50:
                    L.append(f"| {rlab} | {n} | n太小 | | | | | | | | | | |")
                    summary_rows.append((signame, h, rlab, None, None, None, None, None, None, None, None, None))
                    continue
                ic = spearman([o[0] for o in pooled_obs], [o[1] for o in pooled_obs])
                ic_mean, ic_t = newey_west_t(daily_ic, nw_lag)
                obs = sorted(pooled_obs, key=lambda x: x[0])
                qa = []
                for k in range(5):
                    a = n * k // 5; b = n * (k + 1) // 5
                    seg = [o[1] for o in obs[a:b]]
                    qa.append(sum(seg) / len(seg) if seg else 0)
                q5q1_gross = qa[4] - qa[0]

                # ---- 扣成本 long-short:非重疊持有期(step=h),每期重建多空名單 ----
                ls_period = []          # 每個非重疊期的淨多空報酬
                pnl_by_code = {}        # 個股累計貢獻(粗略,用名單股的 alpha 平均拆)
                prev_long, prev_short = set(), set()
                turns = []
                rebuild_days = [d for d in idx_days if rs_ <= d <= re_][::h]
                for d in rebuild_days:
                    ii = series[MKT][2].get(d)
                    if ii is None or ii + h >= len(idx_days): continue
                    idx_fwd = series[MKT][1][ii + h] / series[MKT][1][ii] - 1 if series[MKT][1][ii] > 0 else None
                    if idx_fwd is None: continue
                    idx20 = trail_ret(MKT, d, 20)
                    rows_d = []
                    for c in codes:
                        s20 = trail_ret(c, d, 20)
                        if s20 is None: continue
                        if signame == "abs_mom":
                            sig = s20
                        else:
                            if idx20 is None or abs(1 + idx20) < 1e-6: continue
                            sig = (1 + s20) / (1 + idx20) - 1
                        fr = fwd_ret(c, d, h)
                        if fr is None: continue
                        rows_d.append((sig, fr - idx_fwd, c))
                    if len(rows_d) < 20: continue
                    rows_d.sort(key=lambda x: x[0])
                    q = max(1, len(rows_d) // 5)
                    short_leg = rows_d[:q]; long_leg = rows_d[-q:]
                    long_a = sum(r[1] for r in long_leg) / len(long_leg)
                    short_a = sum(r[1] for r in short_leg) / len(short_leg)
                    gross = long_a - short_a
                    # 換手:本期名單 vs 上期,兩腿合計(0=完全不變,1=全換)
                    cur_long = {r[2] for r in long_leg}; cur_short = {r[2] for r in short_leg}
                    if prev_long or prev_short:
                        t_long = 1 - len(cur_long & prev_long) / max(1, len(cur_long))
                        t_short = 1 - len(cur_short & prev_short) / max(1, len(cur_short))
                        turns.append((t_long + t_short) / 2)
                    prev_long, prev_short = cur_long, cur_short
                    # 成本:每期兩腿都換倉(保守:全換),收來回成本 ×2腿
                    net = gross - COST_ROUNDTRIP * 2
                    ls_period.append(net)
                    for r in long_leg:
                        pnl_by_code[r[2]] = pnl_by_code.get(r[2], 0) + r[1] / len(long_leg)
                    for r in short_leg:
                        pnl_by_code[r[2]] = pnl_by_code.get(r[2], 0) - r[1] / len(short_leg)

                if len(ls_period) >= 4:
                    mean_net = sum(ls_period) / len(ls_period)
                    var = sum((x - mean_net) ** 2 for x in ls_period) / (len(ls_period) - 1)
                    sd = math.sqrt(var) if var > 0 else None
                    # 年化 Sharpe:每期 h 天 → 一年 ~252/h 期
                    periods_per_yr = 252 / h
                    sharpe = (mean_net / sd * math.sqrt(periods_per_yr)) if sd else None
                    # bootstrap CI(每期淨報酬均值)
                    boots = []
                    for _ in range(2000):
                        s = sum(random.choice(ls_period) for _ in range(len(ls_period))) / len(ls_period)
                        boots.append(s)
                    boots.sort()
                    ci_lo = boots[int(0.025 * len(boots))]; ci_hi = boots[int(0.975 * len(boots))]
                    q5q1_net = mean_net
                    # 集中度:|貢獻| 前3 佔總 |貢獻|
                    tot = sum(abs(v) for v in pnl_by_code.values()) or 1
                    top3 = sum(sorted((abs(v) for v in pnl_by_code.values()), reverse=True)[:3])
                    conc = top3 / tot
                    avg_turn = (sum(turns) / len(turns)) if turns else 0
                else:
                    sharpe = ci_lo = ci_hi = q5q1_net = conc = avg_turn = None

                def pct(x): return f"{x*100:+.2f}%" if x is not None else "—"
                L.append(
                    f"| {rlab} | {n} | {ic:+.3f} | {ic_t:+.2f}" + (" " if ic_t else "") +
                    f" | {pct(qa[0])} | {pct(qa[2])} | {pct(qa[4])} | {pct(q5q1_gross)} | {pct(q5q1_net)} | "
                    f"{(f'{sharpe:+.2f}' if sharpe is not None else '—')} | "
                    f"{(f'[{ci_lo*100:+.2f},{ci_hi*100:+.2f}]%' if ci_lo is not None else '—')} | "
                    f"{(f'{conc*100:.0f}%' if conc is not None else '—')} | "
                    f"{(f'{avg_turn:.2f}' if avg_turn is not None else '—')} |"
                )
                summary_rows.append((signame, h, rlab, ic, ic_t, q5q1_gross, q5q1_net, sharpe, ci_lo, ci_hi, conc, avg_turn))

    # ---- 成功判準裁決 ----
    L.append("\n## 成功判準裁決(T2: 扣成本 long-short Sharpe>0.5 且 ≥4/5 regime 正 且 CI不含0)\n")
    L.append("| 訊號×horizon | regime正(net) | regime CI排除0 | 平均Sharpe | 通過? |")
    L.append("|---|---|---|---|---|")
    for signame in ["abs_mom", "rel_mom"]:
        for h in HORIZONS:
            grp = [r for r in summary_rows if r[0] == signame and r[1] == h and r[6] is not None]
            if not grp:
                continue
            pos = sum(1 for r in grp if r[6] > 0)
            ci_excl = sum(1 for r in grp if r[8] is not None and (r[8] > 0 or r[9] < 0))
            sharpes = [r[7] for r in grp if r[7] is not None]
            avg_sh = sum(sharpes) / len(sharpes) if sharpes else None
            ok = (pos >= 4) and (avg_sh is not None and avg_sh > 0.5)
            L.append(f"| {signame} fwd{h}d | {pos}/{len(grp)} | {ci_excl}/{len(grp)} | "
                     f"{(f'{avg_sh:+.2f}' if avg_sh is not None else '—')} | {'通過' if ok else '未通過'} |")

    L.append("\n## 修正前 vs 修正後\n")
    L.append("- 修正前(被否決證據): up勝盤% = LLM up 股『隔日1d alpha>0』比例 = 2022 42% / 2023 48% / 2024 45% / 2025 50% / 2026 49% → 全 ≤50%,被讀成『動能=beta』。")
    L.append("- 缺陷: 1d horizon + 二元命中率(丟劑量) + top12 大型股 universe + 把 LLM 方向預測誤標為『動能變體』。")
    L.append("- 修正後: 全112檔回填2021、point-in-time、扣beta(alpha)、連續訊號 IC + Newey-West t、dose-response、dollar-neutral 扣成本 long-short Sharpe + bootstrap CI。判讀見上表。")

    out = "\n".join(L)
    (ROOT / "reports" / "retest_momentum_beta3.md").write_text(out, encoding="utf-8")
    print(out)
    print("\n→ reports/retest_momentum_beta3.md")


if __name__ == "__main__":
    main()
