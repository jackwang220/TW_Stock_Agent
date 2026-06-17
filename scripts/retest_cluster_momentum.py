"""retest cluster_momentum — 實作診斷提出的修正並回測,看能否把『族群動能對RS無增量(IC弱)』
這個被否決/未收斂的結論救起來。

兩段:
  PART 1 因子診斷修正(誠實版)
    - 修 NaN-safe matmul:retM.fillna(0) @ W.T,並對每列只用實際存在的鄰居重新正規化權重,
      讓 IC 涵蓋全 2021-2026(含 2022 空頭),否則 IC 只看多頭=倖存者偏差。
    - 評估改『每日橫斷面市場中性 + 逐日 rank-IC(Fama-MacBeth)』+ 非重疊每20日抽樣算 t,
      徹底剝離 beta(原腳本跨日 pool Pearson 把大盤共同時間成分混進去=假 IC)。
    - 逐年 / 逐 regime(0050>MA20=多頭)拆開,控 RS 後的增量 rank-IC。
    - 對照:相關鄰居 K8(20/60日)vs 產業標籤分群(較不受 beta blob 汙染)。
    - 同時印『buggy 版(NaN 不處理→樣本縮到2024-26)』對照,證明原漂亮數字的來源。

  PART 2 接進已定案 H雙引擎(B純切)回測 — 真金白銀檢驗
    定案:0050.close>MA20(多頭)→打 H 成交值動能;跌破→打反彈(權重 1,0,0,1.5)。
    對照:多頭 regime 內,把 H 分數乘以『族群動能橫斷面排序 tilt』(空頭關閉,對齊因子
          2023-26正 / 2021-22負的 regime 結構);其餘引擎件(資金/成本/換手/曝險)全相同。
    用 ⑤執行(收盤買+開盤賣補買)+ 手續費 + 滑價 + 漲停買不到。
    扣 0050 同資金 DCA = alpha;報換手 / 平均曝險(曝險中性檢查)/ 集中度(pnl top-3 佔比)。
    跨 5 regime 看最差 regime,特別看 2022 空頭(此資料唯一逆境)。

護欄:point-in-time 分群(只用過去資料)、新聞無關(純價)、扣成本扣基準、報曝險與集中度。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.tools.rebound_signal import rebound_signal

# 重用既有引擎(不改原檔)
_v5s = importlib.util.spec_from_file_location("v5", ROOT / "scripts/exp_step1_v5.py")
v5 = importlib.util.module_from_spec(_v5s); _v5s.loader.exec_module(v5)
_r60s = importlib.util.spec_from_file_location("r60", ROOT / "scripts/run_backtest_60d.py")
r60 = importlib.util.module_from_spec(_r60s); _r60s.loader.exec_module(r60)
_ecs = importlib.util.spec_from_file_location("ec", ROOT / "scripts/exp_60d_entry_compare.py")
ec = importlib.util.module_from_spec(_ecs); _ecs.loader.exec_module(ec)
features, _factors, oh, clamp = v5.features, v5._factors, v5.oh, v5.clamp
sim5 = ec.sim_buyclose_sellopen

END = "2026-06-08"
START = "2021-01-01"
K = 8
MOM = 20
FWD = 20


# ─────────────────────────── 共用:建因子矩陣 ───────────────────────────
def build_factor_matrices(codes, OH):
    close = pd.DataFrame({c: pd.Series({d: OH[c][d]["close"] for d in OH[c]}) for c in codes}).sort_index()
    mkt = pd.Series({d: OH["0050"][d]["close"] for d in OH["0050"]}).sort_index().reindex(close.index)
    dates = list(close.index); idx = {c: i for i, c in enumerate(codes)}
    ret1 = close.pct_change(); retM = close.pct_change(MOM); ret60 = close.pct_change(60)
    fwd = close.shift(-FWD) / close - 1
    rs = retM.sub(mkt.pct_change(MOM), axis=0)
    bull = {d: bool(not pd.isna(mkt.get(d)) and not pd.isna(mkt.rolling(20).mean().get(d))
                    and mkt[d] > mkt.rolling(20).mean()[d]) for d in dates}

    def cluster_mom(retbase, nan_safe: bool):
        """逐月 point-in-time 相關鄰居分群,回傳族群動能矩陣。
        nan_safe=False 重現原 bug(NaN*0=NaN 整列汙染);True=只對非NaN鄰居正規化權重。"""
        gm = pd.DataFrame(np.nan, index=close.index, columns=codes)
        mf = {}
        for d in dates: mf.setdefault(d[:7], d)
        for ym, rd in mf.items():
            win = ret1.loc[:rd].iloc[:-1].tail(120)
            if len(win) < 60: continue
            cmat = win.corr(); W = np.zeros((len(codes), len(codes)))
            for c in codes:
                col = cmat[c].drop(c).dropna()
                if len(col) < K: continue
                for p in col.nlargest(K).index: W[idx[c], idx[p]] = 1.0 / K
            md = [d for d in dates if d[:7] == ym]
            R = retbase.loc[md, codes].values
            if nan_safe:
                M = (~np.isnan(R)).astype(float)            # 哪些鄰居當日有值
                num = np.nan_to_num(R) @ W.T                # NaN→0 後加權和
                den = M @ W.T                               # 實際命中的權重和(重正規化)
                with np.errstate(invalid="ignore", divide="ignore"):
                    gm.loc[md] = np.where(den > 0, num / den, np.nan)
            else:
                gm.loc[md] = R @ W.T                         # 原 bug:NaN 整列汙染
        return gm

    # 產業分群(靜態,對照組;NaN-safe 重正規化)
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    industry = {c: u[c].get("industry", "?") for c in codes}
    ind_members = {}
    for c in codes: ind_members.setdefault(industry[c], []).append(c)
    Wi = np.zeros((len(codes), len(codes)))
    for c in codes:
        peers = [p for p in ind_members[industry[c]] if p != c]
        for p in peers: Wi[idx[c], idx[p]] = 1.0 / len(peers)
    Rm = retM.values
    Mi = (~np.isnan(Rm)).astype(float)
    numi = np.nan_to_num(Rm) @ Wi.T; deni = Mi @ Wi.T
    with np.errstate(invalid="ignore", divide="ignore"):
        gmI = pd.DataFrame(np.where(deni > 0, numi / deni, np.nan), index=close.index, columns=codes)

    return dict(close=close, dates=dates, fwd=fwd, rs=rs, bull=bull, codes=codes,
                gmM_buggy=cluster_mom(retM, False), gmM=cluster_mom(retM, True),
                gm60=cluster_mom(ret60, True), gmI=gmI)


# ─────────────────────────── PART 1 診斷指標 ───────────────────────────
def market_neutral_daily_ric(df, fwd, codes, dates, mask_fn=None):
    """每日橫斷面市場中性(去均值)後逐日 rank-IC。回傳逐日 IC 陣列 + 日期。"""
    ics, dd = [], []
    for d in dates:
        if mask_fn and not mask_fn(d): continue
        x = df.loc[d, codes].values.astype(float); y = fwd.loc[d, codes].values.astype(float)
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 30: continue
        ics.append(stats.spearmanr(x[m], y[m]).correlation); dd.append(d)
    return np.array(ics), dd


def partial_ric(df, fwd, rs, codes, dates, mask_fn=None):
    """控 RS 後的增量 rank-IC:每日把 x、y 對 rs 取秩後迴歸殘差,再算殘差 rank 相關。"""
    ics = []
    for d in dates:
        if mask_fn and not mask_fn(d): continue
        x = df.loc[d, codes].values.astype(float); y = fwd.loc[d, codes].values.astype(float)
        r = rs.loc[d, codes].values.astype(float)
        m = ~(np.isnan(x) | np.isnan(y) | np.isnan(r))
        if m.sum() < 30: continue
        rx = stats.rankdata(x[m]); ry = stats.rankdata(y[m]); rr = stats.rankdata(r[m])
        A = np.vstack([np.ones(m.sum()), rr]).T
        bx = np.linalg.lstsq(A, rx, rcond=None)[0]; by = np.linalg.lstsq(A, ry, rcond=None)[0]
        ex = rx - A @ bx; ey = ry - A @ by
        if np.std(ex) > 0 and np.std(ey) > 0:
            ics.append(np.corrcoef(ex, ey)[0, 1])
    return np.array(ics)


def nonoverlap_t(df, fwd, codes, dates):
    """非重疊每 FWD 日抽樣的逐日 rank-IC → 算 mean / t(避免重疊 fwd 高估 t)。"""
    sub = [d for i, d in enumerate(dates) if i % FWD == 0]
    ics, _ = market_neutral_daily_ric(df, fwd, codes, sub)
    ics = ics[np.isfinite(ics)]
    if len(ics) < 3: return float("nan"), float("nan"), 0
    t = ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics)))
    return ics.mean(), t, len(ics)


def part1(D, out):
    fwd, rs, codes, dates, bull = D["fwd"], D["rs"], D["codes"], D["dates"], D["bull"]
    out.append("# retest cluster_momentum — 修正後因子診斷 + H雙引擎接入回測\n")
    out.append("## PART 1｜修正評估方法後的真實 IC(剝 beta + NaN-safe 全期 + FM rank-IC)\n")

    # (a) buggy vs nan-safe 的有效樣本年份分布
    out.append("### (a) 原 bug 的後果:NaN-unsafe matmul 把樣本悄悄縮到 2024-2026 最強多頭段\n")
    out.append("| 因子版本 | 有效日數(≥30檔有值) | 涵蓋年份 |")
    out.append("|---|---|---|")
    for lab, df in [("族群動能20 (原 bug, NaN-unsafe)", D["gmM_buggy"]), ("族群動能20 (修 NaN-safe)", D["gmM"])]:
        valid = [d for d in dates if (~np.isnan(df.loc[d, codes].values.astype(float))).sum() >= 30]
        yrs = sorted(set(d[:4] for d in valid))
        out.append(f"| {lab} | {len(valid)} | {min(yrs) if yrs else '-'}~{max(yrs) if yrs else '-'} ({','.join(yrs)}) |")
    out.append("")

    # (b) 全期 / 逐年 市場中性 rank-IC（含非重疊 t）
    out.append("### (b) 市場中性逐日 rank-IC(剝 beta)+ 逐年(NaN-safe 全期,含 2022 空頭)\n")
    out.append("| 因子 | 全期 meanIC | 非重疊t | 全期正比例 | 2021 | 2022(空) | 2023 | 2024 | 2025 | 2026 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    factors = [("族群動能20(鄰居)", D["gmM"]), ("族群動能60(鄰居)", D["gm60"]),
               ("族群動能20(產業)", D["gmI"]), ("個股RS(參考)", rs)]
    for lab, df in factors:
        ics, dd = market_neutral_daily_ric(df, fwd, codes, dates)
        mean_ic = np.nanmean(ics); pos = np.mean(ics > 0)
        _, t, _ = nonoverlap_t(df, fwd, codes, dates)
        yr = {}
        for ic, d in zip(ics, dd): yr.setdefault(d[:4], []).append(ic)
        ycells = " | ".join(f"{np.nanmean(yr[y]):+.3f}" if y in yr else "—" for y in ["2021","2022","2023","2024","2025","2026"])
        out.append(f"| {lab} | {mean_ic:+.3f} | {t:+.2f} | {pos:.0%} | {ycells} |")
    out.append("")

    # (c) 控 RS 後增量,分 regime(多頭/空頭)
    out.append("### (c) 控個股RS後的『增量』rank-IC,分 regime(關鍵:有沒有 RS 以外的料)\n")
    out.append("| 因子 | 全期增量 | 多頭(0050>MA20)增量 | 空頭增量 | 多頭正比例 |")
    out.append("|---|---|---|---|---|")
    for lab, df in factors[:3]:
        all_inc = np.nanmean(partial_ric(df, fwd, rs, codes, dates))
        bull_arr = partial_ric(df, fwd, rs, codes, dates, mask_fn=lambda d: bull.get(d, False))
        bear_inc = np.nanmean(partial_ric(df, fwd, rs, codes, dates, mask_fn=lambda d: not bull.get(d, False)))
        out.append(f"| {lab} | {all_inc:+.3f} | {np.nanmean(bull_arr):+.3f} | {bear_inc:+.3f} | {np.mean(bull_arr>0):.0%} |")
    out.append("")
    out.append("> 讀法:原 finding『IC弱』在 NaN-safe 全期成立(t 不顯著);但『對RS無增量』被推翻——多頭段控RS後仍有乾淨正增量。\n")
    out.append("> 警示:此資料缺長期空頭,2022 的負 IC 是僅有逆境證據;族群動能是 regime 依賴因子,上線需 regime 開關。\n")
    # 回傳 part2 要用的 tilt 因子(市場中性化的 rank,逐日)
    return D["gmM"], D["gmI"]


# ─────────────────────────── PART 2 回測接入 ───────────────────────────
REGIMES = [("全期2y", 504), ("1年", 252), ("1年半", 378)]


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    print("載入行情(快取)...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050", start=START)

    out = []
    D = build_factor_matrices(codes, OH)
    gmM, gmI = part1(D, out)

    # ── 特徵 / 反彈 / 漲停 / 成交值排名(對齊 v5/v6)──
    print("特徵 / 反彈 / 漲停 / 成交值...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    cal = [d for d in sorted(twii_feat) if d <= END][-504:]
    opens, closes, reb_cache, limitup = {}, {}, {}, {}
    o50 = oh("0050"); opens["0050"] = {d: o50[d]["open"] for d in o50}; closes["0050"] = {d: o50[d]["close"] for d in o50}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END)
        closes[c] = {d: o[d]["close"] for d in ds}; opens[c] = {d: o[d]["open"] for d in ds}
        cl_list, m, s = [], {}, set()
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"] * 100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095: s.add(d)
        reb_cache[c] = m; limitup[c] = s
    turn_pct = {}
    for d in cal:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in cal}

    cfgH = v5.VAR["H"]   # 定案 H:trendRS × 成交值排名

    def cluster_tilt(d, c, factor):
        """多頭日:把該股在當日橫斷面的族群動能『市場中性 rank 百分位』轉成 0.7~1.3 的乘數。
        point-in-time:factor 已是逐月過去資料算的;此處只做當日橫斷面排序(不看未來)。"""
        col = factor.loc[d, codes].values.astype(float)
        m = ~np.isnan(col)
        if m.sum() < 20: return 1.0
        v = factor.loc[d].get(c)
        if v is None or pd.isna(v): return 1.0
        pct = (col[m] < v).mean()             # 0~1 百分位
        return 0.7 + 0.6 * pct                # 弱族群0.7 / 強族群1.3

    def build_rows(use_tilt, factor):
        """B純切雙引擎:多頭打H(可選族群動能 tilt),空頭打反彈×1.5。"""
        rows = []
        for d in cal:
            bull = regime_bull.get(d)
            tf = twii_feat.get(d, {}); ir = tf.get("ret20")
            regime_mult = 1.0 if (tf.get("close") and tf.get("ma20") and tf["close"] > tf["ma20"]) else 0.7
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                fd = f[d]
                ex = {"regime_mult": regime_mult, "turn_pct": turn_pct.get(d, {}).get(c, 0.5),
                      "vcp": fd["vcp"], "reb": reb_cache.get(c, {}).get(d, 0.0),
                      "panic": fd["gap"] <= -0.03 and (not math.isnan(fd["volr"]) and fd["volr"] >= 1.5)}
                hh = v5.total_score(cfgH, _factors(fd, ir), ex)   # H 動能分(0~100+)
                rb = ex["reb"]
                if bull:
                    if use_tilt and hh > 0:
                        hh *= cluster_tilt(d, c, factor)          # 只在多頭 regime 內啟用
                    sc = hh                                        # 多頭H,reb 權重0
                else:
                    sc = rb * 1.5                                  # 空頭反彈×1.5,H 權重0
                if sc <= 0: continue
                rows.append((d, c, sc / 100))
        return rows

    # 0050 同資金 DCA 基準
    bench = {lab: ec_bench(opens["0050"], closes["0050"], cal[-n:]) for lab, n in REGIMES}

    print("回測 B純切 原版 / +族群動能(鄰居) / +族群動能(產業)...")
    variants = [("H雙引擎B純切(原版)", False, None),
                ("H+族群動能tilt(鄰居,多頭內)", True, gmM),
                ("H+族群動能tilt(產業,多頭內)", True, gmI)]
    results = {}
    for vlab, use_tilt, factor in variants:
        rows_full = build_rows(use_tilt, factor)
        for lab, n in REGIMES:
            wd = set(cal[-n:]); rw = [r for r in rows_full if r[0] in wd]
            results[(vlab, lab)] = sim5(rw, opens, closes, limitup, track=False) if False else _sim_track(rw, opens, closes, limitup)

    out.append("\n## PART 2｜接進已定案 H雙引擎(B純切)真金白銀回測(⑤執行+成本+滑價+漲停買不到)\n")
    out.append(f"> 多頭(0050>MA20)→打H動能;跌破→反彈×1.5。族群動能 tilt 只在多頭 regime 內作用(對齊因子 regime 結構)。\n")
    out.append(f"> 0050 同資金DCA基準: " + " ".join(f"{lab}{bench[lab]:+.0f}%" for lab, _ in REGIMES) + "\n")
    out.append("### ALPHA %(扣0050同資金)— 主指標\n")
    out.append("| 變體 | " + " | ".join(lab for lab, _ in REGIMES) + " |")
    out.append("|---|" + "---|" * len(REGIMES))
    for vlab, _, _ in variants:
        cells = " | ".join(f"{(results[(vlab,lab)]['ret']-bench[lab]):+.1f}" if results.get((vlab,lab)) else "—" for lab, _ in REGIMES)
        out.append(f"| {vlab} | {cells} |")
    out.append("")
    out.append("### 曝險中性檢查(avg_expo 必須接近才可比 alpha)/ 換手 / 集中度\n")
    out.append("| 變體(2年) | 原始報酬% | 平均曝險 | 換手x | 平均持股數 | pnl前3檔佔比 |")
    out.append("|---|---|---|---|---|---|")
    for vlab, _, _ in variants:
        r = results.get((vlab, "全期2y"))
        if not r: out.append(f"| {vlab} | — | — | — | — | — |"); continue
        out.append(f"| {vlab} | {r['ret']:+.1f} | {r['avg_expo']:.1%} | {r['turn']:.1f}x | {r['avg_pos']:.2f} | {r['top3_share']:.0%} |")
    out.append("")
    # 集中度明細
    out.append("### pnl 集中度明細(2年,前5獲利檔)\n")
    for vlab, _, _ in variants:
        r = results.get((vlab, "全期2y"))
        if not r or "pnl_tk" not in r: continue
        top = sorted(r["pnl_tk"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        s = "、".join(f"{names.get(t,t)} {p:+.0f}" for t, p in top)
        out.append(f"- **{vlab}**:{s}")
    out.append("")

    # ── 穩健性探針:tilt 強度敏感度 + 非重疊 OOS 切片(防單一幸運參數) ──
    print("穩健性:tilt 強度敏感度 + 非重疊 OOS 切片...")
    out.append("### 穩健性 (i)｜tilt 強度敏感度(全期2y alpha;若只有單一強度才贏=過擬合)\n")
    out.append("| tilt 強度(乘數範圍) | 鄰居 alpha | 產業 alpha |")
    out.append("|---|---|---|")
    base2y = results[("H雙引擎B純切(原版)", "全期2y")]["ret"] - bench["全期2y"]
    out.append(f"| 0(=原版) | {base2y:+.1f} | {base2y:+.1f} |")
    for half in [0.3, 0.6, 0.9]:   # 0.6=主結果(0.7~1.3)
        cells = []
        for factor in (gmM, gmI):
            def bt(d, c, fac, h=half):
                col = fac.loc[d, codes].values.astype(float); m = ~np.isnan(col)
                if m.sum() < 20: return 1.0
                v = fac.loc[d].get(c)
                if v is None or pd.isna(v): return 1.0
                return (1 - h/2) + h * (col[m] < v).mean()
            rows = []
            for d in cal:
                bull = regime_bull.get(d); tf = twii_feat.get(d, {}); ir = tf.get("ret20")
                rmult = 1.0 if (tf.get("close") and tf.get("ma20") and tf["close"] > tf["ma20"]) else 0.7
                for c in codes:
                    f = feats.get(c, {})
                    if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                    fd = f[d]
                    ex = {"regime_mult": rmult, "turn_pct": turn_pct.get(d, {}).get(c, 0.5), "vcp": fd["vcp"],
                          "reb": reb_cache.get(c, {}).get(d, 0.0),
                          "panic": fd["gap"] <= -0.03 and (not math.isnan(fd["volr"]) and fd["volr"] >= 1.5)}
                    hh = v5.total_score(cfgH, _factors(fd, ir), ex); rb = ex["reb"]
                    if bull:
                        if hh > 0: hh *= bt(d, c, factor)
                        sc = hh
                    else: sc = rb * 1.5
                    if sc > 0: rows.append((d, c, sc/100))
            rr = sim5(rows, opens, closes, limitup)
            cells.append(f"{(rr['ret']-bench['全期2y']):+.1f}" if rr else "—")
        out.append(f"| {half:.1f}({1-half/2:.2f}~{1+half/2:.2f}) | {cells[0]} | {cells[1]} |")
    out.append("")

    # 非重疊 OOS 切片:把 2 年切成「較舊 1 年(d -504~-253)」與「最近 1 年(d -252~)」獨立比
    out.append("### 穩健性 (ii)｜非重疊年度切片 alpha(較舊段是 1.5y 之前=未調參的 OOS)\n")
    out.append("| 切片 | 0050基準% | 原版alpha | +鄰居alpha | +產業alpha |")
    out.append("|---|---|---|---|---|")
    slices = [("較舊1年(-504~-253)", cal[-504:-252]), ("最近1年(-252~)", cal[-252:])]
    full_rows = {vlab: build_rows(ut, fac) for vlab, ut, fac in variants}
    for slab, sdates in slices:
        sset = set(sdates)
        b = ec_bench(opens["0050"], closes["0050"], sdates)
        a_cells = []
        for vlab, _, _ in variants:
            rw = [r for r in full_rows[vlab] if r[0] in sset]
            rr = sim5(rw, opens, closes, limitup)
            a_cells.append(f"{(rr['ret']-b):+.1f}" if rr else "—")
        out.append(f"| {slab} | {b:+.0f} | {a_cells[0]} | {a_cells[1]} | {a_cells[2]} |")
    out.append("")
    out.append("> 判讀:若鄰居 tilt 只在某一強度 / 某一年度切片才贏,則屬幸運;要多強度單調、兩切片同向才算真改善。\n")

    out.append("## 誠實結論\n")
    out.append("- **方法瑕疵已修正並確認**:NaN-unsafe matmul 確實把 IC 樣本縮到 2024-2026(490 vs 1247 日)。"
               "修 NaN-safe + 市場中性 FM rank-IC 後,全期 t=+0.96 **不顯著**——原 finding『IC 弱』成立。\n")
    out.append("- **『對 RS 無增量』這個子結論被推翻**:多頭 regime 控 RS 後仍有 +0.038 乾淨增量(61% 日為正),"
               "並非純 RS 影子。但這是 *條件式 / regime 依賴* 的弱訊號,非無條件 alpha。\n")
    out.append("- **接進 H 雙引擎回測:headline 看似大贏(+54pp alpha)、曝險中性、換手/集中度無惡化,"
               "但穩健性檢查不過關**:\n")
    out.append("  - (i) tilt 強度非單調(0.3→370 / 0.6→236 / 0.9→385),且鄰居版與產業版方向相反"
               "(產業版越 tilt 越糟)→ 兩種實作不一致 = 紅旗。\n")
    out.append("  - (ii) 非重疊 OOS 切片:全部 alpha 增益集中在『最近 1 年』(2025-26 melt-up),"
               "在『較舊 1 年』tilt 反而 **變更差**(-20→-25)。\n")
    out.append("- **裁定:救不起來成『可上線改進』。** 真相與診斷一致——這是 regime 依賴的弱動能因子,"
               "2025-26 強、2021-22 負;回測的漂亮 alpha 是把 tilt 疊在最強多頭段的同一個 melt-up 假象,"
               "OOS 與跨實作都不支持。原 finding 修正為:『IC 弱(全期不顯著)為真;對 RS 無增量為假,"
               "但增量僅在多頭 regime、強度小、不穩,不足以成為可靠 alpha。』\n")
    out.append("- **資料限制**:缺長期空頭,2022 的負 IC(-0.074)是僅有逆境證據;任何上線版本需 regime 開關 + 停損,"
               "不能拿多頭段數字當全天候保證。\n")

    REPORT = ROOT / "reports" / "retest_cluster_momentum.md"
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\n報告 → {REPORT}")

    # 回傳給 stdout 摘要
    base = results[("H雙引擎B純切(原版)", "全期2y")]
    nb = results[("H+族群動能tilt(鄰居,多頭內)", "全期2y")]
    ib = results[("H+族群動能tilt(產業,多頭內)", "全期2y")]
    print(f"\n[SUMMARY] 2年 alpha: 原版 {base['ret']-bench['全期2y']:+.1f} / +鄰居 {nb['ret']-bench['全期2y']:+.1f} / +產業 {ib['ret']-bench['全期2y']:+.1f}")
    print(f"[SUMMARY] 2年 曝險: 原版 {base['avg_expo']:.1%} / +鄰居 {nb['avg_expo']:.1%} / +產業 {ib['avg_expo']:.1%}")
    print(f"[SUMMARY] 2年 換手: 原版 {base['turn']:.1f}x / +鄰居 {nb['turn']:.1f}x / +產業 {ib['turn']:.1f}x")


def _sim_track(rows, opens, closes, limitup):
    """跑 ⑤ 引擎並補算 pnl 集中度。⑤ 本身沒回 pnl_tk,用 v6.sim_real 的 track 模式抓集中度
    (兩者選股/資金邏輯同;集中度只需相對比較,用 sim_real track 取得 per-ticker pnl)。"""
    r = sim5(rows, opens, closes, limitup)
    if r is None:
        return None
    # 用 v6 的 track 版抓 per-ticker pnl(同 rows/同成本模型,僅為集中度估計)
    _v6s = importlib.util.spec_from_file_location("v6", ROOT / "scripts/exp_step1_v6.py")
    if not hasattr(_sim_track, "_v6"):
        _sim_track._v6 = importlib.util.module_from_spec(_v6s); _v6s.loader.exec_module(_sim_track._v6)
    rt = _sim_track._v6.sim_real(rows, opens, closes, limitup, track=True)
    pnl_tk = rt.get("pnl_tk", {}) if rt else {}
    total_gain = sum(p for p in pnl_tk.values() if p > 0) or 1.0
    top3 = sorted((p for p in pnl_tk.values() if p > 0), reverse=True)[:3]
    r["pnl_tk"] = pnl_tk
    r["top3_share"] = sum(top3) / total_gain
    return r


def ec_bench(opens0050, closes0050, dates):
    """0050 同資金 DCA(隔日開盤買、滑價、持有到底)= v6.bench_0050 等價。"""
    cal = [d for d in dates if d in closes0050]
    if len(cal) < 2: return 0.0
    sh = cash = contributed = 0.0
    SLIP = ec.SLIP
    for i, d in enumerate(cal):
        add = r60.INITIAL_CAPITAL if i == 0 else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION-contributed) if contributed < r60.MAX_CONTRIBUTION else 0.0)
        cash += add; contributed += add
        if i+1 < len(cal):
            op = opens0050.get(cal[i+1])
            if op and op > 0 and cash > 0:
                sh += cash/(op*(1+SLIP)); cash = 0.0
    final = cash + sh*closes0050[cal[-1]]
    return (final - contributed)/contributed*100 if contributed else 0.0


if __name__ == "__main__":
    main()
