"""創意軌:News Co-mention Momentum Spillover tilt(共同提及動能外溢正向選股)。

學理來源(WebSearch):
  Diamond cuts diamond: News co-mention momentum spillover prevails in China
  (J. Banking & Finance 2025) —— 當兩檔股票在同一則新聞「共同被提及」,其中一檔近期的
  報酬動能會『外溢』預測另一檔的未來報酬;此 cross-firm momentum 在中文市場(台股最接近)
  最強,且能 unify 其他形式的 cross-firm momentum。產生顯著正 alpha(long-short ~1.9%/月,t>5)。

為何這軌與既有 7 條失敗軌根本不同:
  既有軌都把「A 自己的新聞量/情緒/LLM 多空分數」當訊號 → 全失敗
  (台股標題滯後、只報已漲完的)。本軌完全不看 A 自己的新聞情緒,而是用
  「A 的新聞『共現夥伴』近期『價格』動能」當訊號 —— 利用標題的『網路結構』而非情緒。
  台股「台積電帶 2315 反彈」這種標題天然就是 spillover 訊號。

探針(scripts/news_creative_comention_probe.py)已確認:
  fresh 共現(近10日)+ 夥伴5日動能 → 逐日橫斷面 rank-IC +0.018,IC-t≈+2.15(顯著),
  top-bottom quintile fwd5 spread +0.44%。比所有「自身新聞」軌都強。

本檔:把該 spillover score 逐日橫斷面 z-score(只取正,負不懲罰,符合「消息面只做正向選股」洞察),
tilt 進 baseline H+反彈雙引擎的選股 edge:edge×(1+w·max(0,spill_z)),掃 w。
baseline=v15 現行實盤(統一×1.5,exit_only,⑤收盤買+開盤只賣)。輸出 v15 多窗格式。

防洩漏:訊號日 d 的 spillover 用 date < d 的共現(不含當日)、夥伴動能用 ≤ d-1 還原價。
(比第一輪寬鬆標準更嚴:共現本身就排除當日新聞;唯一殘留洩漏是 finmind 標題時戳本身的盤後性,
 但因只用『過去日』的共現,當日盤後標題不進訊號 → 本軌幾乎 leak-free。)
"""
from __future__ import annotations
import sys, json, importlib.util, math, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

v5 = _load("v5", "scripts/exp_step1_v5.py")
v6 = _load("v6", "scripts/exp_step1_v6.py")
ec = _load("ec", "scripts/exp_60d_entry_compare.py")
features, _factors = v5.features, v5._factors
sim5 = ec.sim_buyclose_sellopen
bench_0050 = v6.bench_0050

START, END = "2021-01-01", "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252), ("1年半", 378), ("2年", 504)]
REGIMES = [("2021復甦", "2021-04-01", "2021-12-31"), ("2022空頭", "2022-01-01", "2022-12-31"),
           ("2023復甦", "2023-01-01", "2023-12-31"), ("2024-25多頭", "2024-01-01", "2025-06-30"),
           ("2025下-26", "2025-07-01", "2026-06-08")]
INC = 1.5

TILT_WS = [0.1, 0.3, 0.6, 1.0, 2.0]
# spillover 參數(探針甜蜜點)
FRESH_DAYS = 10     # 只用近 N 天的新鮮共現
HALFLIFE = 60       # 共現衰減半衰期
PARTNER_MOM = 5     # 夥伴動能回看天數


def load_news_titles(codes):
    out = {}
    cdir = DATA_DIR / "finmind_cache"
    for c in codes:
        p = cdir / f"news_{c}.json"
        m = {}
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                for d, items in j.items():
                    if isinstance(items, list):
                        m[d[:10]] = [it.get("title", "") for it in items]
            except Exception:
                pass
        out[c] = m
    return out


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t + 0.35*rs + 0.15*vo + 0.10*ri + 0.05*ma) * 100 * (0.8 + 0.4*tp)


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", "") for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info("載入全史還原 OHLCV ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    logger.info("特徵 ...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    idx = {d: i for i, d in enumerate(alld)}
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes + ["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes + ["0050"]}

    logger.info("反彈/漲停 ...")
    reb_cache, limitup = {}, {}
    for c in codes:
        ds = sorted(OH[c]); cl = []; m = {}; s = set()
        for j, d in enumerate(ds):
            cl.append(OH[c][d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"]*100
                except Exception: pass
            if j > 0 and OH[c][ds[j-1]]["close"] > 0 and OH[c][d]["close"]/OH[c][ds[j-1]]["close"]-1 >= 0.095: s.add(d)
        reb_cache[c] = m; limitup[c] = s

    turn_pct = {}
    for d in alld:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in alld}

    # ── 夥伴近 PARTNER_MOM 日報酬(point-in-time)──
    ret = {c: {} for c in codes}
    for c in codes:
        ds = sorted(d for d in OH[c] if d <= END); cl = [OH[c][d]["close"] for d in ds]
        for j, d in enumerate(ds):
            if j >= PARTNER_MOM and cl[j-PARTNER_MOM] > 0:
                ret[c][d] = cl[j]/cl[j-PARTNER_MOM] - 1

    # ── 共現邊(對稱):c 的標題提及 c2 → (c,c2) 與 (c2,c) 都登記 ──
    logger.info("掃描新聞標題共現網路 ...")
    news = load_news_titles(codes)
    name_ok = {c: (len(names[c]) >= 2) for c in codes}
    pair_dates = defaultdict(list)   # (c, partner) -> [共現日...]
    for c in codes:
        for d, titles in news[c].items():
            if not (START <= d <= END): continue
            mentioned = set()
            for t in titles:
                for c2 in codes:
                    if c2 == c or c2 in mentioned: continue
                    if (c2 in t) or (name_ok[c2] and names[c2] in t):
                        mentioned.add(c2)
            for c2 in mentioned:
                pair_dates[(c, c2)].append(d)
                pair_dates[(c2, c)].append(d)
    for k in pair_dates: pair_dates[k].sort()
    partners_of = defaultdict(set)
    for (c, p) in pair_dates: partners_of[c].add(p)
    logger.info(f"共現對 {len(pair_dates)},有夥伴的股 {len(partners_of)}")

    def prev_trading(d):
        i = idx.get(d); return alld[i-1] if (i and i > 0) else None

    # ── spillover score(嚴格 date<d 的 fresh 共現,夥伴動能用 d-1)──
    logger.info("計算 spillover score ...")
    spill = {c: {} for c in codes}   # spill[c][d] = 加權平均夥伴動能(僅新鮮共現)
    for c in codes:
        ps = partners_of.get(c)
        if not ps: continue
        for d in alld:
            dp = prev_trading(d)
            if dp is None: continue
            num = 0.0; wsum = 0.0
            for p in ps:
                w = 0.0
                for cd in pair_dates[(c, p)]:
                    if cd >= d: break
                    age = idx[d] - idx.get(cd, idx[d])
                    if age <= FRESH_DAYS:
                        w += math.exp(-age / HALFLIFE)
                if w <= 0: continue
                pr = ret[p].get(dp)
                if pr is None: continue
                num += w * pr; wsum += w
            if wsum > 0:
                spill[c][d] = num / wsum

    # ── 逐日橫斷面 z-score(spillover)──
    spill_z = {}   # (d,c) -> z (只取正在 tilt 端處理)
    for d in alld:
        vals = [(c, spill[c][d]) for c in codes if d in spill[c]]
        if len(vals) < 3: continue
        xs = [v for _, v in vals]; m = statistics.mean(xs); sd = statistics.pstdev(xs)
        if sd <= 0:
            for c, _ in vals: spill_z[(d, c)] = 0.0
        else:
            for c, v in vals: spill_z[(d, c)] = (v - m) / sd

    cover_days = len({d for (d, c) in spill_z})
    logger.info(f"spillover z 覆蓋 {len(spill_z)} 個 (d,c),涵蓋 {cover_days} 交易日")

    # ── baseline rows(H 雙引擎,= v15)──
    base_rows = []
    for d in alld:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            rb = reb_cache.get(c, {}).get(d, 0.0)
            sc = (hh*1.0) if bull else (rb*1.5)
            if sc > 0:
                base_rows.append((d, c, sc/100))

    def tilt_rows(w):
        out = []
        for (d, c, e) in base_rows:
            z = spill_z.get((d, c), 0.0)
            out.append((d, c, e * (1.0 + w * max(0.0, z))))
        return out

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    def run(rows):
        r = {}
        for col in COLS:
            ds = col_dates[col]
            r[col] = sim5([x for x in rows if x[0] in ds], opens, closes, limitup,
                          incumbent=INC, sell_mode="exit_only", open_buy="none")
        return r

    logger.info("跑 baseline ...")
    res = {"baseline": run(base_rows)}
    for w in TILT_WS:
        lab = f"spill×w{w}"
        res[lab] = run(tilt_rows(w))
        logger.info(f"{lab} 完成")

    def a(lab, col):
        r = res[lab].get(col); return (r["ret"]-bench[col]) if r and r.get("ret") is not None else None

    # ── 報告 ──
    L = ["# 創意軌 — News Co-mention Momentum Spillover tilt(共現動能外溢)\n",
         f"> baseline=H雙引擎(統一×1.5,exit_only,⑤收盤買+開盤只賣)｜結束{END}｜還原價｜{len(codes)}檔｜DCA(15000+1000/日上限5萬)\n",
         f"> 訊號:A 的新聞共現夥伴近{PARTNER_MOM}日動能,只用近{FRESH_DAYS}日 fresh 共現(衰減半衰期{HALFLIFE}),逐日橫斷面 z;tilt=edge×(1+w·max(0,spill_z))\n",
         "> 學理:Diamond cuts diamond (JBF 2025,中國市場) co-mention momentum spillover。探針 rank-IC +0.018 (IC-t≈+2.15),quintile spread +0.44%\n",
         f"> 防洩漏:spillover 只用 date<d 的共現(不含當日新聞)+ 夥伴動能用 d-1 還原價 → 本軌幾乎 leak-free\n",
         "> ALPHA=策略−同資金DCA0050;扣成本 買0.14%/賣0.44%+滑價0.1%+漲停買不到;換手/持股/曝險取 2年窗\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) | 持股數(2年) | 曝險(2年) |",
         "|" + "---|" * (len(COLS) + 6)]

    labels = ["baseline"] + [f"spill×w{w}" for w in TILT_WS]
    for lb in labels:
        vals = [a(lb, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res[lb].get("2年")
        turn = r2["turn"] if r2 else 0; npos = r2.get("avg_pos", 0) if r2 else 0
        expo = r2.get("avg_expo", 0) if r2 else 0
        mn = f"{min(valid):+.0f}" if valid else "—"; av = f"{sum(valid)/len(valid):+.0f}" if valid else "—"
        L.append(f"| {lb} | {cells} | **{mn}** | {av} | {turn:.1f}x | {npos:.1f} | {expo*100:.0f}% |")

    L += ["", "## Uplift vs baseline(pp,正=tilt 加分)\n",
          "| 變體 | " + " | ".join(COLS) + " | 平均uplift |", "|" + "---|" * (len(COLS) + 2)]
    for lb in labels[1:]:
        ups = []
        for c in COLS:
            ab = a(lb, c); bb = a("baseline", c)
            ups.append(ab - bb if (ab is not None and bb is not None) else None)
        uvalid = [x for x in ups if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in ups)
        av = f"{sum(uvalid)/len(uvalid):+.1f}" if uvalid else "—"
        L.append(f"| {lb} | {cells} | **{av}** |")

    REPORT = ROOT / "reports" / "news_creative_comention.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")

    print("\n=== 摘要(逐窗 + 平均uplift) ===")
    for lb in labels:
        bavg = [a(lb, c) for c in COLS]; bavg = [x for x in bavg if x is not None]
        up = "—"
        if lb != "baseline":
            ups = [a(lb, c) - a("baseline", c) for c in COLS if a(lb, c) is not None and a("baseline", c) is not None]
            up = f"{sum(ups)/len(ups):+.1f}" if ups else "—"
        print(f"{lb:12s} 60={a(lb,'60天')} 90={a(lb,'90天')} 半年={a(lb,'半年')} 1年={a(lb,'1年')} 2年={a(lb,'2年')} 平均α={sum(bavg)/len(bavg):+.0f} uplift={up}")


if __name__ == "__main__":
    main()
