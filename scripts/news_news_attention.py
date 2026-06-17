"""news_news_attention.py — 新聞關注度因子 (attention factor) 驗證。

假設: 個股新聞「近期提及數上升」= attention/動能前兆 -> 加權買。
方法:
  1. 每檔每日數 news_<ticker>.json 近 N5/N20 日標題數(嚴格只用 date < 決策日)。
  2. attention = log率(近5日) - log率(近20日) 之加速度 -> 逐日橫斷面 z-score。
  3. 先算 forward return 的 rank-IC (attention 是否領先報酬)。
  4. tilt: score *= (1 + w*tanh(attn_z)),掃 w 找單調與甜蜜點;baseline(w=0)=純H雙引擎。
  5. v15 格式多窗表 alpha vs DCA0050 扣成本。

第一輪允許輕微洩漏(註明):
  - 新聞嚴格 date < 決策日(get_historical_news 已防當日盤後)-> 這部分乾淨。
  - 但 attention 因子只在「有 news cache 覆蓋」的股票/日子上算;cache 覆蓋本身是
    事後回填的(哪些日被查詢過有選擇偏差),這是潛在洩漏,Verify 階段需等距回填全universe再驗。
  - 沒覆蓋的(date,股) attn=0 -> tilt=1.0(中性,不加不減)。
"""
from __future__ import annotations
import sys, json, importlib.util, math
from pathlib import Path
from datetime import date, timedelta

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
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
N_SHORT, N_LONG = 5, 20

def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def load_news_counts(codes):
    counts = {}; queried = {}
    for c in codes:
        p = DATA_DIR / "finmind_cache" / f"news_{c}.json"
        if not p.exists():
            counts[c] = {}; queried[c] = set(); continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            counts[c] = {}; queried[c] = set(); continue
        cc = {}; q = set()
        for k, v in d.items():
            k10 = str(k)[:10]; q.add(k10); cc[k10] = len(v)
        counts[c] = cc; queried[c] = q
    return counts, queried


def attention_z(counts, queried, codes, alld):
    attn = {}
    for d in alld:
        d0 = date.fromisoformat(d)
        win_short = [(d0 - timedelta(days=k)).isoformat() for k in range(1, N_SHORT+1)]
        win_long  = [(d0 - timedelta(days=k)).isoformat() for k in range(1, N_LONG+1)]
        row = {}
        for c in codes:
            cc = counts.get(c, {}); q = queried.get(c, set())
            covered_long = [dd for dd in win_long if dd in q]
            if len(covered_long) < N_LONG * 0.4:
                continue
            n_short_cov = sum(1 for dd in win_short if dd in q)
            if n_short_cov == 0: continue
            s_short = sum(cc.get(dd, 0) for dd in win_short if dd in q)
            s_long = sum(cc.get(dd, 0) for dd in covered_long)
            rate_short = s_short / max(1, n_short_cov)
            rate_long  = s_long / max(1, len(covered_long))
            raw = math.log1p(rate_short) - math.log1p(rate_long)
            row[c] = raw
        if row:
            attn[d] = row
    attn_z = {}
    for d, row in attn.items():
        vals = list(row.values())
        if len(vals) < 3:
            attn_z[d] = {c: 0.0 for c in row}; continue
        mu = sum(vals)/len(vals)
        sd = (sum((x-mu)**2 for x in vals)/len(vals))**0.5 or 1.0
        attn_z[d] = {c: (x-mu)/sd for c, x in row.items()}
    return attn_z


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0]*len(xs); i = 0
    while i < len(xs):
        j = i
        while j+1 < len(xs) and xs[order[j+1]] == xs[order[i]]: j += 1
        avg = (i+j)/2 + 1
        for k in range(i, j+1): r[order[k]] = avg
        i = j+1
    return r


def spearman(xs, ys):
    from statistics import mean
    if len(xs) < 3: return None
    rx = _rank(xs); ry = _rank(ys)
    mx = mean(rx); my = mean(ry)
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    dx = (sum((a-mx)**2 for a in rx))**0.5; dy = (sum((b-my)**2 for b in ry))**0.5
    return num/(dx*dy) if dx and dy else None


def rank_ic(attn_z, closes, alld_by_code, fwd=5):
    ics = []
    for d in sorted(attn_z):
        row = attn_z[d]; xs = []; ys = []
        for c, z in row.items():
            ds = alld_by_code.get(c)
            if not ds: continue
            cl = closes.get(c, {})
            cds = [x for x in ds if x >= d]
            if len(cds) < fwd+1: continue
            d_now = cds[0]; d_fut = cds[fwd]
            if d_now != d: continue
            p0 = cl.get(d_now); p1 = cl.get(d_fut)
            if not p0 or not p1: continue
            xs.append(z); ys.append(p1/p0 - 1)
        if len(xs) >= 5:
            ic = spearman(xs, ys)
            if ic is not None: ics.append(ic)
    if not ics: return None, 0
    return sum(ics)/len(ics), len(ics)


def build_base_rows(codes, OH, feats, twii_feat, reb_cache, regime_bull, turn_pct, alld):
    rows = []
    for d in alld:
        ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull[d]
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
            hh = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            rb = reb_cache.get(c, {}).get(d, 0.0)
            sc = max(hh*1.0, rb*0.0) if bull else max(hh*0.0, rb*1.5)
            if sc > 0: rows.append((d, c, sc/100))
    return rows


def tilt_rows(base_rows, attn_z, w):
    out = []
    for d, c, sc in base_rows:
        z = attn_z.get(d, {}).get(c, 0.0)
        mult = max(0.3, 1.0 + w*math.tanh(z))
        out.append((d, c, sc*mult))
    return out


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info("載入全史還原 OHLCV 2021~ ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    logger.info("特徵 ...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c]})
    opens = {c: {d: OH[c][d]["open"] for d in OH[c]} for c in codes + ["0050"]}
    closes = {c: {d: OH[c][d]["close"] for d in OH[c]} for c in codes + ["0050"]}
    alld_by_code = {c: sorted(OH[c]) for c in codes}

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

    logger.info("新聞關注度因子 ...")
    counts, queried = load_news_counts(codes)
    azz = attention_z(counts, queried, codes, alld)
    cov_days = len(azz)
    avg_stocks = (sum(len(v) for v in azz.values())/cov_days) if cov_days else 0
    logger.info(f"attention 覆蓋: {cov_days} 個交易日有訊號, 平均每日 {avg_stocks:.1f} 檔")

    logger.info("rank-IC (forward) ...")
    ic1, n1 = rank_ic(azz, closes, alld_by_code, fwd=1)
    ic5, n5 = rank_ic(azz, closes, alld_by_code, fwd=5)
    ic10, n10 = rank_ic(azz, closes, alld_by_code, fwd=10)
    logger.info(f"rank-IC 1d={ic1} (n={n1})  5d={ic5} (n={n5})  10d={ic10} (n={n10})")

    base_rows = build_base_rows(codes, OH, feats, twii_feat, reb_cache, regime_bull, turn_pct, alld)

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    WEIGHTS = [-1.0, -0.6, -0.3, 0.0, 0.1, 0.3, 0.6, 1.0, 2.0]
    res = {}
    for w in WEIGHTS:
        rws = base_rows if w == 0 else tilt_rows(base_rows, azz, w)
        for col in COLS:
            ds = col_dates[col]
            res[(w, col)] = sim5([x for x in rws if x[0] in ds], opens, closes, limitup,
                                 incumbent=1.5, sell_mode="exit_only", open_buy="none", asym=False)
        logger.info(f"w={w} 完成")

    def a(w, col):
        r = res.get((w, col)); return (r["ret"]-bench[col]) if r else None

    ic1s = f"{ic1:.4f}" if ic1 is not None else "NA"
    ic5s = f"{ic5:.4f}" if ic5 is not None else "NA"
    ic10s = f"{ic10:.4f}" if ic10 is not None else "NA"
    L = ["# news_attention — 新聞關注度因子 tilt (第一輪,輕微洩漏見註)\n",
         f"> baseline=H雙引擎B純切(w=0);approach=score×(1+w·tanh(attn_z));⑤收盤買開盤賣;112檔;DCA;扣成本\n",
         f"> attention=log率(近5日)−log率(近20日)橫斷面z;嚴格 news date<決策日;覆蓋{cov_days}日平均{avg_stocks:.1f}檔/日\n",
         f"> rank-IC(forward,評估用): 1d={ic1s}(n={n1}) 5d={ic5s}(n={n5}) 10d={ic10s}(n={n10})\n",
         "> ALPHA=策略−同資金DCA0050;換手/持股取2年窗\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| tilt w | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) | 持股數(2年) |",
         "|" + "---|" * (len(COLS) + 5)]
    for w in WEIGHTS:
        vals = [a(w, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res.get((w, "2年"))
        turn = r2["turn"] if r2 else 0; npos = r2.get("avg_pos", 0) if r2 else 0
        tag = "(baseline)" if w == 0 else ""
        L.append(f"| {w} {tag} | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {turn:.1f}x | {npos:.1f} |")
    L.append("\n## uplift vs baseline (w − w0), pp\n")
    L.append("| tilt w | " + " | ".join(COLS) + " | 平均 |")
    L.append("|" + "---|" * (len(COLS) + 2))
    for w in WEIGHTS:
        if w == 0: continue
        ups = []
        for c in COLS:
            aw = a(w, c); a0 = a(0.0, c)
            ups.append((aw - a0) if (aw is not None and a0 is not None) else None)
        uv = [x for x in ups if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in ups)
        L.append(f"| {w} | {cells} | {sum(uv)/len(uv):+.0f} |")

    REPORT = ROOT / "reports" / "news_attention.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
