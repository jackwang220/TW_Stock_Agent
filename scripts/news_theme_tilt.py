"""消息面軌 — 主題/族群權重 tilt（重用 v15 引擎，不改既有檔）。

假設:把 universe 按主題分組(被動元件/記憶體/IC設計/CoWoS封裝/散熱/AI伺服器/PCB載板/晶圓代工…),
當某主題「當紅」時(族群近期動能強 或 該主題新聞量近期放大),對 H 雙引擎選股 edge 對該主題成分股加權。

baseline = 不加 tilt 的 H 雙引擎(= v15 統一×1.5 現行實盤,INC=1.5,exit_only)。
approach = baseline 的 edge × (1 + w × theme_heat_z),掃 w 找最佳/單調。

heat 來源(掃):
  mom  = 族群動能:主題成分股近 20 日中位報酬(跨全史可用,較乾淨)
  news = 新聞量:主題成分股近 N 日新聞標題數 vs 基線(只有最近 60/90 天足夠 → 短窗;含當日新聞=輕微洩漏,已註明)
  combo= mom + news 各半

tilt 只在「主題夠紅(heat_z>0)」時放大 edge;heat_z<0 不懲罰(避免變相 veto,符合洞察:消息面只該做正向選股)。

輸出 v15 格式多窗表;alpha vs 同資金 DCA0050;扣成本;報換手/集中度/曝險。
"""
from __future__ import annotations
import sys, json, importlib.util, math, statistics
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
WINDOWS = [("60天",60),("90天",90),("半年",126),("1年",252),("1年半",378),("2年",504)]
REGIMES = [("2021復甦","2021-04-01","2021-12-31"), ("2022空頭","2022-01-01","2022-12-31"),
           ("2023復甦","2023-01-01","2023-12-31"), ("2024-25多頭","2024-01-01","2025-06-30"),
           ("2025下-26","2025-07-01","2026-06-08")]
INC = 1.5   # 與 v15 現行實盤對齊

# tilt 強度掃描
TILT_WS = [0.1, 0.3, 0.6, 1.0, 2.0]
# heat 來源
HEAT_SRCS = ["mom", "news", "combo"]

# ── 主題定義:關鍵字(對股名/產業) → 主題。一檔可屬多主題,heat 取所屬主題 max ──
# 用 base_universe 的 name + industry 比對;另補半導體供應鏈精確成分(companies.json 風格)。
THEME_KEYWORDS = {
    "半導體晶圓": ["台積", "聯電", "世界先進", "力積電", "晶圓"],
    "IC設計":    ["聯發科", "聯詠", "瑞昱", "矽力", "義隆", "創意", "世芯", "祥碩", "群聯", "智原"],
    "記憶體":    ["南亞科", "華邦", "旺宏", "群聯", "鈺創"],
    "被動元件":  ["國巨", "華新科", "禾伸堂", "奇力新", "信昌電"],
    "封裝測試":  ["日月光", "京元電", "矽格", "精測", "頎邦", "南茂"],
    "PCB載板":   ["欣興", "南電", "景碩", "華通", "金像電", "臻鼎", "台光電", "聯茂", "燿華"],
    "散熱":      ["奇鋐", "雙鴻", "建準", "健策", "尼得科"],
    "AI伺服器":  ["廣達", "緯創", "緯穎", "技嘉", "英業達", "鴻海", "華碩", "雲達", "勤誠", "川湖"],
    "光通訊網通": ["智邦", "光聖", "聯亞", "波若威", "上詮", "眾達"],
    "電源散熱模組": ["台達電", "光寶科", "群電"],
    "金融":      ["金", "銀行", "證券", "壽"],
    "航運":      ["長榮", "陽明", "萬海", "海運", "貨櫃"],
    "重電綠能":  ["華城", "士電", "亞力", "中興電", "東元", "綠能", "風電", "太陽能"],
}
# 用 industry 欄補抓(關鍵字漏掉的)
INDUSTRY_THEME = {
    "半導體業": "半導體廣義",
    "電子零組件業": "電子零組件",
    "電腦及週邊設備業": "電腦週邊",
    "金融保險": "金融",
    "航運業": "航運",
    "綠能環保": "重電綠能",
    "綠能環保類": "重電綠能",
}


def build_theme_map(u):
    """code -> set(themes)。優先關鍵字(精確主題),再用 industry 補一個廣義主題。"""
    cm = {}
    for c, info in u.items():
        name = info.get("name", "")
        themes = set()
        for th, kws in THEME_KEYWORDS.items():
            if any(k in name for k in kws):
                themes.add(th)
        ind = info.get("industry", "")
        if ind in INDUSTRY_THEME:
            themes.add(INDUSTRY_THEME[ind])
        if not themes:
            themes.add("其他_" + (ind or "NA"))
        cm[c] = themes
    return cm


def load_news_counts(codes):
    """code -> {date(YYYY-MM-DD): n_titles}。直接讀快取(date<=當日,含當日=輕微洩漏,已註明)。"""
    out = {}
    cdir = DATA_DIR / "finmind_cache"
    for c in codes:
        p = cdir / f"news_{c}.json"
        m = {}
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                for d, items in j.items():
                    m[d[:10]] = len(items) if isinstance(items, list) else 0
            except Exception:
                pass
        out[c] = m
    return out


def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t+0.35*rs+0.15*vo+0.10*ri+0.05*ma)*100*(0.8+0.4*tp)


def main():
    u = json.loads((DATA_DIR/"base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    theme_map = build_theme_map(u)
    # 主題 -> 成分股
    theme_members = {}
    for c, ths in theme_map.items():
        for th in ths:
            theme_members.setdefault(th, []).append(c)
    big_themes = {th: ms for th, ms in theme_members.items() if len(ms) >= 2}
    logger.info(f"主題數(>=2成分): {len(big_themes)}  例: " +
                ", ".join(f"{th}({len(ms)})" for th, ms in sorted(big_themes.items(), key=lambda x:-len(x[1]))[:10]))

    logger.info("載入全史還原 OHLCV 2021~ ...")
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}; OH["0050"] = get_daily_ohlcv("0050", start=START)
    features.__globals__["_OH"] = OH
    logger.info("特徵 ...")
    twii_feat = features("0050"); feats = {c: features(c) for c in codes}
    alld = sorted({d for c in codes for d in OH[c]})
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

    # ── 個股 20 日報酬(供族群動能 heat) ──
    ret20 = {c: {} for c in codes}
    for c in codes:
        ds = sorted(OH[c]); cls = [OH[c][d]["close"] for d in ds]
        for j, d in enumerate(ds):
            if j >= 20 and cls[j-20] > 0:
                ret20[c][d] = cls[j]/cls[j-20] - 1

    # ── 新聞量(全 universe,date->count) ──
    news_counts = load_news_counts(codes)

    # ── 主題 heat(每交易日,每主題):
    #    mom_heat[th][d]  = 該主題成分股當日 ret20 中位數(動能)
    #    news_heat[th][d] = 該主題成分股近 20 日新聞標題數 / 近 120 日日均 - 1(放量倍率,含當日=輕微洩漏)
    #    再對「同日所有主題」做橫斷面 z-score(>0 = 相對其他主題更紅)→ heat_z
    idx = {d: i for i, d in enumerate(alld)}
    mom_heat = {th: {} for th in big_themes}
    for th, ms in big_themes.items():
        for d in alld:
            vals = [ret20[c][d] for c in ms if d in ret20[c]]
            if len(vals) >= 2:
                mom_heat[th][d] = statistics.median(vals)

    # news: 主題每日總標題數 → 近20日和 / 近120日均(每日) 倍率
    theme_news_daily = {th: {} for th in big_themes}
    for th, ms in big_themes.items():
        for d in alld:
            theme_news_daily[th][d] = sum(news_counts.get(c, {}).get(d, 0) for c in ms)
    news_heat = {th: {} for th in big_themes}
    for th in big_themes:
        nd = theme_news_daily[th]
        for j, d in enumerate(alld):
            if j < 120: continue
            recent = sum(nd.get(alld[k], 0) for k in range(j-19, j+1))      # 近20交易日(含當日)
            base = sum(nd.get(alld[k], 0) for k in range(j-119, j+1)) / 120  # 近120日日均
            if base > 0:
                news_heat[th][d] = (recent/20) / base - 1.0  # 放量倍率-1
            elif recent > 0:
                news_heat[th][d] = 1.0
            # base==0 且 recent==0 → 無新聞 → 不設(視為 0 heat)

    def cross_z(heat_map, d):
        """同一日,所有主題的 heat 做 z-score。回傳 {th: z}。"""
        vals = [(th, heat_map[th][d]) for th in big_themes if d in heat_map[th]]
        if len(vals) < 3: return {}
        xs = [v for _, v in vals]
        m = statistics.mean(xs); sd = statistics.pstdev(xs)
        if sd <= 0: return {th: 0.0 for th, _ in vals}
        return {th: (v - m)/sd for th, v in vals}

    # 預算每日 heat_z(每主題),分 mom / news
    momz = {d: cross_z(mom_heat, d) for d in alld}
    newsz = {d: cross_z(news_heat, d) for d in alld}

    def code_heat_z(c, d, src):
        """該股當日的 tilt heat:取所屬主題中最大 z(只取正,負不懲罰)。"""
        ths = theme_map.get(c, set())
        zs = []
        if src in ("mom", "combo"):
            mz = momz.get(d, {})
            zs += [mz[th] for th in ths if th in mz]
        if src in ("news", "combo"):
            nz = newsz.get(d, {})
            zs += [nz[th] for th in ths if th in nz]
        if not zs: return 0.0
        z = max(zs)
        return max(0.0, z)   # 只放大紅主題,不懲罰冷主題

    # ── baseline rows(B純切 H雙引擎,= v15)──
    base_rows = []
    raw = {}  # (d,c)->(sc, bull) 留著 tilt 用
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
                raw[(d, c)] = sc

    def tilt_rows(src, w):
        out = []
        for (d, c, e) in base_rows:
            hz = code_heat_z(c, d, src)
            out.append((d, c, e * (1.0 + w * hz)))
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

    # baseline
    logger.info("跑 baseline(無 tilt)...")
    res = {"baseline": run(base_rows)}
    for src in HEAT_SRCS:
        for w in TILT_WS:
            lab = f"{src}×w{w}"
            res[lab] = run(tilt_rows(src, w))
            logger.info(f"{lab} 完成")

    def a(lab, col):
        r = res[lab].get(col); return (r["ret"]-bench[col]) if r and r.get("ret") is not None else None

    # ── 報告 ──
    L = ["# 消息面軌 — 主題/族群權重 tilt(重用 v15 引擎)\n",
         f"> baseline=H雙引擎(統一×1.5,exit_only,B純切,⑤收盤買+開盤只賣)｜結束{END}｜還原價｜112檔｜DCA(15000+1000/日上限5萬)\n",
         "> tilt: edge×(1+w×heat_z);heat_z=主題橫斷面z(只取正,冷主題不懲罰)｜mom=族群20日中位報酬｜news=主題近20日新聞量/近120日均(含當日=輕微洩漏)｜combo=兩者\n",
         "> ALPHA=策略−同資金DCA0050;換手/持股數/曝險取 2年窗口。新聞量只有最近~60/90天足量,長窗的news欄近似無tilt\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) | 持股數(2年) | 曝險(2年) |",
         "|" + "---|" * (len(COLS) + 6)]

    labels = ["baseline"] + [f"{src}×w{w}" for src in HEAT_SRCS for w in TILT_WS]
    for lb in labels:
        vals = [a(lb, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res[lb].get("2年")
        turn = r2["turn"] if r2 else 0; npos = r2.get("avg_pos", 0) if r2 else 0
        expo = r2.get("avg_expo", 0) if r2 else 0
        mn = f"{min(valid):+.0f}" if valid else "—"; av = f"{sum(valid)/len(valid):+.0f}" if valid else "—"
        L.append(f"| {lb} | {cells} | **{mn}** | {av} | {turn:.1f}x | {npos:.1f} | {expo*100:.0f}% |")

    # uplift 表(approach - baseline,逐窗)
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

    REPORT = ROOT / "reports" / "news_theme_tilt.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    # 簡短 stdout 摘要
    print("\n=== 摘要(2年/60天/平均uplift) ===")
    for lb in labels:
        bavg = [a(lb, c) for c in COLS]; bavg = [x for x in bavg if x is not None]
        up = "—"
        if lb != "baseline":
            ups = [a(lb, c) - a("baseline", c) for c in COLS if a(lb,c) is not None and a("baseline",c) is not None]
            up = f"{sum(ups)/len(ups):+.1f}" if ups else "—"
        print(f"{lb:14s} 2年={a(lb,'2年')} 60天={a(lb,'60天')} 平均α={sum(bavg)/len(bavg):+.0f} uplift={up}")


if __name__ == "__main__":
    main()
