"""消息面軌 · 處置/管制「解除」事件 tilt 回測。

假設：股票剛從處置/注意/管制「出關」(處置期剛結束) → 報復性動能，事件後 N 日加權。
方法：對 universe 用 Google News 抓「處置」公告 → regex 解析處置結束日 → 建事件行事曆
      (event = 某交易日 = 某股處置結束後第一個交易日)。回測時：訊號日 d 若某股在過去
      EVENT_WINDOW 個交易日內剛出關，edge ×(1+tilt)。baseline=不加 tilt 的 H+反彈雙引擎。

兩段式：
  1) build_events()  : 連網掃 Google News(分批/限流)，把每股的處置結束日存到
                       data/disposal_events_cache.json。已存在就直接用(離線可重跑)。
  2) backtest        : 純離線，讀 cache + ohlcv_cache，掃 tilt 權重出 v15 格式多窗表。

第一輪洩漏聲明：
  - 處置「結束日」由新聞標題解析(如「至06/11」)。該結束日的公告通常在處置「開始時」
    就發布(處置一次公告即含起訖)，故「未來會在某日出關」這件事在出關前其實已知
    → 用它當『出關後』事件大致 leak-free。但我們用 search_company_news 的 max_age_hours=0
    歷史搜尋無嚴格 as_of 上界裁切到「公告發布日」，可能納入晚於事件起點才出現的延長公告
    → 視為輕微洩漏(第一輪允許)。Verify 階段再用 search_news_around_date 嚴格 as_of 重跑。

用法：
    uv run python scripts/news_disposition_end.py            # 有 cache 就只回測
    uv run python scripts/news_disposition_end.py --scan     # 強制重新連網掃事件
    uv run python scripts/news_disposition_end.py --scan-only
"""
from __future__ import annotations
import sys, json, importlib.util, math, time, argparse
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.market_status import _parse_disposal_end_date, _parse_pub_date

def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

v3 = _load("v3", "scripts/exp_step1_v3.py")
v5 = _load("v5", "scripts/exp_step1_v5.py")
v6 = _load("v6", "scripts/exp_step1_v6.py")
ec = _load("ec", "scripts/exp_60d_entry_compare.py")
features, _factors, oh = v3.features, v3._factors, v3.oh
sim5 = ec.sim_buyclose_sellopen
bench_0050 = v6.bench_0050

END = "2026-06-08"
WINDOWS = [("60天", 60), ("90天", 90), ("半年", 126), ("1年", 252), ("1年半", 378), ("2年", 504)]
REGIMES = [("2021復甦", "2021-04-01", "2021-12-31"), ("2022空頭", "2022-01-01", "2022-12-31"),
           ("2023復甦", "2023-01-01", "2023-12-31"), ("2024-25多頭", "2024-01-01", "2025-06-30"),
           ("2025下-26", "2025-07-01", "2026-06-08")]
TOPN = 4
INC = 1.5
EVENT_WINDOW = 5      # 出關後幾個交易日內視為「事件熱區」(預設;可掃)
EVENT_CACHE = DATA_DIR / "disposal_events_cache.json"


# ── H+ 分數(與 entry_compare 一致)───────────────────────────────────────────
def h_score(ff, tp):
    if ff is None: return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t + 0.35*rs + 0.15*vo + 0.10*ri + 0.05*ma) * 100 * (0.8 + 0.4*tp)


# ── 第1段：連網掃處置結束事件 → cache ───────────────────────────────────────
def build_events(codes, names, force=False):
    """回傳 {code: [end_date_iso, ...]} 處置結束日清單。

    用 Google News RSS 對每股做歷史搜尋(after:/before: 涵蓋全回測窗),
    收集所有含「處置」的標題,regex 解析結束日,去重。分批限流。
    """
    if EVENT_CACHE.exists() and not force:
        data = json.loads(EVENT_CACHE.read_text(encoding="utf-8"))
        logger.info(f"讀取處置事件 cache: {EVENT_CACHE.name} ({sum(len(v) for v in data.values())} 個結束日 / {len([k for k,v in data.items() if v])} 股)")
        return data

    from tw_stock_agent.news.scanner import search_company_news
    logger.warning("連網掃 Google News 處置公告(限流,可能數分鐘)...")
    events: dict[str, list[str]] = {}
    # 全窗搜尋:after 取回測起點前,before 取結束後一天
    after = "2021-01-01"
    before = (datetime.strptime(END, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()
    today_ref = date.today()
    for idx, c in enumerate(codes, 1):
        nm = names.get(c, c)
        ends: set[str] = set()
        queries = [f"{c} 處置 after:{after} before:{before}",
                   f"{nm} 處置股 after:{after} before:{before}"]
        for q in queries:
            try:
                arts = search_company_news(q, max_articles=20, max_age_hours=0, snippet_chars=60)
            except Exception as exc:
                logger.error(f"{c} 搜尋失敗: {exc}"); arts = []
            for a in arts:
                title = a.get("title", "")
                if "處置" not in title:
                    continue
                pub_d = _parse_pub_date(a.get("published", "") or "")
                ref = pub_d or today_ref
                ed = _parse_disposal_end_date(title, ref)
                if ed and date(2021, 1, 1) <= ed <= datetime.strptime(END, "%Y-%m-%d").date():
                    ends.add(ed.isoformat())
            time.sleep(0.4)   # 限流
        events[c] = sorted(ends)
        if ends:
            logger.info(f"[{idx}/{len(codes)}] {c} {nm}: {len(ends)} 處置結束日 {sorted(ends)}")
        elif idx % 20 == 0:
            logger.info(f"[{idx}/{len(codes)}] ...掃描中")
    EVENT_CACHE.write_text(json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")
    n_end = sum(len(v) for v in events.values()); n_stk = len([k for k,v in events.items() if v])
    logger.success(f"事件 cache 寫入 {EVENT_CACHE} ({n_end} 結束日 / {n_stk} 股)")
    return events


# ── 把「處置結束日」對應到回測的事件熱區交易日集合 ───────────────────────────
def build_event_hot(events, alld, window):
    """{code: set(交易日)}  = 每股「出關後 window 個交易日」的熱區。
    出關日 = 結束日的『下一個交易日』(處置期最後一天為結束日,出關隔天開始正常)。
    """
    alld_sorted = sorted(alld)
    hot: dict[str, set[str]] = {}
    for c, ends in events.items():
        s: set[str] = set()
        for e_iso in ends:
            # 找第一個 > 結束日的交易日 = 出關日
            future = [d for d in alld_sorted if d > e_iso]
            if not future:
                continue
            start_idx = alld_sorted.index(future[0])
            for k in range(window):
                if start_idx + k < len(alld_sorted):
                    s.add(alld_sorted[start_idx + k])
        if s:
            hot[c] = s
    return hot


# ── 主程式 ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="強制重新連網掃事件")
    ap.add_argument("--scan-only", action="store_true", help="只掃事件不回測")
    args = ap.parse_args()

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    events = build_events(codes, names, force=args.scan)
    if args.scan_only:
        return

    logger.info(f"載入特徵({len(codes)} 支)...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}
    opens, closes = {}, {}
    for c in codes + ["0050"]:
        o = oh(c)
        opens[c] = {d: o[d]["open"] for d in o}
        closes[c] = {d: o[d]["close"] for d in o}

    alld = sorted({d for c in codes for d in closes.get(c, {}) if d <= END})
    sig_days = alld[-max(n for _, n in WINDOWS):]
    logger.info(f"訊號範圍 {sig_days[0]} ~ {sig_days[-1]} ({len(sig_days)} 日)")

    # 反彈 + 漲停
    logger.info("反彈/漲停...")
    reb_cache, limitup = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END)
        cl, m, s = [], {}, set()
        for j, d in enumerate(ds):
            cl.append(o[d]["close"])
            if len(cl) >= 25:
                try:
                    g = rebound_signal(cl, turns.get(c, 0.0))
                    if g.get("fired"): m[d] = g["score"] * 100
                except Exception: pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095:
                s.add(d)
        reb_cache[c] = m; limitup[c] = s

    turn_pct = {}
    for d in sig_days:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i+1)/len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in sig_days}

    # 事件熱區覆蓋率診斷
    hot = build_event_hot(events, alld, EVENT_WINDOW)
    n_event_stk = len(hot)
    n_event_days_in_sig = sum(1 for c in hot for d in hot[c] if d in set(sig_days))
    logger.info(f"處置結束事件: {n_event_stk} 股有事件 | 熱區×訊號日交集 {n_event_days_in_sig} 筆(window={EVENT_WINDOW})")

    # ── 基礎雙引擎訊號(未 tilt) rows0 = [(d,c,raw_edge)] ──
    def build_rows(tilt, window):
        hotw = build_event_hot(events, alld, window)
        rows, n_tilt = [], 0
        for d in sig_days:
            ir = twii_feat.get(d, {}).get("ret20"); bull = regime_bull.get(d)
            sc = []
            for c in codes:
                f = feats.get(c, {})
                if d not in f or math.isnan(f[d].get("ma20", float("nan"))): continue
                v = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5)) if bull else reb_cache.get(c, {}).get(d, 0.0)
                if v > 0:
                    if tilt != 0.0 and c in hotw and d in hotw[c]:
                        v *= (1.0 + tilt); n_tilt += 1
                    sc.append((v, c))
            sc.sort(reverse=True)
            for v, c in sc[:TOPN]:
                rows.append((d, c, v/100))
        return rows, n_tilt

    cal_end = [d for d in alld if d <= END]
    col_dates = {}
    for wl, n in WINDOWS: col_dates[wl] = set(cal_end[-n:])
    for lab, s, e in REGIMES: col_dates[lab] = {d for d in alld if s <= d <= e}
    COLS = [wl for wl, _ in WINDOWS] + [lab for lab, _, _ in REGIMES]
    bench = {col: bench_0050(opens["0050"], closes["0050"], sorted(col_dates[col])) for col in COLS}

    # ── 掃 tilt(含 baseline tilt=0)──
    TILTS = [0.0, 0.1, 0.3, 0.6, 1.0, 2.0]
    rows_cache = {}
    res = {}
    for tilt in TILTS:
        rows, n_tilt = build_rows(tilt, EVENT_WINDOW)
        rows_cache[tilt] = (rows, n_tilt)
        for col in COLS:
            ds = col_dates[col]
            res[(tilt, col)] = sim5([x for x in rows if x[0] in ds], opens, closes, limitup,
                                    incumbent=INC, sell_mode="exit_only", open_buy="none")
        logger.info(f"tilt={tilt} 完成 (tilt命中 {n_tilt} 訊號-股-日)")

    def a(tilt, col):
        r = res.get((tilt, col)); return (r["ret"] - bench[col]) if r else None

    def lab(tilt):
        return "baseline(無tilt)" if tilt == 0.0 else f"tilt ×(1+{tilt})"

    L = ["# 消息面軌 · 處置/管制解除事件 tilt(第一輪,輕微洩漏)\n",
         f"> 引擎=H+反彈雙引擎⑤(買收賣開/exit_only/INC1.5)｜結束{END}｜還原價｜{len(codes)}檔｜DCA(15000+1000/日上限5萬)\n",
         f"> 事件=股票處置期『結束日』後 {EVENT_WINDOW} 交易日熱區,該股 edge ×(1+tilt)｜事件來源 Google News 標題解析\n",
         f"> 覆蓋:{n_event_stk}/{len(codes)} 股有處置結束事件,熱區×訊號交集 {n_event_days_in_sig} 筆\n",
         "> ALPHA=策略−同資金DCA0050(扣成本)；換手/持股取 2年窗\n",
         "> 0050 各欄基準: " + " ".join(f"{c}{bench[c]:+.0f}%" for c in COLS) + "\n",
         "| 變體 | " + " | ".join(COLS) + " | 最差 | 平均 | 換手(2年) | 持股數(2年) | tilt命中 |",
         "|" + "---|" * (len(COLS) + 6)]
    for tilt in TILTS:
        vals = [a(tilt, c) for c in COLS]; valid = [x for x in vals if x is not None]
        cells = " | ".join(f"{x:+.0f}" if x is not None else "—" for x in vals)
        r2 = res.get((tilt, "2年")); turn = r2["turn"] if r2 else 0; npos = r2.get("avg_pos", 0) if r2 else 0
        n_tilt = rows_cache[tilt][1]
        L.append(f"| {lab(tilt)} | {cells} | **{min(valid):+.0f}** | {sum(valid)/len(valid):+.0f} | {turn:.1f}x | {npos:.1f} | {n_tilt} |")

    # uplift 列(相對 baseline tilt=0)
    L += ["", "## uplift = approach − baseline (pp,逐窗)\n",
          "| 變體 | " + " | ".join(COLS) + " | 最差Δ | 平均Δ |",
          "|" + "---|" * (len(COLS) + 3)]
    base_a = {c: a(0.0, c) for c in COLS}
    for tilt in TILTS[1:]:
        ups = [(a(tilt, c) - base_a[c]) if (a(tilt, c) is not None and base_a[c] is not None) else None for c in COLS]
        uv = [x for x in ups if x is not None]
        cells = " | ".join(f"{x:+.1f}" if x is not None else "—" for x in ups)
        L.append(f"| {lab(tilt)} | {cells} | {min(uv):+.1f} | {sum(uv)/len(uv):+.1f} |")

    REPORT = ROOT / "reports" / "news_disposition_end.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")

    # 印關鍵摺要供回傳
    print("\n=== SUMMARY ===")
    print(f"event_coverage: {n_event_stk}/{len(codes)} stocks, {n_event_days_in_sig} hot-day hits in signal range")
    for tilt in TILTS:
        vals = [a(tilt, c) for c in COLS]; valid = [x for x in vals if x is not None]
        print(f"tilt={tilt}: 60d={a(tilt,'60天')} 90d={a(tilt,'90天')} 2y={a(tilt,'2年')} worst={min(valid):+.0f} avg={sum(valid)/len(valid):+.0f} hits={rows_cache[tilt][1]}")


if __name__ == "__main__":
    main()
