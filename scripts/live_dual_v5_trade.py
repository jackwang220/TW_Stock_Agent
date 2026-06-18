"""⑤策略實盤下單:買收盤吃隔夜溢價 + exit_only 低換手。可用永豐 Shioaji API 真實下單。

策略(多窗口驗證最優,見 reports/exp_entry_compare_multiwin.md):
  選股 = H+反彈 regime 雙引擎(大盤站上20MA→打H動能;跌破→打跌深反彈),純技術無LLM。
  執行 = ⑤c(2年 alpha +79%,所有窗口最優):
    • buy  leg(收盤前 13:00–13:25): 名單內未達目標的 → 買到目標金額(只買、不減碼)
    • sell leg(隔日開盤 09:00–09:05): 持倉中「掉出今日名單」的 → 全部出場
  edge = 吃強勢股的隔夜跳空溢價(買在收盤=隔夜前、低換手抱住)。

  ⚠️ 風險:這是「收盤滿倉 = 裸隔夜曝險」。回測那 2 年是大多頭,沒測過長期空頭。
     大盤轉空時隔夜會 gap-down 反向吃滿,務必盯著 KILL_SWITCH、用得起才上 --live。

三段安全(務必依序驗證):
  --dry-run(預設): 登入+讀持倉+算單,但「不送任何委託」,只印給你看。
  --sim          : simulation=True 模擬下單(不花真錢)。
  --live         : simulation=False 真實下單。需 env SHIOAJI_LIVE_CONFIRM=YES 才放行。

用法:
  # 收盤前(13:00後)買進腿:
  python scripts/live_dual_v5_trade.py --leg buy --capital 50000            # dry-run 看計畫
  python scripts/live_dual_v5_trade.py --leg buy --capital 50000 --sim
  SHIOAJI_LIVE_CONFIRM=YES python scripts/live_dual_v5_trade.py --leg buy --capital 50000 --live
  # 隔日開盤(09:00)賣出腿(出清掉出名單的):
  python scripts/live_dual_v5_trade.py --leg sell                           # dry-run
  SHIOAJI_LIVE_CONFIRM=YES python scripts/live_dual_v5_trade.py --leg sell --live
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

# 複用 exp_step1_v5 的特徵工程（與回測/dual_engine_targets 完全相同）
_v5 = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("v5", ROOT / "scripts/exp_step1_v5.py"))
importlib.util.spec_from_file_location("v5", ROOT / "scripts/exp_step1_v5.py").loader.exec_module(_v5)
features, _factors = _v5.features, _v5._factors

# ── 策略參數（與 dual_engine_targets / 回測同步）─────────────────────────────
MAX_SIG, EXPO_CAP, EXPO_FLOOR, TIE = 3, 0.90, 0.30, 0.90
INC = 1.75  # 持股加權(incumbent):排名時持股分數 ×INC,讓持股黏著、不易被踢。1.75=穩健中間值(2026-06驗:1.5暴衝/2.0最穩,1.75哪種行情都不太差、不在刀尖)
PROTECT_MA = 60      # 下檔保險:0050 收盤跌破 MAn(下降趨勢)→ 降曝險(2026-06驗:趨勢濾網是唯一可上線改進,2022 −58→−24,多頭爆衝全留)
# PROTECT_SCALE=0.0 → A2 完全空手(★預設;唯一「忠實驗證+純exit_only零新邏輯」,go-live安全)
# PROTECT_SCALE=0.5 → 半倉(需賣腿 trim 才真生效;此 trim 為新增邏輯、回測用P&L overlay理想化過、未經實單驗證 → 暫不建議上線)
PROTECT_SCALE = 0.0
ALT_MIN_SCORE_PCT = 0.85   # 備選分數門檻:須 ≥ 正取最低分 × 此比例,否則不補(避免拉進爛股)

# ── 資金模型(DCA:起始15000、每交易日+1000、5萬封頂;正式上線後可調)────────────
CAP_INITIAL = 15_000           # 起始資金
CAP_DAILY   = 1_000            # 每交易日加碼
CAP_MAX     = 50_000           # 累計上限(封頂)
CAP_START   = "2026-06-18"     # DCA 起算日(上線首日);此日當天=15000,之後每交易日+1000

# ── 安全參數（上線前依你的資金調）────────────────────────────────────────────
MAX_POSITION_TWD  = 20_000     # 單一個股「持有」金額上限
DAILY_BUY_CAP_TWD = 15_000     # 單一個股「當天最多買」金額上限
MAX_TOTAL_BUY_TWD = 50_000     # 本次「總買入」金額上限
MAX_ORDERS        = 12         # 本次最多送幾筆委託
MIN_ORDER_TWD     = 1_000      # 小於此金額不送（避免零碎）
LOT_SHARES        = 1_000      # 整股 1 張 = 1000 股（--common 模式用）
LIMIT_BUFFER      = 0.015      # 限價:買 +1.5% / 賣 -1.5%（求成交但不追市價）
LIMIT_UP_GUARD    = 0.094      # 漲幅 ≥9.4% 視為鎖漲停 → 不買
KILL_SWITCH       = ROOT / "KILL_SWITCH"   # 此檔存在 → 立即中止
SPACING_SEC       = 1.0        # 永豐 API 呼叫間隔
LEDGER            = DATA_DIR / "positions.json"   # 手動帳本(元大手動交易用;--ledger 讀它算持有/損益)
CAP_FILE          = DATA_DIR / "capital.json"      # 手動現金池;有設就覆蓋 DCA(ledger.py cash 設定)
WEIGHTS_FILE      = DATA_DIR / "weights.json"      # 雙引擎權重;有設就覆蓋預設(ledger.py weights 設定)
MANUAL_FILE       = DATA_DIR / "manual_scores.json"  # 手動觀察加減分;{code:{bonus,until,note}};加在排名分上、不被×INC(manual_score.py 設)

# 零股 vs 整股：預設盤中零股(IntradayOdd,1股起,小資金/高價股可買;⑤買在13:00盤中正好可用)
ODD_LOT = True

TODAY = date.today().isoformat()   # 即時訊號模式用,把今日盤中 bar 灌進序列


def fetch_all_snapshots(api) -> dict:
    """用已登入的 api 批次抓 base_universe 全部即時 snapshot,組今日 bar(close=即時價)。"""
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()) + ["0050"]
    cts = []
    for c in codes:
        ct = api.Contracts.Stocks.get(c) if hasattr(api.Contracts.Stocks, "get") else api.Contracts.Stocks[c]
        if ct is not None:
            cts.append((c, ct))
    bars: dict[str, dict] = {}
    need = max(1, len(cts) // 2)          # 至少抓到半數才算成功(開盤瞬間常全 0)
    for attempt in range(3):
        bars = {}
        for i in range(0, len(cts), 100):
            batch = cts[i:i + 100]
            try:
                snaps = api.snapshots([ct for _, ct in batch])
            except Exception as e:
                log(f"  snapshot 批次失敗: {e}"); continue
            for (c, _), sn in zip(batch, snaps):
                close = float(getattr(sn, "close", 0) or 0)
                if close <= 0:
                    continue
                bars[c] = {"open": float(getattr(sn, "open", 0) or close),
                           "high": float(getattr(sn, "high", 0) or close),
                           "low":  float(getattr(sn, "low", 0) or close),
                           "close": close,
                           "volume": float(getattr(sn, "total_volume", 0) or 0),
                           "amount": float(getattr(sn, "total_amount", 0) or 0)}
            time.sleep(0.5)
        if "0050" in bars and len(bars) >= need:        # 0050 必抓到(決策日靠它)
            break
        if attempt < 2:
            log(f"  即時報價只抓到 {len(bars)}/{len(cts)} 檔(0050={'有' if '0050' in bars else '無'})"
                f",3 秒後重試 {attempt + 2}/3 ...")
            time.sleep(3)
    log(f"  抓到 {len(bars)} 檔即時報價")
    return bars


def log(msg: str) -> None:
    print(f"[v5-trade] {msg}", flush=True)


def notify(text: str) -> None:
    """Discord webhook 推播(best-effort);未設 DISCORD_WEBHOOK_URL 則略過。"""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        data = json.dumps({"content": text[:1900]}).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "User-Agent": "tw-stock-agent/1.0 (+discord-webhook)",  # Discord/Cloudflare 會擋無 UA → 403
        })
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"  推播失敗: {e}")


def fmt_plan(leg: str, mode: str, picks: dict, plan: list,
             positions: list | None = None, records: list | None = None) -> str:
    """組人類可讀的下單摘要(給 log/推播/手動照單下單共用)。
    含:本腿買/賣的價量金額、目前持有與個股損益、總投入/市值/未實現損益。"""
    names = picks.get("names", {})
    nm = lambda c: names.get(c, c)
    px = lambda p: f"{p:g}"
    regime = "🟢多頭→H動能" if picks.get("bull") else "🔴空頭→反彈"
    tag = {"buy": "買進腿(收盤前)", "sell": "賣出腿(開盤)"}[leg]
    cap = picks.get("capital")
    cap_s = f" · 可投{cap:,.0f}" if cap else ""
    # 標題在框外(emoji/粗體會渲染);明細包進程式碼框 → 每則訊息=獨立框,不會黏在一起
    head = f"📊 **v5實盤 {tag}** · {picks.get('date')} · {regime} · 曝險{picks.get('expo', 0):.0%}{cap_s} · [{mode}]"
    if picks.get("warn"):
        head += f"\n⚠️ {picks['warn']}"
    sent = {r.get("code"): r for r in (records or [])}
    score_of = {c: s for s, c in picks.get("ranked", [])}       # ranked 含主名單+備選
    for s, c in picks.get("sel", []):
        score_of.setdefault(c, s)
    def sc(code):
        s = score_of.get(code)
        return f"分{s:.0f}" if s is not None else ""
    def mark(code):
        if records is None:
            return ""
        r = sent.get(code)
        return "  ✅" if (r and not r.get("error")) else "  ❌"
    _first = [True]
    def div(t):
        corner = "┌" if _first[0] else "├"
        _first[0] = False
        return f"{corner}─── {t} " + "─" * max(3, 20 - len(t) * 2)

    B = []          # 框內各行
    buys = [o for o in plan if o[1] == "Buy"]
    sells = [o for o in plan if o[1] == "Sell"]
    if buys:
        B.append(div("買進"))
        for code, _a, qty, lp, why in buys:
            alt = " 🔻備選" if "備選" in str(why) else ""
            B.append(f"│ 買 {nm(code)}({code}) {sc(code)}")
            B.append(f"│    {px(lp)} × {qty}股 = {qty*lp:,.0f}元{alt}{mark(code)}")
    if sells:
        B.append(div("賣出"))
        for code, _a, qty, lp, _w in sells:
            B.append(f"│ 賣 {nm(code)}({code})  掉出名單")
            B.append(f"│    {px(lp)} × {qty}股 = {qty*lp:,.0f}元{mark(code)}")
    if not plan:
        B.append(div("今日"))
        B.append("│ → 無需調整(無買賣)")

    tot_cost = tot_mv = 0.0
    if positions:
        B.append(div("目前持有"))
        for p in sorted(positions, key=lambda x: -x["last"] * x["qty"]):
            mv, cost = p["last"] * p["qty"], p["cost"] * p["qty"]
            tot_mv += mv; tot_cost += cost
            pc = (p["last"] / p["cost"] - 1) * 100 if p["cost"] > 0 else 0.0
            B.append(f"│ {nm(p['code'])}({p['code']}) {sc(p['code'])}  {p['qty']}股")
            B.append(f"│    現{px(p['last'])} = {mv:,.0f}元  損益{mv-cost:+,.0f} ({pc:+.1f}%)")
    else:
        B.append(div("目前持有"))
        B.append("│ 無")

    if tot_cost > 0:
        pnl = tot_mv - tot_cost
        B.append(div("總計"))
        B.append(f"│ 投入 {tot_cost:,.0f} · 市值 {tot_mv:,.0f}")
        B.append(f"│ 未實現 {pnl:+,.0f}元 ({pnl/tot_cost*100:+.1f}%)")
    B.append("└" + "─" * 24)
    return head + "\n```\n" + "\n".join(B) + "\n```"


def round_lots(shares: float) -> int:
    return int(shares // LOT_SHARES) * LOT_SHARES


def align_tick(price: float, side: str) -> float:
    """對齊台股最小跳動單位(否則委託會被拒)。買單向上、賣單向下(確保能成交)。"""
    import math
    if   price < 10:   tick = 0.01
    elif price < 50:   tick = 0.05
    elif price < 100:  tick = 0.1
    elif price < 500:  tick = 0.5
    elif price < 1000: tick = 1.0
    else:              tick = 5.0
    steps = price / tick
    steps = math.ceil(steps) if side == "Buy" else math.floor(steps)
    return round(steps * tick, 2)


def h_score(ff, tp: float) -> float:
    if ff is None:
        return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35 * t + 0.35 * rs + 0.15 * vo + 0.10 * ri + 0.05 * ma) * 100 * (0.8 + 0.4 * tp)


# ── 策略：算今日名單 + 目標持倉（純技術，不碰 API）──────────────────────────
def compute_picks(capital: float | None = None, date: str | None = None, refresh: bool = False,
                  live_bars: dict | None = None,
                  weights: tuple = (1.0, 0.0, 0.0, 1.5),
                  held: set | None = None, require_today: bool = False,
                  manual_override: dict | None = None, extra_codes: list | None = None) -> dict:
    """weights=(多頭H, 多頭reb, 空頭H, 空頭reb)。預設(1,0,0,1.5)=B純切。
    v9 H雙引擎C=(1,0.6,0.3,1.3)；v8 雙引擎A溫和=(1,0.7,0.6,1.3)。
    capital=None → 用 DCA 模型自動算(起始15000、每交易日+1000、5萬封頂)。
    held=目前持股代號集合 → 排名時 ×INC 黏著(配重/曝險仍用原始分數;對齊回測)。
    extra_codes=池外要一起算分的股(查詢用);手動加減分名單裡的池外股也會自動納入候選(觀察股加分後可被選/買)。"""
    held = held or set()
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    # 池外候選:呼叫端 extra_codes + 手動加減分名單裡不在 universe 的股(讓觀察股加分後也能進候選)
    _man_codes = (list(manual_override.keys()) if manual_override is not None
                  else (list(json.loads(MANUAL_FILE.read_text(encoding="utf-8")).keys()) if MANUAL_FILE.exists() else []))
    extra = [c for c in dict.fromkeys(list(extra_codes or []) + _man_codes) if c not in u]
    codes = list(u.keys()) + extra
    names = {c: u[c].get("name", c) for c in u}
    for c in extra:
        names.setdefault(c, c)
    turns = {c: u[c].get("avg_turnover", 0.0) for c in u}

    log(f"載入行情({len(codes)} 檔{('+池外'+str(len(extra))) if extra else ''}{'+強制更新' if refresh else ''})...")
    OH = {c: get_daily_ohlcv(c, force_refresh=refresh) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050", force_refresh=refresh)
    for c in extra:                      # 池外股:turns 從成交額算(universe 才有 avg_turnover)
        amts = [OH[c][x].get("amount", 0) for x in sorted(OH.get(c, {}))][-120:]
        turns[c] = sum(amts) / len(amts) if amts else 0.0
    # 即時訊號：把今日盤中即時 bar 灌進去，讓決策日=今天、訊號用今日盤中價（避免追高已反彈的股）
    if live_bars:
        n = 0
        for c, bar in live_bars.items():
            if c in OH and TODAY not in OH[c]:
                OH[c][TODAY] = bar; n += 1
        log(f"灌入 {TODAY} 即時 bar: {n} 檔 → 決策日改用今日盤中訊號")
    # features=v3.features,內部 oh() 讀 v3 模組的 _OH;灌即時必須設到 v3._OH 才生效
    _v5._OH = OH
    _v5.v3._OH = OH
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}

    d = date or max(OH["0050"])
    if d not in twii_feat:
        raise SystemExit(f"❌ 決策日 {d} 無 0050 資料")
    # 即時訊號模式 fail-closed:即時 bar 沒灌成功(決策日≠今天)或歷史過期 → 中止,
    # 絕不拿舊資料去下單/「名單空就出清全部」(這正是 6/18 誤賣廣達的根因)。
    if require_today:
        _hist = [x for x in OH["0050"] if x != TODAY]
        _hmax = max(_hist) if _hist else None
        if d != TODAY:
            m = (f"🛑 TW v5 中止(即時資料異常):決策日={d}≠今天{TODAY},即時報價灌入失敗"
                 f"(開盤瞬間常抓不到)→ 未送任何委託、未動倉位。請稍後手動重跑;多次失敗檢查永豐/網路。")
            log(m); notify(m)
            raise SystemExit(2)
        if _hmax and (datetime.fromisoformat(TODAY) - datetime.fromisoformat(_hmax)).days > 5:
            m = (f"🛑 TW v5 中止(歷史資料過期):最新只到 {_hmax}(今天{TODAY}),FinMind 增量補資料可能失敗"
                 f"→ 未動倉位。請用 --refresh 強制更新後重跑(若逢長假休市可忽略)。")
            log(m); notify(m)
            raise SystemExit(2)
    bull = bool(twii_feat[d].get("close") and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
    # 下檔保險:0050 收盤跌破 MA60 = 下降趨勢 → 當天曝險砍半(見頂部 PROTECT_*)
    _mn = twii_feat[d].get(f"ma{PROTECT_MA}")
    defensive = bool(_mn and not math.isnan(_mn) and twii_feat[d].get("close")
                     and twii_feat[d]["close"] < _mn)
    ir = twii_feat[d].get("ret20")
    # 資金優先序:--capital 指定 > 手動現金池(capital.json) > DCA 自動
    if capital is None:
        override = None
        if CAP_FILE.exists():
            try:
                v = json.loads(CAP_FILE.read_text(encoding="utf-8")).get("capital")
                override = float(v) if v else None
            except Exception as e:
                log(f"  現金池讀取失敗: {e}(改用 DCA)")
        if override is not None:
            capital = override
            log(f"資金(手動現金池):可部署 {capital:,.0f} TWD(ledger.py cash 設定;cash auto 可恢復DCA)")
        else:
            n_td = sum(1 for x in sorted(OH["0050"]) if CAP_START <= x <= d)   # 起算日以來的交易日數
            capital = min(CAP_INITIAL + CAP_DAILY * max(0, n_td - 1), CAP_MAX)
            log(f"資金(DCA):起算 {CAP_START} 以來第 {max(1, n_td)} 個交易日 → 可部署 {capital:,.0f} TWD(封頂{CAP_MAX:,.0f})")
    log(f"決策日 {d}｜大盤 {'多頭(站上20MA)→ 打H動能' if bull else '空頭(跌破20MA)→ 打反彈'}"
        + (f"｜🛡️下檔保險:跌破MA{PROTECT_MA}→{'完全空手' if PROTECT_SCALE<=0 else f'曝險×{PROTECT_SCALE:.0%}'}" if defensive else ""))

    vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                   if d in feats.get(c, {}) and feats[c][d].get("turn", 0) > 0),
                  key=lambda x: x[1])
    tp = {c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    # 手動觀察加減分:override(試算用)優先;否則讀 manual_scores.json,只取在決策日 d 仍未到期(until>=d)的
    if manual_override is not None:
        mscores = {c: float(b) for c, b in manual_override.items()}
    else:
        mscores = {}
        if MANUAL_FILE.exists():
            try:
                for c, e in json.loads(MANUAL_FILE.read_text(encoding="utf-8")).items():
                    if str(e.get("until", "")) >= d:        # 到期(d>until)自動失效
                        mscores[c] = float(e.get("bonus", 0))
            except Exception as ex:
                log(f"  手動加分讀取失敗: {ex}")

    whb, wrb, whs, wrs = weights
    scored = []
    raw_map: dict[str, float] = {}; bon_map: dict[str, float] = {}
    for c in codes:
        f = feats.get(c, {})
        if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
            continue
        hh = h_score(_factors(f[d], ir), tp.get(c, 0.5))
        closes = [OH[c][x]["close"] for x in sorted(OH[c]) if x <= d]
        sig = rebound_signal(closes, turns.get(c, 0.0))
        rb = sig["score"] * 100 if sig.get("fired") else 0.0
        sc = max(hh * whb, rb * wrb) if bull else max(hh * whs, rb * wrs)
        bonus = mscores.get(c, 0.0)
        size_score = sc + bonus              # 加分後分數(sizing/曝險用;bonus 加在分上、不被×INC)
        if size_score > 0:                   # bonus 可把 sc=0 的觀察股拉進候選
            scored.append((size_score, c)); raw_map[c] = sc; bon_map[c] = bonus
    # 排名:持股「原始分」×INC 黏著 + 手動bonus(flat,絕不被×INC);sizing/曝險用 size_score(it[0]=原始+bonus)
    rk = lambda it: raw_map[it[1]] * (INC if it[1] in held else 1.0) + bon_map[it[1]]
    scored = sorted(scored, key=rk, reverse=True)

    sel = scored[:MAX_SIG]
    if len(scored) > MAX_SIG and rk(scored[MAX_SIG]) >= rk(scored[MAX_SIG - 1]) * TIE:
        sel = scored[:MAX_SIG + 1]

    targets: dict[str, float] = {}
    ref_price: dict[str, float] = {}
    if sel:
        avg = sum(s for s, _ in sel) / len(sel) / 100
        expo = min(EXPO_CAP, max(EXPO_FLOOR, avg))
        if defensive:
            if PROTECT_SCALE <= 0:
                sel = []; expo = 0.0          # A2:下降趨勢完全空手(名單清空→賣腿exit_only全出清、買腿不買)
            else:
                expo *= PROTECT_SCALE         # 半倉:減曝險(需賣腿trim,見PROTECT_SCALE註)
        ssum = sum(s for s, _ in sel)
        for s, c in sel:
            targets[c] = round(capital * expo * (s / ssum))
            ref_price[c] = OH[c][max(x for x in OH[c] if x <= d)]["close"]
    else:
        expo = 0.0

    # ranked = 完整排名(給「主名單買不到時往下補備選」用),per-stock 預算 = capital*expo/檔數
    per_slot = round(capital * expo / max(1, len(sel))) if sel else 0
    return {"date": d, "bull": bull, "defensive": defensive, "expo": expo, "targets": targets,
            "sel": sel, "names": names, "ref_price": ref_price, "capital": capital,
            "ranked": scored[:15], "per_slot": per_slot,
            "bonus": {c: bon_map.get(c, 0.0) for c in bon_map if bon_map.get(c, 0.0)},
            "rank_of": {c: (i + 1, s) for i, (s, c) in enumerate(scored)}}   # 全排名查找(試算用):{code:(名次,加分後分數)}


# ── Shioaji 登入 ─────────────────────────────────────────────────────────────
def shioaji_login(simulation: bool, need_ca: bool):
    api_key = os.environ.get("SINOPAC_APIKEY")
    secret_key = os.environ.get("SINOPAC_SECRETKEY")
    ca_path = os.environ.get("SINOPAC_CA_PATH")
    ca_passwd = os.environ.get("SINOPAC_CA_PASSWORD")
    if not api_key or not secret_key:
        log("ERROR: 缺 SINOPAC_APIKEY / SINOPAC_SECRETKEY"); return None, None

    import shioaji as sj
    api = sj.Shioaji(simulation=simulation)
    log("登入 ...")
    accounts = api.login(api_key=api_key, secret_key=secret_key)
    stock_acc = getattr(api, "stock_account", None) or next(
        (a for a in accounts if str(getattr(a, "account_type", "")) in ("S", "AccountType.Stock")), None)
    if stock_acc is None:
        log("ERROR: 找不到證券帳戶"); return None, None
    log(f"帳戶 {stock_acc.account_id} ({stock_acc.username})")

    if need_ca:
        if not ca_path or not ca_passwd:
            log("ERROR: 送單需 SINOPAC_CA_PATH / SINOPAC_CA_PASSWORD"); return None, None
        cap = Path(ca_path); cap = cap if cap.is_absolute() else ROOT / cap
        time.sleep(SPACING_SEC)
        if not api.activate_ca(ca_path=str(cap), ca_passwd=ca_passwd):
            log("ERROR: 憑證啟用失敗"); return None, None
        log("憑證 OK")

    log("下載商品檔 ...")
    try:
        api.fetch_contracts(contract_download=True)
    except Exception as e:
        log(f"  fetch_contracts cb(已知無害): {e}")
    time.sleep(SPACING_SEC)
    return api, stock_acc


def positions_detail(api, stock_acc) -> list[dict]:
    """回持倉明細 [{code, qty, cost(均價), last(現價), pnl}]。抓不到當空倉。"""
    out: list[dict] = []
    try:
        for p in (api.list_positions(stock_acc) or []):
            qty = int(getattr(p, "quantity", 0) or 0)
            if qty == 0:
                continue
            cost = float(getattr(p, "price", 0) or 0)        # 平均成本
            last = float(getattr(p, "last_price", 0) or 0) or cost   # 現價(取不到退回成本)
            out.append({"code": str(p.code), "qty": qty, "cost": cost,
                        "last": last, "pnl": float(getattr(p, "pnl", 0) or 0)})
    except Exception as e:
        log(f"  WARN list_positions: {e}(當作空倉)")
    return out


def cur_positions(api, stock_acc) -> dict[str, int]:
    return {p["code"]: p["qty"] for p in positions_detail(api, stock_acc)}


def read_ledger() -> dict:
    """讀手動帳本 data/positions.json:{code: {qty, cost}}。不存在/壞掉回空。"""
    if LEDGER.exists():
        try:
            return json.loads(LEDGER.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"  WARN 帳本讀取失敗: {e}(當作空倉)")
    return {}


def ledger_positions(api) -> list[dict]:
    """手動帳本 → 持倉明細,現價用永豐即時 snapshot(取不到退回成本)。"""
    out: list[dict] = []
    for code, v in read_ledger().items():
        qty = int(v.get("qty", 0) or 0)
        cost = float(v.get("cost", 0) or 0)
        if qty <= 0:
            continue
        last = 0.0
        if api is not None:
            last, _ = snapshot_price(api, code)
        if last <= 0:
            last = cost
        out.append({"code": str(code), "qty": qty, "cost": cost,
                    "last": last, "pnl": (last - cost) * qty})
    return out


def snapshot_price(api, code: str) -> tuple[float, float]:
    """回 (現價, 昨收)。抓不到回 (0,0)。"""
    ct = api.Contracts.Stocks.get(code) if hasattr(api.Contracts.Stocks, "get") else api.Contracts.Stocks[code]
    if ct is None:
        return 0.0, 0.0
    px = 0.0
    ref = float(getattr(ct, "reference", 0) or 0)
    try:
        snap = api.snapshots([ct])
        if snap:
            px = float(snap[0].close or snap[0].sell_price or ref)
    except Exception:
        pass
    if not px:
        px = ref
    return px, ref


def place(api, stock_acc, code, action, qty_shares, limit_px):
    """qty_shares = 股數。ODD_LOT=True 用盤中零股(IntradayOdd),否則整股(quantity=張)。"""
    import shioaji as sj
    ct = api.Contracts.Stocks[code]
    if ODD_LOT:
        lot = sj.StockOrderLot.IntradayOdd
        quantity = qty_shares                  # 零股:數量=股
    else:
        lot = sj.StockOrderLot.Common
        quantity = qty_shares // LOT_SHARES     # 整股:數量=張
    order = sj.StockOrder(                       # 原 api.Order 已棄用,改 sj.StockOrder(參數相同)
        price=limit_px, quantity=quantity,
        action=sj.Action.Buy if action == "Buy" else sj.Action.Sell,
        price_type=sj.StockPriceType.LMT, order_type=sj.OrderType.ROD,
        order_lot=lot, account=stock_acc)
    time.sleep(SPACING_SEC)
    trade = api.place_order(contract=ct, order=order)
    sid = getattr(getattr(trade, "status", None), "id", "?")
    st = getattr(getattr(trade, "status", None), "status", "?")
    return sid, st


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, choices=["buy", "sell"],
                    help="buy=收盤前買進腿｜sell=開盤出清掉出名單腿")
    ap.add_argument("--capital", type=float, default=None,
                    help="可部署總資金 TWD(buy leg 用)。不給=用 DCA 自動算(起始15000/每交易日+1000/5萬封頂)")
    ap.add_argument("--date", default=None, help="決策日(預設=資料最新一天)")
    ap.add_argument("--refresh", action="store_true", help="先強制更新行情")
    ap.add_argument("--sim", action="store_true", help="模擬下單(simulation=True)")
    ap.add_argument("--live", action="store_true", help="真實下單,需 SHIOAJI_LIVE_CONFIRM=YES")
    ap.add_argument("--common", action="store_true", help="改用整股(1張=1000股);預設盤中零股")
    ap.add_argument("--weights", default=None,
                    help="雙引擎權重 多頭H,多頭reb,空頭H,空頭reb。不給→讀 weights.json(ledger.py weights 設),都沒有→B純切 1,0,0,1.5")
    ap.add_argument("--live-signal", action="store_true",
                    help="用今日盤中即時價算今日訊號(避免追高已反彈的股);否則用最新收盤(昨收)")
    ap.add_argument("--ledger", action="store_true",
                    help="持有/損益改讀 data/positions.json(手動帳本),且強制不下單(純訊號)。元大手動交易用。")
    args = ap.parse_args()

    global ODD_LOT
    ODD_LOT = not args.common
    # 權重優先序:--weights 指定 > weights.json(ledger.py weights) > 預設 B純切
    if args.weights:
        W = tuple(float(x) for x in args.weights.split(","))
    elif WEIGHTS_FILE.exists():
        try:
            W = tuple(float(x) for x in json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))["weights"])
        except Exception:
            W = (1.0, 0.0, 0.0, 1.5)
    else:
        W = (1.0, 0.0, 0.0, 1.5)

    mode = "live" if args.live else ("sim" if args.sim else "dry-run")
    place_orders = mode in ("sim", "live")
    if args.ledger:
        place_orders = False                # 帳本=純訊號模式,絕不下單(你在元大手動下)
    simulation = (mode != "live")
    mode_disp = "訊號·手動帳本" if args.ledger else mode

    # ── 安全閘 ──
    if KILL_SWITCH.exists():
        log(f"🛑 KILL_SWITCH 存在 ({KILL_SWITCH}) → 中止。刪掉它才會執行。"); return 10
    if mode == "live" and os.environ.get("SHIOAJI_LIVE_CONFIRM") != "YES":
        log("🛑 --live 需要 SHIOAJI_LIVE_CONFIRM=YES 才放行(防誤觸真錢)。中止。"); return 11

    load_dotenv(ROOT / ".env", override=False)

    # ── 1) 登入(dry-run 也登入,才能比對真實持倉/取即時價)──
    api, stock_acc = shioaji_login(simulation=simulation, need_ca=place_orders)
    if api is None:
        log("⚠️ 未登入 Shioaji(缺憑證?)。改用收盤資料純策略預覽,未比對實際持倉、未下單。")
        picks = compute_picks(args.capital, date=args.date, refresh=args.refresh, weights=W)
        for s, c in picks["sel"]:
            log(f"  {c} {picks['names'].get(c,c)[:6]:<7} 分數{s:.0f} → 目標 {picks['targets'][c]:,.0f} TWD")
        return 1

    # ── 2) 先讀持倉(供 incumbent 黏著用)──
    pos = ledger_positions(api) if args.ledger else positions_detail(api, stock_acc)
    cur = {p["code"]: p["qty"] for p in pos}
    log(f"{'帳本' if args.ledger else '帳戶'}持倉 {len(cur)} 檔: {cur}")

    # ── 3) 算今日名單(--live-signal 用今日盤中即時訊號;持股分數 ×INC 黏著)──
    live_bars = fetch_all_snapshots(api) if args.live_signal else None
    picks = compute_picks(args.capital, date=args.date, refresh=args.refresh,
                          live_bars=live_bars, weights=W, held=set(cur),
                          require_today=args.live_signal)
    targets, names, sel = picks["targets"], picks["names"], picks["sel"]
    log(f"決策日 {picks['date']}｜今日名單({'多頭H動能' if picks['bull'] else '空頭反彈'}, 曝險 {picks['expo']:.0%}):")
    _bon = picks.get("bonus", {})
    for s, c in sel:
        tags = ('(持股×'+str(INC)+')' if c in cur else '') + (f'(+{_bon[c]:.0f}觀察)' if _bon.get(c) else '')
        log(f"  {c} {names.get(c,c)[:6]:<7} 分數{s:.0f}{tags} → 目標 {targets[c]:,.0f} TWD")
    if not sel:
        log("今日無訊號 → 名單空(buy 不買;sell 會出清全部持倉)。")

    unit = "股" if ODD_LOT else "張"
    min_qty = 1 if ODD_LOT else LOT_SHARES     # 最小可下數量(股)

    # ── 3) 算單(plan 存「股數」)──
    plan = []  # (code, action, qty_shares, limit_px, reason)
    if args.leg == "buy":
        # 名單內未達目標 → 買到目標(只買、不減碼;exit_only 的買腿)
        total_buy = 0.0
        bought: set[str] = set()
        n_limitup = 0   # 主名單「鎖漲停買不到」的檔數(只有這個才啟動備選)

        def try_buy(code, tgt_val, tag):
            """嘗試買一檔。成功 append 並回 True;否則回原因字串。"""
            nonlocal total_buy
            if code in bought:
                return "已買"
            px, ref = snapshot_price(api, code)
            if px <= 0:
                return "取價失敗"
            cur_val = cur.get(code, 0) * px
            buy_val = min(tgt_val - cur_val, DAILY_BUY_CAP_TWD)
            if buy_val < MIN_ORDER_TWD:
                return "已達標"
            if ref > 0 and px / ref - 1 >= LIMIT_UP_GUARD:
                return f"接近漲停({px/ref-1:+.1%})買不到"
            limit_px = align_tick(px * (1 + LIMIT_BUFFER), "Buy")
            qty = round_lots(int(buy_val / limit_px)) if not ODD_LOT else int(buy_val / limit_px)
            if qty < min_qty:
                return f"買得起 {qty} 股 < 最小 {min_qty}{unit}(價{px})"
            cost = qty * limit_px
            if total_buy + cost > MAX_TOTAL_BUY_TWD:
                return "超過單次總買入上限"
            total_buy += cost; bought.add(code)
            plan.append((code, "Buy", qty, limit_px, tag))
            return True

        # ① 主名單
        for s, code in sorted(sel, key=lambda x: -x[0]):
            tgt = min(targets[code], MAX_POSITION_TWD)
            r = try_buy(code, tgt, f"目標{tgt:.0f}")
            if r is not True:
                log(f"  跳過 {code}:{r}")
                if "漲停" in r:
                    n_limitup += 1
        # ② 備選:只有「好票被鎖漲停」才啟動;且備選分數須 ≥ 正取最低分 × ALT_MIN_SCORE_PCT
        if n_limitup > 0 and sel:
            sel_codes = {c for _, c in sel}
            ref = min(s for s, _ in sel)                 # 正取裡最低的分數
            bar = ref * ALT_MIN_SCORE_PCT
            per = min(picks.get("per_slot") or MIN_ORDER_TWD, MAX_POSITION_TWD)
            need = n_limitup
            for s, code in picks.get("ranked", []):
                if need <= 0:
                    break
                if code in sel_codes or code in bought:
                    continue
                if s < bar:    # ranked 已由高到低排序 → 一旦低於門檻,後面更低,直接停
                    log(f"  備選止步:{names.get(code, code)} 分{s:.0f} < 門檻{bar:.0f}"
                        f"(正取最低{ref:.0f}×{ALT_MIN_SCORE_PCT:.0%})→ 不補、留現金")
                    break
                if try_buy(code, per, f"🔻備選 分{s:.0f}") is True:
                    need -= 1
                    log(f"  備選補上 {code} {names.get(code, code)} 分{s:.0f}")
            if any("備選" in str(p[4]) for p in plan):
                picks["warn"] = (f"⚠️ 前{MAX_SIG}名有 {n_limitup} 檔鎖漲停買不到,"
                                 f"下面🔻備選為分數≥正取{ALT_MIN_SCORE_PCT:.0%}的替補,請自行斟酌")
        log(f"買進腿:總買入 ~{total_buy:,.0f} TWD")
    else:  # sell leg:持倉中「掉出名單」的全部出場(exit_only)
        keep = set(targets.keys())
        for code, qty in cur.items():
            if code in keep or qty < min_qty:
                continue
            px, ref = snapshot_price(api, code)
            if px <= 0:
                log(f"  跳過 {code}:取價失敗"); continue
            limit_px = align_tick(px * (1 - LIMIT_BUFFER), "Sell")
            sell_qty = qty if ODD_LOT else round_lots(qty)
            if sell_qty < min_qty:
                continue
            plan.append((code, "Sell", sell_qty, limit_px, f"掉出名單→出清 {qty}股"))
        n_exit = len(plan)
        # 下檔保險:防禦日(0050<MA60),把「還在名單但部位超過半倉目標」的減碼到目標(讓半倉真生效;非防禦日維持exit_only不減碼)
        if picks.get("defensive"):
            for code, qty in cur.items():
                if code not in keep or qty < min_qty:
                    continue
                px, ref = snapshot_price(api, code)
                if px <= 0:
                    continue
                tgt = min(targets.get(code, 0.0), MAX_POSITION_TWD)
                excess = qty * px - tgt
                if excess < MIN_ORDER_TWD:    # 超出不多 → 不動(避免碎單)
                    continue
                limit_px = align_tick(px * (1 - LIMIT_BUFFER), "Sell")
                trim_qty = round_lots(int(excess / limit_px)) if not ODD_LOT else int(excess / limit_px)
                max_sell = qty if ODD_LOT else round_lots(qty)
                trim_qty = min(trim_qty, max_sell)
                if trim_qty < min_qty:
                    continue
                plan.append((code, "Sell", trim_qty, limit_px, f"🛡️防禦減碼→半倉目標{tgt:,.0f}"))
            log(f"賣出腿:出清掉出名單 {n_exit} 檔 + 🛡️防禦減碼 {len(plan)-n_exit} 檔(0050<MA{PROTECT_MA})")
        else:
            log(f"賣出腿:出清掉出名單 {len(plan)} 檔")

    plan = plan[:MAX_ORDERS]

    # ── 4) 印計畫 ──
    log("=" * 60)
    log(f"下單計畫({len(plan)} 筆,mode={mode},{'盤中零股' if ODD_LOT else '整股'}):")
    for code, act, qty, lp, why in plan:
        ct = api.Contracts.Stocks[code]
        disp = f"{qty}股" if ODD_LOT else f"{qty//LOT_SHARES}張"
        log(f"  {act:<4} {code} {getattr(ct,'name','')[:5]:<6} {disp} @ 限價{lp}  ({why})")
    if not plan:
        log("  無需調整。"); notify(fmt_plan(args.leg, mode_disp, picks, plan, pos)); api.logout(); return 0

    # ── 5) 送單(dry-run / ledger 不送)──
    if not place_orders:
        if args.ledger:
            log("📒 帳本訊號模式:以上是「你要在元大手動下的單」,本程式不送任何委託。")
        else:
            log("🟡 dry-run:以上「不會送出」。確認無誤後加 --sim 模擬、或 --live 真實。")
        notify(fmt_plan(args.leg, mode_disp, picks, plan, pos))
        api.logout(); return 0

    log(f"🔴 開始送單(mode={mode})...")
    logf = ROOT / "reports" / f"v5_orders_{args.leg}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for code, act, qty, lp, why in plan:
        disp = f"{qty}股" if ODD_LOT else f"{qty//LOT_SHARES}張"
        try:
            sid, st = place(api, stock_acc, code, act, qty, lp)
            log(f"  ✅ {act} {code} {disp} → id={sid} status={st}")
            records.append({"code": code, "action": act, "shares": qty, "price": lp,
                            "id": str(sid), "status": str(st)})
        except Exception as e:
            log(f"  ❌ {act} {code} 失敗: {e}")
            records.append({"code": code, "action": act, "shares": qty, "price": lp, "error": str(e)})
    logf.write_text(json.dumps({"leg": args.leg, "mode": mode, "time": datetime.now().isoformat(),
                                "orders": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"委託紀錄 → {logf}")
    notify(fmt_plan(args.leg, mode_disp, picks, plan, pos, records))
    api.logout()
    log("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
