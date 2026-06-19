"""Agent 工具庫(Phase 2)——把系統操作整理成「分級、有 schema」的工具集。

這是之後 Phase 3 GPT agent 唯一能呼叫的東西(LLM 不給 shell,只能在這份白名單裡選工具+填參數)。
每個工具標 tier:
  - read  : 唯讀/查詢/測試。純聊天也能呼叫(不改任何狀態)。
  - mutate: 會改設定(manual_scores / weights / capital)。**run_tool 預設拒絕,必須 allow_mutate=True**
            (Phase 3 只有在你下「設定指令」+ 看過試算 + 明確確認後才會帶 allow_mutate=True)。
  - 下單(trade)不在此庫:依設計,agent 永遠不能下真實單,維持手動 --live。

CLI:
  uv run python scripts/agent_tools.py list                         # 列出所有工具(分類)
  uv run python scripts/agent_tools.py schema                       # 印 OpenAI function-tools schema(給 agent 用)
  uv run python scripts/agent_tools.py run query_score '{"codes":["2330","2382"]}'
  uv run python scripts/agent_tools.py run preview_bonus '{"code":"2330","bonus":20,"days":5}'
  uv run python scripts/agent_tools.py run set_manual_score '{"code":"2330","bonus":20,"days":5}' --allow-mutate
"""
from __future__ import annotations
import sys, json, io, contextlib, importlib.util
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR

# 復用既有程式(不重寫邏輯)
_s = importlib.util.spec_from_file_location("v5live", ROOT / "scripts/live_dual_v5_trade.py")
v5live = importlib.util.module_from_spec(_s); _s.loader.exec_module(v5live)
_m = importlib.util.spec_from_file_location("mscore", ROOT / "scripts/manual_score.py")
mscore = importlib.util.module_from_spec(_m); _m.loader.exec_module(mscore)
_e = importlib.util.spec_from_file_location("ec", ROOT / "scripts/exp_60d_entry_compare.py")
ec = importlib.util.module_from_spec(_e); _e.loader.exec_module(ec)
sim5 = ec.sim_buyclose_sellopen        # 真實 ⑤ 模擬(每日換股損益用)

# agent 進程開記憶體快取:同股的還原OHLCV只算一次,大幅加速重複的 compute_picks(實盤不 import 本檔→不受影響)
import tw_stock_agent.tools.finmind_client as _fc
_fc.MEMOIZE = True

UNIV = DATA_DIR / "base_universe.json"


def _silent_picks(**kw):
    """跑 compute_picks 但不噴 log。"""
    old = v5live.log; v5live.log = lambda *a, **k: None
    try:
        return v5live.compute_picks(**kw)
    finally:
        v5live.log = old


_NAMES_CACHE: dict = {}


def _names() -> dict:
    """全市場 {代號:名稱}(FinMind 3000+ 檔),universe 名字覆蓋。同進程快取。"""
    if not _NAMES_CACHE:
        try:
            _NAMES_CACHE.update(_fc.get_stock_names())
        except Exception:
            pass
        _NAMES_CACHE.update({c: v.get("name", c) for c, v in json.loads(UNIV.read_text(encoding="utf-8")).items()})
    return _NAMES_CACHE


def _to_code(x) -> str:
    """把『可能是中文名』的輸入轉成代號(LLM 常把名字當代號傳)。是代號就原樣回。"""
    x = str(x).strip()
    nm = _names()
    if x in nm:                      # 已是代號
        return x
    for c, n in nm.items():          # 名稱→代號
        if n == x:
            return c
    return x                         # 找不到就原樣(讓後續報「資料不足」)


def _holdings() -> set:
    """帳本持倉代號(查分數時當 held,反映 ×INC 黏著=持股實際會不會被續抱)。"""
    p = DATA_DIR / "positions.json"
    if not p.exists():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")).keys())


_PICKS_CACHE: dict = {}     # 同一進程內快取「最新日、給定held、讀檔manual」的選股結果,免每次重算116檔


def _load_weights() -> tuple:
    p = DATA_DIR / "weights.json"
    if p.exists():
        try:
            return tuple(float(x) for x in json.loads(p.read_text(encoding="utf-8"))["weights"])
        except Exception:
            pass
    return (1.0, 0.0, 0.0, 1.5)     # 預設 B純切


def _picks_cached(date_str: str, held: set):
    key = (date_str, frozenset(held))
    if key not in _PICKS_CACHE:
        _PICKS_CACHE[key] = _silent_picks(date=date_str, held=set(held), weights=_load_weights())
    return _PICKS_CACHE[key]


def _invalidate_cache():
    _PICKS_CACHE.clear()    # 任何 mutate(改加分/權重/資金)後呼叫,免用到舊的選股結果


# ─────────────────────── READ(查詢/測試)───────────────────────
def query_score(args: dict) -> str:
    codes = args["codes"] if isinstance(args.get("codes"), list) else [args.get("codes")]
    codes = [_to_code(c) for c in codes]              # 中文名自動轉代號
    held = _holdings()                                # 持股→套 ×INC 黏著,反映「會不會被續抱」
    oh = v5live.get_daily_ohlcv("0050"); d = max(oh)
    uni = set(json.loads(UNIV.read_text(encoding="utf-8")))
    extra = [c for c in codes if c not in uni]        # 池外股要一起算分才查得到
    if extra:
        pk = _silent_picks(date=d, held=held, weights=_load_weights(), extra_codes=extra)
    else:
        pk = _picks_cached(d, held)                   # 快取:同進程重複查不重算
    ro, sel = pk["rank_of"], [c for _, c in pk["sel"]]
    NM = _names()
    out = [f"決策日 {d}（大盤{'多頭' if pk['bull'] else '空頭'}）今日選股: " +
           ", ".join(NM.get(c, c) for _, c in pk["sel"])]
    for c in codes:
        nm = NM.get(c, c)
        tags = ("（持股×1.75黏著）" if c in held else "") + ("（池外觀察）" if c in extra else "")
        if c in ro:
            rk, sc = ro[c]
            st = ("✅會被選/續抱" if c in sel else ("⚠️未進前選（持股可能被賣）" if c in held else "未選上"))
            out.append(f"  {c} {nm}: 分數{sc:.0f}、排名{rk}、{st}{tags}")
        else:
            out.append(f"  {c} {nm}: 分數0（不在動能結構、未進候選）{tags}")
    return "\n".join(out)


def query_universe(args: dict) -> str:
    u = json.loads(UNIV.read_text(encoding="utf-8"))
    act = args.get("action", "count")
    if act == "check":
        c = str(args["code"]); return f"{c} {'在' if c in u else '不在'} universe" + (f"（{u[c]['name']}）" if c in u else "")
    if act == "list":
        return f"universe {len(u)} 檔: " + ", ".join(f"{c}{v['name']}" for c, v in u.items())
    return f"universe 共 {len(u)} 檔"


def query_positions(args: dict) -> str:
    p = DATA_DIR / "positions.json"
    if not p.exists(): return "帳本無持倉"
    pos = json.loads(p.read_text(encoding="utf-8"))
    if not pos: return "帳本無持倉"
    nm = _names(); lines, tot = [], 0.0
    for c, v in pos.items():
        qty, cost = int(v.get("qty", 0)), float(v.get("cost", 0))   # 帳本欄位是 cost(每股均價)
        val = qty * cost; tot += val
        lines.append(f"  {c} {nm.get(c, c)}: {qty}股 均價{cost:.2f}（成本{val:,.0f}）")
    return "帳本持倉:\n" + "\n".join(lines) + f"\n總成本 {tot:,.0f} TWD"


def query_manual_scores(args: dict) -> str:
    m = mscore._load()
    if not m: return "目前沒有任何手動加減分"
    today = date.today().isoformat()
    return "手動加減分:\n" + "\n".join(
        f"  {c} {e.get('bonus',0):+.0f} 到{e.get('until','')} {'(生效)' if str(e.get('until',''))>=today else '(已過期)'} {e.get('note','')}"
        for c, e in m.items())


def query_config(args: dict) -> str:
    cap = json.loads((DATA_DIR/"capital.json").read_text()).get("capital") if (DATA_DIR/"capital.json").exists() else None
    w = json.loads((DATA_DIR/"weights.json").read_text()).get("weights") if (DATA_DIR/"weights.json").exists() else None
    return (f"現金池: {'手動 '+format(cap,',.0f')+' TWD' if cap else 'DCA 自動(15000+1000/日)'}\n"
            f"雙引擎權重: {w if w else '預設 B純切 [1,0,0,1.5]'}")


def resolve_stock(args: dict) -> str:
    """名稱(可能打錯/簡稱)→代號。給 agent 在使用者用名字提到股票時先確認。"""
    import difflib
    q = str(args.get("query", "")).strip()
    if not q:
        return "請給股票名稱或代號"
    NM = _names()
    if q in NM:
        return f"{q} = {NM[q]}（代號直接命中）"
    NM = {c: n for c, n in NM.items() if not any(x in n for x in ("購", "售", "期", "權證"))}  # 濾掉權證/期貨噪音
    rev: dict = {}
    for c, n in NM.items():
        rev.setdefault(n, c)
    if q in rev:
        return f"{rev[q]} {q}（名稱完全符合）"
    subs = [(c, n) for c, n in NM.items() if q in n][:6]
    seen = {c for c, _ in subs}
    fz = [(rev[n], n) for n in difflib.get_close_matches(q, list(rev.keys()), n=6, cutoff=0.5) if rev[n] not in seen]
    cands = (subs + fz)[:6]
    if not cands:
        return f"找不到符合「{q}」的股票,請確認名稱或直接給代號。"
    return f"「{q}」可能是(請跟使用者確認是哪一個):\n" + "\n".join(f"  {c} {n}" for c, n in cands)


def preview_bonus(args: dict) -> str:
    """5日試算:加分後 vs 原本、會不會被選到。唯讀,不寫入。"""
    code, bonus = str(args["code"]), float(args["bonus"]); days = int(args.get("days", 5))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mscore._run_preview(code, bonus, days)
    return buf.getvalue().strip()


def bonus_sweep(args: dict) -> str:
    """算某股要加幾分才會進過去N天的選股,並列出加X分→入選幾天。
       快:每天分數只算一次(N次),掃 bonus 是純算術。用乾淨排名(held=空,同 preview)。"""
    code = str(args["code"]); days = int(args.get("days", 5))
    oh = v5live.get_daily_ohlcv("0050"); alld = sorted(oh)[-days:]
    MAXSIG = v5live.MAX_SIG; nm = _names(); name = nm.get(code, code)
    is_extra = code not in set(json.loads(UNIV.read_text(encoding="utf-8")))   # 池外股要帶 extra_codes 才算得到
    rows = []
    for d in alld:
        ro = (_silent_picks(date=d, held=set(), extra_codes=[code]) if is_extra
              else _picks_cached(d, set()))["rank_of"]      # 當天所有股的原始分
        base = ro.get(code, (None, 0.0))[1]
        others = sorted((s for c, (r, s) in ro.items() if c != code), reverse=True)
        cutoff = others[MAXSIG - 1] if len(others) >= MAXSIG else (others[-1] if others else 0.0)
        rows.append((d, base, max(0.0, cutoff - base)))    # (日, 原始分, 入選最低門檻)
    thrs = sorted(t for _, _, t in rows)
    out = [f"{code} {name} — 要進過去{days}天選股(取前{MAXSIG}名)需加幾分:",
           "  各日原始分/門檻: " + "  ".join(f"{d[5:]}({b:.0f},需+{t:.0f})" for d, b, t in rows),
           "  加分→入選天數:"]
    for k in range(1, days + 1):
        out.append(f"    加 +{thrs[k-1]:.0f}分 → 至少 {k}/{days} 天入選")
    out.append(f"  (門檻=當天第{MAXSIG}名分數−{name}當天原始分;加分不被×1.75。已在名單的天門檻=0)")
    return "\n".join(out)


def _ret_over(code: str, days: int):
    """某股過去 days 個交易日的報酬(用還原收盤價,含息)。"""
    oh = v5live.get_daily_ohlcv(code)
    cl = [oh[d]["close"] for d in sorted(oh)]
    if len(cl) < days + 1:
        return None
    return cl[-1] / cl[-(days + 1)] - 1


def pnl_history(args: dict) -> str:
    """歷史N交易日「照買並持有」的損益試算。可用四種選法:
       mode=holdings 你的持股 / mode=list 今日策略名單(可帶 bonus 試算加減分後的名單) /
       mode=stocks 指定股(weight=equal 等權 或 score 分數加權;單一檔=100%)。"""
    mode = args.get("mode", "list"); days = int(args.get("days", 5))
    # 策略名單預設「分數加權」(對齊真實配重:targets∝分數);其他模式預設等權
    weight = args.get("weight") or ("score" if mode == "list" else "equal")
    nm = _names(); cap = args.get("capital")

    # ── 持倉模式:你的『實際持倉』真實未實現損益(現價 vs 你的成本),點時間、非N天假設 ──
    if mode == "holdings":
        p = DATA_DIR / "positions.json"
        pos = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if not pos:
            return "【持倉模式】你帳本目前沒有持倉。"
        lines = []; tot_cost = tot_mv = 0.0
        for c, v in pos.items():
            qty = float(v.get("qty", 0)); cost = float(v.get("cost", 0))
            o = v5live.get_daily_ohlcv(c); ds = sorted(o)
            if not ds:
                lines.append(f"  {c} {nm.get(c, c)}: 取價失敗"); continue
            last = o[ds[-1]]["close"]; prev = o[ds[-2]]["close"] if len(ds) > 1 else last
            mv = qty * last; cst = qty * cost; tot_mv += mv; tot_cost += cst
            day = (last / prev - 1) * 100 if prev else 0.0
            pnl = (last - cost) * qty; pct = (last / cost - 1) * 100 if cost else 0.0
            lines.append(f"  {c} {nm.get(c, c)}: {qty:g}股 成本{cost:.2f}→現{last:.2f}  損益{pnl:+,.0f}({pct:+.1f}%)  今日{day:+.1f}%")
        tot_pnl = tot_mv - tot_cost
        return ("📊【持倉模式｜你的實際持倉｜現價(最新收盤)vs你的成本｜真實未實現損益】\n" + "\n".join(lines) +
                f"\n→ 總成本 {tot_cost:,.0f} · 市值 {tot_mv:,.0f} · 未實現 {tot_pnl:+,.0f}元 "
                f"({(tot_mv/tot_cost-1)*100 if tot_cost else 0:+.1f}%)｜含每檔『今日』單日漲跌")
    elif mode == "stocks":
        codes = args["codes"] if isinstance(args.get("codes"), list) else [args.get("codes")]
        codes = [_to_code(c) for c in codes]          # 中文名自動轉代號
        if len(codes) > 1 and weight == "score":
            oh = v5live.get_daily_ohlcv("0050"); pk = _picks_cached(max(oh), _holdings())
            sc = {c: pk["rank_of"].get(c, (0, 0.0))[1] for c in codes}
            tot = sum(sc.values()) or 1.0
            basket = [(c, sc[c] / tot) for c in codes]
            title = f"{len(codes)}檔(分數加權)"
        else:
            basket = [(c, 1.0 / len(codes)) for c in codes]
            title = (f"{codes[0]} 單一檔" if len(codes) == 1 else f"{len(codes)}檔(等權)")
    else:  # list = 今日策略名單(可帶加減分試算)
        weight = "score"        # 策略名單一律分數加權(對齊真實配重 targets∝分數;不接受等權,免誤導)
        oh = v5live.get_daily_ohlcv("0050"); d = max(oh)
        ov = None
        if args.get("bonus"):
            b = args["bonus"]; ov = {str(b["code"]): float(b["bonus"])}
        # list 模式用乾淨排名(held=空):純分數選股,讓加減分試算看得出效果、且與 preview_bonus 一致
        pk = (_silent_picks(date=d, held=set(), weights=_load_weights(), manual_override=ov)
              if ov else _picks_cached(d, set()))
        sel = pk["sel"]
        if not sel:
            return "今日策略名單為空,無法計算"
        if weight == "score":
            tot = sum(s for s, _ in sel) or 1.0
            basket = [(c, s / tot) for s, c in sel]
        else:
            basket = [(c, 1.0 / len(sel)) for _, c in sel]
        title = f"今日策略名單({'分數加權' if weight == 'score' else '等權'}{',含加分試算' if ov else ''})"

    lines, port = [], 0.0
    for c, w in basket:
        r = _ret_over(c, days)
        if r is None:
            lines.append(f"  {c} {nm.get(c, c)}: 資料不足"); continue
        port += w * r
        lines.append(f"  {c} {nm.get(c, c)}: {r*100:+.1f}%（權重{w*100:.0f}%）")
    wlab = "分數加權" if weight == "score" else "等權"
    head = f"📊【假設模式｜不是你的持倉｜過去{days}交易日照買持有｜{wlab}】 {title}:"
    tail = f"\n→ 組合報酬 **{port*100:+.1f}%**"
    if cap:
        tail += f";投入 {float(cap):,.0f} TWD → 損益 **{float(cap)*port:+,.0f} TWD**"
    return (head + "\n" + "\n".join(lines) + tail +
            "\n(假設模式=回測,非你實際下的單;要看你真實持倉損益請用『我的持倉損益』)")


def pnl_strategy_daily(args: dict) -> str:
    """過去N交易日「每日換股」真實⑤模擬損益(逐日收盤買/開盤賣掉出名單,含INC黏著),比靜態試算準。"""
    days = int(args.get("days", 5))
    window = sorted(v5live.get_daily_ohlcv("0050"))[-(days + 1):]
    if len(window) < 2:
        return "資料不足"
    rows, involved = [], set()
    for d in window:
        for c, (rk, s) in _picks_cached(d, set())["rank_of"].items():
            rows.append((d, c, s / 100)); involved.add(c)
    opens, closes, limitup = {}, {}, {}
    for c in involved:
        o = v5live.get_daily_ohlcv(c); ds = sorted(o)
        opens[c] = {d: o[d]["open"] for d in o}; closes[c] = {d: o[d]["close"] for d in o}
        limitup[c] = {d for j, d in enumerate(ds) if j > 0 and o[ds[j-1]]["close"] > 0
                      and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095}
    inc = float(args["inc"]) if args.get("inc") is not None else v5live.INC   # 可調換手/黏著參數(實驗用)
    r = sim5(rows, opens, closes, limitup, incumbent=inc, sell_mode="exit_only", open_buy="none")
    if not r:
        return "模擬無結果(視窗太短)"
    note = f" 【實驗:換手/黏著INC={inc:g},非實盤值{v5live.INC}】" if inc != v5live.INC else ""
    return (f"過去{days}交易日「每日換股」⑤模擬損益({window[0]}~{window[-1]}){note}:\n"
            f"  報酬 {r['ret']:+.1f}% · 損益 {r['total_pnl']:+,.0f}元 · 換手 {r['turn']:.1f}x · 手續費 {r['fees']:,.0f}元\n"
            f"  【回顧模擬:逐日收盤買/開盤賣、含×{inc:g}黏著;非你的實際下單,也不影響實盤設定】")


def order_history(args: dict) -> str:
    """過去N交易日,策略每天「買進/賣出」了什麼(逐日換股明細;帶持股黏著)。"""
    days = int(args.get("days", 5))
    inc = float(args["inc"]) if args.get("inc") is not None else None    # 可調換手/黏著(實驗)
    window = sorted(v5live.get_daily_ohlcv("0050"))[-days:]
    NM = _names(); held = set()
    itag = f"(換手/黏著INC={inc:g})" if inc is not None else ""
    out = [f"過去{days}交易日 逐日換股明細{itag}【回顧模擬:假設這N天就照策略每日換股,非你的實際下單】:"]
    for d in window:
        pk = _silent_picks(date=d, held=held, weights=_load_weights(), inc=inc) if inc is not None else _picks_cached(d, held)
        sel = [c for _, c in pk["sel"]]
        buys = [c for c in sel if c not in held]
        sells = [c for c in held if c not in sel]
        seg = []
        if buys: seg.append("買[" + " ".join(NM.get(c, c) for c in buys) + "]")
        if sells: seg.append("賣[" + " ".join(NM.get(c, c) for c in sells) + "]")
        out.append(f"  {d}: {' '.join(seg) if seg else '無變動'}  → 持{len(sel)}檔")
        held = set(sel)
    out.append(f"  (模擬到今天會持有: {', '.join(NM.get(c, c) for c in held) or '無'};你的實際持倉請用 query_positions)")
    return "\n".join(out)


def compare_inc(args: dict) -> str:
    """比較不同『換手/黏著參數 INC』在過去N天每日換股的結果(報酬/換手/損益)。
       回答『換手權重/黏著 INC 該調多少、該不該這麼黏』。INC越低=換手越高。不影響實盤設定。"""
    days = int(args.get("days", 5))
    incs = args.get("incs") or [1.3, 1.5, 1.75, 2.0]
    window = sorted(v5live.get_daily_ohlcv("0050"))[-(days + 1):]
    if len(window) < 2:
        return "資料不足"
    rows, involved = [], set()
    for d in window:
        for c, (rk, s) in _picks_cached(d, set())["rank_of"].items():
            rows.append((d, c, s / 100)); involved.add(c)
    opens, closes, limitup = {}, {}, {}
    for c in involved:
        o = v5live.get_daily_ohlcv(c); ds = sorted(o)
        opens[c] = {d: o[d]["open"] for d in o}; closes[c] = {d: o[d]["close"] for d in o}
        limitup[c] = {d for j, d in enumerate(ds) if j > 0 and o[ds[j-1]]["close"] > 0
                      and o[d]["close"]/o[ds[j-1]]["close"]-1 >= 0.095}
    out = [f"過去{days}交易日 不同『換手/黏著 INC』比較(每日換股⑤模擬,INC越低換手越高):",
           "  INC      報酬     換手    損益"]
    for inc in incs:
        r = sim5(rows, opens, closes, limitup, incumbent=float(inc), sell_mode="exit_only", open_buy="none")
        if r:
            mark = " ←實盤" if float(inc) == v5live.INC else ""
            out.append(f"  {float(inc):<5g}{mark:<5} {r['ret']:+6.1f}%  {r['turn']:4.1f}x  {r['total_pnl']:+,.0f}元")
    out.append("  ⚠️ 這只是過去N天單一視窗(雜訊大);要嚴謹決定 INC 該調多少,需多窗+regime回測(exp_trend_gate 那種),別只看這個。")
    return "\n".join(out)


def top_scores(args: dict) -> str:
    """今日分數最高的前 N 檔(原始分數由高到低排序)。問『分數前N高/最強的股票/排行』用這個,別自己排。"""
    n = int(args.get("n", 5))
    d = max(v5live.get_daily_ohlcv("0050"))
    pk = _picks_cached(d, set())            # held=空→原始分數排名(INC黏著不影響原始分)
    NM = _names()
    ranked = sorted(pk["rank_of"].items(), key=lambda kv: -kv[1][1])[:n]
    lines = [f"📊 今日({d})分數最高 {n} 檔(原始分,大盤{'多頭' if pk['bull'] else '空頭'}):"]
    for i, (c, (rk, s)) in enumerate(ranked, 1):
        lines.append(f"  {i}. {NM.get(c, c)}({c}) 分{s:.0f}")
    lines.append("  (依原始分數排序。注意:換手/黏著 INC 只影響『選股名單(持股黏著)』,不改原始分數排名)")
    return "\n".join(lines)


def today_buy(args: dict) -> str:
    """今天『該買什麼』的當日策略訊號(含目標金額;標出新買vs續抱)。非回測,是今天的動作。"""
    held = _holdings()
    d = max(v5live.get_daily_ohlcv("0050"))
    pk = _picks_cached(d, held)          # 帶持股→INC黏著,跟實盤買腿同邏輯
    NM = _names(); sel = pk["sel"]; tg = pk["targets"]
    if not sel:
        return f"📋【今日買進腿｜決策日{d}】名單空(空頭/下檔保險)→ 今天不買。"
    lines = [f"📋【今日該買 — 買進腿｜決策日{d}｜大盤{'多頭' if pk['bull'] else '空頭'}｜曝險{pk['expo']:.0%}｜可投{pk.get('capital',0):,.0f}】"]
    for s, c in sel:
        lines.append(f"  {'續抱' if c in held else '🆕買'} {NM.get(c, c)}({c}) 分{s:.0f} → 目標 {tg.get(c, 0):,.0f}元")
    lines.append("  (用最新收盤算的今日訊號;實盤13:15會用盤中即時價,數字可能微調。此處不下單)")
    return "\n".join(lines)


def today_sell(args: dict) -> str:
    """今天『該賣什麼』的當日策略訊號(持倉掉出名單→賣;否則續抱)。非回測,是今天的動作。"""
    held = _holdings()
    if not held:
        return "你目前無持倉,今日賣出腿無動作。"
    d = max(v5live.get_daily_ohlcv("0050"))
    pk = _picks_cached(d, held)
    NM = _names(); sel = [c for _, c in pk["sel"]]
    drop = [c for c in held if c not in sel]
    keep = [c for c in held if c in sel]
    head = f"📋【今日該賣 — 賣出腿｜決策日{d}】"
    if not drop:
        return f"{head} 你的持倉都還在名單內 → 全部續抱,今天不賣。\n  續抱: {', '.join(NM.get(c, c) for c in keep)}"
    lines = [f"{head} 掉出名單、今天該賣:"]
    for c in drop:
        lines.append(f"  🔴賣 {NM.get(c, c)}({c})")
    lines.append(f"  續抱: {', '.join(NM.get(c, c) for c in keep) or '無'}")
    lines.append("  (用最新收盤算的今日訊號;實盤09:02會用開盤即時價。此處不下單)")
    return "\n".join(lines)


# ─────────────────────── MUTATE(設定,需閘門)───────────────────────
def set_manual_score(args: dict) -> str:
    code, bonus = str(args["code"]), float(args["bonus"]); days = int(args.get("days", 5))
    until = mscore._forward_td(date.today(), days)
    m = mscore._load()
    m[code] = {"bonus": bonus, "until": until.isoformat(), "set": date.today().isoformat(),
               "days": days, "note": args.get("note", "agent set")}
    mscore._save(m); _invalidate_cache()
    return f"✅ 已設定 {code} {bonus:+.0f}分,生效到 {until}（{days}個交易日）"


def remove_manual_score(args: dict) -> str:
    code = str(args["code"]); m = mscore._load()
    if code in m:
        del m[code]; mscore._save(m); _invalidate_cache(); return f"✅ 已移除 {code} 的手動加減分"
    return f"{code} 不在手動加減分清單"


def set_weights(args: dict) -> str:
    w = args["weights"]
    if w == "auto" or w is None:
        (DATA_DIR/"weights.json").unlink(missing_ok=True); _invalidate_cache()
        return "✅ 雙引擎權重恢復預設 B純切 [1,0,0,1.5]"
    w = [float(x) for x in w]
    assert len(w) == 4, "weights 需 4 個:多頭H,多頭reb,空頭H,空頭reb"
    (DATA_DIR/"weights.json").write_text(json.dumps({"weights": w}), encoding="utf-8"); _invalidate_cache()
    return f"✅ 雙引擎權重設為 {w}"


def set_capital(args: dict) -> str:
    a = args.get("amount")
    if a == "auto" or a is None:
        (DATA_DIR/"capital.json").unlink(missing_ok=True); _invalidate_cache(); return "✅ 現金池恢復 DCA 自動"
    (DATA_DIR/"capital.json").write_text(json.dumps({"capital": float(a)}), encoding="utf-8"); _invalidate_cache()
    return f"✅ 現金池設為 {float(a):,.0f} TWD"


# ─────────────────────── 工具註冊表 ───────────────────────
def _p(props, required=()):
    return {"type": "object", "properties": props, "required": list(required)}

TOOLS = [
    # ---- READ:query ----
    {"name": "query_score", "tier": "read", "category": "query", "fn": query_score,
     "desc": "查指定股票今天的策略分數、排名、會不會被選到(反映目前的手動加減分)。",
     "params": _p({"codes": {"type": "array", "items": {"type": "string"}, "description": "股票代號清單"}}, ["codes"])},
    {"name": "query_universe", "tier": "read", "category": "query", "fn": query_universe,
     "desc": "查選股池(universe):總檔數 / 列出全部 / 檢查某股在不在。",
     "params": _p({"action": {"type": "string", "enum": ["count", "list", "check"]}, "code": {"type": "string"}})},
    {"name": "query_positions", "tier": "read", "category": "query", "fn": query_positions,
     "desc": "查目前帳本持倉。", "params": _p({})},
    {"name": "top_scores", "tier": "read", "category": "query", "fn": top_scores,
     "desc": "今日分數最高的前N檔(原始分由高到低,已排好序)。問『分數前N高/最強的股票/分數排行』一定用這個,不要自己列/自己排序。",
     "params": _p({"n": {"type": "integer", "description": "前幾名,預設5"}})},
    {"name": "today_buy", "tier": "read", "category": "query", "fn": today_buy,
     "desc": "今天『該買什麼』的當日策略訊號(買進腿:今日名單+目標金額,標新買/續抱)。問『今天買什麼/今日該買/今日策略/買進訊號』用這個。不下單。",
     "params": _p({})},
    {"name": "today_sell", "tier": "read", "category": "query", "fn": today_sell,
     "desc": "今天『該賣什麼』的當日策略訊號(賣出腿:持倉掉出名單就賣,否則續抱)。問『今天賣什麼/今日該賣/賣出訊號』用這個。不下單。",
     "params": _p({})},
    {"name": "query_manual_scores", "tier": "read", "category": "query", "fn": query_manual_scores,
     "desc": "查目前生效中的手動加減分。", "params": _p({})},
    {"name": "query_config", "tier": "read", "category": "query", "fn": query_config,
     "desc": "查目前的現金池與雙引擎權重設定。", "params": _p({})},
    {"name": "resolve_stock", "tier": "read", "category": "query", "fn": resolve_stock,
     "desc": "把股票名稱(可能打錯或簡稱)解析成代號。使用者用名字提到股票、或你不確定代號時先用這個;若有多個或不確定,問使用者是哪一個再動作。",
     "params": _p({"query": {"type": "string", "description": "股票名稱或代號"}}, ["query"])},
    # ---- READ:test ----
    {"name": "preview_bonus", "tier": "read", "category": "test", "fn": preview_bonus,
     "desc": "試算:對某股加 N 分、用過去5交易日跑出加分後分數與是否被選,跟原本並排比。設定加分前必須先跑這個給使用者看。",
     "params": _p({"code": {"type": "string"}, "bonus": {"type": "number"}, "days": {"type": "integer"}}, ["code", "bonus"])},
    {"name": "bonus_sweep", "tier": "read", "category": "test", "fn": bonus_sweep,
     "desc": "算某股『要加幾分才會出現在過去N天的選股』,並列出加X分→入選幾天的階梯。回答「加幾分才會被選到/比例」就用這個(快)。",
     "params": _p({"code": {"type": "string"}, "days": {"type": "integer"}}, ["code"])},
    {"name": "pnl_strategy_daily", "tier": "read", "category": "test", "fn": pnl_strategy_daily,
     "desc": "過去N交易日『每日換股』的⑤模擬損益(逐日收盤買/開盤賣掉出名單,含持股黏著)。問「每日換股/每天那種/真實損益」用這個。可帶 inc 用『不同換手/黏著參數』重跑(不影響實盤)。",
     "params": _p({"days": {"type": "integer"}, "inc": {"type": "number", "description": "換手/黏著參數(實盤1.75);越低換手越高"}})},
    {"name": "order_history", "tier": "read", "category": "test", "fn": order_history,
     "desc": "過去N交易日策略每天買進/賣出了什麼(逐日換股明細)。問「下單明細/每天買賣什麼」用這個。可帶 inc 看不同換手參數下的換股。",
     "params": _p({"days": {"type": "integer"}, "inc": {"type": "number"}})},
    {"name": "compare_inc", "tier": "read", "category": "test", "fn": compare_inc,
     "desc": "比較不同『換手權重/黏著參數 INC』(如1.3/1.5/1.75/2.0)在過去N天每日換股的報酬與換手。使用者說「換手權重/黏著 改成X再跑」「該不該這麼黏」「換手權重該調多少」就用這個。INC越低換手越高。不會改實盤設定。",
     "params": _p({"days": {"type": "integer"}, "incs": {"type": "array", "items": {"type": "number"}, "description": "要比較的INC清單,預設[1.3,1.5,1.75,2.0]"}})},
    {"name": "pnl_history", "tier": "read", "category": "test", "fn": pnl_history,
     "desc": ("損益。mode=holdings=你『實際持倉』的真實未實現損益(現價vs你的成本+今日單日漲跌,點時間,非假設)——"
              "問『我的損益/我賺多少/我持股損益』用這個。"
              "mode=list/stocks=『假設模式』回測(過去N日照買持有,非你的持倉):list=今日策略名單(可帶 bonus 試算),"
              "stocks=指定股(weight equal/score)。"),
     "params": _p({"mode": {"type": "string", "enum": ["holdings", "list", "stocks"]},
                   "codes": {"type": "array", "items": {"type": "string"}},
                   "weight": {"type": "string", "enum": ["equal", "score"]},
                   "days": {"type": "integer"}, "capital": {"type": "number"},
                   "bonus": {"type": "object", "properties": {"code": {"type": "string"}, "bonus": {"type": "number"}}}},
                  ["mode"])},
    # ---- MUTATE:set(需閘門)----
    {"name": "set_manual_score", "tier": "mutate", "category": "set", "fn": set_manual_score,
     "desc": "對某股寫入手動加減分,維持 N 個交易日(到期自動失效)。呼叫前必須先 preview_bonus 並取得使用者明確確認。",
     "params": _p({"code": {"type": "string"}, "bonus": {"type": "number"}, "days": {"type": "integer"}, "note": {"type": "string"}}, ["code", "bonus"])},
    {"name": "remove_manual_score", "tier": "mutate", "category": "set", "fn": remove_manual_score,
     "desc": "移除某股的手動加減分。", "params": _p({"code": {"type": "string"}}, ["code"])},
    {"name": "set_weights", "tier": "mutate", "category": "set", "fn": set_weights,
     "desc": "設定雙引擎權重 [多頭H,多頭reb,空頭H,空頭reb],或傳 'auto' 恢復預設 B純切。",
     "params": _p({"weights": {"description": "4個數字的陣列,或字串 'auto'"}}, ["weights"])},
    {"name": "set_capital", "tier": "mutate", "category": "set", "fn": set_capital,
     "desc": "設定可投資金(現金池),或傳 'auto' 恢復 DCA 自動。",
     "params": _p({"amount": {"description": "金額(數字)或字串 'auto'"}}, ["amount"])},
]
_BY_NAME = {t["name"]: t for t in TOOLS}


def run_tool(name: str, args: dict, allow_mutate: bool = False) -> str:
    """執行工具。mutate 級工具預設拒絕,需 allow_mutate=True(Phase 3 確認後才帶)。"""
    t = _BY_NAME.get(name)
    if not t:
        return f"❌ 未知工具: {name}"
    if t["tier"] == "mutate" and not allow_mutate:
        return f"🔒 {name} 是會改設定的工具,需經確認流程(allow_mutate)才能執行——已擋下。"
    try:
        return t["fn"](args or {})
    except Exception as e:
        return f"❌ {name} 執行錯誤: {e}"


def openai_schema(tier: str | None = None) -> list:
    """回傳 OpenAI function-tools 格式(給 Phase 3 GPT 用)。tier 可篩。"""
    return [{"type": "function", "function": {"name": t["name"], "description": t["desc"], "parameters": t["params"]}}
            for t in TOOLS if tier is None or t["tier"] == tier]


def main() -> int:
    a = sys.argv[1:]
    if not a or a[0] == "list":
        cur = None
        for t in sorted(TOOLS, key=lambda x: (x["category"], x["tier"])):
            tag = "🔒設定" if t["tier"] == "mutate" else "🔍唯讀"
            if t["category"] != cur:
                cur = t["category"]; print(f"\n[{cur}]")
            print(f"  {tag} {t['name']:<20} — {t['desc']}")
        print("\n(mutate=設定類需確認;trade 下單不在此庫=agent 永遠不能下真單)")
        return 0
    if a[0] == "schema":
        print(json.dumps(openai_schema(), ensure_ascii=False, indent=2)); return 0
    if a[0] == "run" and len(a) >= 2:
        name = a[1]
        args = json.loads(a[2]) if len(a) >= 3 and not a[2].startswith("--") else {}
        allow = "--allow-mutate" in a
        print(run_tool(name, args, allow_mutate=allow)); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main())
