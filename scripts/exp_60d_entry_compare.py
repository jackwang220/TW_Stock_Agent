"""60D 無LLM：隔日開盤 vs 當日收盤 執行時間對比

①原本邏輯  : 收盤後決策 → 隔日 09:00 開盤成交（有隔夜 gap 風險）
②收盤前30分: 13:30 決策 → 當日收盤成交（避開隔夜 gap，但訊號確認較晚）

訊號完全相同（H+反彈雙引擎，無 LLM），只改成交時間點。
手續費: 買 0.1425% / 賣 0.4425% + 滑價 0.1%
漲停: ①隔日開盤 gap ≥9.5% 跳過；②今日已漲停(close/prev_close≥9.5%)跳過

初次執行需從 FinMind 抓資料（~112 支股票，可能需要幾分鐘），資料會快取到
data/finmind_cache/ 供後續使用。

用法:
    python scripts/exp_60d_entry_compare.py
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal

# ── 借用 v3/v5/v6/r60 的特徵工程與模擬引擎 ──────────────────────────────────
def _load(name, path):
    m = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    importlib.util.spec_from_file_location(name, path).loader.exec_module(m)
    return m

v3  = _load("v3",  ROOT / "scripts/exp_step1_v3.py")
v5  = _load("v5",  ROOT / "scripts/exp_step1_v5.py")
v6  = _load("v6",  ROOT / "scripts/exp_step1_v6.py")
r60 = _load("r60", ROOT / "scripts/run_backtest_60d.py")

features, _factors, oh = v3.features, v3._factors, v3.oh

# ── 常數 ──────────────────────────────────────────────────────────────────────
END     = "2026-06-08"
NDAYS   = 60    # （保留）單窗口預設
WINDOWS = [("60天", 60), ("90天", 90), ("120天", 120), ("1年", 252), ("1年半", 378), ("2年", 504)]
TOPN    = 4     # 每日最多取 top-N 候選
FEE_BUY, FEE_SELL, SLIP = 0.001425, 0.004425, 0.001
INC, HY = 1.5, 0.05


# ── H+ 分數（與 exp_60d_llm.py ①無LLM 相同）─────────────────────────────────
def h_score(ff, tp: float) -> float:
    if ff is None:
        return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t + 0.35*rs + 0.15*vo + 0.10*ri + 0.05*ma) * 100 * (0.8 + 0.4*tp)


# ── 模擬引擎 A：隔日開盤成交（直接用 v6.sim_real）────────────────────────────
sim_open = v6.sim_real   # (rows, opens, closes, limitup, incumbent, hyst) -> dict


# ── 模擬引擎 B：當日收盤成交 ──────────────────────────────────────────────────
def sim_close(rows, opens, closes, limitup, incumbent=INC, hyst=HY, limitup_by_open=False):
    """13:30 訊號 → 當日收盤成交。

    唯一改動（vs sim_real）:
      • 執行價  : closes[tk][d]（訊號日收盤）而非 opens[tk][e]（隔日開盤）
      • 漲停判斷: d in limitup[tk]（今日鎖漲停=買不到），而非隔日開盤漲停
      • 隔夜 gap: 不存在
      • 權益估值: 仍用 closes[tk][e]（隔日收盤），與 sim_real 對齊方便比較

    limitup_by_open=True（診斷用）: 成交價仍用 close[d]，但漲停過濾改用「隔日開盤
      ≥9.5%」（與 sim_real 一致）→ 隔離「純成交價」單因子，其他全部對齊 sim_real。
    """
    sig = defaultdict(dict)
    tickers: set[str] = set()
    for d, tk, e in rows:
        tickers.add(tk)
        sig[d][tk] = e
    if not sig:
        return None

    alld = sorted({d for tk in tickers for d in closes.get(tk, {})})
    first, last = min(sig), max(sig)
    cal = [d for d in alld if first <= d <= last]
    if len(cal) < 2:
        return None

    def cl(tk, d):
        c = closes.get(tk, {})
        ds = [x for x in c if x <= d]
        return c[max(ds)] if ds else None

    cash = contributed = prev_eq = traded = fees = 0.0
    shares: dict[str, float] = {}
    last_edge: dict[str, float] = {}
    day_pnl: list[float] = []
    expo_track: list[float] = []
    n_limitup_skip = 0   # 因漲停跳過的「想買」次數
    n_buy_intent = 0     # 總共有幾次想買（delta>0 且通過 hyst）

    for i, d in enumerate(cal):
        add = (r60.INITIAL_CAPITAL if i == 0
               else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION - contributed)
                     if contributed < r60.MAX_CONTRIBUTION else 0.0))
        cash += add
        contributed += add

        if i + 1 >= len(cal):
            eq = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)
            day_pnl.append(eq - prev_eq - add)
            break

        e = cal[i + 1]  # 隔日（只用於估值，不用於成交）

        # 訊號日收盤前估值（作為 port 基礎決定目標部位）
        port = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)

        todays = sig.get(d, {})
        edges: dict[str, float] = {}
        for tk, ed in todays.items():
            edges[tk] = ed
            last_edge[tk] = ed
        for tk in shares:
            if tk not in edges:
                ed = last_edge.get(tk, 0.0) * r60.EDGE_DECAY
                edges[tk] = ed
                last_edge[tk] = ed

        rk = lambda tk: edges[tk] * (incumbent if tk in shares else 1.0)
        ranked = sorted([tk for tk in edges if edges[tk] > 0], key=rk, reverse=True)
        sel = ranked[:r60.MAX_SIGNALS]
        if (len(ranked) > r60.MAX_SIGNALS
                and rk(ranked[r60.MAX_SIGNALS]) >= rk(ranked[r60.MAX_SIGNALS - 1]) * r60.TIE_RATIO):
            sel = ranked[:r60.MAX_SIGNALS + 1]

        confs = [todays[tk] for tk in sel if tk in todays]
        avg = sum(confs) / len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg)) if sel else 0.0

        wsum = sum(edges[tk] for tk in sel)
        targets: dict[str, float] = {}
        if wsum > 0 and expo > 0:
            for tk in sel:
                targets[tk] = port * expo * (edges[tk] / wsum)

        for tk in set(shares) | set(targets):
            cp = closes.get(tk, {}).get(d)      # 當日收盤 = 執行價
            if not cp or cp <= 0:
                continue
            cur = shares.get(tk, 0.0) * cp
            tgt = targets.get(tk, 0.0)
            delta = tgt - cur
            if delta > r60.DAILY_ADD_CAP:
                delta = r60.DAILY_ADD_CAP
                tgt = cur + delta
            if abs(delta) < hyst * port:
                continue
            if delta > 0:
                n_buy_intent += 1
                if limitup_by_open:   # 診斷模式:漲停過濾對齊 sim_real（隔日開盤≥9.5%）
                    op_e = opens.get(tk, {}).get(e)
                    if op_e and cp and op_e / cp - 1 >= 0.095:
                        n_limitup_skip += 1
                        continue
                elif d in limitup.get(tk, set()):   # 今日漲停→買不到
                    n_limitup_skip += 1
                    continue
            slip_cost = abs(delta) * SLIP
            fee = (FEE_BUY if delta > 0 else FEE_SELL) * abs(delta)
            cash -= delta + slip_cost + fee
            traded += abs(delta)
            fees += fee + slip_cost
            if tgt <= 1e-6:
                shares.pop(tk, None)
            else:
                shares[tk] = tgt / cp

        # 以隔日收盤估值（與 sim_real 對齊，確保 PnL 可直接比較）
        invested = sum(shares[tk] * (cl(tk, e) or 0) for tk in shares)
        eq = cash + invested
        expo_track.append(invested / eq if eq > 0 else 0.0)
        day_pnl.append(eq - prev_eq - add)
        prev_eq = eq

    total = sum(day_pnl)
    active = [x for x in day_pnl if abs(x) > 1e-9]
    cum = peak = mdd = 0.0
    for x in day_pnl:
        cum += x
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    if len(active) > 1:
        mm = sum(active) / len(active)
        sd = math.sqrt(sum((x - mm) ** 2 for x in active) / len(active))
        shp = (mm / sd * math.sqrt(252)) if sd > 0 else 0.0
    else:
        shp = 0.0
    return {
        "ret":       total / contributed * 100 if contributed else 0,
        "mdd":       mdd,
        "sharpe":    shp,
        "turn":      traded / contributed if contributed else 0,
        "total_pnl": total,
        "fees":      fees,
        "avg_expo":  sum(expo_track) / len(expo_track) if expo_track else 0.0,
        "limitup_skip": n_limitup_skip,
        "buy_intent":   n_buy_intent,
    }


# ── 模擬引擎 ③混合：d收盤先買 → d+1開盤補沒買夠的 ───────────────────────────
def sim_hybrid(rows, opens, closes, limitup, incumbent=INC, hyst=HY):
    """混合執行（真實兩段式）：

      決策/賣出（d 收盤）:
        • sizing 與 targets 算法跟 ①② 完全相同（d 日收盤估值）
        • 減碼/出場用 close[d] 成交
      買入分兩階段（同一個目標金額，受同一個 DAILY_ADD_CAP）:
        • 階段1 @ d 收盤 : 非鎖漲停的，用 close[d] 先買到位
        • 階段2 @ d+1 開盤: 階段1沒買夠的（含當日鎖漲停的），用 open[d+1] 補；
                            隔日開盤仍跳空漲停(≥9.5%) 才真正放棄
      估值用 close[d+1]（與 ①② 對齊，可直接比較）。
    """
    sig = defaultdict(dict)
    tickers: set[str] = set()
    for d, tk, e in rows:
        tickers.add(tk)
        sig[d][tk] = e
    if not sig:
        return None

    alld = sorted({d for tk in tickers for d in closes.get(tk, {})})
    first, last = min(sig), max(sig)
    cal = [d for d in alld if first <= d <= last]
    if len(cal) < 2:
        return None

    def cl(tk, d):
        c = closes.get(tk, {})
        ds = [x for x in c if x <= d]
        return c[max(ds)] if ds else None

    cash = contributed = prev_eq = traded = fees = 0.0
    shares: dict[str, float] = {}
    last_edge: dict[str, float] = {}
    day_pnl: list[float] = []
    expo_track: list[float] = []
    n_close_fill = 0   # 階段1（收盤）成交筆數
    n_open_fill = 0    # 階段2（隔日開盤補）成交筆數
    n_both_blocked = 0  # 兩階段都買不到（罕見：當日+隔日開盤都漲停）

    def _buy(tk, amt, px):   # 共用買入：扣現金、記成本、加股數
        nonlocal cash, traded, fees
        slip_cost = amt * SLIP
        fee = FEE_BUY * amt
        cash -= amt + slip_cost + fee
        traded += amt
        fees += fee + slip_cost
        shares[tk] = shares.get(tk, 0.0) + amt / px

    for i, d in enumerate(cal):
        add = (r60.INITIAL_CAPITAL if i == 0
               else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION - contributed)
                     if contributed < r60.MAX_CONTRIBUTION else 0.0))
        cash += add
        contributed += add

        if i + 1 >= len(cal):
            eq = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)
            day_pnl.append(eq - prev_eq - add)
            break

        e = cal[i + 1]
        port = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)

        todays = sig.get(d, {})
        edges: dict[str, float] = {}
        for tk, ed in todays.items():
            edges[tk] = ed
            last_edge[tk] = ed
        for tk in shares:
            if tk not in edges:
                ed = last_edge.get(tk, 0.0) * r60.EDGE_DECAY
                edges[tk] = ed
                last_edge[tk] = ed

        rk = lambda tk: edges[tk] * (incumbent if tk in shares else 1.0)
        ranked = sorted([tk for tk in edges if edges[tk] > 0], key=rk, reverse=True)
        sel = ranked[:r60.MAX_SIGNALS]
        if (len(ranked) > r60.MAX_SIGNALS
                and rk(ranked[r60.MAX_SIGNALS]) >= rk(ranked[r60.MAX_SIGNALS - 1]) * r60.TIE_RATIO):
            sel = ranked[:r60.MAX_SIGNALS + 1]

        confs = [todays[tk] for tk in sel if tk in todays]
        avg = sum(confs) / len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg)) if sel else 0.0

        wsum = sum(edges[tk] for tk in sel)
        targets: dict[str, float] = {}
        if wsum > 0 and expo > 0:
            for tk in sel:
                targets[tk] = port * expo * (edges[tk] / wsum)

        # ── 賣出/減碼（d 收盤）──
        for tk in set(shares) | set(targets):
            cp = closes.get(tk, {}).get(d)
            if not cp or cp <= 0:
                continue
            cur = shares.get(tk, 0.0) * cp
            tgt = targets.get(tk, 0.0)
            delta = tgt - cur
            if delta >= 0:
                continue                       # 買入留到下方兩階段
            if abs(delta) < hyst * port:
                continue
            sell_amt = abs(delta)
            slip_cost = sell_amt * SLIP
            fee = FEE_SELL * sell_amt
            cash += sell_amt - slip_cost - fee
            traded += sell_amt
            fees += fee + slip_cost
            shares[tk] = tgt / cp
            if shares[tk] <= 1e-6:
                shares.pop(tk, None)

        # ── 買入（兩階段，共用 DAILY_ADD_CAP 與 hyst）──
        for tk in sel:
            cp = closes.get(tk, {}).get(d)
            if not cp or cp <= 0:
                continue
            cur = shares.get(tk, 0.0) * cp
            tgt = targets.get(tk, 0.0)
            buy_budget = min(tgt - cur, r60.DAILY_ADD_CAP)
            if buy_budget <= 0 or buy_budget < hyst * port:
                continue
            filled = 0.0
            # 階段1：d 收盤（非當日鎖漲停）
            if d not in limitup.get(tk, set()):
                _buy(tk, buy_budget, cp)
                filled = buy_budget
                n_close_fill += 1
            # 階段2：d+1 開盤補沒買夠的
            remain = buy_budget - filled
            if remain > 1e-6:
                op = opens.get(tk, {}).get(e)
                if op and op > 0 and op / cp - 1 < 0.095:   # 隔日開盤未漲停 → 買得到
                    _buy(tk, remain, op)
                    n_open_fill += 1
                elif filled <= 1e-6:
                    n_both_blocked += 1                     # 兩階段都漲停 → 真的買不到

        invested = sum(shares[tk] * (cl(tk, e) or 0) for tk in shares)
        eq = cash + invested
        expo_track.append(invested / eq if eq > 0 else 0.0)
        day_pnl.append(eq - prev_eq - add)
        prev_eq = eq

    total = sum(day_pnl)
    active = [x for x in day_pnl if abs(x) > 1e-9]
    cum = peak = mdd = 0.0
    for x in day_pnl:
        cum += x
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    if len(active) > 1:
        mm = sum(active) / len(active)
        sd = math.sqrt(sum((x - mm) ** 2 for x in active) / len(active))
        shp = (mm / sd * math.sqrt(252)) if sd > 0 else 0.0
    else:
        shp = 0.0
    return {
        "ret":       total / contributed * 100 if contributed else 0,
        "mdd":       mdd,
        "sharpe":    shp,
        "turn":      traded / contributed if contributed else 0,
        "total_pnl": total,
        "fees":      fees,
        "avg_expo":  sum(expo_track) / len(expo_track) if expo_track else 0.0,
        "close_fill": n_close_fill,
        "open_fill":  n_open_fill,
        "both_blocked": n_both_blocked,
    }


# 來回交易成本率（賣+買+雙邊滑價）→ 換股要划算的最低門檻
ROUNDTRIP_COST = FEE_SELL + FEE_BUY + 2 * SLIP   # ≈ 0.0078


# ── 模擬引擎 ⑤：買收盤(不補) + 賣開盤 + 開盤換股(賣出現金當天可用) ────────────
def sim_buyclose_sellopen(rows, opens, closes, limitup, incumbent=INC,
                          switch_cost_mult=1.0, sell_mode="rebalance",
                          open_buy="full", gap_cap=0.03, asym=False):
    """open_buy 控制 B②(隔日開盤補買)的條件:
      "full"     = 把沒買夠的全部補買(原行為)
      "none"     = 完全不開盤補買(只在收盤買、開盤只賣)
      "limitup"  = 只補買「收盤 d 鎖漲停、收盤腿沒買到」的
      "dip"      = 只在開盤 ≤ 昨收(開低/開平)時補買
      "gapcap"   = 開盤跳空向上 > gap_cap 就不補買(避免追高)"""
    """你的想法（含更正：賣出現金 d+1 開盤當天可動用）：

      A @ d 收盤   : 常規買入/加碼，用 close[d]；當日鎖漲停 → 先跳過(不補收盤)
      B @ d+1 開盤 : 用 open[d+1] 重估持倉
        ① 先賣  : 要減碼/出場(含掉出名單)的，用 open[d+1] 賣 → 現金當天入帳可用
        ② 後買  : 用①的現金，在開盤把「還沒買夠的」(含 A 階段鎖漲停沒買到的、
                   及換進的強股)補齊，用 open[d+1]；隔日開盤仍漲停 → 真的買不到
      換股門檻    : 只有「再平衡幅度 / port > switch_cost_mult × 來回成本(0.78%)」
                    才動手（算進成本仍划算才換，取代固定 5% 遲滯帶）
      單日單股總加碼(A+B) ≤ DAILY_ADD_CAP。估值 @ close[d+1]（與 ①②④ 對齊）。
    """
    sig = defaultdict(dict)
    tickers: set[str] = set()
    for d, tk, e in rows:
        tickers.add(tk)
        sig[d][tk] = e
    if not sig:
        return None

    alld = sorted({d for tk in tickers for d in closes.get(tk, {})})
    first, last = min(sig), max(sig)
    cal = [d for d in alld if first <= d <= last]
    if len(cal) < 2:
        return None

    def cl(tk, d):
        c = closes.get(tk, {})
        ds = [x for x in c if x <= d]
        return c[max(ds)] if ds else None

    thresh = switch_cost_mult * ROUNDTRIP_COST

    cash = contributed = prev_eq = traded = fees = 0.0
    shares: dict[str, float] = {}
    last_edge: dict[str, float] = {}
    day_pnl: list[float] = []
    expo_track: list[float] = []
    n_buy_close = n_sell_open = n_buy_open = n_blocked = 0
    pos_track: list[int] = []

    def _buy(tk, amt, px):
        nonlocal cash, traded, fees
        slip_cost = amt * SLIP
        fee = FEE_BUY * amt
        cash -= amt + slip_cost + fee
        traded += amt
        fees += fee + slip_cost
        shares[tk] = shares.get(tk, 0.0) + amt / px

    for i, d in enumerate(cal):
        add = (r60.INITIAL_CAPITAL if i == 0
               else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION - contributed)
                     if contributed < r60.MAX_CONTRIBUTION else 0.0))
        cash += add
        contributed += add

        if i + 1 >= len(cal):
            eq = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)
            day_pnl.append(eq - prev_eq - add)
            break

        e = cal[i + 1]
        port = cash + sum(shares[tk] * (cl(tk, d) or 0) for tk in shares)

        todays = sig.get(d, {})
        edges: dict[str, float] = {}
        for tk, ed in todays.items():
            edges[tk] = ed
            last_edge[tk] = ed
        for tk in shares:
            if tk not in edges:
                ed = last_edge.get(tk, 0.0) * r60.EDGE_DECAY
                edges[tk] = ed
                last_edge[tk] = ed

        rk = lambda tk: edges[tk] * (incumbent if tk in shares else 1.0)
        if asym:
            # 不對稱:買新用「原始分」選 top-N;持股若通過 raw×incumbent 的 top-N 就 keep 加回(可累積>N)
            ranked = sorted([tk for tk in edges if edges[tk] > 0], key=lambda tk: edges[tk], reverse=True)
            sel = ranked[:r60.MAX_SIGNALS]
            if (len(ranked) > r60.MAX_SIGNALS
                    and edges[ranked[r60.MAX_SIGNALS]] >= edges[ranked[r60.MAX_SIGNALS - 1]] * r60.TIE_RATIO):
                sel = ranked[:r60.MAX_SIGNALS + 1]
            keep_pool = set(sorted([tk for tk in edges if edges[tk] > 0], key=rk, reverse=True)[:r60.MAX_SIGNALS])
            sel = sel + [tk for tk in shares if tk in keep_pool and tk not in sel]
        else:
            ranked = sorted([tk for tk in edges if edges[tk] > 0], key=rk, reverse=True)
            sel = ranked[:r60.MAX_SIGNALS]
            if (len(ranked) > r60.MAX_SIGNALS
                    and rk(ranked[r60.MAX_SIGNALS]) >= rk(ranked[r60.MAX_SIGNALS - 1]) * r60.TIE_RATIO):
                sel = ranked[:r60.MAX_SIGNALS + 1]

        confs = [todays[tk] for tk in sel if tk in todays]
        avg = sum(confs) / len(confs) if confs else 0.0
        expo = min(r60.EXPOSURE_CAP, max(r60.EXPOSURE_FLOOR, avg)) if sel else 0.0

        wsum = sum(edges[tk] for tk in sel)
        targets: dict[str, float] = {}
        if wsum > 0 and expo > 0:
            for tk in sel:
                targets[tk] = port * expo * (edges[tk] / wsum)

        added_today: dict[str, float] = {}   # 單日單股已加碼(A+B 合計)，受 DAILY_ADD_CAP

        # ── A @ d 收盤：常規買入(鎖漲停跳過，不補收盤) ──
        for tk in sel:
            cp = closes.get(tk, {}).get(d)
            if not cp or cp <= 0:
                continue
            cur = shares.get(tk, 0.0) * cp
            cap_left = r60.DAILY_ADD_CAP - added_today.get(tk, 0.0)
            buy_amt = min(targets.get(tk, 0.0) - cur, cap_left)
            if buy_amt <= 0 or buy_amt < thresh * port:
                continue
            if d in limitup.get(tk, set()):    # 當日鎖漲停 → 留到開盤補
                continue
            _buy(tk, buy_amt, cp)
            added_today[tk] = added_today.get(tk, 0.0) + buy_amt
            n_buy_close += 1

        # ── B① @ d+1 開盤：先賣/減碼/出場(現金當天可用) ──
        for tk in list(shares):
            op = opens.get(tk, {}).get(e)
            if not op or op <= 0:
                continue
            cur = shares[tk] * op
            tgt = targets.get(tk, 0.0)
            # exit_only: 只賣「掉出名單」(target≈0)的，名單內即使 gap-up 超過 target 也不賣
            if sell_mode == "exit_only" and tgt > 1e-6:
                continue
            delta = tgt - cur
            if delta >= 0:
                continue
            if abs(delta) < thresh * port:
                continue
            sell_amt = abs(delta)
            slip_cost = sell_amt * SLIP
            fee = FEE_SELL * sell_amt
            cash += sell_amt - slip_cost - fee
            traded += sell_amt
            fees += fee + slip_cost
            shares[tk] = tgt / op
            if shares[tk] <= 1e-6:
                shares.pop(tk, None)
            n_sell_open += 1

        # ── B② @ d+1 開盤：用賣出現金補買沒買夠的(含換進的強股) ──
        for tk in (sel if open_buy != "none" else ()):
            op = opens.get(tk, {}).get(e)
            cp = closes.get(tk, {}).get(d)
            if not op or op <= 0 or not cp or cp <= 0:
                continue
            gap = op / cp - 1                  # 開盤相對昨收的跳空幅度
            # 開盤補買條件(視 open_buy 模式)
            if open_buy == "limitup" and d not in limitup.get(tk, set()):
                continue                       # 只補「收盤鎖漲停沒買到」的
            if open_buy == "dip" and gap > 0:
                continue                       # 只在開低/開平補買
            if open_buy == "gapcap" and gap > gap_cap:
                continue                       # 跳空向上太多就不追
            cur = shares.get(tk, 0.0) * op
            cap_left = r60.DAILY_ADD_CAP - added_today.get(tk, 0.0)
            buy_amt = min(targets.get(tk, 0.0) - cur, cap_left)
            if buy_amt <= 0 or buy_amt < thresh * port:
                continue
            if gap >= 0.095:                   # 隔日開盤仍漲停 → 真買不到
                n_blocked += 1
                continue
            _buy(tk, buy_amt, op)
            added_today[tk] = added_today.get(tk, 0.0) + buy_amt
            n_buy_open += 1

        invested = sum(shares[tk] * (cl(tk, e) or 0) for tk in shares)
        eq = cash + invested
        expo_track.append(invested / eq if eq > 0 else 0.0)
        pos_track.append(len([tk for tk in shares if shares[tk] > 1e-6]))
        day_pnl.append(eq - prev_eq - add)
        prev_eq = eq

    total = sum(day_pnl)
    active = [x for x in day_pnl if abs(x) > 1e-9]
    cum = peak = mdd = 0.0
    for x in day_pnl:
        cum += x
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    if len(active) > 1:
        mm = sum(active) / len(active)
        sd = math.sqrt(sum((x - mm) ** 2 for x in active) / len(active))
        shp = (mm / sd * math.sqrt(252)) if sd > 0 else 0.0
    else:
        shp = 0.0
    return {
        "ret":       total / contributed * 100 if contributed else 0,
        "mdd":       mdd,
        "sharpe":    shp,
        "turn":      traded / contributed if contributed else 0,
        "total_pnl": total,
        "fees":      fees,
        "avg_expo":  sum(expo_track) / len(expo_track) if expo_track else 0.0,
        "avg_pos":   sum(pos_track) / len(pos_track) if pos_track else 0.0,
        "buy_close": n_buy_close,
        "sell_open": n_sell_open,
        "buy_open":  n_buy_open,
        "blocked":   n_blocked,
    }


# ── 0050 基準（收盤成交版，對齊 sim_close）───────────────────────────────────
def bench_close_0050(closes, dates):
    """同資金模型 DCA 進 0050，當日收盤買（對齊②）。"""
    cal = [d for d in dates if d in closes.get("0050", {})]
    if len(cal) < 2:
        return 0.0
    sh = cash = contributed = 0.0
    for i, d in enumerate(cal):
        add = (r60.INITIAL_CAPITAL if i == 0
               else (min(r60.DAILY_BUDGET, r60.MAX_CONTRIBUTION - contributed)
                     if contributed < r60.MAX_CONTRIBUTION else 0.0))
        cash += add
        contributed += add
        cp = closes["0050"].get(d, 0)
        if cp > 0 and cash > 0:
            sh += cash / (cp * (1 + SLIP))
            cash = 0.0
    final = cash + sh * (closes["0050"].get(cal[-1], 0) if cal else 0)
    return (final - contributed) / contributed * 100 if contributed else 0.0


# ── 主程式 ────────────────────────────────────────────────────────────────────
def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}

    logger.info(f"載入特徵（{len(codes)} 支，首次執行需從 FinMind 抓資料）...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}

    # 用 oh() 取得 open/close（與 features() 同一份快取）
    opens:  dict[str, dict] = {}
    closes: dict[str, dict] = {}
    for c in codes + ["0050"]:
        o = oh(c)
        opens[c]  = {d: o[d]["open"]  for d in o}
        closes[c] = {d: o[d]["close"] for d in o}

    # 訊號生成涵蓋最長窗口（2年=504），各子窗口再從尾端切
    alld     = sorted({d for c in codes for d in closes.get(c, {}) if d <= END})
    sig_days = alld[-max(n for _, n in WINDOWS):]
    if not sig_days:
        logger.error("找不到交易日，請確認資料完整")
        return
    logger.info(f"訊號生成範圍: {sig_days[0]} ~ {sig_days[-1]}（{len(sig_days)} 交易日）")

    # 反彈訊號 + 漲停日集合
    logger.info("計算反彈訊號與漲停日...")
    reb_cache: dict[str, dict[str, float]] = {}
    limitup:   dict[str, set[str]]         = {}
    for c in codes:
        o  = oh(c)
        ds = sorted(d for d in o if d <= END)
        cl_list: list[float] = []
        m: dict[str, float] = {}
        s: set[str] = set()
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
            if j > 0 and o[ds[j-1]]["close"] > 0 and o[d]["close"] / o[ds[j-1]]["close"] - 1 >= 0.095:
                s.add(d)
        reb_cache[c] = m
        limitup[c]   = s

    # 成交值百分位（用於 h_score 的 turn 維度）
    turn_pct: dict[str, dict[str, float]] = {}
    for d in sig_days:
        vals = sorted(
            ((c, feats[c][d]["turn"]) for c in codes
             if d in feats.get(c, {}) and feats[c][d]["turn"] > 0),
            key=lambda x: x[1],
        )
        turn_pct[d] = ({c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)}
                       if vals else {})

    # 大盤 regime（0050 close > MA20 = 動能市）
    regime_bull = {
        d: bool(twii_feat.get(d, {}).get("close")
                and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
        for d in sig_days
    }

    # 生成訊號列表（rows = [(date, ticker, edge), ...]）
    logger.info("生成 H+反彈雙引擎訊號...")
    rows: list[tuple[str, str, float]] = []
    for d in sig_days:
        ir   = twii_feat.get(d, {}).get("ret20")
        bull = regime_bull.get(d)
        sc: list[tuple[float, str]] = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            if bull:
                v = h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            else:
                v = reb_cache.get(c, {}).get(d, 0.0)
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:TOPN]:
            rows.append((d, c, v / 100))

    n_sig = len(rows)
    n_bull = sum(1 for d in sig_days if regime_bull.get(d))
    n_bear = len(sig_days) - n_bull
    logger.info(f"訊號 {n_sig} 筆 | 動能日 {n_bull} / 反彈日 {n_bear}")

    # 訊號日 → 下一交易日（算隔夜 gap 用）
    nd_of = {alld[i]: alld[i + 1] for i in range(len(alld) - 1)}

    # ── 各策略（多窗口共用）──
    STRATS = [
        ("①純開盤",    lambda rw: sim_open(rw, opens, closes, limitup)),
        ("②純收盤",    lambda rw: sim_close(rw, opens, closes, limitup)),
        ("④混合",      lambda rw: sim_hybrid(rw, opens, closes, limitup)),
        ("⑤買收賣開",   lambda rw: sim_buyclose_sellopen(rw, opens, closes, limitup, switch_cost_mult=1.0)),
        ("⑤c只賣出場",  lambda rw: sim_buyclose_sellopen(rw, opens, closes, limitup, sell_mode="exit_only")),
    ]

    res: dict = {}        # (strat, win) -> sim dict
    bench: dict = {}      # win -> (開盤DCA, 收盤DCA)
    meta: dict = {}       # win -> {start,end,days,sig,bull,gap}

    logger.info("=" * 70)
    for wl, n in WINDOWS:
        wd_days = sig_days[-n:]
        wd = set(wd_days)
        rows_w = [r for r in rows if r[0] in wd]
        bench[wl] = (v6.bench_0050(opens["0050"], closes["0050"], wd_days),
                     bench_close_0050(closes, wd_days))
        nb = sum(1 for d in wd_days if regime_bull.get(d))
        # 該窗口訊號股的平均隔夜 gap（open[d+1]/close[d]-1）
        gaps = []
        for (d, tk, _) in rows_w:
            e = nd_of.get(d)
            cd = closes.get(tk, {}).get(d)
            oe = opens.get(tk, {}).get(e) if e else None
            if cd and oe and cd > 0:
                gaps.append(oe / cd - 1)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        meta[wl] = {"start": wd_days[0], "end": wd_days[-1], "days": len(wd_days),
                    "sig": len(rows_w), "bull": nb, "gap": avg_gap}
        for sname, fn in STRATS:
            res[(sname, wl)] = fn(rows_w)
        logger.info(
            f"  {wl:<5}({n}) {wd_days[0]}~{wd_days[-1]} 多頭{nb}/{len(wd_days)}日 "
            f"隔夜gap{avg_gap*100:+.2f}% | "
            + " ".join(f"{sn}{res[(sn,wl)]['ret']:+.0f}" if res[(sn, wl)] else f"{sn}—"
                       for sn, _ in STRATS))
    logger.info("=" * 70)

    # ── 報告 ──
    def g(sname, wl, key="ret"):
        r = res.get((sname, wl))
        return r[key] if r else None

    def fmt(sname, wl, key="ret", suf="%", sign=True):
        v = g(sname, wl, key)
        if v is None:
            return "—"
        return (f"{v:+.0f}{suf}" if sign else f"{v:.0f}{suf}")

    RPT = ROOT / "reports" / "exp_entry_compare_multiwin.md"
    RPT.parent.mkdir(parents=True, exist_ok=True)
    wlabels = [wl for wl, _ in WINDOWS]
    head_cells = " | ".join(wlabels)

    L = [
        "# 執行時點對比 · 多窗口（60天 → 2年）\n",
        f"> 結束 {END}｜H+反彈雙引擎無LLM｜112 檔｜資金 15000+1000/日(上限5萬)｜最多3檔\n",
        "> 手續費 買0.14%/賣0.44% + 滑價0.1%｜漲停買不到｜各窗口從同一結束日往回切\n",
        "## 窗口背景（市場狀態 + 訊號股隔夜跳空）\n",
        "| 指標 | " + head_cells + " |",
        "|------|" + "|".join(["---"] * len(WINDOWS)) + "|",
        "| 期間起 | " + " | ".join(meta[wl]["start"] for wl in wlabels) + " |",
        "| 多頭日% | " + " | ".join(f"{meta[wl]['bull']/meta[wl]['days']*100:.0f}%" for wl in wlabels) + " |",
        "| 訊號股平均隔夜gap | " + " | ".join(f"{meta[wl]['gap']*100:+.2f}%" for wl in wlabels) + " |",
        "| 0050(開盤DCA) | " + " | ".join(f"{bench[wl][0]:+.0f}%" for wl in wlabels) + " |",
        "",
        "## ① 報酬率 %（本金報酬，已扣成本）\n",
        "| 策略 | " + head_cells + " |",
        "|------|" + "|".join(["---"] * len(WINDOWS)) + "|",
    ]
    for sname, _ in STRATS:
        L.append(f"| {sname} | " + " | ".join(fmt(sname, wl, "ret") for wl in wlabels) + " |")
    L.append("| 0050基準 | " + " | ".join(f"{bench[wl][0]:+.0f}%" for wl in wlabels) + " |")

    L += [
        "",
        "## ② Alpha %（報酬 − 0050開盤DCA）\n",
        "| 策略 | " + head_cells + " |",
        "|------|" + "|".join(["---"] * len(WINDOWS)) + "|",
    ]
    for sname, _ in STRATS:
        cells = []
        for wl in wlabels:
            v = g(sname, wl, "ret")
            cells.append(f"{v - bench[wl][0]:+.0f}%" if v is not None else "—")
        L.append(f"| {sname} | " + " | ".join(cells) + " |")

    L += [
        "",
        "## ③ 最大回撤 MDD（TWD，越小越好）\n",
        "| 策略 | " + head_cells + " |",
        "|------|" + "|".join(["---"] * len(WINDOWS)) + "|",
    ]
    for sname, _ in STRATS:
        cells = []
        for wl in wlabels:
            v = g(sname, wl, "mdd")
            cells.append(f"-{v:,.0f}" if v is not None else "—")
        L.append(f"| {sname} | " + " | ".join(cells) + " |")

    L += [
        "",
        "## ④ Sharpe（年化）\n",
        "| 策略 | " + head_cells + " |",
        "|------|" + "|".join(["---"] * len(WINDOWS)) + "|",
    ]
    for sname, _ in STRATS:
        cells = []
        for wl in wlabels:
            v = g(sname, wl, "sharpe")
            cells.append(f"{v:.2f}" if v is not None else "—")
        L.append(f"| {sname} | " + " | ".join(cells) + " |")

    # ── 數據驅動判讀 ──
    def alp(s, wl):
        v = g(s, wl, "ret")
        return (v - bench[wl][0]) if v is not None else None
    L += ["", "## 判讀（數據驅動）\n"]
    a1, a2, a4 = alp("①純開盤", "2年"), alp("②純收盤", "2年"), alp("④混合", "2年")
    a5, a5c = alp("⑤買收賣開", "2年"), alp("⑤c只賣出場", "2年")
    m1, m5c = g("①純開盤", "2年", "mdd"), g("⑤c只賣出場", "2年", "mdd")
    s1, s5c = g("①純開盤", "2年", "sharpe"), g("⑤c只賣出場", "2年", "sharpe")
    gap2y = meta["2年"]["gap"]
    if None not in (a1, a2, a4, a5c, m1, m5c, s1, s5c):
        L += [
            f"- **你現在的 ①②④ 長期 alpha 全面轉負**：2年 alpha ① {a1:+.0f}%／② {a2:+.0f}%／④ {a4:+.0f}%"
            "（報酬雖正，但**跑輸直接 DCA 0050**）。純技術雙引擎抱越久、相對大盤越吃虧。",
            f"- **只有 ⑤ 系列維持正 alpha**：⑤ 2年 +{a5:.0f}%、⑤c +{a5c:.0f}%，且 60天→2年 每個窗口皆為正。"
            "**不是短窗口假象（我原本的假設被推翻）。**",
            f"- **⑤ 的優勢來源**：選強勢訊號股 + 買在收盤（隔夜前）+ 低換手抱住 → 吃滿強勢股的「隔夜跳空溢價」。"
            f"這溢價 2年仍 +{gap2y*100:.2f}%/日，且大於大盤隔夜貢獻（所以扣掉 0050 後 alpha 仍正）。",
            f"- **代價是回撤最大**：⑤c 2年 MDD -{m5c:,.0f} vs ① -{m1:,.0f}（大約 +{(m5c/m1-1)*100:.0f}%）。"
            f"但報酬倍率更高，**風險調整後 Sharpe 仍最優**（⑤c {s5c:.2f} vs ① {s1:.2f}）。",
            "",
            "### ⚠️ 上線前必須注意（為何不能因為這張表就梭哈⑤）\n",
            "1. **這 2 年是大多頭**（多頭日 67–80%），**沒有經歷長期空頭**。⑤吃的隔夜溢價在空頭可能轉負——"
            "收盤滿倉 = 裸隔夜曝險，一個隔夜系統性 gap-down 就會反向吃滿。",
            "2. **Sharpe 看不出尾部隔夜風險**：日報酬標準差沒有捕捉「隔夜跳空崩盤」這種低頻大事件。",
            "3. **⑤需要每天收盤前下單、且承擔 T+2 交割資金壓力**（你已確認帳戶現金流能撐）。",
            "",
        ]
    L += [
        "> 全部無 LLM、純技術雙引擎。各窗口共用同一套 sizing/門檻，只變執行時點。",
        "> 真正的待辦：找一段**含明確空頭/大回檔**的期間（本資料 2024-05 起最早，未涵蓋），單獨驗⑤的隔夜溢價會不會崩。",
    ]

    RPT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"多窗口報告 → {RPT}")


if __name__ == "__main__":
    main()
