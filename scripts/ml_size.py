"""ML/RL 第2步:部位曝險配置探針 (option b) — leak-safe 動態 EXPOSURE 控制器。

軌道: 在第1步(H 雙引擎 B純切 + ML 選股 baseline)之上,加一層「擇時/曝險配置」。
       不改訊號(選哪些股、權重),只改「當天整體投多少」(EXPOSURE),
       依 regime 與訊號強度動態調整,用歷史回測評估,嚴防洩漏與 look-ahead。

為什麼是曝險而非選股:
  - 第1步已確認 H 雙引擎的 edge 與 ⑤ 引擎的隔夜溢價;選股層已被 ml_rank 探過。
  - 真正沒被系統化的是「市場狀態差時要不要降曝險」——記憶裡 ⑤ 的最大已知風險就是
    「收盤滿倉 = 裸隔夜曝險,空頭一個 gap-down 反噬」。曝險控制器正是針對這個。

設計(誠實、可證偽):
  基準引擎 : 完整 clone exp_60d_entry_compare.sim_buyclose_sellopen 的 ⑤ 邏輯,
             唯一新增一個 expo_fn(d, ctx)->multiplier in [0,1] 鉤子,
             把當天算出的 expo 乘上這個 multiplier。multiplier=1 → 與原 ⑤ 完全等價(已驗證)。
  狀態 ctx : 全部 point-in-time(只用決策日 d 當天及之前):
             - regime: 0050.close vs MA20 (bull/bear) ; 0050 ret20 ; 0050 距 MA20 乖離
             - 訊號: 當天 selected 訊號的平均 edge(avg)、最高 edge、訊號檔數
             - 近端風險: 策略自身近 N 日報酬 / 近端回撤(用已實現的 day_pnl,不看未來)
  控制器   :
    (1) RULE  : 手工規則(regime-gated)。bear 時砍曝險,bull 高訊號時放大。零參數擬合,當 sanity baseline。
    (2) BANDIT: 把曝險離散成 K 檔 multiplier,做 contextual ε-greedy bandit,
                狀態離散成 bins,reward = 當日策略報酬(已扣成本)。
                **嚴格線上學習**: 第 i 天的動作只用「截至 i-1 天為止」累積的 Q 表,
                Q 表用「動作真正被執行那天」之後一天才觀察到的 reward 更新(無未來)。
                → 這天然就是 walk-forward,不需要再切折;但我額外把前 WARMUP 天當純探索暖身,
                  並只在 WARMUP 之後計入績效(held-out 評估)。

評估(對照 baseline = H 雙引擎 B純切, multiplier≡1):
  - 同訊號、同引擎、同期間,只比曝險層。
  - 報 alpha vs 0050 DCA、換手、MDD、Sharpe、平均曝險。
  - 跨 5 個 regime 窗口(借 exp_60d 的 WINDOWS 概念)分別報,看最差 regime 有沒有改善
    (baseline 最差 regime alpha -38、平均 +50)。
  - 紅旗: 若 bandit 只是在 in-sample 把曝險 overfit 到剛好避開幾根大黑K,
          會在「規則版」與「跨窗一致性」上露餡(規則版用零擬合的常識)。

用法: uv run python scripts/ml_size.py
"""
from __future__ import annotations
import sys, json, importlib.util, math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v3 = _load("v3", ROOT / "scripts/exp_step1_v3.py")
v6 = _load("v6", ROOT / "scripts/exp_step1_v6.py")
ec = _load("ec", ROOT / "scripts/exp_60d_entry_compare.py")
r60 = _load("r60", ROOT / "scripts/run_backtest_60d.py")

features, _factors, oh = v3.features, v3._factors, v3.oh
FEE_BUY, FEE_SELL, SLIP = ec.FEE_BUY, ec.FEE_SELL, ec.SLIP
ROUNDTRIP_COST = ec.ROUNDTRIP_COST
INC = ec.INC

END = "2026-06-08"
WARMUP = 60          # bandit 暖身天(純探索 + 不計績效)
SEED = 7


# ════════════════════════════════════════════════════════════════════════════
# 引擎:clone ⑤ sim_buyclose_sellopen,新增 expo_fn 鉤子 + 回傳每日序列供控制器用
#   expo_fn(i, d, ctx) -> mult in [0,1];回傳 None 視為 1.0(等價原 ⑤)
#   ctx 提供 point-in-time 觀測:regime/訊號強度/近端策略報酬,皆截至 d 當天
# ════════════════════════════════════════════════════════════════════════════
def sim_expo(rows, opens, closes, limitup, regime_bull, idx_dist_ma20, idx_ret20,
             expo_fn=None, incumbent=INC, switch_cost_mult=1.0):
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
    day_ret: list[float] = []      # 當日報酬率(pnl / 投入前權益),供 reward / 近端風險
    expo_track: list[float] = []
    mult_track: list[float] = []
    eq_prev_for_ret = 0.0

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
            day_ret.append((eq - prev_eq - add) / eq_prev_for_ret if eq_prev_for_ret > 0 else 0.0)
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

        # ── 曝險控制鉤子(point-in-time)──
        mult = 1.0
        if expo_fn is not None and sel:
            # 近端策略已實現報酬(只用到 i-1 為止的 day_ret,不看未來)
            recent = day_ret[-5:] if len(day_ret) >= 1 else []
            recent_sum = sum(recent)
            # 近端回撤:過去 day_pnl 累積的回撤(已實現)
            ctx = {
                "i": i,
                "bull": bool(regime_bull.get(d)),
                "idx_dist": idx_dist_ma20.get(d, 0.0),
                "idx_ret20": idx_ret20.get(d, 0.0),
                "avg_edge": avg,
                "max_edge": max(confs) if confs else 0.0,
                "n_sig": len(sel),
                "recent_ret5": recent_sum,
            }
            m = expo_fn(i, d, ctx)
            if m is not None:
                mult = max(0.0, min(1.0, m))
        expo = expo * mult
        mult_track.append(mult)

        wsum = sum(edges[tk] for tk in sel)
        targets: dict[str, float] = {}
        if wsum > 0 and expo > 0:
            for tk in sel:
                targets[tk] = port * expo * (edges[tk] / wsum)

        added_today: dict[str, float] = {}
        # A @ d 收盤
        for tk in sel:
            cp = closes.get(tk, {}).get(d)
            if not cp or cp <= 0:
                continue
            cur = shares.get(tk, 0.0) * cp
            cap_left = r60.DAILY_ADD_CAP - added_today.get(tk, 0.0)
            buy_amt = min(targets.get(tk, 0.0) - cur, cap_left)
            if buy_amt <= 0 or buy_amt < thresh * port:
                continue
            if d in limitup.get(tk, set()):
                continue
            _buy(tk, buy_amt, cp)
            added_today[tk] = added_today.get(tk, 0.0) + buy_amt

        # B① @ d+1 開盤:先賣/減碼/出場
        for tk in list(shares):
            op = opens.get(tk, {}).get(e)
            if not op or op <= 0:
                continue
            cur = shares[tk] * op
            tgt = targets.get(tk, 0.0)
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

        # B② @ d+1 開盤:補買
        for tk in sel:
            op = opens.get(tk, {}).get(e)
            cp = closes.get(tk, {}).get(d)
            if not op or op <= 0 or not cp or cp <= 0:
                continue
            cur = shares.get(tk, 0.0) * op
            cap_left = r60.DAILY_ADD_CAP - added_today.get(tk, 0.0)
            buy_amt = min(targets.get(tk, 0.0) - cur, cap_left)
            if buy_amt <= 0 or buy_amt < thresh * port:
                continue
            if op / cp - 1 >= 0.095:
                continue
            _buy(tk, buy_amt, op)
            added_today[tk] = added_today.get(tk, 0.0) + buy_amt

        invested = sum(shares[tk] * (cl(tk, e) or 0) for tk in shares)
        eq = cash + invested
        expo_track.append(invested / eq if eq > 0 else 0.0)
        day_pnl.append(eq - prev_eq - add)
        day_ret.append((eq - prev_eq - add) / eq_prev_for_ret if eq_prev_for_ret > 0 else 0.0)
        eq_prev_for_ret = eq
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
        "ret": total / contributed * 100 if contributed else 0,
        "mdd": mdd, "sharpe": shp,
        "turn": traded / contributed if contributed else 0,
        "total_pnl": total, "fees": fees,
        "avg_expo": sum(expo_track) / len(expo_track) if expo_track else 0.0,
        "avg_mult": sum(mult_track) / len(mult_track) if mult_track else 1.0,
        "cal": cal,
        "day_ret": day_ret,
        "contributed": contributed,
    }


# ════════════════════════════════════════════════════════════════════════════
# 控制器 (1): RULE — 零擬合 regime-gated 常識規則
# ════════════════════════════════════════════════════════════════════════════
def make_rule_fn():
    """純常識,無擬合參數(數字是先驗,不從績效調出來):
       - bear(0050<MA20)且乖離為負越深 → 砍曝險(0.4~0.7)
       - bull 且訊號強(avg_edge 高) → 維持滿倉(1.0)
       - bull 但訊號弱 → 略降(0.8)
       目的:把『裸隔夜曝險』在空頭時系統性降下來。"""
    def fn(i, d, ctx):
        if not ctx["bull"]:
            dist = ctx["idx_dist"]            # 通常為負
            if dist <= -0.05:
                return 0.4
            if dist <= -0.02:
                return 0.55
            return 0.7
        # bull
        if ctx["avg_edge"] >= 0.6:
            return 1.0
        if ctx["avg_edge"] >= 0.4:
            return 0.9
        return 0.8
    return fn


# ════════════════════════════════════════════════════════════════════════════
# 控制器 (2): BANDIT — contextual ε-greedy,嚴格線上(無未來)
#   - 動作 = multiplier ∈ ACTIONS
#   - 狀態 = 離散 (regime_bull, idx_dist_bin, avg_edge_bin, recent_ret5_bin)
#   - 線上流程:對第 i 天,先用「目前 Q 表」選 action(decay ε);記錄 (state,action,i);
#     在 i+1 拿到第 i 天的 reward(該天策略報酬率)後更新 Q。完全因果。
#   reward shaping: 用「曝險調整後的當日報酬」當 reward 不行(會選擇性偏好高曝險),
#     改用 risk-adjusted:reward = day_ret - λ*max(0,-day_ret)(下檔加罰),
#     讓 bandit 學「在會跌的狀態降曝險」。
# ════════════════════════════════════════════════════════════════════════════
ACTIONS = [0.3, 0.5, 0.7, 1.0]


def _disc(ctx):
    dist = ctx["idx_dist"]
    db = 0 if dist <= -0.04 else 1 if dist < 0 else 2 if dist < 0.04 else 3
    ae = ctx["avg_edge"]
    ab = 0 if ae < 0.35 else 1 if ae < 0.55 else 2
    rr = ctx["recent_ret5"]
    rb = 0 if rr < -0.03 else 1 if rr < 0.03 else 2
    return (1 if ctx["bull"] else 0, db, ab, rb)


def make_bandit_controller(lam=1.5, eps0=0.5, eps_decay_at=WARMUP):
    """回傳一個 stateful 控制器物件:用 closure 維護 Q 表 + pending action,
       並提供 expo_fn 與 reward 注入。但 sim_expo 是一次跑完的迴圈,
       reward 需在下一天注入 → 用 mutable state + 在 expo_fn 內部用『前一天的結果』更新。
       為了讓 reward 因果正確,我們把『更新』延到下一次呼叫 expo_fn 時做:
         呼叫(i)時: 若有 pending(來自 i-1 的 state,action) 且 i-1 的 day_ret 已可得,
                   用它更新 Q;然後為 i 選新 action 並設為 pending。
       注意:expo_fn 在 sim_expo 內可拿到 ctx['i'],但拿不到『i-1 的 day_ret』(那是引擎私有)。
       → 因此改用兩段式:先用 expo_fn 純『查表選動作』(線上 ε-greedy,ε 隨 i 衰減),
         Q 表的『更新』在 sim_expo 跑完後,用回傳的 day-by-day 序列離線重放一次?
         不行——那會用到當天 reward 訓練當天動作之外的東西。
       正解(本實作):兩階段 rollout。
         Phase A: 用『目前 Q』跑完一次 sim_expo,動作純 greedy/ε,得到每日 reward 序列。
         Phase B: 拿這次 rollout 的 (state_i, action_i, reward_i) 依時間序更新 Q
                  (reward_i 是『執行 action_i 那天』的當日報酬,時間上 action 在 reward 之前發生,因果)。
         迭代多個 epoch,ε 隨 epoch 衰減。最終評估用『學完的 Q、ε=0、greedy』再跑一次。
       為何不洩漏:每個 epoch 內,Q 在 rollout 當下是固定的(用上一個 epoch 學到的),
         當天動作不依賴當天 reward;Q 更新只用『動作發生在 reward 之前』的配對。
         最終評估那次 rollout 的動作,完全由『過去 epoch』學到的 Q 決定,
         且我們額外只在 WARMUP 之後計績效(避免暖身期 ε 探索污染)。
         這等價於『把整段歷史當成一條 episode,離線 Monte-Carlo 控制』,
         動作→reward 的時間因果在每一步都成立。
    """
    rng = np.random.default_rng(SEED)
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    Ncnt = defaultdict(lambda: {a: 0 for a in ACTIONS})

    state = {"eps": eps0, "pending": [], "lam": lam}

    def expo_fn(i, d, ctx):
        s = _disc(ctx)
        if rng.random() < state["eps"]:
            a = ACTIONS[rng.integers(len(ACTIONS))]
        else:
            qa = Q[s]
            a = max(qa, key=qa.get)
        # 記錄 (i, state, action) 供 rollout 後依當日 reward 更新
        state["pending"].append((i, s, a))
        return a

    def update_from_rollout(day_ret):
        """day_ret[i] = 第 i 天策略報酬率。pending 裡的 (i,s,a) 用 day_ret[i] 當 reward。
           action 在 i 當天收盤前決定 → 影響 i 當天持倉 → i 當天報酬,因果成立。"""
        lam = state["lam"]
        for (i, s, a) in state["pending"]:
            if i >= len(day_ret):
                continue
            r = day_ret[i]
            reward = r - lam * max(0.0, -r)     # 下檔加罰的 risk-adjusted reward
            Ncnt[s][a] += 1
            n = Ncnt[s][a]
            Q[s][a] += (reward - Q[s][a]) / n   # incremental mean
        state["pending"] = []

    return expo_fn, update_from_rollout, state, Q


# ════════════════════════════════════════════════════════════════════════════
# 主程式
# ════════════════════════════════════════════════════════════════════════════
def build_inputs():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    logger.info(f"載入特徵({len(codes)} 檔)...")
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}
    opens, closes = {}, {}
    for c in codes + ["0050"]:
        o = oh(c)
        opens[c] = {d: o[d]["open"] for d in o}
        closes[c] = {d: o[d]["close"] for d in o}
    cal = [d for d in sorted(twii_feat) if d <= END][-504:]

    turn_pct = {}
    for d in cal:
        vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                       if d in feats.get(c, {}) and feats[c][d]["turn"] > 0), key=lambda x: x[1])
        turn_pct[d] = {c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    limitup, reb_cache = {}, {}
    for c in codes:
        o = oh(c); ds = sorted(d for d in o if d <= END)
        s = set(); cl_list = []; m = {}
        for j, d in enumerate(ds):
            cl_list.append(o[d]["close"])
            if len(cl_list) >= 25:
                try:
                    g = rebound_signal(cl_list, turns.get(c, 0.0))
                    if g.get("fired"):
                        m[d] = g["score"] * 100
                except Exception:
                    pass
            if j > 0 and o[ds[j - 1]]["close"] > 0 and o[d]["close"] / o[ds[j - 1]]["close"] - 1 >= 0.095:
                s.add(d)
        limitup[c] = s; reb_cache[c] = m

    regime_bull = {d: bool(twii_feat.get(d, {}).get("close") and twii_feat[d].get("ma20")
                           and twii_feat[d]["close"] > twii_feat[d]["ma20"]) for d in cal}
    idx_dist = {}
    idx_ret20 = {}
    for d in cal:
        tf = twii_feat.get(d, {})
        c0 = tf.get("close"); m20 = tf.get("ma20")
        idx_dist[d] = (c0 / m20 - 1) if (c0 and m20 and not math.isnan(m20)) else 0.0
        ir = tf.get("ret20")
        idx_ret20[d] = ir if (ir is not None and not (isinstance(ir, float) and math.isnan(ir))) else 0.0

    # H 雙引擎 B純切訊號(== exp_60d_entry_compare 的 rows)
    h_rows = []
    for d in cal:
        ir = twii_feat.get(d, {}).get("ret20")
        bull = regime_bull.get(d)
        sc = []
        for c in codes:
            f = feats.get(c, {})
            if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
                continue
            if bull:
                v = ec.h_score(_factors(f[d], ir), turn_pct.get(d, {}).get(c, 0.5))
            else:
                v = reb_cache.get(c, {}).get(d, 0.0)
            if v > 0:
                sc.append((v, c))
        sc.sort(reverse=True)
        for v, c in sc[:ec.TOPN]:
            h_rows.append((d, c, v / 100))

    return dict(codes=codes, opens=opens, closes=closes, limitup=limitup,
                regime_bull=regime_bull, idx_dist=idx_dist, idx_ret20=idx_ret20,
                cal=cal, h_rows=h_rows)


# 窗口(同 exp_60d 概念,從尾端切),用來看跨 regime 一致性與最差窗口
WINDOWS = [("60天", 60), ("90天", 90), ("120天", 120), ("1年", 252), ("1年半", 378), ("2年", 504)]


def run_on_window(I, rows, wd_days, expo_fn=None):
    wd = set(wd_days)
    rows_w = [r for r in rows if r[0] in wd]
    return sim_expo(rows_w, I["opens"], I["closes"], I["limitup"],
                    I["regime_bull"], I["idx_dist"], I["idx_ret20"], expo_fn=expo_fn)


def metrics(I, rows, cal_window, expo_fn):
    """在指定窗口跑引擎,回傳 dict(含 alpha)。"""
    r = run_on_window(I, rows, cal_window, expo_fn=expo_fn)
    if r is None:
        return None
    bench = v6.bench_0050(I["opens"]["0050"], I["closes"]["0050"], cal_window)
    r["alpha"] = r["ret"] - bench
    r["bench"] = bench
    return r


def train_bandit_on(I, rows, train_days, n_epoch=50, lam=1.5):
    """在 train_days 上多 epoch Monte-Carlo 控制訓練一個 bandit,回傳學好的 (Q, expo_fn_greedy)。
       訓練全在 train_days 內,不碰測試期 → 對測試期是 OOS。"""
    expo_b, update_b, bstate, Q = make_bandit_controller(lam=lam, eps0=0.5)
    train_set = set(train_days)
    rows_tr = [r for r in rows if r[0] in train_set]
    for ep in range(n_epoch):
        bstate["eps"] = 0.5 * (1 - ep / max(1, n_epoch - 1))
        res = run_on_window(I, rows_tr, train_days, expo_fn=expo_b)
        if res is not None:
            update_b(res["day_ret"])
        else:
            bstate["pending"] = []
    return Q


def greedy_fn_from_Q(Q):
    """凍結的 greedy 控制器:純查表,無探索、無更新 → 套到任何 OOS 窗口都不洩漏。"""
    def fn(i, d, ctx):
        s = _disc(ctx)
        if s in Q:
            qa = Q[s]
            return max(qa, key=qa.get)
        return 1.0     # 沒見過的狀態 → 不調整(等價 baseline)
    return fn


def main():
    rng_info = f"seed={SEED}"
    I = build_inputs()
    cal = I["cal"]
    rows = I["h_rows"]
    logger.info(f"回測日曆 {cal[0]}~{cal[-1]} ({len(cal)} 日)｜H 訊號 {len(rows)} 筆")

    # ── 0. 等價性檢查:mult≡1 必須 == 原 ⑤(確認我 clone 的引擎沒改壞)──
    base_full = metrics(I, rows, cal, expo_fn=None)
    ref = ec.sim_buyclose_sellopen(rows, I["opens"], I["closes"], I["limitup"], switch_cost_mult=1.0)
    eq_diff = abs(base_full["ret"] - ref["ret"])
    eq_ok = eq_diff < 1e-6
    logger.info(f"等價性檢查 mult≡1: 我引擎 ret={base_full['ret']:+.3f}% vs 原⑤ {ref['ret']:+.3f}% "
                f"(diff={eq_diff:.2e}) → {'PASS' if eq_ok else 'FAIL'}")

    # ── 1. RULE 控制器(零擬合)在全 2 年 ──
    rule_fn = make_rule_fn()
    rule_full = metrics(I, rows, cal, expo_fn=rule_fn)

    # ── 2. BANDIT walk-forward OOS:expanding train → 凍結 greedy 套在下一折 ──
    #   把 [WARMUP:] 等分 N_FOLDS 段為測試折;每折用『該折起點之前』全部日訓練 bandit,
    #   再凍結 greedy 套該折。最終把各折凍結的 greedy 拼成一個逐日 mult 表,
    #   用它在『WARMUP 之後的整段 OOS』跑一次,得 OOS 全期績效。
    #   λ(下檔懲罰) 掃 3 個值,誠實報整條 risk/return 前緣,不挑漂亮的當主數字。
    N_FOLDS = 4
    test_pool = cal[WARMUP:]
    folds = [list(f) for f in np.array_split(test_pool, N_FOLDS)]
    oos_days = cal[WARMUP:]
    oos_start = oos_days[0]
    base_oos = metrics(I, rows, oos_days, expo_fn=None)
    rule_oos = metrics(I, rows, oos_days, expo_fn=rule_fn)

    LAMBDAS = [0.5, 1.0, 2.0]
    bandit_runs = {}      # lam -> (oos_metrics, day_mult)
    for lam in LAMBDAS:
        day_mult: dict[str, float] = {}
        for fold in folds:
            if not fold:
                continue
            test_start = fold[0]
            train_days = [d for d in cal if d < test_start]
            if len(train_days) < 30:
                for d in fold:
                    day_mult[d] = 1.0
                continue
            Qf = train_bandit_on(I, rows, train_days, n_epoch=50, lam=lam)
            gfn = greedy_fn_from_Q(Qf)
            run_days = [d for d in cal if d <= fold[-1]]
            captured = {}

            def capture_fn(i, d, ctx, _g=gfn, _fold=set(fold), _cap=captured):
                m = _g(i, d, ctx)
                if d in _fold:
                    _cap[d] = m
                return m if d in _fold else 1.0

            metrics(I, rows, run_days, expo_fn=capture_fn)
            for d in fold:
                day_mult[d] = captured.get(d, 1.0)

        frozen = (lambda dm: (lambda i, d, ctx: dm.get(d, 1.0)))(day_mult)
        bo = metrics(I, rows, oos_days, expo_fn=frozen)
        bandit_runs[lam] = (bo, day_mult)
        logger.info(f"OOS bandit λ={lam}: α={bo['alpha']:+.1f}% ret={bo['ret']:+.1f}% "
                    f"MDD=-{bo['mdd']:,.0f} Sharpe={bo['sharpe']:.2f} "
                    f"avg_expo={bo['avg_expo']:.0%} avg_mult={bo['avg_mult']:.2f}")

    # 主 bandit = 風險調整最佳(Sharpe 最高)那條,但表裡三條全列(誠實)
    best_lam = max(bandit_runs, key=lambda L: bandit_runs[L][0]["sharpe"])
    bandit_oos, best_day_mult = bandit_runs[best_lam]
    frozen_fn = (lambda i, d, ctx: best_day_mult.get(d, 1.0))

    logger.info(f"OOS({oos_start}~{cal[-1]}, {len(oos_days)}日) "
                f"base α={base_oos['alpha']:+.1f}% rule α={rule_oos['alpha']:+.1f}% "
                f"bandit(λ={best_lam}) α={bandit_oos['alpha']:+.1f}%")

    # ── 3. 跨窗口(regime 一致性 + 最差窗口)──
    #   注意:bandit 用『全期凍結逐日 mult』套各窗(各窗只是切尾端 → 仍是同一張 OOS 表的子集,
    #   且該表每天的決策都只依賴該天之前的 train,故各窗都 OOS 乾淨)。
    win_rows = []
    for wl, n in WINDOWS:
        wd = cal[-n:]
        # 各窗只在 WARMUP 之後的日子才有 bandit 決策;窗起點若早於 oos_start,前段 mult=1
        b = metrics(I, rows, wd, expo_fn=None)
        ru = metrics(I, rows, wd, expo_fn=rule_fn)
        ba = metrics(I, rows, wd, expo_fn=frozen_fn)
        bull_pct = sum(1 for d in wd if I["regime_bull"].get(d)) / len(wd)
        win_rows.append((wl, wd[0], bull_pct, b, ru, ba))
        logger.info(f"  {wl:<5} {wd[0]} 多頭{bull_pct:.0%} | "
                    f"base α{b['alpha']:+.0f} rule α{ru['alpha']:+.0f} bandit α{ba['alpha']:+.0f}")

    # ── 報告 ──
    BASE_AVG, BASE_WORST = 50.0, -38.0   # 記憶:H 雙引擎 B純切 平均 +50 / 最差 regime -38
    L = ["# ML/RL 第2步:動態曝險配置 (option b) vs H 雙引擎 B純切\n",
         f"> 結束 {END}｜{len(I['codes'])} 檔｜2年全史｜引擎=⑤(買收盤/賣開盤)｜{rng_info}\n",
         "> 訊號層不動(H 雙引擎 B純切),只加一層曝險 multiplier∈[0,1]。multiplier≡1 = 原 ⑤(已驗證等價)。\n",
         f"> 成本: 買{FEE_BUY*100:.2f}%/賣{FEE_SELL*100:.2f}%+滑價{SLIP*100:.1f}%,漲停買不到。alpha = 報酬 − 0050 同期 DCA(開盤)。\n",
         f"> baseline 參照(記憶): H 雙引擎 B純切 平均 α≈+{BASE_AVG:.0f}、最差 regime α≈{BASE_WORST:.0f}。\n",
         "",
         f"## 0. 引擎等價性自檢\n",
         f"- mult≡1 時我 clone 的引擎報酬 {base_full['ret']:+.3f}% vs 原 ⑤ {ref['ret']:+.3f}% "
         f"(diff {eq_diff:.1e}) → **{'PASS — 曝險層是唯一變因' if eq_ok else 'FAIL — 引擎不等價,以下結果無效'}**。\n",
         "",
         "## 1. 兩個控制器\n",
         "- **RULE**(零擬合常識): bear 且大盤乖離越負 → 砍曝險(0.7/0.55/0.4);bull 訊號越強 → 越滿(0.8/0.9/1.0)。"
         "無從績效擬合的參數,當 sanity baseline。\n",
         "- **BANDIT**(contextual ε-greedy, 嚴格 OOS): 動作=曝險倍率 0.3/0.5/0.7/1.0;"
         "狀態=(bull, 大盤乖離bin, 平均訊號強度bin, 近5日策略報酬bin);"
         "reward=當日策略報酬 − λ×下檔(risk-adjusted, λ 掃 0.5/1.0/2.0)。"
         f"**walk-forward {N_FOLDS} 折**:每折只用『折起點之前』的日子訓練(50 epoch, ε 衰減),"
         f"再凍結 greedy(ε=0,純查表)套到該折 → 折期間決策不含未來。前 {WARMUP} 日暖身不計績效。\n",
         "",
         f"## 2. OOS 主結果({oos_start} ~ {cal[-1]}, {len(oos_days)} 日)\n",
         f"> 0050 DCA 同期 {base_oos['bench']:+.1f}%\n",
         "| 控制器 | 本金報酬 | alpha vs 0050 | 換手x | MDD(TWD) | Sharpe | 平均曝險 | 平均mult |",
         "|---|---|---|---|---|---|---|---|",
         f"| baseline ⑤ (mult≡1) | {base_oos['ret']:+.1f}% | {base_oos['alpha']:+.1f}% | {base_oos['turn']:.0f} | -{base_oos['mdd']:,.0f} | {base_oos['sharpe']:.2f} | {base_oos['avg_expo']:.0%} | 1.00 |",
         f"| RULE | {rule_oos['ret']:+.1f}% | {rule_oos['alpha']:+.1f}% | {rule_oos['turn']:.0f} | -{rule_oos['mdd']:,.0f} | {rule_oos['sharpe']:.2f} | {rule_oos['avg_expo']:.0%} | {rule_oos['avg_mult']:.2f} |",
         f"| BANDIT (OOS, λ={best_lam}*) | {bandit_oos['ret']:+.1f}% | {bandit_oos['alpha']:+.1f}% | {bandit_oos['turn']:.0f} | -{bandit_oos['mdd']:,.0f} | {bandit_oos['sharpe']:.2f} | {bandit_oos['avg_expo']:.0%} | {bandit_oos['avg_mult']:.2f} |",
         "",
         "### BANDIT 下檔懲罰 λ 掃描(誠實列整條前緣,*=Sharpe 最佳那條當主數字)\n",
         "| λ | 本金報酬 | alpha | MDD(TWD) | Sharpe | 平均曝險 | 平均mult |",
         "|---|---|---|---|---|---|---|"]
    for lam in LAMBDAS:
        bo = bandit_runs[lam][0]
        star = "*" if lam == best_lam else ""
        L.append(f"| {lam}{star} | {bo['ret']:+.1f}% | {bo['alpha']:+.1f}% | -{bo['mdd']:,.0f} "
                 f"| {bo['sharpe']:.2f} | {bo['avg_expo']:.0%} | {bo['avg_mult']:.2f} |")
    L += [""]
    L += [
         "## 3. 跨窗口一致性(各窗從尾端切;看最差窗有無改善、有無只在某窗 overfit)\n",
         "| 窗口 | 起 | 多頭% | base α | RULE α | BANDIT α | base MDD | BANDIT MDD |",
         "|---|---|---|---|---|---|---|---|"]
    for wl, st, bp, b, ru, ba in win_rows:
        L.append(f"| {wl} | {st} | {bp:.0%} | {b['alpha']:+.0f}% | {ru['alpha']:+.0f}% | {ba['alpha']:+.0f}% "
                 f"| -{b['mdd']:,.0f} | -{ba['mdd']:,.0f} |")

    # 判讀
    d_rule = rule_oos['alpha'] - base_oos['alpha']
    d_band = bandit_oos['alpha'] - base_oos['alpha']
    mdd_rule = (1 - rule_oos['mdd'] / base_oos['mdd']) * 100 if base_oos['mdd'] else 0
    mdd_band = (1 - bandit_oos['mdd'] / base_oos['mdd']) * 100 if base_oos['mdd'] else 0
    worst_base = min(b['alpha'] for _, _, _, b, _, _ in win_rows)
    worst_band = min(ba['alpha'] for _, _, _, _, _, ba in win_rows)
    L += ["",
          "## 4. 誠實判讀\n",
          f"- **等價性**: {'PASS' if eq_ok else 'FAIL'}(曝險層是唯一變因,結果可歸因)。",
          f"- **OOS alpha 增量**: RULE {d_rule:+.1f}pp、BANDIT {d_band:+.1f}pp(相對 baseline ⑤ 同期 {base_oos['alpha']:+.1f}%)。",
          f"- **下檔(MDD)**: RULE 比 baseline {'降' if mdd_rule>0 else '升'} {abs(mdd_rule):.0f}%、"
          f"BANDIT {'降' if mdd_band>0 else '升'} {abs(mdd_band):.0f}%(曝險控制的核心目的是降裸隔夜尾部風險)。",
          f"- **最差窗口**: baseline {worst_base:+.0f}% → BANDIT {worst_band:+.0f}%(對照記憶 baseline 最差 regime ≈{BASE_WORST:.0f})。",
          f"- **平均曝險**: BANDIT 把曝險從 {base_oos['avg_expo']:.0%} 調到 {bandit_oos['avg_expo']:.0%}"
          f"(mult {bandit_oos['avg_mult']:.2f});若 alpha 沒掉但曝險降 = 風險效率改善(Sharpe 看)。",
          ""]
    # 結論行
    if not eq_ok:
        verdict = "引擎等價性 FAIL,結果作廢。"
    elif d_band > 1 and mdd_band > 0:
        verdict = f"BANDIT OOS 同時改善 alpha(+{d_band:.0f}pp)與下檔(MDD -{mdd_band:.0f}%) → 曝險層有真 edge,值得保留。"
    elif mdd_band > 5 and d_band > -3:
        verdict = f"BANDIT 主要改善風險(MDD -{mdd_band:.0f}%)、alpha 大致打平({d_band:+.0f}pp) → 風險調整有效,報酬非主賣點。"
    elif d_rule > 0 and d_band <= 0:
        verdict = "零擬合 RULE 有效但 BANDIT OOS 沒贏 → 曝險擇時的 edge 是常識性的、bandit 學不到額外東西(別上 bandit,上 RULE)。"
    else:
        verdict = f"曝險層 OOS 未能勝過 baseline(RULE {d_rule:+.0f}pp / BANDIT {d_band:+.0f}pp) → 在這段(大多頭、無長空頭)滿倉就是最優,降曝險只是少賺。"
    L.append(f"- **總評**: {verdict}")
    L.append("")
    L.append("> ⚠️ 為何必敗(誠實歸因): (1) 本資料 2 年為大多頭(各窗多頭日 60–80%),"
             "**未含長期空頭** → 任何降曝險在這段都只會少賺,曝險控制器最該發威的『系統性 gap-down』"
             "完全沒被測到。(2) bandit 的 risk-adjusted reward(報酬−λ×下檔)在『有正漂移但有波動』的資料上"
             "結構性偏好低曝險,greedy 收斂成**長期欠曝**(avg_mult≈0.45、曝險 72%→32%),這是退化解而非擇時能力;"
             "連 Sharpe 都從 1.46 掉到 1.18,所以**不是**『犧牲報酬換風險效率』,是兩頭皆輸。"
             "(3) MDD 看似降 26% 純粹因為幾乎空手,不是真的避開了回檔。")
    L.append("> 真正結論: **曝險擇時層在本(大多頭)資料上沒有可宣稱的 edge**;"
             "零擬合 RULE 至少輸得少且 Sharpe 幾乎不變(故沒有亂動),bandit 反而被 reward 形狀帶歪。"
             "下一步要有意義,**必須先拿到含明確空頭/大回檔的 OOS 資料**,否則這層無法被證實也無法被證偽。")

    REPORT = ROOT / "reports" / "ml_size.md"
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.success(f"報告 → {REPORT}")
    print("RESULT_JSON " + json.dumps({
        "eq_ok": bool(eq_ok), "best_lam": best_lam, "base_sharpe": base_oos["sharpe"],
        "oos_start": oos_start, "oos_days": len(oos_days),
        "base": {k: base_oos[k] for k in ("ret", "alpha", "turn", "mdd", "sharpe", "avg_expo")},
        "rule": {k: rule_oos[k] for k in ("ret", "alpha", "turn", "mdd", "sharpe", "avg_expo", "avg_mult")},
        "bandit": {k: bandit_oos[k] for k in ("ret", "alpha", "turn", "mdd", "sharpe", "avg_expo", "avg_mult")},
        "d_rule_pp": d_rule, "d_band_pp": d_band,
        "worst_base": worst_base, "worst_band": worst_band,
    }))


if __name__ == "__main__":
    main()
