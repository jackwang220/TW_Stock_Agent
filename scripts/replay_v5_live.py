"""⑤策略 live replay:從 START 起逐日模擬「上線後的每次買賣」(真實開收盤價)。

執行(⑤c,多窗口最優):買在收盤、賣在隔日開盤、exit_only(只賣掉出名單、名單內抱住)、零股。
資金:固定可投入 capital,曝險×權重(同 dual_engine_targets / 回測)。

時間線(逐日):
  d 收盤   : 買進 d 名單未達目標的(close[d])
  d+1 開盤 : 賣掉「不在 d 名單」的持倉(open[d+1])
FinMind 最新到 6/11,所以 6/10/6/11 可用真實價;6/12(今日未收盤)用 --live-px 抓即時價估。

用法:
  python scripts/replay_v5_live.py --capital 50000
  python scripts/replay_v5_live.py --capital 50000 --live-px   # 6/12 用 Shioaji 即時價補
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

_v5 = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("v5", ROOT / "scripts/exp_step1_v5.py"))
importlib.util.spec_from_file_location("v5", ROOT / "scripts/exp_step1_v5.py").loader.exec_module(_v5)
features, _factors = _v5.features, _v5._factors

START = "2026-05-29"
TODAY = "2026-06-12"
INITIAL, DAILY, MAXC = 15000.0, 1000.0, 50000.0   # 第一天15000、每日+1000、累計達5萬停止加碼
FEE_BUY, FEE_SELL, SLIP = 0.001425, 0.004425, 0.001   # 手續費 買0.14%/賣0.44%(含證交稅) + 滑價0.1%
MAX_SIG, EXPO_CAP, EXPO_FLOOR, TIE = 3, 0.90, 0.30, 0.90


def fetch_live_bars(codes: list[str]) -> dict[str, dict]:
    """登入 Shioaji,批次抓即時 snapshot,組成今日 OHLCV bar(close=即時價)。"""
    import os, time
    import shioaji as sj
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    api = sj.Shioaji(simulation=True)
    print("登入 Shioaji 抓即時報價 ...")
    api.login(api_key=os.environ["SINOPAC_APIKEY"], secret_key=os.environ["SINOPAC_SECRETKEY"])
    try:
        api.fetch_contracts(contract_download=True)
    except Exception:
        pass
    time.sleep(1.0)
    cts = []
    for c in codes:
        ct = api.Contracts.Stocks.get(c) if hasattr(api.Contracts.Stocks, "get") else api.Contracts.Stocks[c]
        if ct is not None:
            cts.append((c, ct))
    bars: dict[str, dict] = {}
    # 分批 snapshot(每批 ≤100)
    for i in range(0, len(cts), 100):
        batch = cts[i:i + 100]
        try:
            snaps = api.snapshots([ct for _, ct in batch])
        except Exception as e:
            print(f"  snapshot 批次失敗: {e}"); continue
        for (c, _), sn in zip(batch, snaps):
            close = float(getattr(sn, "close", 0) or 0)
            if close <= 0:
                continue
            bars[c] = {
                "open":   float(getattr(sn, "open", 0) or close),
                "high":   float(getattr(sn, "high", 0) or close),
                "low":    float(getattr(sn, "low", 0) or close),
                "close":  close,
                "volume": float(getattr(sn, "total_volume", 0) or 0),
                "amount": float(getattr(sn, "total_amount", 0) or 0),
            }
        time.sleep(0.5)
    api.logout()
    print(f"  抓到 {len(bars)} 檔即時報價")
    return bars


def h_score(ff, tp):
    if ff is None:
        return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35 * t + 0.35 * rs + 0.15 * vo + 0.10 * ri + 0.05 * ma) * 100 * (0.8 + 0.4 * tp)


def picks_on(d, equity, codes, feats, twii_feat, OH, turns, names):
    """算決策日 d 的名單 + 目標金額(基於 equity)。回 (regime, [(code,score,target)...])。"""
    if d not in twii_feat:
        return None, []
    bull = bool(twii_feat[d].get("close") and twii_feat[d].get("ma20")
                and twii_feat[d]["close"] > twii_feat[d]["ma20"])
    ir = twii_feat[d].get("ret20")
    vals = sorted(((c, feats[c][d]["turn"]) for c in codes
                   if d in feats.get(c, {}) and feats[c][d].get("turn", 0) > 0), key=lambda x: x[1])
    tp = {c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)} if vals else {}
    scored = []
    for c in codes:
        f = feats.get(c, {})
        if d not in f or math.isnan(f[d].get("ma20", float("nan"))):
            continue
        if bull:
            sc = h_score(_factors(f[d], ir), tp.get(c, 0.5))
        else:
            closes = [OH[c][x]["close"] for x in sorted(OH[c]) if x <= d]
            sig = rebound_signal(closes, turns.get(c, 0.0))
            sc = sig["score"] * 100 if sig.get("fired") else 0.0
        if sc > 0:
            scored.append((sc, c))
    scored.sort(reverse=True)
    sel = scored[:MAX_SIG]
    if len(scored) > MAX_SIG and scored[MAX_SIG][0] >= scored[MAX_SIG - 1][0] * TIE:
        sel = scored[:MAX_SIG + 1]
    if not sel:
        return ("bull" if bull else "bear"), []
    avg = sum(s for s, _ in sel) / len(sel) / 100
    expo = min(EXPO_CAP, max(EXPO_FLOOR, avg))
    ssum = sum(s for s, _ in sel)
    out = [(c, s, round(equity * expo * (s / ssum))) for s, c in sel]
    return ("bull" if bull else "bear"), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50000)
    ap.add_argument("--live-px", action="store_true",
                    help="抓今日 Shioaji 即時報價,讓 6/12 用盤中價算今日訊號(否則 6/12 用昨收名單預估)")
    args = ap.parse_args()
    CAP = args.capital

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys())
    names = {c: u[c].get("name", c) for c in codes}
    turns = {c: u[c].get("avg_turnover", 0.0) for c in codes}
    print(f"載入行情({len(codes)} 檔,本地快取)...")
    OH = {c: get_daily_ohlcv(c) for c in codes}
    OH["0050"] = get_daily_ohlcv("0050")

    # ── 即時價:抓今日 snapshot,灌成 TODAY 的 bar,讓 6/12 用盤中即時訊號 ──
    if args.live_px:
        live = fetch_live_bars(codes + ["0050"])
        n_inject = 0
        for c, bar in live.items():
            if c in OH and TODAY not in OH[c]:
                OH[c][TODAY] = bar
                n_inject += 1
        print(f"  灌入 {TODAY} 即時 bar: {n_inject} 檔（close=盤中即時價當今日收盤近似）")

    # features=v3.features,內部 oh() 讀 v3 模組的 _OH;灌即時必須設到 v3._OH 才生效
    _v5._OH = OH
    _v5.v3._OH = OH
    twii_feat = features("0050")
    feats = {c: features(c) for c in codes}

    alld = sorted({d for c in codes for d in OH[c]})
    days = [d for d in alld if d >= START]
    if not days:
        print(f"❌ {START} 起無交易日"); return 1
    print(f"Replay 交易日: {days}（資金 {INITIAL:,.0f}+{DAILY:,.0f}/日 上限{MAXC:,.0f}、⑤c 買收盤/賣開盤/exit_only/零股）\n")

    def cls(c, d):
        return OH[c].get(d, {}).get("close")

    def opn(c, d):
        return OH[c].get(d, {}).get("open")

    shares: dict[str, float] = {}   # code -> 股數
    cash = contributed = prev_eq = 0.0
    summary = []   # 每日一行:(date, regime, buy_str, sell_str, pnl, pnl_pct, contributed, equity)
    detail: list[str] = []

    for i, d in enumerate(days):
        # ── 加碼:第一天 15000,其後每日 +1000,累計達 5 萬停止 ──
        add = INITIAL if i == 0 else (min(DAILY, MAXC - contributed) if contributed < MAXC else 0.0)
        cash += add
        contributed += add

        equity = cash + sum(sh * (cls(c, d) or 0) for c, sh in shares.items())
        regime, picks = picks_on(d, equity, codes, feats, twii_feat, OH, turns, names)
        sel_codes = {c for c, _, _ in picks}
        eng = "多頭→H動能" if regime == "bull" else "空頭→反彈"
        detail.append(f"## {d}（{eng}，加碼 +{add:,.0f}，決策權益 {equity:,.0f}）\n")

        # ── 開盤賣:賣掉「不在今日名單」的持倉(exit_only) ──
        sells = []
        for c in list(shares):
            if c in sel_codes:
                continue
            op = opn(c, d)
            if not op:
                continue
            val = shares[c] * op
            cash += val * (1 - FEE_SELL - SLIP)   # 賣出扣手續費+證交稅+滑價
            sells.append((c, shares[c], op, val))
            shares.pop(c, None)
        if sells:
            detail.append(f"**開盤賣出（掉出名單，{d} 開盤）**")
            for c, sh, op, val in sells:
                detail.append(f"- 賣 {c} {names.get(c,c)[:5]} {sh:.0f}股 @ {op} = {val:,.0f}")
        elif i > 0:
            detail.append("**開盤賣出**：無（持倉都還在名單內）")

        # ── 收盤買:名單內未達目標的,買到目標(零股,close[d]) ──
        buys = []
        for c, s, t in sorted(picks, key=lambda x: -x[1]):
            cp = cls(c, d)
            if not cp or cp <= 0:
                continue
            cur_val = shares.get(c, 0.0) * cp
            buy_val = t - cur_val
            if buy_val < 1000:
                continue
            qty = int(buy_val / cp)
            if qty < 1:
                continue
            unit_cost = cp * (1 + FEE_BUY + SLIP)   # 每股含手續費+滑價
            cost = qty * unit_cost
            if cost > cash:
                qty = int(cash / unit_cost)
                cost = qty * unit_cost
                if qty < 1:
                    continue
            cash -= cost
            shares[c] = shares.get(c, 0.0) + qty
            buys.append((c, qty, cp, cost, t))
        detail.append(f"\n**收盤買進（名單內未達標，{d} 收盤）**")
        if buys:
            for c, qty, cp, cost, t in buys:
                detail.append(f"- 買 {c} {names.get(c,c)[:5]} {qty}股 @ {cp} = {cost:,.0f}")
        else:
            detail.append("- 無（名單內都已達標或無訊號）")

        # ── 收盤估值 + 當日損益(扣掉當天加碼才是真實賺賠)──
        inv = sum(sh * (cls(c, d) or 0) for c, sh in shares.items())
        eq2 = cash + inv
        pnl = eq2 - prev_eq - add
        pnl_pct = (pnl / prev_eq * 100) if prev_eq > 1e-6 else (pnl / add * 100 if add else 0.0)
        prev_eq = eq2
        hold_str = "、".join(f"{c}({names.get(c,c)[:4]}){sh:.0f}" for c, sh in shares.items()) or "—"
        detail.append(f"\n**{d} 收盤後**：持倉 {hold_str}｜投入市值 {inv:,.0f}｜現金 {cash:,.0f}｜權益 **{eq2:,.0f}**\n")

        buy_str = "、".join(f"{names.get(c,c)[:3]}{qty}" for c, qty, _, _, _ in buys) or "—"
        sell_str = "、".join(f"{names.get(c,c)[:3]}" for c, _, _, _ in sells) or "—"
        summary.append((d, eng, buy_str, sell_str, pnl, pnl_pct, contributed, eq2))

    # ── 組裝報告:標題 + 摘要表(每天一行) + 逐日明細 ──
    final = summary[-1] if summary else None
    tot_pnl = final[7] - final[6] if final else 0.0
    tot_pct = (tot_pnl / final[6] * 100) if final and final[6] else 0.0
    L = [f"# ⑤策略 Live Replay — 從進場日 {days[0]} 開始（到 {days[-1]}）\n",
         f"> 資金:第一天 {INITIAL:,.0f}、每日 +{DAILY:,.0f}、累計達 {MAXC:,.0f} 停止｜"
         f"⑤c:買收盤/賣隔日開盤/exit_only/零股｜大盤站上20MA打H動能、跌破打反彈\n",
         f"> **結算:累計投入 {final[6]:,.0f}｜期末權益 {final[7]:,.0f}｜淨損益 {tot_pnl:+,.0f}（{tot_pct:+.2f}%）**\n",
         "## 每日總表\n",
         "| 日期 | 行情 | 買進 | 賣出 | 當日漲跌% | 當日損益 | 累計投入 | 總權益 |",
         "|------|------|------|------|----------|---------|---------|--------|"]
    for d, eng, bs, ss, pnl, pct, contrib, eq in summary:
        L.append(f"| {d} | {eng} | {bs} | {ss} | {pct:+.2f}% | {pnl:+,.0f} | {contrib:,.0f} | {eq:,.0f} |")
    L.append("")
    if args.live_px and days[-1] == TODAY:
        L.append(f"> ⚠️ 最後一天 {TODAY} 用**盤中即時價**算（收盤前會變;真要下單 13:00 後重跑 `live_dual_v5_trade.py`）。\n")
    L.append("---\n## 逐日明細\n")
    L += detail

    rpt = ROOT / "reports" / "replay_v5_live.md"
    rpt.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:3 + len(summary) + 6]))   # 終端只印摘要表
    print(f"\n✅ 完整報告（含逐日明細）→ {rpt}")


if __name__ == "__main__":
    sys.exit(main() or 0)
