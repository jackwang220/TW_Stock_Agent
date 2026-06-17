"""單股策略分數查詢:印出某幾檔在「最新交易日」的 B純切 分數 + 因子拆解。
這就是 Discord 訊息裡那個「分數」的同一套算法(H雙引擎B純切)。

用法:
    uv run python scripts/score_one.py 2330 2454 2317
    uv run python scripts/score_one.py 2409        # 看你手上的也行
"""
from __future__ import annotations
import sys, json, importlib.util, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.rebound_signal import rebound_signal
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

_spec = importlib.util.spec_from_file_location("v5", ROOT / "scripts/exp_step1_v5.py")
v5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v5)
features, _factors = v5.features, v5._factors


def h_score(ff, tp):
    if ff is None:
        return 0.0
    t, rs, vo, ri, ma, br, bias = ff
    return (0.35*t + 0.35*rs + 0.15*vo + 0.10*ri + 0.05*ma) * 100 * (0.8 + 0.4*tp)


def main() -> int:
    req = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not req:
        print("用法: uv run python scripts/score_one.py <代號> [代號...]"); return 2

    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    allcodes = list(u)
    names = {c: u[c].get("name", c) for c in allcodes}
    turns_avg = {c: u[c].get("avg_turnover", 0.0) for c in allcodes}
    for c in req:
        names.setdefault(c, c)

    refresh = "--refresh" in sys.argv
    print(f"載入行情({len(allcodes)} 檔,算成交值百分位用){'(查詢股強制更新)' if refresh else ''}...")
    OH = {c: get_daily_ohlcv(c) for c in set(allcodes) - set(req)}
    for c in set(req) | {"0050"}:                      # 查詢股+0050:可 --refresh 抓最新
        OH[c] = get_daily_ohlcv(c, force_refresh=refresh)
    features.__globals__["_OH"] = OH
    twii = features("0050")
    d = max(twii)
    tf = twii[d]
    bull = bool(tf.get("close") and tf.get("ma20") and not math.isnan(tf["ma20"]) and tf["close"] > tf["ma20"])
    ir = tf.get("ret20")

    # 成交值百分位(當日,112 檔內)
    vals = sorted(((c, OH[c][d]["close"] * OH[c][d].get("volume", 0))
                   for c in allcodes if d in OH.get(c, {}) and OH[c][d].get("volume", 0) > 0),
                  key=lambda x: x[1])
    tp_map = {c: (i + 1) / len(vals) for i, (c, _) in enumerate(vals)} if vals else {}

    print(f"\n決策日 {d}｜大盤 {'🟢多頭 → 打 H 動能' if bull else '🔴空頭 → 打反彈'}"
          f"  (B純切:多頭只看H分、空頭只看反彈分×1.5)\n")
    for code in req:
        f = features(code)
        valid = [x for x in sorted(f) if x <= d and not math.isnan(f[x].get("ma20", float("nan")))]
        if not valid:
            print(f"{names.get(code,code)}({code}): 無足夠資料"); continue
        dc = valid[-1]                                  # 對不上決策日 → 用該股自己最新日
        stale = f"  ⚠️用{dc}" if dc != d else ""
        ff = _factors(f[dc], ir)
        tp = tp_map.get(code, 0.5)
        hh = h_score(ff, tp)
        cl = [OH[code][x]["close"] for x in sorted(OH[code]) if x <= dc]
        amts = [OH[code][x].get("amount", 0) for x in sorted(OH[code]) if x <= dc][-120:]
        avg_tn = sum(amts) / len(amts) if amts else turns_avg.get(code, 0.0)   # 真實成交額(池外股也準)
        g = rebound_signal(cl, avg_tn)
        rb = g["score"] * 100 if g.get("fired") else 0.0
        ret5 = (cl[-1] / cl[-6] - 1) * 100 if len(cl) > 5 else float("nan")
        tier = "large" if avg_tn >= 5e8 else ("mid" if avg_tn >= 5e7 else "small")
        sc = hh if bull else rb * 1.5     # B純切
        eng = "H動能" if bull else "反彈"
        print(f"■ {names.get(code,code)}({code})  分數 {sc:.0f}  ←{eng}引擎"
              f"   [H分{hh:.0f} / 反彈分{rb:.0f}]  成交{avg_tn/1e8:.0f}億({tier}){stale}")
        if ff:
            t, rs, vo, ri, ma, br, bias = ff
            print(f"    H因子(0~1):趨勢{t:.2f} 相對強度{rs:.2f} 量{vo:.2f} RSI{ri:.2f} 均線{ma:.2f} 突破{br:.2f}"
                  f"｜成交值百分位{tp:.0%}｜乖離{bias:+.1%}")
        else:
            print(f"    (不在動能上升結構:MA5≤MA20 → H分=0)")
        # 反彈診斷(為什麼觸發/沒觸發)
        bias_pct = (cl[-1]/(sum(cl[-20:])/20) - 1)*100 if len(cl) >= 20 else float("nan")
        if g.get("fired"):
            print(f"    反彈:✅觸發({g.get('depth','')})｜近5日{ret5:+.1f}% 乖離{bias_pct:+.1f}%")
        else:
            why = "小型股不適用" if tier == "small" else f"跌不夠深(近5日{ret5:+.1f}%、乖離{bias_pct:+.1f}%;需近5日≤−12%或乖離≤−10%)"
            print(f"    反彈:✗未觸發 — {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
