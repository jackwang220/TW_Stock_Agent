"""跨股類比預測引擎(kNN / analog forecasting)— Phase 2 原型。

你的「記憶起來、去找其他類似情況」= k-近鄰。本檔證明整套機制:
  1. 把每個(股票,某天)編碼成「情境指紋」= 近20日波形 + 條件特徵(自身波動/動能 + 大盤regime)
  2. 查詢時,在全市場歷史庫找最像的 K 個(先條件、再形狀:regime/波動加權)
  3. 看這 K 個歷史案例「後來怎麼走」→ 當預測

護欄(繼承驗證A):
  - 防洩漏:查詢日 D 只配對「結果已在 D 之前完全實現」的歷史窗口
  - regime 用大盤往回看指標機械算(零未來)
  - 純資料、可解釋(會印出實際被匹配到的歷史案例)

⚠️ 原型用現有快取(~225檔, 2024+)證明機制;真正的庫要換成「液性股 + 2016深歷史」。

用法:
    python scripts/analog_engine.py 3711 2026-05-19   # 查日月光在5/19當下,歷史類比怎麼走
    python scripts/analog_engine.py 2330              # 預設用最新一天
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from tw_stock_agent.config import DATA_DIR, TW_STOCK_INDEX

CACHE = DATA_DIR / "finmind_cache"
WIN = 20          # 波形:近 N 日日報酬
HORIZON = 10      # 往後記錄幾天
K = 50            # 取最像的 K 個
CTX_W = 3.0       # 條件特徵(波動/動能/regime)相對波形的權重(>1 = 更重視「情境相同」)


def load_panel() -> dict[str, list]:
    # 庫限定在基本盤(液性股深歷史);沒有就用全部快取
    base_f = DATA_DIR / "base_universe.json"
    only = set(json.loads(base_f.read_text(encoding="utf-8"))) if base_f.exists() else None
    panel = {}
    for f in glob.glob(str(CACHE / "TaiwanStockPrice_*.json")):
        code = Path(f).stem.replace("TaiwanStockPrice_", "")
        if only is not None and code not in only:
            continue
        try:
            rows = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        ser = []
        for r in rows:
            c = r.get("close")
            try:
                c = float(c)
            except (TypeError, ValueError):
                continue
            if c > 0:
                ser.append((r["date"], c))
        ser.sort()
        if len(ser) > WIN + HORIZON + 200:
            panel[code] = ser
    return panel


def build_regime(panel: dict) -> dict[str, tuple]:
    """大盤 regime(每日,只用往回看):趨勢 / 波動 / 回撤。零未來資訊。"""
    rets = defaultdict(list)
    for ser in panel.values():
        for i in range(1, len(ser)):
            if ser[i - 1][1] > 0:
                rets[ser[i][0]].append(ser[i][1] / ser[i - 1][1] - 1)
    dates = sorted(rets)
    med = np.array([np.median(rets[d]) for d in dates])
    idx = np.cumprod(1 + med)
    out = {}
    for k, d in enumerate(dates):
        if k < 200:
            out[d] = None
            continue
        trend = idx[k] / idx[k - 200:k].mean() - 1
        vol = float(np.std(med[k - 20:k]) * np.sqrt(252))
        ddown = idx[k] / idx[k - 60:k + 1].max() - 1
        out[d] = (float(trend), vol, float(ddown))
    return out


def encode(ser: list, i: int, regime: dict):
    """(股票, 第i天) → 情境向量 + 後續逐日報酬 + 結果完成日。資料不足回 None。"""
    if i < max(WIN, 60) or regime.get(ser[i][0]) is None:
        return None
    closes = [c for _, c in ser]
    shape = [closes[i - WIN + 1 + j] / closes[i - WIN + j] - 1 for j in range(WIN)]
    own_vol = float(np.std(shape) * np.sqrt(252))
    own_mom = closes[i] / closes[i - 60] - 1
    tr, vo, dd = regime[ser[i][0]]
    vec = np.array(shape + [own_vol, own_mom, tr, vo, dd], dtype=float)
    fwd, end = None, None
    if i + HORIZON < len(ser):
        fwd = [closes[i + h] / closes[i + h - 1] - 1 for h in range(1, HORIZON + 1)]
        end = ser[i + HORIZON][0]
    return vec, fwd, end


def build_library(panel: dict, regime: dict):
    vecs, fwds, meta = [], [], []
    for code, ser in panel.items():
        for i in range(max(WIN, 60), len(ser) - HORIZON):
            e = encode(ser, i, regime)
            if e is None or e[1] is None:
                continue
            vecs.append(e[0]); fwds.append(e[1]); meta.append((code, ser[i][0], e[2]))
    V = np.array(vecs); F = np.array(fwds)
    mu, sd = V.mean(0), V.std(0)
    sd[sd == 0] = 1.0
    Vz = (V - mu) / sd
    return Vz, F, meta, (mu, sd)


def query(code, as_of, panel, regime, Vz, F, meta, scaler):
    if code not in panel:
        return None, f"{code} 不在快取庫(可能未抓或非液性)"
    ser = panel[code]
    # 找 as_of(或之前最近一天)的 index
    idxs = [k for k, (d, _) in enumerate(ser) if d <= as_of]
    if not idxs:
        return None, f"{code} 在 {as_of} 前無資料"
    i = idxs[-1]
    e = encode(ser, i, regime)
    if e is None:
        return None, f"{code} 在 {as_of} 資料不足以編碼(需 ≥{max(WIN,60)} 日歷史 + regime)"
    mu, sd = scaler
    q = (e[0] - mu) / sd
    # 權重:波形 1、條件 CTX_W
    w = np.ones(len(q)); w[WIN:] = CTX_W
    d = np.sqrt(((Vz - q) ** 2 * w).sum(1))
    # 防洩漏 + 排除自己鄰近窗口
    qdate = ser[i][0]
    valid = np.array([(end is not None and end < as_of and not (c == code and abs(_days(dt, qdate)) < 30))
                      for (c, dt, end) in meta])
    d = np.where(valid, d, np.inf)
    order = np.argsort(d)[:K]
    return (i, qdate, order, d), None


def _days(a: str, b: str) -> int:
    from datetime import date
    return (date.fromisoformat(a) - date.fromisoformat(b)).days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("as_of", nargs="?", default="2026-06-10")
    args = ap.parse_args()

    print("載入快取庫...", flush=True)
    panel = load_panel()
    regime = build_regime(panel)
    Vz, F, meta, scaler = build_library(panel, regime)
    print(f"庫:{len(panel)} 檔、{len(meta)} 個歷史情境窗口\n", flush=True)

    res, err = query(args.code, args.as_of, panel, regime, Vz, F, meta, scaler)
    if err:
        print("查詢失敗:", err); return
    i, qdate, order, dist = res
    idx_name = json.loads(TW_STOCK_INDEX.read_text(encoding="utf-8")).get(args.code, {}).get("name", args.code)
    tr, vo, dd = regime[qdate]
    print(f"=== 查詢:{args.code} {idx_name} @ {qdate} ===")
    print(f"當下大盤 regime:趨勢{tr*100:+.1f}%(對200MA) 波動{vo*100:.0f}% 回撤{dd*100:.1f}%(距60日高)\n")

    print(f"最像的 {len(order)} 個歷史情境(跨股):")
    nm = json.loads(TW_STOCK_INDEX.read_text(encoding="utf-8"))
    for j in order[:12]:
        c, dt, _ = meta[j]
        path = "　".join(f"{F[j][h]*100:+.0f}" for h in range(min(5, HORIZON)))
        print(f"  {c} {nm.get(c,{}).get('name',c)[:5]:6s} @{dt}  距離{dist[j]:.2f}  後5日:{path}")

    # 聚合:這 K 個後來怎麼走
    sub = F[order]
    print(f"\n=== 類比預測(K={len(order)} 個案例的後續分布)===")
    cum = 0
    for h in range(HORIZON):
        day = sub[:, h]
        win = (day > 0).mean()
        cum_path = (np.prod(1 + sub[:, :h + 1], axis=1) - 1).mean()
        print(f"  D+{h+1}: 當日均{day.mean()*100:+.2f}%  勝率{win*100:.0f}%  累積均{cum_path*100:+.2f}%")
    # 連漲維持天數
    streak = [(np.argmax(np.append(row <= 0, True))) for row in sub]
    print(f"  連漲(>0)維持:平均 {np.mean(streak):.1f} 天、中位 {int(np.median(streak))} 天")

    # 若查詢日在過去,印出實際後續對照
    if i + HORIZON < len(panel[args.code]):
        closes = [c for _, c in panel[args.code]]
        actual = [closes[i + h] / closes[i + h - 1] - 1 for h in range(1, 6)]
        print(f"\n  (對照)該股實際後5日:{'　'.join(f'{x*100:+.0f}' for x in actual)}")


if __name__ == "__main__":
    main()