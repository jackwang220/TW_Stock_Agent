"""擴充 universe(偏科技):按產業完整收錄科技子產業 + 流動性下限 2000萬,併入現有112。
解決「用成交值排名建池子會漏掉還沒噴的薄族群」——改成按子產業完整收,只用低門檻濾微型股。
輸出 data/base_universe_v2.json(不覆蓋現有,驗證後再換)。背景跑,逐檔抓、可重跑(cache)。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools import finmind_client as fc
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

TECH_IND = {"半導體業", "電子零組件業", "電腦及週邊設備業", "光電業", "通信網路業",
            "其他電子業", "電子通路業", "其他電子類", "電子工業", "資訊服務業"}
LIQ_MIN = 2e7          # 日均成交金額下限 2000萬
START = "2021-01-01"
OUT = DATA_DIR / "base_universe_v2.json"


def main():
    base = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    print(f"現有 universe {len(base)} 檔(全保留)")

    info = fc._fetch_api("TaiwanStockInfo", "", "", "")
    cand = {}
    for r in info:
        sid = str(r.get("stock_id", "")); ind = r.get("industry_category", "")
        if r.get("type") not in ("twse", "tpex"): continue
        if ind not in TECH_IND: continue
        if not (len(sid) == 4 and sid.isdigit()): continue       # 排除 ETF(00xx是4位但…)/權證/DR
        if sid.startswith("00"): continue                         # ETF
        cand[sid] = {"name": r.get("stock_name", sid), "industry": ind,
                     "market": "TWSE" if r["type"] == "twse" else "TPEX"}
    todo = [c for c in cand if c not in base]
    print(f"科技子產業候選 {len(cand)} 檔,其中 {len(todo)} 檔不在現有池 → 待抓篩")

    uni = dict(base)                       # 先放現有(全保留)
    kept = 0
    for i, c in enumerate(todo, 1):
        try:
            o = get_daily_ohlcv(c, start=START)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {c} 抓取失敗: {e}"); continue
        if not o or len(o) < 120:
            pass
        else:
            ds = sorted(o)[-120:]
            avg = mean(o[d].get("amount", 0) for d in ds)
            if avg >= LIQ_MIN:
                uni[c] = {"name": cand[c]["name"], "market": cand[c]["market"],
                          "industry": cand[c]["industry"], "avg_turnover": avg, "source": "tech-expand"}
                kept += 1
        if i % 50 == 0:
            print(f"  進度 {i}/{len(todo)}｜已收 {kept} 檔｜目前池 {len(uni)}")
            OUT.write_text(json.dumps(uni, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(0.15)

    OUT.write_text(json.dumps(uni, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n完成:新增 {kept} 檔科技股 → 總 {len(uni)} 檔 → {OUT}")
    # 產業分佈
    from collections import Counter
    cnt = Counter(v.get("industry", "?") for v in uni.values())
    print("產業分佈:")
    for k, n in cnt.most_common(15):
        print(f"  {n:>3}  {k}")


if __name__ == "__main__":
    main()
