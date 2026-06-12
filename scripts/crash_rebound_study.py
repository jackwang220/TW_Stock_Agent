"""驗證 A：跨股「暴跌→反彈」研究（純價格數據，無 LLM）。

回答的問題：台股暴跌後,隔日進場「連續每天大漲」的現象是真 edge 還是錯覺?
維持幾天?只存在大型股還是全市場都有?

設計重點（與使用者討論定案）：
  - 全市場(tw_stock_index.json 全普通股),不只前幾檔 → 避免樣本太少。
  - 逐日記錄每天漲幅 + 連漲維持天數 + 逐日續強機率(不是攤平的 N 日總報酬)。
  - 三種暴跌定義：跌停 / 5日-12% / 5日-20%。
  - 現實進場：暴跌隔日「開盤」進場;隔日鎖漲停買不到 → 單獨標記。
  - Alpha：每日減去全市場中位數日報酬(長天期防 beta 騙人)。
  - 分層：流動性(成交額)大/中/小、市場別 → 直接看「會不會被稀釋」。
  - 防洩漏：暴跌判斷只用 ≤當日;後續報酬只用未來。

使用方式：
    python scripts/crash_rebound_study.py            # 全市場(會先抓未快取的 OHLCV)
    python scripts/crash_rebound_study.py --limit 300 # 只跑前 300 檔(測試)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median, mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # 避免 cp950 輸出錯誤
except Exception:
    pass

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")   # 關掉 FinMind 每檔 DEBUG 噪音

from tw_stock_agent.config import TW_STOCK_INDEX, REPORTS_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

OUT_MD = REPORTS_DIR / "crash_rebound_study.md"

START = "2024-06-01"          # 研究起始(FinMind 快取從 2024)
MAX_FWD = 45                  # 最多追蹤未來 45 個交易日
LIMIT_DOWN = -0.095          # 跌停門檻(容錯)
LIMIT_UP = 0.094             # 漲停門檻
COOLDOWN = 10                # 同股同觸發,10 交易日內不重複計
EXAMPLES = {"2059": "川湖", "2327": "國巨", "3711": "日月光投控"}


def fetch_panel(codes: list[str]) -> dict[str, dict]:
    """抓所有股票的 OHLCV(複用快取,未快取的逐檔抓 + 節流/退避)。"""
    panel = {}
    n = len(codes)
    for i, code in enumerate(codes, 1):
        for attempt in range(3):
            try:
                d = get_daily_ohlcv(code, start=START)
                if d:
                    panel[code] = d
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if i % 100 == 0:
            print(f"  OHLCV {i}/{n} ... (panel={len(panel)})", flush=True)
        time.sleep(0.05)
    return panel


def series(ohlcv: dict) -> tuple[list, list, list, list, list]:
    """回傳 (dates, closes, highs, lows, opens) 依日期升冪。"""
    ds = sorted(ohlcv)
    return (ds,
            [ohlcv[d]["close"] for d in ds],
            [ohlcv[d]["high"] for d in ds],
            [ohlcv[d]["low"] for d in ds],
            [ohlcv[d]["open"] for d in ds])


def build_market_median(panel: dict) -> dict[str, float]:
    """每個交易日的全市場『中位數日報酬』,當 Alpha 的大盤基準。"""
    by_date: dict[str, list] = defaultdict(list)
    for code, oh in panel.items():
        ds, cl, *_ = series(oh)
        for i in range(1, len(cl)):
            if cl[i - 1] > 0:
                by_date[ds[i]].append(cl[i] / cl[i - 1] - 1)
    return {d: median(v) for d, v in by_date.items() if v}


def detect_events(panel: dict, trigger: str) -> list[dict]:
    """偵測暴跌事件。trigger: 'limit_down' | 'drop5_12' | 'drop5_20'。

    回傳每個事件的逐日資訊(進場=隔日,逐日 close-to-close 報酬序列)。
    """
    events = []
    for code, oh in panel.items():
        ds, cl, hi, lo, op = series(oh)
        n = len(cl)
        if n < 30:
            continue
        last_evt = -999
        for i in range(5, n - 1):     # i=暴跌日;需要 i+1 才能進場
            if cl[i] <= 0 or cl[i - 1] <= 0:
                continue
            # 暴跌判斷(只用 ≤i 資料)
            if trigger == "limit_down":
                crashed = (cl[i] / cl[i - 1] - 1) <= LIMIT_DOWN
                severity = cl[i] / cl[i - 1] - 1
            else:
                if cl[i - 5] <= 0:
                    continue
                thr = -0.12 if trigger == "drop5_12" else -0.20
                cum5 = cl[i] / cl[i - 5] - 1
                crashed = cum5 <= thr
                severity = cum5
            if not crashed:
                continue
            if i - last_evt < COOLDOWN:
                continue
            last_evt = i
            # 進場 = 隔日(i+1);逐日 close-to-close 報酬 r1..rMAX_FWD
            rets = []
            for k in range(1, MAX_FWD + 1):
                j = i + k
                if j >= n or cl[j - 1] <= 0:
                    break
                rets.append(cl[j] / cl[j - 1] - 1)
            if not rets:
                continue
            # 隔日是否鎖漲停(買不到):r1≥漲停 且 當日振幅 <0.5%
            j1 = i + 1
            locked = (rets[0] >= LIMIT_UP and (hi[j1] - lo[j1]) <= 0.005 * cl[i])
            events.append({
                "code": code, "date": ds[i], "severity": severity,
                "rets": rets, "entry_dates": [ds[i + k] for k in range(1, len(rets) + 1)],
                "locked": locked,
            })
    return events


def streak(rets: list[float], thr: float) -> int:
    """從第1天起連續達標(>=thr)的天數。"""
    s = 0
    for r in rets:
        if r >= thr:
            s += 1
        else:
            break
    return s


def agg(events: list[dict], mkt: dict[str, float]) -> dict:
    """彙總一組事件:逐日平均、連漲天數、續強機率、Alpha。"""
    if not events:
        return {"n": 0}
    n = len(events)
    locked = sum(1 for e in events if e["locked"])
    # 逐日平均報酬 + Alpha(每天用當天市場中位數)
    day_raw: dict[int, list] = defaultdict(list)
    day_alpha: dict[int, list] = defaultdict(list)
    for e in events:
        for k, r in enumerate(e["rets"], 1):
            day_raw[k].append(r)
            m = mkt.get(e["entry_dates"][k - 1])
            if m is not None:
                day_alpha[k].append(r - m)
    # 連漲天數
    su = [streak(e["rets"], 0.0) for e in events]      # 上漲
    s3 = [streak(e["rets"], 0.03) for e in events]     # ≥3%
    s5 = [streak(e["rets"], 0.05) for e in events]     # ≥5%
    # 逐日續強機率(以「上漲」定義):P(第k天還在連漲)
    cont = {}
    for k in range(1, 11):
        still = sum(1 for e in events if streak(e["rets"], 0.0) >= k)
        prev = sum(1 for e in events if streak(e["rets"], 0.0) >= k - 1) if k > 1 else n
        cont[k] = still / prev if prev else 0.0
    return {
        "n": n, "locked_pct": locked / n,
        "day_raw": {k: mean(v) for k, v in day_raw.items()},
        "day_alpha": {k: mean(v) for k, v in day_alpha.items()},
        "streak_up": (mean(su), median(su)),
        "streak_3": (mean(s3), median(s3)),
        "streak_5": (mean(s5), median(s5)),
        "win_d1": sum(1 for e in events if e["rets"][0] > 0) / n,
        "cont": cont,
    }


def fmt_path(a: dict, days=(1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45)) -> str:
    dr = a.get("day_raw", {})
    da = a.get("day_alpha", {})
    rows = []
    for k in days:
        if k in dr:
            rows.append(f"D+{k}: 原始{dr[k]*100:+.2f}% / α{da.get(k,0)*100:+.2f}%")
    return "　".join(rows)


def section(title: str, a: dict) -> list[str]:
    if a.get("n", 0) == 0:
        return [f"### {title}\n", "（無事件）\n"]
    L = [f"### {title}（n={a['n']}）\n",
         f"- 隔日勝率(上漲): **{a['win_d1']*100:.0f}%**　｜　隔日鎖漲停買不到: {a['locked_pct']*100:.0f}%",
         f"- 連漲維持天數: 上漲 平均{a['streak_up'][0]:.1f}/中位{a['streak_up'][1]:.0f}天　"
         f"｜≥3% 平均{a['streak_3'][0]:.1f}天　｜≥5% 平均{a['streak_5'][0]:.1f}天",
         "- 逐日續強機率: " + " ".join(f"D{k}={a['cont'][k]*100:.0f}%" for k in range(1, 8)),
         f"- 逐日報酬路徑: {fmt_path(a)}", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 檔(測試)")
    args = ap.parse_args()

    idx = json.loads(TW_STOCK_INDEX.read_text(encoding="utf-8"))
    codes = list(idx.keys())
    if args.limit:
        codes = codes[:args.limit]
    print(f"universe: {len(codes)} 檔,開始抓 OHLCV(複用快取)...", flush=True)
    panel = fetch_panel(codes)
    print(f"取得 {len(panel)} 檔有效資料。建立市場基準...", flush=True)
    mkt = build_market_median(panel)

    # 流動性分層(用平均成交額)
    avg_amt = {}
    for code, oh in panel.items():
        amts = [v["amount"] for v in oh.values() if v["amount"] > 0]
        if amts:
            avg_amt[code] = mean(amts)
    sorted_amt = sorted(avg_amt.values())
    def tier(code):
        a = avg_amt.get(code, 0)
        if not sorted_amt:
            return "中"
        # 用固定門檻(台幣):大>5億、中5000萬~5億、小<5000萬
        if a >= 5e8:
            return "大型(日均成交>5億)"
        if a >= 5e7:
            return "中型(5000萬~5億)"
        return "小型(<5000萬)"

    lines = ["# 驗證 A：暴跌→反彈 跨股研究\n",
             f"> 研究期間 {START} 起｜universe {len(panel)} 檔（全市場普通股）｜"
             f"進場=暴跌隔日｜逐日 close-to-close｜Alpha=減全市場中位數日報酬\n",
             "> ⚠️ 連漲停=買不到(已單獨標記)；長天期請看 Alpha 不看原始(原始含大盤 beta)\n"]

    for trig, label in [("limit_down", "暴跌定義 A：單日跌停(≤-9.5%)"),
                        ("drop5_12", "暴跌定義 B：5 日跌幅 ≤ -12%"),
                        ("drop5_20", "暴跌定義 C：5 日跌幅 ≤ -20%(極端)")]:
        evts = detect_events(panel, trig)
        lines.append(f"\n## {label} — 共 {len(evts)} 個事件\n")
        # 全市場合併(會被稀釋,當對照)
        lines += section("全市場合併（會被稀釋,僅供對照）", agg(evts, mkt))
        # 流動性分層(解決稀釋疑慮)
        lines.append("#### 依流動性分層（這裡看 edge 到底在哪）\n")
        for t in ["大型(日均成交>5億)", "中型(5000萬~5億)", "小型(<5000萬)"]:
            sub = [e for e in evts if tier(e["code"]) == t]
            lines += section(t, agg(sub, mkt))
        # 市場別
        lines.append("#### 依市場別\n")
        for m in ["TWSE", "TPEX"]:
            sub = [e for e in evts if idx.get(e["code"], {}).get("market") == m]
            lines += section(m, agg(sub, mkt))

    # ── 你的範例逐筆 ──
    lines.append("\n## 你的範例:逐筆實際走勢\n")
    for code, nm in EXAMPLES.items():
        if code not in panel:
            lines.append(f"### {code} {nm}：無資料\n")
            continue
        lines.append(f"### {code} {nm}\n")
        found = False
        for trig in ["limit_down", "drop5_12", "drop5_20"]:
            for e in detect_events({code: panel[code]}, trig):
                seq = "　".join(f"D+{k}:{r*100:+.1f}%" for k, r in enumerate(e["rets"][:10], 1))
                lk = "（隔日鎖漲停買不到）" if e["locked"] else ""
                lines.append(f"- [{trig}] 暴跌日 {e['date']}(跌幅{e['severity']*100:.1f}%){lk}")
                lines.append(f"  進場後前10日: {seq}")
                found = True
        if not found:
            lines.append("- (研究期間內無觸發暴跌定義)")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n完成 → {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()