"""探針:News Co-mention Momentum Spillover(共同提及動能外溢)是否有 forward-return 預測力。

學理:Diamond cuts diamond (J. Banking & Finance 2025, 中國市場) —— 當股票 A 與股票 B
在同一則新聞標題被「共同提及」,B 近期的報酬會外溢預測 A 的未來報酬(cross-firm momentum
透過新聞共現網路傳導)。中文市場(台股最接近)實證最強,且能 unify 其他 cross-firm momentum。

與既有失敗軌的根本差異:
  - 既有軌都用「A 自己的新聞量/情緒/LLM分數」→ 全失敗(台股標題滯後、只報已漲)。
  - 本軌用「A 的新聞『共現夥伴』的近期價格動能」當訊號 → 利用標題的『網路結構』,
    不依賴標題情緒、不依賴 A 自己有沒有新聞。台積電帶 2315 反彈這類標題天然就是 spillover。

本探針只算 rank-IC(訊號 vs forward return),確認訊號存在再做完整回測。
嚴格防洩漏:訊號日 d 用 date < d 的新聞建共現、用 ≤ d 的還原價算夥伴動能;forward = d→d+5 報酬。
(第一輪允許新聞時戳含當日盤後的輕微洩漏,此探針已用 date<d 收緊到不含當日。)
"""
from __future__ import annotations
import sys, json, math, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv

START = "2021-01-01"; END = "2026-06-08"
FWD = 5            # forward horizon (交易日)
COMENT_HALFLIFE = 60   # 共現關係的指數衰減半衰期(交易日)
PARTNER_MOM = 5        # 夥伴動能回看天數
FRESH_DAYS = 10        # 只用近 FRESH_DAYS 天內的「新鮮」共現(spillover 集中在剛共現後)


def load_news(codes):
    out = {}
    cdir = DATA_DIR / "finmind_cache"
    for c in codes:
        p = cdir / f"news_{c}.json"
        m = {}
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                for d, items in j.items():
                    if isinstance(items, list):
                        m[d[:10]] = [it.get("title", "") for it in items]
            except Exception:
                pass
        out[c] = m
    return out


def main():
    u = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
    codes = list(u.keys()); names = {c: u[c].get("name", "") for c in codes}
    print(f"universe {len(codes)} 檔, 載入還原 OHLCV ...", flush=True)
    OH = {c: get_daily_ohlcv(c, start=START) for c in codes}
    alld = sorted({d for c in codes for d in OH[c] if d <= END})
    idx = {d: i for i, d in enumerate(alld)}
    closes = {c: OH[c] for c in codes}

    # 個股近 PARTNER_MOM 日報酬(point-in-time,用 ≤ d)
    ret = {c: {} for c in codes}
    for c in codes:
        ds = sorted(d for d in OH[c] if d <= END)
        cl = [OH[c][d]["close"] for d in ds]
        for j, d in enumerate(ds):
            if j >= PARTNER_MOM and cl[j-PARTNER_MOM] > 0:
                ret[c][d] = cl[j]/cl[j-PARTNER_MOM] - 1
    # forward FWD 日報酬(收盤→收盤,用於 IC,僅評估)
    fwd = {c: {} for c in codes}
    for c in codes:
        ds = sorted(d for d in OH[c] if d <= END)
        cl = [OH[c][d]["close"] for d in ds]
        for j, d in enumerate(ds):
            if j + FWD < len(ds) and cl[j] > 0:
                fwd[c][d] = cl[j+FWD]/cl[j] - 1

    news = load_news(codes)

    # ── 建「每日新聞中的共現邊」:對每股 c 的標題,掃其它股 c2 的名/碼是否出現 → 當日共現(c,c2) ──
    # name 太短(如「金」)會誤命中 → 名長度>=2 才用名比對;碼一律比對。
    name_ok = {c: (len(names[c]) >= 2) for c in codes}
    # 為效率:預先對每檔做「碼+名」token
    tokens = {c: ([c] + ([names[c]] if name_ok[c] else [])) for c in codes}

    # daily_coment[d] = set of frozenset({c, c2}) co-mentioned that day
    # 為控成本,只在 c 有新聞的日子掃。記 per-pair 最後共現日 + 累積權重。
    # spillover 訊號 score[c][d] = Σ_partner  w(c,partner) * partner_recent_ret
    #   w 用「過去共現次數的指數衰減和」(只用 date < d 的共現),partner_recent_ret 用 ret[partner][d_prev]

    # Step 1: 收集所有共現事件 (day, c, partner)
    events = []   # (day, c, partner)
    code_set = set(codes)
    for c in codes:
        for d, titles in news[c].items():
            if not (START <= d <= END):
                continue
            for t in titles:
                for c2 in codes:
                    if c2 == c:
                        continue
                    hit = (c2 in t) or (name_ok[c2] and names[c2] in t)
                    if hit:
                        events.append((d, c, c2))
                        events.append((d, c2, c))   # 對稱:兩檔都登記彼此為夥伴
    print(f"共現事件(對稱後) {len(events)}", flush=True)

    # 以 (c, partner) -> list of co-mention dates
    pair_dates = defaultdict(list)
    for d, c, p in events:
        pair_dates[(c, p)].append(d)
    for k in pair_dates:
        pair_dates[k].sort()

    # Step 2: 逐(訊號日 d, 股 c) 算 spillover score(嚴格 date < d)
    # 候選 (c,d):只在 c 至少有一個共現夥伴、且該股當日有 forward return
    # 為效率,對每個有夥伴的 c,只在 alld 的子集(c 有 fwd 的日子)算
    def prev_trading(d):
        i = idx.get(d)
        return alld[i-1] if (i is not None and i > 0) else None

    # partner set per c
    partners_of = defaultdict(set)
    for (c, p) in pair_dates:
        partners_of[c].add(p)

    sample = []   # (d, score, fwd_ret)
    sig_days = [d for d in alld if "2021-06-01" <= d <= END]
    for c in codes:
        ps = partners_of.get(c)
        if not ps:
            continue
        for d in sig_days:
            if d not in fwd[c]:
                continue
            dp = prev_trading(d)
            if dp is None:
                continue
            num = 0.0; wsum = 0.0; fresh = False
            for p in ps:
                # 共現權重:只用近 FRESH_DAYS 天內的共現(新鮮 spillover),指數衰減
                w = 0.0
                for cd in pair_dates[(c, p)]:
                    if cd >= d:
                        break
                    age = idx[d] - idx.get(cd, idx[d])
                    if age <= FRESH_DAYS:
                        w += math.exp(-age / COMENT_HALFLIFE)
                if w <= 0:
                    continue
                pr = ret[p].get(dp)   # 夥伴近 PARTNER_MOM 日動能(到前一交易日)
                if pr is None:
                    continue
                num += w * pr; wsum += w; fresh = True
            if wsum > 0 and fresh:
                score = num / wsum   # 加權平均夥伴動能(僅新鮮共現)
                sample.append((d, c, score, fwd[c][d]))

    print(f"有效樣本(c,d) {len(sample)}", flush=True)

    # ── 逐日橫斷面 rank-IC(Spearman 近似:用 rank 的 Pearson)──
    byday = defaultdict(list)
    for d, c, s, f in sample:
        byday[d].append((s, f))
    ics = []
    for d, lst in byday.items():
        if len(lst) < 5:
            continue
        ss = [x[0] for x in lst]; ff = [x[1] for x in lst]
        rs = _rank(ss); rf = _rank(ff)
        ic = _pearson(rs, rf)
        if ic is not None:
            ics.append(ic)
    if ics:
        mic = statistics.mean(ics); sic = statistics.pstdev(ics)
        ir = mic / sic * math.sqrt(len(ics)) if sic > 0 else 0
        print(f"\n=== Co-mention spillover rank-IC ===")
        print(f"  日數 {len(ics)}  mean rank-IC {mic:+.4f}  std {sic:.4f}  IC-t≈{ir:+.2f}")
        # 分位數:top-quintile vs bottom-quintile forward return
        allp = [(s, f) for d, c, s, f in sample]
        allp.sort()
        n = len(allp); q = n // 5
        if q > 0:
            bot = statistics.mean(f for s, f in allp[:q])
            top = statistics.mean(f for s, f in allp[-q:])
            print(f"  bottom-quintile fwd{FWD} {bot:+.4%}  top-quintile fwd{FWD} {top:+.4%}  spread {(top-bot):+.4%}")
    else:
        print("無足夠橫斷面樣本")


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0]*len(xs)
    for rank, i in enumerate(order):
        r[i] = rank
    return r


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return None
    ma = sum(a)/n; mb = sum(b)/n
    cov = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    va = sum((x-ma)**2 for x in a); vb = sum((x-mb)**2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    return cov/math.sqrt(va*vb)


if __name__ == "__main__":
    main()
