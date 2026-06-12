"""確認 LLM 行為(守門 PASS>REJECT、選股 alpha)在「近半年」+「大股 vs 中小股」是否一致。
不動任何 config(ablate/equal_weight 全預設關)。寫獨立檔 data/_llm_size.csv。"""
import sys, csv, importlib.util
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import numpy as np
sys.path.insert(0, "src"); sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger; logger.remove()
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import get_daily_ohlcv
from tw_stock_agent.debate.bear import run_debate
spec = importlib.util.spec_from_file_location("cpv", "scripts/crossperiod_validate.py")
cpv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cpv)
_finmind_stock = cpv._finmind_stock

LARGE = {"2330": "台積電", "2454": "聯發科", "2408": "南亞科", "2344": "華邦電", "2308": "台達電"}
SMALL = {"9955": "佳龍", "6806": "森崴能源", "2753": "八方雲集", "8499": "鼎炫", "6275": "元山"}
ALL = {**LARGE, **SMALL}
START, END = "2025-12-01", "2026-06-09"
IDX = get_daily_ohlcv("0050")
cal = [d for d in sorted(get_daily_ohlcv("2330")) if START <= d <= END][::2]  # 每隔一天
print(f"近半年取樣 {len(cal)} 天 × {len(ALL)} 檔")

def idx_ret(as_of):
    le = [d for d in IDX if d <= as_of]; gt = [d for d in IDX if d > as_of]
    return (IDX[min(gt)]["close"]/IDX[max(le)]["close"]-1) if le and gt else None

def one(code, as_of):
    stock = _finmind_stock(code, ALL[code], as_of)
    if stock is None: return None
    oh = get_daily_ohlcv(code); fdays = sorted(d for d in oh if d > as_of)
    if not fdays: return None
    ret = oh[fdays[0]]["close"]/stock["close_price"]-1
    ir = idx_ret(as_of); alpha = (ret-ir) if ir is not None else None
    try:
        res = run_debate(stock, [], historical_date=date.fromisoformat(as_of))
    except Exception:
        return None
    return {"tier": "large" if code in LARGE else "small", "code": code, "as_of": as_of,
            "verdict": res.verdict, "dir": res.predicted_direction,
            "ret": ret*100, "alpha": alpha*100 if alpha is not None else None}

out = []
for as_of in cal:
    with ThreadPoolExecutor(max_workers=8) as ex:
        out += [r for r in ex.map(lambda c: one(c, as_of), ALL) if r]
with open(DATA_DIR/"_llm_size.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["tier","code","as_of","verdict","dir","ret","alpha"]); w.writeheader(); w.writerows(out)

# 分析
def stat(rows, label):
    P = [r["alpha"] for r in rows if r["verdict"]=="PASS" and r["alpha"] is not None]
    R = [r["alpha"] for r in rows if r["verdict"]=="REJECT" and r["alpha"] is not None]
    ups = [r["alpha"] for r in rows if r["dir"]=="up" and r["alpha"] is not None]
    upbeat = sum(1 for x in ups if x>0)/len(ups)*100 if ups else 0
    print(f"\n=== {label}(n={len(rows)})===")
    print(f"  守門: PASS α{np.mean(P) if P else 0:+.2f}%(n{len(P)}) vs REJECT α{np.mean(R) if R else 0:+.2f}%(n{len(R)}) → 價差{(np.mean(P)-np.mean(R)) if P and R else 0:+.2f}%")
    print(f"  選股: up預測 α{np.mean(ups) if ups else 0:+.2f}%(n{len(ups)})、勝大盤{upbeat:.0f}%")
stat(out, "全部")
stat([r for r in out if r["tier"]=="large"], "大股")
stat([r for r in out if r["tier"]=="small"], "中小股")
print("\n判讀:PASS明顯>REJECT=守門在;up α>0且勝盤>50%=有選股力(≈50或負=beta)。比大股vs中小股是否一致。")