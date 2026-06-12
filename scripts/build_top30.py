"""產出 0050 前30大代理(日均成交額排名)→ data/universe_top30.json"""
import sys, json
sys.path.insert(0, "src"); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR

base = json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8"))
idx = json.loads((DATA_DIR / "tw_stock_index.json").read_text(encoding="utf-8"))
# 名稱字典(支援 list 或 dict 結構)
names = {}
if isinstance(idx, dict):
    for k, v in idx.items():
        names[k] = (v.get("name") if isinstance(v, dict) else v) or ""
elif isinstance(idx, list):
    for v in idx:
        names[str(v.get("code") or v.get("stock_id"))] = v.get("name", "")

ranked = sorted(base.items(), key=lambda kv: kv[1].get("avg_turnover", 0), reverse=True)[:30]
out = {}
print(f"{'#':>2} {'代號':<6}{'名稱':<10}{'日均成交額(億)':>12}")
for i, (code, v) in enumerate(ranked, 1):
    nm = names.get(code, "?")
    turn = v.get("avg_turnover", 0)
    out[code] = {"name": nm, "avg_turnover": turn}
    print(f"{i:>2} {code:<6}{nm:<10}{turn/1e8:>12.1f}")
(DATA_DIR / "universe_top30.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n已存 {len(out)} 檔 → data/universe_top30.json")
