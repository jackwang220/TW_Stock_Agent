"""深抓 base_universe 112 檔的三大法人 + 月營收歷史回 2021(供跨期間驗證,修2022空缺)。"""
import sys, json, time
sys.path.insert(0, "src"); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR
from tw_stock_agent.tools.finmind_client import _get_data

START = "2021-01-01"
codes = list(json.loads((DATA_DIR / "base_universe.json").read_text(encoding="utf-8")))
print(f"深抓 {len(codes)} 檔 法人+營收(start={START})...")
inst_ok = rev_ok = 0
for i, code in enumerate(codes, 1):
    inst = _get_data("TaiwanStockInstitutionalInvestorsBuySell", code, start=START, force_refresh=True)
    rev = _get_data("TaiwanStockMonthRevenue", code, start=START, force_refresh=True)
    inst_ok += int(bool(inst) and any(r.get("date", "") <= "2022-06-30" for r in inst))
    rev_ok += int(bool(rev))
    if i % 20 == 0 or i == len(codes):
        print(f"  [{i}/{len(codes)}] {code} 法人{len(inst)}筆 營收{len(rev)}筆")
    time.sleep(0.25)
print(f"完成:{inst_ok} 檔法人有2022前資料,{rev_ok} 檔有營收")