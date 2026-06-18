"""手動觀察加減分系統(Phase 1)。對觀察股設 +N 分、維持 X 個交易日。

- 分數加在「排名分」上、**不被 ×INC**(持股黏著只乘原始分);sizing 也算 bonus(同樣不被乘)。
- bonus 可把原本 sc=0 的觀察股拉進候選(只要它在 universe 內、當天有資料)。
- 寫入 data/manual_scores.json,compute_picks 自動讀取,到期(交易日超過 until)自動失效。

用法:
  uv run python scripts/manual_score.py show                 # 看目前生效的加減分
  uv run python scripts/manual_score.py preview 2330 20       # 試算(不寫入):過去5交易日加分後 vs 原本
  uv run python scripts/manual_score.py add 2330 20 5         # 試算→問確定→寫入(維持5個交易日)
  uv run python scripts/manual_score.py add 2330 20 5 --yes   # 跳過確認直接寫(給之後 agent 用)
  uv run python scripts/manual_score.py rm 2330              # 移除
負分也可:  add 2330 -15 5   (減分)
"""
from __future__ import annotations
import sys, json, importlib.util
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from tw_stock_agent.config import DATA_DIR

_spec = importlib.util.spec_from_file_location("v5live", ROOT / "scripts/live_dual_v5_trade.py")
v5live = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v5live)
compute_picks = v5live.compute_picks
MANUAL_FILE = v5live.MANUAL_FILE
get_ohlcv = v5live.get_daily_ohlcv

# ── 交易日曆(算到期日 + 取最近5交易日)──
_HOL = set()
_hp = DATA_DIR / "tw_holidays.txt"
if _hp.exists():
    for ln in _hp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            _HOL.add(ln.split()[0])

def _is_td(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in _HOL

def _forward_td(start: date, n: int) -> date:
    d, cnt = start, 0
    while cnt < n:
        d += timedelta(days=1)
        if _is_td(d):
            cnt += 1
    return d

def _load() -> dict:
    return json.loads(MANUAL_FILE.read_text(encoding="utf-8")) if MANUAL_FILE.exists() else {}

def _save(m: dict) -> None:
    MANUAL_FILE.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

def _info(pk: dict, code: str):
    """從 compute_picks 結果取某股:(分數, 是否被選, 排名)。用全排名查找表,15名外也查得到。"""
    selected = code in [c for _, c in pk["sel"]]
    ro = pk.get("rank_of", {}).get(code)
    if ro:
        rank, sc = ro
        return sc, selected, rank
    return None, selected, None


def _run_preview(code: str, bonus: float, days: int):
    """印過去5交易日:加分後 vs 原本(無加分)。回傳 (加分後選到天數, 原本選到天數, 到期日)。"""
    oh = get_ohlcv("0050")
    last5 = sorted(oh)[-5:]
    today = date.fromisoformat(max(oh))           # 用最新資料日當「今天」算到期
    until = _forward_td(today, days)
    _log = v5live.log; v5live.log = lambda *a, **k: None   # 靜音 compute_picks
    try:
        try:
            from tw_stock_agent.tools.finmind_client import get_stock_names
            nm = get_stock_names().get(code, code)      # 池外股也查得到中文名
        except Exception:
            nm = code
        hit = 0; base_hit = 0
        print(f"\n📊 試算:{code} {'+' if bonus>=0 else ''}{bonus:.0f}分,維持 {days} 個交易日(到 {until})")
        print(f"{'日期':<12}{'加分後分數':>11}{'排名':>6}{'選上?':>6}   {'原本分數':>9}{'原本選上?':>9}   當日選到名單")
        for d in last5:
            pk = compute_picks(date=d, held=set(), manual_override={code: bonus})
            bs = compute_picks(date=d, held=set(), manual_override={})
            nm = nm or pk["names"].get(code, code)
            sc, sel, rk = _info(pk, code)
            bsc, bsel, _ = _info(bs, code)
            hit += sel; base_hit += bsel
            sellist = ",".join(pk["names"].get(c, c)[:4] for _, c in pk["sel"])
            sc_s = f"{sc:.0f}" if sc is not None else "—"
            bsc_s = f"{bsc:.0f}" if bsc is not None else "0"
            rk_s = f"{rk}" if rk else ">15"
            print(f"{d:<12}{sc_s:>11}{rk_s:>6}{'✅' if sel else '✗':>6}   {bsc_s:>9}{'✅' if bsel else '✗':>9}   {sellist}")
        print(f"\n結論:加 {'+' if bonus>=0 else ''}{bonus:.0f} 後,{nm}({code}) 過去5交易日中 "
              f"**{hit}/5 天**會被選到(原本 {base_hit}/5 天)。")
        return hit, base_hit, until
    finally:
        v5live.log = _log


def main() -> int:
    a = sys.argv[1:]
    if not a:
        print(__doc__); return 2
    cmd = a[0]

    if cmd == "show":
        m = _load()
        if not m:
            print("(目前沒有任何手動加減分)"); return 0
        today = date.today().isoformat()
        print(f"{'代號':<8}{'加減分':>7}{'到期(交易日)':>16}{'狀態':>8}  備註")
        for c, e in m.items():
            alive = str(e.get("until", "")) >= today
            print(f"{c:<8}{e.get('bonus',0):>+7.0f}{str(e.get('until','')):>16}{'生效' if alive else '已過期':>8}  {e.get('note','')}")
        return 0

    if cmd == "rm":
        if len(a) < 2:
            print("用法: rm <代號>"); return 2
        m = _load(); code = a[1]
        if code in m:
            del m[code]; _save(m); print(f"已移除 {code}")
        else:
            print(f"{code} 不在清單")
        return 0

    if cmd in ("preview", "add"):
        if len(a) < 3:
            print(f"用法: {cmd} <代號> <加減分> [維持交易日數(預設5)] [--yes]"); return 2
        code = a[1]; bonus = float(a[2])
        days = 5
        for x in a[3:]:
            if x.lstrip("-").isdigit() and not x.startswith("--"):
                days = int(x)
        hit, base_hit, until = _run_preview(code, bonus, days)
        if cmd == "preview":
            print("\n(這是試算,未寫入。要套用請用 add)")
            return 0
        # add:確認後寫入
        if "--yes" not in a:
            ans = input(f"\n確定要對 {code} 設定 {'+' if bonus>=0 else ''}{bonus:.0f} 分、維持到 {until} 嗎?(y/N) ").strip().lower()
            if ans not in ("y", "yes"):
                print("已取消,未寫入。"); return 0
        m = _load()
        m[code] = {"bonus": bonus, "until": until.isoformat(),
                   "set": date.today().isoformat(), "days": days, "note": "manual watch"}
        _save(m)
        print(f"✅ 已寫入:{code} {'+' if bonus>=0 else ''}{bonus:.0f} 分,生效到 {until}({days}個交易日)。")
        return 0

    print(f"未知指令: {cmd}\n{__doc__}"); return 2


if __name__ == "__main__":
    sys.exit(main())
