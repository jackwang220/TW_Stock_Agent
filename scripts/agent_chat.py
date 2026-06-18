"""股票助理 agent(Phase 3)——GPT-4o-mini + agent_tools 工具庫 + 雙層安全閘門。

安全模型(三道關):
  1) LLM 不給 shell,只能呼叫 agent_tools 白名單裡的工具。
  2) 意圖閘門:只有當你的訊息含「設定意圖關鍵字」(設定/加分/改權重/改資金…),
     才把 mutate(會改設定)工具放進這一輪的工具集;否則純聊天只給唯讀工具,LLM 根本叫不到改設定的工具。
  3) 確認閘門:LLM 想執行 mutate 工具時,先試算給你看 → 問你「確定?」→ 你明確同意才以 allow_mutate 執行。
  下單(trade)永遠不在工具庫 → agent 不可能下真單。

用法(終端機 REPL,先 `OPENAI_API_KEY=...` 寫進 .env):
  uv run python scripts/agent_chat.py
  你> 2330 現在分數多少          (純聊天→唯讀工具)
  你> 幫台積電加20分維持5天       (設定意圖→試算→問確定→寫入)
"""
from __future__ import annotations
import sys, json, os, re, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv(str(ROOT / ".env"))

_s = importlib.util.spec_from_file_location("agent_tools", ROOT / "scripts/agent_tools.py")
AT = importlib.util.module_from_spec(_s); _s.loader.exec_module(AT)

MODEL = "gpt-4o-mini"
# 設定意圖關鍵字 + 正則(false-positive 無害,後面還有確認閘門;false-negative 只是叫不動設定、不會誤改)
MUTATE_INTENT = ["設定", "設成", "加分", "減分", "加幾分", "扣分", "改", "調整", "權重",
                 "weights", "資金", "現金", "capital", "移除", "拿掉", "刪", "set", "bonus", "觀察加"]
# 抓「加/減/扣/設 N 分」這種帶數字的講法(例:加20分、減 15 分、設+30分)
MUTATE_RE = re.compile(r"[加減扣設調][\s+\-]*\d+\s*分")

SYSTEM = (
    "你是使用者的台股策略助理。你只能透過提供的工具操作他的系統,不要編造數字。\n"
    "唯讀工具:查分數 query_score、查universe、查持倉、查設定、查加減分、試算 preview_bonus、"
    "歷史損益 pnl_history(mode=holdings/list/stocks,可帶 weight、days、capital、bonus 試算)。可自由用來回答問題。\n"
    "★ 認股票:使用者用『名字』提到股票(尤其可能打錯字或簡稱)時,先用 resolve_stock 解析成代號;"
    "若結果不只一個或你不確定,問他『你是說 X(代號) 嗎?』確認後再動作,絕不自己亂猜代號。給純數字就當代號。\n"
    "★ 講不清楚就先問,別亂猜:當使用者的要求『缺少必要資訊或有多種解讀』時,先用一句話問清楚再動作。例如:\n"
    "  - 『算損益』沒說算哪些股(你的持股?某幾檔?今日名單?)、幾天、怎麼加權 → 先問。\n"
    "  - 『加分/減分』沒說哪一檔、加幾分、維持幾天 → 先問。\n"
    "  - 只有小細節有合理預設(如天數預設5天、等權)時,可以用預設但要明講你用了什麼,讓他能更正。\n"
    "若使用者要『改設定』(手動加減分、雙引擎權重、可投資金):\n"
    "  1. 先把參數問清楚 → 用 preview_bonus / pnl_history 試算跑給他看。\n"
    "  2. 說明你打算怎麼改,再呼叫對應 set_ 工具(系統會再要求他本人確認)。\n"
    "你絕對不能、也沒有工具可以下任何真實買賣單。回答用繁體中文,簡潔。"
)


def detect_mutate_intent(text: str) -> bool:
    return bool(MUTATE_RE.search(text)) or any(k in text for k in MUTATE_INTENT)


def _confirm_terminal(name: str, args: dict) -> bool:
    ans = input(f"\n⚠️ 即將執行【{name}】參數={args}\n   確定?(y/N) ").strip().lower()
    return ans in ("y", "yes", "確定", "好")


def handle_tool_call(name: str, args: dict, confirm_fn=_confirm_terminal) -> str:
    tool = AT._BY_NAME.get(name)
    if not tool:
        return f"❌ 未知工具 {name}"
    if tool["tier"] != "mutate":
        return AT.run_tool(name, args)                 # 唯讀:直接跑
    # mutate:先試算(加分類)→ 確認 → 執行
    if name == "set_manual_score":
        print("\n— 先試算給你看 —")
        print(AT.run_tool("preview_bonus",
                          {"code": args.get("code"), "bonus": args.get("bonus"), "days": args.get("days", 5)}))
    if confirm_fn(name, args):
        return AT.run_tool(name, args, allow_mutate=True)
    return "使用者未確認,已取消,未改任何設定。"


def chat_turn(client, messages: list, user_msg: str, confirm_fn=_confirm_terminal) -> str:
    messages.append({"role": "user", "content": user_msg})
    mutate_ok = detect_mutate_intent(user_msg)
    tools = AT.openai_schema("read") + (AT.openai_schema("mutate") if mutate_ok else [])
    for _ in range(8):                                  # 最多 8 輪工具呼叫
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content or ""
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = handle_tool_call(tc.function.name, args, confirm_fn)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "(工具呼叫過多,已中止)"


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 沒有 OPENAI_API_KEY。請在 .env 加一行 OPENAI_API_KEY=sk-... 再跑。"); return 2
    from openai import OpenAI
    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM}]
    print("股票助理(GPT-4o-mini)。Ctrl-C 離開。純聊天=唯讀;含『設定/加分/改權重』才會動設定(且要你確認)。\n")
    try:
        while True:
            u = input("你> ").strip()
            if not u:
                continue
            print("🤖", chat_turn(client, messages, u), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\n掰。"); return 0


if __name__ == "__main__":
    sys.exit(main())
