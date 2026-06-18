"""Discord bot(Phase 4)——把 GPT-4o-mini agent 接到 Discord,你私訊就能操作。

安全:
  - **只回應 DISCORD_OWNER_ID 本人**(別人傳一律忽略)——真錢設定不能讓別人碰。
  - 改設定一樣要確認:agent 會先試算 + 問「確定嗎?」,你**回覆「確定」**才執行(Discord 版的確認閘門)。
  - 沿用 agent_tools 白名單,永遠不能下真實單。

需 .env:
  OPENAI_API_KEY=sk-...
  DISCORD_BOT_TOKEN=...        (Discord 開發者後台建 bot 拿)
  DISCORD_OWNER_ID=123456789   (你的 Discord 使用者 ID,只聽你)
Discord 開發者後台 → Bot → 開啟「MESSAGE CONTENT INTENT」。

跑(機器要一直開著):  uv run python scripts/discord_bot.py
指令:傳「reset」清除對話記憶。
"""
from __future__ import annotations
import sys, os, asyncio, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(str(ROOT / ".env"))
import discord

_s = importlib.util.spec_from_file_location("agent_chat", ROOT / "scripts/agent_chat.py")
ac = importlib.util.module_from_spec(_s); _s.loader.exec_module(ac)

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OWNER = os.getenv("DISCORD_OWNER_ID")
AFFIRM = {"確定", "好", "yes", "y", "ok", "對", "可以", "沒問題", "go", "確認"}
SESSIONS: dict = {}     # uid -> messages(對話記憶,跨訊息)

_intents = discord.Intents.default(); _intents.message_content = True
bot = discord.Client(intents=_intents)
_oai = None


@bot.event
async def on_ready():
    print(f"✅ 上線: {bot.user}（只聽 OWNER={OWNER or '未設(任何人都可,危險!)'}）")


@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return
    if OWNER and str(msg.author.id) != str(OWNER):     # 只聽本人
        return
    text = (msg.content or "").strip()
    if not text:
        return
    if text.lower() in ("reset", "清除", "重置"):
        SESSIONS.pop(msg.author.id, None)
        await msg.channel.send("🧹 已清除對話記憶。")
        return
    messages = SESSIONS.setdefault(msg.author.id, [{"role": "system", "content": ac.SYSTEM}])
    # Discord 版確認閘門:這則訊息是不是「確定」=同意執行 mutate(配合 agent 自然的「先試算問→你回確定」流程)
    confirm_fn = lambda name, args: text in AFFIRM
    async with msg.channel.typing():
        try:
            reply = await asyncio.to_thread(ac.chat_turn, _oai, messages, text, confirm_fn)
        except Exception as e:
            reply = f"❌ 出錯: {e}"
    reply = reply or "(無回覆)"
    for i in range(0, len(reply), 1900):               # Discord 單則 2000 字上限
        await msg.channel.send(reply[i:i + 1900])


def main() -> int:
    global _oai
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 沒 OPENAI_API_KEY(.env)"); return 2
    if not TOKEN:
        print("❌ 沒 DISCORD_BOT_TOKEN(.env)。去 https://discord.com/developers 建 bot 拿 token。"); return 2
    if not OWNER:
        print("⚠️ 沒設 DISCORD_OWNER_ID → 任何人都能操作你的系統,強烈建議設成你的 Discord 使用者 ID。")
    from openai import OpenAI
    _oai = OpenAI()
    print("連線中 ...")
    bot.run(TOKEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
