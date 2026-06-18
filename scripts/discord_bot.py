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

# 操作面板按鈕:(標籤, 工具, 參數, 樣式)。都是唯讀工具,點了直接跑。
PANEL_BTNS = [
    ("🟢 今日該買", "today_buy", {}, discord.ButtonStyle.success),
    ("🔴 今日該賣", "today_sell", {}, discord.ButtonStyle.danger),
    ("💰 我的持倉損益", "pnl_history", {"mode": "holdings"}, discord.ButtonStyle.secondary),
    ("📊 我的持倉", "query_positions", {}, discord.ButtonStyle.secondary),
    ("📈 5天每日換股(回顧)", "pnl_strategy_daily", {"days": 5}, discord.ButtonStyle.primary),
]


def make_panel() -> discord.ui.View:
    v = discord.ui.View(timeout=None)
    for label, tool, targs, style in PANEL_BTNS:
        b = discord.ui.Button(label=label, style=style)
        async def cb(interaction: discord.Interaction, _tool=tool, _args=targs):
            if OWNER and str(interaction.user.id) != str(OWNER):
                await interaction.response.send_message("⛔ 只有擁有者能用", ephemeral=True); return
            await interaction.response.defer(thinking=True)
            print(f"[按鈕] {interaction.user} → {_tool}{_args}", flush=True)
            try:
                res = await asyncio.to_thread(ac.AT.run_tool, _tool, _args)
            except Exception as e:
                res = f"❌ 出錯: {e}"
            res = res or "(無回覆)"
            for i in range(0, len(res), 1900):
                await interaction.followup.send(res[i:i + 1900])
        b.callback = cb
        v.add_item(b)
    return v


@bot.event
async def on_ready():
    print(f"✅ 上線: {bot.user}（只聽 OWNER={OWNER or '未設'}）", flush=True)
    for g in bot.guilds:
        chans = [ch for ch in g.text_channels if ch.permissions_for(g.me).send_messages]
        print(f"  伺服器「{g.name}」可發言頻道: {[ch.name for ch in chans] or '無(權限不足)'}", flush=True)
        if chans:
            try:
                await chans[0].send("🤖 TW Stock bot 上線!**請在這個頻道**跟我說話(例:`2330 分數多少`)")
                print(f"  → 已發測試訊息到 #{chans[0].name}", flush=True)
            except Exception as e:
                print(f"  → 發訊息失敗: {e}", flush=True)


@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return
    text = (msg.content or "").strip()
    print(f"[收到] from={msg.author}({msg.author.id}) owner={OWNER} match={str(msg.author.id)==str(OWNER)} "
          f"content={text[:50]!r}", flush=True)
    if OWNER and str(msg.author.id) != str(OWNER):     # 只聽本人
        print("  → 非owner,忽略", flush=True); return
    if not text:
        print("  → 空內容(可能 Message Content Intent 沒生效)", flush=True); return
    if text.lower() in ("reset", "清除", "重置"):
        SESSIONS.pop(msg.author.id, None)
        await msg.channel.send("🧹 已清除對話記憶。")
        return
    if text.lower() in ("選單", "面板", "按鈕", "menu", "panel", "/menu"):
        await msg.channel.send("🎛️ **操作面板**(點按鈕直接查;也可以直接打字問):", view=make_panel())
        return
    messages = SESSIONS.setdefault(msg.author.id, [{"role": "system", "content": ac.SYSTEM}])
    # Discord 版確認閘門:這則訊息是不是「確定」=同意執行 mutate(配合 agent 自然的「先試算問→你回確定」流程)
    confirm_fn = lambda name, args: text in AFFIRM
    print("  → 處理中(呼叫 agent)...", flush=True)
    async with msg.channel.typing():
        try:
            reply = await asyncio.to_thread(ac.chat_turn, _oai, messages, text, confirm_fn)
        except Exception as e:
            import traceback; traceback.print_exc()
            reply = f"❌ 出錯: {e}"
    reply = reply or "(無回覆)"
    print(f"  → 回覆 {len(reply)} 字,送出中 ...", flush=True)
    try:
        for i in range(0, len(reply), 1900):           # Discord 單則 2000 字上限
            await msg.channel.send(reply[i:i + 1900])
        print("  → 已送出 ✅", flush=True)
    except Exception as e:
        print(f"  → ❌ 送出失敗(可能該頻道無發言權限): {e}", flush=True)


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
