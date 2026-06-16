# v5 實盤運維手冊(H雙引擎 B純切 / ⑤ 買收盤吃隔夜溢價)

策略已定案:**H雙引擎 B純切**(多頭→純H動能、空頭→純反彈,開關=0050 收盤>MA20)。
執行=⑤(收盤前買、隔日開盤出清掉出名單)。本檔講「怎麼讓它每天自動跑」與「出錯怎麼辦」。

---

## 0. 目前階段:股票在元大、手動交易(`--ledger` 訊號模式)

Shioaji 只連永豐、**看不到也碰不到元大帳戶**。所以現階段機器人 = **訊號產生器**:
- 用永豐 keys **模擬登入(只抓即時價、不下任何單、免 CA)** → 算 B純切名單 → 推 Discord。
- 持有/損益讀**手動帳本** `data/positions.json`(你在元大買賣後自己記),用永豐即時價估市值。
- 排程(launchd)已預設帶 `--ledger`,**保證不會送任何委託**。

**每天流程:**
1. 09:02 收到 DC 賣出訊號 → 你在元大手動賣 → `uv run python scripts/ledger.py sell <代號> <股數> <成交價>`
2. 13:05 收到 DC 買進訊號 → 你在元大手動買 → `uv run python scripts/ledger.py buy <代號> <股數> <成交價>`
3. 隔天訊號就會用更新後的帳本算「該買/該賣/持有損益」。

**帳本指令**(`scripts/ledger.py`):
```bash
uv run python scripts/ledger.py buy  2330 18 1010   # 買18股@1010(自動算加權均價)
uv run python scripts/ledger.py sell 2330 5  1050   # 賣5股@1050(印已實現損益)
uv run python scripts/ledger.py show                # 看目前帳本
uv run python scripts/ledger.py rm   2330           # 整檔移除
```

手動跑一次看今天訊號(不必等排程):
```bash
uv run python scripts/live_dual_v5_trade.py --leg buy --live-signal --ledger
```
> 之後開永豐戶、要讓機器人自動下單時,才拿掉 `--ledger`、走下面 §3 的 sim→live 流程。

---

## 1. 每天實際發生什麼

| 時間 | 腿 | 動作 |
|---|---|---|
| 09:02 開盤後 | `sell` | 登入永豐 → 讀真實持倉 → 把「今天掉出名單」的部位開盤出清 |
| 13:05 收盤前 | `buy`  | 登入 → 抓盤中即時價 → 算今日 B純切 名單(前~3檔) → 比對持倉 → 把沒買夠的買到目標(只加碼) |

排程器(launchd)每天這兩個時間各觸發一次 `scripts/run_leg.sh`,它先做**交易日防呆**(週末 / `data/tw_holidays.txt` 假日 / `KILL_SWITCH` → 跳過),再跑主程式 `scripts/live_dual_v5_trade.py`。主程式跑完會把下單摘要推播到 Discord。

> 注意:這不是常駐程式。它一天被叫醒兩次、跑完就結束。**Mac 在 09:02 / 13:05 必須開機且未睡眠**(見 §6)。

---

## 2. 檔案地圖

| 檔案 | 角色 |
|---|---|
| `scripts/live_dual_v5_trade.py` | 交易主程式(算單/比對持倉/下單/推播);三段安全閘 |
| `scripts/run_leg.sh` | 排程包:交易日防呆 → 呼叫主程式 → 存當日 log |
| `scripts/launchd/com.twstock.v5.buy.plist` | 買腿排程(13:05) |
| `scripts/launchd/com.twstock.v5.sell.plist` | 賣腿排程(09:02) |
| `data/tw_holidays.txt` | 休市日清單(**需自行依 TWSE 校正**) |
| `.env` | 永豐金鑰、`SHIOAJI_LIVE_CONFIRM`、`DISCORD_WEBHOOK_URL` |
| `KILL_SWITCH`(建立才生效) | 緊急中止:此檔存在 → 主程式與排程包都會拒跑 |
| `reports/live/<日期>_<腿>.log` | 每次執行的完整輸出 |
| `reports/v5_orders_<腿>_<時戳>.log` | 實際送出的委託 JSON 紀錄 |

---

## 3. 上線前三段驗證(務必依序)

主程式預設 **dry-run**(登入+算單+推播,但**不送任何委託**)。

```bash
# ① dry-run:看今天會買/賣什麼(不送單)
uv run python scripts/live_dual_v5_trade.py --leg buy --live-signal

# ② sim:Shioaji 模擬下單(不花真錢),連續跑幾天確認流程
uv run python scripts/live_dual_v5_trade.py --leg buy --live-signal --sim

# ③ live:真實下單(需 .env 設 SHIOAJI_LIVE_CONFIRM=YES)
SHIOAJI_LIVE_CONFIRM=YES uv run python scripts/live_dual_v5_trade.py --leg buy --live-signal --live
# 賣腿:
uv run python scripts/live_dual_v5_trade.py --leg sell            # dry-run
SHIOAJI_LIVE_CONFIRM=YES uv run python scripts/live_dual_v5_trade.py --leg sell --live
```

安全上限寫在主程式頂部(依資金調整):`MAX_POSITION_TWD`(單股持有上限)、`MAX_TOTAL_BUY_TWD`(單次總買入)、`MAX_ORDERS`、`LIMIT_UP_GUARD`(近漲停不買)。

---

## 4. 安裝排程(確認 sim 沒問題後再做)

```bash
# 先把 plist 內 dry-run 跑順;要真的送單,編輯 plist 在 ProgramArguments 末尾加 <string>--live</string>
cp scripts/launchd/com.twstock.v5.buy.plist  ~/Library/LaunchAgents/
cp scripts/launchd/com.twstock.v5.sell.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.twstock.v5.buy.plist
launchctl load -w ~/Library/LaunchAgents/com.twstock.v5.sell.plist

launchctl list | grep twstock          # 確認已載入
# 手動立刻測一次(不必等到時間):
launchctl start com.twstock.v5.buy
```

停用 / 移除:
```bash
launchctl unload -w ~/Library/LaunchAgents/com.twstock.v5.buy.plist
launchctl unload -w ~/Library/LaunchAgents/com.twstock.v5.sell.plist
```

升級下單模式:預設 plist 是 dry-run。要模擬→在 buy/sell plist 的 `<array>` 末尾加 `<string>--sim</string>`;要真實→改成 `<string>--live</string>` 並確保 `.env` 有 `SHIOAJI_LIVE_CONFIRM=YES`。改完 `unload` 再 `load`。

---

## 5. 緊急中止

```bash
touch KILL_SWITCH      # 之後任何腿都會拒跑
rm KILL_SWITCH         # 恢復
```
盤中要砍已掛未成交的單,用永豐 App / 官網手動刪單最快。

---

## 6. 「電腦要開著」的坑(務必處理)

- **睡眠**:Mac 闔蓋/休眠時 launchd 不一定醒。两個做法:
  - 交易時段用 `caffeinate`:`caffeinate -dimsu &`(或設定排程在 08:50 caffeinate、13:30 放掉)。
  - 或 `sudo pmset repeat wakeorpoweron MTWRF 08:55:00`(每個交易日早上自動喚醒)。
- **更穩**:把整包丟到一台**不關機的機器**(家裡常開的 Mac mini / 雲端 VM)。筆電當主力交易機不可靠。
- **時區**:系統需為台北時間(plist 用本機時間)。
- **網路**:09:02 / 13:05 要有網路。

---

## 7. 例行檢查

- [ ] 每天看 Discord 推播:buy/sell 各一則,內容對不對。
- [ ] 每月初用 TWSE 官方休市表更新 `data/tw_holidays.txt`(尤其農曆春節連假)。
- [ ] **永豐 CA 憑證約一年到期** → 到期前重新申請、更新 `SINOPAC_CA_PATH`。
- [ ] 異常(推播說 ❌ 失敗、或沒收到推播)→ 看 `reports/live/<日期>_<腿>.log`。

---

## 8. 已知限制(別忘了)

- ⑤ 收盤滿倉 = **裸隔夜曝險**;回測樣本(2021~2026)多頭偏多,長期深空頭未充分驗證。
- 最差季 alpha 約 **−18%**(B純切);這是策略本質,不是 bug。
- 訊號/universe 用 `base_universe.json` 的 112 檔;成交值/反彈分每天重算。
