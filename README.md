# TW Stock Agent — 台股科技供應鏈 LLM 分析 Agent

用 **LLM(Bear/Bull 辯論)+ 量化篩選 + 供應鏈知識圖 + 經實證的價格 edge(跌深反彈)**,每天從台股科技供應鏈裡挑出值得關注的標的,並用嚴格的防洩漏框架做回測與每日紙上交易帳本。

> ⚠️ 本專案為研究/教育用途,所有產出皆為量化訊號彙整,**非投資建議**。實單請自負風險。

---

## 目錄
- [系統在做什麼](#系統在做什麼)
- [核心設計理念](#核心設計理念)
- [架構:統一 Pipeline](#架構統一-pipeline)
- [已驗證的交易 Edge](#已驗證的交易-edge)
- [安裝](#安裝)
- [設定](#設定)
- [每日使用流程](#每日使用流程)
- [回測與研究](#回測與研究)
- [腳本索引](#腳本索引)
- [資料檔說明](#資料檔說明)
- [已知限制](#已知限制)

---

## 系統在做什麼

每天(盤後)跑一次:

1. **掃描候選股** — 固定掃描池 `base_universe`(~112 檔流動性大型股,0050/0051/00947 等大型 ETF 成分的代理)。
2. **量化初篩** — 量比、相對強度、均線、K 線型態 → PASS / WARN / REJECT。
3. **供應鏈擴張**(live 模式)— 從新聞命中的標的沿供應鏈知識圖 BFS 找受益股。
4. **Bear/Bull LLM 辯論** — 對 Top-N 候選股,LLM 讀「個股新聞 + 三大法人籌碼 + 月營收 + 技術面 + 跌深反彈訊號」做多空辯論,輸出**隔日漲跌方向 + 區間 + 信心**。
5. **產出報告** — `reports/<決策日>.md`(分級 + 每檔催化劑/風險/預測/新聞來源)。
6. **更新帳本** — `reports/live_portfolio.md`(期望值加權再平衡的紙上交易帳本)。

深度 LLM 分析固定鎖在 Top-N(預設 25 檔),所以**成本與掃描池大小脫鉤**——universe 放大也不會讓 LLM 成本爆掉。

---

## 核心設計理念

### 1. 防未來數據洩漏(Look-ahead / Time-travel Leakage)
回測/歷史重生時,**只能用截止「決策日」當天(含盤後)的資料**:
- 新聞:`get_historical_news`(FinMind,`date <= 決策日`,嚴禁看到隔日)。
- 技術面 / 三大法人 / 月營收 / 跌深反彈:全部傳 `as_of_date=決策日` 隔離。
- 價格:`get_ohlcv(as_of_date=…)` 只抓到決策日為止。

### 2. 盤後日期慣例(自動處理「幾點跑」)
`market_calendar.resolve_session()` 把 wall-clock 自動解析成 **(決策日, 交易日)**:

| 你幾點跑 | 決策日 | 交易日 |
|---|---|---|
| 週二 23:00(盤後) | 週二 | 週三 |
| 週三 01:00(半夜) | 週二 | 週三 |
| 週三 08:00(開盤前) | 週二 | 週三 |
| 週三 15:00(盤後) | 週三 | 週四 |
| 週五盤後 / 週末 | 週五 | 下週一 |

→ 解決「晚上 11 點跟半夜 1 點已經跨一個日曆日」的問題:兩者都對應**同一個交易日**。
**報告檔以「決策日」命名;`signal_log` / 帳本以「交易日」命名。**

### 3. Point-in-time 掃描池(零生存者偏誤)
`universe_as_of(as_of)`:用**截止決策日**的成交額排名取流動性大型股。跑 2022 就拿 2022「當時」的大型股,不會用到後來才變大的股票。

### 4. 一套系統(live = 歷史重生 = 回測)
同一個 `run_daily_scan(as_of)`:
- `as_of=None` → live(用現在時間自動解析決策日)。
- `as_of="2026-06-09"` → 重生那天的報告(只用 ≤6/9 資料)。
- 回測 = 迴圈呼叫不同 `as_of`。

---

## 架構:統一 Pipeline

LangGraph `StateGraph`(`src/tw_stock_agent/pipeline/`),7 個節點:

```
fetch_news → extract_entities → supply_chain_bfs → quantitative_screen
           → pattern_match → bear_debate → generate_report
```

| 節點 | live 模式 | as_of 模式(歷史/回測) |
|---|---|---|
| `fetch_news` | 即時 RSS 新聞 | 跳過(辯論層自抓 ≤決策日新聞) |
| `extract_entities` | 新聞命中個股 | 候選股 = `universe_as_of`(固定掃描池) |
| `supply_chain_bfs` | 供應鏈受益股 | 跳過(固定掃描池) |
| `quantitative_screen` | 量化初篩(today=今天) | 量化初篩(`as_of_date`) |
| `pattern_match` | K 線型態 | K 線型態(`as_of_date`) |
| `bear_debate` | Bear/Bull LLM(Top-N) | 同左,`historical_date=決策日` |
| `generate_report` | 寫報告 + signal_log | 報告=決策日、signal_log=交易日 |

LLM provider 可在 `configs/default.yaml` 切換(OpenAI / Claude / Gemini),預設 `gpt-4o-mini`。

---

## 已驗證的交易 Edge

經三方獨立確認(validation A + 防洩漏 kNN + edge 掃描器 OOS)、**樣本外(2022–2026)成立**:

**「大型股 + 跌深 → 短線反彈」家族**(前瞻 5 日,勝率=贏過大盤中位數比例):

| 設定 | 條件 | OOS 勝率 |
|---|---|---|
| 5 日跌 ≤ −20% | 大型股 | **78%** |
| 單日跌停 | 大型股 | **70%** |
| 5 日跌 ≤ −12% | 大型股 | **64%** |
| 乖離 ≤ −10%(遠低於 MA20) | 大型股 | **63%** |

**核心規律**:大型/高流動性股跌越兇 → 反彈越強(均值回歸,有人接刀)。劑量反應單調(−8%/−12%/−20% → 超額勝 +3/+8/+22pts)。

已做成訊號 `tools/rebound_signal.py`(權重=OOS 實證期望淨報酬)並注入 LLM prompt。

**回測(純反彈、2 年、真實成交含手續費)**:只做大型股 + d12 深度門檻 → **+75% / Sharpe 1.84 / 最大回撤 −26%**。

**測過但沒用的想法**(務必實測別信網路說法):新鮮瀑布加權、爆量恐慌、liquidity sweep、generic kNN 形狀比對 → 皆無 edge 或更差。

---

## 安裝

需求:**Python ≥ 3.12** + [uv](https://github.com/astral-sh/uv)(或 pip)。

```bash
# 用 uv(建議)
uv sync

# 或 pip
pip install -e .
```

主要依賴:`langgraph`、`openai` / `anthropic` / `google-genai`、`yfinance`、`pandas` / `numpy` / `scipy`、`networkx`、`feedparser`、`pydantic`、`loguru`。

---

## 設定

### `.env`(API 金鑰 / Token)
複製 `.env.example` → `.env` 填入:

```ini
OPENAI_API_KEY=sk-...           # 預設辯論模型 gpt-4o-mini
GOOGLE_API_KEY=AIza...          # 選用(Gemini)
ANTHROPIC_API_KEY=sk-ant-...    # 選用(Claude)

FINMIND_TOKEN=eyJ...            # FinMind 資料(價格/新聞/法人/營收)— 必填
FINMIND_TOKEN2=eyJ...           # 第二帳號 token,雙 token 分攤額度(選用,建議)

DISCORD_WEBHOOK_URL=            # 通知(選用)
```

> FinMind 免費版約 300 請求/小時;雙 token 輪流 + 撞 402 自動切換 → 約 600/小時。

### `configs/default.yaml`(交易規則,改這裡就改行為)
重點區塊 `trading:`:

```yaml
trading:
  capital:                 # 資金模型
    initial: 15000         # 期初一次投入
    daily_budget: 1000     # 每交易日加碼
    max_contribution: 50000# 總投入上限
  signals:
    combine: "sum"         # sum | max | llm_only | rebound_only
    rebound_large_only: true   # 反彈只在大型股觸發
    rebound_min_depth: "d12"   # 最淺觸發深度 d8/d12/d20
  entry:
    timing: "next_open"    # 隔日開盤進場(真實)
    max_chase_pct: 0.03    # 追高上限
    skip_limit_up: true    # 隔日鎖漲停買不到→跳過
    slippage_pct: 0.001
  exit:
    max_hold_days: 5
    take_profit_pct: 0.10
    stop_loss_pct: -0.06
  cost:
    round_trip_pct: 0.005  # 來回手續費+稅

news:
  historical_source: "finmind"  # 回測歷史新聞源:finmind(2022+,零洩漏) | google
universe:
  as_of_top_n: 0           # as_of 掃描池:0=全部 base_universe;>0=當時流動性前N
```

---

## 每日使用流程

```bash
# 1. 盤後(或半夜/開盤前)跑一次掃描 → 產生當日決策報告 + signal_log
python scripts/daily_stock_scan.py

# 2. 更新紙上交易帳本(算出今天該怎麼配置 + 結算昨日P&L)
python scripts/live_portfolio.py

# 3. 看今天的跌深反彈觸發(快速查,不跑 LLM)
python scripts/rebound_today.py
```

日期會自動解析(不用管你幾點跑)。報告在 `reports/<決策日>.md`,帳本在 `reports/live_portfolio.md`。

### 重生某天的報告(用現在的邏輯 + 只用該天的資料)
```bash
python scripts/regen_report.py 2026-06-09   # 舊報告自動備份成 .OLD.md
```

---

## 回測與研究

```bash
# 真實成交回測(隔日開盤進場、滑價、停利停損、手續費)
python scripts/backtest_realistic.py                      # LLM+反彈合併(讀 LLM CSV)
python scripts/backtest_realistic.py --rebound-only --days 500   # 純反彈 2 年

# 完整 LLM 60 天回測(新聞+辯論,慢)
python scripts/run_backtest_60d.py --base                 # 112 檔

# Edge 掃描器(setup × 條件 × OOS,扣基準算超額勝率)
python scripts/edge_scanner.py

# 跨期間 LLM 驗證(2022/2023/2024/2025 各取樣,測 LLM 是否跨 regime 都行)
python scripts/crossperiod_validate.py    # 跑批(數小時)
python scripts/crossperiod_analyze.py     # 分析各 regime 準確率 vs 純反彈
```

**方法論護欄**(找 edge 最大的敵人是假陽性):①防洩漏 ②樣本外確認(IS 發現/OOS 驗證)③多重檢定 ④扣同條件基準(避免右偏 skew 灌水)⑤扣成本 ⑥要有經濟故事。

---

## 腳本索引

| 腳本 | 用途 |
|---|---|
| `daily_stock_scan.py` | 每日掃描入口(跑統一 pipeline) |
| `live_portfolio.py` | 紙上交易帳本(期望值加權再平衡) |
| `regen_report.py <日期>` | 用統一系統重生某天報告(防洩漏) |
| `rebound_today.py` | 快速列出今日跌深反彈觸發 |
| `backtest_realistic.py` | 真實成交回測 |
| `run_backtest_60d.py` | 完整 LLM 60 天回測 |
| `edge_scanner.py` | Edge 掃描(OOS + 扣基準) |
| `crossperiod_validate.py` / `crossperiod_analyze.py` | 跨期間 LLM robust 驗證 |
| `build_base_universe.py` | 建固定掃描池(流動性排名) |
| `fetch_tw_stock_index.py` | 建股票字典(FinMind TaiwanStockInfo) |
| `build_stock_profile.py` | 建個股歷史型態表現檔 |

---

## 資料檔說明(`data/`)

| 檔案 | 內容 |
|---|---|
| `base_universe.json` | 固定掃描池 ~112 檔(每日強制分析) |
| `tw_stock_index.json` | 全台股字典(~2139 檔,代號↔名稱↔yf_ticker) |
| `companies.json` | 供應鏈知識圖 + watchlist |
| `signal_log.csv` | 每日預測記錄(交易日命名)+ 報酬回填 |
| `finmind_cache/` | FinMind 價格/新聞/法人/營收本地快取 |
| `ohlcv_cache/` | yfinance 價格 pickle 快取 |
| `backtest_llm_results.csv` | LLM 回測逐筆結果 |

---

## 已知限制

- **FinMind 歷史新聞只涵蓋 2022 → 今**;更早(2020 COVID / 2018)需 TEJ 等其他來源。
- **LLM 加值尚未跨 regime 證實**:在近期崩盤反彈窗 LLM-only ≈ +30%(vs 純反彈 +14%),但這是單一窗 + Google 新聞;跨期間驗證(FinMind 新聞)進行中,用來裁決「LLM 那層是真強還是窗運氣」。
- **三大法人 / 月營收的歷史快取**目前以近期為主,深歷史回測時這部分 context 較薄(新聞與價格不受影響)。
- `live_portfolio.py` 進場價仍用「前一交易日收盤」(尚未切到 `trading_rules` 的隔日開盤模型)。
- 小型股不在掃描池(流動性不足、易被操控、買不到)。

---

*由 TW-Stock-Agent 自動分析產出。研究用途,非投資建議。*
