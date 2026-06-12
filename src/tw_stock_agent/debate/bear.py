"""Bear/Bull 辯論模組：把所有最近新聞丟給 AI，讓 AI 自己判斷相關性。"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from tw_stock_agent.config import cfg, get_settings
from tw_stock_agent.news.scanner import search_company_news, search_company_news_enhanced


def _get_gemini_client():
    from google import genai
    s = get_settings()
    return genai.Client(api_key=s.google_api_key)


# ── System prompts ─────────────────────────────────────────────────────────

BEAR_SYSTEM = """你是證據導向的風險審查員，不是反方辯手。
你的任務不是硬找空方理由，而是判斷這筆交易是否存在「足以否決進場」的具體風險。

【語言規定】所有輸出必須使用繁體中文。嚴禁使用英文說明，股票代號除外。

【你會收到】
1. 這支股票的技術面數據
2. 今日所有財經/科技新聞（未經任何篩選）
3. 個股直接相關新聞

你必須自己找出哪些新聞跟這支股票相關——包括：
- 直接提到這家公司的消息
- 影響客戶端的需求（客戶訂單砍單、客戶業績下滑）
- 競爭對手搶市場
- 產業上游漲價、材料短缺
- 宏觀事件（出口管制、關稅、地緣政治）對這公司的間接影響
- 替代技術威脅

【評分鐵則（重要，違反視為分析失敗）】
1. 沒有具體證據時，誠實給低分（bear_score ≤ 4）即可，**不需硬湊風險**。找不到實質利空本身就是合格答案。
2. 「估值偏高」「漲多可能獲利了結」「產業競爭激烈」「總體不確定性」等**推測性理由**，只能列為次要風險，bear_score **不得超過 5**。
3. 要給 bear_score ≥ 6，必須有**具體可驗證**的利空，例如：
   - 月營收 / 獲利明顯衰退（附數字）
   - 法人連續大賣且股價同步轉弱（搭配「嚴重出貨」標籤）
   - 公司展望下修、或明確利空新聞
   - 技術面跌破關鍵支撐、放量收黑、假突破、利多出盡
4. **資料缺漏（unknown）不等於風險**，不得因為「沒有資料」就提高 bear_score。
5. evidence_level 必須誠實：證據不足填 "weak"，且 bear_score ≤ 5。
6. 引用新聞時必須說明為何跟這支股票相關（不能引用無關新聞）。
7. 切勿因推測性風險就否決掉技術面/基本面明確偏多的設定。

【三大法人權重規則（重要）】
三大法人籌碼面僅供參考，動態權重依「外資流量嚴重程度」標籤而定，不可無腦把賣超當利空：
- 標記【嚴重出貨】：屬恐慌性出貨，才可作為提高 bear_score 的有力依據。
- 標記【中度調節】：僅可作為次要佐證，最多讓 bear_score +1，且需搭配其他利空。
- 標記【輕微賣超】或【中性】：視為正常週轉，**不得**因此提高 bear_score（外資調節 ≠ 個股轉弱）。
- 切勿因外資賣超就否決掉技術面/基本面明確偏多的設定，請以技術面、基本面、新聞催化劑為主要評分依據。

【輸出格式】嚴格 JSON，不附加任何說明文字：
{
  "ticker": "<股票代號>",
  "bear_score": <1-10，10=極度悲觀>,
  "evidence_level": "strong|medium|weak",
  "top_risks": ["具體風險1（說明與本股關聯）", "具體風險2", "具體風險3"],
  "relevant_news_titles": ["你認為跟這支股票相關的新聞標題1", "標題2"],
  "verdict": "PASS|WARN|REJECT",
  "reason": "<一句話繁中說明，必須含具體數字或事實>"
}

verdict 規則：bear_score>=7→REJECT, >=5→WARN, <5→PASS"""

BULL_SYSTEM = """你是多方分析師，專門找出股票上漲的具體催化劑與即將發生的事件排程。

【語言規定】所有輸出必須使用繁體中文。嚴禁使用英文說明，股票代號除外。

【你的任務】
你會收到：
1. 今日系統日期（精確，請以此為基準判斷「近期」）
2. 今日所有財經/科技新聞（未篩選）
3. 個股直接相關新聞（含法說/電話會議）

你必須自己判斷哪些新聞對這支股票有利好影響——包括：
- 客戶新訂單、出貨量增加
- 上游材料降價、毛利改善機會
- 新應用場景帶動需求（SpaceX 衛星、AI Server、電動車等）
- 法說會利多、買超、創高
- 競爭對手退出市場

【催化劑規則】
1. 催化劑必須具體可驗證，拒絕「AI 趨勢看好」這類空話。
2. 說明催化劑跟這支股票的具體關聯（例如：SpaceX 上市→衛星需求→射頻元件→誰受惠）。
3. 評估時效性：near（1個月內）vs medium（1-3個月）vs far（已 price in）。

【即將發生事件（Upcoming Events）偵測指令 — 重要！】
像雷達一樣掃描新聞中是否有「具體排程」的即將發生事件：
- 公司法說會、股東會、業績發表日期
- 重大新產品發表（Nvidia 新晶片、蘋果發表會、COMPUTEX 等）
- 處置股解除 / 管制解禁日期
- 新法規生效、政府標案開標日

規則：
1. 必須有明確的時間暗示（「下週三」「6/15」「月底」「本季末」）。
2. 僅說「未來將推出」的不算，歸類為普通催化劑即可。
3. 忠實記錄新聞中的原始時間字串（date_mention），**不需要你自己計算距今幾天**。
4. is_confirmed = true 表示官方確定日期，false 表示市場傳聞。

【輸出格式】嚴格 JSON，不附加任何說明文字：
{
  "ticker": "<股票代號>",
  "bull_score": <1-10，10=極度樂觀>,
  "catalysts": [
    {"event": "<繁中催化劑描述，說明與本股關聯>", "timeframe": "near|medium|far", "confidence": 0.0-1.0}
  ],
  "upcoming_events": [
    {
      "event_name": "<即將發生的重大事件名稱，例如：Q2法說會、蘋果WWDC、解除處置>",
      "date_mention": "<新聞原始時間字串，例如：下週二、6月15日、月底>",
      "is_confirmed": true
    }
  ],
  "relevant_news_titles": ["你認為跟這支股票相關的利多新聞標題1", "標題2"],
  "verdict": "BUY|WATCH|SKIP",
  "reason": "<一句話繁中說明>"
}"""

PREDICT_SYSTEM = """你是量化預測分析師。根據多空雙方的分析結果與技術面數據，分別預測這支股票的短期走勢：
- D+1：明日（下一個交易日）
- D+3：3 個交易日後
- D+5：5 個交易日後

【語言規定】所有輸出必須使用繁體中文。嚴禁使用英文說明，股票代號除外。

【規則】
1. 台股漲跌停限制 ±10%，單日預測範圍不超過此限；3/5日累積幅度可超過。
2. 預測需結合技術面（型態、量比、RS、近5日K線走勢）與基本面（多空評分）。
3. 若多空分歧大（bear_score 高但 bull_score 也高），confidence 應低於 0.4，direction 填 "neutral"。
4. 不確定時寧可預測 "neutral"，不要強行給方向。
5. 漲停隔日通常有三種情境：繼續鎖漲停、開高震盪、開高走低，需評估哪種最可能。
6. 3/5 日預測要考慮獲利了結賣壓、主力是否延續、基本面是否支撐。

【輸出格式】嚴格 JSON，不附加任何說明：
{
  "d1": {
    "direction": "up|down|neutral",
    "low_pct": <保守情境，例如 -2.5>,
    "high_pct": <樂觀情境，例如 +3.0>,
    "center_pct": <最可能，例如 +0.8>,
    "confidence": <0.0-1.0>
  },
  "d3": {
    "direction": "up|down|neutral",
    "low_pct": <3日累積保守>,
    "high_pct": <3日累積樂觀>,
    "center_pct": <3日累積最可能>,
    "confidence": <0.0-1.0>
  },
  "d5": {
    "direction": "up|down|neutral",
    "low_pct": <5日累積保守>,
    "high_pct": <5日累積樂觀>,
    "center_pct": <5日累積最可能>,
    "confidence": <0.0-1.0>
  },
  "key_factor": "<最關鍵驅動因素，30字內>",
  "scenario": "<簡短走勢情境描述，例如『漲停隔日震盪整理後再攻』，50字內>"
}"""


# ── 資料結構 ────────────────────────────────────────────────────────────────

@dataclass
class DebateResult:
    code: str
    name: str
    bear_score: int
    bull_score: int
    evidence_level: str
    top_risks: list[str]
    catalysts: list[dict]
    verdict: str
    bear_reason: str
    bull_reason: str
    negative_news: list[dict]   # [{title, link}]
    positive_news: list[dict]   # [{title, link}]
    # 隔日 (D+1) 預測
    predicted_direction: str = "neutral"
    predicted_low_pct: float = 0.0
    predicted_high_pct: float = 0.0
    predicted_center_pct: float = 0.0
    prediction_confidence: float = 0.0
    # D+3 預測
    d3_direction: str = "neutral"
    d3_low_pct: float = 0.0
    d3_high_pct: float = 0.0
    d3_center_pct: float = 0.0
    d3_confidence: float = 0.0
    # D+5 預測
    d5_direction: str = "neutral"
    d5_low_pct: float = 0.0
    d5_high_pct: float = 0.0
    d5_center_pct: float = 0.0
    d5_confidence: float = 0.0
    # 共用
    prediction_key_factor: str = ""
    prediction_scenario: str = ""
    # 即將發生事件（從 Bull Agent 新聞中萃取）
    upcoming_events: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.upcoming_events is None:
            self.upcoming_events = []


# ── LLM 呼叫 ───────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def _call_gemini(system: str, user: str, model: str | None = None) -> dict:
    import logging, time
    from google.genai import types
    client = _get_gemini_client()
    primary = model or cfg("llm.debate", "gemini-3.5-flash")
    fallback = cfg("llm.fallback", "gemini-2.5-flash-lite")

    def _try_model(m: str, max_retries: int = 4) -> dict:
        delays = [1, 2, 4, 8]
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                raw = resp.text or ""
                result = _parse_json(raw)
                if result:
                    return result
            except Exception as e:
                is_retriable = any(s in str(e) for s in ("503", "429", "499", "UNAVAILABLE", "CANCELLED", "RESOURCE_EXHAUSTED"))
                if is_retriable and attempt < max_retries - 1:
                    wait = delays[attempt]
                    logging.warning(f"Model {m} failed ({e}), retry {attempt+1}/{max_retries-1} in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        return {}

    try:
        return _try_model(primary)
    except Exception as e:
        logging.warning(f"Model {primary} failed ({e}), trying fallback {fallback}")
    try:
        return _try_model(fallback)
    except Exception as e:
        logging.error(f"Fallback {fallback} also failed: {e}")
        raise


def _call_claude(system: str, user: str, model: str | None = None) -> dict:
    import logging, time
    import anthropic
    m = model or cfg("llm.debate", "claude-haiku-4-5-20251001")
    api_key = get_settings().anthropic_api_key
    client = anthropic.Anthropic(api_key=api_key)
    delays = [1, 2, 4, 8]
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=m,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0,
            )
            raw = resp.content[0].text if resp.content else ""
            result = _parse_json(raw)
            if result:
                return result
        except Exception as e:
            is_retriable = any(s in str(e) for s in ("529", "503", "429", "overloaded"))
            if is_retriable and attempt < 3:
                wait = delays[attempt]
                logging.warning(f"Claude {m} failed ({e}), retry {attempt+1}/3 in {wait}s")
                time.sleep(wait)
            else:
                raise
    return {}


def _call_openai(system: str, user: str, model: str | None = None) -> dict:
    import logging, time
    from openai import OpenAI
    m = model or cfg("llm.debate", "gpt-4o-mini")
    api_key = get_settings().openai_api_key
    client = OpenAI(api_key=api_key)
    delays = [1, 2, 4, 8]
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            result = _parse_json(raw)
            if result:
                return result
        except Exception as e:
            is_retriable = any(s in str(e) for s in ("429", "503", "overloaded", "rate_limit"))
            if is_retriable and attempt < 3:
                wait = delays[attempt]
                logging.warning(f"OpenAI {m} failed ({e}), retry {attempt+1}/3 in {wait}s")
                time.sleep(wait)
            else:
                raise
    return {}


def _call_llm(system: str, user: str, model: str | None = None) -> dict:
    """統一 LLM 呼叫入口，依 config 的 llm.provider 決定 openai / claude / gemini。"""
    provider = cfg("llm.provider", "gemini")
    if provider == "openai":
        return _call_openai(system, user, model)
    if provider == "claude":
        return _call_claude(system, user, model)
    return _call_gemini(system, user, model)


# ── 主函式 ─────────────────────────────────────────────────────────────────

def run_debate(
    stock: dict,
    all_recent_news: list[dict],
    historical_date=None,   # datetime.date | None：回測模式下指定日期
) -> DebateResult:
    """Bear/Bull 辯論。

    Args:
        stock: screened stock dict
        all_recent_news: Node 1 抓下來的所有新聞（live 模式）；historical_date 有值時忽略
        historical_date: datetime.date，設定後改用 Google News 歷史搜尋取代 live 新聞
    """
    from tw_stock_agent.news.scanner import search_news_around_date

    code = stock.get("code", "")
    name = stock.get("name", code)

    if historical_date is not None:
        # ── 回測/歷史重生模式：用 ≤historical_date 的歷史新聞（防未來洩漏）──────
        src = cfg("news.historical_source", "finmind")
        if src == "finmind":
            # FinMind TaiwanStockNews：per-stock、含時戳、涵蓋2022→今、免費零洩漏。
            # 使用者定：只用個股新聞，不抓大盤/總體（總體靠價格 regime 特徵）。
            from tw_stock_agent.tools.finmind_client import get_historical_news
            company_articles = get_historical_news(code, historical_date,
                                                   window_days=5, max_articles=15)
            broad_articles = []
        else:
            company_articles = search_news_around_date(name, historical_date,
                                                       window_days=1, max_articles=12)
            broad_articles = search_news_around_date(
                "台灣科技股 OR 半導體 OR AI伺服器 OR 供應鏈",
                historical_date, window_days=1, max_articles=25,
            )
        if cfg("debate.ablate_news", False):   # ablation:關掉新聞輸入
            company_articles = []
            broad_articles = []
        all_recent_news = broad_articles
        _mode = f"historical({historical_date},{src})"
    else:
        # ── Live 模式：用多策略搜尋（含法說/電話會議/英文名） ──────────
        name_en = stock.get("name_en", "")
        company_articles = search_company_news_enhanced(
            name_zh=name,
            code=code,
            name_en=name_en,
            max_articles=20,
            max_age_hours=72,
        )
        # 廣域新聞（若呼叫方未傳入，自行抓 RSS）
        if not all_recent_news:
            from tw_stock_agent.news.scanner import scan_news
            all_recent_news = scan_news(max_age_hours=48)
        _mode = "live"

    def _fmt_news(articles: list[dict], limit: int = 30) -> str:
        if not articles:
            return "（無）"
        lines = []
        for a in articles[:limit]:
            title = a.get("title", "")
            if not title or "[搜尋失敗]" in title or "[ERROR]" in title:
                continue
            link = a.get("link", "")
            pub = (a.get("published", "") or a.get("date", ""))[:16]
            snippet = (a.get("content", "") or a.get("snippet", ""))[:120].strip()
            link_md = f" [連結]({link})" if link else ""
            line = f"- {title}{link_md}（{pub}）"
            if snippet:
                line += f"\n  {snippet}"
            lines.append(line)
        return "\n".join(lines) if lines else "（無）"

    tech_summary = (
        f"股票：{name}（{code}）\n"
        f"收盤價：{stock.get('close_price', 0):.1f} TWD\n"
        f"量比（今日量/20日均量）：{stock.get('volume_ratio', 0):.2f}x\n"
        f"MA5 > MA20：{stock.get('ma5_gt_ma20', False)}\n"
        f"相對強度（20日）：{stock.get('rs_20d', 1.0):.2f}\n"
        f"52週高點警示：{stock.get('weeks_52_warn', False)}\n"
        f"K線型態：{stock.get('pattern_type', 'none')}\n"
    )

    # 近 5 日 K 線（讓 LLM 看連續走勢，不只今日快照）
    recent_bars = stock.get("recent_bars", [])
    if recent_bars:
        tech_summary += "\n近5日K線：\n"
        tech_summary += f"  {'日期':<12} {'開':>7} {'高':>7} {'低':>7} {'收':>7} {'量比':>6} {'漲跌':>8}\n"
        for i, bar in enumerate(recent_bars):
            chg_str = f"{bar['chg_pct']:+.2f}%" if i > 0 else " (基準)"
            tech_summary += (
                f"  {bar['date']:<12} {bar['open']:>7.1f} {bar['high']:>7.1f}"
                f" {bar['low']:>7.1f} {bar['close']:>7.1f}"
                f" {bar['vol_ratio']:>5.2f}x {chg_str:>8}\n"
            )

    # 特殊市場狀態（漲跌停 / 處置 / 注意）
    ms = stock.get("market_status", {})
    if ms:
        if ms.get("limit") == "limit_up":
            tech_summary += "【特殊狀態】當日漲停板（+10%）：大量買盤追捧，隔日可能鎖死或大幅震盪\n"
        elif ms.get("limit") == "limit_down":
            tech_summary += "【特殊狀態】當日跌停板（-10%）：嚴重賣壓，隔日風險極高\n"
        if ms.get("is_disposal"):
            until = ms.get("disposal_until", "未知")
            days = ms.get("disposal_days_left")
            days_str = f"，{days} 天後出關" if days is not None and days >= 0 else ""
            tech_summary += f"【特殊狀態】處置股：至 {until} 止{days_str}\n"
            tech_summary += "  → 交易限制：約每 20 分鐘撮合一次、需全額預繳、無法當沖\n"
            tech_summary += "  → 出關前夕散戶常追漲，但主力可能趁流動性恢復時倒貨，風險極高\n"

    # 注入個股歷史型態表現（有資料才加；回測模式傳入截止日避免 look-ahead）
    profile_text = ""
    try:
        from tw_stock_agent.tools.stock_profile import get_pattern_profile
        profile_text = get_pattern_profile(
            code, stock.get("pattern_type", "none"),
            as_of_date=historical_date,
        )
        if profile_text:
            tech_summary += f"\n{profile_text}\n"
    except Exception:
        pass

    # 注入三大法人籌碼 + 月營收（FinMind；有資料才加；回測模式截止日隔離）
    try:
        from tw_stock_agent.tools.finmind_client import get_institutional_flow, get_revenue_yoy
        inst = get_institutional_flow(code, as_of_date=historical_date)
        if inst.get("summary_text") and not cfg("debate.ablate_inst", False):
            tech_summary += f"\n{inst['summary_text']}\n"
        rev = get_revenue_yoy(code, as_of_date=historical_date)
        if rev.get("summary_text") and not cfg("debate.ablate_rev", False):
            tech_summary += f"\n{rev['summary_text']}\n"
    except Exception:
        pass

    # 注入跌深反彈訊號（confirmed edge；回測模式截止日隔離，防洩漏）
    try:
        from tw_stock_agent.tools.rebound_signal import rebound_signal_for_code, avg_turnover_of
        _aod = historical_date.isoformat() if historical_date is not None else None
        rb = rebound_signal_for_code(code, avg_turnover_of(code), as_of=_aod)
        if rb.get("summary"):
            tech_summary += f"\n{rb['summary']}\n"
    except Exception:
        pass

    # 所有廣域新聞（未篩選，Node 1 直接拿到的）
    all_news_text = _fmt_news(all_recent_news, limit=40)
    # 個股新聞（只搜公司名，無關鍵字）
    company_news_text = _fmt_news(company_articles, limit=12)

    from datetime import date as _date
    # 回測模式用歷史日期，live 模式用今天
    today_str = historical_date.isoformat() if historical_date is not None else _date.today().isoformat()

    bear_prompt = (
        f"今日系統日期：{today_str}\n\n"
        f"{tech_summary}\n\n"
        f"---\n"
        f"【今日所有財經/科技新聞（未篩選，請你自行判斷哪些跟 {name} 相關）】\n"
        f"{all_news_text}\n\n"
        f"---\n"
        f"【個股直接相關新聞（搜尋關鍵字：{name}）】\n"
        f"{company_news_text}\n\n"
        f"---\n"
        f"請分析：上述所有新聞中，哪些對 {name}（{code}）構成風險或利空影響？"
        f"（包括直接和間接影響，例如客戶端、競品、宏觀事件）"
    )

    bull_prompt = (
        f"今日系統日期：{today_str}\n\n"
        f"{tech_summary}\n\n"
        f"---\n"
        f"【今日所有財經/科技新聞（未篩選，請你自行判斷哪些跟 {name} 相關）】\n"
        f"{all_news_text}\n\n"
        f"---\n"
        f"【個股直接相關新聞（含法說/電話會議搜尋，搜尋關鍵字：{name}）】\n"
        f"{company_news_text}\n\n"
        f"---\n"
        f"請分析：上述所有新聞中，哪些對 {name}（{code}）構成催化劑或利多影響？"
        f"並偵測新聞中是否有具體排程的即將發生事件（法說會日期、發表會、解禁日等）。"
    )

    bear_data = _call_llm(BEAR_SYSTEM, bear_prompt)
    bull_data = _call_llm(BULL_SYSTEM, bull_prompt)

    bear_score = int(bear_data.get("bear_score", 5))
    bull_score = int(bull_data.get("bull_score", 5))
    evidence_level = bear_data.get("evidence_level", "weak")
    top_risks = bear_data.get("top_risks", [])
    catalysts = bull_data.get("catalysts", [])

    # Step1 安全閘：弱證據時 bear_score 封頂 5，防止「估值偏高/競爭壓力」等
    # 推測性理由把空方分數灌高（即使 LLM 沒遵守 prompt 也由程式強制）
    if evidence_level == "weak" and bear_score > 5:
        logging.info(f"  bear_score {bear_score}→5（證據 weak，封頂）")
        bear_score = 5

    reject_thresh = cfg("debate.bear_reject_threshold", 7)
    warn_thresh = cfg("debate.bear_warn_threshold", 5)
    if bear_score >= reject_thresh:
        verdict = "REJECT"
    elif bear_score >= warn_thresh:
        verdict = "WARN"
    else:
        verdict = "PASS"

    # 從 LLM 回報的 relevant_news_titles 匹配回帶連結的文章
    bear_relevant = set(bear_data.get("relevant_news_titles", []))
    bull_relevant = set(bull_data.get("relevant_news_titles", []))
    all_articles = list(company_articles) + list(all_recent_news)

    def _match_articles(titles: set[str]) -> list[dict]:
        matched = []
        for a in all_articles:
            t = a.get("title", "")
            if any(t and (title in t or t in title) for title in titles):
                matched.append({"title": t, "link": a.get("link", "")})
        # 若 LLM 沒回報，fallback 用 company_articles
        return matched if matched else [
            {"title": a.get("title", ""), "link": a.get("link", "")}
            for a in company_articles[:5]
            if a.get("title") and "[搜尋失敗]" not in a.get("title", "")
        ]

    # ── 隔日漲跌預測（第三個 LLM call，整合多空結果）──────────────────────────
    predict_prompt = (
        f"股票：{name}（{code}）\n\n"
        f"【空方分析結果】\n"
        f"bear_score: {bear_score}/10，evidence: {evidence_level}\n"
        f"主要風險：{'; '.join((top_risks or [])[:3])}\n"
        f"空方判斷：{bear_data.get('reason', '')}\n\n"
        f"【多方分析結果】\n"
        f"bull_score: {bull_score}/10\n"
        f"主要催化劑：{'; '.join(c.get('event','') for c in (catalysts or [])[:3])}\n"
        f"多方判斷：{bull_data.get('reason', '')}\n\n"
        f"【技術面】\n"
        f"K線型態：{stock.get('pattern_type', 'none')} — {stock.get('pattern_detail', '')}\n"
        f"量比：{stock.get('volume_ratio', 0):.2f}x，RS20d：{stock.get('rs_20d', 1.0):.2f}\n"
        f"MA5>MA20：{stock.get('ma5_gt_ma20', False)}，RSI14：{stock.get('rsi_14', 0):.1f}\n\n"
        + (f"{profile_text}\n\n" if profile_text else "")
        + f"請預測 {name}（{code}）**明日（下一個交易日）**的漲跌幅。"
    )
    pred_data = _call_llm(PREDICT_SYSTEM, predict_prompt)

    # 從 bull_data 解析即將發生事件
    raw_events = bull_data.get("upcoming_events") or []
    upcoming_events: list[dict] = []
    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        event_name = ev.get("event_name", "").strip()
        date_mention = ev.get("date_mention", "").strip()
        is_confirmed = bool(ev.get("is_confirmed", False))
        if not event_name or not date_mention:
            continue
        # Python 計算倒數天數（LLM 只負責萃取原始字串，不算數學）
        days_until: int | None = None
        try:
            import dateparser
            parsed = dateparser.parse(
                date_mention,
                languages=["zh-Hant", "zh-Hans", "en"],
                settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
            )
            if parsed:
                from datetime import datetime as _dt
                delta = (parsed.date() - _dt.now().date()).days
                if -3 <= delta <= 180:   # 合理範圍：3天前 ~ 半年後
                    days_until = delta
        except Exception:
            pass
        upcoming_events.append({
            "event_name": event_name,
            "date_mention": date_mention,
            "is_confirmed": is_confirmed,
            "days_until": days_until,
        })

    d1 = pred_data.get("d1") or {}
    d3 = pred_data.get("d3") or {}
    d5 = pred_data.get("d5") or {}

    # 相容舊格式（單層 predicted_direction）
    if not d1 and "predicted_direction" in pred_data:
        d1 = {
            "direction":  pred_data.get("predicted_direction", "neutral"),
            "low_pct":    pred_data.get("predicted_low_pct", 0.0),
            "high_pct":   pred_data.get("predicted_high_pct", 0.0),
            "center_pct": pred_data.get("predicted_center_pct", 0.0),
            "confidence": pred_data.get("confidence", 0.0),
        }

    return DebateResult(
        code=code, name=name,
        bear_score=bear_score, bull_score=bull_score,
        evidence_level=evidence_level,
        top_risks=top_risks if isinstance(top_risks, list) else [str(top_risks)],
        catalysts=catalysts if isinstance(catalysts, list) else [],
        verdict=verdict,
        bear_reason=bear_data.get("reason", ""),
        bull_reason=bull_data.get("reason", ""),
        negative_news=_match_articles(bear_relevant),
        positive_news=_match_articles(bull_relevant),
        # D+1
        predicted_direction=d1.get("direction", "neutral"),
        predicted_low_pct=float(d1.get("low_pct", 0.0)),
        predicted_high_pct=float(d1.get("high_pct", 0.0)),
        predicted_center_pct=float(d1.get("center_pct", 0.0)),
        prediction_confidence=float(d1.get("confidence", 0.0)),
        # D+3
        d3_direction=d3.get("direction", "neutral"),
        d3_low_pct=float(d3.get("low_pct", 0.0)),
        d3_high_pct=float(d3.get("high_pct", 0.0)),
        d3_center_pct=float(d3.get("center_pct", 0.0)),
        d3_confidence=float(d3.get("confidence", 0.0)),
        # D+5
        d5_direction=d5.get("direction", "neutral"),
        d5_low_pct=float(d5.get("low_pct", 0.0)),
        d5_high_pct=float(d5.get("high_pct", 0.0)),
        d5_center_pct=float(d5.get("center_pct", 0.0)),
        d5_confidence=float(d5.get("confidence", 0.0)),
        # 共用
        prediction_key_factor=pred_data.get("key_factor", ""),
        prediction_scenario=pred_data.get("scenario", ""),
        upcoming_events=upcoming_events,
    )
