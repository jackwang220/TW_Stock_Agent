"""Markdown 報告產生器。"""
from __future__ import annotations

from datetime import datetime


_VERDICT_EMOJI = {"PASS": "🟢", "WARN": "🟡", "REJECT": "🔴", "ERROR": "⚫"}
_LEVEL_EMOJI   = {"PASS": "🟢", "WARN": "🟡", "REJECT": "🔴"}


def _stock_section(s: dict) -> str:
    verdict = s.get("verdict", "")
    emoji = _VERDICT_EMOJI.get(verdict, "⚪")
    name = s.get("name", s.get("code", ""))
    code = s.get("code", "")
    close = s.get("close_price", 0)
    vr = s.get("volume_ratio", 0)
    rs = s.get("rs_20d", 1.0)
    bear = s.get("bear_score", 0)
    bull = s.get("bull_score", 0)
    pattern = s.get("pattern_type", "none")
    pattern_detail = s.get("pattern_detail", "")
    via = s.get("via_path", "")
    w52 = " ⚠️52W高" if s.get("weeks_52_warn") else ""
    evidence = s.get("evidence_level", "")

    risks = s.get("top_risks", [])
    risk_lines = "\n".join(f"  - {r}" for r in risks[:4]) if risks else "  - （無）"

    catalysts = s.get("catalysts", [])
    cat_lines = "\n".join(
        f"  - {c.get('event','')} ｜{c.get('timeframe','')} ｜信心 {c.get('confidence',0):.0%}"
        for c in catalysts[:3]
    ) if catalysts else "  - （無）"

    # 新聞來源連結
    neg_news = s.get("negative_news", [])
    pos_news = s.get("positive_news", [])
    def _news_links(articles: list[dict], limit: int = 4) -> str:
        lines = []
        for a in articles[:limit]:
            t = a.get("title", "")
            lk = a.get("link", "")
            lines.append(f"  - [{t}]({lk})" if lk else f"  - {t}")
        return "\n".join(lines) if lines else "  - （無）"

    # 預測區塊
    pred_dir = s.get("predicted_direction", "neutral")
    pred_low = s.get("predicted_low_pct", 0.0)
    pred_high = s.get("predicted_high_pct", 0.0)
    pred_center = s.get("predicted_center_pct", 0.0)
    pred_conf = s.get("prediction_confidence", 0.0)
    pred_factor = s.get("prediction_key_factor", "")
    dir_emoji = {"up": "📈", "down": "📉", "neutral": "➡️"}.get(pred_dir, "➡️")
    dir_label = {"up": "看漲", "down": "看跌", "neutral": "中性"}.get(pred_dir, "中性")
    conf_bar = "█" * round(pred_conf * 5) + "░" * (5 - round(pred_conf * 5))
    predict_block = (
        f"**【隔日預測】** {dir_emoji} {dir_label}　"
        f"`{pred_low:+.1f}% ～ {pred_center:+.1f}% ～ {pred_high:+.1f}%`　"
        f"把握度 {conf_bar} {pred_conf:.0%}"
        + (f"\n> {pred_factor}" if pred_factor else "")
    )

    return f"""### {emoji} {verdict} — {name}（{code}）{w52}

| 指標 | 數值 |
|---|---|
| 收盤價 | {close:.1f} TWD |
| 量比（今/20日均） | {vr:.2f}x |
| 相對強度（20日） | {rs:.2f} |
| K 線型態 | {pattern} |
| Bear Score | {bear}/10（{evidence}）|
| Bull Score | {bull}/10 |
| 供應鏈路徑 | {via or '直接命中'} |

**K 線細節**：{pattern_detail}

{predict_block}

**多方催化劑**：
{cat_lines}

**空方風險**：
{risk_lines}

**Bear 判斷**：{s.get('bear_reason','')}
**Bull 判斷**：{s.get('bull_reason','')}

<details><summary>📰 新聞來源</summary>

**負面/風險新聞**：
{_news_links(neg_news)}

**正面/成長新聞**：
{_news_links(pos_news)}
</details>
"""


def _screened_row(s: dict) -> str:
    """量化篩選結果的單行摘要（給未進入辯論的股票用）。"""
    lvl = s.get("pass_level", "REJECT")
    emoji = _LEVEL_EMOJI.get(lvl, "⚪")
    code = s.get("code", "")
    name = s.get("name", code)
    close = s.get("close_price", 0.0)
    vr = s.get("volume_ratio", 0.0)
    rs = s.get("rs_20d", 1.0)
    ma = "Y" if s.get("ma5_gt_ma20") else "N"
    reason = s.get("screen_reason", "")
    via = s.get("via_path", "")
    depth = s.get("bfs_depth", 0)
    path_label = f" via {via}" if via else ""
    return (
        f"| {emoji} {lvl} | {code} | {name} | "
        f"{close:.1f} | {vr:.2f}x | {rs:.2f} | {ma} | "
        f"depth={depth}{path_label} | {reason} |"
    )


def build_report(debated: list[dict], today: str,
                 screened: list[dict] | None = None) -> str:
    """把 debated 清單組成完整 Markdown 報告。

    Args:
        debated: 完整 Bear/Bull 辯論結果（PASS/WARN/REJECT verdict）
        today: ISO 日期字串
        screened: 量化篩選全結果（含未進入辯論的 REJECT 股票）
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    is_error = lambda s: str(s.get("bear_reason", "")).startswith("[debate_error]")
    pass_stocks   = [s for s in debated if s.get("verdict") == "PASS"  and not is_error(s)]
    warn_stocks   = [s for s in debated if s.get("verdict") == "WARN"  and not is_error(s)]
    reject_stocks = [s for s in debated if s.get("verdict") == "REJECT" and not is_error(s)]
    error_stocks  = [s for s in debated if is_error(s)]

    # 量化篩選 REJECT（未進入辯論）
    debated_codes = {s.get("code") for s in debated}
    screen_only = [s for s in (screened or [])
                   if s.get("code") not in debated_codes]
    screen_reject = [s for s in screen_only if not s.get("_screen_passed")]

    total = len(debated) + len(screen_reject)

    header = f"""# 台股科技供應鏈每日掃描報告

**日期**：{today}　**產出時間**：{now}

> ⚠️ 本報告為量化訊號彙整，非投資建議。所有判斷僅供參考，請自行評估風險。

---

## 摘要

| 類別 | 數量 |
|---|---|
| 🟢 PASS（建議關注） | {len(pass_stocks)} |
| 🟡 WARN（需觀察） | {len(warn_stocks)} |
| 🔴 REJECT（辯論後） | {len(reject_stocks)} |
| 📊 量化淘汰 | {len(screen_reject)} |
| ⚠️ Debate 失敗 | {len(error_stocks)} |
| 總掃描 | {total} |

"""
    if total == 0:
        return header + "\n今日無候選股（新聞無法命中供應鏈標的或資料不足）。\n"

    sections: list[str] = []

    # ── 詳細辯論分析（PASS + WARN）──────────────────────────
    if pass_stocks or warn_stocks:
        sections.append("---\n\n## 詳細分析（通過篩選）\n")
        for s in pass_stocks + warn_stocks:
            sections.append(_stock_section(s))

    # ── Debate API 失敗的股票（保留量化評級）────────────────
    if error_stocks:
        sections.append("---\n\n## ⚠️ Debate API 失敗（量化評級仍有效）\n")
        sections.append(
            "| 量化評級 | 代號 | 名稱 | 收盤 | 量比 | RS20 | 原因 |\n"
            "|---|---|---|---|---|---|---|"
        )
        for s in error_stocks:
            lvl = s.get("pass_level", s.get("verdict", "?"))
            emoji = _LEVEL_EMOJI.get(lvl, "⚪")
            err_msg = s.get("bear_reason", "").replace("[debate_error] ", "")[:80]
            sections.append(
                f"| {emoji} {lvl} | {s.get('code','')} | {s.get('name','')} | "
                f"{s.get('close_price',0):.1f} | {s.get('volume_ratio',0):.2f}x | "
                f"{s.get('rs_20d',1):.2f} | {err_msg} |"
            )
        sections.append("")

    # ── 辯論後 REJECT ────────────────────────────────────────
    if reject_stocks:
        sections.append("---\n\n## 🔴 辯論後 REJECT（空方論據強）\n")
        for s in reject_stocks:
            sections.append(
                f"- **{s.get('name','')}（{s.get('code','')}）** "
                f"bear={s.get('bear_score',0)} bull={s.get('bull_score',0)} "
                f"— {s.get('bear_reason','')}\n"
            )

    # ── 量化篩選概覽（所有掃描到的股票）────────────────────
    if screened:
        sections.append("---\n\n## 📊 量化篩選概覽（所有候選）\n")
        sections.append(
            "| 評級 | 代號 | 名稱 | 收盤 | 量比 | RS20 | MA5>MA20 | 路徑 | 原因 |\n"
            "|---|---|---|---|---|---|---|---|---|"
        )
        # 先顯示進入辯論的
        for s in debated:
            lvl = s.get("pass_level", s.get("verdict", "REJECT"))
            row_s = {**s, "pass_level": lvl}
            sections.append(_screened_row(row_s))
        # 再顯示量化淘汰的
        for s in screen_reject:
            sections.append(_screened_row(s))
        sections.append("")

    footer = f"""
---

*由 TW-Stock-Agent 自動產出 ／ {now}*
"""
    return header + "\n".join(sections) + footer
