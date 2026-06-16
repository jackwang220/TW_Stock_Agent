#!/bin/bash
# v5 實盤排程包:交易日防呆 → 跑指定腿 → (主程式自己推播)
# 用法: run_leg.sh buy|sell [傳給 live_dual_v5_trade.py 的額外參數...]
#   排程(launchd)每天觸發,本檔負責「今天不是交易日就跳過」。
#   預設 dry-run(不送單)。要真的送單,額外參數加 --sim 或 --live(且 .env 設 SHIOAJI_LIVE_CONFIRM=YES)。
set -euo pipefail

# launchd 的 PATH 很乾淨,補上 uv / brew 路徑
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEG="${1:-}"
if [[ "$LEG" != "buy" && "$LEG" != "sell" ]]; then
  echo "用法: run_leg.sh buy|sell [extra args]"; exit 2
fi
shift || true

TODAY="$(date +%F)"
DOW="$(date +%u)"   # 1=一 ... 7=日
LOGDIR="$ROOT/reports/live"; mkdir -p "$LOGDIR"
RUNLOG="$LOGDIR/${TODAY}_${LEG}.log"

stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNLOG"; }

# ── 交易日防呆 ──
if [[ "$DOW" -ge 6 ]]; then
  stamp "週末($TODAY)休市 → 跳過 $LEG"; exit 0
fi
HOL="$ROOT/data/tw_holidays.txt"
if [[ -f "$HOL" ]] && grep -qE "^${TODAY}([[:space:]]|#|\$)" "$HOL"; then
  stamp "休市日($TODAY,見 tw_holidays.txt)→ 跳過 $LEG"; exit 0
fi
if [[ -f "$ROOT/KILL_SWITCH" ]]; then
  stamp "🛑 KILL_SWITCH 存在 → 跳過 $LEG(刪掉它才會跑)"; exit 0
fi

stamp "▶ 開跑 $LEG  extra=[$*]"
# 主程式會自己做 Discord 推播;這裡把 stdout 也存進當日 log
if uv run python "$ROOT/scripts/live_dual_v5_trade.py" --leg "$LEG" "$@" 2>&1 | tee -a "$RUNLOG"; then
  stamp "✔ $LEG 完成"
else
  stamp "✗ $LEG 失敗(exit=$?)"
fi
