#!/bin/bash
# run_swarm_weekly.sh
#
# This script wraps the broader weekly swarm search.
# It is intended to be called by cron once per week outside peak market hours.
#
# Installation (Mac/Linux):
# 1. Make this script executable: chmod +x run_swarm_weekly.sh
# 2. Open crontab: crontab -e
# 3. Add a line like this to run early Sunday morning in Istanbul time:
#    30 2 * * 0 /Users/chrixchange/.openclaw/workspace/gold-predictor/tools/run_swarm_weekly.sh >> /tmp/gold_swarm_weekly.log 2>&1

set -euo pipefail

echo "================================================="
echo "Gold (XAUUSD) Weekly Swarm Optimization Run: $(date)"
echo "================================================="

PYTHON_EXEC="${PYTHON_EXEC:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/swarm_optimize.py"

cd "$SCRIPT_DIR"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git pull --ff-only origin main >/tmp/gold_swarm_weekly_git_pull.log 2>&1 || true
fi

"$PYTHON_EXEC" -u "$SCRIPT_PATH" --reduced --serial --period 365d --interval 1h --ticker GC=F

echo ""
