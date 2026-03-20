#!/bin/bash
# run_swarm_daily.sh
#
# This script wraps the swarm_optimize.py execution.
# It is intended to be called by cron once per day after market close.
#
# Installation (Mac/Linux):
# 1. Make this script executable: chmod +x run_swarm_daily.sh
# 2. Open crontab: crontab -e
# 3. Add this line to run once daily after market close in Istanbul time.
#    For gold/forex-style markets, a practical "market closed" window is shortly after midnight.
#    Example: run at 01:15 Istanbul time, Tuesday-Saturday, so it evaluates the prior trading day after the session rollover:
# 15 1 * * 2-6 /Users/chrixchange/.gemini/antigravity/skills/xauusd-prediction-agent/tools/run_swarm_daily.sh >> /tmp/gold_swarm.log 2>&1

set -euo pipefail

echo "================================================="
echo "Gold (XAUUSD) Swarm Optimization Run: $(date)"
echo "================================================="

# IMPORTANT: Ensure your python path here is correct.
PYTHON_EXEC="python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/swarm_optimize.py"

$PYTHON_EXEC $SCRIPT_PATH

echo ""
