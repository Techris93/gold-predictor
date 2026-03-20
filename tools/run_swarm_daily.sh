#!/bin/bash
# run_swarm_daily.sh
#
# This script wraps the swarm_optimize.py execution.
# It is intended to be called by cron once per day after market close.
#
# Installation (Mac/Linux):
# 1. Make this script executable: chmod +x run_swarm_daily.sh
# 2. Open crontab: crontab -e
# 3. Add this line to run once daily after market close (example: 6:15 PM server time):
# 15 18 * * 1-5 /Users/chrixchange/.gemini/antigravity/skills/xauusd-prediction-agent/tools/run_swarm_daily.sh >> /tmp/gold_swarm.log 2>&1

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
