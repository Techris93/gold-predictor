#!/bin/bash
# run_gold_predict_hourly.sh
#
# This script wraps the predict_gold.py execution.
# It is now intended to be called by cron every 15 minutes.
#
# Installation (Mac/Linux):
# 1. Make this script executable: chmod +x run_gold_predict_hourly.sh
# 2. Open crontab: crontab -e
# 3. Add this line to run every 15 minutes:
# */15 * * * * /Users/chrixchange/.gemini/antigravity/skills/xauusd-prediction-agent/tools/run_gold_predict_hourly.sh >> /tmp/gold_predictions.log 2>&1

echo "================================================="
echo "Gold (XAUUSD) Prediction Run: $(date)"
echo "================================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" && -x "$PROJECT_ROOT/.venv313/bin/python" ]]; then
	DEFAULT_PYTHON="$PROJECT_ROOT/.venv313/bin/python"
fi
PYTHON_EXEC="${PYTHON_EXEC:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_EXEC" ]]; then
	echo "Python executable not found: $PYTHON_EXEC"
	exit 1
fi

cd "$PROJECT_ROOT" || exit 1
"$PYTHON_EXEC" -m tools.predict_gold

echo ""
