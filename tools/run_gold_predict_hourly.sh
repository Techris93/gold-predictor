#!/bin/bash
# run_gold_predict_hourly.sh
# 
# This script wraps the predict_gold.py execution.
# It can be called by cron to run every 1 hour.
# 
# Installation (Mac/Linux):
# 1. Make this script executable: chmod +x run_gold_predict_hourly.sh
# 2. Open crontab: crontab -e
# 3. Add this line to run at the top of every hour (00 minutes):
# 0 * * * * /Users/chrixchange/.gemini/antigravity/skills/xauusd-prediction-agent/tools/run_gold_predict_hourly.sh >> /tmp/gold_predictions.log 2>&1

echo "================================================="
echo "Gold (XAUUSD) Prediction Run: $(date)"
echo "================================================="

# IMPORTANT: Ensure your python path here is correct.
# If you are using anaconda, you might need to use the explicit path:
# e.g., /Users/chrixchange/anaconda3/bin/python
PYTHON_EXEC="python"
SCRIPT_PATH="/Users/chrixchange/.gemini/antigravity/skills/xauusd-prediction-agent/tools/predict_gold.py"

$PYTHON_EXEC $SCRIPT_PATH

echo ""
