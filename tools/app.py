from flask import Flask, jsonify, render_template
import sys
import os

# Add the tools directory to the path so we can import predict_gold
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import predict_gold

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict')
def get_prediction():
    try:
        # Fetch data using the existing script logic
        ta_data = predict_gold.get_technical_analysis()
        sa_data = predict_gold.get_sentiment_analysis()
        
        # Calculate a basic final verdict based on TA and Volume
        verdict = "Neutral"
        confidence = 50
        
        trend = ta_data.get("ema_trend", "Neutral")
        volume = ta_data.get("volume_analysis", {}).get("overall_volume_signal", "Neutral")
        pa_struct = ta_data.get("price_action", {}).get("structure", "Consolidating")
        pa_pattern = ta_data.get("price_action", {}).get("latest_candle_pattern", "None")
        
        bull_score = 0
        bear_score = 0
        
        # 1. Moving Averages Trend (Max 3.0)
        if trend == "Bullish": bull_score += 3.0
        elif trend == "Bearish": bear_score += 3.0
            
        # 2. Volume & Orderflow (Max 2.0)
        if "Accumulation" in volume: bull_score += 2.0
        elif "Buying Bias" in volume: bull_score += 1.0
        if "Distribution" in volume: bear_score += 2.0
        elif "Selling Bias" in volume: bear_score += 1.0
            
        # 3. Price Action (Max 2.0)
        if "Bullish Structure" in pa_struct: bull_score += 1.0
        elif "Bearish Structure" in pa_struct: bear_score += 1.0
        
        if "Bullish Engulfing" in pa_pattern: bull_score += 1.0
        elif "Bearish Engulfing" in pa_pattern: bear_score += 1.0

        # 4. RSI Momentum (Max 2.0)
        rsi = ta_data.get("rsi_14", 50)
        if rsi < 30: bull_score += 2.0
        elif rsi < 40: bull_score += 1.0
        if rsi > 70: bear_score += 2.0
        elif rsi > 60: bear_score += 1.0
            
        # Calculate dynamic confidence
        # Total possible score for one side is 9.0 points
        max_possible = 9.0
        
        if bull_score > bear_score + 1:
            verdict = "Bullish"
            confidence = 60 + ((bull_score / max_possible) * 35) # Scales from 60% to 95%
        elif bear_score > bull_score + 1:
            verdict = "Bearish"
            confidence = 60 + ((bear_score / max_possible) * 35)
        else:
            verdict = "Neutral"
            confidence = 50 + (abs(bull_score - bear_score) * 5)
            
        # Ensure it stays within reasonable bounds
        confidence = round(min(max(confidence, 50), 95))

        return jsonify({
            "status": "success",
            "verdict": verdict,
            "confidence": confidence,
            "TechnicalAnalysis": ta_data,
            "SentimentalAnalysis": sa_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to allow access from other devices on the local network (like your phone)
    app.run(host='0.0.0.0', port=5001, debug=True)
