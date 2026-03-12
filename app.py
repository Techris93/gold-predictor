from flask import Flask, jsonify, render_template
from tools import predict_gold

app = Flask(__name__)

@app.after_request
def apply_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict')
def get_prediction():
    try:
        # Fetch data using the existing script logic
        ta_data = predict_gold.get_technical_analysis()
        sa_data = predict_gold.get_sentiment_analysis()

        if not isinstance(ta_data, dict):
            ta_data = {"error": "Technical analysis payload is invalid."}

        if not isinstance(sa_data, list):
            sa_data = []

        if ta_data.get("error"):
            return jsonify({
                "status": "error",
                "message": ta_data["error"],
                "verdict": "Neutral",
                "confidence": 50,
                "TechnicalAnalysis": ta_data,
                "SentimentalAnalysis": sa_data
            }), 502
        
        # Calculate verdict using trend + multi-timeframe + regime + price action
        verdict = "Neutral"
        confidence = 50
        
        trend = ta_data.get("ema_trend", "Neutral")
        volume = ta_data.get("volume_analysis", {}).get("overall_volume_signal", "Neutral")
        regime = ta_data.get("volatility_regime", {}).get("market_regime", "Range-Bound")
        adx_14 = ta_data.get("volatility_regime", {}).get("adx_14", 0)
        mtf = ta_data.get("multi_timeframe", {})
        alignment_score = mtf.get("alignment_score", 0)
        alignment_label = mtf.get("alignment_label", "Mixed / Low Alignment")
        pa_struct = ta_data.get("price_action", {}).get("structure", "Consolidating")
        pa_pattern = ta_data.get("price_action", {}).get("latest_candle_pattern", "None")
        
        bull_score = 0.0
        bear_score = 0.0
        
        # 1. 1H trend baseline
        if trend == "Bullish":
            bull_score += 2.5
        elif trend == "Bearish":
            bear_score += 2.5

        # 2. Multi-timeframe alignment
        if alignment_score > 0:
            bull_score += min(alignment_score, 3) * 1.2
        elif alignment_score < 0:
            bear_score += min(abs(alignment_score), 3) * 1.2

        # 3. Regime filter (trending environments should have more confidence)
        if regime == "Trending" and adx_14 >= 22:
            if trend == "Bullish":
                bull_score += 1.0
            elif trend == "Bearish":
                bear_score += 1.0
        elif regime == "Weak Trend":
            if trend == "Bullish":
                bull_score += 0.4
            elif trend == "Bearish":
                bear_score += 0.4
            
        # 4. Volume & orderflow (if available)
        if "Accumulation" in volume:
            bull_score += 2.0
        elif "Buying Bias" in volume:
            bull_score += 1.0
        if "Distribution" in volume:
            bear_score += 2.0
        elif "Selling Bias" in volume:
            bear_score += 1.0
            
        # 5. Price action
        if "Bullish Breakout" in pa_struct:
            bull_score += 2.0
        elif "Bearish Breakdown" in pa_struct:
            bear_score += 2.0
        elif "Bullish Structure" in pa_struct:
            bull_score += 1.0
        elif "Bearish Structure" in pa_struct:
            bear_score += 1.0
        
        if "Bullish Engulfing" in pa_pattern:
            bull_score += 1.0
        elif "Bearish Engulfing" in pa_pattern:
            bear_score += 1.0
        elif "Bullish Hammer" in pa_pattern:
            bull_score += 0.6
        elif "Bearish Shooting Star" in pa_pattern:
            bear_score += 0.6

        # 6. RSI momentum
        rsi = ta_data.get("rsi_14", 50)
        if rsi < 30:
            bull_score += 1.5
        elif rsi < 40:
            bull_score += 0.6
        if rsi > 70:
            bear_score += 1.5
        elif rsi > 60:
            bear_score += 0.6
            
        # Verdict decision using directional margin
        score_diff = bull_score - bear_score
        score_margin = abs(score_diff)
        evidence_total = bull_score + bear_score
        
        if score_diff >= 1.2:
            verdict = "Bullish"
        elif score_diff <= -1.2:
            verdict = "Bearish"
        else:
            verdict = "Neutral"

        # Confidence calibration with penalties for lower-quality contexts
        base_conf = 50 + (score_margin * 8.0) + (evidence_total * 1.4)
        penalty = 0.0

        if regime == "Range-Bound":
            penalty += 8.0
        elif regime == "Weak Trend":
            penalty += 3.0

        if "N/A" in volume:
            penalty += 5.0

        if alignment_label == "Mixed / Low Alignment":
            penalty += 6.0

        confidence = base_conf - penalty
        if verdict == "Neutral":
            confidence = min(confidence, 63)
            
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
    app.run(host='0.0.0.0', port=5001, debug=False)
