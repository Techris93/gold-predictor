#!/usr/bin/env python3
"""
predict_gold.py
Helper tool for the XAUUSD Prediction Agent.
Fetches 1H XAUUSD=X data using yfinance (no API key required for TA!), calculates EMA and RSI.
Also fetches recent news headlines for sentiment analysis.

Prerequisites:
    pip install yfinance pandas ta requests
"""

import sys
import json
try:
    import yfinance as yf
    import pandas as pd
    import ta
    import requests
except ImportError:
    print(json.dumps({
        "error": "Missing dependencies. Please run: pip install yfinance pandas ta requests"
    }))
    sys.exit(1)

def get_technical_analysis():
    """Fetches 1H interval data for Gold (XAUUSD=X) and calculates RSI and EMA."""
    ticker = "GC=F" # Gold Futures (or "XAUUSD=X" for spot)
    try:
        gold = yf.Ticker(ticker)
        # Fetch 1H data for the last 5 days
        df = gold.history(period="5d", interval="1h")
        
        if df.empty:
            # Fallback to daily if hourly fails
            df = gold.history(period="1mo", interval="1d")

        if df.empty:
            return {"error": "Failed to fetch price data."}

        # Calculate price indicators
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Calculate volume/orderflow proxies
        # On-Balance Volume (OBV) measures cumulative buying vs selling pressure
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        # Chaikin Money Flow (CMF) measures accumulation vs distribution over 14 periods
        df['CMF_14'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).chaikin_money_flow()

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Determine basic EMA trend
        ema_trend = "Neutral"
        if latest['Close'] > latest['EMA_20'] and latest['EMA_20'] > latest['EMA_50']:
            ema_trend = "Bullish"
        elif latest['Close'] < latest['EMA_20'] and latest['EMA_20'] < latest['EMA_50']:
            ema_trend = "Bearish"

        # Price Action: Higher Highs / Lower Lows structure (last 3 candles)
        pa_structure = "Consolidating"
        if len(df) >= 3:
            p1, p2, p3 = df.iloc[-3], df.iloc[-2], latest
            if p3['High'] > p2['High'] > p1['High'] and p3['Low'] > p2['Low'] > p1['Low']:
                pa_structure = "Higher Highs / Higher Lows (Bullish Structure)"
            elif p3['High'] < p2['High'] < p1['High'] and p3['Low'] < p2['Low'] < p1['Low']:
                pa_structure = "Lower Highs / Lower Lows (Bearish Structure)"

        # Price Action: Single Candle Patterns (Engulfing)
        candle_pattern = "None"
        prev_body = abs(prev['Close'] - prev['Open'])
        latest_body = abs(latest['Close'] - latest['Open'])
        if latest['Close'] > latest['Open'] and prev['Close'] < prev['Open'] and latest['Close'] > prev['Open'] and latest['Open'] < prev['Close']:
            candle_pattern = "Bullish Engulfing"
        elif latest['Close'] < latest['Open'] and prev['Close'] > prev['Open'] and latest['Close'] < prev['Open'] and latest['Open'] > prev['Close']:
            candle_pattern = "Bearish Engulfing"

        rsi_signal = "Neutral"
        if latest['RSI_14'] > 70:
            rsi_signal = "Overbought (Bearish bias)"
        elif latest['RSI_14'] < 20:
            rsi_signal = "Oversold (Bullish bias)"
            
        # Determine volume/orderflow signal
        volume_signal = "Neutral"
        obv_rising = latest['OBV'] > prev['OBV']
        if latest['CMF_14'] > 0.1 and obv_rising:
            volume_signal = "Strong Buying Pressure (Accumulation)"
        elif latest['CMF_14'] < -0.1 and not obv_rising:
            volume_signal = "Strong Selling Pressure (Distribution)"
        elif latest['CMF_14'] > 0:
            volume_signal = "Slight Buying Bias"
        elif latest['CMF_14'] < 0:
            volume_signal = "Slight Selling Bias"

        return {
            "current_price": round(latest['Close'], 2),
            "ema_trend": ema_trend,
            "ema_20": round(latest['EMA_20'], 2),
            "ema_50": round(latest['EMA_50'], 2),
            "rsi_14": round(latest['RSI_14'], 2),
            "rsi_signal": rsi_signal,
            "price_action": {
                "structure": pa_structure,
                "latest_candle_pattern": candle_pattern
            },
            "volume_analysis": {
                "cmf_14": round(latest['CMF_14'], 4),
                "obv_trend": "Rising" if obv_rising else "Falling",
                "overall_volume_signal": volume_signal
            },
            "data_points_analyzed": len(df)
        }
    except Exception as e:
        return {"error": str(e)}

def get_sentiment_analysis():
    """Fetches recent news headlines related to gold for sentiment analysis."""
    # We use yfinance news as a free built-in proxy without needing API keys
    try:
        gold = yf.Ticker("GC=F")
        news = gold.news
        headlines = []
        for article in news[:5]:
            headlines.append({
                "title": article.get("title", ""),
                "publisher": article.get("publisher", ""),
                "link": article.get("link", "")
            })
        return headlines
    except Exception as e:
        return {"error": str(e)}

def main():
    ta_data = get_technical_analysis()
    sa_data = get_sentiment_analysis()
    
    output = {
        "TechnicalAnalysis": ta_data,
        "SentimentalAnalysis": sa_data,
        "FundamentalAnalysis": {
            "note": "For FA (Inflation, Fed Rates, DXY), the agent should ideally query FRED or analyze the news headlines provided."
        }
    }
    
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
