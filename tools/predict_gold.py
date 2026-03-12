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
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

try:
    import yfinance as yf
    import pandas as pd
    import ta
    from twelvedata import TDClient
except ImportError:
    print(json.dumps({
        "error": "Missing dependencies. Please run: pip install yfinance pandas ta requests twelvedata python-dotenv"
    }))
    sys.exit(1)

# Initialize Twelve Data Client
TD_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
td_client = None
if TD_API_KEY and TD_API_KEY != "your_twelve_data_api_key_here":
    td_client = TDClient(apikey=TD_API_KEY)


def _calc_trend_from_close(close_series):
    """Returns bullish/bearish/neutral trend from EMA stack."""
    if close_series is None or len(close_series) < 55:
        return "Neutral"

    ema_20 = ta.trend.EMAIndicator(close_series, window=20).ema_indicator()
    ema_50 = ta.trend.EMAIndicator(close_series, window=50).ema_indicator()
    latest_close = close_series.iloc[-1]
    latest_ema20 = ema_20.iloc[-1]
    latest_ema50 = ema_50.iloc[-1]

    if pd.isna(latest_ema20) or pd.isna(latest_ema50):
        return "Neutral"

    if latest_close > latest_ema20 > latest_ema50:
        return "Bullish"
    if latest_close < latest_ema20 < latest_ema50:
        return "Bearish"
    return "Neutral"


def _fetch_yf_trend(ticker, interval, period):
    """Fetches a timeframe and returns trend plus datapoints."""
    try:
        df_tf = yf.Ticker(ticker).history(period=period, interval=interval)
        if df_tf.empty or 'Close' not in df_tf.columns:
            return {"trend": "Neutral", "data_points": 0}
        close = pd.to_numeric(df_tf['Close'], errors='coerce').dropna()
        return {
            "trend": _calc_trend_from_close(close),
            "data_points": int(len(close))
        }
    except Exception:
        return {"trend": "Neutral", "data_points": 0}


def _derive_h4_trend_from_h1(ticker):
    """Builds synthetic 4H candles from 1H data to measure higher timeframe trend."""
    try:
        df_h1 = yf.Ticker(ticker).history(period="60d", interval="1h")
        if df_h1.empty:
            return {"trend": "Neutral", "data_points": 0}

        cols = ['Open', 'High', 'Low', 'Close']
        for col in cols:
            if col not in df_h1.columns:
                return {"trend": "Neutral", "data_points": 0}
            df_h1[col] = pd.to_numeric(df_h1[col], errors='coerce')

        h4 = pd.DataFrame()
        h4['Open'] = df_h1['Open'].resample('4h').first()
        h4['High'] = df_h1['High'].resample('4h').max()
        h4['Low'] = df_h1['Low'].resample('4h').min()
        h4['Close'] = df_h1['Close'].resample('4h').last()
        h4 = h4.dropna()

        if h4.empty:
            return {"trend": "Neutral", "data_points": 0}

        return {
            "trend": _calc_trend_from_close(h4['Close']),
            "data_points": int(len(h4))
        }
    except Exception:
        return {"trend": "Neutral", "data_points": 0}

def get_technical_analysis():
    """Fetches 1H interval data for Gold and calculates RSI and EMA."""
    ticker = "GC=F" # Yahoo Ticker
    td_symbol = "XAU/USD" # Twelve Data Symbol
    
    df = pd.DataFrame()
    data_source = "yfinance (Delayed)"
    
    try:
        # Try Twelve Data first if client is available
        if td_client:
            try:
                # 1. Fetch TimeSeries for indicators
                ts = td_client.time_series(symbol=td_symbol, interval="1h", outputsize=100)
                df = ts.as_pandas()
                
                if not df.empty:
                    data_source = "Twelve Data (Real-Time)"
                    # Rename TD columns immediately to prevent KeyErrors later
                    df.columns = [col.capitalize() for col in df.columns]
                    
                    # 2. Separately fetch Real-Time Price for the dashboard tick
                    # This is wrapped so a price failure doesn't kill the TA analysis
                    try:
                        price_data = td_client.price(symbol=td_symbol).as_json()
                        live_price = float(price_data.get('price', 0))
                        if live_price > 0:
                            df.iloc[-1, df.columns.get_loc('Close')] = live_price
                    except Exception as p_err:
                        print(f"Twelve Data Price Tick Error: {p_err}. Using last candle close instead.")

            except Exception as td_err:
                print(f"Twelve Data TimeSeries Error: {td_err}. Falling back to yfinance...")

        # Fallback to yfinance if df is still empty
        if df.empty:
            gold = yf.Ticker(ticker)
            df = gold.history(period="5d", interval="1h")
            
            if df.empty:
                # Last resort: daily data
                df = gold.history(period="1mo", interval="1d")

        if df.empty:
            return {"error": "Failed to fetch price data."}

        # Normalize core OHLCV columns to numeric for stable indicator math
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if df.empty:
            return {"error": "Price data is malformed after cleanup."}

        # Calculate price indicators
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['ADX_14'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        
        # Calculate volume/orderflow proxies if volume data is available
        has_volume = 'Volume' in df.columns and not df['Volume'].empty and (df['Volume'] > 0).any()
        
        if has_volume:
            # On-Balance Volume (OBV) measures cumulative buying vs selling pressure
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            # Chaikin Money Flow (CMF) measures accumulation vs distribution over 14 periods
            df['CMF_14'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).chaikin_money_flow()
        else:
            df['OBV'] = 0
            df['CMF_14'] = 0

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Determine basic EMA trend
        ema_trend = "Neutral"
        if latest['Close'] > latest['EMA_20'] and latest['EMA_20'] > latest['EMA_50']:
            ema_trend = "Bullish"
        elif latest['Close'] < latest['EMA_20'] and latest['EMA_20'] < latest['EMA_50']:
            ema_trend = "Bearish"

        # Market regime filter using ADX (trend strength) + ATR percent (volatility)
        adx_14 = float(latest['ADX_14']) if not pd.isna(latest['ADX_14']) else 0.0
        atr_14 = float(latest['ATR_14']) if not pd.isna(latest['ATR_14']) else 0.0
        atr_pct = (atr_14 / latest['Close'] * 100) if latest['Close'] else 0.0

        market_regime = "Range-Bound"
        if adx_14 >= 22 and atr_pct >= 0.25:
            market_regime = "Trending"
        elif adx_14 >= 18:
            market_regime = "Weak Trend"

        # Price Action: Breakout + swing structure checks
        pa_structure = "Consolidating"
        if len(df) >= 21:
            recent_high = df['High'].iloc[-21:-1].max()
            recent_low = df['Low'].iloc[-21:-1].min()
            if latest['Close'] > recent_high:
                pa_structure = "Bullish Breakout"
            elif latest['Close'] < recent_low:
                pa_structure = "Bearish Breakdown"

        if pa_structure == "Consolidating" and len(df) >= 3:
            p1, p2, p3 = df.iloc[-3], df.iloc[-2], latest
            if p3['High'] > p2['High'] > p1['High'] and p3['Low'] > p2['Low'] > p1['Low']:
                pa_structure = "Higher Highs / Higher Lows (Bullish Structure)"
            elif p3['High'] < p2['High'] < p1['High'] and p3['Low'] < p2['Low'] < p1['Low']:
                pa_structure = "Lower Highs / Lower Lows (Bearish Structure)"

        # Price Action: Single Candle Patterns (Engulfing + reversal candles)
        candle_pattern = "None"
        if latest['Close'] > latest['Open'] and prev['Close'] < prev['Open'] and latest['Close'] > prev['Open'] and latest['Open'] < prev['Close']:
            candle_pattern = "Bullish Engulfing"
        elif latest['Close'] < latest['Open'] and prev['Close'] > prev['Open'] and latest['Close'] < prev['Open'] and latest['Open'] > prev['Close']:
            candle_pattern = "Bearish Engulfing"
        else:
            candle_range = max(latest['High'] - latest['Low'], 1e-8)
            body = abs(latest['Close'] - latest['Open'])
            upper_wick = latest['High'] - max(latest['Close'], latest['Open'])
            lower_wick = min(latest['Close'], latest['Open']) - latest['Low']

            if body / candle_range < 0.15:
                candle_pattern = "Doji"
            elif lower_wick / candle_range > 0.55 and upper_wick / candle_range < 0.2 and latest['Close'] >= latest['Open']:
                candle_pattern = "Bullish Hammer"
            elif upper_wick / candle_range > 0.55 and lower_wick / candle_range < 0.2 and latest['Close'] <= latest['Open']:
                candle_pattern = "Bearish Shooting Star"

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

        # Multi-timeframe trend confirmation (15m + 1h + synthetic 4h)
        m15 = _fetch_yf_trend(ticker=ticker, interval="15m", period="5d")
        h4 = _derive_h4_trend_from_h1(ticker=ticker)
        trend_map = {"Bullish": 1, "Bearish": -1, "Neutral": 0}

        alignment_score = (
            trend_map.get(m15['trend'], 0)
            + trend_map.get(ema_trend, 0)
            + trend_map.get(h4['trend'], 0)
        )

        alignment_label = "Mixed / Low Alignment"
        if alignment_score >= 2:
            alignment_label = "Strong Bullish Alignment"
        elif alignment_score <= -2:
            alignment_label = "Strong Bearish Alignment"

        return {
            "data_source": data_source,
            "current_price": round(latest['Close'], 2),
            "ema_trend": ema_trend,
            "ema_20": round(latest['EMA_20'], 2),
            "ema_50": round(latest['EMA_50'], 2),
            "rsi_14": round(latest['RSI_14'], 2),
            "rsi_signal": rsi_signal,
            "volatility_regime": {
                "market_regime": market_regime,
                "adx_14": round(adx_14, 2),
                "atr_14": round(atr_14, 2),
                "atr_percent": round(atr_pct, 3)
            },
            "multi_timeframe": {
                "m15_trend": m15['trend'],
                "h1_trend": ema_trend,
                "h4_trend": h4['trend'],
                "alignment_score": alignment_score,
                "alignment_label": alignment_label,
                "data_points": {
                    "m15": m15['data_points'],
                    "h4": h4['data_points']
                }
            },
            "price_action": {
                "structure": pa_structure,
                "latest_candle_pattern": candle_pattern
            },
            "volume_analysis": {
                "cmf_14": round(latest['CMF_14'], 4) if has_volume else "N/A",
                "obv_trend": ("Rising" if obv_rising else "Falling") if has_volume else "N/A",
                "overall_volume_signal": volume_signal if has_volume else "N/A (Volume data not available for Spot Gold)"
            },
            "data_points_analyzed": len(df)
        }
    except Exception as e:
        return {"error": str(e)}

def get_sentiment_analysis():
    """Fetches recent news headlines related to gold for sentiment analysis."""
    # GLD is a highly tracked ETF and provides much better news coverage than specific futures tickers
    try:
        gold = yf.Ticker("GLD")
        news = gold.news
        headlines = []
        if news:
            for article in news[:5]:
                # yfinance news structure changed to nest data under 'content'
                content = article.get("content", article) 
                headlines.append({
                    "title": content.get("title", "No Title"),
                    "publisher": content.get("provider", {}).get("displayName", "Unknown"),
                    "link": content.get("clickThroughUrl", {}).get("url", "#")
                })
        
        if not headlines:
            return [{"title": "No recent gold news found.", "publisher": "System", "link": "#"}]
            
        return headlines
    except Exception as e:
        return [{"title": f"Sentiment Query Error: {str(e)}", "publisher": "Error", "link": "#"}]

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
