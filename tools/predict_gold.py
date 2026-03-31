#!/usr/bin/env python3
"""
predict_gold.py
Helper tool for the XAUUSD Prediction Agent.
Uses Twelve Data for price/technical analysis and Yahoo Finance only for
news headlines used in sentiment analysis.

Prerequisites:
    pip install yfinance pandas ta requests twelvedata python-dotenv
"""

import sys
import json
import os
import re
import xml.etree.ElementTree as ET
from html import unescape
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

try:
    import requests
    import yfinance as yf
    import pandas as pd
    import ta
    from twelvedata import TDClient
except ImportError:
    print(json.dumps({
        "error": "Missing dependencies. Please run: pip install yfinance pandas ta requests twelvedata python-dotenv"
    }))
    sys.exit(1)

# Initialize Twelve Data client for all market/technical data.
TD_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
td_client = None
if TD_API_KEY and TD_API_KEY != "your_twelve_data_api_key_here":
    td_client = TDClient(apikey=TD_API_KEY)

DEFAULT_STRATEGY_PARAMS = {
    "ema_short": 20,
    "ema_long": 50,
    "rsi_window": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 20,
    "adx_window": 14,
    "adx_trending_threshold": 22,
    "adx_weak_trend_threshold": 18,
    "atr_window": 14,
    "atr_trending_percent_threshold": 0.25,
    "cmf_window": 14,
    "cmf_strong_buy_threshold": 0.1,
    "cmf_strong_sell_threshold": -0.1,
    "trend_base_weight": 2.5,
    "alignment_weight": 1.2,
    "trend_regime_bonus": 1.0,
    "weak_trend_bonus": 0.4,
    "strong_volume_weight": 2.0,
    "bias_volume_weight": 1.0,
    "breakout_weight": 2.0,
    "structure_weight": 1.0,
    "engulfing_weight": 1.0,
    "reversal_candle_weight": 0.6,
    "rsi_extreme_weight": 1.5,
    "rsi_warning_weight": 0.6,
    "rsi_warning_band": 10,
    "verdict_margin_threshold": 1.2,
    "confidence_margin_multiplier": 8.0,
    "confidence_evidence_multiplier": 1.4,
    "rangebound_penalty": 8.0,
    "weak_trend_penalty": 3.0,
    "volume_unavailable_penalty": 5.0,
    "mixed_alignment_penalty": 6.0,
    "neutral_confidence_cap": 63,
    "mtf_intervals": ["15min", "1h", "4h"],
}


def _load_json_config(relative_path, fallback):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, relative_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            merged = fallback.copy()
            merged.update(data)
            return merged
    except Exception:
        pass
    return fallback.copy()


ACTIVE_STRATEGY_PARAMS = _load_json_config("config/strategy_params.json", DEFAULT_STRATEGY_PARAMS)

RSS_FEEDS = [
    ("Google News Gold", "https://news.google.com/rss/search?q=gold%20OR%20XAUUSD%20OR%20%22Federal%20Reserve%22%20OR%20inflation&hl=en-US&gl=US&ceid=US:en"),
    ("Google News Commodities", "https://news.google.com/rss/search?q=gold%20price%20commodities%20yields%20dollar&hl=en-US&gl=US&ceid=US:en"),
]

BULLISH_TERMS = {
    "gold rises": 2,
    "gold rise": 2,
    "gold gains": 2,
    "gold gain": 2,
    "gold jumps": 2,
    "gold rally": 2,
    "gold higher": 2,
    "record high gold": 3,
    "safe haven": 3,
    "haven demand": 2,
    "geopolitical tensions": 2,
    "fed cuts": 3,
    "rate cut": 3,
    "rate cuts": 3,
    "dovish": 2,
    "easing inflation": 2,
    "inflation cools": 2,
    "soft inflation": 2,
    "weaker dollar": 3,
    "dollar falls": 3,
    "dollar weakens": 3,
    "yields fall": 3,
    "yield falls": 3,
    "real yields fall": 3,
    "central bank buying": 3,
    "buy gold": 2,
    "bullish": 1,
}

BEARISH_TERMS = {
    "gold falls": 2,
    "gold fall": 2,
    "gold drops": 2,
    "gold drop": 2,
    "gold slips": 2,
    "gold declines": 2,
    "gold lower": 2,
    "gold plunge": 3,
    "gold plunges": 3,
    "price crash": 3,
    "gold price crash": 3,
    "worst week": 3,
    "lower u.s. interest rates fade": 3,
    "hopes for lower u.s. interest rates fade": 3,
    "rate-cut bets": 2,
    "rate cut bets": 2,
    "usd rally": 3,
    "fierce usd rally": 3,
    "oil spike": 2,
    "energy prices surge": 2,
    "hawkish": 2,
    "fed hikes": 3,
    "rate hike": 3,
    "rate hikes": 3,
    "higher rates": 3,
    "rates higher for longer": 3,
    "sticky inflation": 2,
    "hot inflation": 2,
    "hot ppi": 3,
    "stronger dollar": 3,
    "dollar rises": 3,
    "dollar strengthens": 3,
    "yields rise": 3,
    "yield rises": 3,
    "real yields rise": 3,
    "profit-taking": 1,
    "sell gold": 2,
    "outflows": 2,
    "bearish": 1,
}


def _calc_trend_from_close(close_series):
    """Returns bullish/bearish/neutral trend from configured EMA stack."""
    if close_series is None or len(close_series) < 55:
        return "Neutral"

    ema_short = int(ACTIVE_STRATEGY_PARAMS.get("ema_short", 20))
    ema_long = int(ACTIVE_STRATEGY_PARAMS.get("ema_long", 50))
    ema_20 = ta.trend.EMAIndicator(close_series, window=ema_short).ema_indicator()
    ema_50 = ta.trend.EMAIndicator(close_series, window=ema_long).ema_indicator()
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


def _fetch_td_trend(symbol, interval, outputsize=200):
    """Fetches trend from Twelve Data timeframe data only."""
    if not td_client:
        return {"trend": "Neutral", "data_points": 0, "source": "none"}

    try:
        ts = td_client.time_series(symbol=symbol, interval=interval, outputsize=outputsize)
        df_tf = ts.as_pandas()
        if df_tf.empty or 'close' not in df_tf.columns:
            return {"trend": "Neutral", "data_points": 0, "source": "twelvedata"}

        close = pd.to_numeric(df_tf['close'], errors='coerce').dropna().sort_index()
        return {
            "trend": _calc_trend_from_close(close),
            "data_points": int(len(close)),
            "source": "twelvedata",
        }
    except Exception:
        return {"trend": "Neutral", "data_points": 0, "source": "twelvedata"}


def _fetch_mtf_trends(td_symbol, h1_trend):
    """Fetches multi-timeframe trends strictly from Twelve Data."""
    mtf_intervals = ACTIVE_STRATEGY_PARAMS.get("mtf_intervals", ["15min", "1h", "4h"])
    m15_interval = mtf_intervals[0] if len(mtf_intervals) > 0 else "15min"
    h4_interval = mtf_intervals[2] if len(mtf_intervals) > 2 else "4h"
    m15 = _fetch_td_trend(symbol=td_symbol, interval=m15_interval, outputsize=200)
    h4 = _fetch_td_trend(symbol=td_symbol, interval=h4_interval, outputsize=200)

    trend_map = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
    alignment_score = (
        trend_map.get(m15.get('trend', 'Neutral'), 0)
        + trend_map.get(h1_trend, 0)
        + trend_map.get(h4.get('trend', 'Neutral'), 0)
    )

    alignment_label = "Mixed / Low Alignment"
    if alignment_score >= 2:
        alignment_label = "Strong Bullish Alignment"
    elif alignment_score <= -2:
        alignment_label = "Strong Bearish Alignment"

    return {
        "m15_trend": m15.get("trend", "Neutral"),
        "h1_trend": h1_trend,
        "h4_trend": h4.get("trend", "Neutral"),
        "alignment_score": alignment_score,
        "alignment_label": alignment_label,
        "data_points": {
            "m15": m15.get("data_points", 0),
            "h4": h4.get("data_points", 0),
        },
        "sources": {
            "m15": m15.get("source", "unknown"),
            "h4": h4.get("source", "unknown"),
        },
    }

def get_technical_analysis():
    """Fetches gold price/technical data from Twelve Data only."""
    td_symbol = "XAU/USD"  # Twelve Data symbol for all technical/price calculations

    if not td_client:
        return {"error": "TWELVE_DATA_API_KEY is missing or invalid. Twelve Data is required."}

    df = pd.DataFrame()
    data_source = "Twelve Data"

    try:
        try:
            ts = td_client.time_series(symbol=td_symbol, interval="1h", outputsize=100)
            df = ts.as_pandas()

            if not df.empty:
                data_source = "Twelve Data (Real-Time)"
                df.columns = [col.capitalize() for col in df.columns]

                try:
                    price_data = td_client.price(symbol=td_symbol).as_json()
                    live_price = float(price_data.get('price', 0))
                    if live_price > 0:
                        df.iloc[-1, df.columns.get_loc('Close')] = live_price
                except Exception as p_err:
                    print(f"Twelve Data Price Tick Error: {p_err}. Using last candle close instead.")
        except Exception as td_err:
            return {"error": f"Twelve Data TimeSeries Error: {td_err}"}

        if df.empty:
            return {"error": "Failed to fetch price data from Twelve Data."}

        # Normalize core OHLCV columns to numeric for stable indicator math
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure oldest->newest ordering for deterministic indicator outputs.
        df = df.sort_index()

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if df.empty:
            return {"error": "Price data is malformed after cleanup."}

        ema_short = int(ACTIVE_STRATEGY_PARAMS.get("ema_short", 20))
        ema_long = int(ACTIVE_STRATEGY_PARAMS.get("ema_long", 50))
        rsi_window = int(ACTIVE_STRATEGY_PARAMS.get("rsi_window", 14))
        atr_window = int(ACTIVE_STRATEGY_PARAMS.get("atr_window", 14))
        adx_window = int(ACTIVE_STRATEGY_PARAMS.get("adx_window", 14))
        cmf_window = int(ACTIVE_STRATEGY_PARAMS.get("cmf_window", 14))

        # Calculate price indicators
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=ema_short).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=ema_long).ema_indicator()
        df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_window).rsi()
        df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_window).average_true_range()
        df['ADX_14'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=adx_window).adx()
        
        # Calculate volume/orderflow proxies if volume data is available
        has_volume = 'Volume' in df.columns and not df['Volume'].empty and (df['Volume'] > 0).any()
        
        if has_volume:
            # On-Balance Volume (OBV) measures cumulative buying vs selling pressure
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            # Chaikin Money Flow (CMF) measures accumulation vs distribution over 14 periods
            df['CMF_14'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=cmf_window).chaikin_money_flow()
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

        adx_trending_threshold = float(ACTIVE_STRATEGY_PARAMS.get("adx_trending_threshold", 22))
        adx_weak_trend_threshold = float(ACTIVE_STRATEGY_PARAMS.get("adx_weak_trend_threshold", 18))
        atr_trending_pct_threshold = float(ACTIVE_STRATEGY_PARAMS.get("atr_trending_percent_threshold", 0.25))

        market_regime = "Range-Bound"
        if adx_14 >= adx_trending_threshold and atr_pct >= atr_trending_pct_threshold:
            market_regime = "Trending"
        elif adx_14 >= adx_weak_trend_threshold:
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

        # If still not broken out/structured, detect directional drift and pressure in range.
        if pa_structure == "Consolidating" and len(df) >= 12:
            recent12 = df.iloc[-12:]
            high_now = recent12['High'].iloc[-4:].mean()
            high_prev = recent12['High'].iloc[-8:-4].mean()
            low_now = recent12['Low'].iloc[-4:].mean()
            low_prev = recent12['Low'].iloc[-8:-4].mean()
            close_now = recent12['Close'].iloc[-3:].mean()
            close_prev = recent12['Close'].iloc[-6:-3].mean()

            if high_now > high_prev and low_now > low_prev and close_now > close_prev:
                pa_structure = "Bullish Drift"
            elif high_now < high_prev and low_now < low_prev and close_now < close_prev:
                pa_structure = "Bearish Drift"

        if pa_structure == "Consolidating" and len(df) >= 20:
            recent20 = df.iloc[-20:]
            range_high = recent20['High'].max()
            range_low = recent20['Low'].min()
            range_size = max(range_high - range_low, 1e-8)
            close_pos = (latest['Close'] - range_low) / range_size

            if ema_trend == "Bullish" and close_pos >= 0.67:
                pa_structure = "Bullish Pressure in Range"
            elif ema_trend == "Bearish" and close_pos <= 0.33:
                pa_structure = "Bearish Pressure in Range"

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

        rsi_overbought = float(ACTIVE_STRATEGY_PARAMS.get("rsi_overbought", 70))
        rsi_oversold = float(ACTIVE_STRATEGY_PARAMS.get("rsi_oversold", 20))
        rsi_signal = "Neutral"
        if latest['RSI_14'] > rsi_overbought:
            rsi_signal = "Overbought (Bearish bias)"
        elif latest['RSI_14'] < rsi_oversold:
            rsi_signal = "Oversold (Bullish bias)"
            
        # Determine volume/orderflow signal
        cmf_strong_buy_threshold = float(ACTIVE_STRATEGY_PARAMS.get("cmf_strong_buy_threshold", 0.10))
        cmf_strong_sell_threshold = float(ACTIVE_STRATEGY_PARAMS.get("cmf_strong_sell_threshold", -0.10))
        volume_signal = "Neutral"
        obv_rising = latest['OBV'] > prev['OBV']
        if latest['CMF_14'] > cmf_strong_buy_threshold and obv_rising:
            volume_signal = "Strong Buying Pressure (Accumulation)"
        elif latest['CMF_14'] < cmf_strong_sell_threshold and not obv_rising:
            volume_signal = "Strong Selling Pressure (Distribution)"
        elif latest['CMF_14'] > 0:
            volume_signal = "Slight Buying Bias"
        elif latest['CMF_14'] < 0:
            volume_signal = "Slight Selling Bias"

        # Multi-timeframe trend confirmation (15m + 1h + 4h) from Twelve Data only.
        mtf = _fetch_mtf_trends(td_symbol=td_symbol, h1_trend=ema_trend)

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
                "m15_trend": mtf['m15_trend'],
                "h1_trend": mtf['h1_trend'],
                "h4_trend": mtf['h4_trend'],
                "alignment_score": mtf['alignment_score'],
                "alignment_label": mtf['alignment_label'],
                "data_points": {
                    "m15": mtf['data_points']['m15'],
                    "h4": mtf['data_points']['h4']
                },
                "sources": mtf['sources']
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
            "data_points_analyzed": len(df),
            "active_strategy_params": ACTIVE_STRATEGY_PARAMS.copy(),
        }
    except Exception as e:
        return {"error": str(e)}

def _clean_headline(text):
    text = unescape((text or "").strip())
    text = re.sub(r"\s+", " ", text)
    return text


def _score_headline_sentiment(title):
    title_lc = (title or "").lower()
    bullish_score = sum(weight for term, weight in BULLISH_TERMS.items() if term in title_lc)
    bearish_score = sum(weight for term, weight in BEARISH_TERMS.items() if term in title_lc)
    net_score = bullish_score - bearish_score

    label = "Neutral"
    if net_score >= 2:
        label = "Bullish"
    elif net_score <= -2:
        label = "Bearish"

    matched_terms = [
        term for term in list(BULLISH_TERMS.keys()) + list(BEARISH_TERMS.keys())
        if term in title_lc
    ][:4]

    return {
        "sentiment_label": label,
        "sentiment_score": net_score,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "matched_terms": matched_terms,
    }


def _fetch_yahoo_sentiment_headlines(limit=5):
    headlines = []
    try:
        gold = yf.Ticker("GLD")
        news = gold.news
        if news:
            for article in news[:limit]:
                content = article.get("content", article)
                title = _clean_headline(content.get("title", "No Title"))
                if not title:
                    continue
                entry = {
                    "title": title,
                    "publisher": content.get("provider", {}).get("displayName", "Yahoo Finance"),
                    "link": content.get("clickThroughUrl", {}).get("url", "#"),
                    "source_type": "yahoo_finance",
                }
                entry.update(_score_headline_sentiment(title))
                headlines.append(entry)
    except Exception:
        pass
    return headlines


def _fetch_rss_sentiment_headlines(limit_per_feed=4):
    headlines = []
    headers = {"User-Agent": "gold-predictor/1.0 (+rss sentiment fetch)"}

    for feed_name, feed_url in RSS_FEEDS:
        try:
            response = requests.get(feed_url, headers=headers, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            items = root.findall(".//item")[:limit_per_feed]

            for item in items:
                title = _clean_headline(item.findtext("title", default=""))
                link = item.findtext("link", default="#") or "#"
                publisher = item.findtext("source", default=feed_name) or feed_name
                if not title:
                    continue
                entry = {
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "source_type": "rss",
                    "feed": feed_name,
                }
                entry.update(_score_headline_sentiment(title))
                headlines.append(entry)
        except Exception:
            continue

    return headlines


def _dedupe_sentiment_headlines(headlines, limit=10):
    deduped = []
    seen = set()

    for item in headlines:
        title_key = re.sub(r"[^a-z0-9]+", " ", (item.get("title") or "").lower()).strip()
        if not title_key or title_key in seen:
            continue
        seen.add(title_key)
        deduped.append(item)
        if len(deduped) >= limit:
            break

    return deduped


def get_sentiment_analysis():
    """Fetches Yahoo + RSS headlines and applies simple local sentiment scoring."""
    try:
        yahoo_headlines = _fetch_yahoo_sentiment_headlines(limit=5)
        rss_headlines = _fetch_rss_sentiment_headlines(limit_per_feed=4)
        combined = yahoo_headlines + rss_headlines
        deduped = _dedupe_sentiment_headlines(combined, limit=10)

        if not deduped:
            return [{
                "title": "No recent gold news found.",
                "publisher": "System",
                "link": "#",
                "source_type": "system",
                "sentiment_label": "Neutral",
                "sentiment_score": 0,
                "bullish_score": 0,
                "bearish_score": 0,
                "matched_terms": [],
            }]

        return deduped
    except Exception as e:
        return [{
            "title": f"Sentiment Query Error: {str(e)}",
            "publisher": "Error",
            "link": "#",
            "source_type": "system",
            "sentiment_label": "Neutral",
            "sentiment_score": 0,
            "bullish_score": 0,
            "bearish_score": 0,
            "matched_terms": [],
        }]

def main():
    ta_data = get_technical_analysis()
    sa_data = get_sentiment_analysis()
    
    output = {
        # TechnicalAnalysis comes from Twelve Data market data.
        "TechnicalAnalysis": ta_data,
        # SentimentalAnalysis is populated from Yahoo Finance headlines only.
        "SentimentalAnalysis": sa_data,
        "FundamentalAnalysis": {
            "note": "For FA (Inflation, Fed Rates, DXY), the agent should ideally query FRED or analyze the news headlines provided."
        }
    }
    
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
