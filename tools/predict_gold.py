#!/usr/bin/env python3
"""
predict_gold.py
Helper tool for the XAUUSD Prediction Agent.
Uses Twelve Data for price and technical analysis.

Prerequisites:
    pip install pandas ta requests twelvedata python-dotenv
"""

import sys
import json
import os
import re
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

try:
    import requests
    import pandas as pd
    import ta
except ImportError:
    print(json.dumps({
        "error": "Missing dependencies. Please run: pip install pandas ta requests twelvedata python-dotenv"
    }))
    sys.exit(1)

from tools.twelvedata_market_data import canonical_gold_symbol, get_td_client, fetch_cross_asset_context
from tools.event_regime import annotate_event_regime_features, compute_event_regime_snapshot
from tools.price_action import classify_price_action

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
    "breakout_weight": 0.0,
    "structure_weight": 0.0,
    "swing_structure_weight": 0.0,
    "drift_weight": 1.0,
    "range_pressure_weight": 0.0,
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
    "event_watch_setup_weight": 0.35,
    "event_breakout_setup_weight": 0.7,
    "event_directional_setup_weight": 1.15,
    "event_momentum_setup_weight": 1.55,
    "event_alignment_boost": 0.35,
    "event_conflict_penalty": 0.85,
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
LAST_SUCCESSFUL_TA = None
LAST_TA_REFRESH_TS = 0
LAST_CROSS_ASSET_CONTEXT = None
LAST_CROSS_ASSET_TS = 0
MTF_TREND_CACHE = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENT_RISK_CONFIG_PATH = os.path.join(BASE_DIR, "config", "event_risk_windows.json")
TECHNICAL_ANALYSIS_CACHE_SECONDS = max(5, int(os.getenv("TECHNICAL_ANALYSIS_CACHE_SECONDS", "20")))
MTF_CACHE_SECONDS = max(15, int(os.getenv("MTF_CACHE_SECONDS", "45")))
CROSS_ASSET_CACHE_SECONDS = max(30, int(os.getenv("CROSS_ASSET_CACHE_SECONDS", "90")))


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
    td_client = get_td_client()
    if not td_client:
        return {"trend": "Neutral", "data_points": 0, "source": "none"}

    cache_key = (str(symbol), str(interval), int(outputsize))
    now_ts = int(time.time())
    cached = MTF_TREND_CACHE.get(cache_key)
    if cached and (now_ts - int(cached.get("ts", 0))) < MTF_CACHE_SECONDS:
        return dict(cached.get("payload", {}))

    try:
        ts = td_client.time_series(symbol=symbol, interval=interval, outputsize=outputsize)
        df_tf = ts.as_pandas()
        if df_tf.empty or 'close' not in df_tf.columns:
            return {"trend": "Neutral", "data_points": 0, "source": "twelvedata"}

        close = pd.to_numeric(df_tf['close'], errors='coerce').dropna().sort_index()
        payload = {
            "trend": _calc_trend_from_close(close),
            "data_points": int(len(close)),
            "source": "twelvedata",
        }
        MTF_TREND_CACHE[cache_key] = {"ts": now_ts, "payload": payload}
        return payload
    except Exception:
        return {"trend": "Neutral", "data_points": 0, "source": "twelvedata"}


def _get_cached_cross_asset_context():
    global LAST_CROSS_ASSET_CONTEXT, LAST_CROSS_ASSET_TS
    now_ts = int(time.time())
    if (
        isinstance(LAST_CROSS_ASSET_CONTEXT, dict)
        and (now_ts - LAST_CROSS_ASSET_TS) < CROSS_ASSET_CACHE_SECONDS
    ):
        return dict(LAST_CROSS_ASSET_CONTEXT)

    context = fetch_cross_asset_context()
    if isinstance(context, dict):
        LAST_CROSS_ASSET_CONTEXT = dict(context)
        LAST_CROSS_ASSET_TS = now_ts
    return context


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


def _load_event_risk_windows():
    try:
        with open(EVENT_RISK_CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        windows = payload.get("windows", []) if isinstance(payload, dict) else []
        return windows if isinstance(windows, list) else []
    except Exception:
        return []


def _event_risk_context(now_ts):
    active = []
    upcoming = []
    for window in _load_event_risk_windows():
        if not isinstance(window, dict):
            continue
        try:
            start = datetime.fromisoformat(str(window.get("start")).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(window.get("end")).replace("Z", "+00:00"))
        except Exception:
            continue
        start_ts = int(start.replace(tzinfo=start.tzinfo or timezone.utc).timestamp())
        end_ts = int(end.replace(tzinfo=end.tzinfo or timezone.utc).timestamp())
        if start_ts <= now_ts <= end_ts:
            active.append(window)
        elif now_ts < start_ts:
            upcoming.append((start_ts, window))

    next_event = None
    if upcoming:
        upcoming.sort(key=lambda item: item[0])
        next_event = upcoming[0][1]

    return {
        "active": bool(active),
        "active_events": active[:3],
        "next_event": next_event,
        "now_utc": datetime.now(timezone.utc).isoformat(),
    }


def _support_resistance_snapshot(df, latest, prev):
    if df.empty:
        return {
            "nearest_support": None,
            "nearest_resistance": None,
            "support_distance_pct": None,
            "resistance_distance_pct": None,
            "reaction": "None",
        }

    latest_close = float(latest["Close"])
    price_levels = []

    by_day = df[["High", "Low"]].copy()
    by_day["session_day"] = by_day.index.date
    daily = by_day.groupby("session_day").agg({"High": "max", "Low": "min"})
    if len(daily) >= 2:
        previous_day = daily.iloc[-2]
        price_levels.append(("Previous Day High", float(previous_day["High"]), "resistance"))
        price_levels.append(("Previous Day Low", float(previous_day["Low"]), "support"))

    lookback = df.iloc[-24:] if len(df) >= 24 else df
    if not lookback.empty:
        price_levels.append(("Recent Swing High", float(lookback["High"].max()), "resistance"))
        price_levels.append(("Recent Swing Low", float(lookback["Low"].min()), "support"))

    supports = [(name, value) for name, value, kind in price_levels if kind == "support" and value <= latest_close]
    resistances = [(name, value) for name, value, kind in price_levels if kind == "resistance" and value >= latest_close]
    nearest_support = max(supports, key=lambda item: item[1]) if supports else None
    nearest_resistance = min(resistances, key=lambda item: item[1]) if resistances else None

    support_distance_pct = ((latest_close - nearest_support[1]) / latest_close * 100.0) if nearest_support else None
    resistance_distance_pct = ((nearest_resistance[1] - latest_close) / latest_close * 100.0) if nearest_resistance else None

    reaction = "None"
    candle_range = max(float(latest["High"] - latest["Low"]), 1e-8)
    bullish_close = latest["Close"] > latest["Open"] and latest["Close"] > prev["Close"]
    bearish_close = latest["Close"] < latest["Open"] and latest["Close"] < prev["Close"]

    if nearest_support and support_distance_pct is not None and support_distance_pct <= 0.18 and bullish_close:
        reaction = "Bullish Support Rejection"
    elif nearest_resistance and resistance_distance_pct is not None and resistance_distance_pct <= 0.18 and bearish_close:
        reaction = "Bearish Resistance Rejection"
    elif candle_range > 0:
        breakout_up = nearest_resistance and latest["Close"] > nearest_resistance[1] and latest["Close"] > latest["Open"]
        breakout_down = nearest_support and latest["Close"] < nearest_support[1] and latest["Close"] < latest["Open"]
        if breakout_up:
            reaction = "Bullish Breakout Through Resistance"
        elif breakout_down:
            reaction = "Bearish Breakdown Through Support"

    return {
        "nearest_support": {"label": nearest_support[0], "price": round(nearest_support[1], 2)} if nearest_support else None,
        "nearest_resistance": {"label": nearest_resistance[0], "price": round(nearest_resistance[1], 2)} if nearest_resistance else None,
        "support_distance_pct": round(support_distance_pct, 3) if support_distance_pct is not None else None,
        "resistance_distance_pct": round(resistance_distance_pct, 3) if resistance_distance_pct is not None else None,
        "reaction": reaction,
    }

def get_technical_analysis():
    """Fetches gold price/technical data from Twelve Data only."""
    global LAST_SUCCESSFUL_TA, LAST_TA_REFRESH_TS
    now_ts = int(time.time())
    if (
        isinstance(LAST_SUCCESSFUL_TA, dict)
        and (now_ts - LAST_TA_REFRESH_TS) < TECHNICAL_ANALYSIS_CACHE_SECONDS
    ):
        cached = dict(LAST_SUCCESSFUL_TA)
        cached["served_from_cache"] = True
        cached["cache_age_seconds"] = max(0, now_ts - LAST_TA_REFRESH_TS)
        return cached

    td_symbol = canonical_gold_symbol("XAU/USD")
    td_client = get_td_client()

    if not td_client:
        return {"error": "TWELVE_DATA_API_KEY is missing or invalid. Twelve Data is required."}

    df = pd.DataFrame()
    data_source = "Twelve Data"

    def _cached_ta(error_message):
        if isinstance(LAST_SUCCESSFUL_TA, dict):
            cached = dict(LAST_SUCCESSFUL_TA)
            cached["stale_data"] = True
            cached["data_warning"] = error_message
            cached["fallback_served_at"] = now_ts
            cached["data_source"] = f"{cached.get('data_source', 'Twelve Data')} (Cached Fallback)"
            return cached
        return {"error": error_message}

    try:
        last_td_error = None
        for _ in range(2):
            try:
                ts = td_client.time_series(symbol=td_symbol, interval="1h", outputsize=100)
                df = ts.as_pandas()

                if df.empty:
                    last_td_error = "Twelve Data returned an empty time series."
                    continue

                data_source = "Twelve Data (Real-Time)"
                df.columns = [col.capitalize() for col in df.columns]

                try:
                    price_data = td_client.price(symbol=td_symbol).as_json()
                    if isinstance(price_data, dict):
                        live_price_raw = price_data.get("price")
                        live_price = float(live_price_raw) if live_price_raw is not None else 0.0
                        if live_price > 0 and "Close" in df.columns:
                            df.iloc[-1, df.columns.get_loc('Close')] = live_price
                except Exception as p_err:
                    print(f"Twelve Data Price Tick Error: {p_err}. Using last candle close instead.")
                break
            except Exception as td_err:
                last_td_error = f"Twelve Data TimeSeries Error: {td_err}"

        if df.empty:
            return _cached_ta(last_td_error or "Failed to fetch price data from Twelve Data.")

        # Normalize core OHLCV columns to numeric for stable indicator math
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure oldest->newest ordering for deterministic indicator outputs.
        df = df.sort_index()

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if df.empty:
            return _cached_ta("Price data is malformed after cleanup.")

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
        df["EMA_TREND"] = "Neutral"
        df.loc[(df["Close"] > df["EMA_20"]) & (df["EMA_20"] > df["EMA_50"]), "EMA_TREND"] = "Bullish"
        df.loc[(df["Close"] < df["EMA_20"]) & (df["EMA_20"] < df["EMA_50"]), "EMA_TREND"] = "Bearish"
        pa_structure, candle_pattern = classify_price_action(df, len(df) - 1)

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

        result = {
            "data_source": data_source,
            "last_updated_at": now_ts,
            "stale_data": False,
            "served_from_cache": False,
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
            "support_resistance": _support_resistance_snapshot(df, latest, prev),
            "event_risk": _event_risk_context(int(time.time())),
            "volume_analysis": {
                "cmf_14": round(latest['CMF_14'], 4) if has_volume else "N/A",
                "obv_trend": ("Rising" if obv_rising else "Falling") if has_volume else "N/A",
                "overall_volume_signal": volume_signal if has_volume else "N/A (Volume data not available for Spot Gold)"
            },
            "data_points_analyzed": len(df),
            "active_strategy_params": ACTIVE_STRATEGY_PARAMS.copy(),
        }
        df = annotate_event_regime_features(df, event_windows=_load_event_risk_windows())
        latest = df.iloc[-1]
        cross_asset_context = _get_cached_cross_asset_context()
        result["cross_asset_context"] = cross_asset_context
        result["event_regime"] = compute_event_regime_snapshot(
            latest,
            trend=ema_trend,
            alignment_label=mtf["alignment_label"],
            market_structure=pa_structure,
            candle_pattern=candle_pattern,
            event_risk=result["event_risk"],
            cross_asset_context=cross_asset_context,
            expansion_watch_threshold=float(ACTIVE_STRATEGY_PARAMS.get("expansion_watch_threshold", 48.0)),
            high_breakout_threshold=float(ACTIVE_STRATEGY_PARAMS.get("high_breakout_threshold", 64.0)),
            directional_expansion_threshold=float(ACTIVE_STRATEGY_PARAMS.get("directional_expansion_threshold", 78.0)),
        )

        LAST_SUCCESSFUL_TA = dict(result)
        LAST_TA_REFRESH_TS = now_ts
        return result
    except Exception as e:
        return _cached_ta(str(e))

def main():
    ta_data = get_technical_analysis()

    output = {
        # TechnicalAnalysis comes from Twelve Data market data.
        "TechnicalAnalysis": ta_data,
        "FundamentalAnalysis": {
            "note": "For macro context such as inflation, Fed rates, and DXY, query dedicated macro data sources separately."
        }
    }
    
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
