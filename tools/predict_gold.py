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
from datetime import datetime, timedelta, timezone
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

from tools.twelvedata_market_data import (
    TD_OUTPUT_TIMEZONE,
    canonical_gold_symbol,
    fetch_cross_asset_context,
    fetch_live_price,
    get_td_client,
    normalize_ohlcv_frame,
)
from tools.event_regime import annotate_event_regime_features, compute_event_regime_snapshot
from tools.price_action import classify_price_action
from tools.signal_engine import build_ta_payload_from_row, normalize_strategy_params, prepare_historical_features

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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_PARAMS_PATH = os.path.join(BASE_DIR, "config", "strategy_params.json")
_ACTIVE_STRATEGY_PARAMS_CACHE = {
    "mtime": None,
    "params": normalize_strategy_params(DEFAULT_STRATEGY_PARAMS.copy()),
}


def get_active_strategy_params():
    try:
        mtime = os.path.getmtime(STRATEGY_PARAMS_PATH)
    except OSError:
        mtime = None

    cached = _ACTIVE_STRATEGY_PARAMS_CACHE
    if cached.get("mtime") == mtime and isinstance(cached.get("params"), dict):
        return dict(cached["params"])

    params = normalize_strategy_params(
        _load_json_config("config/strategy_params.json", DEFAULT_STRATEGY_PARAMS)
    )
    cached["mtime"] = mtime
    cached["params"] = params
    return dict(params)


LAST_SUCCESSFUL_TA = None
LAST_SUCCESSFUL_FRAME = None
LAST_TA_REFRESH_TS = 0
LAST_LIVE_PRICE = None
LAST_LIVE_PRICE_TS = 0
LAST_CROSS_ASSET_CONTEXT = None
LAST_CROSS_ASSET_TS = 0
MTF_TREND_CACHE = {}
EVENT_RISK_CONFIG_PATH = os.path.join(BASE_DIR, "config", "event_risk_windows.json")
EVENT_COUNTDOWN_LOOKAHEAD_MINUTES = max(
    0,
    int(os.getenv("EVENT_COUNTDOWN_LOOKAHEAD_MINUTES", str(7 * 24 * 60))),
)
EVENT_NEAR_LOOKAHEAD_MINUTES = max(
    0,
    int(os.getenv("EVENT_NEAR_LOOKAHEAD_MINUTES", str(72 * 60))),
)
TECHNICAL_ANALYSIS_CACHE_SECONDS = max(2, int(os.getenv("TECHNICAL_ANALYSIS_CACHE_SECONDS", "8")))
MTF_CACHE_SECONDS = max(8, int(os.getenv("MTF_CACHE_SECONDS", "20")))
CROSS_ASSET_CACHE_SECONDS = max(15, int(os.getenv("CROSS_ASSET_CACHE_SECONDS", "45")))
LIVE_PRICE_CACHE_SECONDS = max(1, int(os.getenv("LIVE_PRICE_CACHE_SECONDS", "3")))
LIVE_PRICE_HARD_STALE_SECONDS = max(30, int(os.getenv("LIVE_PRICE_HARD_STALE_SECONDS", "90")))
TECHNICAL_BASE_INTERVAL = str(os.getenv("TECHNICAL_BASE_INTERVAL", "15min") or "15min").strip().lower()
INTERVAL_SECONDS = {
    "15m": 15 * 60,
    "15min": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1day": 24 * 60 * 60,
    "1week": 7 * 24 * 60 * 60,
}


def _interval_seconds(interval):
    return int(INTERVAL_SECONDS.get(str(interval or "").strip().lower(), 60 * 60))


def _frame_age_seconds(df, now_ts):
    if not isinstance(df, pd.DataFrame) or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None
    try:
        last_index = df.index[-1]
        if pd.isna(last_index):
            return None
        last_dt = last_index.to_pydatetime()
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        else:
            last_dt = last_dt.astimezone(timezone.utc)
        return max(0, int(now_ts - int(last_dt.timestamp())))
    except Exception:
        return None


def _merge_data_warning(existing, extra):
    current = str(existing or "").strip()
    addition = str(extra or "").strip()
    if not addition:
        return current or None
    if not current:
        return addition
    if addition in current:
        return current
    return f"{current} {addition}"


def _attach_freshness_metadata(result, df, now_ts):
    if not isinstance(result, dict):
        return result

    bar_age_seconds = _frame_age_seconds(df, now_ts)
    freshness_limit_seconds = max(_interval_seconds(TECHNICAL_BASE_INTERVAL) + 120, int(_interval_seconds(TECHNICAL_BASE_INTERVAL) * 1.35))
    result["bar_age_seconds"] = bar_age_seconds
    result["freshness_limit_seconds"] = freshness_limit_seconds

    if bar_age_seconds is None or bar_age_seconds > freshness_limit_seconds:
        result["stale_data"] = True
        result["data_warning"] = _merge_data_warning(
            result.get("data_warning"),
            f"Latest {TECHNICAL_BASE_INTERVAL.upper()} bar is {bar_age_seconds if bar_age_seconds is not None else 'unknown'}s old.",
        )
        if "Real-Time" in str(result.get("data_source") or ""):
            result["data_source"] = "Twelve Data (Delayed Series)"

    live_tick_age_seconds = result.get("live_price_tick_age_seconds")
    if (
        result.get("live_price_applied")
        and isinstance(live_tick_age_seconds, int)
        and live_tick_age_seconds > LIVE_PRICE_HARD_STALE_SECONDS
    ):
        result["stale_data"] = True
        result["data_warning"] = _merge_data_warning(
            result.get("data_warning"),
            f"Live price tick is {int(live_tick_age_seconds)}s old.",
        )
        if "Real-Time" in str(result.get("data_source") or ""):
            result["data_source"] = "Twelve Data (Delayed Tick)"

    return result


def _calc_trend_from_close(close_series):
    """Returns bullish/bearish/neutral trend from configured EMA stack."""
    if close_series is None or len(close_series) < 55:
        return "Neutral"

    strategy_params = get_active_strategy_params()
    ema_short = int(strategy_params.get("ema_short", 20))
    ema_long = int(strategy_params.get("ema_long", 50))
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
        ts = td_client.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            timezone=TD_OUTPUT_TIMEZONE,
        )
        df_tf = normalize_ohlcv_frame(ts.as_pandas())
        if df_tf.empty or 'Close' not in df_tf.columns:
            return {"trend": "Neutral", "data_points": 0, "source": "twelvedata"}

        close = pd.to_numeric(df_tf['Close'], errors='coerce').dropna().sort_index()
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


def _get_live_price_tick(td_symbol, allow_cached=True):
    global LAST_LIVE_PRICE, LAST_LIVE_PRICE_TS
    now_ts = int(time.time())
    if (
        allow_cached
        and isinstance(LAST_LIVE_PRICE, (int, float))
        and float(LAST_LIVE_PRICE) > 0
        and (now_ts - LAST_LIVE_PRICE_TS) < LIVE_PRICE_CACHE_SECONDS
    ):
        return float(LAST_LIVE_PRICE)

    live_price = fetch_live_price(td_symbol)
    if isinstance(live_price, (int, float)) and float(live_price) > 0:
        LAST_LIVE_PRICE = float(live_price)
        LAST_LIVE_PRICE_TS = now_ts
        return float(live_price)

    if (
        isinstance(LAST_LIVE_PRICE, (int, float))
        and float(LAST_LIVE_PRICE) > 0
        and (now_ts - LAST_LIVE_PRICE_TS) <= LIVE_PRICE_HARD_STALE_SECONDS
    ):
        return float(LAST_LIVE_PRICE)
    return None


def _apply_live_price_tick(df, live_price):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    updated = df.copy()
    if not isinstance(live_price, (int, float)) or float(live_price) <= 0:
        return updated

    if "Close" in updated.columns:
        updated.iat[-1, updated.columns.get_loc("Close")] = float(live_price)
    if "High" in updated.columns:
        last_high = updated.iloc[-1]["High"]
        last_high = float(last_high) if not pd.isna(last_high) else float(live_price)
        updated.iat[-1, updated.columns.get_loc("High")] = max(last_high, float(live_price))
    if "Low" in updated.columns:
        last_low = updated.iloc[-1]["Low"]
        last_low = float(last_low) if not pd.isna(last_low) else float(live_price)
        updated.iat[-1, updated.columns.get_loc("Low")] = min(last_low, float(live_price))
    return updated


def _fetch_mtf_trends(td_symbol, h1_trend=None):
    """Fetches multi-timeframe trends strictly from Twelve Data."""
    strategy_params = get_active_strategy_params()
    mtf_intervals = strategy_params.get("mtf_intervals", ["15min", "1h", "4h"])
    m15_interval = mtf_intervals[0] if len(mtf_intervals) > 0 else "15min"
    h1_interval = mtf_intervals[1] if len(mtf_intervals) > 1 else "1h"
    h4_interval = mtf_intervals[2] if len(mtf_intervals) > 2 else "4h"
    m15 = _fetch_td_trend(symbol=td_symbol, interval=m15_interval, outputsize=200)
    h1 = _fetch_td_trend(symbol=td_symbol, interval=h1_interval, outputsize=200)
    h4 = _fetch_td_trend(symbol=td_symbol, interval=h4_interval, outputsize=200)
    h1_value = h1.get("trend", "Neutral")
    if h1_trend in {"Bullish", "Bearish", "Neutral"}:
        h1_value = str(h1_trend)

    trend_map = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
    alignment_score = (
        trend_map.get(m15.get('trend', 'Neutral'), 0)
        + trend_map.get(h1_value, 0)
        + trend_map.get(h4.get('trend', 'Neutral'), 0)
    )

    alignment_label = "Mixed / Low Alignment"
    if alignment_score >= 2:
        alignment_label = "Strong Bullish Alignment"
    elif alignment_score <= -2:
        alignment_label = "Strong Bearish Alignment"

    return {
        "m15_trend": m15.get("trend", "Neutral"),
        "h1_trend": h1_value,
        "h4_trend": h4.get("trend", "Neutral"),
        "alignment_score": alignment_score,
        "alignment_label": alignment_label,
        "data_points": {
            "m15": m15.get("data_points", 0),
            "h1": h1.get("data_points", 0),
            "h4": h4.get("data_points", 0),
        },
        "sources": {
            "m15": m15.get("source", "unknown"),
            "h1": h1.get("source", "unknown"),
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
    def _parse_utc(value):
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    def _release_offset_minutes(name):
        label = str(name or "").lower()
        if any(token in label for token in ["cpi", "ppi", "employment", "nfp"]):
            return 30
        if "fomc" in label:
            return 45
        return 0

    def _release_ts(window, start_dt):
        release = _parse_utc((window or {}).get("release") or (window or {}).get("release_time"))
        if release is not None:
            return int(release.replace(tzinfo=release.tzinfo or timezone.utc).timestamp())
        return int((start_dt + timedelta(minutes=_release_offset_minutes((window or {}).get("name")))).timestamp())

    active = []
    upcoming = []
    release_candidates = []
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
        release_ts = _release_ts(window, start)
        if start_ts <= now_ts <= end_ts:
            active.append(window)
        elif now_ts < start_ts:
            upcoming.append((start_ts, window))
        if release_ts >= now_ts:
            release_candidates.append((release_ts, window))

    next_event = None
    if active:
        active.sort(key=lambda item: str(item.get("start") or ""))
        next_event = active[0]
    elif upcoming:
        upcoming.sort(key=lambda item: item[0])
        next_event = upcoming[0][1]

    sorted_release_candidates = sorted(release_candidates, key=lambda item: item[0]) if release_candidates else []

    next_release_event = None
    minutes_to_next_release = None
    if sorted_release_candidates:
        next_release_ts, next_release_event = sorted_release_candidates[0]
        minutes_to_next_release = max((next_release_ts - now_ts) / 60.0, 0.0)

    near_releases = []
    if EVENT_NEAR_LOOKAHEAD_MINUTES > 0:
        for release_ts, release_window in sorted_release_candidates:
            delta_minutes = max((release_ts - now_ts) / 60.0, 0.0)
            if delta_minutes > EVENT_NEAR_LOOKAHEAD_MINUTES:
                break
            near_releases.append(
                {
                    "name": str((release_window or {}).get("name") or "").strip(),
                    "minutes": round(delta_minutes, 1),
                    "start": (release_window or {}).get("start"),
                    "release": (release_window or {}).get("release") or (release_window or {}).get("release_time"),
                }
            )
            if len(near_releases) >= 3:
                break

    # Large countdown values are rarely actionable intraday and usually indicate a sparse
    # event calendar relative to the dashboard horizon.
    calendar_sparse = False
    if (
        minutes_to_next_release is not None
        and EVENT_COUNTDOWN_LOOKAHEAD_MINUTES > 0
        and minutes_to_next_release > EVENT_COUNTDOWN_LOOKAHEAD_MINUTES
    ):
        calendar_sparse = True
        minutes_to_next_release = None
        next_release_event = None
        if not active:
            next_event = None

    return {
        "active": bool(active),
        "active_events": active[:3],
        "next_event": next_event,
        "next_release_event": next_release_event,
        "minutes_to_next_release": (round(minutes_to_next_release, 1) if minutes_to_next_release is not None else None),
        "near_releases": near_releases,
        "calendar_sparse": calendar_sparse,
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

def _build_technical_analysis_from_frame(
    df,
    td_symbol,
    now_ts,
    data_source,
    served_from_cache=False,
    stale_data=False,
    data_warning=None,
    cache_age_seconds=0,
    live_price_applied=False,
):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Price data frame is empty.")

    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_index()
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    if df.empty:
        raise ValueError("Price data is malformed after cleanup.")

    active_strategy_params = get_active_strategy_params()
    ema_short = int(active_strategy_params.get("ema_short", 20))
    ema_long = int(active_strategy_params.get("ema_long", 50))
    rsi_window = int(active_strategy_params.get("rsi_window", 14))
    atr_window = int(active_strategy_params.get("atr_window", 14))
    adx_window = int(active_strategy_params.get("adx_window", 14))
    cmf_window = int(active_strategy_params.get("cmf_window", 14))

    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=ema_short).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=ema_long).ema_indicator()
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_window).rsi()
    df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_window).average_true_range()
    df['ADX_14'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=adx_window).adx()

    has_volume = 'Volume' in df.columns and not df['Volume'].empty and (df['Volume'] > 0).any()

    if has_volume:
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['CMF_14'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=cmf_window).chaikin_money_flow()
    else:
        df['OBV'] = 0
        df['CMF_14'] = 0

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    ema_trend = "Neutral"
    if latest['Close'] > latest['EMA_20'] and latest['EMA_20'] > latest['EMA_50']:
        ema_trend = "Bullish"
    elif latest['Close'] < latest['EMA_20'] and latest['EMA_20'] < latest['EMA_50']:
        ema_trend = "Bearish"

    adx_14 = float(latest['ADX_14']) if not pd.isna(latest['ADX_14']) else 0.0
    atr_14 = float(latest['ATR_14']) if not pd.isna(latest['ATR_14']) else 0.0
    atr_pct = (atr_14 / latest['Close'] * 100) if latest['Close'] else 0.0

    adx_trending_threshold = float(active_strategy_params.get("adx_trending_threshold", 22))
    adx_weak_trend_threshold = float(active_strategy_params.get("adx_weak_trend_threshold", 18))
    atr_trending_pct_threshold = float(active_strategy_params.get("atr_trending_percent_threshold", 0.25))

    market_regime = "Range-Bound"
    if adx_14 >= adx_trending_threshold and atr_pct >= atr_trending_pct_threshold:
        market_regime = "Trending"
    elif adx_14 >= adx_weak_trend_threshold:
        market_regime = "Weak Trend"

    df["EMA_TREND"] = "Neutral"
    df.loc[(df["Close"] > df["EMA_20"]) & (df["EMA_20"] > df["EMA_50"]), "EMA_TREND"] = "Bullish"
    df.loc[(df["Close"] < df["EMA_20"]) & (df["EMA_20"] < df["EMA_50"]), "EMA_TREND"] = "Bearish"
    pa_structure, candle_pattern = classify_price_action(df, len(df) - 1)

    rsi_overbought = float(active_strategy_params.get("rsi_overbought", 70))
    rsi_oversold = float(active_strategy_params.get("rsi_oversold", 20))
    rsi_signal = "Neutral"
    if latest['RSI_14'] > rsi_overbought:
        rsi_signal = "Overbought (Bearish bias)"
    elif latest['RSI_14'] < rsi_oversold:
        rsi_signal = "Oversold (Bullish bias)"

    cmf_strong_buy_threshold = float(active_strategy_params.get("cmf_strong_buy_threshold", 0.10))
    cmf_strong_sell_threshold = float(active_strategy_params.get("cmf_strong_sell_threshold", -0.10))
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

    mtf = _fetch_mtf_trends(td_symbol=td_symbol, h1_trend=None)
    effective_trend = mtf.get("h1_trend", ema_trend) if TECHNICAL_BASE_INTERVAL in {"15m", "15min"} else ema_trend

    result = {
        "data_source": data_source,
        "base_interval": TECHNICAL_BASE_INTERVAL,
        "last_updated_at": now_ts,
        "stale_data": bool(stale_data),
        "served_from_cache": bool(served_from_cache),
        "cache_age_seconds": max(0, int(cache_age_seconds)),
        "current_price": round(latest['Close'], 2),
        "ema_trend": effective_trend,
        "execution_trend": ema_trend,
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
                "h1": mtf['data_points']['h1'],
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
        "active_strategy_params": active_strategy_params.copy(),
    }
    if data_warning:
        result["data_warning"] = data_warning

    strategy_params = normalize_strategy_params(active_strategy_params)
    enriched = prepare_historical_features(df, strategy_params)
    latest = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) > 1 else latest
    cross_asset_context = _get_cached_cross_asset_context()
    support_resistance = _support_resistance_snapshot(enriched, latest, prev)
    shared_payload = build_ta_payload_from_row(
        latest,
        strategy_params,
        event_risk=result["event_risk"],
        cross_asset_context=cross_asset_context,
        support_resistance=support_resistance,
    )
    result.update(shared_payload)
    result["data_source"] = data_source
    result["base_interval"] = TECHNICAL_BASE_INTERVAL
    result["last_updated_at"] = now_ts
    result["stale_data"] = bool(stale_data)
    result["served_from_cache"] = bool(served_from_cache)
    result["cache_age_seconds"] = max(0, int(cache_age_seconds))
    result["data_points_analyzed"] = len(df)
    result["cross_asset_context"] = cross_asset_context
    result["live_price_applied"] = bool(live_price_applied)
    result["live_price_tick_age_seconds"] = (max(0, now_ts - LAST_LIVE_PRICE_TS) if LAST_LIVE_PRICE_TS else None)
    return _attach_freshness_metadata(result, df, now_ts)


def get_technical_analysis():
    """Fetches gold price/technical data from Twelve Data only."""
    global LAST_SUCCESSFUL_TA, LAST_SUCCESSFUL_FRAME, LAST_TA_REFRESH_TS
    now_ts = int(time.time())
    td_symbol = canonical_gold_symbol("XAU/USD")
    td_client = get_td_client()

    if not td_client:
        return {"error": "TWELVE_DATA_API_KEY is missing or invalid. Twelve Data is required."}

    def _cached_ta(error_message):
        cache_age_seconds = max(0, now_ts - LAST_TA_REFRESH_TS)
        if isinstance(LAST_SUCCESSFUL_FRAME, pd.DataFrame) and not LAST_SUCCESSFUL_FRAME.empty:
            try:
                live_price = _get_live_price_tick(td_symbol, allow_cached=True)
                live_frame = _apply_live_price_tick(LAST_SUCCESSFUL_FRAME, live_price)
                return _build_technical_analysis_from_frame(
                    live_frame,
                    td_symbol=td_symbol,
                    now_ts=now_ts,
                    data_source=(
                        "Twelve Data (Live Price + Cached Series Fallback)"
                        if live_price is not None
                        else "Twelve Data (Cached Series Fallback)"
                    ),
                    served_from_cache=True,
                    stale_data=True,
                    data_warning=error_message,
                    cache_age_seconds=cache_age_seconds,
                    live_price_applied=live_price is not None,
                )
            except Exception:
                pass
        if isinstance(LAST_SUCCESSFUL_TA, dict):
            cached = dict(LAST_SUCCESSFUL_TA)
            cached["stale_data"] = True
            cached["data_warning"] = error_message
            cached["fallback_served_at"] = now_ts
            cached["data_source"] = f"{cached.get('data_source', 'Twelve Data')} (Cached Fallback)"
            cached["served_from_cache"] = True
            cached["cache_age_seconds"] = cache_age_seconds
            return cached
        return {"error": error_message}

    if (
        isinstance(LAST_SUCCESSFUL_FRAME, pd.DataFrame)
        and not LAST_SUCCESSFUL_FRAME.empty
        and (now_ts - LAST_TA_REFRESH_TS) < TECHNICAL_ANALYSIS_CACHE_SECONDS
    ):
        try:
            live_price = _get_live_price_tick(td_symbol, allow_cached=True)
            live_frame = _apply_live_price_tick(LAST_SUCCESSFUL_FRAME, live_price)
            cached_result = _build_technical_analysis_from_frame(
                live_frame,
                td_symbol=td_symbol,
                now_ts=now_ts,
                data_source=(
                    "Twelve Data (Live Price + Cached Series)"
                    if live_price is not None
                    else "Twelve Data (Cached Series)"
                ),
                served_from_cache=True,
                stale_data=False,
                cache_age_seconds=max(0, now_ts - LAST_TA_REFRESH_TS),
                live_price_applied=live_price is not None,
            )
            LAST_SUCCESSFUL_TA = dict(cached_result)
            return cached_result
        except Exception:
            if isinstance(LAST_SUCCESSFUL_TA, dict):
                cached = dict(LAST_SUCCESSFUL_TA)
                cached["served_from_cache"] = True
                cached["cache_age_seconds"] = max(0, now_ts - LAST_TA_REFRESH_TS)
                return cached

    try:
        df = pd.DataFrame()
        last_td_error = None
        live_price = None
        for _ in range(2):
            try:
                ts = td_client.time_series(
                    symbol=td_symbol,
                    interval=TECHNICAL_BASE_INTERVAL,
                    outputsize=200,
                    timezone=TD_OUTPUT_TIMEZONE,
                )
                df = normalize_ohlcv_frame(ts.as_pandas())

                if df.empty:
                    last_td_error = "Twelve Data returned an empty time series."
                    continue

                live_price = _get_live_price_tick(td_symbol, allow_cached=False)
                df = _apply_live_price_tick(df, live_price)
                break
            except Exception as td_err:
                last_td_error = f"Twelve Data TimeSeries Error: {td_err}"

        if df.empty:
            return _cached_ta(last_td_error or "Failed to fetch price data from Twelve Data.")
        result = _build_technical_analysis_from_frame(
            df,
            td_symbol=td_symbol,
            now_ts=now_ts,
            data_source=(
                "Twelve Data (Real-Time)"
                if live_price is not None
                else "Twelve Data (Series Snapshot)"
            ),
            served_from_cache=False,
            stale_data=False,
            cache_age_seconds=0,
            live_price_applied=live_price is not None,
        )
        LAST_SUCCESSFUL_FRAME = df.copy()
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
