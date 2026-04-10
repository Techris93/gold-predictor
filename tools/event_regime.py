from __future__ import annotations

import math
from datetime import timezone
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _normalized_distance(value: float | None, scale: float = 0.35) -> float:
    if value is None:
        return 0.0
    return _clamp((scale - abs(float(value))) / max(scale, 1e-8), 0.0, 1.0)


def _annotate_event_calendar_features(frame: pd.DataFrame, event_windows: list[dict] | None = None) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["EVENT_ACTIVE"] = 0
    annotated["MINUTES_TO_NEXT_EVENT"] = np.nan

    if not event_windows or annotated.empty or not isinstance(annotated.index, pd.DatetimeIndex):
        return annotated

    next_minutes = []
    event_active = []
    parsed_windows = []
    for window in event_windows:
        if not isinstance(window, dict):
            continue
        try:
            start = pd.Timestamp(str(window.get("start"))).tz_convert("UTC")
            end = pd.Timestamp(str(window.get("end"))).tz_convert("UTC")
        except Exception:
            try:
                start = pd.Timestamp(str(window.get("start")), tz="UTC")
                end = pd.Timestamp(str(window.get("end")), tz="UTC")
            except Exception:
                continue
        parsed_windows.append((start, end))

    for ts in annotated.index:
        ts_utc = pd.Timestamp(ts)
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.tz_localize("UTC")
        else:
            ts_utc = ts_utc.tz_convert("UTC")
        active = 0
        minutes_to_next = np.nan
        upcoming = []
        for start, end in parsed_windows:
            if start <= ts_utc <= end:
                active = 1
                minutes_to_next = 0.0
                break
            if ts_utc < start:
                upcoming.append((start - ts_utc).total_seconds() / 60.0)
        if active == 0 and upcoming:
            minutes_to_next = min(upcoming)
        event_active.append(active)
        next_minutes.append(minutes_to_next)

    annotated["EVENT_ACTIVE"] = event_active
    annotated["MINUTES_TO_NEXT_EVENT"] = next_minutes
    return annotated


def annotate_event_regime_features(frame: pd.DataFrame, event_windows: list[dict] | None = None) -> pd.DataFrame:
    annotated = frame.copy()
    if annotated.empty:
        return annotated

    for col in ["Open", "High", "Low", "Close"]:
        if col in annotated.columns:
            annotated[col] = pd.to_numeric(annotated[col], errors="coerce")

    if "ATR_14" not in annotated.columns:
        prev_close = annotated["Close"].shift(1)
        tr_components = pd.concat(
            [
                annotated["High"] - annotated["Low"],
                (annotated["High"] - prev_close).abs(),
                (annotated["Low"] - prev_close).abs(),
            ],
            axis=1,
        )
        annotated["ATR_14"] = tr_components.max(axis=1).rolling(14, min_periods=2).mean()

    if "ATR_PERCENT" not in annotated.columns:
        annotated["ATR_PERCENT"] = (annotated["ATR_14"] / annotated["Close"]) * 100.0

    prev_close = annotated["Close"].shift(1)
    annotated["TRUE_RANGE"] = pd.concat(
        [
            annotated["High"] - annotated["Low"],
            (annotated["High"] - prev_close).abs(),
            (annotated["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    annotated["RANGE_6"] = annotated["High"].rolling(6, min_periods=2).max() - annotated["Low"].rolling(6, min_periods=2).min()
    annotated["RANGE_24"] = annotated["High"].rolling(24, min_periods=6).max() - annotated["Low"].rolling(24, min_periods=6).min()
    annotated["COMPRESSION_RATIO"] = (annotated["RANGE_6"] / annotated["RANGE_24"]).replace([np.inf, -np.inf], np.nan)
    annotated["ATR_BASELINE_20"] = annotated["ATR_14"].rolling(20, min_periods=5).mean()
    annotated["ATR_EXPANSION_RATIO"] = (annotated["ATR_14"] / annotated["ATR_BASELINE_20"]).replace([np.inf, -np.inf], np.nan)
    annotated["VOL_OF_VOL"] = annotated["ATR_PERCENT"].rolling(12, min_periods=4).std()
    annotated["BAR_VELOCITY"] = ((annotated["Close"] - annotated["Open"]).abs() / annotated["ATR_14"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    annotated["GAP_PCT"] = ((annotated["Open"] - prev_close) / prev_close.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan)
    annotated["WICKINESS"] = (
        ((annotated["High"] - annotated[["Open", "Close"]].max(axis=1)) + (annotated[["Open", "Close"]].min(axis=1) - annotated["Low"]))
        / (annotated["TRUE_RANGE"].replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan)
    annotated["UPPER_WICK"] = (annotated["High"] - annotated[["Open", "Close"]].max(axis=1)).clip(lower=0.0)
    annotated["LOWER_WICK"] = (annotated[["Open", "Close"]].min(axis=1) - annotated["Low"]).clip(lower=0.0)
    annotated["WICK_ASYMMETRY"] = (
        (annotated["UPPER_WICK"] - annotated["LOWER_WICK"])
        / annotated["TRUE_RANGE"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    annotated["WICK_ASYMMETRY_PERSISTENCE"] = annotated["WICK_ASYMMETRY"].abs().rolling(6, min_periods=2).mean()
    annotated["MICRO_RETURN_1"] = annotated["Close"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
    annotated["MICRO_RETURN_BURST"] = (
        annotated["MICRO_RETURN_1"].abs().rolling(3, min_periods=2).mean()
        / annotated["MICRO_RETURN_1"].abs().rolling(12, min_periods=4).mean().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    annotated["VELOCITY_DECAY"] = (
        annotated["BAR_VELOCITY"].rolling(3, min_periods=2).mean()
        / annotated["BAR_VELOCITY"].rolling(12, min_periods=4).mean().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    rolling_high_24 = annotated["High"].rolling(24, min_periods=6).max()
    rolling_low_24 = annotated["Low"].rolling(24, min_periods=6).min()
    annotated["RANGE_POSITION_24"] = ((annotated["Close"] - rolling_low_24) / (rolling_high_24 - rolling_low_24).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    bb_mid = annotated["Close"].rolling(20, min_periods=5).mean()
    bb_std = annotated["Close"].rolling(20, min_periods=5).std()
    bb_width = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid.replace(0, np.nan)
    annotated["SQUEEZE_SCORE"] = ((bb_width / bb_width.rolling(50, min_periods=10).mean()) - 1.0).replace([np.inf, -np.inf], np.nan)
    annotated["SQUEEZE_ON"] = (bb_width <= bb_width.rolling(50, min_periods=10).quantile(0.25)).fillna(False).astype(int)

    if isinstance(annotated.index, pd.DatetimeIndex):
        idx = annotated.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        session_gap_hours = idx.to_series().diff().dt.total_seconds().div(3600.0)
        annotated["SESSION_REOPEN"] = (session_gap_hours.fillna(0) > 12).astype(int).values

        session_day = idx.tz_convert(timezone.utc).date
        temp = pd.DataFrame(index=annotated.index)
        temp["High"] = annotated["High"]
        temp["Low"] = annotated["Low"]
        temp["session_day"] = session_day
        temp["hour"] = idx.hour

        asian_high = temp["High"].where(temp["hour"] < 8).groupby(temp["session_day"]).transform("max")
        asian_low = temp["Low"].where(temp["hour"] < 8).groupby(temp["session_day"]).transform("min")
        asian_range = (asian_high - asian_low).replace(0, np.nan)
        annotated["ASIAN_RANGE_POSITION"] = ((annotated["Close"] - asian_low) / asian_range).replace([np.inf, -np.inf], np.nan)

        daily = temp.groupby("session_day").agg({"High": "max", "Low": "min"})
        prev_daily = daily.shift(1)
        prev_day_high_map = {day: prev_daily.loc[day, "High"] for day in prev_daily.index}
        prev_day_low_map = {day: prev_daily.loc[day, "Low"] for day in prev_daily.index}
        annotated["PRIOR_DAY_HIGH"] = [prev_day_high_map.get(day) for day in session_day]
        annotated["PRIOR_DAY_LOW"] = [prev_day_low_map.get(day) for day in session_day]

        week_period = idx.tz_localize(None).to_period("W")
        weekly = temp.groupby(week_period).agg({"High": "max", "Low": "min"})
        prev_weekly = weekly.shift(1)
        prev_week_high_map = {period: prev_weekly.loc[period, "High"] for period in prev_weekly.index}
        prev_week_low_map = {period: prev_weekly.loc[period, "Low"] for period in prev_weekly.index}
        annotated["PRIOR_WEEK_HIGH"] = [prev_week_high_map.get(period) for period in week_period]
        annotated["PRIOR_WEEK_LOW"] = [prev_week_low_map.get(period) for period in week_period]

        annotated["DIST_PRIOR_DAY_HIGH_PCT"] = (
            (annotated["PRIOR_DAY_HIGH"] - annotated["Close"]) / annotated["Close"].replace(0, np.nan) * 100.0
        )
        annotated["DIST_PRIOR_DAY_LOW_PCT"] = (
            (annotated["Close"] - annotated["PRIOR_DAY_LOW"]) / annotated["Close"].replace(0, np.nan) * 100.0
        )
        annotated["DIST_PRIOR_WEEK_HIGH_PCT"] = (
            (annotated["PRIOR_WEEK_HIGH"] - annotated["Close"]) / annotated["Close"].replace(0, np.nan) * 100.0
        )
        annotated["DIST_PRIOR_WEEK_LOW_PCT"] = (
            (annotated["Close"] - annotated["PRIOR_WEEK_LOW"]) / annotated["Close"].replace(0, np.nan) * 100.0
        )

    annotated = _annotate_event_calendar_features(annotated, event_windows=event_windows)
    return annotated


def summarize_cross_asset_context(cross_asset_context: dict | None) -> dict:
    assets = (cross_asset_context or {}).get("assets") or {}
    bullish_score = 0.0
    bearish_score = 0.0

    def _asset(name: str) -> dict:
        value = assets.get(name) or {}
        return value if isinstance(value, dict) else {}

    dxy = _asset("dxy")
    if dxy.get("available"):
        if _safe_float(dxy.get("pct_1h")) < -0.1:
            bullish_score += 1.4
        elif _safe_float(dxy.get("pct_1h")) > 0.1:
            bearish_score += 1.4

    yields = _asset("us10y")
    if yields.get("available"):
        if _safe_float(yields.get("pct_1h")) < -0.1:
            bullish_score += 1.0
        elif _safe_float(yields.get("pct_1h")) > 0.1:
            bearish_score += 1.0

    silver = _asset("silver")
    if silver.get("available"):
        if _safe_float(silver.get("pct_1h")) > 0.15:
            bullish_score += 1.2
        elif _safe_float(silver.get("pct_1h")) < -0.15:
            bearish_score += 1.2

    oil = _asset("oil")
    if oil.get("available"):
        if _safe_float(oil.get("pct_4h")) > 0.35:
            bullish_score += 0.4
        elif _safe_float(oil.get("pct_4h")) < -0.35:
            bearish_score += 0.4

    spx = _asset("spx")
    if spx.get("available"):
        if _safe_float(spx.get("pct_1h")) < -0.25:
            bullish_score += 0.8
        elif _safe_float(spx.get("pct_1h")) > 0.25:
            bearish_score += 0.4

    usdjpy = _asset("usdjpy")
    if usdjpy.get("available"):
        if _safe_float(usdjpy.get("pct_1h")) < -0.2:
            bullish_score += 0.7
        elif _safe_float(usdjpy.get("pct_1h")) > 0.2:
            bearish_score += 0.4

    bias = "Neutral"
    diff = bullish_score - bearish_score
    if diff >= 0.8:
        bias = "Bullish"
    elif diff <= -0.8:
        bias = "Bearish"

    return {
        "bullish_score": round(bullish_score, 2),
        "bearish_score": round(bearish_score, 2),
        "bias": bias,
        "available_count": int(sum(1 for value in assets.values() if isinstance(value, dict) and value.get("available"))),
    }


def compute_event_regime_snapshot(
    row: pd.Series | dict,
    *,
    trend: str = "Neutral",
    alignment_label: str = "Mixed / Low Alignment",
    market_structure: str = "Consolidating",
    candle_pattern: str = "None",
    event_risk: dict | None = None,
    cross_asset_context: dict | None = None,
    expansion_watch_threshold: float = 48.0,
    high_breakout_threshold: float = 64.0,
    directional_expansion_threshold: float = 78.0,
    previous_state: dict | None = None,
    warning_upshift_buffer: float = 2.0,
    warning_downshift_buffer: float = 4.0,
    min_warning_dwell_bars: int = 3,
    breakout_bias_deadband: float = 0.65,
    breakout_bias_hold_bars: int = 3,
) -> dict:
    get_value = row.get if hasattr(row, "get") else lambda key, default=None: default

    compression_ratio = _safe_float(get_value("COMPRESSION_RATIO"), 1.0)
    atr_expansion_ratio = _safe_float(get_value("ATR_EXPANSION_RATIO"), 1.0)
    vol_of_vol = _safe_float(get_value("VOL_OF_VOL"), 0.0)
    gap_pct = _safe_float(get_value("GAP_PCT"), 0.0)
    bar_velocity = _safe_float(get_value("BAR_VELOCITY"), 0.0)
    micro_return_burst = _safe_float(get_value("MICRO_RETURN_BURST"), 1.0)
    velocity_decay = _safe_float(get_value("VELOCITY_DECAY"), 1.0)
    wick_asym_persistence = _safe_float(get_value("WICK_ASYMMETRY_PERSISTENCE"), 0.0)
    asian_range_pos = get_value("ASIAN_RANGE_POSITION")
    range_pos_24 = get_value("RANGE_POSITION_24")
    atr_14 = _safe_float(get_value("ATR_14"), 0.0)
    session_reopen = bool(int(_safe_float(get_value("SESSION_REOPEN"), 0.0)))
    squeeze_on = bool(int(_safe_float(get_value("SQUEEZE_ON"), 0.0)))
    dist_day_high = get_value("DIST_PRIOR_DAY_HIGH_PCT")
    dist_day_low = get_value("DIST_PRIOR_DAY_LOW_PCT")
    dist_week_high = get_value("DIST_PRIOR_WEEK_HIGH_PCT")
    dist_week_low = get_value("DIST_PRIOR_WEEK_LOW_PCT")

    cross_asset_summary = summarize_cross_asset_context(cross_asset_context)
    event_active = bool((event_risk or {}).get("active")) or bool(int(_safe_float(get_value("EVENT_ACTIVE"), 0.0)))
    minutes_to_next_event = get_value("MINUTES_TO_NEXT_EVENT")
    if minutes_to_next_event is None and isinstance(event_risk, dict):
        next_event = event_risk.get("next_event") or {}
        if isinstance(next_event, dict) and next_event.get("start"):
            try:
                delta = (pd.Timestamp(str(next_event["start"])) - pd.Timestamp.utcnow().tz_localize("UTC")).total_seconds() / 60.0
                minutes_to_next_event = max(delta, 0.0)
            except Exception:
                minutes_to_next_event = None

    compression_score = 0.0
    if compression_ratio <= 0.55:
        compression_score = 18.0
    elif compression_ratio <= 0.75:
        compression_score = 10.0

    expansion_score = 0.0
    if atr_expansion_ratio >= 1.25:
        expansion_score = 16.0
    elif atr_expansion_ratio >= 1.1:
        expansion_score = 8.0

    velocity_score = 0.0
    if bar_velocity >= 1.2:
        velocity_score = 18.0
    elif bar_velocity >= 0.8:
        velocity_score = 10.0

    gap_score = 0.0
    if abs(gap_pct) >= 0.25:
        gap_score = 12.0
    elif abs(gap_pct) >= 0.12:
        gap_score = 6.0

    level_proximity_score = (
        _normalized_distance(dist_day_high)
        + _normalized_distance(dist_day_low)
        + 0.7 * _normalized_distance(dist_week_high, scale=0.55)
        + 0.7 * _normalized_distance(dist_week_low, scale=0.55)
    ) * 7.0

    asian_breakout_score = 0.0
    try:
        asian_value = float(asian_range_pos)
        if asian_value >= 0.85 or asian_value <= 0.15:
            asian_breakout_score = 7.0
    except Exception:
        asian_breakout_score = 0.0

    squeeze_score = 10.0 if squeeze_on else 0.0
    calendar_score = 0.0
    if event_active:
        calendar_score = 24.0
    elif minutes_to_next_event is not None:
        minutes = _safe_float(minutes_to_next_event, 9999.0)
        if minutes <= 30:
            calendar_score = 18.0
        elif minutes <= 90:
            calendar_score = 10.0
        elif minutes <= 240:
            calendar_score = 4.0

    reopen_score = 8.0 if session_reopen else 0.0
    vol_of_vol_score = 6.0 if vol_of_vol >= 0.08 else 0.0
    burst_score = 7.0 if micro_return_burst >= 1.35 else (3.0 if micro_return_burst >= 1.1 else 0.0)
    decay_penalty = 6.0 if velocity_decay <= 0.82 else (2.5 if velocity_decay <= 0.95 else 0.0)
    cross_asset_diff = abs(cross_asset_summary["bullish_score"] - cross_asset_summary["bearish_score"])
    cross_asset_score = cross_asset_diff * 5.0

    compression_setup = compression_ratio <= 0.75 or squeeze_on
    momentum_stack = atr_expansion_ratio >= 1.1 and bar_velocity >= 0.8
    level_pressure = level_proximity_score >= 6.0
    calendar_near = event_active or (_safe_float(minutes_to_next_event, 9999.0) <= 90.0 if minutes_to_next_event is not None else False)
    cross_asset_confluence = cross_asset_summary["bias"] != "Neutral"
    setup_cluster_count = sum(
        1
        for flag in [
            compression_setup,
            momentum_stack,
            level_pressure,
            session_reopen,
            calendar_near,
            asian_breakout_score > 0.0,
            cross_asset_confluence,
        ]
        if flag
    )
    cluster_bonus = 0.0
    if setup_cluster_count >= 5:
        cluster_bonus = 10.0
    elif setup_cluster_count >= 3:
        cluster_bonus = 6.0

    big_move_score = _clamp(
        compression_score
        + expansion_score
        + velocity_score
        + gap_score
        + level_proximity_score
        + asian_breakout_score
        + squeeze_score
        + calendar_score
        + reopen_score
        + vol_of_vol_score
        + burst_score
        - decay_penalty
        + cross_asset_score
        + cluster_bonus
    )

    bullish_bias = 0.0
    bearish_bias = 0.0
    if trend == "Bullish":
        bullish_bias += 1.2
    elif trend == "Bearish":
        bearish_bias += 1.2
    if "Bullish" in alignment_label:
        bullish_bias += 1.1
    elif "Bearish" in alignment_label:
        bearish_bias += 1.1
    if "Bullish" in market_structure:
        bullish_bias += 1.0
    elif "Bearish" in market_structure:
        bearish_bias += 1.0
    if "Bullish" in candle_pattern:
        bullish_bias += 0.5
    elif "Bearish" in candle_pattern:
        bearish_bias += 0.5
    if range_pos_24 is not None:
        range_pos_24 = _safe_float(range_pos_24, 0.5)
        if range_pos_24 >= 0.8:
            bullish_bias += 0.5
        elif range_pos_24 <= 0.2:
            bearish_bias += 0.5
    bullish_bias += cross_asset_summary["bullish_score"]
    bearish_bias += cross_asset_summary["bearish_score"]
    if gap_pct > 0:
        bullish_bias += 0.4
    elif gap_pct < 0:
        bearish_bias += 0.4

    if momentum_stack and trend == "Bullish" and "Bullish" in alignment_label:
        bullish_bias += 0.45
    elif momentum_stack and trend == "Bearish" and "Bearish" in alignment_label:
        bearish_bias += 0.45

    if level_pressure and range_pos_24 is not None:
        range_pos_24 = _safe_float(range_pos_24, 0.5)
        if range_pos_24 >= 0.82 and bullish_bias >= bearish_bias:
            bullish_bias += 0.3
        elif range_pos_24 <= 0.18 and bearish_bias >= bullish_bias:
            bearish_bias += 0.3

    breakout_bias = "Neutral"
    bias_threshold = 1.0
    if cross_asset_summary["bias"] == trend and trend in {"Bullish", "Bearish"}:
        bias_threshold = 0.7
    if (
        ("Drift" in market_structure or "Breakout" in market_structure or "Breakdown" in market_structure)
        and (("Bullish" in alignment_label and trend == "Bullish") or ("Bearish" in alignment_label and trend == "Bearish"))
    ):
        bias_threshold = min(bias_threshold, 0.6)
    if setup_cluster_count >= 5:
        bias_threshold = min(bias_threshold, 0.55)

    if bullish_bias - bearish_bias >= bias_threshold:
        breakout_bias = "Bullish"
    elif bearish_bias - bullish_bias >= bias_threshold:
        breakout_bias = "Bearish"

    directional_confluence_count = 0
    if trend in {"Bullish", "Bearish"} and trend == breakout_bias:
        directional_confluence_count += 1
    if ("Bullish" in alignment_label and breakout_bias == "Bullish") or ("Bearish" in alignment_label and breakout_bias == "Bearish"):
        directional_confluence_count += 1
    if ("Bullish" in market_structure and breakout_bias == "Bullish") or ("Bearish" in market_structure and breakout_bias == "Bearish"):
        directional_confluence_count += 1
    if cross_asset_summary["bias"] == breakout_bias and breakout_bias != "Neutral":
        directional_confluence_count += 1
    if momentum_stack:
        directional_confluence_count += 1
    if level_pressure:
        directional_confluence_count += 1

    immediate_trigger = 0.0
    if atr_expansion_ratio >= 1.15:
        immediate_trigger += 10.0
    if bar_velocity >= 0.9:
        immediate_trigger += 10.0
    if abs(gap_pct) >= 0.18:
        immediate_trigger += 8.0
    if session_reopen:
        immediate_trigger += 6.0
    if event_active:
        immediate_trigger += 12.0

    adaptive_threshold_discount = 0.0
    if compression_setup:
        adaptive_threshold_discount += 4.0
    if momentum_stack:
        adaptive_threshold_discount += 4.0
    if level_pressure:
        adaptive_threshold_discount += 3.0
    if session_reopen:
        adaptive_threshold_discount += 2.0
    if calendar_near:
        adaptive_threshold_discount += 4.0
    if cross_asset_confluence:
        adaptive_threshold_discount += 3.0
    if setup_cluster_count >= 5:
        adaptive_threshold_discount += 6.0
    elif setup_cluster_count >= 3:
        adaptive_threshold_discount += 3.0
    if directional_confluence_count >= 4:
        adaptive_threshold_discount += 5.0
    elif directional_confluence_count >= 3:
        adaptive_threshold_discount += 2.0

    effective_watch_threshold = max(28.0, expansion_watch_threshold - adaptive_threshold_discount)
    effective_high_breakout_threshold = max(42.0, high_breakout_threshold - adaptive_threshold_discount)
    effective_directional_threshold = max(54.0, directional_expansion_threshold - adaptive_threshold_discount)

    expansion_probability_30m = _clamp(big_move_score * 0.62 + immediate_trigger + cluster_bonus * 0.35)
    expansion_probability_60m = _clamp(big_move_score * 0.78 + immediate_trigger * 0.7 + cluster_bonus * 0.45)
    expected_range_expansion = round(atr_14 * (1.0 + max(0.0, atr_expansion_ratio - 1.0) + (0.35 if squeeze_on else 0.0)), 2)

    warning_ladder = "Normal"
    if (
        expansion_probability_30m >= max(effective_directional_threshold + 8.0, 76.0)
        and bar_velocity >= 1.1
        and atr_expansion_ratio >= 1.2
        and micro_return_burst >= 1.15
        and directional_confluence_count >= 4
    ) or (
        bar_velocity >= 1.2 and atr_expansion_ratio >= 1.25 and micro_return_burst >= 1.2
    ):
        warning_ladder = "Active Momentum Event"
    elif (
        expansion_probability_60m >= effective_directional_threshold
        or (
            expansion_probability_60m >= max(effective_high_breakout_threshold, effective_directional_threshold - 8.0)
            and breakout_bias != "Neutral"
            and directional_confluence_count >= 4
        )
    ) and breakout_bias != "Neutral" and directional_confluence_count >= 5 and velocity_decay >= 0.95 and micro_return_burst >= 1.05:
        warning_ladder = "Directional Expansion Likely"
    elif (
        expansion_probability_60m >= effective_high_breakout_threshold
        and (momentum_stack or setup_cluster_count >= 3)
        and velocity_decay >= 0.9
    ):
        warning_ladder = "High Breakout Risk"
    elif expansion_probability_60m >= effective_watch_threshold:
        warning_ladder = "Expansion Watch"

    if (
        warning_ladder in {"High Breakout Risk", "Directional Expansion Likely"}
        and not event_active
        and ("Doji" in candle_pattern or _safe_float(get_value("WICKINESS"), 0.0) >= 0.66)
        and velocity_decay < 1.0
    ):
        warning_ladder = "Expansion Watch"

    event_regime = "normal"
    if event_active:
        event_regime = "event_risk"
    elif warning_ladder == "Active Momentum Event" and breakout_bias != "Neutral":
        if "Doji" in candle_pattern and bar_velocity >= 1.0:
            event_regime = "panic_reversal"
        elif "Drift" in market_structure and "Strong" in alignment_label:
            event_regime = "trend_acceleration"
        else:
            event_regime = "range_expansion"
    elif warning_ladder in {"Directional Expansion Likely", "High Breakout Risk"}:
        event_regime = "breakout_watch"

    fakeout_risk_score = 0.0
    if warning_ladder in {"Expansion Watch", "High Breakout Risk"}:
        wickiness = _safe_float(get_value("WICKINESS"), 0.0)
        if breakout_bias == "Neutral":
            fakeout_risk_score += 2.0
        if not momentum_stack:
            fakeout_risk_score += 1.5
        if "Mixed" in alignment_label:
            fakeout_risk_score += 1.5
        if "Doji" in candle_pattern:
            fakeout_risk_score += 1.0
        if wickiness >= 0.62:
            fakeout_risk_score += 1.0
        if wick_asym_persistence >= 0.38:
            fakeout_risk_score += 0.9
        if velocity_decay <= 0.9:
            fakeout_risk_score += 1.2
        if micro_return_burst < 1.0:
            fakeout_risk_score += 0.8
    fakeout_risk_score = _clamp(fakeout_risk_score, 0.0, 6.0)

    previous_state = previous_state if isinstance(previous_state, dict) else {}
    prev_warning = str(previous_state.get("warning_ladder") or "Normal")
    prev_event_regime = str(previous_state.get("event_regime") or "normal")
    prev_bias = str(previous_state.get("breakout_bias") or "Neutral")
    warning_dwell_bars = max(0, int(_safe_float(previous_state.get("warning_dwell_bars"), 0)))
    breakout_bias_dwell_bars = max(0, int(_safe_float(previous_state.get("breakout_bias_dwell_bars"), 0)))

    ladder_rank = {
        "Normal": 0,
        "Expansion Watch": 1,
        "High Breakout Risk": 2,
        "Directional Expansion Likely": 3,
        "Active Momentum Event": 4,
    }
    stable_warning_rank = ladder_rank.get(prev_warning, ladder_rank.get(warning_ladder, 0))
    target_warning_rank = ladder_rank.get(warning_ladder, 0)

    if target_warning_rank > stable_warning_rank:
        if target_warning_rank <= 2 and expansion_probability_60m < (effective_high_breakout_threshold + float(warning_upshift_buffer)):
            target_warning_rank = stable_warning_rank
        if target_warning_rank >= 3 and expansion_probability_60m < (effective_directional_threshold + float(warning_upshift_buffer)):
            target_warning_rank = stable_warning_rank
    elif target_warning_rank < stable_warning_rank:
        can_downgrade = warning_dwell_bars >= max(1, int(min_warning_dwell_bars))
        if stable_warning_rank >= 3 and expansion_probability_60m >= (effective_high_breakout_threshold - float(warning_downshift_buffer)):
            can_downgrade = False
        if stable_warning_rank == 2 and expansion_probability_60m >= (effective_watch_threshold - float(warning_downshift_buffer)):
            can_downgrade = False
        if not can_downgrade:
            target_warning_rank = stable_warning_rank

    stable_warning_ladder = next(
        (name for name, rank in ladder_rank.items() if rank == target_warning_rank),
        warning_ladder,
    )
    warning_dwell_bars = (warning_dwell_bars + 1) if stable_warning_ladder == prev_warning else 1

    bias_delta = bullish_bias - bearish_bias
    stable_breakout_bias = breakout_bias
    if prev_bias in {"Bullish", "Bearish"} and breakout_bias != prev_bias:
        weak_flip = abs(bias_delta) < max(0.1, float(breakout_bias_deadband))
        if weak_flip or breakout_bias_dwell_bars < max(1, int(breakout_bias_hold_bars)):
            stable_breakout_bias = prev_bias
    breakout_bias_dwell_bars = (breakout_bias_dwell_bars + 1) if stable_breakout_bias == prev_bias else 1

    if event_active:
        stable_event_regime = "event_risk"
    elif stable_warning_ladder == "Active Momentum Event" and stable_breakout_bias != "Neutral":
        if "Doji" in candle_pattern and bar_velocity >= 1.0:
            stable_event_regime = "panic_reversal"
        elif "Drift" in market_structure and "Strong" in alignment_label:
            stable_event_regime = "trend_acceleration"
        else:
            stable_event_regime = "range_expansion"
    elif stable_warning_ladder in {"Directional Expansion Likely", "High Breakout Risk"}:
        stable_event_regime = "breakout_watch"
    else:
        stable_event_regime = "normal"

    if (
        prev_event_regime == "breakout_watch"
        and stable_event_regime == "normal"
        and warning_dwell_bars < max(1, int(min_warning_dwell_bars))
    ):
        stable_event_regime = prev_event_regime

    return {
        "big_move_risk": round(big_move_score, 2),
        "expansion_probability_30m": round(expansion_probability_30m, 2),
        "expansion_probability_60m": round(expansion_probability_60m, 2),
        "expected_range_expansion": expected_range_expansion,
        "breakout_bias": stable_breakout_bias,
        "event_regime": stable_event_regime,
        "warning_ladder": stable_warning_ladder,
        "raw_breakout_bias": breakout_bias,
        "raw_event_regime": event_regime,
        "raw_warning_ladder": warning_ladder,
        "warning_dwell_bars": int(warning_dwell_bars),
        "breakout_bias_dwell_bars": int(breakout_bias_dwell_bars),
        "cross_asset_bias": cross_asset_summary["bias"],
        "cross_asset_available": cross_asset_summary["available_count"],
        "minutes_to_next_event": round(_safe_float(minutes_to_next_event), 1) if minutes_to_next_event is not None else None,
        "event_active": event_active,
        "feature_hits": {
            "compression": compression_ratio <= 0.75,
            "squeeze": squeeze_on,
            "session_reopen": session_reopen,
            "level_proximity": level_proximity_score >= 6.0,
            "atr_expansion": atr_expansion_ratio >= 1.1,
            "bar_velocity": bar_velocity >= 0.8,
            "micro_return_burst": micro_return_burst >= 1.1,
            "velocity_decay_healthy": velocity_decay >= 1.0,
            "wick_asymmetry_persistent": wick_asym_persistence >= 0.3,
            "calendar_risk": event_active or (_safe_float(minutes_to_next_event, 9999.0) <= 90.0 if minutes_to_next_event is not None else False),
            "cross_asset_confirmation": cross_asset_summary["bias"] == stable_breakout_bias and stable_breakout_bias != "Neutral",
            "momentum_stack": momentum_stack,
            "compression_setup": compression_setup,
            "setup_cluster": setup_cluster_count >= 3,
            "directional_confluence": directional_confluence_count >= 3,
            "fakeout_risk": fakeout_risk_score >= 3.0,
        },
        "components": {
            "compression_score": round(compression_score, 2),
            "expansion_score": round(expansion_score, 2),
            "velocity_score": round(velocity_score, 2),
            "gap_score": round(gap_score, 2),
            "level_proximity_score": round(level_proximity_score, 2),
            "asian_breakout_score": round(asian_breakout_score, 2),
            "calendar_score": round(calendar_score, 2),
            "cross_asset_score": round(cross_asset_score, 2),
            "burst_score": round(burst_score, 2),
            "decay_penalty": round(decay_penalty, 2),
            "cluster_bonus": round(cluster_bonus, 2),
            "fakeout_risk_score": round(fakeout_risk_score, 2),
            "directional_confluence_count": int(directional_confluence_count),
            "bias_delta": round(bias_delta, 3),
            "adaptive_threshold_discount": round(adaptive_threshold_discount, 2),
            "effective_watch_threshold": round(effective_watch_threshold, 2),
            "effective_high_breakout_threshold": round(effective_high_breakout_threshold, 2),
            "effective_directional_threshold": round(effective_directional_threshold, 2),
        },
    }
