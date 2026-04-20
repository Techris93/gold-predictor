import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import ta

from tools.event_regime import annotate_event_regime_features, compute_event_regime_snapshot
from tools.price_action import annotate_price_action, extract_price_action_feature_hits

FRANKFURT_TZ = ZoneInfo("Europe/Berlin")
LONDON_TZ = ZoneInfo("Europe/London")
NEW_YORK_TZ = ZoneInfo("America/New_York")


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
    "rsi_divergence_weight": 0.9,
    "macd_early_momentum_weight": 0.55,
    "macd_hist_slope_scale": 0.06,
    "volume_spike_trigger_weight": 0.5,
    "volume_spike_zscore_threshold": 1.8,
    "verdict_margin_threshold": 1.2,
    "confidence_margin_multiplier": 8.0,
    "confidence_evidence_multiplier": 1.4,
    "rangebound_penalty": 8.0,
    "weak_trend_penalty": 3.0,
    "volume_unavailable_penalty": 5.0,
    "mixed_alignment_penalty": 6.0,
    "neutral_confidence_cap": 63,
    "sr_reaction_weight": 1.6,
    "sr_proximity_weight": 0.5,
    "pivot_confluence_weight": 0.35,
    "special_sr_confluence_weight": 0.18,
    "trigger_min_score": 1.3,
    "no_trade_adx_threshold": 18,
    "no_trade_confidence_cap": 60,
    "event_risk_penalty": 15.0,
    "regime_quality_weight": 0.30,
    "alignment_quality_weight": 0.20,
    "structure_quality_weight": 0.20,
    "trigger_quality_weight": 0.15,
    "volume_quality_weight": 0.10,
    "stability_flip_penalty": 12.0,
    "stability_conflict_penalty": 10.0,
    "stability_mixed_alignment_penalty": 10.0,
    "high_tradeability_threshold": 68.0,
    "medium_tradeability_threshold": 52.0,
    "direction_entry_threshold": 8.0,
    "direction_hold_threshold": 5.0,
    "exit_risk_threshold": 6.0,
    "mtf_intervals": ["15min", "1h", "4h"],
    "expansion_watch_threshold": 48.0,
    "high_breakout_threshold": 64.0,
    "directional_expansion_threshold": 78.0,
    "event_watch_setup_weight": 0.35,
    "event_breakout_setup_weight": 0.7,
    "event_directional_setup_weight": 1.15,
    "event_momentum_setup_weight": 1.55,
    "event_alignment_boost": 0.35,
    "event_conflict_penalty": 0.85,
    "anti_chop_margin_buffer": 0.8,
    "anti_chop_trigger_buffer": 0.35,
    "anti_chop_tradeability_floor": 68.0,
    "anti_chop_penalty": 8.0,
    "warning_upshift_buffer": 3.5,
    "warning_downshift_buffer": 7.0,
    "warning_min_dwell_bars": 5,
    "warning_raw_streak_required": 2,
    "breakout_bias_deadband": 0.9,
    "breakout_bias_hold_bars": 4,
    "fakeout_risk_penalty": 6.0,
    "meta_entry_score_threshold": 63.0,
    "meta_fakeout_prob_cap": 0.42,
    "meta_exit_prob_cap": 0.58,
    "min_expected_edge_pct": 0.06,
    "direction_probability_floor": 0.56,
    "transition_setup_tradeability_floor": 54.0,
    "transition_setup_entry_prob_floor": 0.57,
    "breakout_setup_tradeability_floor": 28.0,
    "breakout_setup_direction_prob_30_floor": 0.62,
    "breakout_setup_direction_prob_60_floor": 0.68,
    "breakout_setup_expected_edge_floor": 0.08,
    "breakout_setup_meta_probability_floor": 0.56,
    "breakout_setup_fakeout_probability_cap": 0.48,
    "breakout_setup_exit_probability_cap": 0.38,
    "breakout_setup_projected_move_atr_floor": 1.10,
    "breakout_setup_one_atr_probability_floor": 0.46,
    "breakout_active_tradeability_floor": 32.0,
    "breakout_active_direction_prob_30_floor": 0.72,
    "breakout_active_direction_prob_60_floor": 0.78,
    "breakout_active_expected_edge_floor": 0.10,
    "breakout_active_meta_probability_floor": 0.62,
    "breakout_active_entry_timing_floor": 78.0,
    "breakout_active_fakeout_probability_cap": 0.42,
    "breakout_active_exit_probability_cap": 0.34,
    "breakout_active_projected_move_atr_floor": 1.20,
    "breakout_active_one_atr_probability_floor": 0.50,
    "smooth_adx_center": 20.0,
    "smooth_adx_scale": 2.6,
    "smooth_atr_percent_center": 0.24,
    "smooth_atr_percent_scale": 0.045,
    "move_bucket_base_atr": 1.0,
    "triple_barrier_stop_atr": 0.85,
    "triple_barrier_target_atr": 1.25,
    "triple_barrier_horizon_bars": 8,
    "estimated_avg_win_pct": 0.55,
    "estimated_avg_loss_pct": 0.42,
    "transaction_cost_pct": 0.05,
    "rr_signal_sl_pips": 100,
    "rr_signal_tp_pips": 200,
    "rr_signal_target_move_pips": 200,
    "rr_signal_pip_size": 0.1,
    "rr_signal_min_confidence": 70.0,
    "rr_signal_min_tradeability_score": 56.0,
    "rr_signal_min_move_probability": 0.56,
    "rr_signal_min_expected_edge_pct": 0.05,
    "rr_signal_allow_b_grade": 1,
    "rr_signal_b_min_move_probability": 0.60,
    "rr_signal_b_min_session_quality": 0.72,
    "rr_signal_b_require_active_state": 1,
    "rr_signal_b_min_mtf_matches": 3,
    "rr_signal_strong_stack_target_atr_cap": 1.20,
    "rr_signal_allow_soft_no_trade": 1,
    "rr_signal_soft_no_trade_terms": [
        "Expected value edge is below",
        "Entry timing model is not yet confirming momentum quality",
        "Tradeability is still below the execution floor",
    ],
    "rr_signal_max_hold_bars": 24,
    "rr_signal_max_concurrent_positions": 2,
    "rr_signal_reentry_cooldown_bars": 4,
    "rr_signal_risk_fraction": 0.01,
    "rr_signal_partial_take_profit_pips": 100,
    "rr_signal_move_sl_to_be_after_partial": 1,
    "rr_signal_trade_hours_utc": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "rr_signal_require_m15_trigger": 1,
    "rr_signal_max_signals_per_day": 5,
    "rr_signal_h1_neutral_override_enabled": 1,
    "rr_signal_h1_neutral_min_expansion_60m": 66.0,
    "rr_signal_h1_neutral_min_big_move_risk": 62.0,
    "rr_signal_h1_neutral_min_direction_probability": 0.62,
    "rr_signal_h1_neutral_min_tradeability": 56.0,
    "rr_signal_h1_neutral_min_confidence": 66.0,
    "rr_signal_h1_neutral_min_move_probability": 0.40,
    "rr_signal_h1_neutral_require_h4_non_opposition": 1,
    "rr_signal_h1_neutral_allow_ready": 0,
}

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIDENCE_CALIBRATION_FILE = BASE_DIR / "tools" / "reports" / "confidence_calibration.json"
REGIME_PARAMS_FILE = BASE_DIR / "config" / "regime_params.json"


def normalize_strategy_params(params=None):
    merged = DEFAULT_STRATEGY_PARAMS.copy()
    if isinstance(params, dict):
        merged.update(params)
    return merged


def _safe_float(value, default=0.0):
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _optional_float(value):
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except Exception:
        return None


def _round_or_none(value, digits=2):
    number = _optional_float(value)
    if number is None:
        return None
    return round(number, digits)


def _round_number_step(price):
    normalized_price = abs(_safe_float(price, 0.0))
    if normalized_price >= 1000.0:
        return 5.0
    if normalized_price >= 100.0:
        return 1.0
    if normalized_price >= 10.0:
        return 0.5
    return 0.1


def _level_family(label):
    normalized = str(label or "").strip().lower()
    if "pivot" in normalized:
        return "pivot"
    if "round number" in normalized:
        return "round"
    if "fvg" in normalized:
        return "fvg"
    if "range zone" in normalized:
        return "range"
    if "swing" in normalized:
        return "swing"
    if "previous" in normalized or "prior" in normalized:
        return "prior"
    return "other"


def _serialize_zone(low, high, label=None):
    low_value = _optional_float(low)
    high_value = _optional_float(high)
    if low_value is None or high_value is None:
        return None
    zone_low = min(low_value, high_value)
    zone_high = max(low_value, high_value)
    zone = {
        "low": round(zone_low, 2),
        "high": round(zone_high, 2),
        "mid": round((zone_low + zone_high) / 2.0, 2),
        "width": round(zone_high - zone_low, 2),
    }
    if label:
        zone["label"] = str(label)
    return zone


def _serialize_nearby_level(label, price, current_price):
    level_price = _optional_float(price)
    if level_price is None or current_price <= 0:
        return None
    distance_pct = abs(current_price - level_price) / current_price * 100.0
    return {
        "label": str(label),
        "family": _level_family(label),
        "price": round(level_price, 2),
        "distance_pct": round(distance_pct, 3),
    }


def _collect_nearby_levels(levels, current_price, kind, cutoff_pct=0.25, max_items=5):
    matches = []
    for label, value, level_kind in levels:
        if level_kind != kind:
            continue
        level_price = _optional_float(value)
        if level_price is None or current_price <= 0:
            continue
        distance_pct = abs(current_price - level_price) / current_price * 100.0
        if distance_pct <= (cutoff_pct + 1e-9):
            matches.append((distance_pct, str(label), level_price))
    matches.sort(key=lambda item: (item[0], abs(current_price - item[2]), item[1]))
    serialized = []
    seen = set()
    for _, label, level_price in matches:
        dedupe_key = (label, round(level_price, 4))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        payload = _serialize_nearby_level(label, level_price, current_price)
        if payload:
            serialized.append(payload)
        if len(serialized) >= max_items:
            break
    return serialized


def _nearest_family_distance_pct(levels, family):
    distances = []
    target_family = str(family or "")
    for level in levels:
        if not isinstance(level, dict):
            continue
        if str(level.get("family") or "") != target_family:
            continue
        distance_pct = _optional_float(level.get("distance_pct"))
        if distance_pct is not None:
            distances.append(distance_pct)
    return min(distances) if distances else None


def _round_distance_is_actionable(distance_pct, current_price, round_step):
    distance_value = _optional_float(distance_pct)
    price_value = _optional_float(current_price)
    step_value = _optional_float(round_step)
    if (
        distance_value is None
        or price_value is None
        or price_value <= 0.0
        or step_value is None
        or step_value <= 0.0
    ):
        return False
    distance_price = price_value * (distance_value / 100.0)
    threshold_price = max(step_value * 0.18, price_value * 0.00008)
    return distance_price <= (threshold_price + 1e-9)


def _coerce_signal_state(value):
    return int(round(_safe_float(value, 0.0)))


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def _sigmoid(value, center=0.0, scale=1.0):
    scale = max(abs(float(scale)), 1e-6)
    normalized = (float(value) - float(center)) / scale
    normalized = max(-60.0, min(60.0, normalized))
    return 1.0 / (1.0 + math.exp(-normalized))


def _sigmoid_series(series, center=0.0, scale=1.0):
    scale = max(abs(float(scale)), 1e-6)
    normalized = ((series.astype(float) - float(center)) / scale).clip(-60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-normalized))


def _coerce_utc_datetime(value):
    if value is None:
        return None
    try:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed.to_pydatetime()
    except Exception:
        return None


def _format_utc_time_label(session_dt):
    if not isinstance(session_dt, datetime):
        return None
    return session_dt.astimezone(timezone.utc).strftime("%H:%M UTC")


def _build_session_context_from_datetime(session_dt):
    normalized_dt = _coerce_utc_datetime(session_dt) or datetime.now(timezone.utc)
    hour = int(normalized_dt.hour)
    minute = int(normalized_dt.minute)
    london_dt = normalized_dt.astimezone(LONDON_TZ)
    frankfurt_dt = normalized_dt.astimezone(FRANKFURT_TZ)
    new_york_dt = normalized_dt.astimezone(NEW_YORK_TZ)
    london_hour = int(london_dt.hour)
    london_minute = int(london_dt.minute)
    frankfurt_hour = int(frankfurt_dt.hour)
    new_york_hour = int(new_york_dt.hour)

    is_sydney = hour >= 21 or hour < 6
    is_tokyo = 0 <= hour < 9
    is_asia = is_tokyo
    is_frankfurt = 8 <= frankfurt_hour < 17
    is_london = 8 <= london_hour < 17
    is_new_york = 8 <= new_york_hour < 17
    is_overlap = is_london and is_new_york
    is_tokyo_london_overlap = is_tokyo and is_london
    is_tokyo_frankfurt_london_overlap = is_tokyo and is_frankfurt and is_london
    is_frankfurt_london_overlap = is_frankfurt and is_london
    is_sydney_tokyo_overlap = is_sydney and is_tokyo
    is_london_open = 8 <= london_hour < 10
    is_new_york_open = 8 <= new_york_hour < 10
    is_comex_open = 9 <= new_york_hour < 14
    is_fix_window = london_hour == 16 or (london_hour == 15 and london_minute >= 45)

    label = "Off"
    quality = 0.46
    if is_tokyo_frankfurt_london_overlap:
        label = "Tokyo / Frankfurt / London Overlap"
        quality = 0.95
    elif is_overlap:
        label = "London / New York Overlap"
        quality = 0.92
    elif is_tokyo_london_overlap:
        label = "Tokyo / London Overlap"
        quality = 0.9
    elif is_frankfurt_london_overlap:
        label = "Frankfurt / London Overlap"
        quality = 0.89
    elif is_sydney_tokyo_overlap:
        label = "Sydney / Tokyo Overlap"
        quality = 0.68
    elif is_frankfurt:
        label = "Frankfurt"
        quality = 0.78
    elif is_london:
        label = "London"
        quality = 0.84
    elif is_new_york:
        label = "New York"
        quality = 0.8
    elif is_tokyo:
        label = "Tokyo"
        quality = 0.62
    elif is_sydney:
        label = "Sydney"
        quality = 0.56

    if is_london_open or is_new_york_open:
        quality = min(0.96, quality + 0.06)
    if is_fix_window:
        quality = min(0.98, quality + 0.03)

    return {
        "label": label,
        "hour": hour,
        "minute": minute,
        "quality": round(quality, 4),
        "isSydneyOpen": is_sydney,
        "isTokyoOpen": is_tokyo,
        "isAsiaOpen": is_asia,
        "isFrankfurtOpen": is_frankfurt,
        "isLondonOpen": is_london_open,
        "isNewYorkOpen": is_new_york_open,
        "isComexOpen": is_comex_open,
        "isFixWindow": is_fix_window,
        "isOverlap": is_overlap,
        "timestampUtc": normalized_dt.isoformat(),
        "timeDisplayUtc": _format_utc_time_label(normalized_dt),
        "timezone": "UTC",
    }


def _resolve_bar_datetime(ta_data):
    bar_ts = (ta_data or {}).get("bar_timestamp_utc")
    return _coerce_utc_datetime(bar_ts) or datetime.now(timezone.utc)


def _build_session_context(ta_data):
    return _build_session_context_from_datetime(_resolve_bar_datetime(ta_data))


def _alignment_quality_score(alignment_score, alignment_label):
    label = str(alignment_label or "")
    score = min(abs(int(alignment_score or 0)) / 3.0, 1.0)
    if label.startswith("Strong"):
        score = max(score, 0.82)
    elif "Mixed" in label:
        score = min(score, 0.34)
    return _clamp(score)


def _bias_strength(label, bullish_label="Bullish", bearish_label="Bearish"):
    label = str(label or "")
    if bullish_label in label:
        return 1.0
    if bearish_label in label:
        return -1.0
    return 0.0


def _build_regime_router(
    ta_data,
    regime_state,
    trend,
    alignment_score,
    market_structure,
    candle_pattern,
):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    regime_state = regime_state if isinstance(regime_state, dict) else {}
    market_regime_scores = ta_data.get("market_regime_scores") or {}
    volatility_features = ta_data.get("volatility_features") or {}
    structure_context = ta_data.get("structure_context") or {}
    session_context = _build_session_context(ta_data)

    trend_probability = _safe_float(market_regime_scores.get("trend_probability"), 0.0)
    expansion_probability = _safe_float(market_regime_scores.get("expansion_probability"), 0.0)
    chop_probability = _safe_float(market_regime_scores.get("chop_probability"), 0.0)
    realized_vol_percentile = _safe_float(volatility_features.get("realizedVolPercentile"), 50.0) / 100.0
    atr_percentile = _safe_float(volatility_features.get("atrPercentile"), 50.0) / 100.0
    follow_through = _safe_float(volatility_features.get("trendFollowThrough"), 1.0)
    opening_range_break = _coerce_signal_state(structure_context.get("openingRangeBreak"))
    sweep_reclaim_signal = _coerce_signal_state(structure_context.get("sweepReclaimSignal"))
    sweep_reclaim_quality = _safe_float(structure_context.get("sweepReclaimQuality"), 0.0)
    sweep_signal_strength = min(max(sweep_reclaim_quality, 0.0), 1.0) if sweep_reclaim_signal != 0 else 0.0
    breakout_bias = str(regime_state.get("breakout_bias") or "Neutral")
    cross_asset_bias = str(regime_state.get("cross_asset_bias") or "Neutral")
    warning_ladder = str(regime_state.get("warning_ladder") or "Normal")
    event_regime = str(regime_state.get("event_regime") or "normal")
    big_move_risk = _safe_float(regime_state.get("big_move_risk"), 0.0) / 100.0
    expansion_30m = _safe_float(regime_state.get("expansion_probability_30m"), 0.0) / 100.0
    expansion_60m = _safe_float(regime_state.get("expansion_probability_60m"), 0.0) / 100.0
    event_shock = _safe_float((regime_state.get("components") or {}).get("event_shock_probability"), 0.0)

    structure_stack = 1.0 if (trend == "Bullish" and "Bullish" in str(market_structure or "")) or (trend == "Bearish" and "Bearish" in str(market_structure or "")) else 0.0
    candle_stack = 1.0 if (trend == "Bullish" and "Bullish" in str(candle_pattern or "")) or (trend == "Bearish" and "Bearish" in str(candle_pattern or "")) else 0.0
    bias_confluence = 1.0 if breakout_bias == trend and trend in {"Bullish", "Bearish"} else 0.55 if breakout_bias in {"Bullish", "Bearish"} else 0.35
    cross_confluence = 1.0 if cross_asset_bias == trend and trend in {"Bullish", "Bearish"} else 0.55 if cross_asset_bias in {"Bullish", "Bearish"} else 0.35
    session_quality = _safe_float(session_context.get("quality"), 0.46)

    trend_expert = _clamp(
        (trend_probability * 0.34)
        + (_alignment_quality_score(alignment_score, ta_data.get("multi_timeframe", {}).get("alignment_label")) * 0.18)
        + (structure_stack * 0.12)
        + (candle_stack * 0.05)
        + (follow_through >= 1.0) * 0.08
        + (bias_confluence * 0.11)
        + (cross_confluence * 0.07)
        + (session_quality * 0.05)
        - (event_shock * 0.10)
    )

    breakout_expert = _clamp(
        (expansion_probability * 0.22)
        + (big_move_risk * 0.18)
        + (expansion_60m * 0.18)
        + (expansion_30m * 0.08)
        + (1.0 if opening_range_break != 0 else 0.0) * 0.10
        + sweep_signal_strength * 0.07
        + (1.0 if warning_ladder in {"High Breakout Risk", "Directional Expansion Likely", "Active Momentum Event"} else 0.0) * 0.09
        + (session_quality * 0.08)
        + (realized_vol_percentile * 0.05)
        + (atr_percentile * 0.05)
    )

    mean_reversion_expert = _clamp(
        (chop_probability * 0.35)
        + ((1.0 - expansion_probability) * 0.12)
        + ((1.0 - expansion_60m) * 0.10)
        + sweep_signal_strength * 0.08
        + (1.0 if "Doji" in str(candle_pattern or "") else 0.0) * 0.08
        + ((1.0 - _alignment_quality_score(alignment_score, ta_data.get("multi_timeframe", {}).get("alignment_label"))) * 0.15)
        + ((1.0 - session_quality) * 0.07)
    )

    event_expert = _clamp(
        (event_shock * 0.52)
        + (1.0 if event_regime == "event_risk" else 0.0) * 0.18
        + (big_move_risk * 0.10)
        + (1.0 if warning_ladder == "Active Momentum Event" else 0.0) * 0.08
        + (1.0 if session_context["isNewYorkOpen"] or session_context["isComexOpen"] else 0.0) * 0.06
        + (realized_vol_percentile * 0.06)
    )

    unstable_expert = _clamp(
        1.0 - max(trend_expert, breakout_expert, mean_reversion_expert, event_expert) * 0.92
    )

    expert_scores = {
        "trend_continuation": trend_expert,
        "breakout_transition": breakout_expert,
        "mean_reversion": mean_reversion_expert,
        "event_shock": event_expert,
        "unstable": unstable_expert,
    }
    total = sum(expert_scores.values()) or 1.0
    normalized = {key: round(value / total, 4) for key, value in expert_scores.items()}

    preferred_playbook = max(normalized, key=normalized.get)
    regime_label = "unstable"
    if normalized["event_shock"] >= 0.34 or event_regime == "event_risk":
        regime_label = "event-risk"
    elif normalized["trend_continuation"] >= max(normalized["breakout_transition"], 0.31):
        regime_label = "trend"
    elif normalized["breakout_transition"] >= 0.29:
        regime_label = "transition"
    elif normalized["mean_reversion"] >= 0.30:
        regime_label = "range"

    return {
        "probabilities": normalized,
        "preferred_playbook": preferred_playbook,
        "regime_label": regime_label,
        "session_context": session_context,
    }


def _compute_direction_probabilities(
    score_diff,
    bull_trigger,
    bear_trigger,
    trend,
    alignment_score,
    breakout_bias,
    cross_asset_bias,
    regime_router,
    structure_context,
):
    router = (regime_router or {}).get("probabilities") or {}
    session = (regime_router or {}).get("session_context") or {}
    session_quality = _safe_float(session.get("quality"), 0.46)
    opening_range_break = _coerce_signal_state((structure_context or {}).get("openingRangeBreak"))
    sweep_reclaim_signal = _coerce_signal_state((structure_context or {}).get("sweepReclaimSignal"))
    sweep_reclaim_quality = _safe_float((structure_context or {}).get("sweepReclaimQuality"), 0.0)
    sweep_signal_strength = min(max(sweep_reclaim_quality, 0.0), 1.0) if sweep_reclaim_signal != 0 else 0.0

    cross_bias_score = _bias_strength(cross_asset_bias)
    breakout_bias_score = _bias_strength(breakout_bias)
    trend_score = _bias_strength(trend)
    setup_bias = _safe_float(score_diff) * 0.46
    trigger_bias = (_safe_float(bull_trigger) - _safe_float(bear_trigger)) * 0.78
    alignment_bias = (_safe_float(alignment_score) / 3.0) * 1.15
    range_bias = 0.18 * opening_range_break + ((0.10 + (0.12 * sweep_signal_strength)) * sweep_reclaim_signal)
    router_bias = (
        _safe_float(router.get("trend_continuation"), 0.0) * 0.35
        + _safe_float(router.get("breakout_transition"), 0.0) * 0.55
        - _safe_float(router.get("mean_reversion"), 0.0) * 0.20
        - _safe_float(router.get("event_shock"), 0.0) * 0.14
    )
    base_bias = setup_bias + trigger_bias + alignment_bias + (trend_score * 0.75) + (breakout_bias_score * 0.55) + (cross_bias_score * 0.45) + range_bias + router_bias

    up_15m = _sigmoid(base_bias + (bull_trigger - bear_trigger) * 0.25 + (session_quality - 0.5) * 0.55, center=0.0, scale=1.45)
    up_30m = _sigmoid(base_bias + (trend_score * 0.25) + (breakout_bias_score * 0.18), center=0.0, scale=1.65)
    up_60m = _sigmoid(base_bias + (trend_score * 0.42) + (_safe_float(router.get("trend_continuation"), 0.0) * 0.55), center=0.0, scale=1.85)

    return {
        "up_15m": round(_clamp(up_15m), 4),
        "up_30m": round(_clamp(up_30m), 4),
        "up_60m": round(_clamp(up_60m), 4),
        "down_15m": round(_clamp(1.0 - up_15m), 4),
        "down_30m": round(_clamp(1.0 - up_30m), 4),
        "down_60m": round(_clamp(1.0 - up_60m), 4),
    }


def _nearest_move_bucket_key(target_atr):
    buckets = {
        "0.5_atr": 0.5,
        "1.0_atr": 1.0,
        "1.5_atr": 1.5,
        "2.0_atr": 2.0,
    }
    return min(buckets, key=lambda key: abs(buckets[key] - float(target_atr or 0.0)))


def _compute_move_bucket_state(
    ta_data,
    regime_state,
    direction_probabilities,
    tradeability_score,
    stability_score,
    meta_scores,
    expected_edge_pct,
    strategy_params,
    regime_bucket,
):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    regime_state = regime_state if isinstance(regime_state, dict) else {}
    direction_probabilities = direction_probabilities if isinstance(direction_probabilities, dict) else {}
    meta_scores = meta_scores if isinstance(meta_scores, dict) else {}
    strategy_params = strategy_params if isinstance(strategy_params, dict) else {}

    calib = _load_confidence_calibration()
    empirical_buckets = (calib.get("move_bucket_hit_rates") or {}).get(str(regime_bucket or ""), {}) if isinstance(calib, dict) else {}
    volatility = ta_data.get("volatility_regime") or {}
    volatility_features = ta_data.get("volatility_features") or {}
    session_context = _build_session_context(ta_data)
    atr_percent = _safe_float(volatility.get("atr_percent"), 0.0)
    current_price = _safe_float(ta_data.get("current_price"), 0.0)
    pip_size = max(0.0001, _safe_float(strategy_params.get("rr_signal_pip_size"), 0.1))
    atr_move_pips = ((current_price * (atr_percent / 100.0)) / pip_size) if current_price > 0 and atr_percent > 0 else 0.0

    expansion_30m = _safe_float(regime_state.get("expansion_probability_30m"), 0.0) / 100.0
    expansion_60m = _safe_float(regime_state.get("expansion_probability_60m"), 0.0) / 100.0
    big_move_risk = _safe_float(regime_state.get("big_move_risk"), 0.0) / 100.0
    realized_vol_percentile = _safe_float(volatility_features.get("realizedVolPercentile"), 50.0) / 100.0
    atr_percentile = _safe_float(volatility_features.get("atrPercentile"), 50.0) / 100.0
    trend_follow_through = _safe_float(volatility_features.get("trendFollowThrough"), 1.0)
    session_quality = _safe_float(session_context.get("quality"), 0.46)
    direction_edge = max(
        _safe_float(direction_probabilities.get("up_60m"), 0.5),
        _safe_float(direction_probabilities.get("down_60m"), 0.5),
    )
    tradeability_quality = _clamp(_safe_float(tradeability_score) / 100.0)
    stability_quality = _clamp(_safe_float(stability_score) / 100.0)
    fakeout_penalty = _clamp(_safe_float(meta_scores.get("fakeout_probability"), 0.0))
    exit_penalty = _clamp(_safe_float(meta_scores.get("exit_risk_probability"), 0.0))
    edge_quality = _clamp((_safe_float(expected_edge_pct, 0.0) + 0.15) / 0.55)

    base_score = _clamp(
        (expansion_60m * 0.26)
        + (expansion_30m * 0.10)
        + (big_move_risk * 0.14)
        + (direction_edge * 0.16)
        + (tradeability_quality * 0.10)
        + (stability_quality * 0.08)
        + (session_quality * 0.06)
        + (realized_vol_percentile * 0.05)
        + (atr_percentile * 0.03)
        + (_clamp((trend_follow_through - 0.8) / 0.7) * 0.05)
        + (edge_quality * 0.07)
        - (fakeout_penalty * 0.08)
        - (exit_penalty * 0.06)
    )

    raw_probabilities = {
        "0.5_atr": _clamp(0.26 + (base_score * 0.68) - (fakeout_penalty * 0.08)),
        "1.0_atr": _clamp(0.14 + (base_score * 0.64) - (fakeout_penalty * 0.11) - (exit_penalty * 0.04)),
        "1.5_atr": _clamp(0.06 + (base_score * 0.56) - (fakeout_penalty * 0.14) - (exit_penalty * 0.05)),
        "2.0_atr": _clamp(0.02 + (base_score * 0.47) - (fakeout_penalty * 0.18) - (exit_penalty * 0.06)),
    }

    probabilities = dict(raw_probabilities)
    for key, model_probability in raw_probabilities.items():
        empirical_probability = empirical_buckets.get(key)
        if isinstance(empirical_probability, (int, float)):
            probabilities[key] = _clamp((model_probability * 0.68) + (float(empirical_probability) * 0.32))

    # Enforce monotonically decreasing survival probabilities.
    ordered_keys = ["0.5_atr", "1.0_atr", "1.5_atr", "2.0_atr"]
    previous = 1.0
    for key in ordered_keys:
        probabilities[key] = round(min(previous, _clamp(probabilities[key])), 4)
        previous = probabilities[key]

    projected_move_atr = round(
        0.18
        + (probabilities["0.5_atr"] * 0.40)
        + (probabilities["1.0_atr"] * 0.58)
        + (probabilities["1.5_atr"] * 0.72)
        + (probabilities["2.0_atr"] * 0.86),
        3,
    )
    projected_move_pips = round(projected_move_atr * atr_move_pips, 1) if atr_move_pips > 0 else 0.0
    target_move_pips = max(1.0, _safe_float(strategy_params.get("rr_signal_target_move_pips"), 200.0))
    target_atr = (target_move_pips / atr_move_pips) if atr_move_pips > 0 else 2.0
    probability_target_atr = max(0.5, min(2.0, target_atr))
    target_key = _nearest_move_bucket_key(probability_target_atr)
    target_probability = float(probabilities.get(target_key, 0.0))
    target_probability_is_proxy = bool(target_atr > 2.0 or target_atr < 0.5)

    return {
        "atr_move_pips": round(atr_move_pips, 2),
        "projected_move_atr": projected_move_atr,
        "projected_move_pips": projected_move_pips,
        "probabilities": probabilities,
        "selected_target_key": target_key,
        "selected_target_atr": round(probability_target_atr, 2),
        "target_move_atr_raw": round(target_atr, 2),
        "selected_probability": round(target_probability, 4),
        "selected_probability_is_proxy": target_probability_is_proxy,
    }


def _compute_execution_meta_label(
    direction_probabilities,
    move_bucket_state,
    tradeability_score,
    stability_score,
    meta_scores,
    expected_edge_pct,
    regime_router,
):
    direction_probabilities = direction_probabilities if isinstance(direction_probabilities, dict) else {}
    move_bucket_state = move_bucket_state if isinstance(move_bucket_state, dict) else {}
    meta_scores = meta_scores if isinstance(meta_scores, dict) else {}
    router = (regime_router or {}).get("probabilities") or {}
    session = (regime_router or {}).get("session_context") or {}

    direction_quality = max(
        _safe_float(direction_probabilities.get("up_30m"), 0.5),
        _safe_float(direction_probabilities.get("down_30m"), 0.5),
        _safe_float(direction_probabilities.get("up_60m"), 0.5),
        _safe_float(direction_probabilities.get("down_60m"), 0.5),
    )
    buckets = move_bucket_state.get("probabilities") or {}
    bucket_quality = (
        _safe_float(buckets.get("0.5_atr"), 0.0) * 0.22
        + _safe_float(buckets.get("1.0_atr"), 0.0) * 0.34
        + _safe_float(buckets.get("1.5_atr"), 0.0) * 0.27
        + _safe_float(buckets.get("2.0_atr"), 0.0) * 0.17
    )
    tradeability_quality = _clamp(_safe_float(tradeability_score) / 100.0)
    stability_quality = _clamp(_safe_float(stability_score) / 100.0)
    edge_quality = _clamp((_safe_float(expected_edge_pct, 0.0) + 0.15) / 0.55)
    fakeout_penalty = _clamp(_safe_float(meta_scores.get("fakeout_probability"), 0.0))
    exit_penalty = _clamp(_safe_float(meta_scores.get("exit_risk_probability"), 0.0))
    entry_timing_quality = _clamp(_safe_float(meta_scores.get("entry_timing_score"), 0.0) / 100.0)
    breakout_quality = _safe_float(router.get("breakout_transition"), 0.0)
    trend_quality = _safe_float(router.get("trend_continuation"), 0.0)
    event_penalty = _safe_float(router.get("event_shock"), 0.0) * 0.08
    session_quality = _safe_float(session.get("quality"), 0.46)

    entry_success_probability = _clamp(
        (direction_quality * 0.28)
        + (bucket_quality * 0.24)
        + (tradeability_quality * 0.15)
        + (stability_quality * 0.11)
        + (entry_timing_quality * 0.10)
        + (edge_quality * 0.06)
        + ((breakout_quality + trend_quality) * 0.03)
        + (session_quality * 0.03)
        - (fakeout_penalty * 0.12)
        - (exit_penalty * 0.08)
        - event_penalty
    )

    meta_label_probability = _clamp(
        (entry_success_probability * 0.72)
        + (_safe_float(buckets.get("1.0_atr"), 0.0) * 0.18)
        + (_safe_float(buckets.get("0.5_atr"), 0.0) * 0.10)
    )

    return {
        "entry_success_probability": round(entry_success_probability, 4),
        "meta_label_probability": round(meta_label_probability, 4),
    }


def _load_confidence_calibration():
    try:
        if not CONFIDENCE_CALIBRATION_FILE.exists():
            return {}
        data = json.loads(CONFIDENCE_CALIBRATION_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_regime_overrides():
    try:
        if not REGIME_PARAMS_FILE.exists():
            return {}
        data = json.loads(REGIME_PARAMS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _select_regime_profile(ta_data):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    vol_regime = str(((ta_data.get("volatility_regime") or {}).get("market_regime")) or "")
    event = ta_data.get("event_regime") or {}
    warning = str((event.get("warning_ladder")) or "Normal")
    event_regime = str((event.get("event_regime")) or "normal")
    if event_regime == "panic_reversal":
        return "post_shock_reversal"
    if warning in {"Directional Expansion Likely", "Active Momentum Event"} or event_regime in {"breakout_watch", "range_expansion", "trend_acceleration"}:
        return "event_breakout"
    if vol_regime == "Trending":
        return "trend_continuation"
    if vol_regime == "Range-Bound":
        return "quiet_range"
    return "transition"


def _apply_regime_overrides(base_params, ta_data):
    params = dict(base_params or {})
    profile = _select_regime_profile(ta_data)
    overrides = _load_regime_overrides()
    profile_overrides = {}
    if isinstance(overrides.get("profiles"), dict):
        profile_overrides = overrides["profiles"].get(profile) or {}
    if isinstance(profile_overrides, dict):
        params.update(profile_overrides)
    return params, profile


def _regime_bucket(regime_label, warning_ladder, event_regime):
    regime_label = str(regime_label or "")
    warning_ladder = str(warning_ladder or "")
    event_regime = str(event_regime or "")
    if event_regime == "panic_reversal":
        return "reversal_risk"
    if warning_ladder == "Active Momentum Event" or event_regime in {"trend_acceleration", "range_expansion"}:
        return "active_momentum"
    if warning_ladder in {"Directional Expansion Likely", "High Breakout Risk"} or event_regime == "breakout_watch":
        return "breakout_watch"
    if regime_label == "trend":
        return "trend_continuation"
    if regime_label == "range":
        return "quiet_range"
    return "transition"


def _bucket_from_components(regime_label, directional_bias, tradeability, stability):
    return "|".join(
        [
            str(regime_label or "unknown"),
            str(directional_bias or "Neutral"),
            str(tradeability or "Low"),
            "stable" if float(stability or 0) >= 60 else "unstable",
        ]
    )


def _calibrate_confidence(raw_confidence, regime_label, directional_bias, tradeability, stability):
    calibration = _load_confidence_calibration()
    regime_bucket = _regime_bucket(regime_label, "", "")
    bucket = _bucket_from_components(regime_label, directional_bias, tradeability, stability)
    empirical = None
    if isinstance(calibration.get("confidence_buckets"), dict):
        empirical = calibration["confidence_buckets"].get(bucket)
    if empirical is None:
        empirical = calibration.get(bucket)
    if isinstance(empirical, (int, float)):
        calibrated = max(50, min(95, round(float(empirical))))
        return calibrated, "empirical", bucket
    regime_floor = None
    if isinstance(calibration.get("regime_confidence"), dict):
        regime_floor = calibration["regime_confidence"].get(regime_bucket)
    if isinstance(regime_floor, (int, float)):
        calibrated = round((float(raw_confidence) * 0.55) + (float(regime_floor) * 0.45))
        return max(50, min(95, calibrated)), "regime_blend", bucket
    return int(raw_confidence), "heuristic", bucket


def _flip_direction_label(label):
    text = str(label or "Neutral")
    if text == "Bullish":
        return "Bearish"
    if text == "Bearish":
        return "Bullish"
    return "Neutral"


def _derive_tail_horizon_bias(breakout_bias, directional_bias, forecast_confidence):
    breakout_bias = str(breakout_bias or "Neutral")
    directional_bias = str(directional_bias or "Neutral")
    confidence_value = _safe_float(forecast_confidence, 50.0)
    if breakout_bias in {"Bullish", "Bearish"} and breakout_bias == directional_bias and confidence_value >= 54.0:
        return _flip_direction_label(breakout_bias)
    if breakout_bias in {"Bullish", "Bearish"} and breakout_bias != directional_bias:
        return breakout_bias
    return "Neutral"


def _build_forecast_state(regime_bucket, regime_state, confidence, directional_bias, direction_probabilities=None, move_bucket_state=None, regime_router=None):
    regime_state = regime_state or {}
    direction_probabilities = direction_probabilities if isinstance(direction_probabilities, dict) else {}
    move_bucket_state = move_bucket_state if isinstance(move_bucket_state, dict) else {}
    regime_router = regime_router if isinstance(regime_router, dict) else {}
    breakout_bias = str(regime_state.get("breakout_bias") or "Neutral")
    warning_ladder = str(regime_state.get("warning_ladder") or "Normal")
    event_regime = str(regime_state.get("event_regime") or "normal")
    expansion_30m = float(regime_state.get("expansion_probability_30m") or 0.0)
    expansion_60m = float(regime_state.get("expansion_probability_60m") or 0.0)
    directional_edge = max(
        _safe_float(direction_probabilities.get("up_30m"), 0.5),
        _safe_float(direction_probabilities.get("down_30m"), 0.5),
        _safe_float(direction_probabilities.get("up_60m"), 0.5),
        _safe_float(direction_probabilities.get("down_60m"), 0.5),
    )
    forecast_confidence = round(
        min(
            95.0,
            max(
                50.0,
                (float(confidence or 50) * 0.55)
                + (expansion_60m * 0.30)
                + (expansion_30m * 0.15)
                + (directional_edge * 12.0),
            ),
        )
    )
    tail_horizon_bias = _derive_tail_horizon_bias(breakout_bias, directional_bias, forecast_confidence)
    return {
        "regimeBucket": regime_bucket,
        "moveProbability30m": round(expansion_30m, 2),
        "moveProbability60m": round(expansion_60m, 2),
        "directionalBias": directional_bias,
        "breakoutBias": breakout_bias,
        "tailHorizonBias": tail_horizon_bias,
        "eventRegime": event_regime,
        "warningLadder": warning_ladder,
        "expectedRangeExpansion": regime_state.get("expected_range_expansion"),
        "crossAssetBias": regime_state.get("cross_asset_bias", "Neutral"),
        "minutesToNextEvent": regime_state.get("minutes_to_next_event"),
        "forecastConfidence": forecast_confidence,
        "directionProbabilities": direction_probabilities,
        "moveBuckets": move_bucket_state.get("probabilities", {}),
        "projectedMoveAtr": move_bucket_state.get("projected_move_atr"),
        "projectedMovePips": move_bucket_state.get("projected_move_pips"),
        "preferredPlaybook": regime_router.get("preferred_playbook"),
        "regimeRouter": regime_router.get("probabilities", {}),
    }


def _build_execution_state(action_state, action, tradeability_label, no_trade_reasons, anti_chop_reasons, confidence):
    actionable = action_state in {"LONG_ACTIVE", "SHORT_ACTIVE"}
    forming = action_state in {"SETUP_LONG", "SETUP_SHORT"}
    exit_recommended = action_state == "EXIT_RISK"
    if exit_recommended:
        status = "exit"
    elif actionable:
        status = "enter"
    elif forming:
        status = "prepare"
    else:
        status = "stand_aside"
    return {
        "status": status,
        "actionState": action_state,
        "action": action,
        "entryAllowed": actionable,
        "exitRecommended": exit_recommended,
        "tradeability": tradeability_label,
        "blockers": list(no_trade_reasons or []),
        "antiChopActive": bool(anti_chop_reasons),
        "antiChopReasons": list(anti_chop_reasons or []),
        "executionConfidence": round(float(confidence or 50)),
    }


def _compute_meta_model_scores(
    direction_score,
    trigger_score,
    tradeability_score,
    stability_score,
    fakeout_risk_score,
    exit_risk_score,
    warning_ladder,
    alignment_label,
):
    warning_weight = {
        "Normal": 0.0,
        "Expansion Watch": 0.12,
        "High Breakout Risk": 0.20,
        "Directional Expansion Likely": 0.26,
        "Active Momentum Event": 0.30,
    }.get(str(warning_ladder or "Normal"), 0.0)
    alignment_bonus = 0.08 if str(alignment_label).startswith("Strong") else 0.0
    entry_timing_score = max(
        0.0,
        min(
            100.0,
            (float(direction_score) * 5.2)
            + (float(trigger_score) * 8.5)
            + (float(tradeability_score) * 0.42)
            + (float(stability_score) * 0.28)
            + (warning_weight * 100.0)
            + (alignment_bonus * 100.0)
            - (float(fakeout_risk_score) * 8.0),
        ),
    )
    fakeout_probability = max(
        0.0,
        min(
            1.0,
            (float(fakeout_risk_score) / 6.5)
            + (0.12 if "Mixed" in str(alignment_label) else 0.0)
            + (0.08 if str(warning_ladder) in {"Expansion Watch", "High Breakout Risk"} else 0.0),
        ),
    )
    exit_risk_probability = max(
        0.0,
        min(1.0, (float(exit_risk_score) / 10.0) + (0.06 if "Mixed" in str(alignment_label) else 0.0)),
    )
    return {
        "entry_timing_score": round(entry_timing_score, 2),
        "fakeout_probability": round(fakeout_probability, 4),
        "exit_risk_probability": round(exit_risk_probability, 4),
    }


def _expected_value_edge_pct(confidence, strategy_params, regime_bucket):
    confidence = float(confidence or 50.0)
    calib = _load_confidence_calibration()
    bucket_stats = (calib.get("ev_buckets") or {}) if isinstance(calib, dict) else {}
    stats = bucket_stats.get(str(regime_bucket or "")) if isinstance(bucket_stats, dict) else None
    if stats is None and isinstance(bucket_stats, dict):
        band_floor = int(max(50, min(90, (int(confidence // 10) * 10))))
        band_key = f"{band_floor:02d}-{min(99, band_floor + 9):02d}"
        stats = bucket_stats.get(band_key)

    avg_win = float(strategy_params.get("estimated_avg_win_pct", 0.55))
    avg_loss = float(strategy_params.get("estimated_avg_loss_pct", 0.42))
    tx_cost = float(strategy_params.get("transaction_cost_pct", 0.05))
    if isinstance(stats, dict):
        avg_win = float(stats.get("avg_win_pct", avg_win) or avg_win)
        avg_loss = float(stats.get("avg_loss_pct", avg_loss) or avg_loss)

    p_win = max(0.0, min(1.0, confidence / 100.0))
    edge = (p_win * avg_win) - ((1.0 - p_win) * avg_loss) - tx_cost
    return round(edge, 4)


def _build_rr_signal_state(
    ta_data,
    strategy_params,
    verdict,
    directional_bias,
    directional_bias_source,
    direction_timeframe,
    regime_label,
    alignment_label,
    action_state,
    confidence,
    direction_score,
    tradeability_score,
    stability_score,
    expected_edge_pct,
    meta_scores,
    regime_state,
    no_trade_reasons,
    h1_neutral_override_active=False,
    direction_probabilities=None,
    move_bucket_state=None,
):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    regime_state = regime_state if isinstance(regime_state, dict) else {}
    meta_scores = meta_scores if isinstance(meta_scores, dict) else {}
    direction_probabilities = direction_probabilities if isinstance(direction_probabilities, dict) else {}
    move_bucket_state = move_bucket_state if isinstance(move_bucket_state, dict) else {}

    sl_pips = max(1.0, float(strategy_params.get("rr_signal_sl_pips", 100) or 100))
    tp_pips = max(1.0, float(strategy_params.get("rr_signal_tp_pips", 200) or 200))
    target_move_pips = max(1.0, float(strategy_params.get("rr_signal_target_move_pips", 200) or 200))
    pip_size = max(0.0001, float(strategy_params.get("rr_signal_pip_size", 0.1) or 0.1))
    min_confidence = float(strategy_params.get("rr_signal_min_confidence", 80.0) or 80.0)
    min_tradeability_score = float(strategy_params.get("rr_signal_min_tradeability_score", 68.0) or 68.0)
    min_move_probability = float(strategy_params.get("rr_signal_min_move_probability", 0.64) or 0.64)
    min_expected_edge = float(strategy_params.get("rr_signal_min_expected_edge_pct", 0.12) or 0.12)
    allow_b_grade = bool(int(strategy_params.get("rr_signal_allow_b_grade", 1) or 0))
    b_min_move_probability = float(strategy_params.get("rr_signal_b_min_move_probability", 0.64) or 0.64)
    b_min_session_quality = float(strategy_params.get("rr_signal_b_min_session_quality", 0.78) or 0.78)
    b_require_active_state = bool(int(strategy_params.get("rr_signal_b_require_active_state", 1) or 0))
    b_min_mtf_matches = int(round(float(strategy_params.get("rr_signal_b_min_mtf_matches", 3) or 3)))
    b_min_mtf_matches = max(2, min(3, b_min_mtf_matches))
    allow_soft_no_trade = bool(int(strategy_params.get("rr_signal_allow_soft_no_trade", 1) or 0))
    soft_no_trade_terms = strategy_params.get("rr_signal_soft_no_trade_terms", [])
    soft_no_trade_terms = soft_no_trade_terms if isinstance(soft_no_trade_terms, list) else []
    allowed_trade_hours = strategy_params.get("rr_signal_trade_hours_utc", [])
    allowed_trade_hours = [int(h) for h in allowed_trade_hours if isinstance(h, (int, float, str)) and str(h).strip().lstrip("-").isdigit()]
    allowed_trade_hours = sorted(set(h for h in allowed_trade_hours if 0 <= h <= 23))
    require_m15_trigger = bool(int(strategy_params.get("rr_signal_require_m15_trigger", 1) or 0))
    h1_neutral_override_enabled = bool(int(strategy_params.get("rr_signal_h1_neutral_override_enabled", 1) or 0))
    h1_neutral_min_expansion_60m = float(strategy_params.get("rr_signal_h1_neutral_min_expansion_60m", 66.0) or 66.0)
    h1_neutral_min_big_move_risk = float(strategy_params.get("rr_signal_h1_neutral_min_big_move_risk", 62.0) or 62.0)
    h1_neutral_min_direction_probability = float(strategy_params.get("rr_signal_h1_neutral_min_direction_probability", 0.62) or 0.62)
    h1_neutral_min_tradeability = float(strategy_params.get("rr_signal_h1_neutral_min_tradeability", 56.0) or 56.0)
    h1_neutral_min_confidence = float(strategy_params.get("rr_signal_h1_neutral_min_confidence", 66.0) or 66.0)
    h1_neutral_min_move_probability = float(strategy_params.get("rr_signal_h1_neutral_min_move_probability", 0.40) or 0.40)
    h1_neutral_require_h4_non_opposition = bool(int(strategy_params.get("rr_signal_h1_neutral_require_h4_non_opposition", 1) or 0))
    h1_neutral_allow_ready = bool(int(strategy_params.get("rr_signal_h1_neutral_allow_ready", 0) or 0))

    current_price = float(ta_data.get("current_price") or 0.0)
    atr_percent = float(((ta_data.get("volatility_regime") or {}).get("atr_percent")) or 0.0)
    mtf = ta_data.get("multi_timeframe", {}) if isinstance(ta_data.get("multi_timeframe"), dict) else {}
    m15_trend = str(mtf.get("m15_trend") or "Neutral")
    h1_trend = str(mtf.get("h1_trend") or "Neutral")
    h4_trend = str(mtf.get("h4_trend") or "Neutral")
    support_resistance = ta_data.get("support_resistance", {}) if isinstance(ta_data.get("support_resistance"), dict) else {}
    event_risk = ta_data.get("event_risk", {}) if isinstance(ta_data.get("event_risk"), dict) else {}
    price_action = ta_data.get("price_action", {}) if isinstance(ta_data.get("price_action"), dict) else {}
    pa_structure = str(price_action.get("structure") or "")
    pa_pattern = str(price_action.get("latest_candle_pattern") or "")
    expansion_30m = float(regime_state.get("expansion_probability_30m") or 0.0)
    expansion_60m = float(regime_state.get("expansion_probability_60m") or 0.0)
    big_move_risk = float(regime_state.get("big_move_risk") or 0.0)
    breakout_bias = str(regime_state.get("breakout_bias") or "Neutral")
    warning_ladder = str(regime_state.get("warning_ladder") or "Normal")
    bucket_probabilities = move_bucket_state.get("probabilities") if isinstance(move_bucket_state.get("probabilities"), dict) else {}
    selected_bucket_key = str(move_bucket_state.get("selected_target_key") or "2.0_atr")
    selected_target_atr = float(move_bucket_state.get("selected_target_atr") or 2.0)
    target_move_atr = float(move_bucket_state.get("target_move_atr_raw") or selected_target_atr)
    projected_move_atr = float(move_bucket_state.get("projected_move_atr") or 0.0)
    bucket_move_probability = float(move_bucket_state.get("selected_probability") or 0.0)
    bucket_probability_is_proxy = bool(move_bucket_state.get("selected_probability_is_proxy"))

    direction = "Neutral"
    if verdict in {"Bullish", "Bearish"}:
        direction = verdict
    elif directional_bias in {"Bullish", "Bearish"}:
        direction = directional_bias
    elif breakout_bias in {"Bullish", "Bearish"}:
        direction = breakout_bias

    mtf_votes = [m15_trend, h1_trend, h4_trend]
    mtf_directional_matches = sum(1 for value in mtf_votes if value == direction and direction in {"Bullish", "Bearish"})
    mtf_conflict_count = sum(1 for value in mtf_votes if value in {"Bullish", "Bearish"} and value != direction and direction in {"Bullish", "Bearish"})
    mtf_quality = mtf_directional_matches / 3.0 if direction in {"Bullish", "Bearish"} else 0.0
    h1_filter_pass = direction in {"Bullish", "Bearish"} and h1_trend == direction
    h4_filter_pass = direction in {"Bullish", "Bearish"} and h4_trend in {direction, "Neutral"}
    h1_neutral = direction in {"Bullish", "Bearish"} and h1_trend == "Neutral"
    h4_opposes_candidate = direction in {"Bullish", "Bearish"} and h4_trend in {"Bullish", "Bearish"} and h4_trend != direction

    session_context = _build_session_context(ta_data)
    now_hour_utc = int(session_context.get("hour") or datetime.now(timezone.utc).hour)
    session_allowed = (not allowed_trade_hours) or (now_hour_utc in allowed_trade_hours)
    session_quality = float(session_context.get("quality") or 0.55)

    trend_quality = max(0.0, min(1.0, (float(direction_score) - 5.0) / 5.0))
    confidence_quality = max(0.0, min(1.0, (float(confidence) - 60.0) / 35.0))
    tradeability_quality = max(0.0, min(1.0, float(tradeability_score) / 100.0))
    stability_quality = max(0.0, min(1.0, float(stability_score) / 100.0))
    alignment_quality = 1.0 if str(alignment_label).startswith("Strong") else 0.45
    expansion_quality = max(0.0, min(1.0, (expansion_60m * 0.7 + expansion_30m * 0.3) / 100.0))
    event_quality = max(0.0, min(1.0, big_move_risk / 100.0))
    directional_quality = max(
        float(direction_probabilities.get("up_60m") or 0.5),
        float(direction_probabilities.get("down_60m") or 0.5),
    )
    bucket_quality = max(
        float(bucket_probabilities.get("1.0_atr") or 0.0),
        float(bucket_probabilities.get("1.5_atr") or 0.0),
        float(bucket_probabilities.get("2.0_atr") or 0.0),
    )
    directional_probability = max(
        float(direction_probabilities.get("up_60m") or 0.5) if direction == "Bullish" else 0.0,
        float(direction_probabilities.get("down_60m") or 0.5) if direction == "Bearish" else 0.0,
    )
    fakeout_penalty = max(0.0, min(1.0, float(meta_scores.get("fakeout_probability") or 0.0)))
    exit_penalty = max(0.0, min(1.0, float(meta_scores.get("exit_risk_probability") or 0.0)))

    quant_score = (
        trend_quality * 22.0
        + confidence_quality * 22.0
        + tradeability_quality * 18.0
        + stability_quality * 10.0
        + expansion_quality * 12.0
        + event_quality * 8.0
        + alignment_quality * 5.0
        + (mtf_quality * 8.0)
        + (session_quality * 3.0)
        + (directional_quality * 6.0)
        + (bucket_quality * 6.0)
    )
    quant_score -= (fakeout_penalty * 12.0) + (exit_penalty * 8.0)
    if mtf_conflict_count >= 2:
        quant_score -= 10.0
    elif mtf_conflict_count == 1:
        quant_score -= 4.0
    if event_risk.get("active"):
        quant_score -= 6.0
    quant_score = max(0.0, min(100.0, quant_score))

    move_probability = (
        confidence_quality * 0.28
        + trend_quality * 0.19
        + expansion_quality * 0.24
        + tradeability_quality * 0.14
        + alignment_quality * 0.06
        + (mtf_quality * 0.08)
        + (session_quality * 0.03)
        + event_quality * 0.09
        + (directional_quality * 0.08)
    )
    move_probability -= (fakeout_penalty * 0.18) + (exit_penalty * 0.10)
    if mtf_conflict_count >= 2:
        move_probability -= 0.12
    elif mtf_conflict_count == 1:
        move_probability -= 0.05
    move_probability = max(0.0, min(1.0, move_probability))

    atr_move_pips = float(move_bucket_state.get("atr_move_pips") or 0.0)
    if atr_move_pips <= 0 and current_price > 0 and atr_percent > 0:
        atr_move_pips = ((current_price * (atr_percent / 100.0)) / pip_size)
    projected_move_pips = float(move_bucket_state.get("projected_move_pips") or 0.0)
    if projected_move_pips <= 0:
        projected_move_pips = atr_move_pips * (0.75 + (expansion_60m / 100.0) + (trend_quality * 0.55))
    projected_move_pips = max(0.0, projected_move_pips)
    if bucket_move_probability > 0:
        move_probability = max(move_probability, bucket_move_probability)

    momentum_override_qualified = (
        h1_neutral_override_enabled
        and h1_neutral
        and direction in {"Bullish", "Bearish"}
        and m15_trend == direction
        and float(confidence) >= h1_neutral_min_confidence
        and float(tradeability_score) >= h1_neutral_min_tradeability
        and move_probability >= h1_neutral_min_move_probability
        and directional_probability >= h1_neutral_min_direction_probability
        and expansion_60m >= h1_neutral_min_expansion_60m
        and big_move_risk >= h1_neutral_min_big_move_risk
        and (not h1_neutral_require_h4_non_opposition or not h4_opposes_candidate)
    )
    strong_directional_stack = (
        direction in {"Bullish", "Bearish"}
        and directional_probability >= max(min_move_probability, 0.60)
        and float(tradeability_score) >= max(min_tradeability_score - 6.0, 50.0)
        and (mtf_directional_matches >= 2 or momentum_override_qualified)
    )
    effective_min_confidence = min_confidence
    effective_min_expected_edge = min_expected_edge
    if strong_directional_stack:
        # Allow earlier arming when direction quality is strong and stack agreement is clean.
        effective_min_confidence = min(min_confidence, 60.0)
        effective_min_expected_edge = min(min_expected_edge, 0.008)

    tier = "watch"
    grade = "Watchlist"
    if quant_score >= 84 and move_probability >= 0.68:
        tier = "a_plus"
        grade = "A+ (Quant)"
    elif quant_score >= 74 and move_probability >= 0.56:
        tier = "a"
        grade = "A (High Accuracy)"
    elif quant_score >= 66 and move_probability >= 0.48:
        tier = "b"
        grade = "B (Qualified)"
    else:
        tier = "c"
        grade = "C (Low Confidence)"

    blockers = []
    if direction not in {"Bullish", "Bearish"}:
        blockers.append("No directional bias")
    if float(confidence) < effective_min_confidence:
        blockers.append(f"Confidence below {int(effective_min_confidence)}%")
    if float(tradeability_score) < min_tradeability_score:
        blockers.append(f"Tradeability below {round(min_tradeability_score, 1)}")
    if move_probability < min_move_probability:
        blockers.append(f"RR move probability below {round(min_move_probability * 100)}%")
    if projected_move_pips < target_move_pips:
        blockers.append(f"Projected move below fixed {int(target_move_pips)} pip target")
    if float(expected_edge_pct) < effective_min_expected_edge:
        blockers.append(f"Expected edge below {round(effective_min_expected_edge, 3)}")
    if action_state not in {"LONG_ACTIVE", "SHORT_ACTIVE", "SETUP_LONG", "SETUP_SHORT"} and not momentum_override_qualified:
        blockers.append("Execution state not directional")
    if mtf_directional_matches < 2 and not momentum_override_qualified:
        blockers.append("Insufficient multi-timeframe directional agreement")
    if direction in {"Bullish", "Bearish"} and not h1_filter_pass and not momentum_override_qualified:
        blockers.append("H1 directional filter is not aligned")
    if direction in {"Bullish", "Bearish"} and h4_trend in {"Bullish", "Bearish"} and not h4_filter_pass:
        blockers.append("H4 trend opposes H1 direction")
    if direction in {"Bullish", "Bearish"} and m15_trend in {"Bullish", "Bearish"} and m15_trend != direction:
        blockers.append("M15 trend disagrees with H1 direction")
    if no_trade_reasons:
        reason = str(no_trade_reasons[0])
        soft_match = any(term in reason for term in soft_no_trade_terms)
        if not (allow_soft_no_trade and soft_match and tier in {"a_plus", "a"}):
            blockers.append(reason)

    if event_risk.get("active") and tier in {"b", "c"}:
        blockers.append("Event-risk window is active")
    if not session_allowed:
        blockers.append("Outside preferred London/NY session window")
    m15_trigger_ok = (
        not require_m15_trigger
        or (
            (direction == "Bullish" and m15_trend == "Bullish" and ("Bullish" in pa_structure or "Bullish" in pa_pattern))
            or (direction == "Bearish" and m15_trend == "Bearish" and ("Bearish" in pa_structure or "Bearish" in pa_pattern))
        )
    )
    if not m15_trigger_ok:
        blockers.append("M15 trigger confirmation is missing")
    if tier == "b":
        if mtf_directional_matches < b_min_mtf_matches:
            if b_min_mtf_matches >= 3:
                blockers.append("B-grade requires full multi-timeframe alignment")
            else:
                blockers.append("B-grade requires at least two aligned timeframes")
        if move_probability < max(min_move_probability, b_min_move_probability):
            blockers.append("B-grade move probability is too weak")
        if session_quality < b_min_session_quality:
            blockers.append("B-grade is disabled in low-liquidity session")
        if b_require_active_state and action_state not in {"LONG_ACTIVE", "SHORT_ACTIVE"}:
            blockers.append("B-grade requires active directional state")

    allowed_grades = {"A+ (Quant)", "A (High Accuracy)"}
    if allow_b_grade:
        allowed_grades.add("B (Qualified)")
    send_signal = not blockers and grade in allowed_grades and (h1_neutral_allow_ready or not h1_neutral)
    status = "ready" if send_signal else ("arming" if (grade.startswith("A") or momentum_override_qualified) and direction in {"Bullish", "Bearish"} else "standby")
    display_direction = direction if status in {"arming", "ready"} else "Neutral"

    trigger = "breakout_continuation"
    sr_reaction = str(support_resistance.get("reaction") or "None")
    if "Rejection" in sr_reaction:
        trigger = "pullback_rejection"
    elif "Consolidating" in str(ta_data.get("price_action", {}).get("structure", "")):
        trigger = "range_break_pending"
    reentry_eligible = (
        direction in {"Bullish", "Bearish"}
        and tier in {"a_plus", "a", "b"}
        and (
            ("Support Rejection" in sr_reaction and direction == "Bullish")
            or ("Resistance Rejection" in sr_reaction and direction == "Bearish")
            or mtf_directional_matches >= 2
        )
    )

    sl_distance = sl_pips * pip_size
    tp_distance = tp_pips * pip_size
    entry_price = round(current_price, 2) if current_price > 0 and display_direction in {"Bullish", "Bearish"} else None
    stop_loss = None
    take_profit = None
    if current_price > 0 and display_direction in {"Bullish", "Bearish"}:
        if display_direction == "Bullish":
            stop_loss = round(current_price - sl_distance, 2)
            take_profit = round(current_price + tp_distance, 2)
        else:
            stop_loss = round(current_price + sl_distance, 2)
            take_profit = round(current_price - tp_distance, 2)

    status_text = "Stand by. Current RR direction is not actionable yet."
    if status == "arming":
        if momentum_override_qualified and h1_neutral:
            status_text = "15M momentum acceleration detected while H1 is neutral; waiting for H1 trend confirmation."
        else:
            status_text = "Directional setup detected; waiting for full fixed-target confirmation."
    elif status == "ready":
        status_text = "High-accuracy RR 1:2 signal is ready for alerting."
    if status != "ready" and blockers:
        blocker_text = " and ".join(str(item) for item in blockers[:2] if str(item).strip())
        if blocker_text and blocker_text.lower() not in status_text.lower():
            status_text = f"{status_text} Blocked by {blocker_text}."

    partial_tp_pips = float(strategy_params.get("rr_signal_partial_take_profit_pips", sl_pips) or sl_pips)
    move_sl_to_be = bool(int(strategy_params.get("rr_signal_move_sl_to_be_after_partial", 1) or 0))

    return {
        "enabled": send_signal,
        "status": status,
        "statusText": status_text,
        "tier": tier,
        "grade": grade,
        "quantScore": round(quant_score, 2),
        "direction": display_direction,
        "candidateDirection": direction,
        "directionSource": str(directional_bias_source or "h1"),
        "directionTimeframe": str(direction_timeframe or "1h"),
        "entryTriggerTimeframe": "15m",
        "mtfAgreement": mtf_directional_matches,
        "mtfConflict": mtf_conflict_count,
        "h1DirectionalFilterPass": h1_filter_pass,
        "h1Neutral": h1_neutral,
        "h1NeutralOverrideActive": bool(h1_neutral_override_active),
        "h1Trend": h1_trend,
        "h4DirectionalFilterPass": h4_filter_pass,
        "h4Trend": h4_trend,
        "m15Trend": m15_trend,
        "momentumOverrideQualified": momentum_override_qualified,
        "targetMovePips": round(target_move_pips, 1),
        "targetMoveAtr": round(target_move_atr, 2),
        "targetBucket": selected_bucket_key,
        "targetBucketAtr": round(selected_target_atr, 2),
        "targetBucketProbability": round(bucket_move_probability, 4),
        "targetBucketIsProxy": bucket_probability_is_proxy,
        "projectedMovePips": round(projected_move_pips, 1),
        "projectedMoveAtr": round(projected_move_atr, 2),
        "moveProbability": round(move_probability, 4),
        "moveBuckets": bucket_probabilities,
        "directionProbabilities": direction_probabilities,
        "rrRatio": round(tp_pips / max(sl_pips, 1.0), 2),
        "slPips": round(sl_pips, 1),
        "tpPips": round(tp_pips, 1),
        "partialTakeProfitPips": round(partial_tp_pips, 1),
        "moveStopToBreakevenAfterPartial": move_sl_to_be,
        "entryPrice": entry_price,
        "stopLossPrice": stop_loss,
        "takeProfitPrice": take_profit,
        "pipSize": pip_size,
        "expectedEdgePct": round(float(expected_edge_pct), 4),
        "warningLadder": warning_ladder,
        "regime": regime_label,
        "entryTrigger": trigger,
        "reentryEligible": reentry_eligible,
        "sessionAllowed": session_allowed,
        "allowedTradeHoursUtc": allowed_trade_hours,
        "m15TriggerConfirmed": m15_trigger_ok,
        "blockers": blockers[:4],
    }


def _warning_rank(value):
    ranks = {
        "Normal": 0,
        "Expansion Watch": 1,
        "High Breakout Risk": 2,
        "Directional Expansion Likely": 3,
        "Active Momentum Event": 4,
    }
    return ranks.get(str(value or "Normal"), 0)


def _warning_from_rank(rank):
    mapping = {
        0: "Normal",
        1: "Expansion Watch",
        2: "High Breakout Risk",
        3: "Directional Expansion Likely",
        4: "Active Momentum Event",
    }
    return mapping.get(int(rank), "Normal")


def _stabilize_runtime_regime_state(regime_state, memory, strategy_params):
    regime_state = dict(regime_state or {})
    memory = dict(memory or {})

    raw_warning = str(regime_state.get("warning_ladder") or "Normal")
    raw_event_regime = str(regime_state.get("event_regime") or "normal")
    raw_breakout_bias = str(regime_state.get("breakout_bias") or "Neutral")
    big_move_risk = float(regime_state.get("big_move_risk") or 0.0)
    expansion_60m = float(regime_state.get("expansion_probability_60m") or 0.0)
    bias_delta = float(((regime_state.get("components") or {}).get("bias_delta")) or 0.0)

    watch_threshold = float(strategy_params.get("expansion_watch_threshold", 48.0))
    high_threshold = float(strategy_params.get("high_breakout_threshold", 64.0))
    directional_threshold = float(strategy_params.get("directional_expansion_threshold", 78.0))
    up_buffer = float(strategy_params.get("warning_upshift_buffer", 3.5))
    down_buffer = float(strategy_params.get("warning_downshift_buffer", 7.0))
    min_dwell = max(1, int(strategy_params.get("warning_min_dwell_bars", 5) or 5))
    raw_warning_streak_required = max(1, int(strategy_params.get("warning_raw_streak_required", 2) or 2))
    bias_deadband = float(strategy_params.get("breakout_bias_deadband", 0.9))
    bias_hold = max(1, int(strategy_params.get("breakout_bias_hold_bars", 4) or 4))

    prev_warning = str(memory.get("warning_ladder") or raw_warning)
    prev_event_regime = str(memory.get("event_regime") or raw_event_regime)
    prev_bias = str(memory.get("breakout_bias") or raw_breakout_bias)
    prev_raw_warning = str(memory.get("raw_warning_ladder") or raw_warning)
    raw_warning_streak = max(0, int(memory.get("raw_warning_streak", 0) or 0))
    warning_dwell = max(0, int(memory.get("warning_dwell_bars", 0) or 0))
    bias_dwell = max(0, int(memory.get("breakout_bias_dwell_bars", 0) or 0))
    raw_warning_streak = (raw_warning_streak + 1) if raw_warning == prev_raw_warning else 1

    raw_rank = _warning_rank(raw_warning)
    stable_rank = _warning_rank(prev_warning)
    threshold_by_rank = {
        1: watch_threshold,
        2: high_threshold,
        3: directional_threshold,
        4: directional_threshold + 8.0,
    }

    target_rank = raw_rank
    if raw_rank != stable_rank and raw_warning_streak < raw_warning_streak_required:
        target_rank = stable_rank
    elif raw_rank > stable_rank:
        required = float(threshold_by_rank.get(raw_rank, watch_threshold))
        if expansion_60m < (required + up_buffer) and big_move_risk < (required + up_buffer):
            target_rank = stable_rank
    elif raw_rank < stable_rank:
        required = float(threshold_by_rank.get(stable_rank, watch_threshold))
        can_downgrade = warning_dwell >= min_dwell
        if expansion_60m > (required - down_buffer) or big_move_risk > (required - down_buffer):
            can_downgrade = False
        if not can_downgrade:
            target_rank = stable_rank

    stable_warning = _warning_from_rank(target_rank)
    warning_dwell = (warning_dwell + 1) if stable_warning == prev_warning else 1

    stable_bias = raw_breakout_bias
    if prev_bias in {"Bullish", "Bearish"} and raw_breakout_bias != prev_bias:
        if abs(bias_delta) < max(0.1, bias_deadband) or bias_dwell < bias_hold:
            stable_bias = prev_bias
    bias_dwell = (bias_dwell + 1) if stable_bias == prev_bias else 1

    if raw_event_regime == "event_risk":
        stable_event = "event_risk"
    elif stable_warning == "Normal":
        stable_event = "normal"
    elif stable_warning == "Expansion Watch":
        stable_event = "breakout_watch" if stable_bias != "Neutral" else "normal"
    elif stable_warning in {"High Breakout Risk", "Directional Expansion Likely"}:
        stable_event = "breakout_watch"
    elif stable_warning == "Active Momentum Event":
        if raw_event_regime in {"panic_reversal", "trend_acceleration", "range_expansion"}:
            stable_event = raw_event_regime
        else:
            stable_event = "range_expansion" if stable_bias != "Neutral" else "normal"
    else:
        stable_event = "normal"

    regime_state["raw_warning_ladder"] = raw_warning
    regime_state["raw_event_regime"] = raw_event_regime
    regime_state["raw_breakout_bias"] = raw_breakout_bias
    regime_state["warning_ladder"] = stable_warning
    regime_state["event_regime"] = stable_event
    regime_state["breakout_bias"] = stable_bias
    regime_state["warning_dwell_bars"] = int(warning_dwell)
    regime_state["breakout_bias_dwell_bars"] = int(bias_dwell)

    next_memory = {
        "warning_ladder": stable_warning,
        "event_regime": stable_event,
        "breakout_bias": stable_bias,
        "raw_warning_ladder": raw_warning,
        "raw_warning_streak": int(raw_warning_streak),
        "warning_dwell_bars": int(warning_dwell),
        "breakout_bias_dwell_bars": int(bias_dwell),
    }
    return regime_state, next_memory


def _bias_vote(*biases):
    bullish = sum(1 for bias in biases if bias == "Bullish")
    bearish = sum(1 for bias in biases if bias == "Bearish")
    if bullish > bearish:
        return "Bullish", bullish, bearish
    if bearish > bullish:
        return "Bearish", bullish, bearish
    return "Neutral", bullish, bearish


def _price_action_bias(structure, pattern):
    structure = str(structure or "")
    pattern = str(pattern or "")
    bullish_markers = ("Bullish", "Hammer")
    bearish_markers = ("Bearish", "Shooting Star")
    bullish_score = sum(1 for marker in bullish_markers if marker in structure or marker in pattern)
    bearish_score = sum(1 for marker in bearish_markers if marker in structure or marker in pattern)
    if bullish_score > bearish_score:
        return "Bullish"
    if bearish_score > bullish_score:
        return "Bearish"
    return "Neutral"


def _nearby_level_has_family(levels, family):
    for level in levels:
        if isinstance(level, dict) and str(level.get("family") or "") == str(family or ""):
            return True
    return False


def _nearby_level_special_family_count(levels):
    families = {
        str(level.get("family") or "")
        for level in levels
        if isinstance(level, dict)
    }
    return len(families & {"pivot", "round", "fvg", "range"})


def _direction_probability_for_label(direction, bullish_prob_30, bearish_prob_30, bullish_prob_60, bearish_prob_60):
    if direction == "Bullish":
        return _safe_float(bullish_prob_30, 0.5), _safe_float(bullish_prob_60, 0.5)
    if direction == "Bearish":
        return _safe_float(bearish_prob_30, 0.5), _safe_float(bearish_prob_60, 0.5)
    return 0.5, 0.5


def _resolve_directional_bias(
    *,
    h1_trend,
    h4_trend,
    m15_trend,
    base_trend,
    probabilistic_direction,
    event_breakout_bias,
    price_action_bias,
    confirmed_candle_bias,
    opening_range_break,
    sweep_reclaim_signal,
    score_diff,
    bullish_prob_30,
    bearish_prob_30,
    bullish_prob_60,
    bearish_prob_60,
    warning_ladder,
    expansion_probability_60m,
    big_move_risk,
    direction_probability_floor,
    verdict_margin_threshold,
):
    h1_direction = h1_trend if h1_trend in {"Bullish", "Bearish"} else "Neutral"
    h4_direction = h4_trend if h4_trend in {"Bullish", "Bearish"} else "Neutral"
    candidate_direction = (
        probabilistic_direction if probabilistic_direction in {"Bullish", "Bearish"} else "Neutral"
    )

    directional_bias = h1_direction
    directional_bias_source = "h1"
    direction_timeframe = "1h"
    h1_neutral_override_active = False

    if h1_direction == "Neutral" and candidate_direction in {"Bullish", "Bearish"}:
        candidate_prob_30, candidate_prob_60 = _direction_probability_for_label(
            candidate_direction,
            bullish_prob_30,
            bearish_prob_30,
            bullish_prob_60,
            bearish_prob_60,
        )
        breakout_support = 0
        breakout_conflicts = 0

        if m15_trend == candidate_direction:
            breakout_support += 1
        if base_trend == candidate_direction:
            breakout_support += 1
        if price_action_bias == candidate_direction:
            breakout_support += 1
        if event_breakout_bias == candidate_direction:
            breakout_support += 1
        if confirmed_candle_bias == candidate_direction:
            breakout_support += 1
        if (opening_range_break > 0 and candidate_direction == "Bullish") or (
            opening_range_break < 0 and candidate_direction == "Bearish"
        ):
            breakout_support += 1
        if (sweep_reclaim_signal > 0 and candidate_direction == "Bullish") or (
            sweep_reclaim_signal < 0 and candidate_direction == "Bearish"
        ):
            breakout_support += 1
        if (candidate_direction == "Bullish" and score_diff >= verdict_margin_threshold * 0.75) or (
            candidate_direction == "Bearish" and score_diff <= -(verdict_margin_threshold * 0.75)
        ):
            breakout_support += 1
        if warning_ladder in {"High Breakout Risk", "Directional Expansion Likely", "Active Momentum Event"}:
            breakout_support += 1
        if float(expansion_probability_60m) >= 58.0 or float(big_move_risk) >= 58.0:
            breakout_support += 1

        if event_breakout_bias in {"Bullish", "Bearish"} and event_breakout_bias != candidate_direction:
            breakout_conflicts += 1
        if price_action_bias in {"Bullish", "Bearish"} and price_action_bias != candidate_direction:
            breakout_conflicts += 1
        if confirmed_candle_bias in {"Bullish", "Bearish"} and confirmed_candle_bias != candidate_direction:
            breakout_conflicts += 1

        h4_opposes_candidate = h4_direction in {"Bullish", "Bearish"} and h4_direction != candidate_direction
        breakout_override_qualified = (
            not h4_opposes_candidate
            and candidate_prob_60 >= max(float(direction_probability_floor) + 0.02, 0.60)
            and candidate_prob_30 >= 0.54
            and breakout_support >= 4
            and breakout_conflicts == 0
            and (m15_trend == candidate_direction or base_trend == candidate_direction)
        )

        if breakout_override_qualified:
            directional_bias = candidate_direction
            directional_bias_source = "breakout_stack"
            direction_timeframe = "60m"
            h1_neutral_override_active = True

    h4_opposes_direction = (
        directional_bias in {"Bullish", "Bearish"}
        and h4_direction in {"Bullish", "Bearish"}
        and h4_direction != directional_bias
    )
    h4_filter_pass = not h4_opposes_direction
    m15_trigger_aligned = directional_bias in {"Bullish", "Bearish"} and m15_trend == directional_bias
    m15_trigger_conflict = (
        directional_bias in {"Bullish", "Bearish"}
        and m15_trend in {"Bullish", "Bearish"}
        and m15_trend != directional_bias
    )

    return {
        "directional_bias": directional_bias,
        "directional_bias_source": directional_bias_source,
        "direction_timeframe": direction_timeframe,
        "h1_direction": h1_direction,
        "h4_direction": h4_direction,
        "h1_neutral_override_active": h1_neutral_override_active,
        "h4_opposes_direction": h4_opposes_direction,
        "h4_filter_pass": h4_filter_pass,
        "m15_trigger_aligned": m15_trigger_aligned,
        "m15_trigger_conflict": m15_trigger_conflict,
    }


def compute_trade_guidance(ta_data, confidence):
    if not isinstance(ta_data, dict):
        return {
            "sellLevel": "Weak",
            "buyLevel": "Weak",
            "exitLevel": "Low",
            "summary": "Trade guidance unavailable.",
        }

    trend = ta_data.get("ema_trend", "Neutral")
    regime = (ta_data.get("volatility_regime") or {}).get("market_regime", "Range-Bound")
    alignment = (ta_data.get("multi_timeframe") or {}).get("alignment_label", "Mixed / Low Alignment")
    structure = (ta_data.get("price_action") or {}).get("structure", "Consolidating")
    pattern = (ta_data.get("price_action") or {}).get("latest_candle_pattern", "None")

    sell_score = 0
    buy_score = 0
    exit_score = 0

    if trend == "Bearish":
        sell_score += 2
    elif trend == "Bullish":
        buy_score += 2

    if "Bearish" in alignment:
        sell_score += 2
    elif "Bullish" in alignment:
        buy_score += 2

    if regime == "Trending":
        sell_score += 1 if trend == "Bearish" else 0
        buy_score += 1 if trend == "Bullish" else 0

    if "Bearish" in structure:
        sell_score += 2
    elif "Bullish" in structure:
        buy_score += 2

    if "Bearish" in pattern:
        sell_score += 1
    if "Bullish" in pattern:
        buy_score += 1

    if confidence >= 75:
        if sell_score > buy_score:
            sell_score += 1
        elif buy_score > sell_score:
            buy_score += 1

    if trend == "Bearish" and ("Bullish" in structure or "Bullish" in pattern):
        exit_score += 2
    if trend == "Bullish" and ("Bearish" in structure or "Bearish" in pattern):
        exit_score += 2
    if "Mixed" in alignment:
        exit_score += 1
    if confidence < 65:
        exit_score += 1

    def level(score):
        if score >= 6:
            return "Strong"
        if score >= 4:
            return "Watch"
        return "Weak"

    if exit_score >= 3:
        exit_level = "High"
    elif exit_score >= 2:
        exit_level = "Medium"
    else:
        exit_level = "Low"

    sell_level = level(sell_score)
    buy_level = level(buy_score)
    summary = "Wait for cleaner confirmation."
    if sell_score >= 6 and sell_score > buy_score + 1:
        summary = "Sell bias is favored; wait for bearish continuation or rejection confirmation."
    elif buy_score >= 6 and buy_score > sell_score + 1:
        summary = "Buy bias is favored; wait for bullish continuation or breakout confirmation."
    elif exit_score >= 3:
        summary = "Existing position should be monitored closely for exit or risk reduction."

    return {
        "sellLevel": sell_level,
        "buyLevel": buy_level,
        "exitLevel": exit_level,
        "summary": summary,
    }


def compute_prediction_from_ta(ta_data):
    if not isinstance(ta_data, dict):
        return {
            "verdict": "Neutral",
            "confidence": 50,
            "TradeGuidance": compute_trade_guidance({}, 50),
        }

    strategy_params = normalize_strategy_params(ta_data.get("active_strategy_params"))
    strategy_params, active_regime_profile = _apply_regime_overrides(strategy_params, ta_data)

    verdict = "Neutral"
    mtf = ta_data.get("multi_timeframe", {})
    m15_trend = str(mtf.get("m15_trend") or ta_data.get("execution_trend") or "Neutral")
    h1_trend = str(mtf.get("h1_trend") or ta_data.get("ema_trend") or "Neutral")
    h4_trend = str(mtf.get("h4_trend") or "Neutral")
    trend = h1_trend if h1_trend in {"Bullish", "Bearish"} else str(ta_data.get("ema_trend") or "Neutral")
    volume = (ta_data.get("volume_analysis") or {}).get("overall_volume_signal", "Neutral")
    regime = (ta_data.get("volatility_regime") or {}).get("market_regime", "Range-Bound")
    adx_14 = (ta_data.get("volatility_regime") or {}).get("adx_14", 0)
    alignment_score = mtf.get("alignment_score", 0)
    alignment_label = mtf.get("alignment_label", "Mixed / Low Alignment")
    pa_struct = (ta_data.get("price_action") or {}).get("structure", "Consolidating")
    pa_pattern = (ta_data.get("price_action") or {}).get("latest_candle_pattern", "None")
    momentum_features = ta_data.get("momentum_features", {}) if isinstance(ta_data.get("momentum_features"), dict) else {}
    macd_hist = _safe_float(momentum_features.get("macdHistogram"), 0.0)
    macd_hist_slope = _safe_float(momentum_features.get("macdHistogramSlope"), 0.0)
    rsi_bullish_divergence = bool(int(_safe_float(momentum_features.get("rsiBullishDivergence"), 0.0)))
    rsi_bearish_divergence = bool(int(_safe_float(momentum_features.get("rsiBearishDivergence"), 0.0)))
    rsi_divergence_strength = _safe_float(momentum_features.get("rsiDivergenceStrength"), 0.0)
    volume_zscore = _safe_float(momentum_features.get("volumeZScore"), 0.0)
    volume_spike = bool(int(_safe_float(momentum_features.get("volumeSpike"), 0.0)))
    feature_hits = extract_price_action_feature_hits(pa_struct, pa_pattern)
    structure_context = ta_data.get("structure_context", {}) if isinstance(ta_data.get("structure_context"), dict) else {}
    market_regime_scores = ta_data.get("market_regime_scores", {}) if isinstance(ta_data.get("market_regime_scores"), dict) else {}
    regime_state = ta_data.get("event_regime", {}) if isinstance(ta_data.get("event_regime"), dict) else {}
    regime_memory = ta_data.get("_regime_memory", {}) if isinstance(ta_data.get("_regime_memory"), dict) else {}
    regime_state, next_regime_memory = _stabilize_runtime_regime_state(
        regime_state=regime_state,
        memory=regime_memory,
        strategy_params=strategy_params,
    )
    support_resistance = ta_data.get("support_resistance", {}) if isinstance(ta_data.get("support_resistance"), dict) else {}
    event_risk = ta_data.get("event_risk", {}) if isinstance(ta_data.get("event_risk"), dict) else {}
    trend_base_weight = float(strategy_params.get("trend_base_weight", 2.5))
    alignment_weight = float(strategy_params.get("alignment_weight", 1.2))
    trend_regime_bonus = float(strategy_params.get("trend_regime_bonus", 1.0))
    weak_trend_bonus = float(strategy_params.get("weak_trend_bonus", 0.4))
    strong_volume_weight = float(strategy_params.get("strong_volume_weight", 2.0))
    bias_volume_weight = float(strategy_params.get("bias_volume_weight", 1.0))
    breakout_weight = float(strategy_params.get("breakout_weight", 2.0))
    structure_weight = float(strategy_params.get("structure_weight", 1.0))
    swing_structure_weight = float(strategy_params.get("swing_structure_weight", structure_weight))
    drift_weight = float(strategy_params.get("drift_weight", structure_weight))
    range_pressure_weight = float(strategy_params.get("range_pressure_weight", structure_weight * 0.2))
    engulfing_weight = float(strategy_params.get("engulfing_weight", 1.0))
    reversal_candle_weight = float(strategy_params.get("reversal_candle_weight", 0.6))
    rsi_divergence_weight = float(strategy_params.get("rsi_divergence_weight", 0.9))
    macd_early_momentum_weight = float(strategy_params.get("macd_early_momentum_weight", 0.55))
    macd_hist_slope_scale = float(strategy_params.get("macd_hist_slope_scale", 0.06))
    volume_spike_trigger_weight = float(strategy_params.get("volume_spike_trigger_weight", 0.5))
    verdict_margin_threshold = float(strategy_params.get("verdict_margin_threshold", 1.2))
    confidence_margin_multiplier = float(strategy_params.get("confidence_margin_multiplier", 8.0))
    confidence_evidence_multiplier = float(strategy_params.get("confidence_evidence_multiplier", 1.4))
    rangebound_penalty = float(strategy_params.get("rangebound_penalty", 8.0))
    weak_trend_penalty = float(strategy_params.get("weak_trend_penalty", 3.0))
    volume_unavailable_penalty = float(strategy_params.get("volume_unavailable_penalty", 5.0))
    mixed_alignment_penalty = float(strategy_params.get("mixed_alignment_penalty", 6.0))
    neutral_confidence_cap = float(strategy_params.get("neutral_confidence_cap", 63))
    sr_reaction_weight = float(strategy_params.get("sr_reaction_weight", 1.6))
    sr_proximity_weight = float(strategy_params.get("sr_proximity_weight", 0.5))
    pivot_confluence_weight = float(strategy_params.get("pivot_confluence_weight", 0.35))
    special_sr_confluence_weight = float(strategy_params.get("special_sr_confluence_weight", 0.18))
    trigger_min_score = float(strategy_params.get("trigger_min_score", 1.3))
    no_trade_adx_threshold = float(strategy_params.get("no_trade_adx_threshold", 18))
    no_trade_confidence_cap = float(strategy_params.get("no_trade_confidence_cap", 60))
    event_risk_penalty = float(strategy_params.get("event_risk_penalty", 15.0))
    directional_expansion_threshold = float(strategy_params.get("directional_expansion_threshold", 78.0))
    event_watch_setup_weight = float(strategy_params.get("event_watch_setup_weight", 0.35))
    event_breakout_setup_weight = float(strategy_params.get("event_breakout_setup_weight", 0.7))
    event_directional_setup_weight = float(strategy_params.get("event_directional_setup_weight", 1.15))
    event_momentum_setup_weight = float(strategy_params.get("event_momentum_setup_weight", 1.55))
    event_alignment_boost = float(strategy_params.get("event_alignment_boost", 0.35))
    anti_chop_margin_buffer = float(strategy_params.get("anti_chop_margin_buffer", 0.8))
    anti_chop_trigger_buffer = float(strategy_params.get("anti_chop_trigger_buffer", 0.35))
    anti_chop_tradeability_floor = float(strategy_params.get("anti_chop_tradeability_floor", 68.0))
    anti_chop_penalty = float(strategy_params.get("anti_chop_penalty", 8.0))
    fakeout_risk_penalty = float(strategy_params.get("fakeout_risk_penalty", 6.0))

    event_breakout_bias = str(regime_state.get("breakout_bias", "Neutral"))
    cross_asset_bias = str(regime_state.get("cross_asset_bias", "Neutral"))
    big_move_risk = float(regime_state.get("big_move_risk", 0.0) or 0.0)
    expansion_probability_30m = float(regime_state.get("expansion_probability_30m", 0.0) or 0.0)
    expansion_probability_60m = float(regime_state.get("expansion_probability_60m", 0.0) or 0.0)
    warning_ladder = str(regime_state.get("warning_ladder", "Normal") or "Normal")
    event_regime_label = str(regime_state.get("event_regime", "normal") or "normal")

    bull_setup = 0.0
    bear_setup = 0.0
    bull_trigger = 0.0
    bear_trigger = 0.0
    no_trade_reasons = []
    hard_no_trade_reasons = []

    if trend == "Bullish":
        bull_setup += trend_base_weight
    elif trend == "Bearish":
        bear_setup += trend_base_weight

    if alignment_score > 0:
        bull_setup += min(alignment_score, 3) * alignment_weight
    elif alignment_score < 0:
        bear_setup += min(abs(alignment_score), 3) * alignment_weight

    trend_probability = _safe_float(market_regime_scores.get("trend_probability"), 0.0)
    expansion_probability = _safe_float(market_regime_scores.get("expansion_probability"), 0.0)
    if trend == "Bullish":
        bull_setup += trend_regime_bonus * trend_probability
        bull_setup += weak_trend_bonus * expansion_probability * 0.7
    elif trend == "Bearish":
        bear_setup += trend_regime_bonus * trend_probability
        bear_setup += weak_trend_bonus * expansion_probability * 0.7

    if "Accumulation" in volume:
        bull_setup += strong_volume_weight
    elif "Buying Bias" in volume:
        bull_setup += bias_volume_weight
    if "Distribution" in volume:
        bear_setup += strong_volume_weight
    elif "Selling Bias" in volume:
        bear_setup += bias_volume_weight
    if cross_asset_bias == "Bullish":
        bull_setup += 0.45
    elif cross_asset_bias == "Bearish":
        bear_setup += 0.45

    sr_reaction = str(support_resistance.get("reaction", "None") or "None")
    support_distance_pct = support_resistance.get("support_distance_pct")
    resistance_distance_pct = support_resistance.get("resistance_distance_pct")
    nearby_supports = support_resistance.get("nearby_supports") if isinstance(support_resistance.get("nearby_supports"), list) else []
    nearby_resistances = support_resistance.get("nearby_resistances") if isinstance(support_resistance.get("nearby_resistances"), list) else []
    opening_range_break = _coerce_signal_state(structure_context.get("openingRangeBreak"))
    sweep_reclaim_signal = _coerce_signal_state(structure_context.get("sweepReclaimSignal"))
    sweep_reclaim_quality = _safe_float(structure_context.get("sweepReclaimQuality"), 0.0)
    support_pivot_distance_pct = _nearest_family_distance_pct(nearby_supports, "pivot")
    resistance_pivot_distance_pct = _nearest_family_distance_pct(nearby_resistances, "pivot")
    sweep_signal_strength = min(max(sweep_reclaim_quality, 0.0), 1.0) if sweep_reclaim_signal != 0 else 0.0
    support_special_families = _nearby_level_special_family_count(nearby_supports)
    resistance_special_families = _nearby_level_special_family_count(nearby_resistances)
    bullish_sweep_reversal_context = (
        sweep_reclaim_signal > 0
        and (
            sr_reaction == "Bullish Support Rejection"
            or (isinstance(support_distance_pct, (int, float)) and support_distance_pct <= 0.2)
            or trend != "Bullish"
            or alignment_score <= 0
            or "Bearish" in pa_struct
        )
    )
    bearish_sweep_reversal_context = (
        sweep_reclaim_signal < 0
        and (
            sr_reaction == "Bearish Resistance Rejection"
            or (isinstance(resistance_distance_pct, (int, float)) and resistance_distance_pct <= 0.2)
            or trend != "Bearish"
            or alignment_score >= 0
            or "Bullish" in pa_struct
        )
    )
    bull_pattern_confirmed = False
    bear_pattern_confirmed = False

    if "Bullish Breakout" in pa_struct:
        bull_trigger += breakout_weight
    elif "Bearish Breakdown" in pa_struct:
        bear_trigger += breakout_weight
    elif "Bullish Structure" in pa_struct:
        bull_trigger += swing_structure_weight
    elif "Bearish Structure" in pa_struct:
        bear_trigger += swing_structure_weight
    elif "Bullish Drift" in pa_struct:
        bull_trigger += drift_weight
    elif "Bearish Drift" in pa_struct:
        bear_trigger += drift_weight
    elif "Bullish Pressure" in pa_struct:
        bull_trigger += range_pressure_weight
    elif "Bearish Pressure" in pa_struct:
        bear_trigger += range_pressure_weight

    if "Bullish Engulfing" in pa_pattern:
        if bullish_sweep_reversal_context:
            bull_trigger += engulfing_weight * 0.30
            bull_pattern_confirmed = True
    elif "Bearish Engulfing" in pa_pattern:
        if bearish_sweep_reversal_context:
            bear_trigger += engulfing_weight * 0.30
            bear_pattern_confirmed = True
    elif "Bullish Hammer" in pa_pattern:
        if bullish_sweep_reversal_context:
            bull_trigger += reversal_candle_weight * 0.40
            bull_pattern_confirmed = True
    elif "Bearish Shooting Star" in pa_pattern:
        if bearish_sweep_reversal_context:
            bear_trigger += reversal_candle_weight * 0.40
            bear_pattern_confirmed = True

    if opening_range_break > 0:
        bull_trigger += 0.8
    elif opening_range_break < 0:
        bear_trigger += 0.8
    if sweep_reclaim_signal > 0:
        bull_trigger += 0.5 + (0.4 * sweep_signal_strength)
    elif sweep_reclaim_signal < 0:
        bear_trigger += 0.5 + (0.4 * sweep_signal_strength)

    confirmed_candle_bias = "Neutral"
    if bull_pattern_confirmed and not bear_pattern_confirmed:
        confirmed_candle_bias = "Bullish"
    elif bear_pattern_confirmed and not bull_pattern_confirmed:
        confirmed_candle_bias = "Bearish"

    event_setup_boost = 0.0
    if warning_ladder == "Expansion Watch":
        event_setup_boost = event_watch_setup_weight
    elif warning_ladder == "High Breakout Risk":
        event_setup_boost = event_breakout_setup_weight
    elif warning_ladder == "Directional Expansion Likely":
        event_setup_boost = event_directional_setup_weight
    elif warning_ladder == "Active Momentum Event":
        event_setup_boost = event_momentum_setup_weight

    if event_breakout_bias == "Bullish" and event_setup_boost > 0:
        bull_setup += event_setup_boost
        if trend == "Bullish" or alignment_score > 0:
            bull_setup += event_alignment_boost
    elif event_breakout_bias == "Bearish" and event_setup_boost > 0:
        bear_setup += event_setup_boost
        if trend == "Bearish" or alignment_score < 0:
            bear_setup += event_alignment_boost

    if rsi_bullish_divergence and not rsi_bearish_divergence:
        bull_trigger += rsi_divergence_weight
        if rsi_divergence_strength > 0.0:
            bull_trigger += min(0.35, rsi_divergence_strength * 0.1)
    elif rsi_bearish_divergence and not rsi_bullish_divergence:
        bear_trigger += rsi_divergence_weight
        if rsi_divergence_strength < 0.0:
            bear_trigger += min(0.35, abs(rsi_divergence_strength) * 0.1)

    if abs(macd_hist_slope) >= 0.003:
        slope_strength = _clamp(abs(macd_hist_slope) / max(macd_hist_slope_scale, 1e-4), 0.0, 1.4)
        if macd_hist_slope > 0.0:
            bull_setup += macd_early_momentum_weight * slope_strength
            if macd_hist > 0.0:
                bull_trigger += macd_early_momentum_weight * 0.25
        else:
            bear_setup += macd_early_momentum_weight * slope_strength
            if macd_hist < 0.0:
                bear_trigger += macd_early_momentum_weight * 0.25

    if volume_spike or volume_zscore >= 1.4:
        volume_impulse = max(1.0, min(2.0, 1.0 + max(0.0, volume_zscore - 1.4) * 0.35))
        if opening_range_break > 0 or sweep_reclaim_signal > 0 or "Bullish" in pa_struct:
            bull_trigger += volume_spike_trigger_weight * volume_impulse
        elif opening_range_break < 0 or sweep_reclaim_signal < 0 or "Bearish" in pa_struct:
            bear_trigger += volume_spike_trigger_weight * volume_impulse
        elif trend == "Bullish" and alignment_score > 0:
            bull_setup += volume_spike_trigger_weight * 0.55 * min(volume_impulse, 1.6)
        elif trend == "Bearish" and alignment_score < 0:
            bear_setup += volume_spike_trigger_weight * 0.55 * min(volume_impulse, 1.6)

    if sr_reaction == "Bullish Support Rejection" or sr_reaction == "Bullish Breakout Through Resistance":
        bull_trigger += sr_reaction_weight
    elif sr_reaction == "Bearish Resistance Rejection" or sr_reaction == "Bearish Breakdown Through Support":
        bear_trigger += sr_reaction_weight

    if isinstance(support_distance_pct, (int, float)) and support_distance_pct <= 0.2:
        bull_setup += sr_proximity_weight
    if isinstance(resistance_distance_pct, (int, float)) and resistance_distance_pct <= 0.2:
        bear_setup += sr_proximity_weight

    if support_special_families >= 2:
        bull_setup += min(0.55, (support_special_families - 1) * (special_sr_confluence_weight * 2.0))
    if resistance_special_families >= 2:
        bear_setup += min(0.55, (resistance_special_families - 1) * (special_sr_confluence_weight * 2.0))

    if (
        _nearby_level_has_family(nearby_supports, "pivot")
        and isinstance(support_pivot_distance_pct, (int, float))
        and support_pivot_distance_pct <= 0.22
    ):
        bull_setup += pivot_confluence_weight
    if (
        _nearby_level_has_family(nearby_resistances, "pivot")
        and isinstance(resistance_pivot_distance_pct, (int, float))
        and resistance_pivot_distance_pct <= 0.22
    ):
        bear_setup += pivot_confluence_weight

    bull_score = bull_setup + bull_trigger
    bear_score = bear_setup + bear_trigger
    score_diff = bull_score - bear_score
    score_margin = abs(score_diff)
    evidence_total = bull_score + bear_score

    direction_hold_threshold = float(strategy_params.get("direction_hold_threshold", 5.0))
    exit_risk_threshold = float(strategy_params.get("exit_risk_threshold", 6.0))
    regime_quality_weight = float(strategy_params.get("regime_quality_weight", 0.30))
    alignment_quality_weight = float(strategy_params.get("alignment_quality_weight", 0.20))
    structure_quality_weight = float(strategy_params.get("structure_quality_weight", 0.20))
    trigger_quality_weight = float(strategy_params.get("trigger_quality_weight", 0.15))
    volume_quality_weight = float(strategy_params.get("volume_quality_weight", 0.10))
    stability_flip_penalty = float(strategy_params.get("stability_flip_penalty", 12.0))
    stability_conflict_penalty = float(strategy_params.get("stability_conflict_penalty", 10.0))
    stability_mixed_alignment_penalty = float(strategy_params.get("stability_mixed_alignment_penalty", 10.0))
    high_tradeability_threshold = float(strategy_params.get("high_tradeability_threshold", 68.0))
    medium_tradeability_threshold = float(strategy_params.get("medium_tradeability_threshold", 52.0))
    direction_probability_floor = float(strategy_params.get("direction_probability_floor", 0.56))
    transition_setup_tradeability_floor = float(strategy_params.get("transition_setup_tradeability_floor", 54.0))
    transition_setup_entry_prob_floor = float(strategy_params.get("transition_setup_entry_prob_floor", 0.57))
    breakout_setup_tradeability_floor = float(strategy_params.get("breakout_setup_tradeability_floor", 28.0))
    breakout_setup_direction_prob_30_floor = float(strategy_params.get("breakout_setup_direction_prob_30_floor", 0.62))
    breakout_setup_direction_prob_60_floor = float(strategy_params.get("breakout_setup_direction_prob_60_floor", 0.68))
    breakout_setup_expected_edge_floor = float(strategy_params.get("breakout_setup_expected_edge_floor", 0.08))
    breakout_setup_meta_probability_floor = float(strategy_params.get("breakout_setup_meta_probability_floor", 0.56))
    breakout_setup_fakeout_probability_cap = float(strategy_params.get("breakout_setup_fakeout_probability_cap", 0.48))
    breakout_setup_exit_probability_cap = float(strategy_params.get("breakout_setup_exit_probability_cap", 0.38))
    breakout_setup_projected_move_atr_floor = float(strategy_params.get("breakout_setup_projected_move_atr_floor", 1.10))
    breakout_setup_one_atr_probability_floor = float(strategy_params.get("breakout_setup_one_atr_probability_floor", 0.46))
    breakout_active_tradeability_floor = float(strategy_params.get("breakout_active_tradeability_floor", 32.0))
    breakout_active_direction_prob_30_floor = float(strategy_params.get("breakout_active_direction_prob_30_floor", 0.72))
    breakout_active_direction_prob_60_floor = float(strategy_params.get("breakout_active_direction_prob_60_floor", 0.78))
    breakout_active_expected_edge_floor = float(strategy_params.get("breakout_active_expected_edge_floor", 0.10))
    breakout_active_meta_probability_floor = float(strategy_params.get("breakout_active_meta_probability_floor", 0.62))
    breakout_active_entry_timing_floor = float(strategy_params.get("breakout_active_entry_timing_floor", 78.0))
    breakout_active_fakeout_probability_cap = float(strategy_params.get("breakout_active_fakeout_probability_cap", 0.42))
    breakout_active_exit_probability_cap = float(strategy_params.get("breakout_active_exit_probability_cap", 0.34))
    breakout_active_projected_move_atr_floor = float(strategy_params.get("breakout_active_projected_move_atr_floor", 1.20))
    breakout_active_one_atr_probability_floor = float(strategy_params.get("breakout_active_one_atr_probability_floor", 0.50))

    regime_router = _build_regime_router(
        ta_data=ta_data,
        regime_state=regime_state,
        trend=trend,
        alignment_score=alignment_score,
        market_structure=pa_struct,
        candle_pattern=pa_pattern,
    )
    regime_probs = regime_router.get("probabilities", {}) if isinstance(regime_router.get("probabilities"), dict) else {}
    regime_label = str(regime_router.get("regime_label") or "transition")
    if event_regime_label == "panic_reversal":
        regime_label = "event-risk"

    direction_probabilities = _compute_direction_probabilities(
        score_diff=score_diff,
        bull_trigger=bull_trigger,
        bear_trigger=bear_trigger,
        trend=trend,
        alignment_score=alignment_score,
        breakout_bias=event_breakout_bias,
        cross_asset_bias=cross_asset_bias,
        regime_router=regime_router,
        structure_context=structure_context,
    )
    bullish_prob_30 = _safe_float(direction_probabilities.get("up_30m"), 0.5)
    bearish_prob_30 = _safe_float(direction_probabilities.get("down_30m"), 0.5)
    bullish_prob_60 = _safe_float(direction_probabilities.get("up_60m"), 0.5)
    bearish_prob_60 = _safe_float(direction_probabilities.get("down_60m"), 0.5)

    probabilistic_direction = "Neutral"
    if bullish_prob_60 >= direction_probability_floor and bullish_prob_60 >= bearish_prob_60 + 0.06:
        probabilistic_direction = "Bullish"
    elif bearish_prob_60 >= direction_probability_floor and bearish_prob_60 >= bullish_prob_60 + 0.06:
        probabilistic_direction = "Bearish"
    elif score_diff >= verdict_margin_threshold * 1.15:
        probabilistic_direction = "Bullish"
    elif score_diff <= -verdict_margin_threshold * 1.15:
        probabilistic_direction = "Bearish"

    price_action_bias = _price_action_bias(pa_struct, pa_pattern)
    directional_resolution = _resolve_directional_bias(
        h1_trend=h1_trend,
        h4_trend=h4_trend,
        m15_trend=m15_trend,
        base_trend=trend,
        probabilistic_direction=probabilistic_direction,
        event_breakout_bias=event_breakout_bias,
        price_action_bias=price_action_bias,
        confirmed_candle_bias=confirmed_candle_bias,
        opening_range_break=opening_range_break,
        sweep_reclaim_signal=sweep_reclaim_signal,
        score_diff=score_diff,
        bullish_prob_30=bullish_prob_30,
        bearish_prob_30=bearish_prob_30,
        bullish_prob_60=bullish_prob_60,
        bearish_prob_60=bearish_prob_60,
        warning_ladder=warning_ladder,
        expansion_probability_60m=expansion_probability_60m,
        big_move_risk=big_move_risk,
        direction_probability_floor=direction_probability_floor,
        verdict_margin_threshold=verdict_margin_threshold,
    )
    h1_direction = directional_resolution["h1_direction"]
    directional_bias = directional_resolution["directional_bias"]
    directional_bias_source = directional_resolution["directional_bias_source"]
    direction_timeframe = directional_resolution["direction_timeframe"]
    h1_neutral_override_active = directional_resolution["h1_neutral_override_active"]
    h4_opposes_direction = directional_resolution["h4_opposes_direction"]
    h4_filter_pass = directional_resolution["h4_filter_pass"]
    m15_trigger_aligned = directional_resolution["m15_trigger_aligned"]
    m15_trigger_conflict = directional_resolution["m15_trigger_conflict"]

    if warning_ladder in {"Directional Expansion Likely", "Active Momentum Event"} and event_breakout_bias != "Neutral":
        if directional_bias != "Neutral" and directional_bias != event_breakout_bias:
            message = "Directional event regime conflicts with the core setup."
            if warning_ladder == "Active Momentum Event":
                hard_no_trade_reasons.append(message)
            else:
                no_trade_reasons.append(message)

    if directional_bias in {"Bullish", "Bearish"}:
        verdict = directional_bias

    if h1_direction == "Neutral" and not h1_neutral_override_active:
        no_trade_reasons.append("H1 direction is neutral.")
    if h4_opposes_direction:
        hard_no_trade_reasons.append("H4 trend opposes the selected direction.")
    if m15_trigger_conflict:
        if directional_bias_source == "breakout_stack":
            no_trade_reasons.append("M15 trend disagrees with the breakout direction.")
        else:
            no_trade_reasons.append("M15 trend disagrees with H1 direction.")

    conflict_count = 0
    doji_is_indecision = (
        "Doji" in pa_pattern
        and opening_range_break == 0
        and sweep_reclaim_signal == 0
        and sr_reaction == "None"
    )

    if trend == "Bullish" and ("Bearish" in pa_struct or confirmed_candle_bias == "Bearish"):
        conflict_count += 1
    if trend == "Bearish" and ("Bullish" in pa_struct or confirmed_candle_bias == "Bullish"):
        conflict_count += 1
    if alignment_label == "Mixed / Low Alignment":
        conflict_count += 1
    if regime_label in {"event-risk", "unstable"}:
        conflict_count += 1
    elif regime_label == "transition" and _safe_float(regime_probs.get("breakout_transition"), 0.0) < 0.34:
        conflict_count += 1
    if rsi_bullish_divergence and rsi_bearish_divergence:
        conflict_count += 1

    stability_score = 100.0
    stability_score -= conflict_count * stability_conflict_penalty
    if alignment_label == "Mixed / Low Alignment":
        stability_score -= stability_mixed_alignment_penalty
    if regime_label == "transition":
        stability_score -= stability_flip_penalty * (0.18 if _safe_float(regime_probs.get("breakout_transition"), 0.0) >= 0.34 else 0.40)
    elif regime_label == "unstable":
        stability_score -= stability_flip_penalty * 0.55
    if doji_is_indecision:
        stability_score -= 8.0
    if no_trade_reasons or hard_no_trade_reasons:
        stability_score -= 8.0
    stability_score = max(0.0, min(100.0, stability_score))

    regime_quality = _clamp(
        (_safe_float(regime_probs.get("trend_continuation"), 0.0) * 0.78)
        + (_safe_float(regime_probs.get("breakout_transition"), 0.0) * 0.88)
        - (_safe_float(regime_probs.get("mean_reversion"), 0.0) * 0.08)
        - (_safe_float(regime_probs.get("event_shock"), 0.0) * 0.16),
        0.0,
        1.0,
    ) * 100.0
    alignment_quality = _clamp(
        (_alignment_quality_score(alignment_score, alignment_label) * 0.58)
        + ((1.0 if cross_asset_bias == directional_bias and directional_bias in {"Bullish", "Bearish"} else 0.38) * 0.20)
        + ((1.0 if event_breakout_bias == directional_bias and directional_bias in {"Bullish", "Bearish"} else 0.42) * 0.22),
        0.0,
        1.0,
    ) * 100.0
    structure_quality = _clamp(
        ((max(bull_trigger, bear_trigger) / max(trigger_min_score + 2.3, 1.0)) * 0.55)
        + ((1.0 if any(token in pa_struct for token in ["Breakout", "Breakdown", "Bullish Drift", "Bearish Drift", "Bullish Structure", "Bearish Structure"]) else (0.62 if "Pressure" in pa_struct else 0.28)) * 0.23)
        + ((1.0 if opening_range_break != 0 else 0.0) * 0.10)
        + ((1.0 if sweep_reclaim_signal != 0 else 0.0) * 0.12),
        0.0,
        1.0,
    ) * 100.0
    trigger_quality = _clamp(
        ((max(bull_trigger, bear_trigger) / max(trigger_min_score + 2.1, 1.0)) * 0.70)
        + (((max(bullish_prob_30, bearish_prob_30) - 0.5) / 0.5) * 0.30),
        0.0,
        1.0,
    ) * 100.0
    volume_quality = 75.0 if "Strong" in volume else 55.0 if "Bias" in volume else 40.0
    if volume_spike:
        volume_quality = max(volume_quality, 72.0)
    tradeability_score = (
        regime_quality * regime_quality_weight
        + alignment_quality * alignment_quality_weight
        + structure_quality * structure_quality_weight
        + trigger_quality * trigger_quality_weight
        + volume_quality * volume_quality_weight
    )
    tradeability_score = max(0.0, min(100.0, tradeability_score * (stability_score / 100.0)))

    if tradeability_score >= high_tradeability_threshold:
        tradeability_label = "High"
    elif tradeability_score >= medium_tradeability_threshold:
        tradeability_label = "Medium"
    else:
        tradeability_label = "Low"

    direction_score = round(max(bull_score, bear_score, bullish_prob_60 * 10.0, bearish_prob_60 * 10.0), 2)
    exit_risk_score = 0.0
    if verdict == "Bullish":
        exit_risk_score = max(
            bear_trigger + (2.0 if alignment_label == "Mixed / Low Alignment" else 0.0) + (2.0 if "Bearish" in pa_pattern else 0.0),
            (bearish_prob_30 * 10.0) + (_safe_float(regime_probs.get("event_shock"), 0.0) * 2.0),
        )
    elif verdict == "Bearish":
        exit_risk_score = max(
            bull_trigger + (2.0 if alignment_label == "Mixed / Low Alignment" else 0.0) + (2.0 if "Bullish" in pa_pattern else 0.0),
            (bullish_prob_30 * 10.0) + (_safe_float(regime_probs.get("event_shock"), 0.0) * 2.0),
        )
    else:
        exit_risk_score = 4.0 if (no_trade_reasons or hard_no_trade_reasons) else 2.0

    action_state = "WAIT"
    if (
        directional_bias in {"Bullish", "Bearish"}
        and h4_filter_pass
        and not m15_trigger_conflict
        and regime_label not in {"event-risk", "unstable"}
        and tradeability_score >= transition_setup_tradeability_floor
    ):
        if directional_bias == "Bullish":
            if (
                m15_trigger_aligned
                and bullish_prob_30 >= 0.64
                and bullish_prob_60 >= 0.62
                and bull_trigger >= trigger_min_score
                and tradeability_score >= medium_tradeability_threshold
            ):
                action_state = "LONG_ACTIVE"
            elif bullish_prob_60 >= transition_setup_entry_prob_floor and direction_score >= direction_hold_threshold:
                action_state = "SETUP_LONG"
        elif directional_bias == "Bearish":
            if (
                m15_trigger_aligned
                and bearish_prob_30 >= 0.64
                and bearish_prob_60 >= 0.62
                and bear_trigger >= trigger_min_score
                and tradeability_score >= medium_tradeability_threshold
            ):
                action_state = "SHORT_ACTIVE"
            elif bearish_prob_60 >= transition_setup_entry_prob_floor and direction_score >= direction_hold_threshold:
                action_state = "SETUP_SHORT"

    if regime_label == "range" and _safe_float(regime_probs.get("breakout_transition"), 0.0) < 0.33 and max(bull_trigger, bear_trigger) < (trigger_min_score + 0.1):
        action_state = "WAIT"

    if verdict in {"Bullish", "Bearish"} and exit_risk_score >= exit_risk_threshold:
        action_state = "EXIT_RISK"

    regime_bucket = _regime_bucket(regime_label, warning_ladder, event_regime_label)
    anti_chop_reasons = []
    if warning_ladder in {"Expansion Watch", "High Breakout Risk"} and event_breakout_bias == "Neutral":
        anti_chop_reasons.append("Expansion risk is rising without directional confirmation.")
    if warning_ladder in {"Expansion Watch", "High Breakout Risk"} and tradeability_score < anti_chop_tradeability_floor and regime_label != "transition":
        anti_chop_reasons.append("Tradeability is still below the execution floor for breakout conditions.")
    if alignment_label == "Mixed / Low Alignment" and score_margin < (verdict_margin_threshold + anti_chop_margin_buffer) and max(bullish_prob_60, bearish_prob_60) < 0.60:
        anti_chop_reasons.append("Directional edge is still too narrow in mixed alignment.")
    if doji_is_indecision and warning_ladder != "Active Momentum Event":
        anti_chop_reasons.append("Indecision candle is weakening the trigger.")
    if (
        action_state in {"SETUP_LONG", "SETUP_SHORT"}
        and max(bull_trigger, bear_trigger) < (trigger_min_score + anti_chop_trigger_buffer)
        and max(bullish_prob_30, bearish_prob_30) < 0.60
    ):
        anti_chop_reasons.append("Trigger quality is still too weak for setup promotion.")
    if (
        warning_ladder == "Directional Expansion Likely"
        and event_breakout_bias != "Neutral"
        and directional_bias != "Neutral"
        and event_breakout_bias != directional_bias
    ):
        anti_chop_reasons.append("Forecast direction conflicts with the execution direction.")

    action = "hold"
    if action_state == "LONG_ACTIVE":
        action = "buy"
    elif action_state == "SHORT_ACTIVE":
        action = "sell"
    elif action_state == "EXIT_RISK":
        action = "exit"

    base_conf = (
        50
        + ((max(bullish_prob_60, bearish_prob_60) - 0.5) * 56.0)
        + (score_margin * confidence_margin_multiplier * 0.55)
        + (evidence_total * confidence_evidence_multiplier)
        + (stability_score * 0.10)
    )
    penalty = 0.0

    if regime_label == "range":
        penalty += rangebound_penalty
    elif regime_label == "transition" and _safe_float(regime_probs.get("breakout_transition"), 0.0) < 0.33:
        penalty += weak_trend_penalty
    elif regime == "Weak Trend":
        penalty += weak_trend_penalty

    if "N/A" in volume:
        penalty += volume_unavailable_penalty
    if alignment_label == "Mixed / Low Alignment":
        penalty += mixed_alignment_penalty
    if warning_ladder == "Directional Expansion Likely" and event_breakout_bias == directional_bias and directional_bias != "Neutral":
        base_conf += 4.0
    elif warning_ladder == "Active Momentum Event" and event_breakout_bias == directional_bias and directional_bias != "Neutral":
        base_conf += 6.0
    if event_risk.get("active"):
        penalty += event_risk_penalty
        if directional_bias == "Neutral" or event_breakout_bias == "Neutral":
            hard_no_trade_reasons.append("Major macro event window is active.")
        else:
            no_trade_reasons.append("Major macro event window is active.")
    elif big_move_risk >= directional_expansion_threshold and event_breakout_bias == "Neutral":
        no_trade_reasons.append("Large move risk is elevated, but directional bias is still unclear.")
    elif warning_ladder in {"Directional Expansion Likely", "Active Momentum Event"} and event_breakout_bias != "Neutral" and directional_bias != "Neutral" and event_breakout_bias != directional_bias:
        penalty += 6.0
    if regime == "Range-Bound" and alignment_label == "Mixed / Low Alignment":
        no_trade_reasons.append("Range-bound regime with mixed alignment.")
    if adx_14 < no_trade_adx_threshold and _safe_float(regime_probs.get("breakout_transition"), 0.0) < 0.34:
        no_trade_reasons.append("Trend strength is too weak.")
    if verdict == "Neutral" and max(bull_trigger, bear_trigger) < (trigger_min_score * 0.75) and max(bullish_prob_30, bearish_prob_30) < 0.58:
        no_trade_reasons.append("No clean trigger is present.")
    fakeout_risk_score = float(((regime_state.get("components") or {}).get("fakeout_risk_score")) or 0.0)
    directional_conviction = max(bullish_prob_60, bearish_prob_60)
    fakeout_hard_block = (
        fakeout_risk_score >= 3.6
        and not (
            directional_conviction >= 0.74
            and expansion_probability_60m >= 58.0
            and tradeability_score >= breakout_setup_tradeability_floor
        )
    )
    if fakeout_hard_block:
        anti_chop_reasons.append("Breakout risk is elevated, but fakeout risk remains high.")

    if anti_chop_reasons:
        penalty += anti_chop_penalty
    if fakeout_hard_block:
        penalty += fakeout_risk_penalty
        if action_state in {"SETUP_LONG", "SETUP_SHORT"}:
            action_state = "WAIT"
            action = "hold"
        if not no_trade_reasons and anti_chop_reasons:
            no_trade_reasons.append(anti_chop_reasons[0])

    confidence = base_conf - penalty
    breakout_confidence_floor = None
    if directional_bias_source == "breakout_stack" and verdict in {"Bullish", "Bearish"}:
        breakout_confidence_floor = 54.0 + max(
            0.0,
            min(8.0, (max(bullish_prob_60, bearish_prob_60) - 0.60) * 20.0),
        )
        confidence = max(confidence, breakout_confidence_floor)
    if hard_no_trade_reasons:
        verdict = "Neutral"
        confidence = min(confidence, no_trade_confidence_cap)
        action_state = "WAIT" if action_state != "EXIT_RISK" else "EXIT_RISK"
        action = "hold" if action_state == "WAIT" else "exit"
    elif verdict == "Neutral":
        confidence = min(confidence, neutral_confidence_cap)
    confidence = round(min(max(confidence, 50), 95))
    confidence, confidence_mode, confidence_bucket = _calibrate_confidence(
        confidence,
        regime_label,
        directional_bias,
        tradeability_label,
        stability_score,
    )
    if breakout_confidence_floor is not None and verdict in {"Bullish", "Bearish"}:
        confidence = max(confidence, round(breakout_confidence_floor))
    meta_scores = _compute_meta_model_scores(
        direction_score=direction_score,
        trigger_score=max(bull_trigger, bear_trigger),
        tradeability_score=tradeability_score,
        stability_score=stability_score,
        fakeout_risk_score=fakeout_risk_score,
        exit_risk_score=exit_risk_score,
        warning_ladder=warning_ladder,
        alignment_label=alignment_label,
    )
    expected_edge_pct = _expected_value_edge_pct(confidence, strategy_params, regime_bucket)
    move_bucket_state = _compute_move_bucket_state(
        ta_data=ta_data,
        regime_state=regime_state,
        direction_probabilities=direction_probabilities,
        tradeability_score=tradeability_score,
        stability_score=stability_score,
        meta_scores=meta_scores,
        expected_edge_pct=expected_edge_pct,
        strategy_params=strategy_params,
        regime_bucket=regime_bucket,
    )
    execution_meta = _compute_execution_meta_label(
        direction_probabilities=direction_probabilities,
        move_bucket_state=move_bucket_state,
        tradeability_score=tradeability_score,
        stability_score=stability_score,
        meta_scores=meta_scores,
        expected_edge_pct=expected_edge_pct,
        regime_router=regime_router,
    )
    breakout_setup_prob_30, breakout_setup_prob_60 = _direction_probability_for_label(
        directional_bias,
        bullish_prob_30,
        bearish_prob_30,
        bullish_prob_60,
        bearish_prob_60,
    )
    breakout_move_buckets = (
        move_bucket_state.get("probabilities")
        if isinstance(move_bucket_state.get("probabilities"), dict)
        else {}
    )
    breakout_one_atr_probability = _safe_float(breakout_move_buckets.get("1.0_atr"), 0.0)
    breakout_projected_move_atr = _safe_float(move_bucket_state.get("projected_move_atr"), 0.0)
    breakout_setup_eligible = (
        action_state == "WAIT"
        and directional_bias_source == "breakout_stack"
        and directional_bias in {"Bullish", "Bearish"}
        and not hard_no_trade_reasons
        and h4_filter_pass
        and not m15_trigger_conflict
        and m15_trigger_aligned
        and regime_label in {"transition", "trend"}
        and tradeability_score >= breakout_setup_tradeability_floor
        and breakout_setup_prob_30 >= breakout_setup_direction_prob_30_floor
        and breakout_setup_prob_60 >= breakout_setup_direction_prob_60_floor
        and breakout_one_atr_probability >= breakout_setup_one_atr_probability_floor
        and breakout_projected_move_atr >= breakout_setup_projected_move_atr_floor
        and meta_scores["fakeout_probability"] <= breakout_setup_fakeout_probability_cap
        and meta_scores["exit_risk_probability"] <= breakout_setup_exit_probability_cap
        and execution_meta["meta_label_probability"] >= breakout_setup_meta_probability_floor
        and expected_edge_pct >= breakout_setup_expected_edge_floor
    )
    breakout_active_eligible = (
        action_state in {"WAIT", "SETUP_LONG", "SETUP_SHORT"}
        and directional_bias_source == "breakout_stack"
        and directional_bias in {"Bullish", "Bearish"}
        and not hard_no_trade_reasons
        and not no_trade_reasons
        and not anti_chop_reasons
        and h4_filter_pass
        and not m15_trigger_conflict
        and m15_trigger_aligned
        and regime_label in {"transition", "trend"}
        and tradeability_score >= breakout_active_tradeability_floor
        and breakout_setup_prob_30 >= breakout_active_direction_prob_30_floor
        and breakout_setup_prob_60 >= breakout_active_direction_prob_60_floor
        and breakout_one_atr_probability >= breakout_active_one_atr_probability_floor
        and breakout_projected_move_atr >= breakout_active_projected_move_atr_floor
        and meta_scores["entry_timing_score"] >= breakout_active_entry_timing_floor
        and meta_scores["fakeout_probability"] <= breakout_active_fakeout_probability_cap
        and meta_scores["exit_risk_probability"] <= breakout_active_exit_probability_cap
        and execution_meta["meta_label_probability"] >= breakout_active_meta_probability_floor
        and expected_edge_pct >= breakout_active_expected_edge_floor
    )
    if breakout_active_eligible:
        action_state = "LONG_ACTIVE" if directional_bias == "Bullish" else "SHORT_ACTIVE"
        action = "buy" if directional_bias == "Bullish" else "sell"
    elif breakout_setup_eligible:
        action_state = "SETUP_LONG" if directional_bias == "Bullish" else "SETUP_SHORT"
        action = "hold"
    entry_threshold = float(strategy_params.get("meta_entry_score_threshold", 63.0))
    fakeout_cap = float(strategy_params.get("meta_fakeout_prob_cap", 0.42))
    exit_cap = float(strategy_params.get("meta_exit_prob_cap", 0.58))
    min_expected_edge_pct = float(strategy_params.get("min_expected_edge_pct", 0.06))
    effective_min_expected_edge_pct = min_expected_edge_pct
    if directional_bias_source == "breakout_stack":
        effective_min_expected_edge_pct = min(min_expected_edge_pct, 0.005)
    elif regime_label == "trend" and directional_bias in {"Bullish", "Bearish"}:
        effective_min_expected_edge_pct = min(min_expected_edge_pct, 0.005)
    effective_fakeout_cap = fakeout_cap
    if directional_bias_source == "breakout_stack":
        if action_state in {"LONG_ACTIVE", "SHORT_ACTIVE"}:
            effective_fakeout_cap = max(fakeout_cap, breakout_active_fakeout_probability_cap)
        elif action_state in {"SETUP_LONG", "SETUP_SHORT"}:
            effective_fakeout_cap = max(fakeout_cap, breakout_setup_fakeout_probability_cap)
    promoted_by_breakout = breakout_active_eligible or breakout_setup_eligible

    if action_state in {"LONG_ACTIVE", "SHORT_ACTIVE", "SETUP_LONG", "SETUP_SHORT"}:
        if directional_bias_source == "breakout_stack" and promoted_by_breakout:
            if meta_scores["fakeout_probability"] > (effective_fakeout_cap + 0.10):
                no_trade_reasons.append("Fakeout probability is elevated for the current setup.")
            if meta_scores["exit_risk_probability"] > (exit_cap + 0.10) and action_state in {"SETUP_LONG", "SETUP_SHORT"}:
                no_trade_reasons.append("Exit risk model is too high for a fresh entry.")
        else:
            if meta_scores["entry_timing_score"] < entry_threshold:
                no_trade_reasons.append("Entry timing model is not yet confirming momentum quality.")
            if meta_scores["fakeout_probability"] > effective_fakeout_cap:
                no_trade_reasons.append("Fakeout probability is elevated for the current setup.")
            if meta_scores["exit_risk_probability"] > exit_cap and action_state in {"SETUP_LONG", "SETUP_SHORT"}:
                no_trade_reasons.append("Exit risk model is too high for a fresh entry.")
            if execution_meta["meta_label_probability"] < transition_setup_entry_prob_floor:
                no_trade_reasons.append("Meta-label probability is not strong enough yet.")
        if expected_edge_pct < effective_min_expected_edge_pct:
            no_trade_reasons.append("Expected value edge is below the execution threshold.")

        if no_trade_reasons and action_state in {"LONG_ACTIVE", "SHORT_ACTIVE", "SETUP_LONG", "SETUP_SHORT"}:
            action_state = "WAIT"
            action = "hold"
            confidence = min(confidence, max(no_trade_confidence_cap + 8, 68.0))

    all_no_trade_reasons = list(hard_no_trade_reasons) + list(no_trade_reasons)
    primary_hard_no_trade_reason = hard_no_trade_reasons[0] if hard_no_trade_reasons else ""
    primary_soft_no_trade_reason = no_trade_reasons[0] if no_trade_reasons else ""
    primary_no_trade_reason = primary_hard_no_trade_reason or primary_soft_no_trade_reason

    if verdict in {"Bullish", "Bearish"} and not hard_no_trade_reasons and not no_trade_reasons:
        directional_prob_60 = max(bullish_prob_60, bearish_prob_60)
        if (
            action_state in {"LONG_ACTIVE", "SHORT_ACTIVE"}
            and directional_prob_60 >= 0.68
            and tradeability_score >= medium_tradeability_threshold
        ):
            confidence = max(confidence, 66)
        elif (
            action_state in {"SETUP_LONG", "SETUP_SHORT"}
            and directional_prob_60 >= 0.64
            and tradeability_score >= transition_setup_tradeability_floor
        ):
            confidence = max(confidence, 60)

    forecast_state = _build_forecast_state(
        regime_bucket,
        regime_state,
        confidence,
        directional_bias,
        direction_probabilities=direction_probabilities,
        move_bucket_state=move_bucket_state,
        regime_router=regime_router,
    )
    execution_state = _build_execution_state(
        action_state,
        action,
        tradeability_label,
        all_no_trade_reasons,
        anti_chop_reasons,
        confidence,
    )
    execution_state["entryTimingScore"] = meta_scores["entry_timing_score"]
    execution_state["fakeoutProbability"] = meta_scores["fakeout_probability"]
    execution_state["exitRiskProbability"] = meta_scores["exit_risk_probability"]
    execution_state["expectedEdgePct"] = expected_edge_pct
    execution_state["entrySuccessProbability"] = execution_meta["entry_success_probability"]
    execution_state["metaLabelProbability"] = execution_meta["meta_label_probability"]
    execution_state["directionProbabilities"] = direction_probabilities
    execution_state["moveBuckets"] = move_bucket_state.get("probabilities", {})
    execution_state["projectedMoveAtr"] = move_bucket_state.get("projected_move_atr")
    execution_state["projectedMovePips"] = move_bucket_state.get("projected_move_pips")
    trade_guidance = compute_trade_guidance(
        ta_data,
        confidence,
    )
    higher_timeframe_bias, _, _ = _bias_vote(
        trend,
        "Bullish" if alignment_score > 0 else ("Bearish" if alignment_score < 0 else "Neutral"),
        directional_bias,
    )
    coordination_bias, bullish_votes, bearish_votes = _bias_vote(
        directional_bias,
        event_breakout_bias,
        cross_asset_bias,
        price_action_bias,
    )
    conflicted_bias_stack = bullish_votes > 0 and bearish_votes > 0
    opposing_details = []
    if price_action_bias != "Neutral" and higher_timeframe_bias in {"Bullish", "Bearish"} and price_action_bias != higher_timeframe_bias:
        opposing_details.append(f"price action is {price_action_bias.lower()}")
    if event_breakout_bias != "Neutral" and higher_timeframe_bias in {"Bullish", "Bearish"} and event_breakout_bias != higher_timeframe_bias:
        opposing_details.append(f"breakout pressure is {event_breakout_bias.lower()}")
    if cross_asset_bias != "Neutral" and higher_timeframe_bias in {"Bullish", "Bearish"} and cross_asset_bias != higher_timeframe_bias:
        opposing_details.append(f"cross-asset context is {cross_asset_bias.lower()}")

    def _conflicted_summary(reason_text):
        if higher_timeframe_bias in {"Bullish", "Bearish"} and opposing_details:
            return (
                f"Higher-timeframe trend remains {higher_timeframe_bias.lower()}, but "
                + ", ".join(opposing_details[:-1] + ([f"and {opposing_details[-1]}"] if len(opposing_details) > 1 else opposing_details))
                + f". Conditions are conflicted, so no trade is allowed yet. {reason_text}"
            )
        if coordination_bias in {"Bullish", "Bearish"}:
            return f"Signals are conflicted even though the near-term vote leans {coordination_bias.lower()}. No trade is allowed yet. {reason_text}"
        return f"Signals are conflicted across trend, price action, and breakout context. No trade is allowed yet. {reason_text}"

    if all_no_trade_reasons:
        if primary_hard_no_trade_reason or conflicted_bias_stack:
            trade_guidance["sellLevel"] = "Weak"
            trade_guidance["buyLevel"] = "Weak"
        elif directional_bias == "Bullish":
            trade_guidance["sellLevel"] = "Weak"
            if trade_guidance.get("buyLevel") == "Weak":
                trade_guidance["buyLevel"] = "Watch"
        elif directional_bias == "Bearish":
            trade_guidance["buyLevel"] = "Weak"
            if trade_guidance.get("sellLevel") == "Weak":
                trade_guidance["sellLevel"] = "Watch"

        if conflicted_bias_stack:
            trade_guidance["summary"] = _conflicted_summary(all_no_trade_reasons[0])
        elif directional_bias == "Bearish":
            trade_guidance["summary"] = f"Bearish bias is intact, but conditions are not clean enough for execution yet. {all_no_trade_reasons[0]}"
        elif directional_bias == "Bullish":
            trade_guidance["summary"] = f"Bullish bias is intact, but conditions are not clean enough for execution yet. {all_no_trade_reasons[0]}"
        else:
            trade_guidance["summary"] = all_no_trade_reasons[0]

    if action_state == "SETUP_LONG":
        trade_guidance["summary"] = "Long setup is forming, but trigger confirmation is still pending."
    elif action_state == "LONG_ACTIVE":
        if directional_bias_source == "breakout_stack":
            trade_guidance["summary"] = "Long breakout entry is active; momentum and projected follow-through are strong enough for execution."
        else:
            trade_guidance["summary"] = "Long setup confirmed with acceptable tradeability."
    elif action_state == "SETUP_SHORT":
        trade_guidance["summary"] = "Short setup is forming, but trigger confirmation is still pending."
    elif action_state == "SHORT_ACTIVE":
        if directional_bias_source == "breakout_stack":
            trade_guidance["summary"] = "Short breakout entry is active; momentum and projected follow-through are strong enough for execution."
        else:
            trade_guidance["summary"] = "Short setup confirmed with acceptable tradeability."
    elif action_state == "EXIT_RISK":
        trade_guidance["summary"] = "Exit risk is elevated; the active directional state is deteriorating."
    elif action_state == "WAIT" and directional_bias in {"Bullish", "Bearish"}:
        blockers = []
        if tradeability_label == "Low":
            blockers.append("tradeability is low")
        if regime_label in {"unstable", "range", "event-risk"}:
            blockers.append(f"regime is {regime_label}")
        blocker_text = " and ".join(blockers)
        if conflicted_bias_stack:
            trade_guidance["summary"] = _conflicted_summary(
                f"Because {blocker_text}." if blocker_text else "Execution quality is still too weak."
            )
        elif directional_bias == "Bearish" and trade_guidance["sellLevel"] in {"Watch", "Strong"}:
            trade_guidance["summary"] = (
                "Sell bias is favored, but conditions are not clean enough for execution yet"
                + (f" because {blocker_text}." if blocker_text else ".")
            )
        elif directional_bias == "Bullish" and trade_guidance["buyLevel"] in {"Watch", "Strong"}:
            trade_guidance["summary"] = (
                "Buy bias is favored, but conditions are not clean enough for execution yet"
                + (f" because {blocker_text}." if blocker_text else ".")
            )

    late_breakout_chase = (
        action_state == "WAIT"
        and directional_bias_source == "breakout_stack"
        and directional_bias in {"Bullish", "Bearish"}
        and (
            breakout_projected_move_atr < breakout_setup_projected_move_atr_floor
            or breakout_one_atr_probability < breakout_setup_one_atr_probability_floor
        )
    )
    if late_breakout_chase:
        trade_guidance["summary"] = (
            f"{directional_bias} breakout is already extended; do not chase here. "
            "Wait for a pullback, reset, or fresh H1 confirmation."
        )

    if warning_ladder == "Expansion Watch":
        trade_guidance["summary"] += " Expansion watch is active."
    elif warning_ladder == "High Breakout Risk":
        trade_guidance["summary"] += " High breakout risk is building."
    elif warning_ladder == "Directional Expansion Likely":
        trade_guidance["summary"] += f" Directional expansion is likely with {event_breakout_bias.lower()} bias."
    elif warning_ladder == "Active Momentum Event":
        trade_guidance["summary"] += " Active momentum event conditions are in force."

    return {
        "raw_verdict": verdict,
        "verdict": verdict,
        "confidence": confidence,
        "noTradeReason": primary_no_trade_reason,
        "noTradeReasonHard": primary_hard_no_trade_reason,
        "noTradeReasonSoft": primary_soft_no_trade_reason,
        "hasHardNoTrade": bool(primary_hard_no_trade_reason),
        "setupScore": round(max(bull_setup, bear_setup), 2),
        "triggerScore": round(max(bull_trigger, bear_trigger), 2),
        "directionScore": direction_score,
        "tradeabilityScore": round(tradeability_score, 2),
        "exitRiskScore": round(exit_risk_score, 2),
        "stabilityScore": round(stability_score, 2),
        "regime": regime_label,
        "regimeBucket": regime_bucket,
        "directionalBias": directional_bias,
        "tradeability": tradeability_label,
        "actionState": action_state,
        "action": action,
        "confidenceMode": confidence_mode,
        "confidenceBucket": confidence_bucket,
        "antiChopActive": bool(anti_chop_reasons),
        "antiChopReasons": anti_chop_reasons,
        "FeatureHits": feature_hits,
        "TradeGuidance": trade_guidance,
        "RegimeState": {
            "big_move_risk": round(big_move_risk, 2),
            "expansion_probability_30m": round(expansion_probability_30m, 2),
            "expansion_probability_60m": round(expansion_probability_60m, 2),
            "expected_range_expansion": regime_state.get("expected_range_expansion"),
            "active_regime_profile": active_regime_profile,
            "breakout_bias": event_breakout_bias,
            "event_regime": event_regime_label,
            "warning_ladder": warning_ladder,
            "raw_breakout_bias": regime_state.get("raw_breakout_bias", event_breakout_bias),
            "raw_event_regime": regime_state.get("raw_event_regime", event_regime_label),
            "raw_warning_ladder": regime_state.get("raw_warning_ladder", warning_ladder),
            "warning_dwell_bars": regime_state.get("warning_dwell_bars"),
            "breakout_bias_dwell_bars": regime_state.get("breakout_bias_dwell_bars"),
            "cross_asset_bias": regime_state.get("cross_asset_bias", "Neutral"),
            "cross_asset_available": regime_state.get("cross_asset_available", 0),
            "minutes_to_next_event": regime_state.get("minutes_to_next_event"),
            "next_event_name": regime_state.get("next_event_name"),
            "near_events": regime_state.get("near_events", []),
            "feature_hits": regime_state.get("feature_hits", {}),
            "components": regime_state.get("components", {}),
        },
        "ForecastState": forecast_state,
        "ExecutionState": execution_state,
        "_regime_memory": next_regime_memory,
    }


def _trend_series_from_close(close_series, ema_short, ema_long):
    ema_short_series = ta.trend.EMAIndicator(close_series, window=ema_short).ema_indicator()
    ema_long_series = ta.trend.EMAIndicator(close_series, window=ema_long).ema_indicator()
    trend = pd.Series("Neutral", index=close_series.index, dtype="object")
    bull_mask = (close_series > ema_short_series) & (ema_short_series > ema_long_series)
    bear_mask = (close_series < ema_short_series) & (ema_short_series < ema_long_series)
    trend.loc[bull_mask.fillna(False)] = "Bullish"
    trend.loc[bear_mask.fillna(False)] = "Bearish"
    return trend


def _build_support_resistance_from_row(row):
    current_price = _safe_float(row.get("Close"), 0.0)
    if current_price <= 0:
        return {
            "nearest_support": None,
            "nearest_resistance": None,
            "support_distance_pct": None,
            "resistance_distance_pct": None,
            "reaction": "None",
            "nearby_supports": [],
            "nearby_resistances": [],
            "support_confluence": 0,
            "resistance_confluence": 0,
            "support_family_confluence": 0,
            "resistance_family_confluence": 0,
            "pivot_levels": {
                "pp": None,
                "r1": None,
                "s1": None,
                "r2": None,
                "s2": None,
            },
        }

    levels = []
    for label, key, kind in [
        ("Previous Day High", "PRIOR_DAY_HIGH", "resistance"),
        ("Previous Day Low", "PRIOR_DAY_LOW", "support"),
        ("Previous Week High", "PRIOR_WEEK_HIGH", "resistance"),
        ("Previous Week Low", "PRIOR_WEEK_LOW", "support"),
        ("Recent Swing High", "RECENT_SWING_HIGH_24", "resistance"),
        ("Recent Swing Low", "RECENT_SWING_LOW_24", "support"),
        ("Major Swing High", "RECENT_SWING_HIGH_96", "resistance"),
        ("Major Swing Low", "RECENT_SWING_LOW_96", "support"),
    ]:
        value = row.get(key)
        if isinstance(value, (int, float)) and not pd.isna(value):
            levels.append((label, float(value), kind))

    pivot_point = _optional_float(row.get("PIVOT_POINT"))
    pivot_r1 = _optional_float(row.get("PIVOT_R1"))
    pivot_s1 = _optional_float(row.get("PIVOT_S1"))
    pivot_r2 = _optional_float(row.get("PIVOT_R2"))
    pivot_s2 = _optional_float(row.get("PIVOT_S2"))
    if pivot_point is not None:
        levels.append(("Daily Pivot Point", pivot_point, "support" if pivot_point <= current_price else "resistance"))
    if pivot_r1 is not None:
        levels.append(("Daily Pivot Resistance 1", pivot_r1, "resistance"))
    if pivot_s1 is not None:
        levels.append(("Daily Pivot Support 1", pivot_s1, "support"))
    if pivot_r2 is not None:
        levels.append(("Daily Pivot Resistance 2", pivot_r2, "resistance"))
    if pivot_s2 is not None:
        levels.append(("Daily Pivot Support 2", pivot_s2, "support"))

    supports = [(name, value) for name, value, kind in levels if kind == "support" and value <= current_price]
    resistances = [(name, value) for name, value, kind in levels if kind == "resistance" and value >= current_price]
    nearest_support = max(supports, key=lambda item: item[1]) if supports else None
    nearest_resistance = min(resistances, key=lambda item: item[1]) if resistances else None

    support_distance_pct = ((current_price - nearest_support[1]) / current_price * 100.0) if nearest_support else None
    resistance_distance_pct = ((nearest_resistance[1] - current_price) / current_price * 100.0) if nearest_resistance else None

    opening_range_break = _coerce_signal_state(row.get("OPENING_RANGE_BREAK"))
    sweep_reclaim = _coerce_signal_state(row.get("SWEEP_RECLAIM_SIGNAL"))
    reaction = "None"
    if sweep_reclaim > 0:
        reaction = "Bullish Support Rejection"
    elif sweep_reclaim < 0:
        reaction = "Bearish Resistance Rejection"
    elif opening_range_break > 0:
        reaction = "Bullish Breakout Through Resistance"
    elif opening_range_break < 0:
        reaction = "Bearish Breakdown Through Support"

    nearby_supports = _collect_nearby_levels(levels, current_price, "support")
    nearby_resistances = _collect_nearby_levels(levels, current_price, "resistance")
    support_families = {level.get("family") for level in nearby_supports if isinstance(level, dict)}
    resistance_families = {level.get("family") for level in nearby_resistances if isinstance(level, dict)}

    return {
        "nearest_support": {"label": nearest_support[0], "price": round(nearest_support[1], 2)} if nearest_support else None,
        "nearest_resistance": {"label": nearest_resistance[0], "price": round(nearest_resistance[1], 2)} if nearest_resistance else None,
        "support_distance_pct": round(support_distance_pct, 3) if support_distance_pct is not None else None,
        "resistance_distance_pct": round(resistance_distance_pct, 3) if resistance_distance_pct is not None else None,
        "reaction": reaction,
        "nearby_supports": nearby_supports,
        "nearby_resistances": nearby_resistances,
        "support_confluence": len(nearby_supports),
        "resistance_confluence": len(nearby_resistances),
        "support_family_confluence": len(support_families),
        "resistance_family_confluence": len(resistance_families),
        "pivot_levels": {
            "pp": _round_or_none(pivot_point),
            "r1": _round_or_none(pivot_r1),
            "s1": _round_or_none(pivot_s1),
            "r2": _round_or_none(pivot_r2),
            "s2": _round_or_none(pivot_s2),
        },
    }


def _ensure_volume_column(frame):
    if "Volume" in frame.columns:
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
    else:
        volume = pd.Series(np.nan, index=frame.index, dtype=float)

    has_observed_volume = bool(volume.notna().any())
    price_range = (frame["High"] - frame["Low"]).abs().fillna(0.0)
    close_delta = frame["Close"].diff().abs().fillna(0.0)
    volume_proxy = (price_range + close_delta).rolling(5, min_periods=1).mean()
    volume_proxy = volume_proxy.where(volume_proxy > 0.0, 1.0) * 100000.0

    if has_observed_volume:
        positive_volume = volume.where(volume > 0.0)
        fallback_volume = positive_volume.dropna().median()
        if pd.isna(fallback_volume) or float(fallback_volume) <= 0.0:
            fallback_volume = np.nan
        volume = volume.where(volume > 0.0)
        volume = volume.fillna(fallback_volume)
        frame["HAS_VOLUME_DATA"] = 1
    else:
        frame["HAS_VOLUME_DATA"] = 0

    frame["Volume"] = volume.fillna(volume_proxy).clip(lower=1.0)
    return frame


def _annotate_daily_pivot_levels(frame):
    annotated = frame.copy()
    pivot_columns = ["PIVOT_POINT", "PIVOT_R1", "PIVOT_S1", "PIVOT_R2", "PIVOT_S2"]
    for column in pivot_columns:
        annotated[column] = np.nan

    if annotated.empty or not isinstance(annotated.index, pd.DatetimeIndex):
        return annotated

    idx = annotated.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    session_day = idx.date

    session_frame = pd.DataFrame(index=annotated.index)
    session_frame["session_day"] = session_day
    session_frame["High"] = pd.to_numeric(annotated["High"], errors="coerce")
    session_frame["Low"] = pd.to_numeric(annotated["Low"], errors="coerce")
    session_frame["Close"] = pd.to_numeric(annotated["Close"], errors="coerce")
    daily = session_frame.groupby("session_day").agg({"High": "max", "Low": "min", "Close": "last"})
    previous_daily = daily.shift(1)

    pivot_point_map = {}
    pivot_r1_map = {}
    pivot_s1_map = {}
    pivot_r2_map = {}
    pivot_s2_map = {}
    for day, values in previous_daily.iterrows():
        high_value = _optional_float(values.get("High"))
        low_value = _optional_float(values.get("Low"))
        close_value = _optional_float(values.get("Close"))
        if high_value is None or low_value is None or close_value is None:
            continue
        pivot_point = (high_value + low_value + close_value) / 3.0
        day_range = high_value - low_value
        pivot_point_map[day] = pivot_point
        pivot_r1_map[day] = (2.0 * pivot_point) - low_value
        pivot_s1_map[day] = (2.0 * pivot_point) - high_value
        pivot_r2_map[day] = pivot_point + day_range
        pivot_s2_map[day] = pivot_point - day_range

    annotated["PIVOT_POINT"] = [pivot_point_map.get(day) for day in session_day]
    annotated["PIVOT_R1"] = [pivot_r1_map.get(day) for day in session_day]
    annotated["PIVOT_S1"] = [pivot_s1_map.get(day) for day in session_day]
    annotated["PIVOT_R2"] = [pivot_r2_map.get(day) for day in session_day]
    annotated["PIVOT_S2"] = [pivot_s2_map.get(day) for day in session_day]
    return annotated


def _annotate_round_number_levels(frame):
    annotated = frame.copy()
    round_columns = [
        "ROUND_NUMBER_STEP",
        "ROUND_NUMBER_SUPPORT",
        "ROUND_NUMBER_RESISTANCE",
        "MAJOR_ROUND_NUMBER_SUPPORT",
        "MAJOR_ROUND_NUMBER_RESISTANCE",
        "ROUND_SUPPORT_DISTANCE_PCT",
        "ROUND_RESISTANCE_DISTANCE_PCT",
    ]
    for column in round_columns:
        annotated[column] = np.nan

    if annotated.empty:
        return annotated

    close_series = pd.to_numeric(annotated["Close"], errors="coerce")

    def _levels_for_price(price):
        numeric_price = _optional_float(price)
        if numeric_price is None or numeric_price <= 0.0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        step = _round_number_step(numeric_price)
        major_step = step * 2.0
        support = math.floor(numeric_price / step) * step
        resistance = math.ceil(numeric_price / step) * step
        major_support = math.floor(numeric_price / major_step) * major_step
        major_resistance = math.ceil(numeric_price / major_step) * major_step
        return (step, support, resistance, major_support, major_resistance)

    levels = close_series.apply(_levels_for_price)
    level_frame = pd.DataFrame(
        levels.tolist(),
        index=annotated.index,
        columns=[
            "ROUND_NUMBER_STEP",
            "ROUND_NUMBER_SUPPORT",
            "ROUND_NUMBER_RESISTANCE",
            "MAJOR_ROUND_NUMBER_SUPPORT",
            "MAJOR_ROUND_NUMBER_RESISTANCE",
        ],
    )
    for column in level_frame.columns:
        annotated[column] = level_frame[column]

    close_safe = close_series.replace(0, np.nan)
    annotated["ROUND_SUPPORT_DISTANCE_PCT"] = (
        (close_series - annotated["ROUND_NUMBER_SUPPORT"]) / close_safe * 100.0
    ).replace([np.inf, -np.inf], np.nan)
    annotated["ROUND_RESISTANCE_DISTANCE_PCT"] = (
        (annotated["ROUND_NUMBER_RESISTANCE"] - close_series) / close_safe * 100.0
    ).replace([np.inf, -np.inf], np.nan)
    return annotated


def _annotate_fair_value_gaps(frame):
    annotated = frame.copy()
    columns = [
        "BULLISH_FVG_LOW",
        "BULLISH_FVG_HIGH",
        "BULLISH_FVG_DISTANCE_PCT",
        "BEARISH_FVG_LOW",
        "BEARISH_FVG_HIGH",
        "BEARISH_FVG_DISTANCE_PCT",
    ]
    for column in columns:
        annotated[column] = np.nan
    annotated["IN_BULLISH_FVG"] = 0
    annotated["IN_BEARISH_FVG"] = 0

    if annotated.empty or len(annotated) < 3:
        return annotated

    highs = pd.to_numeric(annotated["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(annotated["Low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(annotated["Close"], errors="coerce").to_numpy(dtype=float)
    if "ATR_14" in annotated.columns:
        atr_values = pd.to_numeric(annotated["ATR_14"], errors="coerce").to_numpy(dtype=float)
    else:
        atr_values = np.full(len(annotated), np.nan, dtype=float)

    bullish_low = np.full(len(annotated), np.nan, dtype=float)
    bullish_high = np.full(len(annotated), np.nan, dtype=float)
    bullish_distance = np.full(len(annotated), np.nan, dtype=float)
    bearish_low = np.full(len(annotated), np.nan, dtype=float)
    bearish_high = np.full(len(annotated), np.nan, dtype=float)
    bearish_distance = np.full(len(annotated), np.nan, dtype=float)
    in_bullish = np.zeros(len(annotated), dtype=int)
    in_bearish = np.zeros(len(annotated), dtype=int)

    active_bullish = []
    active_bearish = []

    for index in range(len(annotated)):
        current_low = lows[index]
        current_high = highs[index]
        current_close = closes[index]

        if np.isfinite(current_low):
            active_bullish = [zone for zone in active_bullish if current_low > zone["low"]]
        if np.isfinite(current_high):
            active_bearish = [zone for zone in active_bearish if current_high < zone["high"]]

        if index >= 2 and np.isfinite(current_close):
            gap_threshold = max(
                max(_safe_float(atr_values[index], 0.0), 0.0) * 0.05,
                abs(current_close) * 0.00015,
            )
            previous_two_high = highs[index - 2]
            previous_two_low = lows[index - 2]

            if np.isfinite(current_low) and np.isfinite(previous_two_high):
                bullish_gap = current_low - previous_two_high
                if bullish_gap > gap_threshold:
                    active_bullish.append({"low": float(previous_two_high), "high": float(current_low)})

            if np.isfinite(current_high) and np.isfinite(previous_two_low):
                bearish_gap = previous_two_low - current_high
                if bearish_gap > gap_threshold:
                    active_bearish.append({"low": float(current_high), "high": float(previous_two_low)})

        if not np.isfinite(current_close) or current_close <= 0.0:
            continue

        bullish_zone = None
        bullish_inside = [zone for zone in active_bullish if zone["low"] <= current_close <= zone["high"]]
        if bullish_inside:
            bullish_zone = max(bullish_inside, key=lambda zone: zone["high"])
        else:
            bullish_below = [zone for zone in active_bullish if zone["high"] <= current_close]
            if bullish_below:
                bullish_zone = max(bullish_below, key=lambda zone: zone["high"])

        if bullish_zone:
            bullish_low[index] = bullish_zone["low"]
            bullish_high[index] = bullish_zone["high"]
            in_bullish[index] = int(bullish_zone["low"] <= current_close <= bullish_zone["high"])
            if in_bullish[index]:
                bullish_distance[index] = 0.0
            else:
                bullish_distance[index] = ((current_close - bullish_zone["high"]) / current_close) * 100.0

        bearish_zone = None
        bearish_inside = [zone for zone in active_bearish if zone["low"] <= current_close <= zone["high"]]
        if bearish_inside:
            bearish_zone = min(bearish_inside, key=lambda zone: zone["low"])
        else:
            bearish_above = [zone for zone in active_bearish if zone["low"] >= current_close]
            if bearish_above:
                bearish_zone = min(bearish_above, key=lambda zone: zone["low"])

        if bearish_zone:
            bearish_low[index] = bearish_zone["low"]
            bearish_high[index] = bearish_zone["high"]
            in_bearish[index] = int(bearish_zone["low"] <= current_close <= bearish_zone["high"])
            if in_bearish[index]:
                bearish_distance[index] = 0.0
            else:
                bearish_distance[index] = ((bearish_zone["low"] - current_close) / current_close) * 100.0

    annotated["BULLISH_FVG_LOW"] = bullish_low
    annotated["BULLISH_FVG_HIGH"] = bullish_high
    annotated["BULLISH_FVG_DISTANCE_PCT"] = bullish_distance
    annotated["BEARISH_FVG_LOW"] = bearish_low
    annotated["BEARISH_FVG_HIGH"] = bearish_high
    annotated["BEARISH_FVG_DISTANCE_PCT"] = bearish_distance
    annotated["IN_BULLISH_FVG"] = in_bullish
    annotated["IN_BEARISH_FVG"] = in_bearish
    return annotated


def _annotate_range_zones(frame, window=12):
    annotated = frame.copy()
    columns = [
        "RANGE_ZONE_LOW",
        "RANGE_ZONE_HIGH",
        "RANGE_ZONE_POSITION",
        "RANGE_ZONE_WIDTH_PCT",
        "RANGE_ZONE_TOUCH_SCORE",
    ]
    for column in columns:
        annotated[column] = np.nan
    annotated["RANGE_ZONE_ACTIVE"] = 0
    annotated["RANGE_ZONE_BREAK"] = 0
    annotated["IN_RANGE_ZONE"] = 0

    if annotated.empty:
        return annotated

    range_high = annotated["High"].rolling(window, min_periods=6).max().shift(1)
    range_low = annotated["Low"].rolling(window, min_periods=6).min().shift(1)
    range_width = range_high - range_low
    atr_reference = pd.to_numeric(annotated.get("ATR_14"), errors="coerce")
    compression_ratio = pd.to_numeric(annotated.get("COMPRESSION_RATIO"), errors="coerce")
    squeeze_on = pd.to_numeric(annotated.get("SQUEEZE_ON"), errors="coerce").fillna(0.0)
    adx_values = pd.to_numeric(annotated.get("ADX_14"), errors="coerce")
    width_atr = range_width / atr_reference.replace(0, np.nan)
    tolerance = pd.Series(
        np.maximum(range_width.fillna(0.0) * 0.18, atr_reference.fillna(0.0) * 0.18),
        index=annotated.index,
    )
    lower_touch = (annotated["Low"].shift(1) <= (range_low + tolerance)).fillna(False)
    upper_touch = (annotated["High"].shift(1) >= (range_high - tolerance)).fillna(False)
    lower_touch_count = lower_touch.rolling(window, min_periods=6).sum()
    upper_touch_count = upper_touch.rolling(window, min_periods=6).sum()
    active_zone = (
        range_width.notna()
        & (range_width > 0.0)
        & width_atr.between(0.9, 4.5)
        & (lower_touch_count >= 2)
        & (upper_touch_count >= 2)
        & ((compression_ratio <= 0.90) | (squeeze_on >= 1.0) | (adx_values <= 22.0))
    )
    breakout_buffer = pd.Series(
        np.maximum(atr_reference.fillna(0.0) * 0.08, annotated["Close"].abs().fillna(0.0) * 0.0002),
        index=annotated.index,
    )
    prior_active_zone = active_zone.shift(1).fillna(False)
    break_up = prior_active_zone & (annotated["Close"] > (range_high + breakout_buffer))
    break_down = prior_active_zone & (annotated["Close"] < (range_low - breakout_buffer))
    zone_context = active_zone | prior_active_zone | break_up | break_down
    zone_position = ((annotated["Close"] - range_low) / range_width.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    annotated["RANGE_ZONE_ACTIVE"] = active_zone.astype(int)
    annotated.loc[break_up, "RANGE_ZONE_BREAK"] = 1
    annotated.loc[break_down, "RANGE_ZONE_BREAK"] = -1
    annotated["RANGE_ZONE_LOW"] = range_low.where(zone_context)
    annotated["RANGE_ZONE_HIGH"] = range_high.where(zone_context)
    annotated["RANGE_ZONE_POSITION"] = zone_position.where(zone_context)
    annotated["RANGE_ZONE_WIDTH_PCT"] = (
        range_width / annotated["Close"].replace(0, np.nan) * 100.0
    ).where(zone_context)
    annotated["RANGE_ZONE_TOUCH_SCORE"] = pd.concat([lower_touch_count, upper_touch_count], axis=1).min(axis=1).where(zone_context)
    annotated["IN_RANGE_ZONE"] = (zone_context & zone_position.between(0.0, 1.0, inclusive="both")).astype(int)
    return annotated


def _annotate_rsi_divergence(frame, rsi_col="RSI_14", pivot_max_spacing=72):
    lows = pd.to_numeric(frame.get("Low"), errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(frame.get("High"), errors="coerce").to_numpy(dtype=float)
    rsi_values = pd.to_numeric(frame.get(rsi_col), errors="coerce").to_numpy(dtype=float)

    bullish_strength = np.zeros(len(frame), dtype=float)
    bearish_strength = np.zeros(len(frame), dtype=float)
    low_pivots = []
    high_pivots = []

    for i in range(2, len(frame)):
        pivot_idx = i - 1

        if np.isfinite(lows[i - 2]) and np.isfinite(lows[pivot_idx]) and np.isfinite(lows[i]):
            if lows[i - 2] > lows[pivot_idx] and lows[i] > lows[pivot_idx]:
                low_pivots.append((pivot_idx, lows[pivot_idx], rsi_values[pivot_idx]))
                if len(low_pivots) >= 2:
                    prev_idx, prev_low, prev_rsi = low_pivots[-2]
                    curr_idx, curr_low, curr_rsi = low_pivots[-1]
                    if (
                        curr_idx - prev_idx <= max(8, int(pivot_max_spacing))
                        and np.isfinite(prev_rsi)
                        and np.isfinite(curr_rsi)
                        and curr_low < prev_low
                        and curr_rsi > (prev_rsi + 1.5)
                    ):
                        price_drop_pct = (prev_low - curr_low) / max(abs(prev_low), 1e-8)
                        rsi_lift = (curr_rsi - prev_rsi) / 10.0
                        bullish_strength[i] = max(0.0, min(3.0, (price_drop_pct * 100.0) + rsi_lift))

        if np.isfinite(highs[i - 2]) and np.isfinite(highs[pivot_idx]) and np.isfinite(highs[i]):
            if highs[i - 2] < highs[pivot_idx] and highs[i] < highs[pivot_idx]:
                high_pivots.append((pivot_idx, highs[pivot_idx], rsi_values[pivot_idx]))
                if len(high_pivots) >= 2:
                    prev_idx, prev_high, prev_rsi = high_pivots[-2]
                    curr_idx, curr_high, curr_rsi = high_pivots[-1]
                    if (
                        curr_idx - prev_idx <= max(8, int(pivot_max_spacing))
                        and np.isfinite(prev_rsi)
                        and np.isfinite(curr_rsi)
                        and curr_high > prev_high
                        and curr_rsi < (prev_rsi - 1.5)
                    ):
                        price_rise_pct = (curr_high - prev_high) / max(abs(prev_high), 1e-8)
                        rsi_drop = (prev_rsi - curr_rsi) / 10.0
                        bearish_strength[i] = max(0.0, min(3.0, (price_rise_pct * 100.0) + rsi_drop))

    bullish_series = pd.Series(bullish_strength, index=frame.index)
    bearish_series = pd.Series(bearish_strength, index=frame.index)
    frame["RSI_BULLISH_DIVERGENCE"] = (bullish_series.rolling(4, min_periods=1).max() > 0.0).astype(int)
    frame["RSI_BEARISH_DIVERGENCE"] = (bearish_series.rolling(4, min_periods=1).max() > 0.0).astype(int)
    frame["RSI_DIVERGENCE_STRENGTH"] = (
        bullish_series.rolling(3, min_periods=1).mean() - bearish_series.rolling(3, min_periods=1).mean()
    ).fillna(0.0)
    return frame


def prepare_historical_features(df, params=None):
    strategy_params = normalize_strategy_params(params)
    frame = df.copy().sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame = _ensure_volume_column(frame)
    ema_short = int(strategy_params.get("ema_short", 20))
    ema_long = int(strategy_params.get("ema_long", 50))
    rsi_window = int(strategy_params.get("rsi_window", 14))
    atr_window = int(strategy_params.get("atr_window", 14))
    adx_window = int(strategy_params.get("adx_window", 14))
    cmf_window = int(strategy_params.get("cmf_window", 14))

    frame["EMA_20"] = ta.trend.EMAIndicator(frame["Close"], window=ema_short).ema_indicator()
    frame["EMA_50"] = ta.trend.EMAIndicator(frame["Close"], window=ema_long).ema_indicator()
    frame["RSI_14"] = ta.momentum.RSIIndicator(frame["Close"], window=rsi_window).rsi()
    frame["ATR_14"] = ta.volatility.AverageTrueRange(
        frame["High"], frame["Low"], frame["Close"], window=atr_window
    ).average_true_range()
    frame["ADX_14"] = ta.trend.ADXIndicator(
        frame["High"], frame["Low"], frame["Close"], window=adx_window
    ).adx()
    frame["OBV"] = ta.volume.OnBalanceVolumeIndicator(frame["Close"], frame["Volume"]).on_balance_volume()
    frame["OBV_PREV"] = frame["OBV"].shift(1)
    frame["CMF_14"] = ta.volume.ChaikinMoneyFlowIndicator(
        frame["High"], frame["Low"], frame["Close"], frame["Volume"], window=cmf_window
    ).chaikin_money_flow()

    macd = ta.trend.MACD(frame["Close"], window_slow=26, window_fast=12, window_sign=9)
    frame["MACD_LINE"] = macd.macd()
    frame["MACD_SIGNAL"] = macd.macd_signal()
    frame["MACD_HIST"] = macd.macd_diff()
    frame["MACD_HIST_SLOPE"] = frame["MACD_HIST"].diff().rolling(3, min_periods=1).mean().fillna(0.0)

    volume_spike_threshold = float(strategy_params.get("volume_spike_zscore_threshold", 1.8))
    volume_mean = frame["Volume"].rolling(20, min_periods=5).mean()
    volume_std = frame["Volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
    frame["VOLUME_ZSCORE"] = ((frame["Volume"] - volume_mean) / volume_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["VOLUME_SPIKE"] = (frame["VOLUME_ZSCORE"] >= volume_spike_threshold).astype(int)
    frame = _annotate_rsi_divergence(frame, rsi_col="RSI_14")

    frame["EMA_TREND"] = _trend_series_from_close(frame["Close"], ema_short, ema_long)

    inferred_minutes = 60.0
    try:
        if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index) >= 3:
            diffs = frame.index.to_series().diff().dropna()
            if not diffs.empty:
                inferred_minutes = float(diffs.dt.total_seconds().median() / 60.0)
    except Exception:
        inferred_minutes = 60.0
    base_is_15m = inferred_minutes <= 20.0

    trend_4h = _trend_series_from_close(
        frame["Close"].resample("4h").last().dropna(),
        ema_short,
        ema_long,
    )
    trend_1h = _trend_series_from_close(
        frame["Close"].resample("1h").last().dropna(),
        ema_short,
        ema_long,
    )
    frame["H4_TREND"] = trend_4h.reindex(frame.index, method="ffill").fillna("Neutral")
    frame["H1_TREND"] = trend_1h.reindex(frame.index, method="ffill").fillna("Neutral")
    if base_is_15m:
        frame["M15_TREND"] = frame["EMA_TREND"]
    else:
        frame["M15_TREND"] = frame["H1_TREND"]

    trend_map = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
    frame["ALIGNMENT_SCORE"] = (
        frame["M15_TREND"].map(trend_map).fillna(0)
        + frame["H1_TREND"].map(trend_map).fillna(0)
        + frame["H4_TREND"].map(trend_map).fillna(0)
    )
    frame["ALIGNMENT_LABEL"] = "Mixed / Low Alignment"
    frame.loc[frame["ALIGNMENT_SCORE"] >= 2, "ALIGNMENT_LABEL"] = "Strong Bullish Alignment"
    frame.loc[frame["ALIGNMENT_SCORE"] <= -2, "ALIGNMENT_LABEL"] = "Strong Bearish Alignment"

    atr_percent = (frame["ATR_14"] / frame["Close"]) * 100
    adx_trending_threshold = float(strategy_params.get("adx_trending_threshold", 22))
    adx_weak_trend_threshold = float(strategy_params.get("adx_weak_trend_threshold", 18))
    atr_trending_pct_threshold = float(strategy_params.get("atr_trending_percent_threshold", 0.25))
    frame["ATR_PERCENT"] = atr_percent
    frame["MARKET_REGIME"] = "Range-Bound"
    frame.loc[
        (frame["ADX_14"] >= adx_trending_threshold) & (frame["ATR_PERCENT"] >= atr_trending_pct_threshold),
        "MARKET_REGIME",
    ] = "Trending"
    frame.loc[
        (frame["MARKET_REGIME"] == "Range-Bound") & (frame["ADX_14"] >= adx_weak_trend_threshold),
        "MARKET_REGIME",
    ] = "Weak Trend"

    cmf_buy_threshold = float(strategy_params.get("cmf_strong_buy_threshold", 0.10))
    cmf_sell_threshold = float(strategy_params.get("cmf_strong_sell_threshold", -0.10))
    frame["VOLUME_SIGNAL"] = "Neutral"
    frame.loc[(frame["CMF_14"] > cmf_buy_threshold) & (frame["OBV"] > frame["OBV_PREV"]), "VOLUME_SIGNAL"] = "Strong Buying Pressure (Accumulation)"
    frame.loc[(frame["CMF_14"] < cmf_sell_threshold) & (frame["OBV"] <= frame["OBV_PREV"]), "VOLUME_SIGNAL"] = "Strong Selling Pressure (Distribution)"
    frame.loc[(frame["VOLUME_SIGNAL"] == "Neutral") & (frame["CMF_14"] > 0), "VOLUME_SIGNAL"] = "Slight Buying Bias"
    frame.loc[(frame["VOLUME_SIGNAL"] == "Neutral") & (frame["CMF_14"] < 0), "VOLUME_SIGNAL"] = "Slight Selling Bias"

    frame["RECENT_SWING_HIGH_24"] = frame["High"].rolling(24, min_periods=6).max().shift(1)
    frame["RECENT_SWING_LOW_24"] = frame["Low"].rolling(24, min_periods=6).min().shift(1)
    frame["RECENT_SWING_HIGH_96"] = frame["High"].rolling(96, min_periods=24).max().shift(1)
    frame["RECENT_SWING_LOW_96"] = frame["Low"].rolling(96, min_periods=24).min().shift(1)

    frame = annotate_price_action(frame)
    frame = annotate_event_regime_features(frame)
    frame = _annotate_daily_pivot_levels(frame)
    smooth_adx_center = float(strategy_params.get("smooth_adx_center", 20.0))
    smooth_adx_scale = float(strategy_params.get("smooth_adx_scale", 2.6))
    smooth_atr_center = float(strategy_params.get("smooth_atr_percent_center", 0.24))
    smooth_atr_scale = float(strategy_params.get("smooth_atr_percent_scale", 0.045))
    alignment_strength = frame["ALIGNMENT_SCORE"].abs().clip(lower=0, upper=3) / 3.0
    trend_probability = (_sigmoid_series(frame["ADX_14"].fillna(0.0), center=smooth_adx_center, scale=smooth_adx_scale) * 0.72) + (alignment_strength * 0.28)
    expansion_probability = (
        _sigmoid_series(frame["ATR_PERCENT"].fillna(0.0), center=smooth_atr_center, scale=smooth_atr_scale) * 0.68
        + _sigmoid_series(frame["ATR_EXPANSION_RATIO"].fillna(1.0), center=1.05, scale=0.12) * 0.22
        + (frame["REALIZED_VOL_PERCENTILE"].fillna(50.0) / 100.0) * 0.10
    )
    frame["TREND_PROBABILITY"] = trend_probability.clip(0.0, 1.0)
    frame["EXPANSION_PROBABILITY"] = expansion_probability.clip(0.0, 1.0)
    frame["CHOP_PROBABILITY"] = (1.0 - (frame["TREND_PROBABILITY"] * 0.66) - (frame["EXPANSION_PROBABILITY"] * 0.34)).clip(0.0, 1.0)
    return frame


def build_ta_payload_from_row(
    row,
    params=None,
    regime_memory=None,
    event_risk=None,
    cross_asset_context=None,
    support_resistance=None,
):
    strategy_params = normalize_strategy_params(params)
    bar_dt = _coerce_utc_datetime(getattr(row, "name", None))
    bar_ts = bar_dt.isoformat() if bar_dt else None
    bar_session_context = _build_session_context_from_datetime(bar_dt)
    current_session_context = _build_session_context_from_datetime(datetime.now(timezone.utc))
    support_resistance = support_resistance if isinstance(support_resistance, dict) else _build_support_resistance_from_row(row)
    event_risk_context = event_risk if isinstance(event_risk, dict) else {
        "active": bool(row.get("EVENT_ACTIVE", 0)),
    }
    return {
        "bar_timestamp_utc": bar_ts,
        "current_price": round(float(row["Close"]), 2),
        "ema_trend": row.get("EMA_TREND", "Neutral"),
        "ema_20": round(float(row.get("EMA_20", 0.0)), 2),
        "ema_50": round(float(row.get("EMA_50", 0.0)), 2),
        "volatility_regime": {
            "market_regime": row.get("MARKET_REGIME", "Range-Bound"),
            "adx_14": round(float(row.get("ADX_14", 0.0)), 2),
            "atr_14": round(float(row.get("ATR_14", 0.0)), 2),
            "atr_percent": round(float(row.get("ATR_PERCENT", 0.0)), 3),
        },
        "multi_timeframe": {
            "m15_trend": row.get("M15_TREND", "Neutral"),
            "h1_trend": row.get("H1_TREND", row.get("EMA_TREND", "Neutral")),
            "h4_trend": row.get("H4_TREND", "Neutral"),
            "alignment_score": int(row.get("ALIGNMENT_SCORE", 0)),
            "alignment_label": row.get("ALIGNMENT_LABEL", "Mixed / Low Alignment"),
            "sources": {
                "m15": "derived_proxy_or_base",
                "h1": "derived_resample",
                "h4": "derived_resample",
            },
        },
        "price_action": {
            "structure": row.get("PA_STRUCTURE", "Consolidating"),
            "latest_candle_pattern": row.get("CANDLE_PATTERN", "None"),
            "basis": "latest_bar_utc",
            "bar_timestamp_utc": bar_ts,
        },
        "support_resistance": support_resistance,
        "volume_analysis": {
            "cmf_14": round(float(row.get("CMF_14", 0.0)), 4),
            "obv_trend": "Rising" if row.get("OBV", 0.0) > row.get("OBV_PREV", row.get("OBV", 0.0)) else "Falling",
            "overall_volume_signal": row.get("VOLUME_SIGNAL", "Neutral"),
        },
        "momentum_features": {
            "macdLine": round(_safe_float(row.get("MACD_LINE"), 0.0), 6),
            "macdSignal": round(_safe_float(row.get("MACD_SIGNAL"), 0.0), 6),
            "macdHistogram": round(_safe_float(row.get("MACD_HIST"), 0.0), 6),
            "macdHistogramSlope": round(_safe_float(row.get("MACD_HIST_SLOPE"), 0.0), 6),
            "rsiBullishDivergence": int(_safe_float(row.get("RSI_BULLISH_DIVERGENCE"), 0.0)),
            "rsiBearishDivergence": int(_safe_float(row.get("RSI_BEARISH_DIVERGENCE"), 0.0)),
            "rsiDivergenceStrength": round(_safe_float(row.get("RSI_DIVERGENCE_STRENGTH"), 0.0), 4),
            "volumeZScore": round(_safe_float(row.get("VOLUME_ZSCORE"), 0.0), 4),
            "volumeSpike": int(_safe_float(row.get("VOLUME_SPIKE"), 0.0)),
        },
        "session_context": {
            "label": bar_session_context.get("label", "Off"),
            "hour": int(bar_session_context.get("hour", 0)),
            "minute": int(bar_session_context.get("minute", 0)),
            "quality": round(_safe_float(bar_session_context.get("quality"), 0.46), 4),
            "isSydneyOpen": bool(bar_session_context.get("isSydneyOpen", False)),
            "isAsiaOpen": bool(bar_session_context.get("isAsiaOpen", False)),
            "isFrankfurtOpen": bool(bar_session_context.get("isFrankfurtOpen", False)),
            "isLondonOpen": bool(bar_session_context.get("isLondonOpen", False)),
            "isNewYorkOpen": bool(bar_session_context.get("isNewYorkOpen", False)),
            "isComexOpen": bool(bar_session_context.get("isComexOpen", False)),
            "isFixWindow": bool(bar_session_context.get("isFixWindow", False)),
            "isOverlap": bool(bar_session_context.get("isOverlap", False)),
            "timezone": "UTC",
            "basis": "latest_bar_utc",
            "barTimestampUtc": bar_ts,
            "barTimeDisplayUtc": bar_session_context.get("timeDisplayUtc"),
            "currentLabel": current_session_context.get("label", "Off"),
            "currentTimestampUtc": current_session_context.get("timestampUtc"),
            "currentTimeDisplayUtc": current_session_context.get("timeDisplayUtc"),
            "bar_session": bar_session_context,
            "current_session": current_session_context,
        },
        "structure_context": {
            "openingRangeBreak": _coerce_signal_state(row.get("OPENING_RANGE_BREAK")),
            "sweepReclaimSignal": _coerce_signal_state(row.get("SWEEP_RECLAIM_SIGNAL")),
            "sweepReclaimQuality": round(_safe_float(row.get("SWEEP_RECLAIM_QUALITY"), 0.0), 4),
            "sessionVwap": round(_safe_float(row.get("SESSION_VWAP"), 0.0), 2),
            "distSessionVwapPct": round(_safe_float(row.get("DIST_SESSION_VWAP_PCT"), 0.0), 4),
            "recentSwingHigh": round(_safe_float(row.get("RECENT_SWING_HIGH_24"), 0.0), 2),
            "recentSwingLow": round(_safe_float(row.get("RECENT_SWING_LOW_24"), 0.0), 2),
            "pivotPoint": _round_or_none(row.get("PIVOT_POINT")),
            "pivotResistance1": _round_or_none(row.get("PIVOT_R1")),
            "pivotSupport1": _round_or_none(row.get("PIVOT_S1")),
            "pivotResistance2": _round_or_none(row.get("PIVOT_R2")),
            "pivotSupport2": _round_or_none(row.get("PIVOT_S2")),
        },
        "volatility_features": {
            "realizedVol8": round(_safe_float(row.get("REALIZED_VOL_8"), 0.0), 6),
            "realizedVol32": round(_safe_float(row.get("REALIZED_VOL_32"), 0.0), 6),
            "realizedVolRatio": round(_safe_float(row.get("REALIZED_VOL_RATIO"), 0.0), 4),
            "realizedVolPercentile": round(_safe_float(row.get("REALIZED_VOL_PERCENTILE"), 50.0), 2),
            "atrPercentile": round(_safe_float(row.get("ATR_PERCENTILE"), 50.0), 2),
            "trendFollowThrough": round(_safe_float(row.get("TREND_FOLLOW_THROUGH"), 1.0), 4),
        },
        "market_regime_scores": {
            "trend_probability": round(_safe_float(row.get("TREND_PROBABILITY"), 0.0), 4),
            "expansion_probability": round(_safe_float(row.get("EXPANSION_PROBABILITY"), 0.0), 4),
            "chop_probability": round(_safe_float(row.get("CHOP_PROBABILITY"), 0.0), 4),
        },
        "active_strategy_params": strategy_params,
        "cross_asset_context": cross_asset_context if isinstance(cross_asset_context, dict) else {},
        "event_risk": event_risk_context,
        "event_regime": compute_event_regime_snapshot(
            row,
            trend=row.get("EMA_TREND", "Neutral"),
            alignment_label=row.get("ALIGNMENT_LABEL", "Mixed / Low Alignment"),
            market_structure=row.get("PA_STRUCTURE", "Consolidating"),
            candle_pattern=row.get("CANDLE_PATTERN", "None"),
            event_risk=event_risk_context,
            cross_asset_context=cross_asset_context,
            expansion_watch_threshold=float(strategy_params.get("expansion_watch_threshold", 48.0)),
            high_breakout_threshold=float(strategy_params.get("high_breakout_threshold", 64.0)),
            directional_expansion_threshold=float(strategy_params.get("directional_expansion_threshold", 78.0)),
            previous_state=regime_memory if isinstance(regime_memory, dict) else None,
            warning_upshift_buffer=float(strategy_params.get("warning_upshift_buffer", 3.5)),
            warning_downshift_buffer=float(strategy_params.get("warning_downshift_buffer", 7.0)),
            min_warning_dwell_bars=int(strategy_params.get("warning_min_dwell_bars", 5)),
            breakout_bias_deadband=float(strategy_params.get("breakout_bias_deadband", 0.9)),
            breakout_bias_hold_bars=int(strategy_params.get("breakout_bias_hold_bars", 4)),
        ),
        "_regime_memory": regime_memory if isinstance(regime_memory, dict) else {},
    }
