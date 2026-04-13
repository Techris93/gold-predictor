import pandas as pd
import ta
from pathlib import Path
import json
from datetime import datetime, timezone

from tools.event_regime import annotate_event_regime_features, compute_event_regime_snapshot
from tools.price_action import annotate_price_action, extract_price_action_feature_hits


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
    "sr_reaction_weight": 1.6,
    "sr_proximity_weight": 0.5,
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
    "estimated_avg_win_pct": 0.55,
    "estimated_avg_loss_pct": 0.42,
    "transaction_cost_pct": 0.05,
    "rr_signal_sl_pips": 100,
    "rr_signal_tp_pips": 200,
    "rr_signal_target_move_pips": 200,
    "rr_signal_pip_size": 0.01,
    "rr_signal_min_confidence": 72.0,
    "rr_signal_min_tradeability_score": 58.0,
    "rr_signal_min_move_probability": 0.52,
    "rr_signal_min_expected_edge_pct": 0.05,
    "rr_signal_allow_b_grade": 1,
    "rr_signal_b_min_move_probability": 0.64,
    "rr_signal_b_min_session_quality": 0.78,
    "rr_signal_b_require_active_state": 1,
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
}

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIDENCE_CALIBRATION_FILE = BASE_DIR / "tools" / "reports" / "confidence_calibration.json"
REGIME_PARAMS_FILE = BASE_DIR / "config" / "regime_params.json"


def normalize_strategy_params(params=None):
    merged = DEFAULT_STRATEGY_PARAMS.copy()
    if isinstance(params, dict):
        merged.update(params)
    return merged


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


def _build_forecast_state(regime_bucket, regime_state, confidence, directional_bias):
    regime_state = regime_state or {}
    breakout_bias = str(regime_state.get("breakout_bias") or "Neutral")
    warning_ladder = str(regime_state.get("warning_ladder") or "Normal")
    event_regime = str(regime_state.get("event_regime") or "normal")
    expansion_30m = float(regime_state.get("expansion_probability_30m") or 0.0)
    expansion_60m = float(regime_state.get("expansion_probability_60m") or 0.0)
    forecast_confidence = round(
        min(
            95.0,
            max(
                50.0,
                (float(confidence or 50) * 0.55)
                + (expansion_60m * 0.30)
                + (expansion_30m * 0.15),
            ),
        )
    )
    return {
        "regimeBucket": regime_bucket,
        "moveProbability30m": round(expansion_30m, 2),
        "moveProbability60m": round(expansion_60m, 2),
        "directionalBias": directional_bias,
        "breakoutBias": breakout_bias,
        "eventRegime": event_regime,
        "warningLadder": warning_ladder,
        "expectedRangeExpansion": regime_state.get("expected_range_expansion"),
        "crossAssetBias": regime_state.get("cross_asset_bias", "Neutral"),
        "minutesToNextEvent": regime_state.get("minutes_to_next_event"),
        "forecastConfidence": forecast_confidence,
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
):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    regime_state = regime_state if isinstance(regime_state, dict) else {}
    meta_scores = meta_scores if isinstance(meta_scores, dict) else {}

    sl_pips = max(1.0, float(strategy_params.get("rr_signal_sl_pips", 100) or 100))
    tp_pips = max(1.0, float(strategy_params.get("rr_signal_tp_pips", 200) or 200))
    target_move_pips = max(1.0, float(strategy_params.get("rr_signal_target_move_pips", 200) or 200))
    pip_size = max(0.0001, float(strategy_params.get("rr_signal_pip_size", 0.01) or 0.01))
    min_confidence = float(strategy_params.get("rr_signal_min_confidence", 80.0) or 80.0)
    min_tradeability_score = float(strategy_params.get("rr_signal_min_tradeability_score", 68.0) or 68.0)
    min_move_probability = float(strategy_params.get("rr_signal_min_move_probability", 0.64) or 0.64)
    min_expected_edge = float(strategy_params.get("rr_signal_min_expected_edge_pct", 0.12) or 0.12)
    allow_b_grade = bool(int(strategy_params.get("rr_signal_allow_b_grade", 1) or 0))
    b_min_move_probability = float(strategy_params.get("rr_signal_b_min_move_probability", 0.64) or 0.64)
    b_min_session_quality = float(strategy_params.get("rr_signal_b_min_session_quality", 0.78) or 0.78)
    b_require_active_state = bool(int(strategy_params.get("rr_signal_b_require_active_state", 1) or 0))
    allow_soft_no_trade = bool(int(strategy_params.get("rr_signal_allow_soft_no_trade", 1) or 0))
    soft_no_trade_terms = strategy_params.get("rr_signal_soft_no_trade_terms", [])
    soft_no_trade_terms = soft_no_trade_terms if isinstance(soft_no_trade_terms, list) else []

    current_price = float(ta_data.get("current_price") or 0.0)
    atr_percent = float(((ta_data.get("volatility_regime") or {}).get("atr_percent")) or 0.0)
    mtf = ta_data.get("multi_timeframe", {}) if isinstance(ta_data.get("multi_timeframe"), dict) else {}
    m15_trend = str(mtf.get("m15_trend") or "Neutral")
    h1_trend = str(mtf.get("h1_trend") or "Neutral")
    h4_trend = str(mtf.get("h4_trend") or "Neutral")
    support_resistance = ta_data.get("support_resistance", {}) if isinstance(ta_data.get("support_resistance"), dict) else {}
    event_risk = ta_data.get("event_risk", {}) if isinstance(ta_data.get("event_risk"), dict) else {}
    expansion_30m = float(regime_state.get("expansion_probability_30m") or 0.0)
    expansion_60m = float(regime_state.get("expansion_probability_60m") or 0.0)
    big_move_risk = float(regime_state.get("big_move_risk") or 0.0)
    breakout_bias = str(regime_state.get("breakout_bias") or "Neutral")
    warning_ladder = str(regime_state.get("warning_ladder") or "Normal")

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

    bar_ts = ta_data.get("bar_timestamp_utc")
    now_hour_utc = datetime.now(timezone.utc).hour
    if bar_ts:
        try:
            parsed = datetime.fromisoformat(str(bar_ts).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            now_hour_utc = parsed.astimezone(timezone.utc).hour
        except Exception:
            pass
    session_quality = 0.55
    if 7 <= now_hour_utc <= 16:
        session_quality = 0.85  # London + overlap
    elif 12 <= now_hour_utc <= 20:
        session_quality = 0.78  # New York window
    elif 0 <= now_hour_utc <= 5:
        session_quality = 0.45

    trend_quality = max(0.0, min(1.0, (float(direction_score) - 5.0) / 5.0))
    confidence_quality = max(0.0, min(1.0, (float(confidence) - 60.0) / 35.0))
    tradeability_quality = max(0.0, min(1.0, float(tradeability_score) / 100.0))
    stability_quality = max(0.0, min(1.0, float(stability_score) / 100.0))
    alignment_quality = 1.0 if str(alignment_label).startswith("Strong") else 0.45
    expansion_quality = max(0.0, min(1.0, (expansion_60m * 0.7 + expansion_30m * 0.3) / 100.0))
    event_quality = max(0.0, min(1.0, big_move_risk / 100.0))
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
    )
    move_probability -= (fakeout_penalty * 0.18) + (exit_penalty * 0.10)
    if mtf_conflict_count >= 2:
        move_probability -= 0.12
    elif mtf_conflict_count == 1:
        move_probability -= 0.05
    move_probability = max(0.0, min(1.0, move_probability))

    atr_move_pips = 0.0
    if current_price > 0 and atr_percent > 0:
        atr_move_pips = ((current_price * (atr_percent / 100.0)) / pip_size)
    projected_move_pips = atr_move_pips * (0.75 + (expansion_60m / 100.0) + (trend_quality * 0.55))
    projected_move_pips = max(0.0, projected_move_pips)

    tier = "watch"
    grade = "Watchlist"
    if quant_score >= 88 and move_probability >= 0.72:
        tier = "a_plus"
        grade = "A+ (Quant)"
    elif quant_score >= 80 and move_probability >= 0.64:
        tier = "a"
        grade = "A (High Accuracy)"
    elif quant_score >= 72 and move_probability >= 0.56:
        tier = "b"
        grade = "B (Qualified)"
    else:
        tier = "c"
        grade = "C (Low Confidence)"

    blockers = []
    if direction not in {"Bullish", "Bearish"}:
        blockers.append("No directional bias")
    if float(confidence) < min_confidence:
        blockers.append(f"Confidence below {int(min_confidence)}%")
    if float(tradeability_score) < min_tradeability_score:
        blockers.append(f"Tradeability below {round(min_tradeability_score, 1)}")
    if move_probability < min_move_probability:
        blockers.append(f"200-pip probability below {round(min_move_probability * 100)}%")
    if projected_move_pips < target_move_pips:
        blockers.append(f"Projected move below {int(target_move_pips)} pips")
    if float(expected_edge_pct) < min_expected_edge:
        blockers.append(f"Expected edge below {round(min_expected_edge, 2)}")
    if action_state not in {"LONG_ACTIVE", "SHORT_ACTIVE", "SETUP_LONG", "SETUP_SHORT"}:
        blockers.append("Execution state not directional")
    if mtf_directional_matches < 2:
        blockers.append("Insufficient multi-timeframe directional agreement")
    if no_trade_reasons:
        reason = str(no_trade_reasons[0])
        soft_match = any(term in reason for term in soft_no_trade_terms)
        if not (allow_soft_no_trade and soft_match and tier in {"a_plus", "a"}):
            blockers.append(reason)

    if event_risk.get("active") and tier in {"b", "c"}:
        blockers.append("Event-risk window is active")
    if tier == "b":
        if mtf_directional_matches < 3:
            blockers.append("B-grade requires full multi-timeframe alignment")
        if move_probability < max(min_move_probability, b_min_move_probability):
            blockers.append("B-grade move probability is too weak")
        if session_quality < b_min_session_quality:
            blockers.append("B-grade is disabled in low-liquidity session")
        if b_require_active_state and action_state not in {"LONG_ACTIVE", "SHORT_ACTIVE"}:
            blockers.append("B-grade requires active directional state")

    allowed_grades = {"A+ (Quant)", "A (High Accuracy)"}
    if allow_b_grade:
        allowed_grades.add("B (Qualified)")
    send_signal = not blockers and grade in allowed_grades
    status = "ready" if send_signal else ("arming" if grade.startswith("A") and direction in {"Bullish", "Bearish"} else "standby")

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
    entry_price = round(current_price, 2) if current_price > 0 else None
    stop_loss = None
    take_profit = None
    if current_price > 0 and direction in {"Bullish", "Bearish"}:
        if direction == "Bullish":
            stop_loss = round(current_price - sl_distance, 2)
            take_profit = round(current_price + tp_distance, 2)
        else:
            stop_loss = round(current_price + sl_distance, 2)
            take_profit = round(current_price - tp_distance, 2)

    status_text = "Stand by until higher-quality directional conditions form."
    if status == "arming":
        status_text = "Directional setup detected; waiting for full 200-pip probability confirmation."
    elif status == "ready":
        status_text = "High-accuracy RR 1:2 signal is ready for alerting."

    partial_tp_pips = float(strategy_params.get("rr_signal_partial_take_profit_pips", sl_pips) or sl_pips)
    move_sl_to_be = bool(int(strategy_params.get("rr_signal_move_sl_to_be_after_partial", 1) or 0))

    return {
        "enabled": send_signal,
        "status": status,
        "statusText": status_text,
        "tier": tier,
        "grade": grade,
        "quantScore": round(quant_score, 2),
        "direction": direction,
        "mtfAgreement": mtf_directional_matches,
        "mtfConflict": mtf_conflict_count,
        "targetMovePips": round(target_move_pips, 1),
        "projectedMovePips": round(projected_move_pips, 1),
        "moveProbability": round(move_probability, 4),
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

    stable_event = raw_event_regime
    if stable_warning == "Normal":
        stable_event = "normal"
    elif stable_warning in {"Expansion Watch", "High Breakout Risk", "Directional Expansion Likely"}:
        stable_event = "breakout_watch"
    elif stable_warning == "Active Momentum Event" and stable_event == "normal":
        stable_event = "range_expansion"
    if prev_event_regime == "breakout_watch" and stable_event == "normal" and warning_dwell < min_dwell:
        stable_event = prev_event_regime

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


def compute_trade_guidance(ta_data, confidence):
    if not isinstance(ta_data, dict):
        return {
            "sellLevel": "Weak",
            "buyLevel": "Weak",
            "exitLevel": "Low",
            "summary": "Trade guidance unavailable.",
        }

    trend = ta_data.get("ema_trend", "Neutral")
    rsi = float(ta_data.get("rsi_14", 50) or 50)
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

    if rsi < 45:
        sell_score += 1
    elif rsi > 55:
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
    trend = ta_data.get("ema_trend", "Neutral")
    volume = (ta_data.get("volume_analysis") or {}).get("overall_volume_signal", "Neutral")
    regime = (ta_data.get("volatility_regime") or {}).get("market_regime", "Range-Bound")
    adx_14 = (ta_data.get("volatility_regime") or {}).get("adx_14", 0)
    mtf = ta_data.get("multi_timeframe", {})
    alignment_score = mtf.get("alignment_score", 0)
    alignment_label = mtf.get("alignment_label", "Mixed / Low Alignment")
    pa_struct = (ta_data.get("price_action") or {}).get("structure", "Consolidating")
    pa_pattern = (ta_data.get("price_action") or {}).get("latest_candle_pattern", "None")
    feature_hits = extract_price_action_feature_hits(pa_struct, pa_pattern)
    regime_state = ta_data.get("event_regime", {}) if isinstance(ta_data.get("event_regime"), dict) else {}
    regime_memory = ta_data.get("_regime_memory", {}) if isinstance(ta_data.get("_regime_memory"), dict) else {}
    regime_state, next_regime_memory = _stabilize_runtime_regime_state(
        regime_state=regime_state,
        memory=regime_memory,
        strategy_params=strategy_params,
    )
    support_resistance = ta_data.get("support_resistance", {}) if isinstance(ta_data.get("support_resistance"), dict) else {}
    event_risk = ta_data.get("event_risk", {}) if isinstance(ta_data.get("event_risk"), dict) else {}
    rsi = ta_data.get("rsi_14", 50)
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
    rsi_extreme_weight = float(strategy_params.get("rsi_extreme_weight", 1.5))
    rsi_warning_weight = float(strategy_params.get("rsi_warning_weight", 0.6))
    rsi_warning_band = float(strategy_params.get("rsi_warning_band", 10))
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
    trigger_min_score = float(strategy_params.get("trigger_min_score", 1.3))
    no_trade_adx_threshold = float(strategy_params.get("no_trade_adx_threshold", 18))
    no_trade_confidence_cap = float(strategy_params.get("no_trade_confidence_cap", 60))
    event_risk_penalty = float(strategy_params.get("event_risk_penalty", 15.0))
    expansion_watch_threshold = float(strategy_params.get("expansion_watch_threshold", 48.0))
    high_breakout_threshold = float(strategy_params.get("high_breakout_threshold", 64.0))
    directional_expansion_threshold = float(strategy_params.get("directional_expansion_threshold", 78.0))
    event_watch_setup_weight = float(strategy_params.get("event_watch_setup_weight", 0.35))
    event_breakout_setup_weight = float(strategy_params.get("event_breakout_setup_weight", 0.7))
    event_directional_setup_weight = float(strategy_params.get("event_directional_setup_weight", 1.15))
    event_momentum_setup_weight = float(strategy_params.get("event_momentum_setup_weight", 1.55))
    event_alignment_boost = float(strategy_params.get("event_alignment_boost", 0.35))
    event_conflict_penalty = float(strategy_params.get("event_conflict_penalty", 0.85))
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

    if trend == "Bullish":
        bull_setup += trend_base_weight
    elif trend == "Bearish":
        bear_setup += trend_base_weight

    if alignment_score > 0:
        bull_setup += min(alignment_score, 3) * alignment_weight
    elif alignment_score < 0:
        bear_setup += min(abs(alignment_score), 3) * alignment_weight

    adx_trending_threshold = float(strategy_params.get("adx_trending_threshold", 22))
    if regime == "Trending" and adx_14 >= adx_trending_threshold:
        if trend == "Bullish":
            bull_setup += trend_regime_bonus
        elif trend == "Bearish":
            bear_setup += trend_regime_bonus
    elif regime == "Weak Trend":
        if trend == "Bullish":
            bull_setup += weak_trend_bonus
        elif trend == "Bearish":
            bear_setup += weak_trend_bonus

    if "Accumulation" in volume:
        bull_setup += strong_volume_weight
    elif "Buying Bias" in volume:
        bull_setup += bias_volume_weight
    if "Distribution" in volume:
        bear_setup += strong_volume_weight
    elif "Selling Bias" in volume:
        bear_setup += bias_volume_weight

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
        bull_trigger += engulfing_weight
    elif "Bearish Engulfing" in pa_pattern:
        bear_trigger += engulfing_weight
    elif "Bullish Hammer" in pa_pattern:
        bull_trigger += reversal_candle_weight
    elif "Bearish Shooting Star" in pa_pattern:
        bear_trigger += reversal_candle_weight

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

    rsi_oversold = float(strategy_params.get("rsi_oversold", 20))
    rsi_overbought = float(strategy_params.get("rsi_overbought", 70))
    bullish_rsi_warning = min(rsi_oversold + rsi_warning_band, 50)
    bearish_rsi_warning = max(rsi_overbought - rsi_warning_band, 50)
    if rsi < rsi_oversold:
        bull_trigger += rsi_extreme_weight
    elif rsi < bullish_rsi_warning:
        bull_trigger += rsi_warning_weight
    if rsi > rsi_overbought:
        bear_trigger += rsi_extreme_weight
    elif rsi > bearish_rsi_warning:
        bear_trigger += rsi_warning_weight

    sr_reaction = support_resistance.get("reaction", "None")
    support_distance_pct = support_resistance.get("support_distance_pct")
    resistance_distance_pct = support_resistance.get("resistance_distance_pct")
    if sr_reaction == "Bullish Support Rejection" or sr_reaction == "Bullish Breakout Through Resistance":
        bull_trigger += sr_reaction_weight
    elif sr_reaction == "Bearish Resistance Rejection" or sr_reaction == "Bearish Breakdown Through Support":
        bear_trigger += sr_reaction_weight

    if isinstance(support_distance_pct, (int, float)) and support_distance_pct <= 0.2:
        bull_setup += sr_proximity_weight
    if isinstance(resistance_distance_pct, (int, float)) and resistance_distance_pct <= 0.2:
        bear_setup += sr_proximity_weight

    bull_score = bull_setup + bull_trigger
    bear_score = bear_setup + bear_trigger
    score_diff = bull_score - bear_score
    score_margin = abs(score_diff)
    evidence_total = bull_score + bear_score

    direction_entry_threshold = float(strategy_params.get("direction_entry_threshold", 8.0))
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

    regime_label = "transition"
    if event_risk.get("active"):
        regime_label = "event-risk"
    elif event_regime_label in {"range_expansion", "trend_acceleration", "panic_reversal"}:
        regime_label = event_regime_label
    elif regime == "Range-Bound":
        regime_label = "range"
    elif alignment_label == "Mixed / Low Alignment" or ("Consolidating" in pa_struct and trend == "Neutral"):
        regime_label = "transition"
    elif regime == "Trending" and trend in {"Bullish", "Bearish"}:
        regime_label = "trend"
    else:
        regime_label = "unstable"

    directional_bias = "Neutral"
    if score_diff >= verdict_margin_threshold:
        directional_bias = "Bullish"
    elif score_diff <= -verdict_margin_threshold:
        directional_bias = "Bearish"

    if warning_ladder in {"Directional Expansion Likely", "Active Momentum Event"} and event_breakout_bias != "Neutral":
        if directional_bias == "Neutral":
            directional_bias = event_breakout_bias
        elif directional_bias != event_breakout_bias:
            if event_breakout_bias == "Bullish":
                bull_setup = max(0.0, bull_setup - event_conflict_penalty)
                bear_setup += event_conflict_penalty
            else:
                bear_setup = max(0.0, bear_setup - event_conflict_penalty)
                bull_setup += event_conflict_penalty
            no_trade_reasons.append("Directional event regime conflicts with the core setup.")
            bull_score = bull_setup + bull_trigger
            bear_score = bear_setup + bear_trigger
            score_diff = bull_score - bear_score
            score_margin = abs(score_diff)
            evidence_total = bull_score + bear_score
            if score_diff >= verdict_margin_threshold:
                directional_bias = "Bullish"
            elif score_diff <= -verdict_margin_threshold:
                directional_bias = "Bearish"
            else:
                directional_bias = "Neutral"

    if directional_bias == "Bullish" and bull_trigger >= trigger_min_score:
        verdict = "Bullish"
    elif directional_bias == "Bearish" and bear_trigger >= trigger_min_score:
        verdict = "Bearish"

    conflict_count = 0
    if trend == "Bullish" and ("Bearish" in pa_struct or "Bearish" in pa_pattern):
        conflict_count += 1
    if trend == "Bearish" and ("Bullish" in pa_struct or "Bullish" in pa_pattern):
        conflict_count += 1
    if alignment_label == "Mixed / Low Alignment":
        conflict_count += 1
    if regime_label in {"transition", "event-risk", "unstable"}:
        conflict_count += 1

    stability_score = 100.0
    stability_score -= conflict_count * stability_conflict_penalty
    if alignment_label == "Mixed / Low Alignment":
        stability_score -= stability_mixed_alignment_penalty
    if regime_label in {"transition", "unstable"}:
        stability_score -= stability_flip_penalty * 0.5
    if "Doji" in pa_pattern:
        stability_score -= 8.0
    if no_trade_reasons:
        stability_score -= 8.0
    stability_score = max(0.0, min(100.0, stability_score))

    regime_quality = 100.0 if regime_label == "trend" else 65.0 if regime_label == "transition" else 35.0 if regime_label == "range" else 20.0 if regime_label == "event-risk" else 30.0
    alignment_quality = 85.0 if alignment_label.startswith("Strong") else 35.0
    structure_quality = 80.0 if any(token in pa_struct for token in ["Breakout", "Breakdown", "Bullish Drift", "Bearish Drift", "Bullish Structure", "Bearish Structure"]) else 35.0
    trigger_quality = min(100.0, max(bull_trigger, bear_trigger) * 20.0)
    volume_quality = 75.0 if "Strong" in volume else 55.0 if "Bias" in volume else 40.0
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

    direction_score = round(max(bull_score, bear_score), 2)
    exit_risk_score = 0.0
    if verdict == "Bullish":
        exit_risk_score = bear_trigger + (2.0 if alignment_label == "Mixed / Low Alignment" else 0.0) + (2.0 if "Bearish" in pa_pattern else 0.0)
    elif verdict == "Bearish":
        exit_risk_score = bull_trigger + (2.0 if alignment_label == "Mixed / Low Alignment" else 0.0) + (2.0 if "Bullish" in pa_pattern else 0.0)
    else:
        exit_risk_score = 4.0 if no_trade_reasons else 2.0

    action_state = "WAIT"
    if regime_label in {"event-risk", "range", "unstable"} or tradeability_label == "Low":
        action_state = "WAIT"
    elif directional_bias == "Bullish":
        if direction_score >= direction_entry_threshold and bull_trigger >= trigger_min_score and tradeability_label == "High":
            action_state = "LONG_ACTIVE"
        elif direction_score >= direction_hold_threshold:
            action_state = "SETUP_LONG"
    elif directional_bias == "Bearish":
        if direction_score >= direction_entry_threshold and bear_trigger >= trigger_min_score and tradeability_label == "High":
            action_state = "SHORT_ACTIVE"
        elif direction_score >= direction_hold_threshold:
            action_state = "SETUP_SHORT"

    if verdict in {"Bullish", "Bearish"} and exit_risk_score >= exit_risk_threshold:
        action_state = "EXIT_RISK"

    regime_bucket = _regime_bucket(regime_label, warning_ladder, event_regime_label)
    anti_chop_reasons = []
    if warning_ladder in {"Expansion Watch", "High Breakout Risk"} and event_breakout_bias == "Neutral":
        anti_chop_reasons.append("Expansion risk is rising without directional confirmation.")
    if warning_ladder in {"Expansion Watch", "High Breakout Risk"} and tradeability_score < anti_chop_tradeability_floor:
        anti_chop_reasons.append("Tradeability is still below the execution floor for breakout conditions.")
    if alignment_label == "Mixed / Low Alignment" and score_margin < (verdict_margin_threshold + anti_chop_margin_buffer):
        anti_chop_reasons.append("Directional edge is still too narrow in mixed alignment.")
    if "Doji" in pa_pattern and warning_ladder != "Active Momentum Event":
        anti_chop_reasons.append("Indecision candle is weakening the trigger.")
    if (
        action_state in {"SETUP_LONG", "SETUP_SHORT"}
        and max(bull_trigger, bear_trigger) < (trigger_min_score + anti_chop_trigger_buffer)
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

    base_conf = 50 + (score_margin * confidence_margin_multiplier) + (evidence_total * confidence_evidence_multiplier) + (stability_score * 0.12)
    penalty = 0.0

    if regime == "Range-Bound":
        penalty += rangebound_penalty
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
        no_trade_reasons.append("Major macro event window is active.")
    elif big_move_risk >= directional_expansion_threshold and event_breakout_bias == "Neutral":
        no_trade_reasons.append("Large move risk is elevated, but directional bias is still unclear.")
    elif warning_ladder in {"Directional Expansion Likely", "Active Momentum Event"} and event_breakout_bias != "Neutral" and directional_bias != "Neutral" and event_breakout_bias != directional_bias:
        penalty += 6.0
    if regime == "Range-Bound" and alignment_label == "Mixed / Low Alignment":
        no_trade_reasons.append("Range-bound regime with mixed alignment.")
    if adx_14 < no_trade_adx_threshold:
        no_trade_reasons.append("Trend strength is too weak.")
    if verdict == "Neutral" and max(bull_trigger, bear_trigger) < trigger_min_score:
        no_trade_reasons.append("No clean trigger is present.")
    fakeout_risk_score = float(((regime_state.get("components") or {}).get("fakeout_risk_score")) or 0.0)
    if fakeout_risk_score >= 3.0:
        anti_chop_reasons.append("Breakout risk is elevated, but fakeout risk remains high.")

    if anti_chop_reasons:
        penalty += anti_chop_penalty
    if fakeout_risk_score >= 3.0:
        penalty += fakeout_risk_penalty
        if action_state in {"SETUP_LONG", "SETUP_SHORT"}:
            action_state = "WAIT"
            action = "hold"
        if not no_trade_reasons:
            no_trade_reasons.append(anti_chop_reasons[0])

    confidence = base_conf - penalty
    if no_trade_reasons:
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
    entry_threshold = float(strategy_params.get("meta_entry_score_threshold", 63.0))
    fakeout_cap = float(strategy_params.get("meta_fakeout_prob_cap", 0.42))
    exit_cap = float(strategy_params.get("meta_exit_prob_cap", 0.58))
    min_expected_edge_pct = float(strategy_params.get("min_expected_edge_pct", 0.06))

    if action_state in {"LONG_ACTIVE", "SHORT_ACTIVE", "SETUP_LONG", "SETUP_SHORT"}:
        if meta_scores["entry_timing_score"] < entry_threshold:
            no_trade_reasons.append("Entry timing model is not yet confirming momentum quality.")
        if meta_scores["fakeout_probability"] > fakeout_cap:
            no_trade_reasons.append("Fakeout probability is elevated for the current setup.")
        if meta_scores["exit_risk_probability"] > exit_cap and action_state in {"SETUP_LONG", "SETUP_SHORT"}:
            no_trade_reasons.append("Exit risk model is too high for a fresh entry.")
        if expected_edge_pct < min_expected_edge_pct:
            no_trade_reasons.append("Expected value edge is below the execution threshold.")

        if no_trade_reasons and action_state in {"LONG_ACTIVE", "SHORT_ACTIVE", "SETUP_LONG", "SETUP_SHORT"}:
            action_state = "WAIT"
            action = "hold"
            verdict = "Neutral"
            confidence = min(confidence, no_trade_confidence_cap)

    forecast_state = _build_forecast_state(
        regime_bucket,
        regime_state,
        confidence,
        directional_bias,
    )
    execution_state = _build_execution_state(
        action_state,
        action,
        tradeability_label,
        no_trade_reasons,
        anti_chop_reasons,
        confidence,
    )
    execution_state["entryTimingScore"] = meta_scores["entry_timing_score"]
    execution_state["fakeoutProbability"] = meta_scores["fakeout_probability"]
    execution_state["exitRiskProbability"] = meta_scores["exit_risk_probability"]
    execution_state["expectedEdgePct"] = expected_edge_pct
    rr_signal_state = _build_rr_signal_state(
        ta_data=ta_data,
        strategy_params=strategy_params,
        verdict=verdict,
        directional_bias=directional_bias,
        regime_label=regime_label,
        alignment_label=alignment_label,
        action_state=action_state,
        confidence=confidence,
        direction_score=direction_score,
        tradeability_score=tradeability_score,
        stability_score=stability_score,
        expected_edge_pct=expected_edge_pct,
        meta_scores=meta_scores,
        regime_state=regime_state,
        no_trade_reasons=no_trade_reasons,
    )

    trade_guidance = compute_trade_guidance(
        ta_data,
        confidence,
    )
    price_action_bias = _price_action_bias(pa_struct, pa_pattern)
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

    if no_trade_reasons:
        trade_guidance["sellLevel"] = "Weak"
        trade_guidance["buyLevel"] = "Weak"
        if conflicted_bias_stack:
            trade_guidance["summary"] = _conflicted_summary(no_trade_reasons[0])
        elif directional_bias == "Bearish":
            trade_guidance["summary"] = f"Bearish bias is intact, but conditions are not clean enough for execution yet. {no_trade_reasons[0]}"
        elif directional_bias == "Bullish":
            trade_guidance["summary"] = f"Bullish bias is intact, but conditions are not clean enough for execution yet. {no_trade_reasons[0]}"
        else:
            trade_guidance["summary"] = no_trade_reasons[0]

    if action_state == "SETUP_LONG":
        trade_guidance["summary"] = "Long setup is forming, but trigger confirmation is still pending."
    elif action_state == "LONG_ACTIVE":
        trade_guidance["summary"] = "Long setup confirmed with acceptable tradeability."
    elif action_state == "SETUP_SHORT":
        trade_guidance["summary"] = "Short setup is forming, but trigger confirmation is still pending."
    elif action_state == "SHORT_ACTIVE":
        trade_guidance["summary"] = "Short setup confirmed with acceptable tradeability."
    elif action_state == "EXIT_RISK":
        trade_guidance["summary"] = "Exit risk is elevated; the active directional state is deteriorating."
    elif action_state == "WAIT" and directional_bias in {"Bullish", "Bearish"}:
        blockers = []
        if tradeability_label == "Low":
            blockers.append("tradeability is low")
        if regime_label in {"unstable", "range", "event-risk", "transition"}:
            blockers.append(f"regime is {regime_label}")
        blocker_text = " and ".join(blockers)
        if conflicted_bias_stack:
            trade_guidance["summary"] = _conflicted_summary(
                f"Because {blocker_text}." if blocker_text else "Execution quality is still too weak."
            )
        elif directional_bias == "Bearish" and trade_guidance["sellLevel"] in {"Watch", "Strong"}:
            trade_guidance["summary"] = (
                f"Sell bias is favored, but conditions are not clean enough for execution yet"
                + (f" because {blocker_text}." if blocker_text else ".")
            )
        elif directional_bias == "Bullish" and trade_guidance["buyLevel"] in {"Watch", "Strong"}:
            trade_guidance["summary"] = (
                f"Buy bias is favored, but conditions are not clean enough for execution yet"
                + (f" because {blocker_text}." if blocker_text else ".")
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
        "raw_verdict": "Bullish" if score_diff >= verdict_margin_threshold else ("Bearish" if score_diff <= -verdict_margin_threshold else "Neutral"),
        "verdict": verdict,
        "confidence": confidence,
        "noTradeReason": no_trade_reasons[0] if no_trade_reasons else "",
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
        "MarketState": {
            "regime": regime_label,
            "regime_bucket": regime_bucket,
            "active_regime_profile": active_regime_profile,
            "directional_bias": directional_bias,
            "tradeability": tradeability_label,
            "action": action,
            "action_state": action_state,
            "anti_chop_active": bool(anti_chop_reasons),
            "anti_chop_reasons": anti_chop_reasons,
            "entry_timing_score": meta_scores["entry_timing_score"],
            "fakeout_probability": meta_scores["fakeout_probability"],
            "exit_risk_probability": meta_scores["exit_risk_probability"],
            "expected_edge_pct": expected_edge_pct,
            "scores": {
                "direction": round(direction_score, 2),
                "tradeability": round(tradeability_score, 2),
                "exit_risk": round(exit_risk_score, 2),
                "stability": round(stability_score, 2),
            },
            "quality_components": {
                "regime_quality": round(regime_quality, 2),
                "alignment_quality": round(alignment_quality, 2),
                "structure_quality": round(structure_quality, 2),
                "trigger_quality": round(trigger_quality, 2),
                "volume_quality": round(volume_quality, 2),
            },
            "confidence_mode": confidence_mode,
            "confidence_bucket": confidence_bucket,
            "rr_signal": rr_signal_state,
        },
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
            "feature_hits": regime_state.get("feature_hits", {}),
            "components": regime_state.get("components", {}),
        },
        "ForecastState": forecast_state,
        "ExecutionState": execution_state,
        "RR200Signal": rr_signal_state,
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


def prepare_historical_features(df, params=None):
    strategy_params = normalize_strategy_params(params)
    frame = df.copy().sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
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

    frame["EMA_TREND"] = _trend_series_from_close(frame["Close"], ema_short, ema_long)

    trend_4h = _trend_series_from_close(
        frame["Close"].resample("4h").last().dropna(),
        ema_short,
        ema_long,
    )
    frame["H4_TREND"] = trend_4h.reindex(frame.index, method="ffill").fillna("Neutral")
    frame["M15_TREND"] = frame["EMA_TREND"]

    trend_map = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
    frame["ALIGNMENT_SCORE"] = (
        frame["M15_TREND"].map(trend_map).fillna(0)
        + frame["EMA_TREND"].map(trend_map).fillna(0)
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

    frame = annotate_price_action(frame)
    return annotate_event_regime_features(frame)


def build_ta_payload_from_row(row, params=None, regime_memory=None):
    strategy_params = normalize_strategy_params(params)
    bar_ts = None
    try:
        if getattr(row, "name", None) is not None:
            bar_ts = pd.Timestamp(row.name).tz_localize("UTC").isoformat() if pd.Timestamp(row.name).tzinfo is None else pd.Timestamp(row.name).tz_convert("UTC").isoformat()
    except Exception:
        bar_ts = None
    return {
        "bar_timestamp_utc": bar_ts,
        "current_price": round(float(row["Close"]), 2),
        "ema_trend": row.get("EMA_TREND", "Neutral"),
        "ema_20": round(float(row.get("EMA_20", 0.0)), 2),
        "ema_50": round(float(row.get("EMA_50", 0.0)), 2),
        "rsi_14": round(float(row.get("RSI_14", 50.0)), 2),
        "volatility_regime": {
            "market_regime": row.get("MARKET_REGIME", "Range-Bound"),
            "adx_14": round(float(row.get("ADX_14", 0.0)), 2),
            "atr_14": round(float(row.get("ATR_14", 0.0)), 2),
            "atr_percent": round(float(row.get("ATR_PERCENT", 0.0)), 3),
        },
        "multi_timeframe": {
            "m15_trend": row.get("M15_TREND", "Neutral"),
            "h1_trend": row.get("EMA_TREND", "Neutral"),
            "h4_trend": row.get("H4_TREND", "Neutral"),
            "alignment_score": int(row.get("ALIGNMENT_SCORE", 0)),
            "alignment_label": row.get("ALIGNMENT_LABEL", "Mixed / Low Alignment"),
            "sources": {
                "m15": "derived_proxy",
                "h4": "derived_resample",
            },
        },
        "price_action": {
            "structure": row.get("PA_STRUCTURE", "Consolidating"),
            "latest_candle_pattern": row.get("CANDLE_PATTERN", "None"),
        },
        "volume_analysis": {
            "cmf_14": round(float(row.get("CMF_14", 0.0)), 4),
            "obv_trend": "Rising" if row.get("OBV", 0.0) > row.get("OBV_PREV", row.get("OBV", 0.0)) else "Falling",
            "overall_volume_signal": row.get("VOLUME_SIGNAL", "Neutral"),
        },
        "active_strategy_params": strategy_params,
        "event_regime": compute_event_regime_snapshot(
            row,
            trend=row.get("EMA_TREND", "Neutral"),
            alignment_label=row.get("ALIGNMENT_LABEL", "Mixed / Low Alignment"),
            market_structure=row.get("PA_STRUCTURE", "Consolidating"),
            candle_pattern=row.get("CANDLE_PATTERN", "None"),
            event_risk={
                "active": bool(row.get("EVENT_ACTIVE", 0)),
            },
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
