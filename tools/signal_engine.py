import pandas as pd
import ta
from pathlib import Path
import json


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
    "sentiment_quality_weight": 0.05,
    "stability_flip_penalty": 12.0,
    "stability_conflict_penalty": 10.0,
    "stability_mixed_alignment_penalty": 10.0,
    "high_tradeability_threshold": 68.0,
    "medium_tradeability_threshold": 52.0,
    "direction_entry_threshold": 8.0,
    "direction_hold_threshold": 5.0,
    "exit_risk_threshold": 6.0,
    "mtf_intervals": ["15min", "1h", "4h"],
}

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIDENCE_CALIBRATION_FILE = BASE_DIR / "tools" / "reports" / "confidence_calibration.json"

SENTIMENT_WEIGHTED_SIGNALS = [
    {
        "terms": [
            "safe haven",
            "geopolitical",
            "war",
            "conflict",
            "middle east tension",
            "risk-off",
        ],
        "weight": 2,
        "side": "bull",
    },
    {
        "terms": [
            "gold rises",
            "gold rise",
            "higher",
            "rebound",
            "gains",
            "surges",
            "bullish",
            "buy",
            "demand",
            "record high",
            "inflows",
            "strong demand",
        ],
        "weight": 1,
        "side": "bull",
    },
    {
        "terms": [
            "dollar rises",
            "dollar rise",
            "stronger dollar",
            "dollar strength",
            "dollar index rises",
            "dxy rises",
            "yields rise",
            "yield rises",
            "treasury yields rise",
            "real yields rise",
            "hawkish fed",
            "hawkish",
            "rate hike",
            "rates stay high",
            "higher rates",
        ],
        "weight": 2,
        "side": "bear",
    },
    {
        "terms": [
            "gold lower",
            "gold falls",
            "gold fall",
            "drops",
            "drop",
            "slips",
            "bearish",
            "sell",
            "outflows",
            "profit-taking",
            "weaker",
        ],
        "weight": 1,
        "side": "bear",
    },
]


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
    bucket = _bucket_from_components(regime_label, directional_bias, tradeability, stability)
    empirical = calibration.get(bucket)
    if isinstance(empirical, (int, float)):
        calibrated = max(50, min(95, round(float(empirical))))
        return calibrated, "empirical", bucket
    return int(raw_confidence), "heuristic", bucket


def classify_market_sentiment(news_list):
    if not isinstance(news_list, list):
        news_list = []

    bullish_score = 0
    bearish_score = 0
    driver_counts = {}

    for news in news_list:
        title = str((news or {}).get("title", "")).lower()
        for signal_group in SENTIMENT_WEIGHTED_SIGNALS:
            for term in signal_group["terms"]:
                if term in title:
                    if signal_group["side"] == "bull":
                        bullish_score += signal_group["weight"]
                    else:
                        bearish_score += signal_group["weight"]
                    driver_counts[term] = driver_counts.get(term, 0) + 1

    net_score = bullish_score - bearish_score
    label = "Neutral"
    if net_score >= 2:
        label = "Bullish"
    elif net_score <= -1:
        label = "Bearish"

    intensity = abs(net_score)
    confidence_band = "Low"
    if intensity >= 6:
        confidence_band = "High"
    elif intensity >= 3:
        confidence_band = "Medium"

    top_drivers = [
        term for term, _count in sorted(driver_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    ]

    return {
        "label": label,
        "confidenceBand": confidence_band,
        "bullishScore": bullish_score,
        "bearishScore": bearish_score,
        "netScore": net_score,
        "topDrivers": top_drivers,
    }


def compute_trade_guidance(ta_data, sentiment_label, confidence):
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

    if sentiment_label == "Bearish":
        sell_score += 1
    elif sentiment_label == "Bullish":
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


def compute_prediction_from_ta(ta_data, sentiment_summary=None):
    if not isinstance(ta_data, dict):
        return {
            "verdict": "Neutral",
            "confidence": 50,
            "TradeGuidance": compute_trade_guidance({}, "Neutral", 50),
        }

    sentiment_summary = sentiment_summary if isinstance(sentiment_summary, dict) else {}
    strategy_params = normalize_strategy_params(ta_data.get("active_strategy_params"))

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
    elif "Bullish Structure" in pa_struct or "Bullish Drift" in pa_struct or "Bullish Pressure" in pa_struct:
        bull_trigger += structure_weight
    elif "Bearish Structure" in pa_struct or "Bearish Drift" in pa_struct or "Bearish Pressure" in pa_struct:
        bear_trigger += structure_weight

    if "Bullish Engulfing" in pa_pattern:
        bull_trigger += engulfing_weight
    elif "Bearish Engulfing" in pa_pattern:
        bear_trigger += engulfing_weight
    elif "Bullish Hammer" in pa_pattern:
        bull_trigger += reversal_candle_weight
    elif "Bearish Shooting Star" in pa_pattern:
        bear_trigger += reversal_candle_weight

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
    sentiment_quality_weight = float(strategy_params.get("sentiment_quality_weight", 0.05))
    stability_flip_penalty = float(strategy_params.get("stability_flip_penalty", 12.0))
    stability_conflict_penalty = float(strategy_params.get("stability_conflict_penalty", 10.0))
    stability_mixed_alignment_penalty = float(strategy_params.get("stability_mixed_alignment_penalty", 10.0))
    high_tradeability_threshold = float(strategy_params.get("high_tradeability_threshold", 68.0))
    medium_tradeability_threshold = float(strategy_params.get("medium_tradeability_threshold", 52.0))

    regime_label = "transition"
    if event_risk.get("active"):
        regime_label = "event-risk"
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
    sentiment_quality = 70.0 if sentiment_summary.get("label") in {"Bullish", "Bearish"} else 45.0

    tradeability_score = (
        regime_quality * regime_quality_weight
        + alignment_quality * alignment_quality_weight
        + structure_quality * structure_quality_weight
        + trigger_quality * trigger_quality_weight
        + volume_quality * volume_quality_weight
        + sentiment_quality * sentiment_quality_weight
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
    if event_risk.get("active"):
        penalty += event_risk_penalty
        no_trade_reasons.append("Major macro event window is active.")
    if regime == "Range-Bound" and alignment_label == "Mixed / Low Alignment":
        no_trade_reasons.append("Range-bound regime with mixed alignment.")
    if adx_14 < no_trade_adx_threshold:
        no_trade_reasons.append("Trend strength is too weak.")
    if verdict == "Neutral" and max(bull_trigger, bear_trigger) < trigger_min_score:
        no_trade_reasons.append("No clean trigger is present.")

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

    trade_guidance = compute_trade_guidance(
        ta_data,
        sentiment_summary.get("label", "Neutral"),
        confidence,
    )
    if no_trade_reasons:
        trade_guidance["sellLevel"] = "Weak"
        trade_guidance["buyLevel"] = "Weak"
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
        "directionalBias": directional_bias,
        "tradeability": tradeability_label,
        "actionState": action_state,
        "action": action,
        "confidenceMode": confidence_mode,
        "confidenceBucket": confidence_bucket,
        "MarketState": {
            "regime": regime_label,
            "directional_bias": directional_bias,
            "tradeability": tradeability_label,
            "action": action,
            "action_state": action_state,
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
                "sentiment_quality": round(sentiment_quality, 2),
            },
            "confidence_mode": confidence_mode,
            "confidence_bucket": confidence_bucket,
        },
        "TradeGuidance": trade_guidance,
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

    frame["PA_STRUCTURE"] = "Consolidating"
    frame["CANDLE_PATTERN"] = "None"
    for i in range(len(frame)):
        current = frame.iloc[i]
        prev = frame.iloc[i - 1] if i >= 1 else current
        prev2 = frame.iloc[i - 2] if i >= 2 else prev

        pa_structure = "Consolidating"
        if i >= 20:
            recent_high = frame["High"].iloc[i - 20:i].max()
            recent_low = frame["Low"].iloc[i - 20:i].min()
            if current["Close"] > recent_high:
                pa_structure = "Bullish Breakout"
            elif current["Close"] < recent_low:
                pa_structure = "Bearish Breakdown"

        if pa_structure == "Consolidating" and i >= 2:
            if current["High"] > prev["High"] > prev2["High"] and current["Low"] > prev["Low"] > prev2["Low"]:
                pa_structure = "Higher Highs / Higher Lows (Bullish Structure)"
            elif current["High"] < prev["High"] < prev2["High"] and current["Low"] < prev["Low"] < prev2["Low"]:
                pa_structure = "Lower Highs / Lower Lows (Bearish Structure)"

        if pa_structure == "Consolidating" and i >= 11:
            recent12 = frame.iloc[i - 11:i + 1]
            high_now = recent12["High"].iloc[-4:].mean()
            high_prev = recent12["High"].iloc[-8:-4].mean()
            low_now = recent12["Low"].iloc[-4:].mean()
            low_prev = recent12["Low"].iloc[-8:-4].mean()
            close_now = recent12["Close"].iloc[-3:].mean()
            close_prev = recent12["Close"].iloc[-6:-3].mean()

            if high_now > high_prev and low_now > low_prev and close_now > close_prev:
                pa_structure = "Bullish Drift"
            elif high_now < high_prev and low_now < low_prev and close_now < close_prev:
                pa_structure = "Bearish Drift"

        if pa_structure == "Consolidating" and i >= 19:
            recent20 = frame.iloc[i - 19:i + 1]
            range_high = recent20["High"].max()
            range_low = recent20["Low"].min()
            range_size = max(range_high - range_low, 1e-8)
            close_pos = (current["Close"] - range_low) / range_size
            if current["EMA_TREND"] == "Bullish" and close_pos >= 0.67:
                pa_structure = "Bullish Pressure in Range"
            elif current["EMA_TREND"] == "Bearish" and close_pos <= 0.33:
                pa_structure = "Bearish Pressure in Range"

        candle_pattern = "None"
        if (
            current["Close"] > current["Open"]
            and prev["Close"] < prev["Open"]
            and current["Close"] > prev["Open"]
            and current["Open"] < prev["Close"]
        ):
            candle_pattern = "Bullish Engulfing"
        elif (
            current["Close"] < current["Open"]
            and prev["Close"] > prev["Open"]
            and current["Close"] < prev["Open"]
            and current["Open"] > prev["Close"]
        ):
            candle_pattern = "Bearish Engulfing"
        else:
            candle_range = max(current["High"] - current["Low"], 1e-8)
            body = abs(current["Close"] - current["Open"])
            upper_wick = current["High"] - max(current["Close"], current["Open"])
            lower_wick = min(current["Close"], current["Open"]) - current["Low"]

            if body / candle_range < 0.15:
                candle_pattern = "Doji"
            elif lower_wick / candle_range > 0.55 and upper_wick / candle_range < 0.2 and current["Close"] >= current["Open"]:
                candle_pattern = "Bullish Hammer"
            elif upper_wick / candle_range > 0.55 and lower_wick / candle_range < 0.2 and current["Close"] <= current["Open"]:
                candle_pattern = "Bearish Shooting Star"

        frame.iloc[i, frame.columns.get_loc("PA_STRUCTURE")] = pa_structure
        frame.iloc[i, frame.columns.get_loc("CANDLE_PATTERN")] = candle_pattern

    return frame


def build_ta_payload_from_row(row, params=None):
    strategy_params = normalize_strategy_params(params)
    return {
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
    }
