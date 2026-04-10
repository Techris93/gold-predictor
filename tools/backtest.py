import json
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

from tools.signal_engine import (
    build_ta_payload_from_row,
    compute_prediction_from_ta,
    normalize_strategy_params,
    prepare_historical_features,
)
from tools.twelvedata_market_data import fetch_history

DEFAULT_BACKTEST_PARAMS = {
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


ACTIVE_BACKTEST_PARAMS = _load_json_config("config/backtest_params.json", DEFAULT_BACKTEST_PARAMS)
FEATURE_REPORT_FILE = os.path.join(BASE_DIR, "tools", "reports", "backtest_feature_analysis.json")
CONFIDENCE_CALIBRATION_FILE = os.path.join(BASE_DIR, "tools", "reports", "confidence_calibration.json")
OUTCOME_SUMMARY_FILE = os.path.join(BASE_DIR, "tools", "reports", "signal_outcome_summary.json")

def generate_signals(df, params=None):
    params = normalize_strategy_params(params or ACTIVE_BACKTEST_PARAMS)
    enriched = prepare_historical_features(df, params)
    signals = pd.Series('Neutral', index=enriched.index)
    states = []
    regime_memory = {}

    for i in range(len(enriched)):
        ta_payload = build_ta_payload_from_row(enriched.iloc[i], params, regime_memory=regime_memory)
        prediction = compute_prediction_from_ta(ta_payload)
        if isinstance(prediction.get("_regime_memory"), dict):
            regime_memory = dict(prediction.get("_regime_memory") or {})
        verdict = prediction["verdict"]
        if verdict == 'Bullish':
            signals.iloc[i] = 'Buy'
        elif verdict == 'Bearish':
            signals.iloc[i] = 'Sell'
        states.append(
            {
                "timestamp": str(enriched.index[i]),
                "verdict": verdict,
                "action_state": prediction.get("actionState"),
                "action": prediction.get("action"),
                "regime": prediction.get("regime"),
                "directional_bias": prediction.get("directionalBias"),
                "tradeability": prediction.get("tradeability"),
                "confidence": prediction.get("confidence"),
                "confidence_bucket": prediction.get("confidenceBucket"),
                "direction_score": prediction.get("directionScore"),
                "tradeability_score": prediction.get("tradeabilityScore"),
                "exit_risk_score": prediction.get("exitRiskScore"),
                "stability_score": prediction.get("stabilityScore"),
                "regime_bucket": prediction.get("regimeBucket"),
                "anti_chop_active": prediction.get("antiChopActive"),
                "anti_chop_reasons": prediction.get("antiChopReasons", []),
                "feature_hits": prediction.get("FeatureHits", {}),
                "regime_state": prediction.get("RegimeState", {}),
                "forecast_state": prediction.get("ForecastState", {}),
                "execution_state": prediction.get("ExecutionState", {}),
            }
        )

    return signals, states, enriched


def summarize_transition_metrics(df, states):
    if not states:
        return {}

    actionable_states = {"LONG_ACTIVE", "SHORT_ACTIVE"}
    transition_count = 0
    false_entry_count = 0
    whipsaw_count = 0
    persistence_lengths = []
    favorable_moves = []
    adverse_moves = []
    reversal_lengths = []

    last_action_state = states[0].get("action_state")
    current_persistence = 1

    for i in range(1, len(states)):
        state = states[i]
        prev = states[i - 1]
        if state.get("action_state") != prev.get("action_state"):
            transition_count += 1
            persistence_lengths.append(current_persistence)
            current_persistence = 1
        else:
            current_persistence += 1

        prev_state = prev.get("action_state")
        cur_state = state.get("action_state")
        if prev_state in actionable_states and cur_state == "EXIT_RISK":
            reversal_lengths.append(1)
        if prev_state == "LONG_ACTIVE" and cur_state in {"EXIT_RISK", "WAIT"}:
            entry_price = float(df["Close"].iloc[i - 1])
            future = df.iloc[i:min(i + 4, len(df))]
            if not future.empty:
                favorable = (future["High"].max() - entry_price) / max(entry_price, 1e-8)
                adverse = (future["Low"].min() - entry_price) / max(entry_price, 1e-8)
                favorable_moves.append(float(favorable))
                adverse_moves.append(float(adverse))
                if favorable < 0.0025:
                    false_entry_count += 1
        if prev_state == "SHORT_ACTIVE" and cur_state in {"EXIT_RISK", "WAIT"}:
            entry_price = float(df["Close"].iloc[i - 1])
            future = df.iloc[i:min(i + 4, len(df))]
            if not future.empty:
                favorable = (entry_price - future["Low"].min()) / max(entry_price, 1e-8)
                adverse = (entry_price - future["High"].max()) / max(entry_price, 1e-8)
                favorable_moves.append(float(favorable))
                adverse_moves.append(float(adverse))
                if favorable < 0.0025:
                    false_entry_count += 1
        if prev_state in {"LONG_ACTIVE", "SHORT_ACTIVE"} and cur_state in {"LONG_ACTIVE", "SHORT_ACTIVE"} and prev_state != cur_state:
            whipsaw_count += 1

    persistence_lengths.append(current_persistence)
    actionable_entries = sum(1 for item in states if item.get("action_state") in actionable_states)

    return {
        "transition_count": transition_count,
        "actionable_entries": actionable_entries,
        "false_entry_rate": round(false_entry_count / actionable_entries, 4) if actionable_entries else 0.0,
        "whipsaw_rate": round(whipsaw_count / actionable_entries, 4) if actionable_entries else 0.0,
        "avg_signal_persistence": round(float(np.mean(persistence_lengths)), 2) if persistence_lengths else 0.0,
        "avg_favorable_excursion": round(float(np.mean(favorable_moves)) * 100, 3) if favorable_moves else 0.0,
        "avg_adverse_excursion": round(float(np.mean(adverse_moves)) * 100, 3) if adverse_moves else 0.0,
        "avg_time_to_reversal": round(float(np.mean(reversal_lengths)), 2) if reversal_lengths else 0.0,
    }


def summarize_feature_hit_metrics(df, states):
    if not states or len(df) < 4:
        return {}

    tracked = {}
    max_index = min(len(states), len(df) - 3)
    for i in range(max_index):
        state = states[i]
        hits = state.get("feature_hits") or {}
        verdict = state.get("verdict")
        if verdict not in {"Bullish", "Bearish"}:
            continue

        entry_price = float(df["Close"].iloc[i])
        future_close = float(df["Close"].iloc[i + 3])
        realized = (future_close - entry_price) / max(entry_price, 1e-8)
        directional_success = realized > 0 if verdict == "Bullish" else realized < 0

        feature_names = []
        if hits.get("structure_breakout"):
            feature_names.append("structure_breakout")
        if hits.get("structure_swing"):
            feature_names.append("structure_swing")
        if hits.get("structure_drift"):
            feature_names.append("structure_drift")
        if hits.get("structure_range_pressure"):
            feature_names.append("structure_range_pressure")
        if hits.get("candle_engulfing"):
            feature_names.append("candle_engulfing")
        if hits.get("candle_reversal"):
            feature_names.append("candle_reversal")
        if hits.get("candle_doji"):
            feature_names.append("candle_doji")

        for feature_name in feature_names:
            bucket = tracked.setdefault(feature_name, {"hits": 0, "wins": 0, "returns": []})
            bucket["hits"] += 1
            bucket["wins"] += 1 if directional_success else 0
            bucket["returns"].append(realized if verdict == "Bullish" else -realized)

    summary = {}
    for feature_name, bucket in tracked.items():
        avg_return = float(np.mean(bucket["returns"])) * 100 if bucket["returns"] else 0.0
        summary[feature_name] = {
            "hits": int(bucket["hits"]),
            "directional_accuracy": round(bucket["wins"] / bucket["hits"], 4) if bucket["hits"] else 0.0,
            "avg_3bar_directional_return_pct": round(avg_return, 4),
        }
    return summary


def _simulate_from_signals(df, signals):
    position = 0
    entry_price = 0
    trades = []

    for i in range(len(df) - 1):
        signal = signals.iloc[i]
        next_open = df['Open'].iloc[i + 1]

        if position == 0:
            if signal == 'Buy':
                position = 1
                entry_price = next_open
            elif signal == 'Sell':
                position = -1
                entry_price = next_open
        elif position == 1:
            if signal == 'Sell':
                pnl = (next_open - entry_price) / entry_price
                trades.append({'type': 'Long', 'pnl': pnl})
                position = -1
                entry_price = next_open
        elif position == -1:
            if signal == 'Buy':
                pnl = (entry_price - next_open) / entry_price
                trades.append({'type': 'Short', 'pnl': pnl})
                position = 1
                entry_price = next_open

    if position == 1:
        pnl = (df['Close'].iloc[-1] - entry_price) / entry_price
        trades.append({'type': 'Long', 'pnl': pnl})
    elif position == -1:
        pnl = (entry_price - df['Close'].iloc[-1]) / entry_price
        trades.append({'type': 'Short', 'pnl': pnl})

    return trades


def _trade_summary(trades):
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "final_capital": 10000.0,
        }

    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(winning_trades) / len(trades) * 100
    avg_win = np.mean([t['pnl'] for t in winning_trades]) * 100 if winning_trades else 0.0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) * 100 if losing_trades else 0.0
    capital = 10000.0
    for trade in trades:
        capital = capital * (1 + trade['pnl'])
    roi = ((capital - 10000.0) / 10000.0) * 100
    return {
        "trades": len(trades),
        "win_rate": round(win_rate, 2),
        "roi": round(roi, 2),
        "average_win": round(avg_win, 2),
        "average_loss": round(avg_loss, 2),
        "final_capital": round(capital, 2),
    }


def summarize_ablation_metrics(df, base_params):
    base_params = normalize_strategy_params(base_params)
    variants = {
        "full_model": base_params,
        "without_market_structure": {
            **base_params,
            "breakout_weight": 0.0,
            "structure_weight": 0.0,
            "swing_structure_weight": 0.0,
            "drift_weight": 0.0,
            "range_pressure_weight": 0.0,
        },
        "without_candle_pattern": {
            **base_params,
            "engulfing_weight": 0.0,
            "reversal_candle_weight": 0.0,
        },
        "without_price_action": {
            **base_params,
            "breakout_weight": 0.0,
            "structure_weight": 0.0,
            "swing_structure_weight": 0.0,
            "drift_weight": 0.0,
            "range_pressure_weight": 0.0,
            "engulfing_weight": 0.0,
            "reversal_candle_weight": 0.0,
        },
    }

    summaries = {}
    for label, params in variants.items():
        signals, _states, _enriched = generate_signals(df, params=params)
        summaries[label] = _trade_summary(_simulate_from_signals(df, signals))

    base = summaries["full_model"]
    summaries["delta_vs_full_model"] = {
        label: {
            "roi_delta": round(summary["roi"] - base["roi"], 2),
            "win_rate_delta": round(summary["win_rate"] - base["win_rate"], 2),
            "trade_delta": int(summary["trades"] - base["trades"]),
        }
        for label, summary in summaries.items()
        if label != "full_model"
    }
    return summaries


def summarize_big_move_metrics(df, states):
    if not states or len(df) < 6:
        return {}

    watch_hits_60m = 0
    watch_true_60m = 0
    watch_hits_4h = 0
    watch_true_4h = 0
    active_hits_4h = 0
    active_true_4h = 0
    total_true_60m = 0
    total_true_4h = 0
    lead_times = []
    regime_counts = {}

    max_index = min(len(states), len(df) - 4)
    for i in range(max_index):
        entry = float(df["Close"].iloc[i])
        next_bar = df.iloc[i + 1]
        next_four = df.iloc[i + 1 : i + 5]
        future_move_60m = max(abs(float(next_bar["High"]) - entry), abs(entry - float(next_bar["Low"])))
        future_move_4h = max(abs(float(next_four["High"].max()) - entry), abs(entry - float(next_four["Low"].min())))
        true_60m = future_move_60m >= 8.0
        true_4h = future_move_4h >= 15.0
        total_true_60m += 1 if true_60m else 0
        total_true_4h += 1 if true_4h else 0

        regime_state = state_regime = states[i].get("regime_state") or {}
        ladder = str(regime_state.get("warning_ladder", "Normal"))
        event_regime = str(regime_state.get("event_regime", "normal"))
        regime_counts[event_regime] = regime_counts.get(event_regime, 0) + 1
        watch_flag = ladder in {"Expansion Watch", "High Breakout Risk", "Directional Expansion Likely", "Active Momentum Event"}
        active_flag = ladder in {"Directional Expansion Likely", "Active Momentum Event"}

        if watch_flag:
            watch_hits_60m += 1
            watch_hits_4h += 1
            watch_true_60m += 1 if true_60m else 0
            watch_true_4h += 1 if true_4h else 0
        if active_flag:
            active_hits_4h += 1
            active_true_4h += 1 if true_4h else 0

        if true_4h:
            for lookback in range(1, min(4, i + 1)):
                previous_ladder = str((states[i - lookback].get("regime_state") or {}).get("warning_ladder", "Normal"))
                if previous_ladder in {"Expansion Watch", "High Breakout Risk", "Directional Expansion Likely", "Active Momentum Event"}:
                    lead_times.append(lookback)
                    break

    def _safe_ratio(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "label_definition": {
            "big_move_60m_price_points": 8.0,
            "big_move_4h_price_points": 15.0,
        },
        "label_prevalence": {
            "true_60m_rate": _safe_ratio(total_true_60m, max_index),
            "true_4h_rate": _safe_ratio(total_true_4h, max_index),
        },
        "warning_watch": {
            "precision_60m": _safe_ratio(watch_true_60m, watch_hits_60m),
            "recall_60m": _safe_ratio(watch_true_60m, total_true_60m),
            "precision_4h": _safe_ratio(watch_true_4h, watch_hits_4h),
            "recall_4h": _safe_ratio(watch_true_4h, total_true_4h),
        },
        "warning_directional": {
            "precision_4h": _safe_ratio(active_true_4h, active_hits_4h),
            "recall_4h": _safe_ratio(active_true_4h, total_true_4h),
        },
        "avg_lead_bars_4h": round(float(np.mean(lead_times)), 2) if lead_times else None,
        "event_regime_distribution": regime_counts,
    }


def summarize_large_move_labels(df, states):
    if len(df) < 8:
        return {}

    total = min(len(states), len(df) - 4)
    label_counts = {
        "abs_move_gt_300pips_30m": 0,
        "abs_move_gt_500pips_60m": 0,
        "abs_move_gt_1000pips_4h": 0,
        "directional_expansion_after_compression": 0,
    }
    for i in range(total):
        entry = float(df["Close"].iloc[i])
        one = df.iloc[i + 1]
        four = df.iloc[i + 1 : i + 5]
        move_30m = max(abs(float(one["High"]) - entry), abs(entry - float(one["Low"])))
        move_60m = move_30m
        move_4h = max(abs(float(four["High"].max()) - entry), abs(entry - float(four["Low"].min())))
        if move_30m >= 4.0:
            label_counts["abs_move_gt_300pips_30m"] += 1
        if move_60m >= 8.0:
            label_counts["abs_move_gt_500pips_60m"] += 1
        if move_4h >= 15.0:
            label_counts["abs_move_gt_1000pips_4h"] += 1

        state = states[i] if i < len(states) else {}
        feature_hits = (state.get("regime_state") or {}).get("feature_hits") or {}
        compressed = bool(feature_hits.get("compression") or feature_hits.get("compression_setup"))
        breakout = bool(feature_hits.get("atr_expansion") and feature_hits.get("bar_velocity"))
        if compressed and breakout and move_60m >= 8.0:
            label_counts["directional_expansion_after_compression"] += 1

    return {
        "samples": int(total),
        "label_counts": label_counts,
        "label_rates": {
            key: round(value / total, 4) if total else 0.0
            for key, value in label_counts.items()
        },
    }


def summarize_decision_quality_metrics(df, states):
    if not states or len(df) < 6:
        return {}

    total = min(len(states), len(df) - 4)
    abstain_total = 0
    abstain_correct = 0
    enter_total = 0
    enter_correct = 0
    decision_regret = []

    for i in range(total):
        state = states[i]
        execution = state.get("execution_state") or {}
        status = str(execution.get("status") or "stand_aside")
        verdict = str(state.get("verdict") or "Neutral")
        entry = float(df["Close"].iloc[i])
        future_close = float(df["Close"].iloc[i + 4])
        realized_pct = ((future_close - entry) / max(entry, 1e-8)) * 100.0
        abs_realized_pct = abs(realized_pct)
        realized_dir = "Bullish" if realized_pct > 0 else ("Bearish" if realized_pct < 0 else "Neutral")

        if status in {"stand_aside", "prepare"}:
            abstain_total += 1
            if abs_realized_pct < 0.25:
                abstain_correct += 1
            regret = 0.0
            if abs_realized_pct >= 0.4:
                regret = abs_realized_pct
            decision_regret.append(regret)
            continue

        if status == "enter":
            enter_total += 1
            if verdict in {"Bullish", "Bearish"} and verdict == realized_dir:
                enter_correct += 1
            signed_model_return = realized_pct if verdict == "Bullish" else (-realized_pct if verdict == "Bearish" else 0.0)
            decision_regret.append(max(0.0, -signed_model_return))
            continue

        decision_regret.append(0.0)

    return {
        "abstain_total": int(abstain_total),
        "abstain_correct_rate": round(abstain_correct / abstain_total, 4) if abstain_total else 0.0,
        "enter_total": int(enter_total),
        "enter_correct_rate": round(enter_correct / enter_total, 4) if enter_total else 0.0,
        "avg_regret_vs_stand_aside_pct": round(float(np.mean(decision_regret)), 4) if decision_regret else 0.0,
    }


def summarize_confidence_reliability(df, states):
    if not states or len(df) < 4:
        return {}

    buckets = {}
    regime_scores = {}
    calibration_table = {}
    brier_terms = []
    rolling_points = []
    max_index = min(len(states), len(df) - 3)
    for i in range(max_index):
        state = states[i]
        verdict = state.get("verdict")
        if verdict not in {"Bullish", "Bearish"}:
            continue
        entry_price = float(df["Close"].iloc[i])
        future_close = float(df["Close"].iloc[i + 3])
        realized = (future_close - entry_price) / max(entry_price, 1e-8)
        success = realized > 0 if verdict == "Bullish" else realized < 0
        confidence = float(state.get("confidence") or 50.0)
        predicted_prob = max(0.0, min(1.0, confidence / 100.0))
        actual = 1.0 if success else 0.0
        brier_terms.append((predicted_prob - actual) ** 2)
        confidence_band = f"{int(confidence // 10) * 10:02d}-{min(99, int(confidence // 10) * 10 + 9):02d}"
        bucket = buckets.setdefault(confidence_band, {"count": 0, "wins": 0})
        bucket["count"] += 1
        bucket["wins"] += 1 if success else 0
        rolling_points.append(
            {
                "timestamp": state.get("timestamp"),
                "predicted_prob": predicted_prob,
                "actual": actual,
            }
        )

        regime_bucket = str(state.get("regime_bucket") or "transition")
        regime_record = regime_scores.setdefault(regime_bucket, {"count": 0, "wins": 0, "confidence_sum": 0.0})
        regime_record["count"] += 1
        regime_record["wins"] += 1 if success else 0
        regime_record["confidence_sum"] += confidence

        tradeability = "High" if float(state.get("tradeability_score") or 0.0) >= 68.0 else "Medium" if float(state.get("tradeability_score") or 0.0) >= 52.0 else "Low"
        stability = "stable" if float(state.get("stability_score") or 0.0) >= 60.0 else "unstable"
        calib_key = "|".join([str(state.get("regime") or "transition"), str(state.get("directional_bias") or "Neutral"), tradeability, stability])
        calib_entry = calibration_table.setdefault(calib_key, {"count": 0, "wins": 0})
        calib_entry["count"] += 1
        calib_entry["wins"] += 1 if success else 0

    reliability_curve = {
        key: {
            "count": value["count"],
            "hit_rate": round(value["wins"] / value["count"], 4) if value["count"] else 0.0,
        }
        for key, value in sorted(buckets.items())
    }
    regime_confidence = {
        key: round((value["wins"] / value["count"]) * 100.0, 2)
        for key, value in regime_scores.items()
        if value["count"]
    }
    confidence_buckets = {
        key: round((value["wins"] / value["count"]) * 100.0, 2)
        for key, value in calibration_table.items()
        if value["count"] >= 5
    }
    payload = {
        "regime_confidence": regime_confidence,
        "confidence_buckets": confidence_buckets,
        "reliability_curve": reliability_curve,
        "brier_score": round(float(np.mean(brier_terms)), 6) if brier_terms else None,
    }
    if rolling_points:
        df_roll = pd.DataFrame(rolling_points)
        window = 200
        if len(df_roll) >= max(30, window // 2):
            df_roll["brier"] = (df_roll["predicted_prob"] - df_roll["actual"]) ** 2
            df_roll["rolling_brier"] = df_roll["brier"].rolling(window=min(window, len(df_roll)), min_periods=30).mean()
            sampled = df_roll.dropna(subset=["rolling_brier"]).iloc[:: max(1, len(df_roll) // 40)]
            payload["rolling_brier"] = [
                {
                    "timestamp": str(row["timestamp"]),
                    "value": round(float(row["rolling_brier"]), 6),
                }
                for _, row in sampled.iterrows()
            ]
    if isinstance(payload.get("reliability_curve"), dict):
        ev_buckets = {}
        for key, value in payload["reliability_curve"].items():
            hit = float(value.get("hit_rate", 0.0) or 0.0)
            avg_win_pct = 0.55 + (0.15 * hit)
            avg_loss_pct = 0.42 + (0.08 * (1.0 - hit))
            ev_buckets[str(key)] = {
                "hit_rate": round(hit, 4),
                "avg_win_pct": round(avg_win_pct, 4),
                "avg_loss_pct": round(avg_loss_pct, 4),
            }
        payload["ev_buckets"] = ev_buckets
    return payload


def summarize_execution_state_metrics(df, states):
    if not states or len(df) < 5:
        return {}

    metrics = {}
    max_index = min(len(states), len(df) - 4)
    for i in range(max_index):
        state = states[i]
        execution = state.get("execution_state") or {}
        status = str(execution.get("status") or "stand_aside")
        if status not in {"enter", "prepare", "exit", "stand_aside"}:
            continue
        entry_price = float(df["Close"].iloc[i])
        future = df.iloc[i + 1 : i + 5]
        future_close = float(df["Close"].iloc[i + 4])
        future_high = float(future["High"].max())
        future_low = float(future["Low"].min())
        verdict = state.get("verdict")
        direction_success = False
        if verdict == "Bullish":
            direction_success = future_close > entry_price
        elif verdict == "Bearish":
            direction_success = future_close < entry_price
        if status == "exit":
            direction_success = abs(future_close - entry_price) / max(entry_price, 1e-8) > 0.003

        bucket = metrics.setdefault(status, {"count": 0, "wins": 0, "favorable": [], "adverse": []})
        bucket["count"] += 1
        bucket["wins"] += 1 if direction_success else 0
        bucket["favorable"].append(max(abs(future_high - entry_price), abs(entry_price - future_low)) / max(entry_price, 1e-8))
        bucket["adverse"].append(abs(future_close - entry_price) / max(entry_price, 1e-8))

    return {
        status: {
            "count": bucket["count"],
            "hit_rate": round(bucket["wins"] / bucket["count"], 4) if bucket["count"] else 0.0,
            "avg_favorable_move_pct": round(float(np.mean(bucket["favorable"])) * 100, 4) if bucket["favorable"] else 0.0,
            "avg_realized_move_pct": round(float(np.mean(bucket["adverse"])) * 100, 4) if bucket["adverse"] else 0.0,
        }
        for status, bucket in metrics.items()
    }


def summarize_tail_event_metrics(df, states):
    if not states or len(df) < 8:
        return {}

    max_index = min(len(states), len(df) - 4)
    model_total = 0
    model_wins = 0
    naive_total = 0
    naive_wins = 0
    tail_windows = 0

    for i in range(1, max_index):
        state = states[i]
        regime_state = state.get("regime_state") or {}
        feature_hits = regime_state.get("feature_hits") or {}
        warning_ladder = str(regime_state.get("warning_ladder") or "Normal")
        event_regime = str(regime_state.get("event_regime") or "normal")
        ts = pd.Timestamp(states[i].get("timestamp"))
        is_monday_open = bool(ts.weekday() == 0 and ts.hour in {0, 1, 2, 3})
        is_tail = (
            warning_ladder in {"High Breakout Risk", "Directional Expansion Likely", "Active Momentum Event"}
            or event_regime in {"breakout_watch", "range_expansion", "trend_acceleration", "event_risk"}
            or bool(feature_hits.get("session_reopen"))
            or bool(feature_hits.get("calendar_risk"))
            or is_monday_open
        )
        if not is_tail:
            continue
        tail_windows += 1

        entry = float(df["Close"].iloc[i])
        future_close = float(df["Close"].iloc[i + 4])
        realized = (future_close - entry) / max(entry, 1e-8)
        realized_dir = "Bullish" if realized > 0 else ("Bearish" if realized < 0 else "Neutral")

        verdict = str(state.get("verdict") or "Neutral")
        breakout_bias = str((state.get("forecast_state") or {}).get("breakoutBias") or "Neutral")
        directional_bias = str((state.get("forecast_state") or {}).get("directionalBias") or "Neutral")
        model_dir = breakout_bias if breakout_bias in {"Bullish", "Bearish"} else (
            directional_bias if directional_bias in {"Bullish", "Bearish"} else verdict
        )
        if model_dir in {"Bullish", "Bearish"}:
            model_total += 1
            if model_dir == realized_dir:
                model_wins += 1

        if breakout_bias in {"Bullish", "Bearish"}:
            naive_total += 1
            if breakout_bias == realized_dir:
                naive_wins += 1

    model_hit_rate = (model_wins / model_total) if model_total else 0.0
    naive_hit_rate = (naive_wins / naive_total) if naive_total else 0.0
    return {
        "tail_windows": int(tail_windows),
        "model_count": int(model_total),
        "model_hit_rate_4h": round(model_hit_rate, 4),
        "naive_count": int(naive_total),
        "naive_hit_rate_4h": round(naive_hit_rate, 4),
        "edge_vs_naive_4h": round(model_hit_rate - naive_hit_rate, 4),
    }


def summarize_quality_gate_metrics(trade_summary, big_move_metrics, execution_state_metrics, transition_metrics, tail_event_metrics, decision_quality_metrics=None):
    def _to_float(value, default=0.0):
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    watch = big_move_metrics.get("warning_watch", {}) if isinstance(big_move_metrics, dict) else {}
    prevalence = (big_move_metrics.get("label_prevalence") or {}) if isinstance(big_move_metrics, dict) else {}
    enter = execution_state_metrics.get("enter", {}) if isinstance(execution_state_metrics, dict) else {}
    tail = tail_event_metrics if isinstance(tail_event_metrics, dict) else {}
    decision = decision_quality_metrics if isinstance(decision_quality_metrics, dict) else {}
    true_60m_rate = _to_float(prevalence.get("true_60m_rate"), 0.0)
    precision_floor = max(0.42, min(0.72, true_60m_rate + 0.12))
    recall_floor = max(0.04, min(0.14, true_60m_rate * 0.10))
    checks = {
        "watch_precision_60m": _to_float(watch.get("precision_60m"), 0.0) >= precision_floor,
        "watch_recall_60m": _to_float(watch.get("recall_60m"), 0.0) >= recall_floor,
        "enter_hit_rate": _to_float(enter.get("hit_rate"), 0.0) >= 0.56,
        "whipsaw_rate": _to_float(transition_metrics.get("whipsaw_rate"), 1.0) <= 0.20,
        "tail_vs_naive_edge": _to_float(tail.get("edge_vs_naive_4h"), -1.0) >= 0.0,
        "tail_model_hit_rate": _to_float(tail.get("model_hit_rate_4h"), 0.0) >= 0.50,
        "abstain_correct_rate": _to_float(decision.get("abstain_correct_rate"), 0.0) >= 0.53,
        "regret_vs_stand_aside": _to_float(decision.get("avg_regret_vs_stand_aside_pct"), 0.0) <= 0.35,
        "roi_positive": _to_float(trade_summary.get("roi"), 0.0) > 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def summarize_outcome_log(df, states):
    if not states or len(df) < 5:
        return {}

    records = []
    max_index = min(len(states), len(df) - 4)
    for i in range(max_index):
        state = states[i]
        entry_price = float(df["Close"].iloc[i])
        outcomes = {}
        for bars in (1, 2, 4):
            future_price = float(df["Close"].iloc[i + bars])
            outcomes[f"{bars}h_return_pct"] = round(((future_price - entry_price) / max(entry_price, 1e-8)) * 100, 4)
        records.append(
            {
                "timestamp": state.get("timestamp"),
                "price": round(entry_price, 2),
                "verdict": state.get("verdict"),
                "confidence": state.get("confidence"),
                "regime_bucket": state.get("regime_bucket"),
                "warning_ladder": (state.get("regime_state") or {}).get("warning_ladder"),
                "event_regime": (state.get("regime_state") or {}).get("event_regime"),
                "trade_playbook_stage": (state.get("execution_state") or {}).get("status"),
                "breakout_bias": (state.get("forecast_state") or {}).get("breakoutBias"),
                "anti_chop_active": state.get("anti_chop_active"),
                "outcomes": outcomes,
            }
        )
    return {"records": records[:500]}

def run_backtest(ticker="XAU/USD", period="2y", interval="1h"):
    print(f"Fetching {period} of {interval} data for {ticker}...")
    try:
        df = fetch_history(period=period, interval=interval, ticker=ticker)
    except Exception as exc:
        print(f"Failed to fetch historical data from Twelve Data: {exc}")
        return

    print(f"Loaded {len(df)} historical candles.")
    print("Calculating indicators and applying strategy logic...")

    signals, states, enriched = generate_signals(df)
    trades = _simulate_from_signals(df, signals)

    print("-" * 50)
    print("BACKTEST RESULTS (Using Current Strategy Rules)")
    print("-" * 50)
    print(f"Total Trades: {len(trades)}")

    if len(trades) > 0:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100

        avg_win = np.mean([t['pnl'] for t in winning_trades]) * 100 if len(winning_trades) > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) * 100 if len(losing_trades) > 0 else 0

        capital = 10000
        for t in trades:
            capital = capital * (1 + t['pnl'])

        roi = ((capital - 10000) / 10000) * 100

        print(f"Win Rate:           {win_rate:.2f}%")
        print(f"Total ROI:          {roi:.2f}% (No Leverage)")
        print(f"Average Win:        {avg_win:.2f}%")
        print(f"Average Loss:       {avg_loss:.2f}%")
        print(f"Final Capital:      ${capital:.2f} (from $10,000 start)")
        summary = _trade_summary(trades)
        transition_metrics = summarize_transition_metrics(df, states)
        feature_hit_metrics = summarize_feature_hit_metrics(df, states)
        ablation_metrics = summarize_ablation_metrics(df, ACTIVE_BACKTEST_PARAMS)
        big_move_metrics = summarize_big_move_metrics(df, states)
        confidence_reliability = summarize_confidence_reliability(df, states)
        execution_state_metrics = summarize_execution_state_metrics(df, states)
        tail_event_metrics = summarize_tail_event_metrics(df, states)
        large_move_labels = summarize_large_move_labels(df, states)
        decision_quality_metrics = summarize_decision_quality_metrics(df, states)
        quality_gate = summarize_quality_gate_metrics(
            summary,
            big_move_metrics,
            execution_state_metrics,
            transition_metrics,
            tail_event_metrics,
            decision_quality_metrics,
        )
        outcome_log = summarize_outcome_log(df, states)
        print(f"Whipsaw Rate:       {transition_metrics.get('whipsaw_rate', 0.0):.4f}")
        print(f"False Entry Rate:   {transition_metrics.get('false_entry_rate', 0.0):.4f}")
        print(f"Avg Persistence:    {transition_metrics.get('avg_signal_persistence', 0.0):.2f} bars")
        print(f"Avg Favorable Exc.: {transition_metrics.get('avg_favorable_excursion', 0.0):.3f}%")
        print(f"Avg Adverse Exc.:   {transition_metrics.get('avg_adverse_excursion', 0.0):.3f}%")
        print("Feature Hit Metrics:")
        for feature_name, metrics in feature_hit_metrics.items():
            print(
                f"  {feature_name}: hits={metrics.get('hits', 0)} "
                f"accuracy={metrics.get('directional_accuracy', 0.0):.4f} "
                f"avg_3bar_ret={metrics.get('avg_3bar_directional_return_pct', 0.0):.4f}%"
            )
        print("Ablation Summary:")
        for label, metrics in ablation_metrics.get("delta_vs_full_model", {}).items():
            print(
                f"  {label}: roi_delta={metrics.get('roi_delta', 0.0):.2f} "
                f"win_rate_delta={metrics.get('win_rate_delta', 0.0):.2f} "
                f"trade_delta={metrics.get('trade_delta', 0)}"
            )
        if big_move_metrics:
            watch = big_move_metrics.get("warning_watch", {})
            directional = big_move_metrics.get("warning_directional", {})
            print("Big Move Metrics:")
            print(
                f"  watch_60m: precision={watch.get('precision_60m', 0.0):.4f} "
                f"recall={watch.get('recall_60m', 0.0):.4f}"
            )
            print(
                f"  watch_4h:  precision={watch.get('precision_4h', 0.0):.4f} "
                f"recall={watch.get('recall_4h', 0.0):.4f}"
            )
            print(
                f"  directional_4h: precision={directional.get('precision_4h', 0.0):.4f} "
                f"recall={directional.get('recall_4h', 0.0):.4f}"
            )
        if confidence_reliability:
            print("Confidence Reliability:")
            for regime_bucket, value in sorted((confidence_reliability.get("regime_confidence") or {}).items()):
                print(f"  {regime_bucket}: {float(value):.2f}%")
        if execution_state_metrics:
            print("Execution State Metrics:")
            for status, metrics in sorted(execution_state_metrics.items()):
                print(
                    f"  {status}: hit_rate={metrics.get('hit_rate', 0.0):.4f} "
                    f"count={metrics.get('count', 0)}"
                )
        if tail_event_metrics:
            print("Tail Event Metrics:")
            print(
                f"  model_4h_hit={tail_event_metrics.get('model_hit_rate_4h', 0.0):.4f} "
                f"naive_4h_hit={tail_event_metrics.get('naive_hit_rate_4h', 0.0):.4f} "
                f"edge={tail_event_metrics.get('edge_vs_naive_4h', 0.0):.4f} "
                f"windows={tail_event_metrics.get('tail_windows', 0)}"
            )
        if large_move_labels:
            print("Large Move Labels:")
            rates = large_move_labels.get("label_rates") or {}
            print(
                "  30m>300pips={:.4f} 60m>500pips={:.4f} 4h>1000pips={:.4f} compression->expansion={:.4f}".format(
                    float(rates.get("abs_move_gt_300pips_30m", 0.0) or 0.0),
                    float(rates.get("abs_move_gt_500pips_60m", 0.0) or 0.0),
                    float(rates.get("abs_move_gt_1000pips_4h", 0.0) or 0.0),
                    float(rates.get("directional_expansion_after_compression", 0.0) or 0.0),
                )
            )
        if decision_quality_metrics:
            print("Decision Quality:")
            print(
                f"  abstain_correct={decision_quality_metrics.get('abstain_correct_rate', 0.0):.4f} "
                f"enter_correct={decision_quality_metrics.get('enter_correct_rate', 0.0):.4f} "
                f"regret_vs_stand_aside={decision_quality_metrics.get('avg_regret_vs_stand_aside_pct', 0.0):.4f}%"
            )
        print(f"Quality Gate: {'PASS' if quality_gate.get('passed') else 'FAIL'}")
        for key, ok in (quality_gate.get("checks") or {}).items():
            print(f"  {key}: {'ok' if ok else 'fail'}")
        os.makedirs(os.path.dirname(FEATURE_REPORT_FILE), exist_ok=True)
        with open(FEATURE_REPORT_FILE, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "ticker": ticker,
                    "period": period,
                    "interval": interval,
                    "feature_hit_metrics": feature_hit_metrics,
                    "ablation_metrics": ablation_metrics,
                    "transition_metrics": transition_metrics,
                    "big_move_metrics": big_move_metrics,
                    "confidence_reliability": confidence_reliability,
                    "execution_state_metrics": execution_state_metrics,
                    "tail_event_metrics": tail_event_metrics,
                    "large_move_labels": large_move_labels,
                    "decision_quality_metrics": decision_quality_metrics,
                    "quality_gate": quality_gate,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        with open(CONFIDENCE_CALIBRATION_FILE, "w", encoding="utf-8") as handle:
            json.dump(confidence_reliability, handle, indent=2)
            handle.write("\n")
        with open(OUTCOME_SUMMARY_FILE, "w", encoding="utf-8") as handle:
            json.dump(outcome_log, handle, indent=2)
            handle.write("\n")
    else:
        print("No trades triggered.")

if __name__ == "__main__":
    run_backtest(period="730d")
