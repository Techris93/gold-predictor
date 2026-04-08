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

def generate_signals(df, params=None):
    params = normalize_strategy_params(params or ACTIVE_BACKTEST_PARAMS)
    enriched = prepare_historical_features(df, params)
    signals = pd.Series('Neutral', index=enriched.index)
    states = []

    for i in range(len(enriched)):
        ta_payload = build_ta_payload_from_row(enriched.iloc[i], params)
        prediction = compute_prediction_from_ta(ta_payload)
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
                "direction_score": prediction.get("directionScore"),
                "tradeability_score": prediction.get("tradeabilityScore"),
                "exit_risk_score": prediction.get("exitRiskScore"),
                "stability_score": prediction.get("stabilityScore"),
                "feature_hits": prediction.get("FeatureHits", {}),
                "regime_state": prediction.get("RegimeState", {}),
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
        true_60m = future_move_60m >= 5.0
        true_4h = future_move_4h >= 10.0
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
            "big_move_60m_price_points": 5.0,
            "big_move_4h_price_points": 10.0,
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
        transition_metrics = summarize_transition_metrics(df, states)
        feature_hit_metrics = summarize_feature_hit_metrics(df, states)
        ablation_metrics = summarize_ablation_metrics(df, ACTIVE_BACKTEST_PARAMS)
        big_move_metrics = summarize_big_move_metrics(df, states)
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
                },
                handle,
                indent=2,
            )
            handle.write("\n")
    else:
        print("No trades triggered.")

if __name__ == "__main__":
    run_backtest(period="730d")
