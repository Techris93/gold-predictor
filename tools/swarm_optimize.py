import itertools
import json
import os
import sys
import subprocess
import argparse
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import pandas as pd
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
from tools.twelvedata_market_data import fetch_history as fetch_td_history
STRATEGY_PARAMS_FILE = os.path.join(BASE_DIR, "config", "strategy_params.json")
BACKTEST_PARAMS_FILE = os.path.join(BASE_DIR, "config", "backtest_params.json")
LATEST_RESULT_FILE = os.path.join(BASE_DIR, "data", "swarm", "latest_result.json")
PROMOTED_RESULT_FILE = os.path.join(BASE_DIR, "data", "swarm", "promoted_result.json")
PROMOTION_DECISION_FILE = os.path.join(BASE_DIR, "data", "swarm", "promotion_decision.json")
CONFIDENCE_CALIBRATION_FILE = os.path.join(BASE_DIR, "tools", "reports", "confidence_calibration.json")
HISTORICAL_CACHE_DIR = os.path.join(BASE_DIR, "data", "swarm", "cache")

ENABLE_TELEGRAM_NOTIFICATIONS = os.environ.get("GOLD_PREDICTOR_ENABLE_TELEGRAM", "").strip().lower() in {"1", "true", "yes", "on"}
NOTIFY_TELEGRAM_TARGET = os.environ.get("GOLD_PREDICTOR_NOTIFY_TELEGRAM", "623118122") if ENABLE_TELEGRAM_NOTIFICATIONS else ""
NOTIFY_CHANNELS = [
    ("telegram", NOTIFY_TELEGRAM_TARGET),
]
MIN_ROI_IMPROVEMENT = 1.5
MIN_PASS_RATE = 0.30
MAX_PASS_RATE_DROP = 0.03
MIN_TRADE_COUNT = 10
MAX_DRAWDOWN_WORSENING_RATIO = 0.15
MIN_PROFIT_FACTOR = 1.4
MIN_EXPECTANCY = 0.0
FEATURE_PARAM_KEYS = (
    "ema_short",
    "ema_long",
    "rsi_window",
    "rsi_overbought",
    "rsi_oversold",
    "adx_window",
    "adx_trending_threshold",
    "adx_weak_trend_threshold",
    "atr_window",
    "atr_trending_percent_threshold",
    "cmf_window",
    "cmf_strong_buy_threshold",
    "cmf_strong_sell_threshold",
)
THRESHOLD_PARAM_KEYS = (
    "alignment_weight",
    "strong_volume_weight",
    "verdict_margin_threshold",
    "confidence_margin_multiplier",
    "rangebound_penalty",
    "mixed_alignment_penalty",
)


def _split_param_sets(params):
    normalized = normalize_strategy_params(params)
    feature_params = {key: normalized[key] for key in FEATURE_PARAM_KEYS}
    score_params = {key: value for key, value in normalized.items() if key not in FEATURE_PARAM_KEYS}
    return feature_params, score_params


def _merge_param_sets(feature_params, score_params):
    merged = {}
    merged.update(feature_params or {})
    merged.update(score_params or {})
    return normalize_strategy_params(merged)


def generate_signals_custom(df_or_enriched, params):
    """Generates signals using the live predictor scoring engine."""
    params = normalize_strategy_params(params)
    if isinstance(df_or_enriched, pd.DataFrame) and "EMA_TREND" in df_or_enriched.columns:
        enriched = df_or_enriched
    else:
        enriched = prepare_historical_features(df_or_enriched, params)
    signals = pd.Series("Neutral", index=enriched.index)

    for i in range(len(enriched)):
        ta_payload = build_ta_payload_from_row(enriched.iloc[i], params)
        verdict = compute_prediction_from_ta(ta_payload)["verdict"]
        if verdict == "Bullish":
            signals.iloc[i] = "Buy"
        elif verdict == "Bearish":
            signals.iloc[i] = "Sell"

    return signals


def _safe_round(value, digits=4):
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return None


def test_params(args):
    """Worker function to test a specific parameter set."""
    frame, params = args
    signals = generate_signals_custom(frame, params)

    capital = 10000.0
    peak_capital = capital
    max_drawdown = 0.0
    position = 0
    entry_price = 0.0
    trades = []

    def close_position(next_open, next_signal_idx):
        nonlocal capital, peak_capital, max_drawdown, position, entry_price
        if position == 0:
            return
        if position == 1:
            pnl = (next_open - entry_price) / entry_price
        else:
            pnl = (entry_price - next_open) / entry_price
        capital *= 1 + pnl
        peak_capital = max(peak_capital, capital)
        if peak_capital > 0:
            max_drawdown = max(max_drawdown, (peak_capital - capital) / peak_capital)
        trades.append(
            {
                "trade_index": next_signal_idx,
                "direction": "long" if position == 1 else "short",
                "entry_price": entry_price,
                "exit_price": next_open,
                "pnl": pnl,
            }
        )
        position = 0
        entry_price = 0.0

    for i in range(len(frame) - 1):
        signal = signals.iloc[i]
        next_open = float(frame["Open"].iloc[i + 1])

        if position == 0:
            if signal == "Buy":
                position = 1
                entry_price = next_open
            elif signal == "Sell":
                position = -1
                entry_price = next_open
            continue

        if position == 1 and signal == "Sell":
            close_position(next_open, i + 1)
            position = -1
            entry_price = next_open
        elif position == -1 and signal == "Buy":
            close_position(next_open, i + 1)
            position = 1
            entry_price = next_open

    if position != 0:
        final_close = float(frame["Close"].iloc[-1])
        close_position(final_close, len(frame) - 1)

    roi = ((capital - 10000) / 10000) * 100
    trade_pnls = [trade["pnl"] for trade in trades]
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl < 0]
    pass_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    expectancy = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "params": params,
        "summary": {
            "roi": _safe_round(roi, 2),
            "trades": len(trade_pnls),
            "pass_rate": _safe_round(pass_rate),
            "max_drawdown": _safe_round(max_drawdown),
            "profit_factor": _safe_round(profit_factor),
            "expectancy": _safe_round(expectancy),
            "ending_capital": _safe_round(capital, 2),
        },
    }


def _write_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _write_confidence_calibration(best_result):
    summary = (best_result or {}).get("summary") or {}
    pass_rate = summary.get("pass_rate")
    if not isinstance(pass_rate, (int, float)):
        return

    calibrated = round(max(50.0, min(95.0, 45.0 + (float(pass_rate) * 100.0))), 2)
    payload = {
        "trend|Bullish|High|stable": calibrated,
        "trend|Bearish|High|stable": calibrated,
        "transition|Bullish|Medium|stable": round(max(50.0, calibrated - 10.0), 2),
        "transition|Bearish|Medium|stable": round(max(50.0, calibrated - 10.0), 2),
        "range|Neutral|Low|unstable": 52.0,
        "event-risk|Neutral|Low|unstable": 50.0,
    }
    _write_json(CONFIDENCE_CALIBRATION_FILE, payload)


def _notify_user(message):
    openclaw_bin = (
        os.environ.get("OPENCLAW_BIN")
        or shutil.which("openclaw")
        or os.path.expanduser("~/.npm-global/bin/openclaw")
        or "/Users/chrixchange/.npm-global/bin/openclaw"
    )
    for channel, target in NOTIFY_CHANNELS:
        if not target:
            continue
        try:
            subprocess.run(
                [
                    openclaw_bin,
                    "message",
                    "send",
                    "--channel",
                    channel,
                    "--target",
                    target,
                    "--message",
                    message,
                ],
                cwd=BASE_DIR,
                check=True,
                timeout=60,
                capture_output=True,
                text=True,
            )
            print(f"📣 Sent {channel} notification to {target}")
        except Exception as e:
            print(f"⚠️ Failed to send {channel} notification to {target}: {e}")


def _read_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_file_path(ticker, period, interval):
    safe_ticker = ticker.replace("/", "_").replace("=", "_").replace("^", "")
    safe_period = period.replace("/", "_")
    safe_interval = interval.replace("/", "_")
    return os.path.join(HISTORICAL_CACHE_DIR, f"{safe_ticker}_{safe_period}_{safe_interval}.csv")


def _normalize_history_frame(frame):
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    df = frame.copy()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [col for col in ["Open", "High", "Low", "Close"] if col in df.columns]
    if required:
        df = df.dropna(subset=required)

    return df


def _load_cached_history(ticker, period, interval):
    cache_path = _cache_file_path(ticker, period, interval)
    if not os.path.exists(cache_path):
        return None
    try:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = _normalize_history_frame(df)
        return df if not df.empty else None
    except Exception:
        return None


def _save_cached_history(ticker, period, interval, df):
    cache_path = _cache_file_path(ticker, period, interval)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    _normalize_history_frame(df).to_csv(cache_path)


def _fetch_historical_data(ticker, period, interval, retries=2):
    errors = []
    for attempt in range(1, retries + 1):
        try:
            frame = _normalize_history_frame(fetch_td_history(period=period, interval=interval, ticker=ticker))
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                _save_cached_history(ticker, period, interval, frame)
                return frame, None
            errors.append(f"Twelve Data attempt {attempt} returned no data")
        except Exception as exc:
            errors.append(f"Twelve Data attempt {attempt} failed: {exc}")

        if attempt < retries:
            time.sleep(2)

    cached = _load_cached_history(ticker, period, interval)
    if cached is not None and not cached.empty:
        return cached, "Using cached historical data after Twelve Data fetch failure. " + "; ".join(errors)

    return None, "; ".join(errors)


def _build_predict_params(params):
    return {
        "ema_short": params["ema_short"],
        "ema_long": params["ema_long"],
        "rsi_window": 14,
        "rsi_overbought": params["rsi_overbought"],
        "rsi_oversold": params["rsi_oversold"],
        "adx_window": 14,
        "adx_trending_threshold": 22,
        "adx_weak_trend_threshold": 18,
        "atr_window": 14,
        "atr_trending_percent_threshold": 0.25,
        "cmf_window": params["cmf_window"],
        "cmf_strong_buy_threshold": 0.10,
        "cmf_strong_sell_threshold": -0.10,
        "alignment_weight": params["alignment_weight"],
        "strong_volume_weight": params["strong_volume_weight"],
        "verdict_margin_threshold": params["verdict_margin_threshold"],
        "confidence_margin_multiplier": params["confidence_margin_multiplier"],
        "rangebound_penalty": params["rangebound_penalty"],
        "mixed_alignment_penalty": params["mixed_alignment_penalty"],
        "mtf_intervals": ["15min", "1h", "4h"],
    }


def _build_backtest_params(params):
    return {
        "ema_short": params["ema_short"],
        "ema_long": params["ema_long"],
        "rsi_window": 14,
        "rsi_overbought": params["rsi_overbought"],
        "rsi_oversold": params["rsi_oversold"],
        "adx_window": 14,
        "adx_trending_threshold": 22,
        "adx_weak_trend_threshold": 18,
        "atr_window": 14,
        "atr_trending_percent_threshold": 0.25,
        "cmf_window": params["cmf_window"],
        "cmf_strong_buy_threshold": 0.10,
        "cmf_strong_sell_threshold": -0.10,
        "alignment_weight": params["alignment_weight"],
        "strong_volume_weight": params["strong_volume_weight"],
        "verdict_margin_threshold": params["verdict_margin_threshold"],
        "confidence_margin_multiplier": params["confidence_margin_multiplier"],
        "rangebound_penalty": params["rangebound_penalty"],
        "mixed_alignment_penalty": params["mixed_alignment_penalty"],
        "mtf_intervals": ["15min", "1h", "4h"],
    }


def _evaluate_baseline(df):
    strategy_params = normalize_strategy_params(_read_json(STRATEGY_PARAMS_FILE) or {})
    baseline_params = {
        "ema_short": int(strategy_params.get("ema_short", 20)),
        "ema_long": int(strategy_params.get("ema_long", 50)),
        "rsi_overbought": int(strategy_params.get("rsi_overbought", 70)),
        "rsi_oversold": int(strategy_params.get("rsi_oversold", 20)),
        "cmf_window": int(strategy_params.get("cmf_window", 14)),
        "alignment_weight": float(strategy_params.get("alignment_weight", 1.2)),
        "strong_volume_weight": float(strategy_params.get("strong_volume_weight", 2.0)),
        "verdict_margin_threshold": float(strategy_params.get("verdict_margin_threshold", 1.2)),
        "confidence_margin_multiplier": float(strategy_params.get("confidence_margin_multiplier", 8.0)),
        "rangebound_penalty": float(strategy_params.get("rangebound_penalty", 8.0)),
        "mixed_alignment_penalty": float(strategy_params.get("mixed_alignment_penalty", 6.0)),
    }
    return test_params((df, baseline_params))


def _metric(summary, name, default=None):
    value = (summary or {}).get(name, default)
    return value if isinstance(value, (int, float)) else default


def _compare_candidate_to_baseline(candidate, baseline):
    candidate_summary = candidate.get("summary") or {}
    baseline_summary = baseline.get("summary") or {}

    candidate_roi = _metric(candidate_summary, "roi", 0.0)
    baseline_roi = _metric(baseline_summary, "roi", 0.0)
    candidate_pass_rate = _metric(candidate_summary, "pass_rate", 0.0)
    baseline_pass_rate = _metric(baseline_summary, "pass_rate", 0.0)
    candidate_trades = _metric(candidate_summary, "trades", 0)
    baseline_drawdown = _metric(baseline_summary, "max_drawdown", 0.0)
    candidate_drawdown = _metric(candidate_summary, "max_drawdown", 0.0)
    candidate_profit_factor = _metric(candidate_summary, "profit_factor", 0.0)
    candidate_expectancy = _metric(candidate_summary, "expectancy", 0.0)

    roi_improvement = candidate_roi - baseline_roi
    pass_rate_delta = candidate_pass_rate - baseline_pass_rate
    allowed_drawdown = baseline_drawdown * (1 + MAX_DRAWDOWN_WORSENING_RATIO) if baseline_drawdown > 0 else MAX_DRAWDOWN_WORSENING_RATIO

    # This strategy family wins with asymmetric payoff more than with raw hit rate,
    # so pass rate is only a floor while profit factor and expectancy stay primary.
    checks = {
        "roi_improvement": {
            "pass": roi_improvement >= MIN_ROI_IMPROVEMENT,
            "value": _safe_round(roi_improvement, 2),
            "threshold": MIN_ROI_IMPROVEMENT,
        },
        "pass_rate_floor": {
            "pass": candidate_pass_rate >= MIN_PASS_RATE,
            "value": _safe_round(candidate_pass_rate),
            "threshold": MIN_PASS_RATE,
        },
        "pass_rate_regression": {
            "pass": pass_rate_delta >= -MAX_PASS_RATE_DROP,
            "value": _safe_round(pass_rate_delta),
            "threshold": -MAX_PASS_RATE_DROP,
        },
        "trade_count": {
            "pass": candidate_trades >= MIN_TRADE_COUNT,
            "value": int(candidate_trades),
            "threshold": MIN_TRADE_COUNT,
        },
        "drawdown_guardrail": {
            "pass": candidate_drawdown <= allowed_drawdown,
            "value": _safe_round(candidate_drawdown),
            "threshold": _safe_round(allowed_drawdown),
        },
        "profit_factor": {
            "pass": candidate_profit_factor >= MIN_PROFIT_FACTOR,
            "value": _safe_round(candidate_profit_factor),
            "threshold": MIN_PROFIT_FACTOR,
        },
        "expectancy": {
            "pass": candidate_expectancy > MIN_EXPECTANCY,
            "value": _safe_round(candidate_expectancy),
            "threshold": MIN_EXPECTANCY,
        },
    }

    promote = all(item["pass"] for item in checks.values())
    failed_checks = [name for name, item in checks.items() if not item["pass"]]
    reason = (
        "Promoted: candidate cleared ROI, robustness, and risk gates."
        if promote
        else f"Not promoted: failed gates -> {', '.join(failed_checks)}"
    )

    return {
        "promote": promote,
        "promotion_reason": reason,
        "checks": checks,
        "candidate": candidate,
        "baseline": baseline,
    }


def _format_top_results(results):
    payload = []
    for idx, result in enumerate(results[:10]):
        params = result.get("params") or {}
        summary = result.get("summary") or {}
        payload.append(
            {
                "rank": idx + 1,
                **params,
                "roi": summary.get("roi"),
                "trades": summary.get("trades"),
                "pass_rate": summary.get("pass_rate"),
                "max_drawdown": summary.get("max_drawdown"),
                "profit_factor": summary.get("profit_factor"),
                "expectancy": summary.get("expectancy"),
            }
        )
    return payload


def _build_param_grid(reduced=False):
    if reduced:
        ema_shorts = [12, 20]
        ema_longs = [50, 100]
        rsi_obs = [70, 75]
        rsi_oss = [20, 25]
        cmf_wins = [14]
        alignment_weights = [1.0, 1.2]
        strong_volume_weights = [1.5, 2.0]
        verdict_margin_thresholds = [1.0, 1.2]
        confidence_margin_multipliers = [8.0]
        rangebound_penalties = [8.0]
        mixed_alignment_penalties = [6.0]
    else:
        ema_shorts = [9, 12, 20]
        ema_longs = [26, 50, 100]
        rsi_obs = [70, 75, 80]
        rsi_oss = [20, 25, 30]
        cmf_wins = [14, 20]
        alignment_weights = [1.0, 1.2]
        strong_volume_weights = [1.5, 2.0]
        verdict_margin_thresholds = [1.0, 1.2]
        confidence_margin_multipliers = [7.0, 8.0]
        rangebound_penalties = [6.0, 8.0]
        mixed_alignment_penalties = [4.0, 6.0]

    param_combinations = itertools.product(
        ema_shorts,
        ema_longs,
        rsi_obs,
        rsi_oss,
        cmf_wins,
        alignment_weights,
        strong_volume_weights,
        verdict_margin_thresholds,
        confidence_margin_multipliers,
        rangebound_penalties,
        mixed_alignment_penalties,
    )
    return [
        {
            "ema_short": ema_s,
            "ema_long": ema_l,
            "rsi_overbought": rsi_ob,
            "rsi_oversold": rsi_os,
            "cmf_window": cmf_w,
            "alignment_weight": alignment_w,
            "strong_volume_weight": volume_w,
            "verdict_margin_threshold": verdict_margin,
            "confidence_margin_multiplier": conf_margin,
            "rangebound_penalty": range_penalty,
            "mixed_alignment_penalty": align_penalty,
        }
        for (
            ema_s,
            ema_l,
            rsi_ob,
            rsi_os,
            cmf_w,
            alignment_w,
            volume_w,
            verdict_margin,
            conf_margin,
            range_penalty,
            align_penalty,
        ) in param_combinations
        if ema_s < ema_l
    ]


def _build_threshold_only_grid(reduced=False):
    strategy_params = normalize_strategy_params(_read_json(STRATEGY_PARAMS_FILE) or {})
    base_params = {
        "ema_short": int(strategy_params.get("ema_short", 20)),
        "ema_long": int(strategy_params.get("ema_long", 50)),
        "rsi_overbought": int(strategy_params.get("rsi_overbought", 70)),
        "rsi_oversold": int(strategy_params.get("rsi_oversold", 20)),
        "cmf_window": int(strategy_params.get("cmf_window", 14)),
    }

    if reduced:
        alignment_weights = [1.0, 1.2]
        strong_volume_weights = [1.5, 2.0]
        verdict_margin_thresholds = [1.0, 1.2]
        confidence_margin_multipliers = [8.0]
        rangebound_penalties = [8.0]
        mixed_alignment_penalties = [6.0]
    else:
        alignment_weights = [1.0, 1.2]
        strong_volume_weights = [1.5, 2.0]
        verdict_margin_thresholds = [1.0, 1.2]
        confidence_margin_multipliers = [7.0, 8.0]
        rangebound_penalties = [6.0, 8.0]
        mixed_alignment_penalties = [4.0, 6.0]

    param_combinations = itertools.product(
        alignment_weights,
        strong_volume_weights,
        verdict_margin_thresholds,
        confidence_margin_multipliers,
        rangebound_penalties,
        mixed_alignment_penalties,
    )
    return [
        {
            **base_params,
            "alignment_weight": alignment_w,
            "strong_volume_weight": volume_w,
            "verdict_margin_threshold": verdict_margin,
            "confidence_margin_multiplier": conf_margin,
            "rangebound_penalty": range_penalty,
            "mixed_alignment_penalty": align_penalty,
        }
        for (
            alignment_w,
            volume_w,
            verdict_margin,
            conf_margin,
            range_penalty,
            align_penalty,
        ) in param_combinations
    ]


def _evaluate_param_grid(df, valid_params, parallel=True):
    grouped = {}
    for params in valid_params:
        feature_params, score_params = _split_param_sets(params)
        feature_key = tuple(sorted(feature_params.items()))
        grouped.setdefault(feature_key, {"feature_params": feature_params, "score_params_list": []})
        grouped[feature_key]["score_params_list"].append(score_params)

    results = []
    if parallel:
        args_list = []
        for item in grouped.values():
            enriched = prepare_historical_features(df, item["feature_params"])
            for score_params in item["score_params_list"]:
                args_list.append((enriched, _merge_param_sets(item["feature_params"], score_params)))
        with ProcessPoolExecutor() as executor:
            for result in executor.map(test_params, args_list):
                results.append(result)
        return results

    total = len(valid_params)
    completed = 0
    for item in grouped.values():
        enriched = prepare_historical_features(df, item["feature_params"])
        for score_params in item["score_params_list"]:
            params = _merge_param_sets(item["feature_params"], score_params)
            results.append(test_params((enriched, params)))
            completed += 1
            if completed % 16 == 0 or completed == total:
                print(f"Completed {completed}/{total} candidates...", flush=True)
    return results


def run_swarm(reduced=False, serial=False, threshold_only=False, period="730d", interval="1h", ticker="XAU/USD"):
    print("🐝 Igniting Autoresearch Swarm for Gold Strategy Optimization...", flush=True)
    print(f"Fetching historical data ({period}, {interval} timeframe) for {ticker}...", flush=True)

    df, fetch_error = _fetch_historical_data(ticker, period, interval)
    if df is None or df.empty:
        message = (
            "Gold predictor swarm run failed: historical data fetch returned no data, "
            "so no optimization or promotion decision was made. "
            f"Ticker={ticker}, period={period}, interval={interval}. "
            f"Details: {fetch_error or 'unknown fetch error'}"
        )
        print("Failed to fetch historical data.", flush=True)
        if fetch_error:
            print(f"Fetch details: {fetch_error}", flush=True)
        _notify_user(message)
        return
    if fetch_error:
        print(f"Data warning: {fetch_error}", flush=True)

    valid_params = (
        _build_threshold_only_grid(reduced=reduced)
        if threshold_only
        else _build_param_grid(reduced=reduced)
    )

    print(f"🧬 Generated {len(valid_params)} unique strategy genetic combinations.", flush=True)
    print("Simulating trading strategies across historical candles. Please wait...\n", flush=True)
    if reduced:
        print("Using reduced search mode.", flush=True)
    if threshold_only:
        print("Using threshold-only search mode around the current live indicator settings.", flush=True)
    if serial:
        print("Using serial evaluation mode.", flush=True)
    results = _evaluate_param_grid(df, valid_params, parallel=not serial)

    results.sort(key=lambda item: item["summary"]["roi"], reverse=True)

    print("🏆 SWARM OPTIMIZATION LEADERBOARD (Top 5 Strategies)\n", flush=True)
    print(f"{'Rank':<5} | {'EMA':<9} | {'RSI':<9} | {'CMF':<5} | {'AlignW':<6} | {'VolW':<5} | {'Margin':<6} | {'ROI':<8}", flush=True)
    print("-" * 96, flush=True)
    for i, result in enumerate(results[:5]):
        params = result["params"]
        summary = result["summary"]
        print(
            f"#{i+1:<4} | {params['ema_short']}/{params['ema_long']:<6} | "
            f"{params['rsi_overbought']}/{params['rsi_oversold']:<6} | "
            f"{params['cmf_window']:<5} | {params['alignment_weight']:<6} | "
            f"{params['strong_volume_weight']:<5} | {params['verdict_margin_threshold']:<6} | "
            f"{summary['roi']:>6.2f}%",
            flush=True,
        )

    best_result = results[0]
    baseline_result = _evaluate_baseline(df)
    promotion_decision = _compare_candidate_to_baseline(best_result, baseline_result)
    generated_at = datetime.now(timezone.utc).isoformat()

    latest_payload = {
        "status": "success",
        "generated_at": generated_at,
        "promote": promotion_decision["promote"],
        "promotion_reason": promotion_decision["promotion_reason"],
        "best_params": {
            **best_result["params"],
            "roi": best_result["summary"]["roi"],
        },
        "summary": best_result["summary"],
        "top_results": _format_top_results(results),
        "baseline": baseline_result,
        "checks": promotion_decision["checks"],
    }

    decision_payload = {
        "generated_at": generated_at,
        "promote": promotion_decision["promote"],
        "promotion_reason": promotion_decision["promotion_reason"],
        "candidate": best_result,
        "baseline": baseline_result,
        "checks": promotion_decision["checks"],
    }

    print("\n📝 Writing latest swarm run artifacts...", flush=True)
    _write_json(LATEST_RESULT_FILE, latest_payload)
    _write_json(PROMOTION_DECISION_FILE, decision_payload)
    _write_confidence_calibration(best_result)

    if not promotion_decision["promote"]:
        print(f"⚖️ Promotion gate rejected candidate. {promotion_decision['promotion_reason']}", flush=True)
        print("📦 Updated latest run artifacts only. Active promoted strategy remains unchanged.", flush=True)
        candidate = best_result["params"]
        candidate_summary = best_result["summary"]
        rejection_message = (
            "Gold predictor swarm completed with no promotion. "
            f"Best candidate was EMA {candidate['ema_short']}/{candidate['ema_long']}, "
            f"RSI {candidate['rsi_overbought']}/{candidate['rsi_oversold']}, CMF {candidate['cmf_window']}, "
            f"ROI {candidate_summary['roi']:.2f}%. {promotion_decision['promotion_reason']}"
        )
        _notify_user(rejection_message)
        return

    print("\n⚡ Promotion gate passed. Updating active strategy JSON...", flush=True)
    predict_params = _build_predict_params(best_result["params"])
    backtest_params = _build_backtest_params(best_result["params"])
    promoted_payload = {
        **latest_payload,
        "status": "promoted",
    }

    try:
        _write_json(STRATEGY_PARAMS_FILE, predict_params)
        _write_json(BACKTEST_PARAMS_FILE, backtest_params)
        _write_json(PROMOTED_RESULT_FILE, promoted_payload)

        print(
            "✅ Promoted strategy to active config: "
            f"EMA {best_result['params']['ema_short']}/{best_result['params']['ema_long']}, "
            f"RSI {best_result['params']['rsi_overbought']}/{best_result['params']['rsi_oversold']}, "
            f"CMF {best_result['params']['cmf_window']}",
            flush=True,
        )

        status_output = subprocess.run(["git", "status", "--porcelain"], cwd=BASE_DIR, capture_output=True, text=True)
        tracked_targets = [
            "config/strategy_params.json",
            "config/backtest_params.json",
            "data/swarm/latest_result.json",
            "data/swarm/promoted_result.json",
            "data/swarm/promotion_decision.json",
        ]
        commit_hash = None
        if any(target in status_output.stdout for target in tracked_targets):
            commit_msg = (
                f"Auto-Evolve: New Best Strategy Found | ROI: {best_result['summary']['roi']:.2f}% | "
                f"EMA: {best_result['params']['ema_short']}/{best_result['params']['ema_long']} "
                f"CMF: {best_result['params']['cmf_window']}"
            )
            subprocess.run(["git", "add", *tracked_targets], cwd=BASE_DIR, check=True)
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=BASE_DIR, check=True)
            commit_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, check=True, capture_output=True, text=True).stdout.strip()
            print("💾 Saved promoted strategy state. You can now run `git push` to deploy the new strategy.", flush=True)
        else:
            print("⚖️ Promotion passed but tracked artifacts are unchanged. No commit needed.", flush=True)

        promoted_message = (
            "Gold predictor promoted a new strategy. "
            f"EMA {best_result['params']['ema_short']}/{best_result['params']['ema_long']}, "
            f"RSI {best_result['params']['rsi_overbought']}/{best_result['params']['rsi_oversold']}, "
            f"CMF {best_result['params']['cmf_window']}, ROI {best_result['summary']['roi']:.2f}%."
        )
        if commit_hash:
            promoted_message += f" Commit {commit_hash}."
        _notify_user(promoted_message)
    except Exception as e:
        failure_message = f"Gold predictor swarm run failed while updating promoted strategy state: {e}"
        print("⚠️ Failed to update promoted strategy state:", e, flush=True)
        _notify_user(failure_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize gold predictor strategy parameters.")
    parser.add_argument("--reduced", action="store_true", help="Run a reduced parameter grid for faster same-day tuning.")
    parser.add_argument("--serial", action="store_true", help="Evaluate candidates serially instead of using a process pool.")
    parser.add_argument("--threshold-only", action="store_true", help="Keep indicator settings fixed and tune only live decision thresholds.")
    parser.add_argument("--period", default="730d", help="Historical window to fetch, for example 365d or 730d.")
    parser.add_argument("--interval", default="1h", help="Candle interval to fetch, for example 1h.")
    parser.add_argument("--ticker", default="XAU/USD", help="Market symbol to backtest against.")
    args = parser.parse_args()
    try:
        run_swarm(
            reduced=args.reduced,
            serial=args.serial,
            threshold_only=args.threshold_only,
            period=args.period,
            interval=args.interval,
            ticker=args.ticker,
        )
    except Exception as e:
        print(f"❌ Unhandled swarm failure: {e}", flush=True)
        _notify_user(f"Gold predictor swarm run failed with an unhandled error: {e}")
        raise
