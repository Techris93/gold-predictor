import itertools
import json
import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools.signal_engine import (
    build_ta_payload_from_row,
    compute_prediction_from_ta,
    normalize_strategy_params,
    prepare_historical_features,
)
STRATEGY_PARAMS_FILE = os.path.join(BASE_DIR, "config", "strategy_params.json")
BACKTEST_PARAMS_FILE = os.path.join(BASE_DIR, "config", "backtest_params.json")
LATEST_RESULT_FILE = os.path.join(BASE_DIR, "data", "swarm", "latest_result.json")
PROMOTED_RESULT_FILE = os.path.join(BASE_DIR, "data", "swarm", "promoted_result.json")
PROMOTION_DECISION_FILE = os.path.join(BASE_DIR, "data", "swarm", "promotion_decision.json")

NOTIFY_WHATSAPP_TARGET = os.environ.get("GOLD_PREDICTOR_NOTIFY_WHATSAPP", "+905528493671")
NOTIFY_TELEGRAM_TARGET = os.environ.get("GOLD_PREDICTOR_NOTIFY_TELEGRAM", "623118122")
NOTIFY_CHANNELS = [
    ("whatsapp", NOTIFY_WHATSAPP_TARGET),
    ("telegram", NOTIFY_TELEGRAM_TARGET),
]

MIN_ROI_IMPROVEMENT = 1.5
MIN_PASS_RATE = 0.60
MAX_PASS_RATE_DROP = 0.03
MIN_TRADE_COUNT = 10
MAX_DRAWDOWN_WORSENING_RATIO = 0.15
MIN_PROFIT_FACTOR = 1.2
MIN_EXPECTANCY = 0.0


def generate_signals_custom(df, params):
    """Generates signals using the live predictor scoring engine."""
    params = normalize_strategy_params(params)
    enriched = prepare_historical_features(df, params)
    signals = pd.Series("Neutral", index=enriched.index)
    sentiment_summary = {"label": "Neutral"}

    for i in range(len(enriched)):
        ta_payload = build_ta_payload_from_row(enriched.iloc[i], params)
        verdict = compute_prediction_from_ta(ta_payload, sentiment_summary)["verdict"]
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
    df, params = args
    signals = generate_signals_custom(df, params)

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

    for i in range(len(df) - 1):
        signal = signals.iloc[i]
        next_open = float(df["Open"].iloc[i + 1])

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
        final_close = float(df["Close"].iloc[-1])
        close_position(final_close, len(df) - 1)

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


def _notify_user(message):
    for channel, target in NOTIFY_CHANNELS:
        if not target:
            continue
        try:
            subprocess.run(
                [
                    "openclaw",
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


def run_swarm():
    print("🐝 Igniting Autoresearch Swarm for Gold Strategy Optimization...")
    print("Fetching historical data (730 days, 1H timeframe)...")

    df = yf.Ticker("GC=F").history(period="730d", interval="1h")
    if df.empty:
        message = "Gold predictor swarm run failed: historical data fetch returned no data, so no optimization or promotion decision was made."
        print("Failed to fetch historical data.")
        _notify_user(message)
        return

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
    valid_params = [
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

    print(f"🧬 Generated {len(valid_params)} unique strategy genetic combinations.")
    print("Simulating trading strategies across 13,000+ historical hours. Please wait...\n")

    results = []
    with ProcessPoolExecutor() as executor:
        args_list = [(df, p) for p in valid_params]
        for r in executor.map(test_params, args_list):
            results.append(r)

    results.sort(key=lambda item: item["summary"]["roi"], reverse=True)

    print("🏆 SWARM OPTIMIZATION LEADERBOARD (Top 5 Strategies)\n")
    print(f"{'Rank':<5} | {'EMA':<9} | {'RSI':<9} | {'CMF':<5} | {'AlignW':<6} | {'VolW':<5} | {'Margin':<6} | {'ROI':<8}")
    print("-" * 96)
    for i, result in enumerate(results[:5]):
        params = result["params"]
        summary = result["summary"]
        print(
            f"#{i+1:<4} | {params['ema_short']}/{params['ema_long']:<6} | "
            f"{params['rsi_overbought']}/{params['rsi_oversold']:<6} | "
            f"{params['cmf_window']:<5} | {params['alignment_weight']:<6} | "
            f"{params['strong_volume_weight']:<5} | {params['verdict_margin_threshold']:<6} | "
            f"{summary['roi']:>6.2f}%"
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

    print("\n📝 Writing latest swarm run artifacts...")
    _write_json(LATEST_RESULT_FILE, latest_payload)
    _write_json(PROMOTION_DECISION_FILE, decision_payload)

    if not promotion_decision["promote"]:
        print(f"⚖️ Promotion gate rejected candidate. {promotion_decision['promotion_reason']}")
        print("📦 Updated latest run artifacts only. Active promoted strategy remains unchanged.")
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

    print("\n⚡ Promotion gate passed. Updating active strategy JSON...")
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
            f"CMF {best_result['params']['cmf_window']}"
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
            print("💾 Saved promoted strategy state. You can now run `git push` to deploy the new strategy.")
        else:
            print("⚖️ Promotion passed but tracked artifacts are unchanged. No commit needed.")

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
        print("⚠️ Failed to update promoted strategy state:", e)
        _notify_user(failure_message)


if __name__ == "__main__":
    try:
        run_swarm()
    except Exception as e:
        print(f"❌ Unhandled swarm failure: {e}")
        _notify_user(f"Gold predictor swarm run failed with an unhandled error: {e}")
        raise
