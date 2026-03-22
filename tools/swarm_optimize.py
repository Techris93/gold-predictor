import itertools
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import ta
import yfinance as yf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_PARAMS_FILE = os.path.join(BASE_DIR, "config", "strategy_params.json")
BACKTEST_PARAMS_FILE = os.path.join(BASE_DIR, "config", "backtest_params.json")
LATEST_RESULT_FILE = os.path.join(BASE_DIR, "data", "swarm", "latest_result.json")
PROMOTED_RESULT_FILE = os.path.join(BASE_DIR, "data", "swarm", "promoted_result.json")
PROMOTION_DECISION_FILE = os.path.join(BASE_DIR, "data", "swarm", "promotion_decision.json")

MIN_ROI_IMPROVEMENT = 1.5
MIN_PASS_RATE = 0.60
MAX_PASS_RATE_DROP = 0.03
MIN_TRADE_COUNT = 10
MAX_DRAWDOWN_WORSENING_RATIO = 0.15
MIN_PROFIT_FACTOR = 1.2
MIN_EXPECTANCY = 0.0


def generate_signals_custom(df, ema_short_win, ema_long_win, rsi_ob, rsi_os, cmf_win):
    """Generates signals using custom parameters for the swarm optimization."""
    df = df.copy()

    df["EMA_S"] = ta.trend.EMAIndicator(df["Close"], window=ema_short_win).ema_indicator()
    df["EMA_L"] = ta.trend.EMAIndicator(df["Close"], window=ema_long_win).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["CMF"] = ta.volume.ChaikinMoneyFlowIndicator(
        df["High"], df["Low"], df["Close"], df["Volume"], window=cmf_win
    ).chaikin_money_flow()

    signals = pd.Series("Neutral", index=df.index)

    for i in range(2, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        bull_trend = current["Close"] > current["EMA_S"] > current["EMA_L"]
        bear_trend = current["Close"] < current["EMA_S"] < current["EMA_L"]

        obv_rising = current["OBV"] > prev["OBV"]
        cmf_bull = current["CMF"] > 0
        cmf_bear = current["CMF"] < 0

        pa_bull = current["High"] > prev["High"] > prev2["High"] and current["Low"] > prev["Low"] > prev2["Low"]
        pa_bear = current["High"] < prev["High"] < prev2["High"] and current["Low"] < prev["Low"] < prev2["Low"]

        rsi_oversold = current["RSI"] < rsi_os
        rsi_overbought = current["RSI"] > rsi_ob

        if bull_trend and obv_rising and cmf_bull and (pa_bull or rsi_oversold):
            signals.iloc[i] = "Buy"
        elif bear_trend and (not obv_rising) and cmf_bear and (pa_bear or rsi_overbought):
            signals.iloc[i] = "Sell"

    return signals


def _safe_round(value, digits=4):
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return None


def _params_dict(ema_s, ema_l, rsi_ob, rsi_os, cmf_w):
    return {
        "ema_short": ema_s,
        "ema_long": ema_l,
        "rsi_overbought": rsi_ob,
        "rsi_oversold": rsi_os,
        "cmf_window": cmf_w,
    }


def test_params(args):
    """Worker function to test a specific parameter set."""
    df, p = args
    ema_s, ema_l, rsi_ob, rsi_os, cmf_w = p

    signals = generate_signals_custom(df, ema_s, ema_l, rsi_ob, rsi_os, cmf_w)

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
        "params": _params_dict(ema_s, ema_l, rsi_ob, rsi_os, cmf_w),
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
        "mtf_intervals": ["15min", "1h", "4h"],
    }


def _build_backtest_params(params):
    return {
        "ema_short": params["ema_short"],
        "ema_long": params["ema_long"],
        "rsi_window": 14,
        "rsi_overbought": params["rsi_overbought"],
        "rsi_oversold": params["rsi_oversold"],
        "cmf_window": params["cmf_window"],
    }


def _evaluate_baseline(df):
    strategy_params = _read_json(STRATEGY_PARAMS_FILE) or {}
    ema_s = int(strategy_params.get("ema_short", 20))
    ema_l = int(strategy_params.get("ema_long", 50))
    rsi_ob = int(strategy_params.get("rsi_overbought", 70))
    rsi_os = int(strategy_params.get("rsi_oversold", 20))
    cmf_w = int(strategy_params.get("cmf_window", 14))
    return test_params((df, (ema_s, ema_l, rsi_ob, rsi_os, cmf_w)))


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
        print("Failed to fetch historical data.")
        return

    ema_shorts = [9, 12, 20]
    ema_longs = [26, 50, 100]
    rsi_obs = [70, 75, 80]
    rsi_oss = [20, 25, 30]
    cmf_wins = [14, 20]

    param_combinations = list(itertools.product(ema_shorts, ema_longs, rsi_obs, rsi_oss, cmf_wins))
    valid_params = [p for p in param_combinations if p[0] < p[1]]

    print(f"🧬 Generated {len(valid_params)} unique strategy genetic combinations.")
    print("Simulating trading strategies across 13,000+ historical hours. Please wait...\n")

    results = []
    with ProcessPoolExecutor() as executor:
        args_list = [(df, p) for p in valid_params]
        for r in executor.map(test_params, args_list):
            results.append(r)

    results.sort(key=lambda item: item["summary"]["roi"], reverse=True)

    print("🏆 SWARM OPTIMIZATION LEADERBOARD (Top 5 Strategies)\n")
    print(f"{'Rank':<5} | {'EMA Short':<10} | {'EMA Long':<9} | {'RSI OB':<7} | {'RSI OS':<7} | {'CMF Win':<8} | {'ROI':<10}")
    print("-" * 75)
    for i, result in enumerate(results[:5]):
        params = result["params"]
        summary = result["summary"]
        print(
            f"#{i+1:<4} | {params['ema_short']:<10} | {params['ema_long']:<9} | {params['rsi_overbought']:<7} | "
            f"{params['rsi_oversold']:<7} | {params['cmf_window']:<8} | {summary['roi']:>6.2f}%"
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
        if any(target in status_output.stdout for target in tracked_targets):
            commit_msg = (
                f"Auto-Evolve: New Best Strategy Found | ROI: {best_result['summary']['roi']:.2f}% | "
                f"EMA: {best_result['params']['ema_short']}/{best_result['params']['ema_long']} "
                f"CMF: {best_result['params']['cmf_window']}"
            )
            subprocess.run(["git", "add", *tracked_targets], cwd=BASE_DIR, check=True)
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=BASE_DIR, check=True)
            print("💾 Saved promoted strategy state. You can now run `git push` to deploy the new strategy.")
        else:
            print("⚖️ Promotion passed but tracked artifacts are unchanged. No commit needed.")
    except Exception as e:
        print("⚠️ Failed to update promoted strategy state:", e)


if __name__ == "__main__":
    run_swarm()
