#!/usr/bin/env python3
"""
autoresearch_loop.py

Autonomous strategy research loop for XAU/USD via Twelve Data.
This script performs parameter search with walk-forward evaluation and
risk-aware scoring, then emits a promotion recommendation.

Usage examples:
  python tools/autoresearch_loop.py
  python tools/autoresearch_loop.py --period 730d --max-runs 40
  python tools/autoresearch_loop.py --ticker XAU/USD --interval 1h --max-runs 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ta
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from tools.twelvedata_market_data import fetch_history as fetch_td_history
from tools.signal_engine import (
    build_ta_payload_from_row,
    compute_prediction_from_ta,
    normalize_strategy_params,
    prepare_historical_features,
)


@dataclass(frozen=True)
class StrategyParams:
    ema_short: int
    ema_long: int
    rsi_overbought: int
    rsi_oversold: int
    cmf_window: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "ema_short": self.ema_short,
            "ema_long": self.ema_long,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
            "cmf_window": self.cmf_window,
        }


def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return fetch_td_history(period=period, interval=interval, ticker=ticker)


def _to_strategy_dict(p: StrategyParams) -> Dict[str, int]:
    return {
        "ema_short": p.ema_short,
        "ema_long": p.ema_long,
        "rsi_window": 14,
        "rsi_overbought": p.rsi_overbought,
        "rsi_oversold": p.rsi_oversold,
        "adx_window": 14,
        "adx_trending_threshold": 22,
        "adx_weak_trend_threshold": 18,
        "atr_window": 14,
        "atr_trending_percent_threshold": 0.25,
        "cmf_window": p.cmf_window,
        "cmf_strong_buy_threshold": 0.10,
        "cmf_strong_sell_threshold": -0.10,
        "alignment_weight": 1.0,
        "strong_volume_weight": 1.5,
        "verdict_margin_threshold": 1.2,
        "confidence_margin_multiplier": 8.0,
        "rangebound_penalty": 8.0,
        "mixed_alignment_penalty": 6.0,
        "mtf_intervals": ["15min", "1h", "4h"],
    }


def compute_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    return prepare_historical_features(df, normalize_strategy_params(_to_strategy_dict(p)))


def generate_signals(df: pd.DataFrame, p: StrategyParams) -> pd.Series:
    enriched = compute_indicators(df, p)
    signals = pd.Series("Neutral", index=enriched.index, dtype="object")
    strategy_params = normalize_strategy_params(_to_strategy_dict(p))

    for i in range(len(enriched)):
        ta_payload = build_ta_payload_from_row(enriched.iloc[i], strategy_params)
        prediction = compute_prediction_from_ta(ta_payload)
        verdict = prediction.get("verdict")
        if verdict == "Bullish":
            signals.iloc[i] = "Buy"
        elif verdict == "Bearish":
            signals.iloc[i] = "Sell"

    return signals


def simulate_trades(df: pd.DataFrame, signals: pd.Series, fee_bps: float, slippage_bps: float) -> List[float]:
    fee = fee_bps / 10000.0
    slippage = slippage_bps / 10000.0

    position = 0
    entry_price = 0.0
    trade_returns: List[float] = []

    for i in range(len(df) - 1):
        signal = signals.iloc[i]
        next_open = float(df["Open"].iloc[i + 1])

        long_fill = next_open * (1 + slippage)
        short_fill = next_open * (1 - slippage)

        if position == 0:
            if signal == "Buy":
                position = 1
                entry_price = long_fill
            elif signal == "Sell":
                position = -1
                entry_price = short_fill
        elif position == 1 and signal == "Sell":
            exit_price = short_fill
            gross = (exit_price - entry_price) / entry_price
            net = gross - (2 * fee)
            trade_returns.append(net)
            position = -1
            entry_price = short_fill
        elif position == -1 and signal == "Buy":
            exit_price = long_fill
            gross = (entry_price - exit_price) / entry_price
            net = gross - (2 * fee)
            trade_returns.append(net)
            position = 1
            entry_price = long_fill

    final_close = float(df["Close"].iloc[-1])
    if position == 1:
        gross = (final_close - entry_price) / entry_price
        trade_returns.append(gross - (2 * fee))
    elif position == -1:
        gross = (entry_price - final_close) / entry_price
        trade_returns.append(gross - (2 * fee))

    return trade_returns


def max_drawdown_from_equity(equity_curve: np.ndarray) -> float:
    if equity_curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(abs(np.min(dd)))


def compute_metrics(trade_returns: List[float], bars_in_test: int) -> Dict[str, float]:
    if len(trade_returns) == 0:
        return {
            "trades": 0,
            "roi": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
        }

    arr = np.array(trade_returns, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    equity = np.cumprod(1 + arr)
    total_return = float(equity[-1] - 1)

    # Convert bars to years for hourly bars.
    years = max((bars_in_test / (24 * 365)), 1e-6)
    cagr = float((equity[-1] ** (1 / years)) - 1) if equity[-1] > 0 else -1.0

    max_dd = max_drawdown_from_equity(equity)
    win_rate = float((len(wins) / len(arr)) * 100)

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (9.99 if gross_profit > 0 else 0.0)

    expectancy = float(np.mean(arr))
    std = float(np.std(arr))
    sharpe = float((np.mean(arr) / std) * math.sqrt(len(arr))) if std > 0 else 0.0

    return {
        "trades": int(len(arr)),
        "roi": total_return * 100,
        "cagr": cagr * 100,
        "max_drawdown": max_dd * 100,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy * 100,
        "sharpe": sharpe,
    }


def composite_score(metrics: Dict[str, float]) -> float:
    # Reward return quality; penalize drawdown and very low trade counts.
    trade_penalty = 0.0 if metrics["trades"] >= 20 else (20 - metrics["trades"]) * 0.4
    score = (
        0.45 * metrics["sharpe"]
        + 0.30 * metrics["cagr"]
        - 0.20 * metrics["max_drawdown"]
        + 0.05 * metrics["win_rate"]
        + 0.10 * min(metrics["profit_factor"], 3.0)
        - trade_penalty
    )
    return float(score)


def walkforward_ranges(n: int, train_bars: int, test_bars: int, step_bars: int) -> List[Tuple[int, int, int, int]]:
    ranges = []
    i = 0
    while True:
        train_start = i
        train_end = train_start + train_bars
        test_start = train_end
        test_end = test_start + test_bars
        if test_end > n:
            break
        ranges.append((train_start, train_end, test_start, test_end))
        i += step_bars
    return ranges


def evaluate_params(
    base_df: pd.DataFrame,
    p: StrategyParams,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, object]:
    spans = walkforward_ranges(len(base_df), train_bars, test_bars, step_bars)
    if not spans:
        raise RuntimeError("Not enough data for walk-forward evaluation.")

    fold_scores: List[float] = []
    fold_metrics: List[Dict[str, float]] = []

    for train_start, train_end, test_start, test_end in spans:
        warmup_start = max(train_start, train_end - max(60, p.ema_long + 5))
        fold_df = base_df.iloc[warmup_start:test_end].copy()

        with_ind = compute_indicators(fold_df, p)
        if with_ind.empty:
            continue

        signals = generate_signals(with_ind, p)

        test_index = with_ind.index[(with_ind.index >= base_df.index[test_start]) & (with_ind.index <= base_df.index[test_end - 1])]
        if len(test_index) < 5:
            continue

        local = with_ind.loc[test_index]
        local_signals = signals.loc[test_index]

        trades = simulate_trades(local, local_signals, fee_bps=fee_bps, slippage_bps=slippage_bps)
        metrics = compute_metrics(trades, bars_in_test=len(local))
        score = composite_score(metrics)

        fold_metrics.append(metrics)
        fold_scores.append(score)

    if not fold_scores:
        return {
            "params": p.as_dict(),
            "median_score": -999.0,
            "mean_score": -999.0,
            "folds": 0,
            "pass_rate": 0.0,
            "summary": {},
        }

    def med(key: str) -> float:
        vals = [m[key] for m in fold_metrics]
        return float(np.median(vals))

    pass_count = 0
    for m in fold_metrics:
        if m["profit_factor"] >= 1.15 and m["max_drawdown"] <= 20 and m["expectancy"] > 0:
            pass_count += 1

    folds = len(fold_metrics)
    pass_rate = pass_count / folds

    summary = {
        "trades": med("trades"),
        "roi": med("roi"),
        "cagr": med("cagr"),
        "max_drawdown": med("max_drawdown"),
        "win_rate": med("win_rate"),
        "profit_factor": med("profit_factor"),
        "expectancy": med("expectancy"),
        "sharpe": med("sharpe"),
    }

    return {
        "params": p.as_dict(),
        "median_score": float(np.median(fold_scores)),
        "mean_score": float(np.mean(fold_scores)),
        "folds": folds,
        "pass_rate": float(pass_rate),
        "summary": summary,
    }


def param_grid() -> List[StrategyParams]:
    grid = []
    for es, el, rob, ros, cmf in itertools.product(
        [9, 12, 20],
        [26, 50, 100],
        [65, 70, 75],
        [20, 25, 30],
        [14, 20],
    ):
        if es >= el:
            continue
        grid.append(StrategyParams(es, el, rob, ros, cmf))
    return grid


def make_report(results: List[Dict[str, object]], baseline: Dict[str, object]) -> Dict[str, object]:
    ranked = sorted(results, key=lambda r: r["median_score"], reverse=True)
    best = ranked[0] if ranked else None

    promote = False
    promotion_mode = "auto"
    reason = "No result"
    if best is not None:
        baseline_score = float(baseline["median_score"])
        improved_by = float(best["median_score"]) - baseline_score
        robust = float(best["pass_rate"]) >= 0.6 and float(best["summary"].get("profit_factor", 0)) >= 1.2
        if improved_by >= 0.5 and robust:
            promote = True
            reason = f"Promote: improved score by {improved_by:.3f} with pass_rate {best['pass_rate']:.2f}."
        else:
            reason = (
                f"Hold: score delta {improved_by:.3f}, pass_rate {best['pass_rate']:.2f}, "
                f"pf {best['summary'].get('profit_factor', 0):.2f}."
            )

        # Manual override (keeps auto logic intact for normal runs).
        manual_promote = os.getenv("MANUAL_PROMOTE", "").strip().lower() in {"1", "true", "yes", "on"}
        if manual_promote:
            promote = True
            promotion_mode = "manual"
            manual_reason = os.getenv("MANUAL_PROMOTION_REASON", "").strip()
            if manual_reason:
                reason = f"Manual promote: {manual_reason}"
            else:
                reason = "Manual promote: operator override enabled via MANUAL_PROMOTE."

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_5": ranked[:5],
        "baseline": baseline,
        "best": best,
        "promote": promote,
        "promotion_mode": promotion_mode,
        "promotion_reason": reason,
    }


def maybe_commit_promoted_result(
    report: Dict[str, object],
    latest_file: Path,
    reports_dir: Path,
    auto_branch: bool,
    commit_on_promote: bool,
    push: bool,
    branch_prefix: str,
) -> None:
    """Optionally creates an experiment branch and commits report outputs when promoted."""
    if not commit_on_promote:
        return

    if not bool(report.get("promote")):
        print("[autoresearch] promotion gate not passed; skipping git commit.")
        return

    repo_root = Path(__file__).resolve().parents[1]
    rec_file = reports_dir / "autoresearch_recommendation.json"

    recommendation = {
        "generated_at": report.get("generated_at"),
        "promote": report.get("promote"),
        "promotion_reason": report.get("promotion_reason"),
        "best": report.get("best"),
        "baseline": report.get("baseline"),
    }
    rec_file.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    branch_name = f"{branch_prefix.rstrip('/')}/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    try:
        if auto_branch:
            subprocess.run(["git", "checkout", "-B", branch_name], cwd=repo_root, check=True)
            print(f"[autoresearch] switched to branch: {branch_name}")

        latest_rel = str(latest_file.relative_to(repo_root))
        rec_rel = str(rec_file.relative_to(repo_root))
        subprocess.run(["git", "add", latest_rel, rec_rel], cwd=repo_root, check=True)

        # If nothing is staged, avoid empty commit errors.
        staged_diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=False)
        if staged_diff.returncode == 0:
            print("[autoresearch] no staged changes to commit.")
            return

        best = report.get("best") or {}
        best_params = best.get("params", {}) if isinstance(best, dict) else {}
        commit_msg = (
            "autoresearch: promoted walk-forward candidate "
            f"EMA {best_params.get('ema_short', '?')}/{best_params.get('ema_long', '?')}"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_root, check=True)
        print("[autoresearch] committed promoted research artifacts.")

        if push and auto_branch:
            subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=repo_root, check=True)
            print("[autoresearch] pushed promotion branch to origin.")
    except subprocess.CalledProcessError as exc:
        print(f"[autoresearch] git automation failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch walk-forward optimizer for XAUUSD strategy.")
    parser.add_argument("--ticker", default="XAU/USD")
    parser.add_argument("--period", default="730d")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--train-bars", type=int, default=1000)
    parser.add_argument("--test-bars", type=int, default=250)
    parser.add_argument("--step-bars", type=int, default=125)
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of parameter sets for quick runs.")
    parser.add_argument("--auto-branch", action="store_true", help="Create/switch to an autoresearch feature branch.")
    parser.add_argument("--commit-on-promote", action="store_true", help="Commit report artifacts only when promotion gate passes.")
    parser.add_argument("--push", action="store_true", help="Push feature branch after successful promotion commit.")
    parser.add_argument("--branch-prefix", default="autoresearch", help="Prefix for auto-created research branches.")
    args = parser.parse_args()

    print("[autoresearch] fetching data...")
    base_df = fetch_history(args.ticker, args.period, args.interval)
    print(f"[autoresearch] candles loaded: {len(base_df)}")

    candidates = param_grid()
    if args.max_runs > 0:
        candidates = candidates[: args.max_runs]
    print(f"[autoresearch] testing {len(candidates)} parameter sets")

    baseline_file = BASE_DIR / "config" / "strategy_params.json"
    try:
        active = json.loads(baseline_file.read_text(encoding="utf-8"))
    except Exception:
        active = {}
    baseline_params = StrategyParams(
        int(active.get("ema_short", 20)),
        int(active.get("ema_long", 100)),
        int(active.get("rsi_overbought", 70)),
        int(active.get("rsi_oversold", 20)),
        int(active.get("cmf_window", 14)),
    )
    baseline = evaluate_params(
        base_df,
        baseline_params,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    results = []
    for idx, params in enumerate(candidates, start=1):
        out = evaluate_params(
            base_df,
            params,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        results.append(out)

        if idx % 10 == 0 or idx == len(candidates):
            print(f"[autoresearch] completed {idx}/{len(candidates)}")

    report = make_report(results, baseline)

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    latest_file = reports_dir / "autoresearch_last.json"
    latest_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    top = report["best"]
    print("\n[autoresearch] ===== SUMMARY =====")
    if top is None:
        print("No valid strategy results.")
        return

    print("Best params:", top["params"])
    print("Best median score:", round(float(top["median_score"]), 4))
    print("Best pass rate:", round(float(top["pass_rate"]), 3))
    print("Best median metrics:", top["summary"])
    print("Promotion decision:", report["promote"], "|", report["promotion_reason"])
    print("Report file:", str(latest_file))

    maybe_commit_promoted_result(
        report=report,
        latest_file=latest_file,
        reports_dir=reports_dir,
        auto_branch=args.auto_branch,
        commit_on_promote=args.commit_on_promote,
        push=args.push,
        branch_prefix=args.branch_prefix,
    )


if __name__ == "__main__":
    main()
