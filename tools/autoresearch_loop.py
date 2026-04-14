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
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from tools.twelvedata_market_data import fetch_history as fetch_td_history
from tools.backtest import ACTIVE_BACKTEST_PARAMS
from tools.signal_engine import (
    build_ta_payload_from_row,
    compute_prediction_from_ta,
    normalize_strategy_params,
    prepare_historical_features,
)


ACTIVE_RESEARCH_TEMPLATE_SOURCE = BASE_DIR / "config" / "backtest_params.json"
ACTIVE_RESEARCH_TEMPLATE = normalize_strategy_params(ACTIVE_BACKTEST_PARAMS)
ACTIVE_STRATEGY_PARAMS_SOURCE = BASE_DIR / "config" / "strategy_params.json"
ACTIVE_BACKTEST_PARAMS_SOURCE = BASE_DIR / "config" / "backtest_params.json"
ACTIVE_RESEARCH_SNAPSHOT_FILE = BASE_DIR / "tools" / "reports" / "autoresearch_active.json"
TRACKED_STRATEGY_KEYS = ("ema_short", "ema_long", "rsi_overbought", "rsi_oversold", "cmf_window")
FALLBACK_HISTORY_FILE = BASE_DIR / "data" / "cache" / "market_history" / "XAU_USD_365d_1h.csv"


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


def _read_json_dict(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tracked_param_subset(params: Dict[str, object]) -> Dict[str, object]:
    params = params if isinstance(params, dict) else {}
    return {key: params.get(key) for key in TRACKED_STRATEGY_KEYS if key in params}


def _param_summary(params: Dict[str, object]) -> Dict[str, object]:
    params = params if isinstance(params, dict) else {}
    return {
        "winning_ema": f"{params.get('ema_short', '-')}/{params.get('ema_long', '-')}",
        "winning_rsi": f"{params.get('rsi_overbought', '-')}/{params.get('rsi_oversold', '-')}",
        "winning_cmf": params.get("cmf_window"),
    }


def write_active_strategy_snapshot(report: Dict[str, object], latest_file: Path) -> None:
    strategy_params = _read_json_dict(ACTIVE_STRATEGY_PARAMS_SOURCE)
    backtest_params = _read_json_dict(ACTIVE_BACKTEST_PARAMS_SOURCE)
    best = report.get("best") or {}
    best_params = best.get("params") if isinstance(best, dict) else {}
    tracked_active = _tracked_param_subset(strategy_params)
    tracked_best = _tracked_param_subset(best_params if isinstance(best_params, dict) else {})

    snapshot = {
        "generated_at": report.get("generated_at"),
        "report_generated_at": report.get("generated_at"),
        "report_file": str(latest_file),
        "active_strategy": {
            "source": "config_snapshot",
            "strategy_params_file": str(ACTIVE_STRATEGY_PARAMS_SOURCE),
            "backtest_params_file": str(ACTIVE_BACKTEST_PARAMS_SOURCE),
            "strategy_params": strategy_params,
            "backtest_params": backtest_params,
            "tracked_params": tracked_active,
            "summary": _param_summary(strategy_params),
        },
        "recommendation": {
            "promote": bool(report.get("promote")),
            "promotion_mode": report.get("promotion_mode"),
            "promotion_reason": report.get("promotion_reason", ""),
            "best_params": best_params if isinstance(best_params, dict) else {},
            "tracked_best_params": tracked_best,
            "matches_active_strategy": bool(tracked_active and tracked_best and tracked_active == tracked_best),
        },
    }

    ACTIVE_RESEARCH_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_RESEARCH_SNAPSHOT_FILE.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        return fetch_td_history(period=period, interval=interval, ticker=ticker)
    except Exception:
        if FALLBACK_HISTORY_FILE.exists():
            return pd.read_csv(FALLBACK_HISTORY_FILE, index_col=0, parse_dates=True).sort_index()
        raise


def _to_strategy_dict(p: StrategyParams) -> Dict[str, int]:
    merged = dict(ACTIVE_RESEARCH_TEMPLATE)
    merged.update(
        {
            "ema_short": p.ema_short,
            "ema_long": p.ema_long,
            "rsi_overbought": p.rsi_overbought,
            "rsi_oversold": p.rsi_oversold,
            "cmf_window": p.cmf_window,
        }
    )
    return normalize_strategy_params(merged)


def compute_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    return prepare_historical_features(df, normalize_strategy_params(_to_strategy_dict(p)))


def interval_to_minutes(interval: str) -> int:
    raw = str(interval or "1h").strip().lower()
    mapping = {
        "15m": 15,
        "15min": 15,
        "30m": 30,
        "30min": 30,
        "60m": 60,
        "60min": 60,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1day": 1440,
    }
    if raw in mapping:
        return mapping[raw]
    if raw.endswith("min") and raw[:-3].isdigit():
        return max(1, int(raw[:-3]))
    if raw.endswith("m") and raw[:-1].isdigit():
        return max(1, int(raw[:-1]))
    if raw.endswith("h") and raw[:-1].isdigit():
        return max(1, int(raw[:-1]) * 60)
    if raw.endswith("d") and raw[:-1].isdigit():
        return max(1, int(raw[:-1]) * 1440)
    return 60


def generate_signals(df: pd.DataFrame, p: StrategyParams) -> pd.Series:
    enriched = compute_indicators(df, p)
    signals = pd.Series("Neutral", index=enriched.index, dtype="object")
    strategy_params = _to_strategy_dict(p)
    regime_memory: Dict[str, object] = {}

    for i in range(len(enriched)):
        ta_payload = build_ta_payload_from_row(enriched.iloc[i], strategy_params, regime_memory=regime_memory)
        prediction = compute_prediction_from_ta(ta_payload)
        if isinstance(prediction.get("_regime_memory"), dict):
            regime_memory = dict(prediction.get("_regime_memory") or {})
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


def compute_metrics(trade_returns: List[float], bars_in_test: int, bar_minutes: int = 60) -> Dict[str, float]:
    test_horizon_days = float((bars_in_test * max(1, bar_minutes)) / (60 * 24))
    if len(trade_returns) == 0:
        return {
            "trades": 0,
            "roi": 0.0,
            "test_horizon_days": test_horizon_days,
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
        "test_horizon_days": test_horizon_days,
        "max_drawdown": max_dd * 100,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy * 100,
        "sharpe": sharpe,
    }


def composite_score(metrics: Dict[str, float]) -> float:
    # Use capped components so one short-lived outlier fold does not dominate ranking.
    trade_penalty = 0.0 if metrics["trades"] >= 20 else (20 - metrics["trades"]) * 0.4
    roi_component = float(np.clip(metrics["roi"], -8.0, 8.0))
    sharpe_component = float(np.clip(metrics["sharpe"], -3.0, 3.0))
    expectancy_component = float(np.clip(metrics["expectancy"], -0.25, 0.25)) * 10.0
    pf_component = (float(np.clip(metrics["profit_factor"], 0.0, 2.5)) - 1.0) * 6.0
    drawdown_component = float(np.clip(metrics["max_drawdown"], 0.0, 12.0))
    score = (
        0.45 * roi_component
        + 0.20 * sharpe_component
        + 0.15 * expectancy_component
        + 0.10 * pf_component
        - 0.20 * drawdown_component
        - trade_penalty
    )
    return float(score)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = float(sorted_weights.sum()) * 0.5
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")])


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
    interval_minutes: int,
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
        metrics = compute_metrics(trades, bars_in_test=len(local), bar_minutes=interval_minutes)
        score = composite_score(metrics)

        fold_metrics.append(metrics)
        fold_scores.append(score)

    if not fold_scores:
        return {
            "params": p.as_dict(),
            "median_score": -999.0,
            "mean_score": -999.0,
            "weighted_score": -999.0,
            "weighted_median_score": -999.0,
            "ranking_score": -999.0,
            "folds": 0,
            "pass_rate": 0.0,
            "recency_weighted_pass_rate": 0.0,
            "summary": {},
        }

    folds = len(fold_metrics)
    recency_weights = np.linspace(0.75, 1.35, folds)
    fold_score_array = np.array(fold_scores, dtype=float)
    weighted_median_score = weighted_median(fold_score_array, recency_weights)
    recent_median_score = float(np.median(fold_score_array[-min(3, len(fold_score_array)):]))
    ranking_score = float(
        (0.50 * weighted_median_score)
        + (0.30 * float(np.median(fold_score_array)))
        + (0.20 * recent_median_score)
    )

    def weighted(key: str) -> float:
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        return float(np.average(vals, weights=recency_weights)) if vals.size else 0.0

    pass_count = 0
    pass_flags: List[float] = []
    for m in fold_metrics:
        passed = m["profit_factor"] >= 1.15 and m["max_drawdown"] <= 20 and m["expectancy"] > 0
        if passed:
            pass_count += 1
        pass_flags.append(1.0 if passed else 0.0)

    pass_rate = pass_count / folds
    weighted_score = float(np.average(fold_score_array, weights=recency_weights))
    recency_weighted_pass_rate = float(np.average(np.array(pass_flags, dtype=float), weights=recency_weights)) if pass_flags else 0.0

    summary = {
        "trades": weighted("trades"),
        "roi": weighted("roi"),
        "test_horizon_days": weighted("test_horizon_days"),
        "max_drawdown": weighted("max_drawdown"),
        "win_rate": weighted("win_rate"),
        "profit_factor": weighted("profit_factor"),
        "expectancy": weighted("expectancy"),
        "sharpe": weighted("sharpe"),
        "recent_median_score": recent_median_score,
        "score_std": float(np.std(fold_score_array)),
        "score_min": float(np.min(fold_score_array)),
        "score_max": float(np.max(fold_score_array)),
    }

    return {
        "params": p.as_dict(),
        "median_score": float(np.median(fold_score_array)),
        "mean_score": float(np.mean(fold_score_array)),
        "weighted_score": weighted_score,
        "weighted_median_score": weighted_median_score,
        "ranking_score": ranking_score,
        "folds": folds,
        "pass_rate": float(pass_rate),
        "recency_weighted_pass_rate": recency_weighted_pass_rate,
        "summary": summary,
    }


def parse_int_values(raw: str) -> List[int]:
    tokens = [token.strip() for token in str(raw or "").split(",") if token.strip()]
    if not tokens:
        return []

    values: List[int] = []
    seen = set()
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid integer value '{token}' in comma-separated list: {raw}") from exc
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def param_grid(
    ema_short_values: List[int] | None = None,
    ema_long_values: List[int] | None = None,
    rsi_overbought_values: List[int] | None = None,
    rsi_oversold_values: List[int] | None = None,
    cmf_window_values: List[int] | None = None,
) -> List[StrategyParams]:
    ema_short_values = ema_short_values or [9, 12, 20]
    ema_long_values = ema_long_values or [26, 50, 100]
    rsi_overbought_values = rsi_overbought_values or [65, 70, 75]
    rsi_oversold_values = rsi_oversold_values or [20, 25, 30]
    cmf_window_values = cmf_window_values or [14, 20]

    grid = []
    for es, el, rob, ros, cmf in itertools.product(
        ema_short_values,
        ema_long_values,
        rsi_overbought_values,
        rsi_oversold_values,
        cmf_window_values,
    ):
        if es >= el:
            continue
        grid.append(StrategyParams(es, el, rob, ros, cmf))
    return grid


def make_report(results: List[Dict[str, object]], baseline: Dict[str, object]) -> Dict[str, object]:
    ranked = sorted(
        results,
        key=lambda r: (
            r.get("ranking_score", r.get("weighted_median_score", r["median_score"])),
            r.get("recency_weighted_pass_rate", r["pass_rate"]),
            r["median_score"],
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None

    promote = False
    promotion_mode = "auto"
    reason = "No result"
    if best is not None:
        baseline_score = float(baseline.get("ranking_score", baseline.get("weighted_median_score", baseline["median_score"])))
        improved_by = float(best.get("ranking_score", best.get("weighted_median_score", best["median_score"]))) - baseline_score
        robust = (
            float(best.get("recency_weighted_pass_rate", best["pass_rate"])) >= 0.6
            and float(best["summary"].get("profit_factor", 0)) >= 1.2
            and float(best.get("weighted_median_score", best["median_score"])) >= float(baseline.get("weighted_median_score", baseline["median_score"]))
        )
        if improved_by >= 0.5 and robust:
            promote = True
            reason = (
                f"Promote: improved robust score by {improved_by:.3f} "
                f"with weighted pass_rate {best.get('recency_weighted_pass_rate', best['pass_rate']):.2f}."
            )
        else:
            reason = (
                f"Hold: robust delta {improved_by:.3f}, weighted pass_rate {best.get('recency_weighted_pass_rate', best['pass_rate']):.2f}, "
                f"weighted_median {best.get('weighted_median_score', best['median_score']):.2f}, pf {best['summary'].get('profit_factor', 0):.2f}."
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
        "ranking_method": "robust_median_recency_blend",
        "promote": promote,
        "promotion_mode": promotion_mode,
        "promotion_reason": reason,
    }


def build_regime_param_overrides(best_params: Dict[str, object]) -> Dict[str, object]:
    p = best_params if isinstance(best_params, dict) else {}
    base = {
        "direction_entry_threshold": 8.0,
        "direction_hold_threshold": 5.0,
        "anti_chop_tradeability_floor": 68.0,
        "anti_chop_penalty": 8.0,
        "meta_entry_score_threshold": 63.0,
        "meta_fakeout_prob_cap": 0.42,
        "meta_exit_prob_cap": 0.58,
        "min_expected_edge_pct": 0.06,
    }
    quiet_range = {
        **base,
        "direction_entry_threshold": 9.5,
        "direction_hold_threshold": 6.2,
        "anti_chop_tradeability_floor": 74.0,
        "anti_chop_penalty": 10.0,
        "meta_entry_score_threshold": 70.0,
        "meta_fakeout_prob_cap": 0.32,
        "meta_exit_prob_cap": 0.54,
        "min_expected_edge_pct": 0.08,
    }
    trend_continuation = {
        **base,
        "direction_entry_threshold": 7.0,
        "direction_hold_threshold": 4.5,
        "anti_chop_tradeability_floor": 64.0,
        "meta_entry_score_threshold": 60.0,
        "meta_fakeout_prob_cap": 0.46,
    }
    event_breakout = {
        **base,
        "direction_entry_threshold": 8.0,
        "direction_hold_threshold": 5.0,
        "anti_chop_tradeability_floor": 68.0,
        "anti_chop_penalty": 9.0,
        "meta_entry_score_threshold": 64.0,
        "meta_fakeout_prob_cap": 0.36,
        "meta_exit_prob_cap": 0.54,
        "min_expected_edge_pct": 0.07,
        "warning_min_dwell_bars": 5,
    }
    post_shock_reversal = {
        **base,
        "direction_entry_threshold": 8.5,
        "direction_hold_threshold": 5.5,
        "anti_chop_tradeability_floor": 70.0,
        "meta_entry_score_threshold": 66.0,
        "meta_fakeout_prob_cap": 0.34,
        "meta_exit_prob_cap": 0.52,
    }
    transition = {
        **base,
        "direction_entry_threshold": 8.5,
        "direction_hold_threshold": 5.5,
        "anti_chop_tradeability_floor": 70.0,
        "meta_entry_score_threshold": 66.0,
        "meta_fakeout_prob_cap": 0.38,
    }
    if "ema_short" in p and "ema_long" in p:
        for profile in [quiet_range, trend_continuation, event_breakout, post_shock_reversal, transition]:
            profile["ema_short"] = int(p["ema_short"])
            profile["ema_long"] = int(p["ema_long"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profiles": {
            "quiet_range": quiet_range,
            "trend_continuation": trend_continuation,
            "event_breakout": event_breakout,
            "post_shock_reversal": post_shock_reversal,
            "transition": transition,
        },
    }


def maybe_commit_research_artifacts(
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
    regime_file = repo_root / "config" / "regime_params.json"

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
        add_targets = [latest_rel, rec_rel]
        if regime_file.exists():
            add_targets.append(str(regime_file.relative_to(repo_root)))
        subprocess.run(["git", "add", *add_targets], cwd=repo_root, check=True)

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
    parser.add_argument("--monthly", action="store_true", help="Use monthly walk-forward cadence (train 180d / test 30d / step 30d).")
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of parameter sets for quick runs.")
    parser.add_argument("--ema-short-values", default="", help="Comma-separated EMA short values for the search grid.")
    parser.add_argument("--ema-long-values", default="", help="Comma-separated EMA long values for the search grid.")
    parser.add_argument("--rsi-overbought-values", default="", help="Comma-separated RSI overbought values for the search grid.")
    parser.add_argument("--rsi-oversold-values", default="", help="Comma-separated RSI oversold values for the search grid.")
    parser.add_argument("--cmf-window-values", default="", help="Comma-separated CMF window values for the search grid.")
    parser.add_argument(
        "--apply-promoted-regime",
        action="store_true",
        help="Apply promoted regime overrides to config/regime_params.json. Without this flag, research runs never touch live regime config.",
    )
    parser.add_argument("--auto-branch", action="store_true", help="Create/switch to an autoresearch feature branch.")
    parser.add_argument("--commit-on-promote", action="store_true", help="Commit report artifacts only when promotion gate passes.")
    parser.add_argument("--push", action="store_true", help="Push feature branch after successful promotion commit.")
    parser.add_argument("--branch-prefix", default="autoresearch", help="Prefix for auto-created research branches.")
    args = parser.parse_args()

    if args.monthly:
        args.train_bars = 24 * 180
        args.test_bars = 24 * 30
        args.step_bars = 24 * 30

    print("[autoresearch] fetching data...")
    base_df = fetch_history(args.ticker, args.period, args.interval)
    interval_minutes = interval_to_minutes(args.interval)
    print(f"[autoresearch] candles loaded: {len(base_df)}")
    if args.monthly and (args.train_bars + args.test_bars) > len(base_df):
        train_bars = max(500, int(len(base_df) * 0.70))
        test_bars = max(120, int(len(base_df) * 0.15))
        if train_bars + test_bars >= len(base_df):
            test_bars = max(120, len(base_df) - train_bars - 1)
        args.train_bars = train_bars
        args.test_bars = test_bars
        args.step_bars = max(120, min(test_bars, int(len(base_df) * 0.15)))
        print(
            "[autoresearch] adjusted monthly windows for available data:",
            f"train={args.train_bars}, test={args.test_bars}, step={args.step_bars}",
        )

    ema_short_values = parse_int_values(args.ema_short_values) or [9, 12, 20]
    ema_long_values = parse_int_values(args.ema_long_values) or [26, 50, 100]
    rsi_overbought_values = parse_int_values(args.rsi_overbought_values) or [65, 70, 75]
    rsi_oversold_values = parse_int_values(args.rsi_oversold_values) or [20, 25, 30]
    cmf_window_values = parse_int_values(args.cmf_window_values) or [14, 20]

    candidates = param_grid(
        ema_short_values=ema_short_values,
        ema_long_values=ema_long_values,
        rsi_overbought_values=rsi_overbought_values,
        rsi_oversold_values=rsi_oversold_values,
        cmf_window_values=cmf_window_values,
    )
    if args.max_runs > 0:
        candidates = candidates[: args.max_runs]
    print(f"[autoresearch] testing {len(candidates)} parameter sets")

    baseline_params = StrategyParams(
        int(ACTIVE_RESEARCH_TEMPLATE.get("ema_short", 20)),
        int(ACTIVE_RESEARCH_TEMPLATE.get("ema_long", 100)),
        int(ACTIVE_RESEARCH_TEMPLATE.get("rsi_overbought", 70)),
        int(ACTIVE_RESEARCH_TEMPLATE.get("rsi_oversold", 20)),
        int(ACTIVE_RESEARCH_TEMPLATE.get("cmf_window", 14)),
    )
    baseline = evaluate_params(
        base_df,
        baseline_params,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        interval_minutes=interval_minutes,
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
            interval_minutes=interval_minutes,
        )
        results.append(out)

        if idx % 10 == 0 or idx == len(candidates):
            print(f"[autoresearch] completed {idx}/{len(candidates)}")

    report = make_report(results, baseline)
    report["evaluation_mode"] = "monthly_walkforward" if args.monthly else "default_walkforward"
    report["parameter_surface_file"] = str(ACTIVE_RESEARCH_TEMPLATE_SOURCE)
    report["cost_assumptions"] = {
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
    }
    report["search_space"] = {
        "ema_short": ema_short_values,
        "ema_long": ema_long_values,
        "rsi_overbought": rsi_overbought_values,
        "rsi_oversold": rsi_oversold_values,
        "cmf_window": cmf_window_values,
        "candidate_count": len(candidates),
    }

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    regime_params_file = BASE_DIR / "config" / "regime_params.json"
    candidate_regime_file = reports_dir / "autoresearch_regime_candidate.json"
    best_params = (report.get("best") or {}).get("params") or {}
    regime_overrides = build_regime_param_overrides(best_params)
    candidate_regime_file.write_text(json.dumps(regime_overrides, indent=2), encoding="utf-8")

    live_regime_applied = False
    if report.get("promote") and args.apply_promoted_regime:
        regime_params_file.write_text(json.dumps(regime_overrides, indent=2), encoding="utf-8")

        live_regime_applied = True

    report["candidate_regime_overrides_file"] = str(candidate_regime_file)
    report["live_regime_applied"] = live_regime_applied
    report["active_strategy_snapshot_file"] = str(ACTIVE_RESEARCH_SNAPSHOT_FILE)

    latest_file = reports_dir / "autoresearch_last.json"
    latest_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_active_strategy_snapshot(report, latest_file)

    top = report["best"]
    print("\n[autoresearch] ===== SUMMARY =====")
    if top is None:
        print("No valid strategy results.")
        return

    print("Best params:", top["params"])
    print("Best median score:", round(float(top["median_score"]), 4))
    print("Best robust score:", round(float(top.get("ranking_score", top["median_score"])), 4))
    print("Best pass rate:", round(float(top["pass_rate"]), 3))
    print("Best recency-weighted metrics:", top["summary"])
    print("Promotion decision:", report["promote"], "|", report["promotion_reason"])
    print("Report file:", str(latest_file))
    print("Active strategy snapshot file:", str(ACTIVE_RESEARCH_SNAPSHOT_FILE))
    print("Candidate regime overrides file:", str(candidate_regime_file))
    if live_regime_applied:
        print("Regime overrides file:", str(regime_params_file))
    elif report.get("promote"):
        print("Live regime overrides not applied; rerun with --apply-promoted-regime to update config/regime_params.json")

    maybe_commit_research_artifacts(
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
