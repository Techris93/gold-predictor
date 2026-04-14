#!/usr/bin/env python3
"""
autoresearch_loop.py

Autonomous strategy research loop for XAU/USD with Yahoo Finance as the
primary source for 1h and higher intervals.
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
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from tools.yahoo_market_data import fetch_history as fetch_yf_history
from tools.autoresearch_job import (
    AutoResearchJobPaths,
    append_jsonl,
    atomic_write_json,
    candidate_key,
    load_json,
    load_jsonl,
    utc_now_iso,
)
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
DEFAULT_FEATURE_CACHE_SIZE = max(12, min(96, int(os.getenv("AUTORESEARCH_FEATURE_CACHE_SIZE", "48") or "48")))

_WORKER_BASE_DF: pd.DataFrame | None = None
_FEATURE_FRAME_CACHE: OrderedDict[tuple[object, ...], pd.DataFrame] = OrderedDict()
_FEATURE_FRAME_CACHE_LIMIT = DEFAULT_FEATURE_CACHE_SIZE


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


def _default_worker_count() -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, min(8, cpu_total - 1 if cpu_total > 1 else 1))


def _reset_feature_cache(limit: int | None = None) -> None:
    global _FEATURE_FRAME_CACHE, _FEATURE_FRAME_CACHE_LIMIT
    _FEATURE_FRAME_CACHE = OrderedDict()
    if isinstance(limit, int) and limit > 0:
        _FEATURE_FRAME_CACHE_LIMIT = limit


def _get_feature_cache(cache_key: tuple[object, ...]) -> pd.DataFrame | None:
    cached = _FEATURE_FRAME_CACHE.get(cache_key)
    if cached is None:
        return None
    _FEATURE_FRAME_CACHE.move_to_end(cache_key)
    return cached


def _set_feature_cache(cache_key: tuple[object, ...], frame: pd.DataFrame) -> None:
    _FEATURE_FRAME_CACHE[cache_key] = frame
    _FEATURE_FRAME_CACHE.move_to_end(cache_key)
    while len(_FEATURE_FRAME_CACHE) > _FEATURE_FRAME_CACHE_LIMIT:
        _FEATURE_FRAME_CACHE.popitem(last=False)


def _init_evaluation_worker(base_df: pd.DataFrame, feature_cache_size: int) -> None:
    global _WORKER_BASE_DF
    _WORKER_BASE_DF = base_df
    _reset_feature_cache(feature_cache_size)


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


def _load_cached_history_fallback(ticker: str, interval: str, warnings: List[str]) -> pd.DataFrame | None:
    if interval_to_minutes(interval) != 60 or not FALLBACK_HISTORY_FILE.exists():
        return None

    cached = pd.read_csv(FALLBACK_HISTORY_FILE, index_col=0, parse_dates=True).sort_index()
    cached.attrs["data_source"] = "Local 1h cache fallback"
    cached.attrs["data_symbol"] = ticker
    if warnings:
        cached.attrs["data_warning"] = " | ".join(warnings)
    return cached


def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    fetch_errors = []
    try:
        return fetch_yf_history(period=period, interval=interval, ticker=ticker)
    except Exception as exc:
        fetch_errors.append(f"Yahoo Finance: {exc}")
    try:
        td_frame = fetch_td_history(period=period, interval=interval, ticker=ticker)
        td_frame.attrs["data_source"] = "Twelve Data fallback"
        td_frame.attrs["data_symbol"] = ticker
        if fetch_errors:
            td_frame.attrs["data_warning"] = " | ".join(fetch_errors)
        return td_frame
    except Exception as exc:
        fetch_errors.append(f"Twelve Data: {exc}")

    cached = _load_cached_history_fallback(ticker, interval, fetch_errors)
    if cached is not None:
        return cached

    raise RuntimeError("Failed to fetch autoresearch history. " + " | ".join(fetch_errors))


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


def compute_indicators(
    df: pd.DataFrame,
    p: StrategyParams,
    cache_key: tuple[object, ...] | None = None,
) -> pd.DataFrame:
    if cache_key is not None:
        cached = _get_feature_cache(cache_key)
        if cached is not None:
            return cached

    enriched = prepare_historical_features(df, normalize_strategy_params(_to_strategy_dict(p)))
    if cache_key is not None:
        _set_feature_cache(cache_key, enriched)
    return enriched


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


def generate_signals(df: pd.DataFrame, p: StrategyParams, precomputed: bool = False) -> pd.Series:
    enriched = df if precomputed else compute_indicators(df, p)
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


def select_walkforward_ranges(
    spans: List[Tuple[int, int, int, int]],
    max_folds: int | None = None,
) -> List[Tuple[int, int, int, int]]:
    if not spans:
        return []
    if not isinstance(max_folds, int) or max_folds <= 0 or len(spans) <= max_folds:
        return spans

    indices = np.linspace(0, len(spans) - 1, num=max_folds, dtype=int)
    selected_indices: List[int] = []
    seen = set()
    for idx in indices.tolist():
        if idx not in seen:
            selected_indices.append(idx)
            seen.add(idx)
    if (len(spans) - 1) not in seen:
        selected_indices.append(len(spans) - 1)
    return [spans[idx] for idx in sorted(selected_indices)]


def summarize_fold_results(
    p: StrategyParams,
    fold_scores: List[float],
    fold_metrics: List[Dict[str, float]],
    planned_folds: int,
) -> Dict[str, object]:
    if not fold_scores:
        return {
            "params": p.as_dict(),
            "median_score": -999.0,
            "mean_score": -999.0,
            "weighted_score": -999.0,
            "weighted_median_score": -999.0,
            "ranking_score": -999.0,
            "folds": 0,
            "completed_folds": 0,
            "planned_folds": int(planned_folds),
            "pass_count": 0,
            "pass_rate": 0.0,
            "recency_weighted_pass_rate": 0.0,
            "summary": {},
            "pruned": False,
            "prune_reason": "",
        }

    folds = len(fold_metrics)
    recency_weights = np.linspace(0.75, 1.35, folds)
    fold_score_array = np.array(fold_scores, dtype=float)
    weighted_median_score = weighted_median(fold_score_array, recency_weights)
    recent_median_score = float(np.median(fold_score_array[-min(3, len(fold_score_array)) :]))
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
    for metrics in fold_metrics:
        passed = (
            metrics["profit_factor"] >= 1.15
            and metrics["max_drawdown"] <= 20
            and metrics["expectancy"] > 0
        )
        if passed:
            pass_count += 1
        pass_flags.append(1.0 if passed else 0.0)

    pass_rate = pass_count / folds
    weighted_score = float(np.average(fold_score_array, weights=recency_weights))
    recency_weighted_pass_rate = (
        float(np.average(np.array(pass_flags, dtype=float), weights=recency_weights)) if pass_flags else 0.0
    )

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
        "completed_folds": folds,
        "planned_folds": int(planned_folds),
        "pass_count": int(pass_count),
        "pass_rate": float(pass_rate),
        "recency_weighted_pass_rate": recency_weighted_pass_rate,
        "summary": summary,
        "pruned": False,
        "prune_reason": "",
    }


def maybe_prune_candidate(
    partial_result: Dict[str, object],
    baseline_reference: Dict[str, object] | None,
    prune_config: Dict[str, float] | None,
) -> str:
    if not isinstance(prune_config, dict) or not prune_config:
        return ""

    completed_folds = int(partial_result.get("completed_folds", partial_result.get("folds", 0)) or 0)
    planned_folds = int(partial_result.get("planned_folds", completed_folds) or completed_folds)
    min_folds = int(prune_config.get("min_folds", 0) or 0)
    if completed_folds < min_folds:
        return ""

    summary = partial_result.get("summary") if isinstance(partial_result.get("summary"), dict) else {}
    baseline_score = float((baseline_reference or {}).get("ranking_score", -999.0) or -999.0)
    baseline_margin = float(prune_config.get("baseline_margin", 0.75) or 0.75)
    min_profit_factor = float(prune_config.get("min_profit_factor", 0.0) or 0.0)
    min_expectancy = float(prune_config.get("min_expectancy", 0.0) or 0.0)
    min_roi = float(prune_config.get("min_roi", -999.0) or -999.0)
    required_pass_rate = float(prune_config.get("required_pass_rate", 0.0) or 0.0)

    current_ranking = float(partial_result.get("ranking_score", -999.0) or -999.0)
    current_profit_factor = float(summary.get("profit_factor", 0.0) or 0.0)
    current_expectancy = float(summary.get("expectancy", 0.0) or 0.0)
    current_roi = float(summary.get("roi", 0.0) or 0.0)

    if current_profit_factor < min_profit_factor and current_expectancy <= min_expectancy:
        return f"Pruned after {completed_folds} folds: profit factor {current_profit_factor:.2f} and expectancy {current_expectancy:.3f} stayed too weak."

    if baseline_score > -900 and current_ranking < (baseline_score - baseline_margin):
        return f"Pruned after {completed_folds} folds: ranking score {current_ranking:.3f} fell materially below baseline {baseline_score:.3f}."

    remaining_folds = max(0, planned_folds - completed_folds)
    pass_count = int(partial_result.get("pass_count", 0) or 0)
    max_possible_pass_rate = (pass_count + remaining_folds) / max(1, planned_folds)
    if required_pass_rate > 0 and max_possible_pass_rate < required_pass_rate:
        return f"Pruned after {completed_folds} folds: maximum reachable pass rate {max_possible_pass_rate:.2f} cannot clear the stage floor {required_pass_rate:.2f}."

    if completed_folds >= (min_folds + 1) and current_roi < min_roi and current_profit_factor < min_profit_factor:
        return f"Pruned after {completed_folds} folds: ROI {current_roi:.3f} and profit factor {current_profit_factor:.2f} are not recovering."

    return ""


def evaluate_params(
    base_df: pd.DataFrame,
    p: StrategyParams,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    fee_bps: float,
    slippage_bps: float,
    interval_minutes: int,
    max_folds: int | None = None,
    baseline_reference: Dict[str, object] | None = None,
    prune_config: Dict[str, float] | None = None,
    stage_name: str = "",
) -> Dict[str, object]:
    spans = walkforward_ranges(len(base_df), train_bars, test_bars, step_bars)
    if not spans:
        raise RuntimeError("Not enough data for walk-forward evaluation.")
    selected_spans = select_walkforward_ranges(spans, max_folds=max_folds)

    fold_scores: List[float] = []
    fold_metrics: List[Dict[str, float]] = []

    for train_start, train_end, test_start, test_end in selected_spans:
        warmup_start = max(train_start, train_end - max(60, p.ema_long + 5))
        fold_df = base_df.iloc[warmup_start:test_end].copy()
        feature_cache_key = (
            int(warmup_start),
            int(test_end),
            int(p.ema_short),
            int(p.ema_long),
            int(p.cmf_window),
        )

        with_ind = compute_indicators(fold_df, p, cache_key=feature_cache_key)
        if with_ind.empty:
            continue

        signals = generate_signals(with_ind, p, precomputed=True)

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

        partial_result = summarize_fold_results(p, fold_scores, fold_metrics, planned_folds=len(selected_spans))
        prune_reason = maybe_prune_candidate(partial_result, baseline_reference, prune_config)
        if prune_reason:
            partial_result["stage"] = stage_name
            partial_result["pruned"] = True
            partial_result["prune_reason"] = prune_reason
            return partial_result

    final_result = summarize_fold_results(p, fold_scores, fold_metrics, planned_folds=len(selected_spans))
    final_result["stage"] = stage_name
    return final_result


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


def rank_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        list(results or []),
        key=lambda r: (
            r.get("ranking_score", r.get("weighted_median_score", r.get("median_score", -999.0))),
            r.get("recency_weighted_pass_rate", r.get("pass_rate", 0.0)),
            r.get("median_score", -999.0),
        ),
        reverse=True,
    )


def result_brief(result: Dict[str, object]) -> Dict[str, object]:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    return {
        "candidate_key": result.get("candidate_key"),
        "params": result.get("params", {}),
        "ranking_score": result.get("ranking_score"),
        "weighted_median_score": result.get("weighted_median_score"),
        "pass_rate": result.get("pass_rate"),
        "recency_weighted_pass_rate": result.get("recency_weighted_pass_rate"),
        "profit_factor": summary.get("profit_factor"),
        "roi": summary.get("roi"),
        "max_drawdown": summary.get("max_drawdown"),
        "folds": result.get("folds"),
        "planned_folds": result.get("planned_folds"),
        "pruned": bool(result.get("pruned")),
        "prune_reason": result.get("prune_reason", ""),
        "error": result.get("error", ""),
    }


def params_from_dict(params: Dict[str, object]) -> StrategyParams:
    return StrategyParams(
        int(params.get("ema_short", 20)),
        int(params.get("ema_long", 50)),
        int(params.get("rsi_overbought", 70)),
        int(params.get("rsi_oversold", 20)),
        int(params.get("cmf_window", 14)),
    )


def sort_candidates(candidates: List[StrategyParams]) -> List[StrategyParams]:
    unique: Dict[Tuple[int, int, int, int, int], StrategyParams] = {}
    for candidate in candidates:
        unique[(candidate.ema_short, candidate.ema_long, candidate.rsi_overbought, candidate.rsi_oversold, candidate.cmf_window)] = candidate
    return sorted(
        unique.values(),
        key=lambda p: (p.ema_short, p.ema_long, p.cmf_window, p.rsi_overbought, p.rsi_oversold),
    )


def trim_candidate_list(candidates: List[StrategyParams], limit: int | None = None) -> List[StrategyParams]:
    ordered = sort_candidates(candidates)
    if isinstance(limit, int) and limit > 0:
        return ordered[:limit]
    return ordered


def _expand_parameter_values(
    base_values: List[int],
    deltas: List[int],
    minimum: int,
    maximum: int,
    limit: int,
) -> List[int]:
    if not base_values:
        return []
    expanded = set()
    for value in base_values:
        for delta in deltas:
            candidate = value + delta
            if minimum <= candidate <= maximum:
                expanded.add(candidate)
    center = int(round(float(np.median(np.array(base_values, dtype=float)))))
    ordered = sorted(expanded, key=lambda item: (abs(item - center), item))
    return sorted(ordered[:limit])


def _candidate_distance(candidate: StrategyParams, seeds: List[StrategyParams]) -> float:
    if not seeds:
        return 0.0
    distances = []
    for seed in seeds:
        distance = (
            abs(candidate.ema_short - seed.ema_short) * 1.2
            + abs(candidate.ema_long - seed.ema_long) * 0.35
            + abs(candidate.rsi_overbought - seed.rsi_overbought) * 0.55
            + abs(candidate.rsi_oversold - seed.rsi_oversold) * 0.55
            + abs(candidate.cmf_window - seed.cmf_window) * 0.65
        )
        distances.append(distance)
    return min(distances)


def order_candidates_by_seed_proximity(
    candidates: List[StrategyParams],
    seeds: List[StrategyParams],
) -> List[StrategyParams]:
    return sorted(
        sort_candidates(candidates),
        key=lambda candidate: (
            _candidate_distance(candidate, seeds),
            candidate.ema_short,
            candidate.ema_long,
            candidate.cmf_window,
            candidate.rsi_overbought,
            candidate.rsi_oversold,
        ),
    )


def derive_refined_candidates(
    coarse_results: List[Dict[str, object]],
    seed_count: int,
    max_candidates: int,
) -> List[StrategyParams]:
    ranked = [result for result in rank_results(coarse_results) if int(result.get("folds", 0) or 0) > 0]
    seeds = [params_from_dict(result.get("params") or {}) for result in ranked[:seed_count]]
    if not seeds:
        return []

    ema_short_values = _expand_parameter_values([seed.ema_short for seed in seeds], [-2, -1, 0, 1, 2], 5, 30, 6)
    ema_long_values = _expand_parameter_values([seed.ema_long for seed in seeds], [-8, -4, 0, 4, 8], 16, 140, 7)
    rsi_overbought_values = _expand_parameter_values([seed.rsi_overbought for seed in seeds], [-4, -2, 0, 2, 4], 58, 85, 5)
    rsi_oversold_values = _expand_parameter_values([seed.rsi_oversold for seed in seeds], [-4, -2, 0, 2, 4], 10, 35, 5)
    cmf_window_values = _expand_parameter_values([seed.cmf_window for seed in seeds], [-4, -2, 0, 2, 4], 8, 28, 4)

    refined = param_grid(
        ema_short_values=ema_short_values,
        ema_long_values=ema_long_values,
        rsi_overbought_values=rsi_overbought_values,
        rsi_oversold_values=rsi_oversold_values,
        cmf_window_values=cmf_window_values,
    )
    ordered = order_candidates_by_seed_proximity(refined, seeds)
    if max_candidates > 0:
        ordered = ordered[:max_candidates]
    return ordered


def derive_confirm_candidates(
    refine_results: List[Dict[str, object]],
    coarse_results: List[Dict[str, object]],
    top_k: int,
) -> List[StrategyParams]:
    combined: List[StrategyParams] = []
    seen = set()
    for source in [refine_results, coarse_results]:
        for result in rank_results(source):
            if int(result.get("folds", 0) or 0) <= 0:
                continue
            candidate = params_from_dict(result.get("params") or {})
            key = (candidate.ema_short, candidate.ema_long, candidate.rsi_overbought, candidate.rsi_oversold, candidate.cmf_window)
            if key in seen:
                continue
            seen.add(key)
            combined.append(candidate)
            if len(combined) >= top_k:
                return combined
    return combined


def build_stage_definition(
    name: str,
    description: str,
    candidates: List[StrategyParams],
    train_bars: int,
    test_bars: int,
    step_bars: int,
    max_folds: int | None,
    prune_config: Dict[str, float] | None,
    candidate_limit: int | None = None,
) -> Dict[str, object]:
    candidate_list = trim_candidate_list(candidates, candidate_limit)
    return {
        "name": name,
        "description": description,
        "candidates": [candidate.as_dict() for candidate in candidate_list],
        "train_bars": int(train_bars),
        "test_bars": int(test_bars),
        "step_bars": int(step_bars),
        "max_folds": int(max_folds) if isinstance(max_folds, int) and max_folds > 0 else None,
        "prune_config": dict(prune_config or {}),
        "baseline": None,
    }


def build_initial_stage_plan(
    args: argparse.Namespace,
    coarse_candidates: List[StrategyParams],
) -> Dict[str, object]:
    candidate_cap = args.max_runs if args.max_runs > 0 else None
    if args.search_mode == "flat":
        return {
            "stage_order": ["flat"],
            "stages": {
                "flat": build_stage_definition(
                    name="flat",
                    description="Single-stage flat parameter search.",
                    candidates=coarse_candidates,
                    train_bars=args.train_bars,
                    test_bars=args.test_bars,
                    step_bars=args.step_bars,
                    max_folds=None,
                    prune_config={
                        "min_folds": 3,
                        "baseline_margin": 0.8,
                        "min_profit_factor": 1.0,
                        "min_expectancy": 0.0,
                        "required_pass_rate": 0.45,
                        "min_roi": -0.25,
                    },
                    candidate_limit=candidate_cap,
                )
            },
        }

    return {
        "stage_order": ["coarse", "refine", "confirm"],
        "stages": {
            "coarse": build_stage_definition(
                name="coarse",
                description="Fast coarse screening across the seed grid.",
                candidates=coarse_candidates,
                train_bars=args.train_bars,
                test_bars=args.test_bars,
                step_bars=args.step_bars,
                max_folds=args.coarse_max_folds,
                prune_config={
                    "min_folds": 2,
                    "baseline_margin": 1.0,
                    "min_profit_factor": 0.95,
                    "min_expectancy": 0.0,
                    "required_pass_rate": 0.35,
                    "min_roi": -0.5,
                },
                candidate_limit=candidate_cap,
            ),
            "refine": None,
            "confirm": None,
        },
    }


def ensure_stage_definition(
    plan: Dict[str, object],
    stage_name: str,
    args: argparse.Namespace,
    stage_results: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object] | None:
    stages = plan.setdefault("stages", {})
    existing = stages.get(stage_name)
    if isinstance(existing, dict):
        return existing

    candidate_cap = args.max_runs if args.max_runs > 0 else None
    if stage_name == "refine":
        refined_candidates = derive_refined_candidates(
            coarse_results=stage_results.get("coarse", []),
            seed_count=args.refine_seed_count,
            max_candidates=min(args.refine_max_candidates, candidate_cap) if candidate_cap else args.refine_max_candidates,
        )
        if not refined_candidates:
            return None
        stages[stage_name] = build_stage_definition(
            name="refine",
            description="Focused neighborhood search around the best coarse candidates.",
            candidates=refined_candidates,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            max_folds=args.refine_max_folds,
            prune_config={
                "min_folds": 3,
                "baseline_margin": 0.7,
                "min_profit_factor": 1.02,
                "min_expectancy": 0.0,
                "required_pass_rate": 0.5,
                "min_roi": -0.2,
            },
            candidate_limit=None,
        )
        return stages[stage_name]

    if stage_name == "confirm":
        confirm_candidates = derive_confirm_candidates(
            refine_results=stage_results.get("refine", []),
            coarse_results=stage_results.get("coarse", []),
            top_k=min(args.confirm_top_k, candidate_cap) if candidate_cap else args.confirm_top_k,
        )
        if not confirm_candidates:
            return None
        stages[stage_name] = build_stage_definition(
            name="confirm",
            description="Full confirmation pass on the strongest refined candidates.",
            candidates=confirm_candidates,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            max_folds=None,
            prune_config={},
            candidate_limit=None,
        )
        return stages[stage_name]

    return None


def _build_failed_result(
    params: StrategyParams,
    stage_name: str,
    planned_folds: int,
    error_message: str,
) -> Dict[str, object]:
    failed = summarize_fold_results(params, [], [], planned_folds=planned_folds)
    failed["stage"] = stage_name
    failed["error"] = error_message
    failed["pruned"] = True
    failed["prune_reason"] = error_message
    return failed


def evaluate_baseline_for_stage(
    base_df: pd.DataFrame,
    stage_definition: Dict[str, object],
    fee_bps: float,
    slippage_bps: float,
    interval_minutes: int,
) -> Dict[str, object]:
    baseline_params = StrategyParams(
        int(ACTIVE_RESEARCH_TEMPLATE.get("ema_short", 20)),
        int(ACTIVE_RESEARCH_TEMPLATE.get("ema_long", 100)),
        int(ACTIVE_RESEARCH_TEMPLATE.get("rsi_overbought", 70)),
        int(ACTIVE_RESEARCH_TEMPLATE.get("rsi_oversold", 20)),
        int(ACTIVE_RESEARCH_TEMPLATE.get("cmf_window", 14)),
    )
    return evaluate_params(
        base_df,
        baseline_params,
        train_bars=int(stage_definition.get("train_bars", 1000)),
        test_bars=int(stage_definition.get("test_bars", 250)),
        step_bars=int(stage_definition.get("step_bars", 125)),
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        interval_minutes=interval_minutes,
        max_folds=stage_definition.get("max_folds"),
        stage_name=str(stage_definition.get("name") or "baseline"),
    )


def write_job_artifacts(
    job_paths: AutoResearchJobPaths,
    job_state: Dict[str, object],
) -> None:
    atomic_write_json(job_paths.state_file, job_state)
    leaderboard_payload = {
        "job_name": job_state.get("job_name"),
        "updated_at": job_state.get("updated_at"),
        "status": job_state.get("status"),
        "current_stage": job_state.get("current_stage"),
        "stages": {
            stage_name: (stage_state.get("top_candidates") if isinstance(stage_state, dict) else [])
            for stage_name, stage_state in (job_state.get("stages") or {}).items()
        },
    }
    atomic_write_json(job_paths.leaderboard_file, leaderboard_payload)
    current_stage = str(job_state.get("current_stage") or "")
    current_stage_state = ((job_state.get("stages") or {}).get(current_stage) if current_stage else {}) or {}
    heartbeat_payload = {
        "job_name": job_state.get("job_name"),
        "updated_at": job_state.get("updated_at"),
        "status": job_state.get("status"),
        "current_stage": current_stage,
        "current_stage_completed": current_stage_state.get("completed_count", 0),
        "current_stage_candidates": current_stage_state.get("candidate_count", 0),
        "job_dir": str(job_paths.job_dir),
    }
    atomic_write_json(job_paths.heartbeat_file, heartbeat_payload)


def refresh_stage_state(
    job_state: Dict[str, object],
    stage_name: str,
    stage_definition: Dict[str, object],
    results: List[Dict[str, object]],
    baseline: Dict[str, object],
    status: str,
) -> None:
    stages = job_state.setdefault("stages", {})
    stage_state = stages.setdefault(stage_name, {})
    ranked = rank_results(results)
    existing_started_at = str(stage_state.get("started_at") or utc_now_iso())
    stage_state.update(
        {
            "name": stage_name,
            "description": stage_definition.get("description", ""),
            "status": status,
            "candidate_count": len(stage_definition.get("candidates") or []),
            "completed_count": len(results),
            "pending_count": max(0, len(stage_definition.get("candidates") or []) - len(results)),
            "pruned_count": sum(1 for result in results if bool(result.get("pruned"))),
            "started_at": existing_started_at,
            "completed_at": utc_now_iso() if status == "completed" else stage_state.get("completed_at"),
            "baseline": result_brief(baseline),
            "top_candidates": [result_brief(result) for result in ranked[:5]],
        }
    )
    job_state["current_stage"] = stage_name
    job_state["updated_at"] = utc_now_iso()


def evaluate_candidate_task(task: Dict[str, Any]) -> Dict[str, object]:
    if _WORKER_BASE_DF is None:
        raise RuntimeError("Autoresearch evaluation worker was not initialized.")
    params = params_from_dict(task.get("params") or {})
    return evaluate_params(
        _WORKER_BASE_DF,
        params,
        train_bars=int(task.get("train_bars", 1000)),
        test_bars=int(task.get("test_bars", 250)),
        step_bars=int(task.get("step_bars", 125)),
        fee_bps=float(task.get("fee_bps", 3.0)),
        slippage_bps=float(task.get("slippage_bps", 2.0)),
        interval_minutes=int(task.get("interval_minutes", 60)),
        max_folds=task.get("max_folds"),
        baseline_reference=task.get("baseline_reference") if isinstance(task.get("baseline_reference"), dict) else None,
        prune_config=task.get("prune_config") if isinstance(task.get("prune_config"), dict) else None,
        stage_name=str(task.get("stage_name") or ""),
    )


def run_stage(
    job_paths: AutoResearchJobPaths,
    job_state: Dict[str, object],
    plan: Dict[str, object],
    stage_name: str,
    stage_definition: Dict[str, object],
    base_df: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    interval_minutes: int,
    workers: int,
    feature_cache_size: int,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    results_file = job_paths.stage_results_file(stage_name)
    existing_results = load_jsonl(results_file)
    completed_keys = {
        str(result.get("candidate_key") or candidate_key(stage_name, result.get("params") or {}))
        for result in existing_results
        if isinstance(result, dict)
    }

    baseline = stage_definition.get("baseline") if isinstance(stage_definition.get("baseline"), dict) else None
    if not isinstance(baseline, dict) or not baseline:
        baseline = evaluate_baseline_for_stage(
            base_df=base_df,
            stage_definition=stage_definition,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            interval_minutes=interval_minutes,
        )
        stage_definition["baseline"] = baseline
        atomic_write_json(job_paths.plan_file, plan)

    refresh_stage_state(job_state, stage_name, stage_definition, existing_results, baseline, status="running")
    job_state["status"] = f"running_{stage_name}"
    write_job_artifacts(job_paths, job_state)

    pending_candidates = [
        params_from_dict(candidate_params)
        for candidate_params in (stage_definition.get("candidates") or [])
        if candidate_key(stage_name, candidate_params) not in completed_keys
    ]
    pending_candidates = sort_candidates(pending_candidates)
    if not pending_candidates:
        refresh_stage_state(job_state, stage_name, stage_definition, existing_results, baseline, status="completed")
        job_state["status"] = "running"
        write_job_artifacts(job_paths, job_state)
        return existing_results, baseline

    print(f"[autoresearch] stage={stage_name} pending {len(pending_candidates)}/{len(stage_definition.get('candidates') or [])}")
    task_payloads = [
        {
            "stage_name": stage_name,
            "params": candidate.as_dict(),
            "train_bars": int(stage_definition.get("train_bars", 1000)),
            "test_bars": int(stage_definition.get("test_bars", 250)),
            "step_bars": int(stage_definition.get("step_bars", 125)),
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "interval_minutes": interval_minutes,
            "max_folds": stage_definition.get("max_folds"),
            "baseline_reference": baseline,
            "prune_config": stage_definition.get("prune_config") if isinstance(stage_definition.get("prune_config"), dict) else {},
        }
        for candidate in pending_candidates
    ]

    def handle_result(task_payload: Dict[str, Any], result: Dict[str, object]) -> None:
        result["candidate_key"] = candidate_key(stage_name, result.get("params") or task_payload.get("params") or {})
        result["stage"] = stage_name
        result["evaluated_at"] = utc_now_iso()
        append_jsonl(results_file, result)
        existing_results.append(result)
        refresh_stage_state(job_state, stage_name, stage_definition, existing_results, baseline, status="running")
        job_state["status"] = f"running_{stage_name}"
        write_job_artifacts(job_paths, job_state)

    if workers > 1 and len(task_payloads) > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_evaluation_worker,
            initargs=(base_df, feature_cache_size),
        ) as executor:
            futures = {executor.submit(evaluate_candidate_task, payload): payload for payload in task_payloads}
            for idx, future in enumerate(as_completed(futures), start=1):
                payload = futures[future]
                params = params_from_dict(payload.get("params") or {})
                try:
                    result = future.result()
                except Exception as exc:
                    result = _build_failed_result(
                        params=params,
                        stage_name=stage_name,
                        planned_folds=int(stage_definition.get("max_folds") or 0),
                        error_message=f"Candidate evaluation failed: {exc}",
                    )
                handle_result(payload, result)
                if idx % 5 == 0 or idx == len(task_payloads):
                    print(f"[autoresearch] stage={stage_name} completed {len(existing_results)}/{len(stage_definition.get('candidates') or [])}")
    else:
        _init_evaluation_worker(base_df, feature_cache_size)
        for idx, payload in enumerate(task_payloads, start=1):
            params = params_from_dict(payload.get("params") or {})
            try:
                result = evaluate_candidate_task(payload)
            except Exception as exc:
                result = _build_failed_result(
                    params=params,
                    stage_name=stage_name,
                    planned_folds=int(stage_definition.get("max_folds") or 0),
                    error_message=f"Candidate evaluation failed: {exc}",
                )
            handle_result(payload, result)
            if idx % 5 == 0 or idx == len(task_payloads):
                print(f"[autoresearch] stage={stage_name} completed {len(existing_results)}/{len(stage_definition.get('candidates') or [])}")

    refresh_stage_state(job_state, stage_name, stage_definition, existing_results, baseline, status="completed")
    job_state["status"] = "running"
    write_job_artifacts(job_paths, job_state)
    return existing_results, baseline


def make_report(results: List[Dict[str, object]], baseline: Dict[str, object]) -> Dict[str, object]:
    ranked = rank_results(results)
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
    parser.add_argument("--search-mode", choices=["staged", "flat"], default="staged")
    parser.add_argument("--job-name", default="", help="Optional job name used for checkpoint/resume files.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing autoresearch job by --job-name.")
    parser.add_argument("--workers", type=int, default=_default_worker_count(), help="Number of candidate evaluation workers.")
    parser.add_argument("--train-bars", type=int, default=1000)
    parser.add_argument("--test-bars", type=int, default=250)
    parser.add_argument("--step-bars", type=int, default=125)
    parser.add_argument("--monthly", action="store_true", help="Use monthly walk-forward cadence (train 180d / test 30d / step 30d).")
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of parameter sets for quick runs.")
    parser.add_argument("--coarse-max-folds", type=int, default=6, help="Maximum walk-forward folds evaluated during the coarse stage.")
    parser.add_argument("--refine-max-folds", type=int, default=10, help="Maximum walk-forward folds evaluated during the refine stage.")
    parser.add_argument("--refine-seed-count", type=int, default=6, help="How many coarse winners seed the refine neighborhood.")
    parser.add_argument("--refine-max-candidates", type=int, default=240, help="Maximum number of refined candidates to evaluate.")
    parser.add_argument("--confirm-top-k", type=int, default=12, help="How many refined winners are fully confirmed in the final stage.")
    parser.add_argument("--feature-cache-size", type=int, default=DEFAULT_FEATURE_CACHE_SIZE, help="Per-process feature-frame cache size.")
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

    if args.resume and not args.job_name:
        parser.error("--resume requires --job-name so the existing checkpoint can be found.")

    if args.monthly:
        args.train_bars = 24 * 180
        args.test_bars = 24 * 30
        args.step_bars = 24 * 30

    print("[autoresearch] fetching data...")
    base_df = fetch_history(args.ticker, args.period, args.interval)
    data_source = str(base_df.attrs.get("data_source") or "Unknown")
    data_symbol = str(base_df.attrs.get("data_symbol") or args.ticker)
    data_warning = str(base_df.attrs.get("data_warning") or "").strip()
    interval_minutes = interval_to_minutes(args.interval)
    print(f"[autoresearch] candles loaded: {len(base_df)}")
    print(f"[autoresearch] data source: {data_source} ({data_symbol})")
    if data_warning:
        print(f"[autoresearch] data warning: {data_warning}")
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

    coarse_candidates = param_grid(
        ema_short_values=ema_short_values,
        ema_long_values=ema_long_values,
        rsi_overbought_values=rsi_overbought_values,
        rsi_oversold_values=rsi_oversold_values,
        cmf_window_values=cmf_window_values,
    )
    if not coarse_candidates:
        raise RuntimeError("No valid parameter candidates were generated for the search space.")

    timestamp_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_name = args.job_name or f"autoresearch_{args.search_mode}_{args.interval}_{timestamp_suffix}"
    job_paths = AutoResearchJobPaths(BASE_DIR, job_name)
    job_paths.ensure()

    plan = load_json(job_paths.plan_file, {}) if args.resume else {}
    if not isinstance(plan, dict) or not plan:
        plan = build_initial_stage_plan(args, coarse_candidates)
        atomic_write_json(job_paths.plan_file, plan)

    stage_order = list(plan.get("stage_order") or (["flat"] if args.search_mode == "flat" else ["coarse", "refine", "confirm"]))
    job_state = load_json(job_paths.state_file, {}) if args.resume else {}
    if not isinstance(job_state, dict) or not job_state:
        job_state = {
            "job_name": job_name,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "completed_at": None,
            "status": "queued",
            "search_mode": args.search_mode,
            "job_dir": str(job_paths.job_dir),
            "stage_order": stage_order,
            "current_stage": "",
            "workers": int(max(1, args.workers)),
            "data_source": data_source,
            "data_symbol": data_symbol,
            "data_warning": data_warning,
            "args": vars(args),
            "stages": {},
        }
    else:
        job_state["updated_at"] = utc_now_iso()
        job_state["search_mode"] = str(job_state.get("search_mode") or args.search_mode)
        job_state["job_dir"] = str(job_paths.job_dir)
        job_state["stage_order"] = stage_order
        job_state["data_source"] = data_source
        job_state["data_symbol"] = data_symbol
        job_state["data_warning"] = data_warning
        job_state["workers"] = int(max(1, args.workers))
        job_state["args"] = vars(args)

    print(f"[autoresearch] job name: {job_name}")
    print(f"[autoresearch] job dir: {job_paths.job_dir}")

    job_state["status"] = "running"
    write_job_artifacts(job_paths, job_state)

    stage_results: Dict[str, List[Dict[str, object]]] = {}
    stage_baselines: Dict[str, Dict[str, object]] = {}
    for stage_name in stage_order:
        stage_definition = ensure_stage_definition(plan, stage_name, args, stage_results)
        if stage_definition is None or not isinstance(stage_definition, dict):
            continue
        if not stage_definition.get("candidates"):
            continue
        atomic_write_json(job_paths.plan_file, plan)
        results, baseline = run_stage(
            job_paths=job_paths,
            job_state=job_state,
            plan=plan,
            stage_name=stage_name,
            stage_definition=stage_definition,
            base_df=base_df,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            interval_minutes=interval_minutes,
            workers=max(1, int(args.workers)),
            feature_cache_size=max(1, int(args.feature_cache_size)),
        )
        stage_results[stage_name] = results
        stage_baselines[stage_name] = baseline

    final_stage_name = "flat"
    if args.search_mode == "staged":
        if stage_results.get("confirm"):
            final_stage_name = "confirm"
        elif stage_results.get("refine"):
            final_stage_name = "refine"
        elif stage_results.get("coarse"):
            final_stage_name = "coarse"

    final_results = stage_results.get(final_stage_name, [])
    final_baseline = stage_baselines.get(final_stage_name)
    if not isinstance(final_baseline, dict):
        final_definition = ensure_stage_definition(plan, final_stage_name, args, stage_results)
        if not isinstance(final_definition, dict):
            raise RuntimeError(f"Missing final stage definition for {final_stage_name}.")
        final_baseline = evaluate_baseline_for_stage(
            base_df=base_df,
            stage_definition=final_definition,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            interval_minutes=interval_minutes,
        )

    report = make_report(final_results, final_baseline)
    report["evaluation_mode"] = "monthly_walkforward" if args.monthly else "default_walkforward"
    report["parameter_surface_file"] = str(ACTIVE_RESEARCH_TEMPLATE_SOURCE)
    report["cost_assumptions"] = {
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
    }
    report["search_mode"] = args.search_mode
    report["job_name"] = job_name
    report["job_dir"] = str(job_paths.job_dir)
    report["job_state_file"] = str(job_paths.state_file)
    report["leaderboard_file"] = str(job_paths.leaderboard_file)
    report["final_stage"] = final_stage_name
    report["market_data_source"] = data_source
    report["market_data_symbol"] = data_symbol
    if data_warning:
        report["market_data_warning"] = data_warning
    report["search_space"] = {
        "ema_short": ema_short_values,
        "ema_long": ema_long_values,
        "rsi_overbought": rsi_overbought_values,
        "rsi_oversold": rsi_oversold_values,
        "cmf_window": cmf_window_values,
        "candidate_count": len(coarse_candidates),
    }
    report["stage_summaries"] = {
        stage_name: dict(stage_state)
        for stage_name, stage_state in (job_state.get("stages") or {}).items()
        if isinstance(stage_state, dict)
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
    atomic_write_json(job_paths.final_report_file, report)

    job_state["status"] = "completed"
    job_state["completed_at"] = utc_now_iso()
    job_state["current_stage"] = final_stage_name
    job_state["final_report_file"] = str(job_paths.final_report_file)
    job_state["final_best"] = result_brief(report.get("best") or {}) if isinstance(report.get("best"), dict) else {}
    write_job_artifacts(job_paths, job_state)

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
    print("Job state file:", str(job_paths.state_file))
    print("Leaderboard file:", str(job_paths.leaderboard_file))
    print("Active strategy snapshot file:", str(ACTIVE_RESEARCH_SNAPSHOT_FILE))
    print("Candidate regime overrides file:", str(candidate_regime_file))
    if live_regime_applied:
        print("Regime overrides file:", str(regime_params_file))
    elif report.get("promote"):
        print("Live regime overrides not applied; rerun with --apply-promoted-regime to update config/regime_params.json")
    print(
        "Resume command:",
        f"python tools/autoresearch_loop.py --resume --job-name {job_name} --interval {args.interval} --period {args.period}",
    )

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
