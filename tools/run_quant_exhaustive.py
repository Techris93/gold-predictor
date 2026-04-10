import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

BASE = Path("/Users/chrixchange/.openclaw/workspace/gold-predictor")
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tools.backtest import (
    _simulate_from_signals,
    _trade_summary,
    generate_signals,
    summarize_big_move_metrics,
    summarize_decision_quality_metrics,
    summarize_execution_state_metrics,
    summarize_large_move_labels,
    summarize_quality_gate_metrics,
    summarize_tail_event_metrics,
    summarize_transition_metrics,
)
from tools.signal_engine import normalize_strategy_params


DATA_FILE = BASE / "data/swarm/cache/XAU_USD_365d_1h.csv"
CONFIG_FILE = BASE / "config/strategy_params.json"
OUT_FILE = BASE / "tools/reports/quant_tuning_results.json"


def score_candidate(summary, big_move, exec_metrics, tail_metrics, transition_metrics, quality_gate, decision_quality):
    roi = float(summary.get("roi", 0.0) or 0.0)
    trades = float(summary.get("trades", 0) or 0)

    watch = big_move.get("warning_watch", {})
    directional = big_move.get("warning_directional", {})
    enter = exec_metrics.get("enter", {})
    tail_edge = float((tail_metrics or {}).get("edge_vs_naive_4h", 0.0) or 0.0)
    tail_hit = float((tail_metrics or {}).get("model_hit_rate_4h", 0.0) or 0.0)
    whipsaw_rate = float((transition_metrics or {}).get("whipsaw_rate", 0.0) or 0.0)
    abstain_correct = float((decision_quality or {}).get("abstain_correct_rate", 0.0) or 0.0)
    regret = float((decision_quality or {}).get("avg_regret_vs_stand_aside_pct", 0.0) or 0.0)
    gate_pass = bool((quality_gate or {}).get("passed"))

    enter_hit = float(enter.get("hit_rate", 0.0) or 0.0)
    enter_count = float(enter.get("count", 0) or 0)

    p60 = float(watch.get("precision_60m", 0.0) or 0.0)
    r60 = float(watch.get("recall_60m", 0.0) or 0.0)
    p4 = float(watch.get("precision_4h", 0.0) or 0.0)
    r4 = float(watch.get("recall_4h", 0.0) or 0.0)

    dp4 = float(directional.get("precision_4h", 0.0) or 0.0)
    dr4 = float(directional.get("recall_4h", 0.0) or 0.0)

    penalty = 0.0
    if p60 < 0.75:
        penalty += (0.75 - p60) * 300
    if p4 < 0.75:
        penalty += (0.75 - p4) * 300
    if dp4 < 0.88:
        penalty += (0.88 - dp4) * 220
    if enter_hit < 0.58:
        penalty += (0.58 - enter_hit) * 260
    if trades < 25:
        penalty += (25 - trades) * 1.2
    if whipsaw_rate > 0.20:
        penalty += (whipsaw_rate - 0.20) * 260
    if tail_hit < 0.54:
        penalty += (0.54 - tail_hit) * 180
    if tail_edge < 0.0:
        penalty += abs(tail_edge) * 220
    if abstain_correct < 0.55:
        penalty += (0.55 - abstain_correct) * 120
    if regret > 0.35:
        penalty += (regret - 0.35) * 90
    if not gate_pass:
        penalty += 80.0

    return (
        roi * 1.4
        + trades * 0.12
        + enter_hit * 220
        + p60 * 90
        + r60 * 180
        + p4 * 90
        + r4 * 180
        + dp4 * 120
        + dr4 * 90
        + tail_hit * 120
        + tail_edge * 200
        + abstain_correct * 80
        + max(0.0, 0.4 - regret) * 50
        + max(0.0, 0.25 - whipsaw_rate) * 120
        + min(enter_count, 120) * 0.18
        - penalty
    )


def build_search_space():
    return {
        "expansion_watch_threshold": [34.0, 38.0, 40.0],
        "high_breakout_threshold": [50.0, 54.0, 56.0],
        "directional_expansion_threshold": [64.0, 68.0, 70.0],
        "event_directional_setup_weight": [1.15, 1.35, 1.55],
        "event_momentum_setup_weight": [1.55, 1.75],
        "anti_chop_margin_buffer": [0.45, 0.6, 0.8],
        "anti_chop_trigger_buffer": [0.2, 0.3, 0.4],
        "anti_chop_tradeability_floor": [60.0, 64.0, 68.0],
        "anti_chop_penalty": [4.0, 6.0, 8.0],
        "warning_upshift_buffer": [1.5, 2.0, 2.5],
        "warning_downshift_buffer": [3.0, 4.0, 5.0],
        "warning_min_dwell_bars": [2, 3, 4],
        "breakout_bias_deadband": [0.55, 0.65, 0.8],
        "breakout_bias_hold_bars": [2, 3, 4],
        "fakeout_risk_penalty": [4.0, 6.0, 8.0],
        "direction_entry_threshold": [6.0, 7.0, 8.0],
        "direction_hold_threshold": [4.0, 5.0],
    }


def evaluate_candidate(df, current, search, keys, index_tuple):
    params_update = {}
    for key, idx in zip(keys, index_tuple):
        params_update[key] = search[key][idx]
    params = normalize_strategy_params({**current, **params_update})

    signals, states, _ = generate_signals(df, params=params)
    trades = _simulate_from_signals(df, signals)
    summary = _trade_summary(trades)
    big_move = summarize_big_move_metrics(df, states)
    exec_metrics = summarize_execution_state_metrics(df, states)
    tail_metrics = summarize_tail_event_metrics(df, states)
    transition_metrics = summarize_transition_metrics(df, states)
    decision_quality = summarize_decision_quality_metrics(df, states)
    label_metrics = summarize_large_move_labels(df, states)
    quality_gate = summarize_quality_gate_metrics(
        summary,
        big_move,
        exec_metrics,
        transition_metrics,
        tail_metrics,
        decision_quality,
    )
    score = score_candidate(summary, big_move, exec_metrics, tail_metrics, transition_metrics, quality_gate, decision_quality)

    return {
        "score": round(score, 4),
        "params": {k: params[k] for k in keys},
        "trade_summary": summary,
        "big_move": big_move,
        "execution": exec_metrics,
        "tail_event": tail_metrics,
        "transition": transition_metrics,
        "decision_quality": decision_quality,
        "labels": label_metrics,
        "quality_gate": quality_gate,
        "_index_tuple": list(index_tuple),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Staged quant tuning search (replaces impractical full exhaustive grid).")
    parser.add_argument("--random-evals", type=int, default=20000, help="Stage-1 random samples.")
    parser.add_argument("--refine-topk", type=int, default=50, help="How many top random candidates to locally refine.")
    parser.add_argument("--max-evals", type=int, default=25000, help="Hard cap on total evaluations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    current = json.loads(CONFIG_FILE.read_text())
    search = build_search_space()
    keys = list(search)

    full_space = 1
    for key in keys:
        full_space *= len(search[key])

    rng = random.Random(args.seed)
    tested = set()
    results = []
    best = None
    count = 0

    random_target = min(max(1, int(args.random_evals)), full_space, max(1, int(args.max_evals)))
    print(
        "STAGED_SEARCH",
        json.dumps(
            {
                "full_space": full_space,
                "stage1_random_target": random_target,
                "refine_topk": int(args.refine_topk),
                "max_evals": int(args.max_evals),
                "seed": int(args.seed),
            }
        ),
        flush=True,
    )

    while len(tested) < random_target and count < int(args.max_evals):
        idx_tuple = tuple(rng.randrange(len(search[k])) for k in keys)
        if idx_tuple in tested:
            continue
        tested.add(idx_tuple)
        count += 1
        item = evaluate_candidate(df, current, search, keys, idx_tuple)
        results.append(item)
        if best is None or item["score"] > best["score"]:
            best = item
            print(
                "NEW_BEST",
                json.dumps(
                    {
                        "score": best["score"],
                        "params": best["params"],
                        "trade_summary": best["trade_summary"],
                        "watch": best["big_move"].get("warning_watch", {}),
                        "directional": best["big_move"].get("warning_directional", {}),
                        "enter": best["execution"].get("enter", {}),
                        "stage": "random",
                    },
                    default=float,
                ),
                flush=True,
            )
        if count % 100 == 0:
            print("PROGRESS", count, flush=True)

    topk = min(max(1, int(args.refine_topk)), len(results))
    seeds = sorted(results, key=lambda x: x["score"], reverse=True)[:topk]

    for seed_item in seeds:
        base = tuple(int(x) for x in seed_item.get("_index_tuple", []))
        if len(base) != len(keys):
            continue
        neighborhood = {base}
        for dim in range(len(keys)):
            for step in (-1, 1):
                nxt = list(base)
                nxt[dim] += step
                if 0 <= nxt[dim] < len(search[keys[dim]]):
                    neighborhood.add(tuple(nxt))
        for idx_tuple in neighborhood:
            if count >= int(args.max_evals):
                break
            if idx_tuple in tested:
                continue
            tested.add(idx_tuple)
            count += 1
            item = evaluate_candidate(df, current, search, keys, idx_tuple)
            results.append(item)
            if best is None or item["score"] > best["score"]:
                best = item
                print(
                    "NEW_BEST",
                    json.dumps(
                        {
                            "score": best["score"],
                            "params": best["params"],
                            "trade_summary": best["trade_summary"],
                            "watch": best["big_move"].get("warning_watch", {}),
                            "directional": best["big_move"].get("warning_directional", {}),
                            "enter": best["execution"].get("enter", {}),
                            "stage": "refine",
                        },
                        default=float,
                    ),
                    flush=True,
                )
            if count % 100 == 0:
                print("PROGRESS", count, flush=True)
        if count >= int(args.max_evals):
            break

    OUT_FILE.write_text(
        json.dumps(
            {
                "mode": "staged_search",
                "tested": count,
                "full_space": full_space,
                "coverage_ratio": round((count / full_space), 8) if full_space else 0.0,
                "best": best,
                "top10": sorted(results, key=lambda x: x["score"], reverse=True)[:10],
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )

    print("FINAL", str(OUT_FILE), flush=True)
    print("TESTED", count, flush=True)
    if best is not None:
        print(json.dumps(best, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
