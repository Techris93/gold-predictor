import json
from pathlib import Path
import sys

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
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

DATA_FILE = BASE / "data" / "cache" / "market_history" / "XAU_USD_365d_1h.csv"
OUT_FILE = BASE / "tools" / "reports" / "tail_event_backtest.json"


def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"missing data file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    params = normalize_strategy_params()
    signals, states, _ = generate_signals(df, params=params)
    trades = _simulate_from_signals(df, signals)
    summary = _trade_summary(trades)
    big_move = summarize_big_move_metrics(df, states)
    execution = summarize_execution_state_metrics(df, states)
    transitions = summarize_transition_metrics(df, states)
    tail = summarize_tail_event_metrics(df, states)
    labels = summarize_large_move_labels(df, states)
    decision_quality = summarize_decision_quality_metrics(df, states)
    quality_gate = summarize_quality_gate_metrics(
        summary,
        big_move,
        execution,
        transitions,
        tail,
        decision_quality,
    )

    payload = {
        "trade_summary": summary,
        "big_move_metrics": big_move,
        "execution_state_metrics": execution,
        "transition_metrics": transitions,
        "tail_event_metrics": tail,
        "large_move_labels": labels,
        "decision_quality_metrics": decision_quality,
        "quality_gate": quality_gate,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("tail_windows", tail.get("tail_windows"))
    print("model_hit_rate_4h", tail.get("model_hit_rate_4h"))
    print("naive_hit_rate_4h", tail.get("naive_hit_rate_4h"))
    print("edge_vs_naive_4h", tail.get("edge_vs_naive_4h"))
    print("decision_quality", decision_quality)
    print("quality_gate", quality_gate)
    print("report", OUT_FILE)


if __name__ == "__main__":
    main()
