import json
import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools.signal_engine import (
    build_ta_payload_from_row,
    compute_prediction_from_ta,
    normalize_strategy_params,
    prepare_historical_features,
)

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
    "breakout_weight": 2.0,
    "structure_weight": 1.0,
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

def generate_signals(df):
    params = normalize_strategy_params(ACTIVE_BACKTEST_PARAMS)
    enriched = prepare_historical_features(df, params)
    signals = pd.Series('Neutral', index=enriched.index)
    sentiment_summary = {"label": "Neutral"}

    for i in range(len(enriched)):
        ta_payload = build_ta_payload_from_row(enriched.iloc[i], params)
        verdict = compute_prediction_from_ta(ta_payload, sentiment_summary)["verdict"]
        if verdict == 'Bullish':
            signals.iloc[i] = 'Buy'
        elif verdict == 'Bearish':
            signals.iloc[i] = 'Sell'

    return signals

def run_backtest(ticker="GC=F", period="2y", interval="1h"):
    print(f"Fetching {period} of {interval} data for {ticker}...")
    gold = yf.Ticker(ticker)
    df = gold.history(period=period, interval=interval)

    if df.empty:
        print("Failed to fetch historical data.")
        return

    print(f"Loaded {len(df)} historical candles.")
    print("Calculating indicators and applying strategy logic...")

    signals = generate_signals(df)

    position = 0
    entry_price = 0
    trades = []

    for i in range(len(df) - 1):
        signal = signals.iloc[i]
        next_open = df['Open'].iloc[i+1]

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
    else:
        print("No trades triggered.")

if __name__ == "__main__":
    run_backtest(period="730d")
