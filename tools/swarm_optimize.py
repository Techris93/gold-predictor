import json
import yfinance as yf
import pandas as pd
import ta
import itertools
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor


def generate_signals_custom(df, ema_short_win, ema_long_win, rsi_ob, rsi_os, cmf_win):
    """Generates signals using custom parameters for the swarm optimization."""
    df = df.copy()

    df['EMA_S'] = ta.trend.EMAIndicator(df['Close'], window=ema_short_win).ema_indicator()
    df['EMA_L'] = ta.trend.EMAIndicator(df['Close'], window=ema_long_win).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=cmf_win).chaikin_money_flow()

    signals = pd.Series('Neutral', index=df.index)

    for i in range(2, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        prev2 = df.iloc[i-2]

        bull_trend = current['Close'] > current['EMA_S'] > current['EMA_L']
        bear_trend = current['Close'] < current['EMA_S'] < current['EMA_L']

        obv_rising = current['OBV'] > prev['OBV']
        cmf_bull = current['CMF'] > 0
        cmf_bear = current['CMF'] < 0

        pa_bull = current['High'] > prev['High'] > prev2['High'] and current['Low'] > prev['Low'] > prev2['Low']
        pa_bear = current['High'] < prev['High'] < prev2['High'] and current['Low'] < prev['Low'] < prev2['Low']

        rsi_oversold = current['RSI'] < rsi_os
        rsi_overbought = current['RSI'] > rsi_ob

        if bull_trend and obv_rising and cmf_bull and (pa_bull or rsi_oversold):
            signals.iloc[i] = 'Buy'
        elif bear_trend and not obv_rising and cmf_bear and (pa_bear or rsi_overbought):
            signals.iloc[i] = 'Sell'

    return signals


def test_params(args):
    """Worker function to test a specific parameter set."""
    df, p = args
    ema_s, ema_l, rsi_ob, rsi_os, cmf_w = p

    signals = generate_signals_custom(df, ema_s, ema_l, rsi_ob, rsi_os, cmf_w)

    position = 0
    entry_price = 0
    capital = 10000

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
        elif position == 1 and signal == 'Sell':
            pnl = (next_open - entry_price) / entry_price
            capital *= (1 + pnl)
            position = -1
            entry_price = next_open
        elif position == -1 and signal == 'Buy':
            pnl = (entry_price - next_open) / entry_price
            capital *= (1 + pnl)
            position = 1
            entry_price = next_open

    roi = ((capital - 10000) / 10000) * 100
    return (p, roi)


def _replace_param_block(file_path, var_name, params):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = rf"(# SWARM_PARAM_BLOCK_START\n){var_name} = \{{.*?\}}\n(# SWARM_PARAM_BLOCK_END)"
    replacement = (
        f"# SWARM_PARAM_BLOCK_START\n"
        f"{var_name} = {json.dumps(params, indent=4)}\n"
        f"# SWARM_PARAM_BLOCK_END"
    )

    updated, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Could not safely update parameter block in {file_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated)


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

    results.sort(key=lambda x: x[1], reverse=True)

    print("🏆 SWARM OPTIMIZATION LEADERBOARD (Top 5 Strategies)\n")
    print(f"{'Rank':<5} | {'EMA Short':<10} | {'EMA Long':<9} | {'RSI OB':<7} | {'RSI OS':<7} | {'CMF Win':<8} | {'ROI':<10}")
    print("-" * 75)

    for i, (p, roi) in enumerate(results[:5]):
        print(f"#{i+1:<4} | {p[0]:<10} | {p[1]:<9} | {p[2]:<7} | {p[3]:<7} | {p[4]:<8} | {roi:>6.2f}%")

    best_params = results[0][0]
    best_roi = results[0][1]
    best_ema_s, best_ema_l, best_rsi_ob, best_rsi_os, best_cmf = best_params

    print("\n⚡ AUTO-APPLYING WINNING STRATEGY TO AGENT...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    predict_file = os.path.join(base_dir, "predict_gold.py")
    backtest_file = os.path.join(base_dir, "backtest.py")

    predict_params = {
        "ema_short": best_ema_s,
        "ema_long": best_ema_l,
        "rsi_window": 14,
        "rsi_overbought": best_rsi_ob,
        "rsi_oversold": best_rsi_os,
        "adx_window": 14,
        "adx_trending_threshold": 22,
        "adx_weak_trend_threshold": 18,
        "atr_window": 14,
        "atr_trending_percent_threshold": 0.25,
        "cmf_window": best_cmf,
        "cmf_strong_buy_threshold": 0.10,
        "cmf_strong_sell_threshold": -0.10,
        "mtf_intervals": ["15min", "1h", "4h"],
    }

    backtest_params = {
        "ema_short": best_ema_s,
        "ema_long": best_ema_l,
        "rsi_window": 14,
        "rsi_overbought": best_rsi_ob,
        "rsi_oversold": best_rsi_os,
        "cmf_window": best_cmf,
    }

    try:
        _replace_param_block(predict_file, "ACTIVE_STRATEGY_PARAMS", predict_params)
        _replace_param_block(backtest_file, "ACTIVE_BACKTEST_PARAMS", backtest_params)

        print(f"✅ Successfully updated agent code with optimal parameters: EMA {best_ema_s}/{best_ema_l}, RSI {best_rsi_ob}/{best_rsi_os}, CMF {best_cmf}")

        status_output = subprocess.run(["git", "status", "--porcelain"], cwd=base_dir, capture_output=True, text=True)
        if "predict_gold.py" in status_output.stdout or "backtest.py" in status_output.stdout:
            print("🚀 New superior strategy detected! Committing to repository...")
            commit_msg = f"Auto-Evolve: New Best Strategy Found | ROI: {best_roi:.2f}% | EMA: {best_ema_s}/{best_ema_l} CMF: {best_cmf}"
            subprocess.run(["git", "add", "predict_gold.py", "backtest.py"], cwd=base_dir, check=True)
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=base_dir, check=True)
            print("💾 Saved! You can now run `git push` to deploy the new strategy.")
        else:
            print("⚖️ The current applied strategy is still the most optimal. No changes made.")

    except Exception as e:
        print("⚠️ Failed to auto-apply parameters:", e)


if __name__ == "__main__":
    run_swarm()
