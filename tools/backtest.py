import yfinance as yf
import pandas as pd
import ta
import numpy as np

def generate_signals(df):
    """
    Simulates the agent's logic across historical data to generate Buy/Sell signals.
    Strategy:
    - Bullish if EMA 20 > EMA 50 AND CMF > 0 AND OBV is rising AND (Price Action is Bullish Structure OR Bullish Engulfing OR RSI < 30)
    - Bearish if EMA 20 < EMA 50 AND CMF < 0 AND OBV is falling AND (Price Action is Bearish Structure OR Bearish Engulfing OR RSI > 70)
    """
    # Calculate indicators
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['CMF_14'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).chaikin_money_flow()
    
    signals = pd.Series('Neutral', index=df.index)
    
    # Needs at least 3 periods for price action structure calculation
    for i in range(2, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        
        # Trend
        bull_trend = current['Close'] > current['EMA_20'] > current['EMA_50']
        bear_trend = current['Close'] < current['EMA_20'] < current['EMA_50']
        
        # Volume
        obv_rising = current['OBV'] > prev['OBV']
        cmf_bull = current['CMF_14'] > 0
        cmf_bear = current['CMF_14'] < 0
        
        # Price Action
        pa_bull_struct = current['High'] > prev['High'] > prev2['High'] and current['Low'] > prev['Low'] > prev2['Low']
        pa_bear_struct = current['High'] < prev['High'] < prev2['High'] and current['Low'] < prev['Low'] < prev2['Low']
        
        bull_engulfing = current['Close'] > current['Open'] and prev['Close'] < prev['Open'] and current['Close'] > prev['Open'] and current['Open'] < prev['Close']
        bear_engulfing = current['Close'] < current['Open'] and prev['Close'] > prev['Open'] and current['Close'] < prev['Open'] and current['Open'] > prev['Close']
        
        rsi_oversold = current['RSI_14'] < 30
        rsi_overbought = current['RSI_14'] > 70

        if bull_trend and obv_rising and cmf_bull and (pa_bull_struct or bull_engulfing or rsi_oversold):
            signals.iloc[i] = 'Buy'
        elif bear_trend and not obv_rising and cmf_bear and (pa_bear_struct or bear_engulfing or rsi_overbought):
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
    
    # Basic Backtest Engine
    position = 0 # 0 = flat, 1 = long, -1 = short
    entry_price = 0
    trades = []
    
    # We enter on the Open of the next candle after a signal
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
                # Close long position, open short
                pnl = (next_open - entry_price) / entry_price
                trades.append({'type': 'Long', 'pnl': pnl})
                position = -1
                entry_price = next_open
        elif position == -1:
            if signal == 'Buy':
                # Close short position, open long
                pnl = (entry_price - next_open) / entry_price
                trades.append({'type': 'Short', 'pnl': pnl})
                position = 1
                entry_price = next_open

    # Close any open positions at the end of the period
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
        
        # Using compounding starting from $10,000 without leverage
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
    run_backtest(period="730d") # Max hourly data from Yahoo usually ~730d
