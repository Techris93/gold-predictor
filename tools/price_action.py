import pandas as pd


def classify_price_action(frame: pd.DataFrame, index: int) -> tuple[str, str]:
    current = frame.iloc[index]
    prev = frame.iloc[index - 1] if index >= 1 else current
    prev2 = frame.iloc[index - 2] if index >= 2 else prev

    pa_structure = "Consolidating"
    if index >= 20:
        recent_high = frame["High"].iloc[index - 20:index].max()
        recent_low = frame["Low"].iloc[index - 20:index].min()
        if current["Close"] > recent_high:
            pa_structure = "Bullish Breakout"
        elif current["Close"] < recent_low:
            pa_structure = "Bearish Breakdown"

    if pa_structure == "Consolidating" and index >= 2:
        if current["High"] > prev["High"] > prev2["High"] and current["Low"] > prev["Low"] > prev2["Low"]:
            pa_structure = "Higher Highs / Higher Lows (Bullish Structure)"
        elif current["High"] < prev["High"] < prev2["High"] and current["Low"] < prev["Low"] < prev2["Low"]:
            pa_structure = "Lower Highs / Lower Lows (Bearish Structure)"

    if pa_structure == "Consolidating" and index >= 11:
        recent12 = frame.iloc[index - 11:index + 1]
        high_now = recent12["High"].iloc[-4:].mean()
        high_prev = recent12["High"].iloc[-8:-4].mean()
        low_now = recent12["Low"].iloc[-4:].mean()
        low_prev = recent12["Low"].iloc[-8:-4].mean()
        close_now = recent12["Close"].iloc[-3:].mean()
        close_prev = recent12["Close"].iloc[-6:-3].mean()

        if high_now > high_prev and low_now > low_prev and close_now > close_prev:
            pa_structure = "Bullish Drift"
        elif high_now < high_prev and low_now < low_prev and close_now < close_prev:
            pa_structure = "Bearish Drift"

    if pa_structure == "Consolidating" and index >= 19:
        recent20 = frame.iloc[index - 19:index + 1]
        range_high = recent20["High"].max()
        range_low = recent20["Low"].min()
        range_size = max(range_high - range_low, 1e-8)
        close_pos = (current["Close"] - range_low) / range_size
        ema_trend = current.get("EMA_TREND", "Neutral")
        if ema_trend == "Bullish" and close_pos >= 0.67:
            pa_structure = "Bullish Pressure in Range"
        elif ema_trend == "Bearish" and close_pos <= 0.33:
            pa_structure = "Bearish Pressure in Range"

    candle_pattern = "None"
    if (
        current["Close"] > current["Open"]
        and prev["Close"] < prev["Open"]
        and current["Close"] > prev["Open"]
        and current["Open"] < prev["Close"]
    ):
        candle_pattern = "Bullish Engulfing"
    elif (
        current["Close"] < current["Open"]
        and prev["Close"] > prev["Open"]
        and current["Close"] < prev["Open"]
        and current["Open"] > prev["Close"]
    ):
        candle_pattern = "Bearish Engulfing"
    else:
        candle_range = max(current["High"] - current["Low"], 1e-8)
        body = abs(current["Close"] - current["Open"])
        upper_wick = current["High"] - max(current["Close"], current["Open"])
        lower_wick = min(current["Close"], current["Open"]) - current["Low"]

        if body / candle_range < 0.15:
            candle_pattern = "Doji"
        elif lower_wick / candle_range > 0.55 and upper_wick / candle_range < 0.2 and current["Close"] >= current["Open"]:
            candle_pattern = "Bullish Hammer"
        elif upper_wick / candle_range > 0.55 and lower_wick / candle_range < 0.2 and current["Close"] <= current["Open"]:
            candle_pattern = "Bearish Shooting Star"

    return pa_structure, candle_pattern


def annotate_price_action(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["PA_STRUCTURE"] = "Consolidating"
    annotated["CANDLE_PATTERN"] = "None"

    for index in range(len(annotated)):
        pa_structure, candle_pattern = classify_price_action(annotated, index)
        annotated.iloc[index, annotated.columns.get_loc("PA_STRUCTURE")] = pa_structure
        annotated.iloc[index, annotated.columns.get_loc("CANDLE_PATTERN")] = candle_pattern

    return annotated


def extract_price_action_feature_hits(structure: str, candle_pattern: str) -> dict:
    structure = structure or "Consolidating"
    candle_pattern = candle_pattern or "None"
    structure_breakout = "Breakout" in structure or "Breakdown" in structure
    structure_swing = "Structure" in structure
    structure_drift = "Drift" in structure
    structure_range_pressure = "Pressure" in structure
    return {
        "market_structure": structure,
        "candle_pattern": candle_pattern,
        "structure_breakout": structure_breakout,
        "structure_swing": structure_swing,
        "structure_drift": structure_drift,
        "structure_trend": structure_swing or structure_drift,
        "structure_range_pressure": structure_range_pressure,
        "candle_engulfing": "Engulfing" in candle_pattern,
        "candle_reversal": candle_pattern in {"Bullish Hammer", "Bearish Shooting Star"},
        "candle_doji": candle_pattern == "Doji",
        "bullish_signal": "Bullish" in structure or "Bullish" in candle_pattern,
        "bearish_signal": "Bearish" in structure or "Bearish" in candle_pattern,
    }
