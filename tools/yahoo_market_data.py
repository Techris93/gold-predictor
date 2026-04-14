from __future__ import annotations

import os
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


YF_OUTPUT_TIMEZONE = "UTC"
YF_GOLD_SYMBOL = os.getenv("YAHOO_GOLD_SYMBOL", "GC=F").strip() or "GC=F"


def canonical_gold_symbol(ticker: Optional[str] = None) -> str:
    raw = str(ticker or "").strip().upper()
    if not raw or raw in {"XAU/USD", "XAUUSD", "XAU-USD", "GOLD", "GC=F"}:
        return YF_GOLD_SYMBOL
    return raw


def interval_to_yahoo(interval: str) -> tuple[str, Optional[str]]:
    raw = str(interval or "1h").strip().lower()
    mapping = {
        "15m": ("15m", None),
        "15min": ("15m", None),
        "30m": ("30m", None),
        "30min": ("30m", None),
        "60m": ("60m", None),
        "60min": ("60m", None),
        "1h": ("60m", None),
        "1hr": ("60m", None),
        "4h": ("60m", "4h"),
        "240m": ("60m", "4h"),
        "240min": ("60m", "4h"),
        "1d": ("1d", None),
        "1day": ("1d", None),
        "1wk": ("1wk", None),
        "1w": ("1wk", None),
        "1week": ("1wk", None),
    }
    return mapping.get(raw, (raw, None))


def _period_to_days(period: str) -> int:
    raw = str(period or "365d").strip().lower()
    if raw.endswith("d"):
        return int(raw[:-1] or "0")
    if raw.endswith("mo"):
        return int(raw[:-2] or "0") * 30
    if raw.endswith("y"):
        return int(raw[:-1] or "0") * 365
    return 365


def _cap_period_for_interval(period: str, interval: str) -> tuple[str, Optional[str]]:
    raw_period = str(period or "365d").strip().lower()
    days = _period_to_days(raw_period)
    if interval in {"15m", "30m"} and days > 60:
        return (
            "60d",
            f"Yahoo Finance caps {interval} history to the last 60d; autoresearch is using 60d plus recent tail bars.",
        )
    return raw_period, None


def _recent_overlay_period(interval: str) -> str:
    if interval in {"15m", "30m"}:
        return "7d"
    if interval == "60m":
        return "30d"
    if interval == "1d":
        return "90d"
    if interval == "1wk":
        return "180d"
    return "7d"


def _normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(part[-1]) for part in df.columns.to_flat_index()]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize(YF_OUTPUT_TIMEZONE)
    else:
        df.index = df.index.tz_convert(YF_OUTPUT_TIMEZONE)

    selected_columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in df.columns]
    df = df[selected_columns].copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    else:
        df["Volume"] = df["Volume"].fillna(0.0)

    df = df.sort_index()
    df = df.dropna(subset=[column for column in ["Open", "High", "Low", "Close"] if column in df.columns])
    return df


def _fetch_symbol_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed; Yahoo Finance autoresearch data is unavailable.")
    history = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        prepost=False,
    )
    return _normalize_history_frame(history)


def _merge_frames(primary: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [primary, recent] if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def _resample_frame(frame: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if not rule or frame.empty:
        return frame
    resampled = frame.resample(rule, label="right", closed="right").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
    return resampled.sort_index()


def fetch_history(period: str = "365d", interval: str = "1h", ticker: Optional[str] = None) -> pd.DataFrame:
    symbol = canonical_gold_symbol(ticker)
    yahoo_interval, resample_rule = interval_to_yahoo(interval)
    effective_period, period_warning = _cap_period_for_interval(period, yahoo_interval)
    primary = _fetch_symbol_history(symbol, effective_period, yahoo_interval)
    if primary.empty:
        raise RuntimeError(f"No Yahoo Finance historical data fetched for {symbol} ({effective_period}, {yahoo_interval}).")

    recent = _fetch_symbol_history(symbol, _recent_overlay_period(yahoo_interval), yahoo_interval)
    merged = _merge_frames(primary, recent)
    merged = _resample_frame(merged, resample_rule)
    if merged.empty:
        raise RuntimeError(f"Yahoo Finance returned empty merged history for {symbol}.")

    merged.attrs["data_source"] = "Yahoo Finance"
    merged.attrs["data_symbol"] = symbol
    if period_warning:
        merged.attrs["data_warning"] = period_warning
    return merged