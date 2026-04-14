from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import load_dotenv

try:
    import pandas as pd
    from twelvedata import TDClient
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing dependencies for Twelve Data integration.") from exc


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TD_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
TD_SYMBOL = os.getenv("TWELVE_DATA_SYMBOL", "XAU/USD").strip() or "XAU/USD"
FRED_REAL_YIELD_SERIES = os.getenv("FRED_REAL_YIELD_SERIES", "DFII10").strip() or "DFII10"
FRED_VIX_SERIES = os.getenv("FRED_VIX_SERIES", "VIXCLS").strip() or "VIXCLS"
FRED_DXY_SERIES = os.getenv("FRED_DXY_SERIES", "DTWEXBGS").strip() or "DTWEXBGS"
MAX_OUTPUTSIZE = 5000
TD_OUTPUT_TIMEZONE = "UTC"

_TD_CLIENT: Optional[TDClient] = None


def _symbol_candidates_from_env(env_name: str, defaults: list[str]) -> list[str]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return defaults

    values = []
    seen = set()
    for token in raw.split(","):
        symbol = token.strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        values.append(symbol)

    return values or defaults

_CROSS_ASSET_SYMBOLS = {
    "dxy": _symbol_candidates_from_env("CROSS_ASSET_DXY_SYMBOLS", ["DX-Y.NYB", "DXY", "UUP"]),
    "silver": ["XAG/USD"],
    "oil": ["USOIL", "WTI"],
    "spx": ["SPY", "ES"],
    "usdjpy": ["USD/JPY"],
    "us10y": ["US10Y", "TNX"],
    "vix": _symbol_candidates_from_env("CROSS_ASSET_VIX_SYMBOLS", ["VIX", "^VIX", "CBOE:VIX", "VIXY", "UVXY"]),
    "real_yield": [FRED_REAL_YIELD_SERIES],
}


def get_td_client() -> Optional[TDClient]:
    global _TD_CLIENT
    if _TD_CLIENT is not None:
        return _TD_CLIENT
    if not TD_API_KEY or TD_API_KEY == "your_twelve_data_api_key_here":
        return None
    try:
        _TD_CLIENT = TDClient(apikey=TD_API_KEY)
    except Exception:
        return None
    return _TD_CLIENT


def canonical_gold_symbol(_ticker: Optional[str] = None) -> str:
    return TD_SYMBOL


def interval_to_twelvedata(interval: str) -> str:
    raw = str(interval or "1h").strip().lower()
    mapping = {
        "1h": "1h",
        "60m": "1h",
        "60min": "1h",
        "1hr": "1h",
        "15m": "15min",
        "15min": "15min",
        "4h": "4h",
        "240m": "4h",
        "240min": "4h",
        "1d": "1day",
        "1day": "1day",
        "1wk": "1week",
        "1w": "1week",
        "1week": "1week",
    }
    return mapping.get(raw, raw)


def bars_for_period(period: str, interval: str) -> int:
    raw_period = str(period or "365d").strip().lower()
    raw_interval = interval_to_twelvedata(interval)

    if raw_period.endswith("d"):
        days = int(raw_period[:-1] or "0")
    elif raw_period.endswith("mo"):
        days = int(raw_period[:-2] or "0") * 30
    elif raw_period.endswith("y"):
        days = int(raw_period[:-1] or "0") * 365
    else:
        days = 365

    bars_per_day = {
        "15min": 24 * 4,
        "1h": 24,
        "4h": 6,
        "1day": 1,
        "1week": 1 / 7,
    }.get(raw_interval, 24)

    bars = int(math.ceil(days * bars_per_day))
    return max(100, min(MAX_OUTPUTSIZE, bars))


def _to_utc_datetime_index(values) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if not isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.DatetimeIndex(parsed)
    if parsed.tz is None:
        return parsed.tz_localize(TD_OUTPUT_TIMEZONE)
    return parsed.tz_convert(TD_OUTPUT_TIMEZONE)


def normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    df = frame.copy()
    df.columns = [str(col).capitalize() for col in df.columns]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Datetime" in df.columns:
        df["Datetime"] = _to_utc_datetime_index(df["Datetime"])
        df = df.set_index("Datetime")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = _to_utc_datetime_index(df.index)
        except Exception:
            pass
    elif df.index.tz is None:
        df.index = df.index.tz_localize(TD_OUTPUT_TIMEZONE)
    else:
        df.index = df.index.tz_convert(TD_OUTPUT_TIMEZONE)

    df = df.sort_index()
    df = df.dropna(subset=[col for col in ["Open", "High", "Low", "Close"] if col in df.columns])

    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    else:
        df["Volume"] = df["Volume"].fillna(0.0)

    return df


def fetch_history(period: str = "365d", interval: str = "1h", ticker: Optional[str] = None) -> pd.DataFrame:
    client = get_td_client()
    if not client:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing or invalid.")

    symbol = canonical_gold_symbol(ticker)
    td_interval = interval_to_twelvedata(interval)
    outputsize = bars_for_period(period, td_interval)

    ts = client.time_series(
        symbol=symbol,
        interval=td_interval,
        outputsize=outputsize,
        timezone=TD_OUTPUT_TIMEZONE,
    )
    df = normalize_ohlcv_frame(ts.as_pandas())
    if df.empty:
        raise RuntimeError(f"No historical data fetched from Twelve Data for {symbol}.")
    return df


def fetch_live_price(ticker: Optional[str] = None) -> Optional[float]:
    client = get_td_client()
    if not client:
        return None
    symbol = canonical_gold_symbol(ticker)
    try:
        payload = client.price(symbol=symbol).as_json()
        if isinstance(payload, dict) and payload.get("price") is not None:
            return float(payload["price"])
    except Exception:
        return None
    return None


def _fetch_symbol_snapshot(client: TDClient, candidates: list[str], interval: str = "1h", outputsize: int = 8) -> dict:
    for symbol in candidates:
        try:
            ts = client.time_series(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                timezone=TD_OUTPUT_TIMEZONE,
            )
            frame = normalize_ohlcv_frame(ts.as_pandas())
            if frame.empty or "Close" not in frame.columns or len(frame) < 2:
                continue
            latest_close = float(frame["Close"].iloc[-1])
            prev_close = float(frame["Close"].iloc[-2])
            pct_1h = ((latest_close - prev_close) / max(prev_close, 1e-8)) * 100.0
            pct_4h = pct_1h
            if len(frame) >= 5:
                close_4h = float(frame["Close"].iloc[-5])
                pct_4h = ((latest_close - close_4h) / max(close_4h, 1e-8)) * 100.0
            trend = "flat"
            if pct_1h > 0.05:
                trend = "up"
            elif pct_1h < -0.05:
                trend = "down"
            return {
                "available": True,
                "symbol": symbol,
                "latest": round(latest_close, 4),
                "pct_1h": round(pct_1h, 4),
                "pct_4h": round(pct_4h, 4),
                "trend": trend,
            }
        except Exception:
            continue
    return {"available": False}


def _fetch_fred_series_snapshot(series_id: str, *, fallback_symbol: str = "", attempts: int = 2, timeout_seconds: int = 8) -> dict:
    if not FRED_API_KEY:
        return {"available": False}

    tries = max(1, int(attempts))
    for _ in range(tries):
        try:
            query = urlencode(
                {
                    "series_id": series_id,
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 128,
                }
            )
            url = f"https://api.stlouisfed.org/fred/series/observations?{query}"
            with urlopen(url, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))

            observations = payload.get("observations") if isinstance(payload, dict) else None
            if not isinstance(observations, list):
                continue

            values = []
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                raw = str(obs.get("value", "")).strip()
                if raw in {"", ".", "NaN", "nan"}:
                    continue
                try:
                    values.append(float(raw))
                except Exception:
                    continue

            if len(values) < 2:
                continue

            latest = float(values[0])
            prev_1 = float(values[1])
            prev_5 = float(values[5]) if len(values) >= 6 else prev_1

            pct_1h = ((latest - prev_1) / max(abs(prev_1), 1e-8)) * 100.0
            pct_4h = ((latest - prev_5) / max(abs(prev_5), 1e-8)) * 100.0

            trend = "flat"
            if pct_1h > 0.20:
                trend = "up"
            elif pct_1h < -0.20:
                trend = "down"

            return {
                "available": True,
                "symbol": fallback_symbol or series_id,
                "latest": round(latest, 4),
                "pct_1h": round(pct_1h, 4),
                "pct_4h": round(pct_4h, 4),
                "trend": trend,
                "source": "fred",
            }
        except Exception:
            continue

    return {"available": False}


def fetch_cross_asset_context() -> dict:
    client = get_td_client()

    assets = {}
    for name, candidates in _CROSS_ASSET_SYMBOLS.items():
        snapshot = {"available": False}
        if client:
            snapshot = _fetch_symbol_snapshot(client, candidates)
        if not snapshot.get("available") and name == "real_yield":
            # Real-yield series may be unavailable via market feed; use FRED when configured.
            snapshot = _fetch_fred_series_snapshot(
                FRED_REAL_YIELD_SERIES,
                fallback_symbol=f"FRED:{FRED_REAL_YIELD_SERIES}",
            )
        elif not snapshot.get("available") and name == "vix":
            snapshot = _fetch_fred_series_snapshot(
                FRED_VIX_SERIES,
                fallback_symbol=f"FRED:{FRED_VIX_SERIES}",
            )
        elif not snapshot.get("available") and name == "dxy":
            snapshot = _fetch_fred_series_snapshot(
                FRED_DXY_SERIES,
                fallback_symbol=f"FRED:{FRED_DXY_SERIES}",
            )
        assets[name] = snapshot

    return {
        "available": any(isinstance(item, dict) and item.get("available") for item in assets.values()),
        "assets": assets,
    }
