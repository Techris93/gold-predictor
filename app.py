from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
from tools import predict_gold
from tools.signal_engine import (
    compute_prediction_from_ta,
)

try:
    from pywebpush import WebPushException, webpush
except ImportError:
    WebPushException = Exception
    webpush = None

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_interval=20,
    ping_timeout=60,
)

_monitor_lock = threading.Lock()
_monitor_state = {
    "started": False,
    "clients": 0,
}


def _read_int_env(name, default, minimum):
    raw = os.getenv(name, str(default))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _read_bool_env(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


MONITOR_INTERVAL_SECONDS = _read_int_env("INDICATOR_MONITOR_INTERVAL_SECONDS", 10, 2)
NOTIFY_MIN_INTERVAL_SECONDS = _read_int_env("INDICATOR_NOTIFY_MIN_INTERVAL_SECONDS", 6, 1)
PREDICTION_CONFIRMATION_COUNT = _read_int_env("PREDICTION_CONFIRMATION_COUNT", 2, 1)
DECISION_CONFIRMATION_COUNT = _read_int_env("DECISION_CONFIRMATION_COUNT", 2, 1)
ENTER_DECISION_CONFIRMATION_COUNT = _read_int_env("ENTER_DECISION_CONFIRMATION_COUNT", 2, 1)
EXIT_DECISION_CONFIRMATION_COUNT = _read_int_env("EXIT_DECISION_CONFIRMATION_COUNT", 3, 1)
WAIT_DECISION_CONFIRMATION_COUNT = _read_int_env("WAIT_DECISION_CONFIRMATION_COUNT", 4, 1)
DIRECTIONAL_REVERSAL_CONFIRMATION_COUNT = _read_int_env("DIRECTIONAL_REVERSAL_CONFIRMATION_COUNT", 4, 1)
ALERT_COOLDOWN_SECONDS = _read_int_env("ALERT_COOLDOWN_SECONDS", 900, 0)
ALERT_CLASS_COOLDOWN_SECONDS = _read_int_env("ALERT_CLASS_COOLDOWN_SECONDS", 600, 0)
ALERT_CONTEXT_COOLDOWN_SECONDS = _read_int_env("ALERT_CONTEXT_COOLDOWN_SECONDS", 900, 0)
PRICE_ACTION_ALERT_COOLDOWN_SECONDS = _read_int_env("PRICE_ACTION_ALERT_COOLDOWN_SECONDS", 900, 0)
DECISION_FLIP_MIN_HOLD_SECONDS = _read_int_env("DECISION_FLIP_MIN_HOLD_SECONDS", 300, 0)
PLAYBOOK_CONFIRMATION_COUNT = _read_int_env("PLAYBOOK_CONFIRMATION_COUNT", 2, 1)
ENTER_PLAYBOOK_CONFIRMATION_COUNT = _read_int_env("ENTER_PLAYBOOK_CONFIRMATION_COUNT", 3, 1)
EXIT_PLAYBOOK_CONFIRMATION_COUNT = _read_int_env("EXIT_PLAYBOOK_CONFIRMATION_COUNT", 2, 1)
PLAYBOOK_FLIP_MIN_HOLD_SECONDS = _read_int_env("PLAYBOOK_FLIP_MIN_HOLD_SECONDS", 600, 0)
BOUNDARY_WOBBLE_COOLDOWN_SECONDS = _read_int_env("BOUNDARY_WOBBLE_COOLDOWN_SECONDS", 1200, 0)
ENTER_STAGE_PROTECT_SECONDS = _read_int_env("ENTER_STAGE_PROTECT_SECONDS", 900, 0)
NOTIFY_EXIT_READS = _read_bool_env("NOTIFY_EXIT_READS", True)
RR200_MAX_SIGNALS_PER_DAY = _read_int_env("RR200_MAX_SIGNALS_PER_DAY", 5, 1)
RR200_MIN_SIGNAL_SPACING_SECONDS = _read_int_env("RR200_MIN_SIGNAL_SPACING_SECONDS", 2700, 0)
PUSH_EXCLUDED_FIELDS = {
    "rsi_14",
    "ema_20",
    "ema_50",
    "adx_14",
    "atr_percent",
    "micro_vwap_band",
}

BASE_DIR = Path(__file__).resolve().parent
SUBSCRIPTIONS_FILE = BASE_DIR / "tools" / "reports" / "webpush_subscriptions.json"
PREDICTION_STATE_FILE = BASE_DIR / "tools" / "reports" / "prediction_state.json"
DECISION_STATE_FILE = BASE_DIR / "tools" / "reports" / "decision_state.json"
ALERT_STATE_FILE = BASE_DIR / "tools" / "reports" / "alert_state.json"
PLAYBOOK_STATE_FILE = BASE_DIR / "tools" / "reports" / "playbook_state.json"
REGIME_MEMORY_FILE = BASE_DIR / "tools" / "reports" / "regime_memory_state.json"
LIVE_SIGNAL_OUTCOMES_FILE = BASE_DIR / "tools" / "reports" / "live_signal_outcomes.json"
LIVE_SIGNAL_SUMMARY_FILE = BASE_DIR / "tools" / "reports" / "live_signal_summary.json"
RR200_SIGNAL_COUNTER_FILE = BASE_DIR / "tools" / "reports" / "rr200_signal_counter.json"
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_CLAIMS_SUBJECT = os.getenv("VAPID_CLAIMS_SUBJECT", "mailto:alerts@example.com")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID", "").strip()
ENABLE_TELEGRAM_NOTIFICATIONS = _read_bool_env("GOLD_PREDICTOR_ENABLE_TELEGRAM", False)
CHANGE_SUMMARY_ORDER = [
    "trade_playbook_stage",
    "warning_ladder",
    "event_regime",
    "breakout_bias",
    "rr_signal_status",
    "rr_signal_grade",
    "rr_signal_direction",
    "market_structure",
    "micro_vwap_band",
    "micro_vwap_bias",
    "micro_orb_state",
    "micro_sweep_state",
    "verdict",
    "confidence_bucket",
    "execution_permission",
    "entry_readiness",
    "exit_urgency",
]
def _load_subscriptions():
    if not SUBSCRIPTIONS_FILE.exists():
        return []
    try:
        data = json.loads(SUBSCRIPTIONS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_json_file(path, default):
    try:
        if not path.exists():
            return default
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else default
    except Exception:
        return default


def _humanize_value(value):
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ")
    words = []
    for word in text.split():
        if word.upper() in {"AI", "RSI", "EMA", "ATR", "ADX", "XAUUSD"}:
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _save_json_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _save_subscriptions(subscriptions):
    SUBSCRIPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUBSCRIPTIONS_FILE.write_text(
        json.dumps(subscriptions, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _upsert_subscription(subscription):
    if not isinstance(subscription, dict):
        return
    endpoint = subscription.get("endpoint")
    keys = subscription.get("keys") if isinstance(subscription.get("keys"), dict) else {}
    if not endpoint or not keys.get("p256dh") or not keys.get("auth"):
        return

    subscriptions = _load_subscriptions()
    for idx, existing in enumerate(subscriptions):
        if existing.get("endpoint") == endpoint:
            subscriptions[idx] = subscription
            _save_subscriptions(subscriptions)
            return

    subscriptions.append(subscription)
    _save_subscriptions(subscriptions)


def _remove_subscription_endpoint(endpoint):
    if not endpoint:
        return
    subscriptions = _load_subscriptions()
    filtered = [item for item in subscriptions if item.get("endpoint") != endpoint]
    if len(filtered) != len(subscriptions):
        _save_subscriptions(filtered)


def _has_push_subscribers():
    return len(_load_subscriptions()) > 0


def _filter_notification_changes(changes):
    """Remove fields that should not appear in push notifications."""
    if not isinstance(changes, dict):
        return {}
    filtered = {
        key: val
        for key, val in changes.items()
        if key not in PUSH_EXCLUDED_FIELDS
    }
    if "micro_vwap_bias" not in filtered:
        filtered.pop("micro_vwap_delta_pct", None)
    return filtered


def _summarize_changes_for_push(changes):
    labels = {
        "trade_playbook_stage": "Trade Playbook",
        "warning_ladder": "Warning Ladder",
        "event_regime": "Event Regime",
        "breakout_bias": "Breakout Bias",
        "rr_signal_status": "RR200 Signal",
        "rr_signal_grade": "Quant Grade",
        "rr_signal_direction": "RR Direction",
        "market_structure": "Market Structure",
        "micro_vwap_band": "VWAP Band",
        "micro_vwap_bias": "VWAP Bias",
        "micro_orb_state": "ORB",
        "micro_sweep_state": "Sweep",
        "verdict": "Verdict",
        "confidence_bucket": "AI Confidence",
        "execution_permission": "Execution Permission",
        "entry_readiness": "Entry Readiness",
        "exit_urgency": "Exit Urgency",
    }
    ordered_keys = [key for key in CHANGE_SUMMARY_ORDER if key in changes]
    ordered_keys.extend(key for key in changes if key not in ordered_keys)
    parts = []
    for key in ordered_keys[:3]:
        val = changes.get(key, {})
        prev = val.get("previous") if isinstance(val, dict) else None
        cur = val.get("current") if isinstance(val, dict) else None
        parts.append(
            f"{labels.get(key, key)}: {_humanize_alert_value(key, prev)} -> {_humanize_alert_value(key, cur)}"
        )
    return " | ".join(parts) if parts else "Execution permission changed"


def _notification_title_for_changes(changes):
    changed_keys = set((changes or {}).keys())
    has_playbook = "trade_playbook_stage" in changed_keys
    has_structure = "market_structure" in changed_keys
    has_warning = "warning_ladder" in changed_keys
    has_event_regime = "event_regime" in changed_keys
    has_breakout_bias = "breakout_bias" in changed_keys
    has_rr_status = "rr_signal_status" in changed_keys
    has_rr_grade = "rr_signal_grade" in changed_keys
    has_rr_direction = "rr_signal_direction" in changed_keys
    has_micro_vwap = "micro_vwap_band" in changed_keys or "micro_vwap_bias" in changed_keys
    has_micro_orb = "micro_orb_state" in changed_keys
    has_micro_sweep = "micro_sweep_state" in changed_keys
    has_microstructure = has_micro_vwap or has_micro_orb or has_micro_sweep
    has_verdict = "verdict" in changed_keys
    has_confidence = "confidence_bucket" in changed_keys
    has_permission = "execution_permission" in changed_keys
    has_entry_readiness = "entry_readiness" in changed_keys
    has_exit_urgency = "exit_urgency" in changed_keys

    if has_rr_status or has_rr_grade or has_rr_direction:
        return "XAUUSD RR200 Signal Changed"
    if has_microstructure and not has_structure and not has_permission:
        return "XAUUSD Microstructure Changed"
    if has_playbook and has_warning and not has_structure and not has_permission:
        return "XAUUSD Trade Setup Changed"
    if has_playbook and (has_breakout_bias or has_entry_readiness or has_exit_urgency) and not has_structure and not has_permission:
        return "XAUUSD Trade Setup Changed"
    if has_playbook and not has_warning and not has_event_regime and not has_structure and not has_permission:
        return "XAUUSD Trade Playbook Changed"
    if (has_warning or has_event_regime or has_breakout_bias) and not has_structure and not has_permission and not has_verdict and not has_confidence:
        return "XAUUSD Trade Context Changed"
    if has_verdict and not has_structure and not has_permission and not has_confidence:
        return "XAUUSD Verdict Changed"
    if has_confidence and not has_structure and not has_permission and not has_verdict:
        return "XAUUSD Confidence Changed"
    if has_structure and not has_permission:
        return "XAUUSD Market Structure Changed"
    if has_exit_urgency and not has_structure and not has_permission:
        return "XAUUSD Risk State Changed"
    if has_entry_readiness and not has_structure and not has_permission:
        return "XAUUSD Trade Setup Changed"
    if has_permission and not has_structure:
        return "XAUUSD Execution Permission Changed"
    if (has_playbook or has_verdict or has_confidence or has_warning or has_event_regime or has_breakout_bias or has_entry_readiness or has_exit_urgency or has_rr_status or has_rr_grade or has_rr_direction) and not has_structure and not has_permission:
        return "XAUUSD State Changed"
    if has_permission and (has_structure or has_verdict or has_confidence or has_warning or has_event_regime or has_playbook or has_breakout_bias or has_entry_readiness or has_exit_urgency):
        return "XAUUSD Structure / Execution Changed"
    if has_playbook or has_verdict or has_confidence or has_warning or has_event_regime or has_breakout_bias or has_entry_readiness or has_exit_urgency or has_rr_status or has_rr_grade or has_rr_direction:
        return "XAUUSD State Changed"
    if has_microstructure:
        return "XAUUSD Microstructure Changed"
    return "XAUUSD Execution Permission Changed"


def _effective_rr_direction(rr_signal):
    rr_signal = rr_signal if isinstance(rr_signal, dict) else {}
    direction = str(rr_signal.get("direction") or "Neutral").strip() or "Neutral"
    if direction in {"Bullish", "Bearish"}:
        return direction
    candidate_direction = str(rr_signal.get("candidateDirection") or "").strip()
    if candidate_direction in {"Bullish", "Bearish"}:
        return candidate_direction
    return direction


def _format_rr_direction_grade(rr_signal):
    rr_signal = rr_signal if isinstance(rr_signal, dict) else {}
    direction = _effective_rr_direction(rr_signal)
    grade = str(rr_signal.get("grade") or "N/A").strip() or "N/A"
    return f"{direction} · {grade}"


def _format_rr_target_bucket_probability(rr_signal):
    rr_signal = rr_signal if isinstance(rr_signal, dict) else {}
    target_bucket = str(rr_signal.get("targetBucket") or "").strip()
    if target_bucket:
        target_bucket = target_bucket.replace("_atr", " ATR").replace("_", " ")
    if not target_bucket:
        target_bucket_atr = rr_signal.get("targetBucketAtr")
        if isinstance(target_bucket_atr, (int, float)):
            target_bucket = f"{float(target_bucket_atr):.1f} ATR"
        else:
            target_move_atr = rr_signal.get("targetMoveAtr")
            if isinstance(target_move_atr, (int, float)):
                target_bucket = f"{float(target_move_atr):.1f} ATR"
    if target_bucket and bool(rr_signal.get("targetBucketIsProxy")):
        target_bucket = f"{target_bucket} proxy"

    move_probability = rr_signal.get("targetBucketProbability")
    if not isinstance(move_probability, (int, float)):
        move_probability = rr_signal.get("moveProbability")
    probability_text = "N/A"
    if isinstance(move_probability, (int, float)):
        probability_text = f"{float(move_probability) * 100.0:.1f}%"

    return f"{target_bucket or 'N/A'} · {probability_text}"


def _format_market_structure_change(changes, market_structure):
    changes = changes if isinstance(changes, dict) else {}
    structure_change = changes.get("market_structure")
    current_structure = str(market_structure or "").strip()
    if isinstance(structure_change, dict):
        previous = str(structure_change.get("previous") or "").strip()
        current = str(structure_change.get("current") or current_structure).strip()
        if previous and current and previous != current:
            return f"{previous} -> {current}"
        if current:
            return current
        if previous:
            return previous
    return current_structure or "N/A"


def _orb_state_label(value):
    try:
        numeric = int(round(float(value)))
    except Exception:
        numeric = 0
    return "Bullish" if numeric > 0 else ("Bearish" if numeric < 0 else "Neutral")


def _sweep_state_label(value):
    try:
        numeric = int(round(float(value)))
    except Exception:
        numeric = 0
    return "Bullish" if numeric > 0 else ("Bearish" if numeric < 0 else "None")


def _vwap_band_label(value):
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    band_floor = math.floor(numeric * 10.0) / 10.0
    band_ceil = band_floor + 0.1
    return f"{band_floor:+.1f}% to {band_ceil:+.1f}%"


def _vwap_bias_label(value):
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    if numeric >= 0.30:
        return "Bullish"
    if numeric >= 0.10:
        return "Mild Bullish"
    if numeric > -0.10:
        return "Neutral"
    if numeric > -0.30:
        return "Mild Bearish"
    return "Bearish"


def _format_vwap_microstructure_text(value):
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    return f"VWAP {numeric:.2f}% {_vwap_bias_label(numeric)}"


def _vwap_bias_strategy_note(bias_label):
    bias = str(bias_label or "").strip()
    if bias == "Bullish":
        return "hold longs, buy pullbacks"
    if bias == "Mild Bullish":
        return "cautious long"
    if bias == "Neutral":
        return "wait for break"
    if bias == "Mild Bearish":
        return "cautious short"
    if bias == "Bearish":
        return "hold shorts, sell rallies"
    return ""


def _format_microstructure_change(changes):
    changes = changes if isinstance(changes, dict) else {}
    parts = []
    vwap_band_delta = changes.get("micro_vwap_band")
    vwap_bias_delta = changes.get("micro_vwap_bias")
    vwap_value_delta = changes.get("micro_vwap_delta_pct")
    if isinstance(vwap_bias_delta, dict) or isinstance(vwap_band_delta, dict):
        prev_bias = str((vwap_bias_delta or {}).get("previous") or "")
        cur_bias = str((vwap_bias_delta or {}).get("current") or "")
        prev_val = (vwap_value_delta or {}).get("previous") if isinstance(vwap_value_delta, dict) else None
        cur_val = (vwap_value_delta or {}).get("current") if isinstance(vwap_value_delta, dict) else None
        value_suffix = ""
        if isinstance(prev_val, (int, float)) and isinstance(cur_val, (int, float)):
            value_suffix = f" ({float(prev_val):.2f}% -> {float(cur_val):.2f}%)"
        if prev_bias and cur_bias and prev_bias != cur_bias:
            strategy_note = _vwap_bias_strategy_note(cur_bias)
            strategy_suffix = f": {strategy_note}" if strategy_note else ""
            parts.append(f"VWAP {prev_bias} -> {cur_bias}{value_suffix}{strategy_suffix}")
    orb_delta = changes.get("micro_orb_state")
    if isinstance(orb_delta, dict):
        prev_orb = _orb_state_label(orb_delta.get("previous"))
        cur_orb = _orb_state_label(orb_delta.get("current"))
        if prev_orb != cur_orb:
            parts.append(f"ORB {prev_orb} -> {cur_orb}")
    sweep_delta = changes.get("micro_sweep_state")
    if isinstance(sweep_delta, dict):
        prev_sweep = _sweep_state_label(sweep_delta.get("previous"))
        cur_sweep = _sweep_state_label(sweep_delta.get("current"))
        if prev_sweep != cur_sweep:
            parts.append(f"Sweep {prev_sweep} -> {cur_sweep}")
    return " · ".join(parts)


def _format_bar_session_microstructure(ta_data):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    session = ta_data.get("session_context") if isinstance(ta_data.get("session_context"), dict) else {}
    bar_session = session.get("bar_session") if isinstance(session.get("bar_session"), dict) else {}
    current_session = session.get("current_session") if isinstance(session.get("current_session"), dict) else {}
    structure = ta_data.get("structure_context") if isinstance(ta_data.get("structure_context"), dict) else {}

    bar_label = str(session.get("label") or bar_session.get("label") or "Off")
    bar_time = str(session.get("barTimeDisplayUtc") or bar_session.get("timeDisplayUtc") or "N/A")
    current_label = str(session.get("currentLabel") or current_session.get("label") or "Off")
    current_time = str(
        session.get("currentTimeDisplayUtc")
        or current_session.get("timeDisplayUtc")
        or datetime.now(timezone.utc).strftime("%H:%M UTC")
    )

    try:
        vwap_delta = float(structure.get("distSessionVwapPct") or 0.0)
    except Exception:
        vwap_delta = 0.0
    try:
        orb = int(round(float(structure.get("openingRangeBreak") or 0.0)))
    except Exception:
        orb = 0
    try:
        sweep = int(round(float(structure.get("sweepReclaimSignal") or 0.0)))
    except Exception:
        sweep = 0

    vwap_text = _format_vwap_microstructure_text(vwap_delta)
    orb_text = f"ORB {_orb_state_label(orb)}"
    sweep_text = "Sweep Bullish" if sweep > 0 else ("Sweep Bearish" if sweep < 0 else "No Sweep")

    return (
        f"Bar {bar_label} {bar_time} · Now {current_label} {current_time} "
        f"· {vwap_text} · {orb_text} · {sweep_text}"
    )


def _join_readable_list(items):
    filtered = [str(item or "").strip() for item in items if str(item or "").strip()]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} and {filtered[1]}"
    return f"{', '.join(filtered[:-1])}, and {filtered[-1]}"


def _dashboard_action_tone(label):
    if label == "Buy":
        return "value-bullish"
    if label == "Sell":
        return "value-bearish"
    if label == "Exit":
        return "value-watch"
    return ""


def _summarize_microstructure_context(ta_data):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    structure = (
        ta_data.get("structure_context")
        if isinstance(ta_data.get("structure_context"), dict)
        else {}
    )
    try:
        vwap_delta = float(structure.get("distSessionVwapPct") or 0.0)
    except Exception:
        vwap_delta = 0.0
    try:
        orb_state = int(round(float(structure.get("openingRangeBreak") or 0.0)))
    except Exception:
        orb_state = 0
    try:
        sweep_state = int(round(float(structure.get("sweepReclaimSignal") or 0.0)))
    except Exception:
        sweep_state = 0

    parts = []
    vwap_bias = _vwap_bias_label(vwap_delta)
    if vwap_bias != "Neutral":
        parts.append(f"VWAP {vwap_bias.lower()}")
    if orb_state != 0:
        parts.append(f"ORB {_orb_state_label(orb_state).lower()}")
    if sweep_state != 0:
        parts.append(f"{_sweep_state_label(sweep_state).lower()} sweep")
    return _join_readable_list(parts)


def _summarize_level_pressure(ta_data):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    support_resistance = (
        ta_data.get("support_resistance")
        if isinstance(ta_data.get("support_resistance"), dict)
        else {}
    )
    nearby_supports = (
        list(support_resistance.get("nearby_supports"))
        if isinstance(support_resistance.get("nearby_supports"), list)
        else []
    )
    nearby_resistances = (
        list(support_resistance.get("nearby_resistances"))
        if isinstance(support_resistance.get("nearby_resistances"), list)
        else []
    )
    support_tone = int(
        support_resistance.get("support_family_confluence")
        or support_resistance.get("support_confluence")
        or len(nearby_supports)
        or 0
    )
    resistance_tone = int(
        support_resistance.get("resistance_family_confluence")
        or support_resistance.get("resistance_confluence")
        or len(nearby_resistances)
        or 0
    )
    if support_tone > resistance_tone:
        return "support is closer and cleaner"
    if resistance_tone > support_tone:
        return "resistance is closer and cleaner"
    return ""


def _coerce_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _level_is_near(current_price, level_price, threshold_pct=0.08):
    price = _coerce_float(current_price, 0.0) or 0.0
    level = _coerce_float(level_price, None)
    if price <= 0.0 or level is None:
        return False
    return abs(price - level) / price * 100.0 <= threshold_pct + 1e-9


def _zone_distance_pct(zone, current_price):
    zone = zone if isinstance(zone, dict) else {}
    price = _coerce_float(current_price, 0.0) or 0.0
    low = _coerce_float(zone.get("low"), None)
    high = _coerce_float(zone.get("high"), None)
    if price <= 0.0 or low is None or high is None:
        return None
    zone_low = min(low, high)
    zone_high = max(low, high)
    if zone_low <= price <= zone_high:
        return 0.0
    if price < zone_low:
        return (zone_low - price) / price * 100.0
    return (price - zone_high) / price * 100.0


def _levels_have_family(levels, families, max_distance_pct=None):
    target_families = {str(family) for family in (families or set())}
    for level in levels if isinstance(levels, list) else []:
        if not isinstance(level, dict):
            continue
        family = str(level.get("family") or "")
        if family not in target_families:
            continue
        distance_pct = _coerce_float(level.get("distance_pct"), None)
        if max_distance_pct is None or (
            distance_pct is not None and distance_pct <= max_distance_pct + 1e-9
        ):
            return True
    return False


def _nearest_level_distance_pct(levels, families=None):
    candidates = []
    target_families = {str(family) for family in (families or set())}
    for level in levels if isinstance(levels, list) else []:
        if not isinstance(level, dict):
            continue
        if target_families and str(level.get("family") or "") not in target_families:
            continue
        distance_pct = _coerce_float(level.get("distance_pct"), None)
        if distance_pct is not None:
            candidates.append(distance_pct)
    return min(candidates) if candidates else None


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _derive_dashboard_action(payload, ta_data=None):
    payload = payload if isinstance(payload, dict) else {}
    ta_data = ta_data if isinstance(ta_data, dict) else (
        payload.get("TechnicalAnalysis")
        if isinstance(payload.get("TechnicalAnalysis"), dict)
        else {}
    )
    market_state = (
        payload.get("MarketState")
        if isinstance(payload.get("MarketState"), dict)
        else {}
    )
    regime_state = (
        payload.get("RegimeState")
        if isinstance(payload.get("RegimeState"), dict)
        else {}
    )
    trade_guidance = (
        payload.get("TradeGuidance")
        if isinstance(payload.get("TradeGuidance"), dict)
        else {}
    )
    decision_status = (
        payload.get("DecisionStatus")
        if isinstance(payload.get("DecisionStatus"), dict)
        else {}
    )
    execution_permission = (
        payload.get("ExecutionPermission")
        if isinstance(payload.get("ExecutionPermission"), dict)
        else {}
    )
    execution_state = (
        payload.get("ExecutionState")
        if isinstance(payload.get("ExecutionState"), dict)
        else {}
    )
    price_action = (
        ta_data.get("price_action")
        if isinstance(ta_data.get("price_action"), dict)
        else {}
    )
    structure_context = (
        ta_data.get("structure_context")
        if isinstance(ta_data.get("structure_context"), dict)
        else {}
    )
    support_resistance = (
        ta_data.get("support_resistance")
        if isinstance(ta_data.get("support_resistance"), dict)
        else {}
    )

    current_price = _coerce_float(ta_data.get("current_price"), 0.0) or 0.0
    structure = str(price_action.get("structure") or "Consolidating").strip()
    breakout_bias = str(regime_state.get("breakout_bias") or "Neutral").strip()
    cross_asset_bias = str(regime_state.get("cross_asset_bias") or "Neutral").strip()
    warning_ladder = str(regime_state.get("warning_ladder") or "Normal").strip()
    event_regime = str(regime_state.get("event_regime") or "normal").strip()
    big_move_risk = _coerce_float(regime_state.get("big_move_risk"), 0.0) or 0.0
    action_state = str(
        market_state.get("action_state")
        or execution_state.get("actionState")
        or "WAIT"
    ).strip()
    raw_action = str(
        market_state.get("action") or execution_state.get("action") or "hold"
    ).strip().lower()
    execution_status = str(execution_state.get("status") or "stand_aside").strip()
    permission_status = str(execution_permission.get("status") or "no_trade").strip()
    direction_bias = str(
        market_state.get("directional_bias")
        or payload.get("directionalBias")
        or "Neutral"
    ).strip()
    buy_level = str(trade_guidance.get("buyLevel") or "Weak").strip()
    sell_level = str(trade_guidance.get("sellLevel") or "Weak").strip()
    exit_level = str(trade_guidance.get("exitLevel") or "Low").strip()
    summary = str(trade_guidance.get("summary") or "").strip()
    sr_reaction = str(support_resistance.get("reaction") or "None").strip()

    nearby_supports = (
        list(support_resistance.get("nearby_supports"))
        if isinstance(support_resistance.get("nearby_supports"), list)
        else []
    )
    nearby_resistances = (
        list(support_resistance.get("nearby_resistances"))
        if isinstance(support_resistance.get("nearby_resistances"), list)
        else []
    )
    support_tone = int(
        support_resistance.get("support_family_confluence")
        or support_resistance.get("support_confluence")
        or len(nearby_supports)
        or 0
    )
    resistance_tone = int(
        support_resistance.get("resistance_family_confluence")
        or support_resistance.get("resistance_confluence")
        or len(nearby_resistances)
        or 0
    )
    support_distance_pct = _coerce_float(
        support_resistance.get("support_distance_pct"),
        None,
    )
    resistance_distance_pct = _coerce_float(
        support_resistance.get("resistance_distance_pct"),
        None,
    )

    pivot_levels = (
        support_resistance.get("pivot_levels")
        if isinstance(support_resistance.get("pivot_levels"), dict)
        else {}
    )
    round_numbers = (
        support_resistance.get("round_numbers")
        if isinstance(support_resistance.get("round_numbers"), dict)
        else {}
    )
    active_fvgs = (
        support_resistance.get("active_fvgs")
        if isinstance(support_resistance.get("active_fvgs"), dict)
        else {}
    )
    range_zone = (
        support_resistance.get("range_zone")
        if isinstance(support_resistance.get("range_zone"), dict)
        else {}
    )

    pivot_point = _coerce_float(
        _first_non_empty(
            structure_context.get("pivotPoint"),
            pivot_levels.get("pp"),
        ),
        None,
    )
    pivot_r1 = _coerce_float(
        _first_non_empty(
            structure_context.get("pivotResistance1"),
            pivot_levels.get("r1"),
        ),
        None,
    )
    pivot_s1 = _coerce_float(
        _first_non_empty(
            structure_context.get("pivotSupport1"),
            pivot_levels.get("s1"),
        ),
        None,
    )
    pivot_r2 = _coerce_float(
        _first_non_empty(
            structure_context.get("pivotResistance2"),
            pivot_levels.get("r2"),
        ),
        None,
    )
    pivot_s2 = _coerce_float(
        _first_non_empty(
            structure_context.get("pivotSupport2"),
            pivot_levels.get("s2"),
        ),
        None,
    )
    round_support = _coerce_float(
        _first_non_empty(
            structure_context.get("roundNumberSupport"),
            round_numbers.get("support"),
        ),
        None,
    )
    round_resistance = _coerce_float(
        _first_non_empty(
            structure_context.get("roundNumberResistance"),
            round_numbers.get("resistance"),
        ),
        None,
    )
    major_round_support = _coerce_float(
        _first_non_empty(
            structure_context.get("majorRoundNumberSupport"),
            round_numbers.get("majorSupport"),
        ),
        None,
    )
    major_round_resistance = _coerce_float(
        _first_non_empty(
            structure_context.get("majorRoundNumberResistance"),
            round_numbers.get("majorResistance"),
        ),
        None,
    )
    bullish_fvg = (
        structure_context.get("bullishFvg")
        if isinstance(structure_context.get("bullishFvg"), dict)
        else active_fvgs.get("bullish")
        if isinstance(active_fvgs.get("bullish"), dict)
        else {}
    )
    bearish_fvg = (
        structure_context.get("bearishFvg")
        if isinstance(structure_context.get("bearishFvg"), dict)
        else active_fvgs.get("bearish")
        if isinstance(active_fvgs.get("bearish"), dict)
        else {}
    )
    if not range_zone:
        range_zone = (
            structure_context.get("rangeZone")
            if isinstance(structure_context.get("rangeZone"), dict)
            else {}
        )

    bullish_fvg_distance_pct = _coerce_float(
        _first_non_empty(
            structure_context.get("bullishFvgDistancePct"),
            _zone_distance_pct(bullish_fvg, current_price),
        ),
        None,
    )
    bearish_fvg_distance_pct = _coerce_float(
        _first_non_empty(
            structure_context.get("bearishFvgDistancePct"),
            _zone_distance_pct(bearish_fvg, current_price),
        ),
        None,
    )
    in_bullish_fvg = bool(structure_context.get("inBullishFvg") or 0)
    in_bearish_fvg = bool(structure_context.get("inBearishFvg") or 0)
    range_zone_active = bool(structure_context.get("rangeZoneActive") or 0)
    in_range_zone = bool(structure_context.get("inRangeZone") or 0)
    range_zone_break = int(round(_coerce_float(structure_context.get("rangeZoneBreak"), 0.0) or 0.0))
    range_zone_position = _coerce_float(structure_context.get("rangeZonePosition"), None)

    vwap_delta = _coerce_float(structure_context.get("distSessionVwapPct"), 0.0) or 0.0
    vwap_bias = _vwap_bias_label(vwap_delta)
    orb_state = int(round(_coerce_float(structure_context.get("openingRangeBreak"), 0.0) or 0.0))
    sweep_state = int(round(_coerce_float(structure_context.get("sweepReclaimSignal"), 0.0) or 0.0))
    micro_bull_count = int(vwap_bias in {"Mild Bullish", "Bullish"}) + int(orb_state > 0) + int(sweep_state > 0)
    micro_bear_count = int(vwap_bias in {"Mild Bearish", "Bearish"}) + int(orb_state < 0) + int(sweep_state < 0)

    bullish_structure = any(
        token in structure
        for token in (
            "Bullish Breakout",
            "Bullish Structure",
            "Bullish Drift",
            "Bullish Pressure in Range",
        )
    )
    bearish_structure = any(
        token in structure
        for token in (
            "Bearish Breakdown",
            "Bearish Structure",
            "Bearish Drift",
            "Bearish Pressure in Range",
        )
    )
    consolidating = "Consolidating" in structure
    bullish_reaction = sr_reaction in {
        "Bullish Support Rejection",
        "Bullish Breakout Through Resistance",
    }
    bearish_reaction = sr_reaction in {
        "Bearish Resistance Rejection",
        "Bearish Breakdown Through Support",
    }
    event_direction = (
        breakout_bias
        if breakout_bias in {"Bullish", "Bearish"}
        else direction_bias
        if direction_bias in {"Bullish", "Bearish"}
        else ""
    )
    bullish_event = (
        event_regime in {"trend_acceleration", "range_expansion", "breakout_watch"}
        and event_direction == "Bullish"
    )
    bearish_event = (
        event_regime in {"trend_acceleration", "range_expansion", "breakout_watch"}
        and event_direction == "Bearish"
    )
    directional_warning = warning_ladder in {
        "Directional Expansion Likely",
        "Active Momentum Event",
    }
    hard_risk_regime = event_regime in {"panic_reversal", "event_risk"}
    support_pressure = support_tone > resistance_tone
    resistance_pressure = resistance_tone > support_tone
    support_cluster_dense_near = bool(
        support_tone >= 2
        and support_distance_pct is not None
        and support_distance_pct <= 0.08
    )
    resistance_cluster_dense_near = bool(
        resistance_tone >= 2
        and resistance_distance_pct is not None
        and resistance_distance_pct <= 0.08
    )
    both_sides_tight = bool(
        support_distance_pct is not None
        and resistance_distance_pct is not None
        and support_distance_pct <= 0.08
        and resistance_distance_pct <= 0.08
    )

    pivot_or_round_support_near = _levels_have_family(
        nearby_supports,
        {"pivot", "round"},
        max_distance_pct=0.10,
    )
    pivot_or_round_resistance_near = _levels_have_family(
        nearby_resistances,
        {"pivot", "round"},
        max_distance_pct=0.10,
    )
    range_or_fvg_support_near = _levels_have_family(
        nearby_supports,
        {"range", "fvg"},
        max_distance_pct=0.10,
    )
    range_or_fvg_resistance_near = _levels_have_family(
        nearby_resistances,
        {"range", "fvg"},
        max_distance_pct=0.10,
    )

    bullish_pivot_round = bool(
        (pivot_point is not None and current_price >= pivot_point)
        or (bullish_reaction and pivot_or_round_support_near)
        or _level_is_near(current_price, pivot_s1, threshold_pct=0.10)
        or _level_is_near(current_price, round_support, threshold_pct=0.10)
        or _level_is_near(current_price, major_round_support, threshold_pct=0.10)
    )
    bearish_pivot_round = bool(
        (pivot_point is not None and current_price <= pivot_point)
        or (
            bearish_reaction
            and (
                pivot_or_round_resistance_near
                or _level_is_near(current_price, pivot_r1, threshold_pct=0.10)
                or _level_is_near(current_price, round_resistance, threshold_pct=0.10)
                or _level_is_near(current_price, major_round_resistance, threshold_pct=0.10)
            )
        )
    )
    bullish_pivot_round_conflict = bool(
        (pivot_point is not None and current_price < pivot_point)
        or (
            bearish_reaction
            and (
                pivot_or_round_resistance_near
                or _level_is_near(current_price, pivot_r1, threshold_pct=0.10)
                or _level_is_near(current_price, round_resistance, threshold_pct=0.10)
                or _level_is_near(current_price, major_round_resistance, threshold_pct=0.10)
            )
        )
    )
    bearish_pivot_round_conflict = bool(
        (pivot_point is not None and current_price > pivot_point)
        or (
            bullish_reaction
            and (
                pivot_or_round_support_near
                or _level_is_near(current_price, pivot_s1, threshold_pct=0.10)
                or _level_is_near(current_price, round_support, threshold_pct=0.10)
                or _level_is_near(current_price, major_round_support, threshold_pct=0.10)
            )
        )
    )
    bullish_fvg_range = bool(
        in_bullish_fvg
        or (
            bullish_fvg_distance_pct is not None
            and bullish_fvg_distance_pct <= 0.08
        )
        or range_zone_break > 0
        or (
            range_zone_active
            and range_zone_position is not None
            and range_zone_position <= 0.35
        )
        or (bullish_reaction and range_or_fvg_support_near)
    )
    bearish_fvg_range = bool(
        in_bearish_fvg
        or (
            bearish_fvg_distance_pct is not None
            and bearish_fvg_distance_pct <= 0.08
        )
        or range_zone_break < 0
        or (
            range_zone_active
            and range_zone_position is not None
            and range_zone_position >= 0.65
        )
        or (bearish_reaction and range_or_fvg_resistance_near)
    )
    no_fvg_edge = bool(
        not bullish_fvg_range
        and not bearish_fvg_range
        and not range_zone_active
        and not in_range_zone
    )

    bullish_variant = ""
    if (
        ("Bullish Breakout" in structure or sr_reaction == "Bullish Breakout Through Resistance")
        and orb_state > 0
    ):
        bullish_variant = "Bullish Breakout + ORB Bullish"
    elif (
        "Bullish Drift" in structure
        and vwap_bias in {"Mild Bullish", "Bullish"}
        and sr_reaction == "Bullish Support Rejection"
    ):
        bullish_variant = "Bullish Drift + bullish VWAP + support rejection"
    elif (
        "Bullish Pressure in Range" in structure
        and sr_reaction == "Bullish Support Rejection"
        and range_zone_break > 0
    ):
        bullish_variant = "Bullish Pressure in Range + support hold + Bull Break"
    elif (
        warning_ladder == "Active Momentum Event"
        and breakout_bias == "Bullish"
        and not resistance_cluster_dense_near
    ):
        bullish_variant = "Active Momentum Event + bullish bias + no nearby heavy resistance"

    bearish_variant = ""
    if (
        ("Bearish Breakdown" in structure or sr_reaction == "Bearish Breakdown Through Support")
        and orb_state < 0
    ):
        bearish_variant = "Bearish Breakdown + ORB Bearish"
    elif (
        "Bearish Drift" in structure
        and vwap_bias in {"Mild Bearish", "Bearish"}
        and sr_reaction == "Bearish Resistance Rejection"
    ):
        bearish_variant = "Bearish Drift + bearish VWAP + resistance rejection"
    elif (
        "Bearish Pressure in Range" in structure
        and sr_reaction == "Bearish Resistance Rejection"
        and range_zone_break < 0
    ):
        bearish_variant = "Bearish Pressure in Range + resistance hold + Bear Break"
    elif (
        warning_ladder == "Active Momentum Event"
        and breakout_bias == "Bearish"
        and not support_cluster_dense_near
    ):
        bearish_variant = "Active Momentum Event + bearish bias + no nearby strong support"

    bullish_evidence = []
    bearish_evidence = []
    if bullish_structure or bullish_reaction:
        bullish_evidence.append(structure if bullish_structure else sr_reaction)
    if bearish_structure or bearish_reaction:
        bearish_evidence.append(structure if bearish_structure else sr_reaction)
    if breakout_bias == "Bullish":
        bullish_evidence.append("breakout bias is bullish")
    elif breakout_bias == "Bearish":
        bearish_evidence.append("breakout bias is bearish")
    if directional_warning:
        if breakout_bias == "Bullish":
            bullish_evidence.append(warning_ladder)
        elif breakout_bias == "Bearish":
            bearish_evidence.append(warning_ladder)
    if bullish_event:
        bullish_evidence.append(f"event regime is {_humanize_words(event_regime)}")
    if bearish_event:
        bearish_evidence.append(f"event regime is {_humanize_words(event_regime)}")
    if micro_bull_count >= 1:
        bullish_evidence.append(_summarize_microstructure_context(ta_data) or "microstructure is bullish")
    if micro_bear_count >= 1:
        bearish_evidence.append(_summarize_microstructure_context(ta_data) or "microstructure is bearish")
    if support_pressure:
        bullish_evidence.append("support cluster is stronger than resistance")
    if resistance_pressure:
        bearish_evidence.append("resistance cluster is stronger than support")
    if bullish_pivot_round:
        bullish_evidence.append("pivot and round levels support the long side")
    if bearish_pivot_round:
        bearish_evidence.append("pivot and round levels support the short side")
    if bullish_fvg_range:
        bullish_evidence.append("bullish FVG or range-zone edge is active")
    if bearish_fvg_range:
        bearish_evidence.append("bearish FVG or range-zone edge is active")

    bullish_major_conflicts = []
    bearish_major_conflicts = []
    bullish_soft_conflicts = []
    bearish_soft_conflicts = []

    if breakout_bias == "Bearish":
        bullish_major_conflicts.append("breakout bias is bearish")
    elif breakout_bias == "Bullish":
        bearish_major_conflicts.append("breakout bias is bullish")

    if hard_risk_regime:
        risk_text = _humanize_words(event_regime)
        bullish_major_conflicts.append(f"{risk_text} is active")
        bearish_major_conflicts.append(f"{risk_text} is active")

    if bearish_structure and not bullish_structure:
        bullish_major_conflicts.append(f"price action is {structure.lower()}")
    if bullish_structure and not bearish_structure:
        bearish_major_conflicts.append(f"price action is {structure.lower()}")

    if bearish_reaction or micro_bear_count >= 2:
        bullish_major_conflicts.append(
            sr_reaction.lower() if bearish_reaction else "microstructure has turned bearish"
        )
    if bullish_reaction or micro_bull_count >= 2:
        bearish_major_conflicts.append(
            sr_reaction.lower() if bullish_reaction else "microstructure has turned bullish"
        )

    if resistance_cluster_dense_near or (
        resistance_pressure
        and resistance_distance_pct is not None
        and resistance_distance_pct <= 0.10
    ):
        bullish_major_conflicts.append("dense nearby resistance is overhead")
    if support_cluster_dense_near or (
        support_pressure
        and support_distance_pct is not None
        and support_distance_pct <= 0.10
    ):
        bearish_major_conflicts.append("dense nearby support is underneath")

    if bullish_pivot_round_conflict:
        bullish_major_conflicts.append("price is below or rejecting pivot/round resistance")
    if bearish_pivot_round_conflict:
        bearish_major_conflicts.append("price is above or reclaiming pivot/round support")

    if bearish_fvg_range:
        bullish_major_conflicts.append("bearish FVG or range resistance is active")
    if bullish_fvg_range:
        bearish_major_conflicts.append("bullish FVG or range support is active")

    if cross_asset_bias == "Bearish":
        bullish_soft_conflicts.append("cross-asset context is bearish")
    elif cross_asset_bias == "Bullish":
        bearish_soft_conflicts.append("cross-asset context is bullish")

    if both_sides_tight:
        bullish_soft_conflicts.append("support and resistance are both very close")
        bearish_soft_conflicts.append("support and resistance are both very close")

    if consolidating:
        bullish_soft_conflicts.append("price action is consolidating")
        bearish_soft_conflicts.append("price action is consolidating")

    bullish_major_conflicts = _dedupe_preserve_order(bullish_major_conflicts)
    bearish_major_conflicts = _dedupe_preserve_order(bearish_major_conflicts)
    bullish_soft_conflicts = _dedupe_preserve_order(bullish_soft_conflicts)
    bearish_soft_conflicts = _dedupe_preserve_order(bearish_soft_conflicts)
    bullish_evidence = _dedupe_preserve_order(bullish_evidence)
    bearish_evidence = _dedupe_preserve_order(bearish_evidence)

    buy_watch = buy_level in {"Watch", "Strong"}
    sell_watch = sell_level in {"Watch", "Strong"}
    active_long = action_state == "LONG_ACTIVE"
    active_short = action_state == "SHORT_ACTIVE"
    in_position = active_long or active_short

    long_exit_reasons = []
    short_exit_reasons = []
    if event_regime == "panic_reversal":
        long_exit_reasons.append("panic reversal is active")
        short_exit_reasons.append("panic reversal is active")
    if event_regime == "event_risk":
        long_exit_reasons.append("event risk is active")
        short_exit_reasons.append("event risk is active")
    if active_long and sr_reaction == "Bearish Resistance Rejection":
        long_exit_reasons.append("long position faces bearish resistance rejection")
    if active_short and sr_reaction == "Bullish Support Rejection":
        short_exit_reasons.append("short position faces bullish support rejection")
    if active_long and resistance_cluster_dense_near:
        long_exit_reasons.append("price has reached a dense nearby resistance cluster")
    if active_short and support_cluster_dense_near:
        short_exit_reasons.append("price has reached a dense nearby support cluster")
    if active_long and any(
        _level_is_near(current_price, level, threshold_pct=0.06)
        for level in (pivot_r1, pivot_r2, major_round_resistance)
    ):
        long_exit_reasons.append("price is pressing into R1/R2 or major round resistance")
    if active_short and any(
        _level_is_near(current_price, level, threshold_pct=0.06)
        for level in (pivot_s1, pivot_s2, major_round_support)
    ):
        short_exit_reasons.append("price is pressing into S1/S2 or major round support")
    if active_long and warning_ladder in {"Normal", "Expansion Watch"} and micro_bear_count >= 1:
        long_exit_reasons.append("follow-through has faded while risk conditions cooled")
    if active_short and warning_ladder in {"Normal", "Expansion Watch"} and micro_bull_count >= 1:
        short_exit_reasons.append("follow-through has faded while risk conditions cooled")
    if active_long and consolidating and resistance_cluster_dense_near:
        long_exit_reasons.append("price has gone back to consolidation near resistance")
    if active_short and consolidating and support_cluster_dense_near:
        short_exit_reasons.append("price has gone back to consolidation near support")
    if active_long and in_bearish_fvg and bearish_reaction:
        long_exit_reasons.append("opposing bearish FVG rejection is active")
    if active_short and in_bullish_fvg and bullish_reaction:
        short_exit_reasons.append("opposing bullish FVG rejection is active")

    long_exit_reasons = _dedupe_preserve_order(long_exit_reasons)
    short_exit_reasons = _dedupe_preserve_order(short_exit_reasons)

    if active_long and (long_exit_reasons or bullish_major_conflicts):
        reasons = long_exit_reasons or bullish_major_conflicts
        exit_variant = reasons[0]
        return {
            "label": "Exit",
            "tone": _dashboard_action_tone("Exit"),
            "reason": f"Exit / reduce long: {exit_variant}.",
            "variant": exit_variant,
            "bullish_evidence": bullish_evidence,
            "bearish_evidence": bearish_evidence,
            "bullish_major_conflicts": bullish_major_conflicts,
            "bullish_soft_conflicts": bullish_soft_conflicts,
        }
    if active_short and (short_exit_reasons or bearish_major_conflicts):
        reasons = short_exit_reasons or bearish_major_conflicts
        exit_variant = reasons[0]
        return {
            "label": "Exit",
            "tone": _dashboard_action_tone("Exit"),
            "reason": f"Exit / reduce short: {exit_variant}.",
            "variant": exit_variant,
            "bullish_evidence": bullish_evidence,
            "bearish_evidence": bearish_evidence,
            "bearish_major_conflicts": bearish_major_conflicts,
            "bearish_soft_conflicts": bearish_soft_conflicts,
        }

    if permission_status == "exit_recommended" or action_state == "EXIT_RISK" or execution_status == "exit":
        return {
            "label": "Exit",
            "tone": _dashboard_action_tone("Exit"),
            "reason": summary or "Exit risk is elevated against the current directional state.",
            "variant": "Execution permission marked exit risk",
            "bullish_evidence": bullish_evidence,
            "bearish_evidence": bearish_evidence,
        }

    if hard_risk_regime and not in_position:
        risk_label = _humanize_words(event_regime)
        return {
            "label": "Stand Aside",
            "tone": _dashboard_action_tone("Stand Aside"),
            "reason": f"{risk_label} is active, so no fresh entry is justified.",
            "variant": risk_label,
            "bullish_evidence": bullish_evidence,
            "bearish_evidence": bearish_evidence,
        }

    if (
        big_move_risk >= 56.0
        and breakout_bias == "Neutral"
        and direction_bias == "Neutral"
        and micro_bull_count == 0
        and micro_bear_count == 0
    ):
        return {
            "label": "Stand Aside",
            "tone": _dashboard_action_tone("Stand Aside"),
            "reason": "High big-move risk is present, but direction is still undefined.",
            "variant": "High Big Move Risk but no direction",
            "bullish_evidence": bullish_evidence,
            "bearish_evidence": bearish_evidence,
        }

    if (
        consolidating
        and breakout_bias == "Neutral"
        and warning_ladder == "Normal"
        and micro_bull_count == 0
        and micro_bear_count == 0
    ):
        return {
            "label": "Stand Aside",
            "tone": _dashboard_action_tone("Stand Aside"),
            "reason": "Price is consolidating with neutral bias, normal warning ladder, and no microstructure edge.",
            "variant": "Consolidating + neutral stack",
            "bullish_evidence": bullish_evidence,
            "bearish_evidence": bearish_evidence,
        }

    bullish_score = len(bullish_evidence)
    bearish_score = len(bearish_evidence)
    bullish_candidate = bool(
        buy_watch
        and bullish_score >= 5
        and bullish_score >= bearish_score + 2
    )
    bearish_candidate = bool(
        sell_watch
        and bearish_score >= 5
        and bearish_score >= bullish_score + 2
    )

    if bullish_candidate and not bearish_candidate:
        if bullish_major_conflicts:
            conflict_text = _join_readable_list(bullish_major_conflicts[:2])
            return {
                "label": "Stand Aside",
                "tone": _dashboard_action_tone("Stand Aside"),
                "reason": f"Bullish setup is present, but {len(bullish_major_conflicts)} major conflict blocks entry: {conflict_text}.",
                "variant": bullish_variant or "Bullish setup blocked",
                "bullish_evidence": bullish_evidence,
                "bullish_major_conflicts": bullish_major_conflicts,
                "bullish_soft_conflicts": bullish_soft_conflicts,
            }
        if bullish_soft_conflicts:
            soft_text = _join_readable_list(bullish_soft_conflicts[:2])
            return {
                "label": "Stand Aside",
                "tone": _dashboard_action_tone("Stand Aside"),
                "reason": f"Bullish setup is forming, but it stays prepare-only because {soft_text}.",
                "variant": bullish_variant or "Bullish setup forming",
                "bullish_evidence": bullish_evidence,
                "bullish_major_conflicts": bullish_major_conflicts,
                "bullish_soft_conflicts": bullish_soft_conflicts,
            }
        if permission_status != "entry_allowed":
            return {
                "label": "Stand Aside",
                "tone": _dashboard_action_tone("Stand Aside"),
                "reason": summary or "Bullish stack is aligned, but execution is not fully cleared yet.",
                "variant": bullish_variant or "Bullish setup forming",
                "bullish_evidence": bullish_evidence,
            }
        variant_text = bullish_variant or "Bullish stack aligned"
        basis = _join_readable_list(bullish_evidence[:4])
        return {
            "label": "Buy",
            "tone": _dashboard_action_tone("Buy"),
            "reason": f"Buy setup confirmed: {variant_text}. {basis}.",
            "variant": variant_text,
            "bullish_evidence": bullish_evidence,
            "bullish_major_conflicts": bullish_major_conflicts,
            "bullish_soft_conflicts": bullish_soft_conflicts,
        }

    if bearish_candidate and not bullish_candidate:
        if bearish_major_conflicts:
            conflict_text = _join_readable_list(bearish_major_conflicts[:2])
            return {
                "label": "Stand Aside",
                "tone": _dashboard_action_tone("Stand Aside"),
                "reason": f"Bearish setup is present, but {len(bearish_major_conflicts)} major conflict blocks entry: {conflict_text}.",
                "variant": bearish_variant or "Bearish setup blocked",
                "bearish_evidence": bearish_evidence,
                "bearish_major_conflicts": bearish_major_conflicts,
                "bearish_soft_conflicts": bearish_soft_conflicts,
            }
        if bearish_soft_conflicts:
            soft_text = _join_readable_list(bearish_soft_conflicts[:2])
            return {
                "label": "Stand Aside",
                "tone": _dashboard_action_tone("Stand Aside"),
                "reason": f"Bearish setup is forming, but it stays prepare-only because {soft_text}.",
                "variant": bearish_variant or "Bearish setup forming",
                "bearish_evidence": bearish_evidence,
                "bearish_major_conflicts": bearish_major_conflicts,
                "bearish_soft_conflicts": bearish_soft_conflicts,
            }
        if permission_status != "entry_allowed":
            return {
                "label": "Stand Aside",
                "tone": _dashboard_action_tone("Stand Aside"),
                "reason": summary or "Bearish stack is aligned, but execution is not fully cleared yet.",
                "variant": bearish_variant or "Bearish setup forming",
                "bearish_evidence": bearish_evidence,
            }
        variant_text = bearish_variant or "Bearish stack aligned"
        basis = _join_readable_list(bearish_evidence[:4])
        return {
            "label": "Sell",
            "tone": _dashboard_action_tone("Sell"),
            "reason": f"Sell setup confirmed: {variant_text}. {basis}.",
            "variant": variant_text,
            "bearish_evidence": bearish_evidence,
            "bearish_major_conflicts": bearish_major_conflicts,
            "bearish_soft_conflicts": bearish_soft_conflicts,
        }

    if buy_watch and bullish_score >= 4 and bullish_score > bearish_score:
        blockers = _join_readable_list((bullish_major_conflicts + bullish_soft_conflicts)[:2])
        return {
            "label": "Stand Aside",
            "tone": _dashboard_action_tone("Stand Aside"),
            "reason": (
                f"Bullish setup is building, but entry is not clean enough yet because {blockers}."
                if blockers
                else summary or "Bullish setup is forming, but execution is not cleared yet."
            ),
            "variant": bullish_variant or "Bullish setup forming",
            "bullish_evidence": bullish_evidence,
            "bullish_major_conflicts": bullish_major_conflicts,
            "bullish_soft_conflicts": bullish_soft_conflicts,
        }
    if sell_watch and bearish_score >= 4 and bearish_score > bullish_score:
        blockers = _join_readable_list((bearish_major_conflicts + bearish_soft_conflicts)[:2])
        return {
            "label": "Stand Aside",
            "tone": _dashboard_action_tone("Stand Aside"),
            "reason": (
                f"Bearish setup is building, but entry is not clean enough yet because {blockers}."
                if blockers
                else summary or "Bearish setup is forming, but execution is not cleared yet."
            ),
            "variant": bearish_variant or "Bearish setup forming",
            "bearish_evidence": bearish_evidence,
            "bearish_major_conflicts": bearish_major_conflicts,
            "bearish_soft_conflicts": bearish_soft_conflicts,
        }

    if both_sides_tight or no_fvg_edge or (bullish_score and bearish_score):
        mixed_parts = []
        if bullish_evidence:
            mixed_parts.append(f"bullish: {_join_readable_list(bullish_evidence[:2])}")
        if bearish_evidence:
            mixed_parts.append(f"bearish: {_join_readable_list(bearish_evidence[:2])}")
        return {
            "label": "Stand Aside",
            "tone": _dashboard_action_tone("Stand Aside"),
            "reason": f"Signals are mixed, so no fresh entry is justified. {_join_readable_list(mixed_parts)}.",
            "variant": "Mixed stack",
            "bullish_evidence": bullish_evidence,
            "bearish_evidence": bearish_evidence,
        }

    return {
        "label": "Stand Aside",
        "tone": _dashboard_action_tone("Stand Aside"),
        "reason": summary or "The directional stack is too weak or mixed for a clean trade.",
        "variant": "No clean edge",
        "bullish_evidence": bullish_evidence,
        "bearish_evidence": bearish_evidence,
        "bullish_major_conflicts": bullish_major_conflicts,
        "bearish_major_conflicts": bearish_major_conflicts,
    }


def _is_rr_signal_actionable(rr_signal):
    rr_signal = rr_signal if isinstance(rr_signal, dict) else {}
    status = str(rr_signal.get("status") or "").strip()
    direction = str(rr_signal.get("direction") or "").strip()
    grade = str(rr_signal.get("grade") or "").strip()
    return (
        status in {"arming", "ready"}
        and direction in {"Bullish", "Bearish"}
        and grade in {"A+ (Quant)", "A (High Accuracy)", "B (Qualified)"}
    )


def _build_signal_notification(changes, rr_signal, market_structure, ta_data=None, payload=None):
    changed_keys = set((changes or {}).keys())
    title = _notification_title_for_changes(changes)
    body_lines = []
    dashboard_action = _derive_dashboard_action(payload, ta_data=ta_data)
    dashboard_label = str(dashboard_action.get("label") or "").strip()
    dashboard_reason = str(dashboard_action.get("reason") or "").strip()
    if dashboard_label:
        body_lines.append(f"Dashboard Label: {dashboard_label}")
    if dashboard_reason:
        body_lines.append(f"Label Basis: {dashboard_reason}")
    if "market_structure" in changed_keys:
        body_lines.append(f"Market Structure: {_format_market_structure_change(changes, market_structure)}")

    rr_direction_grade = _format_rr_direction_grade(rr_signal)
    rr_target_probability = _format_rr_target_bucket_probability(rr_signal)
    if rr_direction_grade != "Neutral · N/A" or rr_target_probability != "N/A · N/A":
        body_lines.append(f"Direction / Grade: {rr_direction_grade}")
        body_lines.append(f"Target Bucket Probability: {rr_target_probability}")
    micro_change = _format_microstructure_change(changes)
    if micro_change:
        body_lines.append(f"Microstructure Change: {micro_change}")
    body_lines.append(f"Bar Session / Microstructure: {_format_bar_session_microstructure(ta_data)}")

    if not body_lines:
        body_lines.append(f"Market Structure: {_format_market_structure_change(changes, market_structure)}")

    body = "\n".join(body_lines)
    return {
        "title": title,
        "body": body,
        "dashboard_action": dashboard_action,
    }


def _humanize_words(value):
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ")
    acronyms = {"AI", "RSI", "EMA", "ATR", "ADX", "XAUUSD"}
    words = []
    for word in text.split():
        upper = word.upper()
        if upper in acronyms:
            words.append(upper)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _humanize_alert_value(key, value):
    text = str(value or "").strip()
    if not text:
        return "None"
    humanized_keys = {
        "trade_playbook_stage",
        "event_regime",
        "breakout_bias",
        "rr_signal_status",
        "rr_signal_grade",
        "rr_signal_direction",
        "entry_readiness",
        "exit_urgency",
        "execution_permission",
        "market_regime",
    }
    if key in humanized_keys:
        return _humanize_words(text)
    return text


def _normalize_notification_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _confidence_bucket(confidence):
    try:
        value = int(confidence or 0)
    except Exception:
        value = 0
    if value >= 85:
        return "Very High"
    if value >= 70:
        return "High"
    if value >= 60:
        return "Guarded"
    return "Low"


def _big_move_risk_bucket(score):
    try:
        value = float(score or 0)
    except Exception:
        value = 0.0
    if value >= 70:
        return "Extreme"
    if value >= 56:
        return "Elevated"
    if value >= 40:
        return "Watch"
    return "Low"


def _warning_ladder_rank(value):
    ranks = {
        "Normal": 0,
        "Expansion Watch": 1,
        "High Breakout Risk": 2,
        "Directional Expansion Likely": 3,
        "Active Momentum Event": 4,
    }
    return ranks.get(str(value or "Normal"), 0)


def _entry_readiness_rank(value):
    ranks = {
        "blocked": 0,
        "closed": 0,
        "low": 1,
        "guarded": 2,
        "medium": 3,
        "high": 4,
    }
    return ranks.get(str(value or "low"), 1)


def _should_append_trade_summary(permission_text, trade_summary):
    permission_text = str(permission_text or "").strip()
    trade_summary = str(trade_summary or "").strip()
    if not trade_summary:
        return False
    if not permission_text:
        return True

    permission_norm = _normalize_notification_text(permission_text)
    trade_norm = _normalize_notification_text(trade_summary)
    if not trade_norm or trade_norm == permission_norm:
        return False

    if trade_norm in permission_norm or permission_norm in trade_norm:
        return False

    tracked_phrases = (
        "bullish",
        "bearish",
        "buy",
        "sell",
        "watchlist",
        "no trade",
        "exit",
        "tradeability is low",
        "low tradeability",
        "regime is unstable",
        "unstable regime",
        "range regime",
        "event risk",
        "transition regime",
        "no clean trigger",
        "conditions are not clean enough",
        "execution is blocked",
    )
    permission_markers = {phrase for phrase in tracked_phrases if phrase in permission_text.lower()}
    trade_markers = {phrase for phrase in tracked_phrases if phrase in trade_summary.lower()}
    if trade_markers and trade_markers.issubset(permission_markers):
        return False

    return True


def _telegram_enabled():
    return bool(ENABLE_TELEGRAM_NOTIFICATIONS and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _has_background_alert_channels():
    return _has_push_subscribers() or _telegram_enabled()


def _send_telegram_notification(changes, rr_signal, market_structure, ta_data=None, payload=None):
    if not _telegram_enabled():
        return

    notification = _build_signal_notification(
        changes,
        rr_signal,
        market_structure,
        ta_data=ta_data,
        payload=payload,
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "\n".join(
            line for line in [notification.get("title"), notification.get("body")] if line
        ),
        "disable_web_page_preview": True,
    }
    if TELEGRAM_THREAD_ID:
        payload["message_thread_id"] = TELEGRAM_THREAD_ID

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        # Keep the monitor loop alive even if Telegram is unavailable.
        return


def _send_web_push_notifications(changes, rr_signal, market_structure, ta_data=None, payload=None):
    if webpush is None or not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
        return

    subscriptions = _load_subscriptions()
    if not subscriptions:
        return

    notification = _build_signal_notification(
        changes,
        rr_signal,
        market_structure,
        ta_data=ta_data,
        payload=payload,
    )
    push_payload = {
        "title": notification.get("title"),
        "body": notification.get("body"),
        "dashboard_action": notification.get("dashboard_action"),
        "rr_signal": rr_signal,
        "market_structure": market_structure,
        "url": "/",
        "ts": int(time.time()),
    }

    dead_endpoints = []
    for sub in subscriptions:
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps(push_payload, ensure_ascii=True),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_CLAIMS_SUBJECT},
            )
        except WebPushException as ex:
            status = getattr(ex, "response", None)
            code = getattr(status, "status_code", None)
            if code in (404, 410):
                dead_endpoints.append(sub.get("endpoint"))
        except Exception:
            continue

    for endpoint in dead_endpoints:
        _remove_subscription_endpoint(endpoint)


def _is_material_change(changes):
    """Filter out minor numeric noise so notifications stay actionable."""
    if not isinstance(changes, dict) or not changes:
        return False

    always_material = {
        "trade_playbook_stage",
        "verdict",
        "confidence_bucket",
        "warning_ladder",
        "event_regime",
        "breakout_bias",
        "rr_signal_status",
        "rr_signal_grade",
        "rr_signal_direction",
        "entry_readiness",
        "exit_urgency",
        "trend",
        "market_structure",
        "market_regime",
        "trade_sell_setup",
        "trade_buy_setup",
        "trade_exit_warning",
        "micro_vwap_bias",
        "micro_orb_state",
        "micro_sweep_state",
    }
    numeric_thresholds = {
        "rsi_14": 0.6,
        "ema_20": 0.25,
        "ema_50": 0.25,
        "adx_14": 0.4,
        "atr_percent": 0.02,
        "micro_vwap_delta_pct": 0.10,
    }

    for key, delta in changes.items():
        if key in always_material:
            return True

        if key in numeric_thresholds:
            prev = delta.get("previous") if isinstance(delta, dict) else None
            cur = delta.get("current") if isinstance(delta, dict) else None
            if isinstance(prev, (int, float)) and isinstance(cur, (int, float)):
                if abs(cur - prev) >= numeric_thresholds[key]:
                    return True
            elif prev != cur:
                return True
            continue

        if isinstance(delta, dict) and delta.get("previous") != delta.get("current"):
            return True

    return False


def _warning_alert_tier(value):
    value = str(value or "")
    if value in {"Directional Expansion Likely", "Active Momentum Event"}:
        return "high"
    if value == "High Breakout Risk":
        return "medium"
    return "low"


def _playbook_stage_class(value):
    value = str(value or "")
    if value in {"enter", "hold"}:
        return "actionable"
    if value in {"prepare", "stalk_entry"}:
        return "forming"
    if value == "exit":
        return "exit"
    return "standby"


def _is_major_playbook_transition(previous_stage, current_stage):
    prev_class = _playbook_stage_class(previous_stage)
    cur_class = _playbook_stage_class(current_stage)
    if prev_class != cur_class:
        return True
    return False


def _warning_tier_rank(value):
    tier = _warning_alert_tier(value)
    return {"low": 0, "medium": 1, "high": 2}.get(tier, 0)


def _is_context_flip_noise(previous_warning, current_warning, previous_event_regime, current_event_regime):
    prev_warning = str(previous_warning or "")
    cur_warning = str(current_warning or "")
    prev_event = str(previous_event_regime or "")
    cur_event = str(current_event_regime or "")
    return (
        {prev_warning, cur_warning}.issubset({"Expansion Watch", "High Breakout Risk", ""})
        and {prev_event, cur_event}.issubset({"normal", "breakout_watch", ""})
    )


def _is_warning_boundary_wobble(
    previous_warning,
    current_warning,
    previous_event_regime,
    current_event_regime,
    trade_playbook_stage,
    execution_status,
):
    if execution_status not in {"no_trade", "watchlist_only"}:
        return False
    if trade_playbook_stage not in {"prepare", "stalk_entry", "stand_aside"}:
        return False
    return _is_context_flip_noise(
        previous_warning,
        current_warning,
        previous_event_regime,
        current_event_regime,
    )


def _is_forming_setup_wobble(changes, trade_playbook, execution_permission, decision_status):
    if not isinstance(changes, dict) or not changes:
        return False

    execution_status = str((execution_permission or {}).get("status") or "no_trade")
    if execution_status not in {"no_trade", "watchlist_only"}:
        return False

    changed_keys = set(changes.keys())
    allowed_keys = {
        "trade_playbook_stage",
        "warning_ladder",
        "event_regime",
        "breakout_bias",
        "entry_readiness",
    }
    if not changed_keys.issubset(allowed_keys):
        return False

    stage_prev = str(((changes.get("trade_playbook_stage") or {}).get("previous")) or "")
    stage_cur = str(((changes.get("trade_playbook_stage") or {}).get("current")) or "")
    warning_prev = str(((changes.get("warning_ladder") or {}).get("previous")) or "")
    warning_cur = str(((changes.get("warning_ladder") or {}).get("current")) or "")
    event_prev = str(((changes.get("event_regime") or {}).get("previous")) or "")
    event_cur = str(((changes.get("event_regime") or {}).get("current")) or "")
    readiness_prev = str(((changes.get("entry_readiness") or {}).get("previous")) or "")
    readiness_cur = str(((changes.get("entry_readiness") or {}).get("current")) or "")
    bias_cur = str((trade_playbook or {}).get("breakoutBias") or "")

    stage_pair = {stage_prev, stage_cur}
    warning_pair = {warning_prev, warning_cur}
    event_pair = {event_prev, event_cur}
    readiness_pair = {readiness_prev, readiness_cur}

    return (
        stage_pair.issubset({"prepare", "stalk_entry", ""})
        and warning_pair.issubset({"Expansion Watch", "High Breakout Risk", ""})
        and event_pair.issubset({"normal", "breakout_watch", ""})
        and readiness_pair.issubset({"guarded", "medium", ""})
        and bias_cur in {"", "Neutral", "Bullish", "Bearish"}
    )


def _evaluate_decision_status(verdict, confidence, ta_data, trade_guidance):
    market_state = (ta_data or {}).get("_prediction_market_state") or {}
    action_state = str(market_state.get("action_state") or "")
    trend = str((ta_data or {}).get("ema_trend") or "Neutral")
    price_action = (ta_data or {}).get("price_action") or {}
    mtf = (ta_data or {}).get("multi_timeframe") or {}
    structure = str(price_action.get("structure") or "")
    alignment = str(mtf.get("alignment_label") or "Mixed / Low Alignment")
    summary = str((trade_guidance or {}).get("summary") or "")
    buy_level = str((trade_guidance or {}).get("buyLevel") or "Weak")
    sell_level = str((trade_guidance or {}).get("sellLevel") or "Weak")
    exit_level = str((trade_guidance or {}).get("exitLevel") or "Low")
    has_bullish_structure = bool(re.search(r"bullish|breakout|continuation", structure, re.IGNORECASE))
    has_bearish_structure = bool(re.search(r"bearish|breakdown|rejection", structure, re.IGNORECASE))
    has_bullish_trigger = bool(re.search(r"bullish|breakout|continuation|support rejection", f"{structure} {summary}", re.IGNORECASE))
    has_bearish_trigger = bool(re.search(r"bearish|breakdown|rejection|resistance rejection", f"{structure} {summary}", re.IGNORECASE))
    has_indecision_signal = bool(re.search(r"doji|indecision|spinning top|no clean trigger", summary, re.IGNORECASE))
    no_clean_trigger = "no clean trigger" in summary.lower()
    bullish_alignment = "bullish" in alignment.lower() and "mixed" not in alignment.lower()
    bearish_alignment = "bearish" in alignment.lower() and "mixed" not in alignment.lower()
    structure_conflict = (
        (trend == "Bullish" and "bearish" in structure.lower())
        or (trend == "Bearish" and "bullish" in structure.lower())
    )
    if action_state:
        text = "Stand aside. Checklist does not support a clean trade yet."
        status = "wait"
        tradeability = str(market_state.get("tradeability") or "Low")
        regime = str(market_state.get("regime") or "unstable")
        blocked_reasons = []
        if tradeability.lower() == "low":
            blocked_reasons.append("tradeability is still low")
        if regime.lower() in {"unstable", "range", "event-risk"}:
            blocked_reasons.append(f"the market regime is {regime.lower()}")
        blocked_reason = " and ".join(blocked_reasons) if blocked_reasons else None
        if action_state == "SETUP_LONG":
            text = "Long setup forming. Bullish bias is developing, but trigger confirmation is still pending."
            status = "wait"
        elif action_state == "LONG_ACTIVE":
            text = "Safer to look for a buy. Long state is confirmed with acceptable tradeability."
            status = "buy"
        elif action_state == "SETUP_SHORT":
            text = "Short setup forming. Bearish bias is developing, but trigger confirmation is still pending."
            status = "wait"
        elif action_state == "SHORT_ACTIVE":
            text = "Safer to look for a sell. Short state is confirmed with acceptable tradeability."
            status = "sell"
        elif action_state == "EXIT_RISK":
            text = "Safer to exit or stand aside. Exit risk is confirmed against the active directional state."
            status = "exit"

        buy_checks = [
            verdict == "Bullish" and confidence >= 70,
            trend == "Bullish" and has_bullish_structure,
            bullish_alignment,
            not has_indecision_signal and not no_clean_trigger and has_bullish_trigger,
            buy_level in {"Watch", "Strong"} and exit_level in {"Low", "Medium"},
        ]
        sell_checks = [
            verdict == "Bearish" and confidence >= 70,
            trend == "Bearish" and has_bearish_structure,
            bearish_alignment,
            not has_indecision_signal and not no_clean_trigger and has_bearish_trigger,
            sell_level in {"Watch", "Strong"} and exit_level in {"Low", "Medium"},
        ]
        exit_checks = [
            verdict == "Neutral" or "mixed" in alignment.lower() or no_clean_trigger,
            exit_level == "High" or confidence < 60,
            structure_conflict,
            has_indecision_signal,
            buy_level == "Weak" and sell_level == "Weak",
        ]
        buy_passed = sum(1 for item in buy_checks if item)
        sell_passed = sum(1 for item in sell_checks if item)

        if action_state == "WAIT":
            if sell_passed == len(sell_checks) and sell_passed > buy_passed:
                reason_suffix = f" because {blocked_reason}" if blocked_reason else ""
                text = (
                    "Watchlist Only: sell checklist is confirmed, but execution is blocked"
                    f"{reason_suffix}."
                )
            elif buy_passed == len(buy_checks) and buy_passed > sell_passed:
                reason_suffix = f" because {blocked_reason}" if blocked_reason else ""
                text = (
                    "Watchlist Only: buy checklist is confirmed, but execution is blocked"
                    f"{reason_suffix}."
                )
            elif sell_passed >= 3 and sell_passed > buy_passed:
                reason_suffix = f" because {blocked_reason}" if blocked_reason else ""
                text = (
                    "Watchlist Only: sell conditions are mostly aligned, but execution is blocked"
                    f"{reason_suffix}."
                )
            elif buy_passed >= 3 and buy_passed > sell_passed:
                reason_suffix = f" because {blocked_reason}" if blocked_reason else ""
                text = (
                    "Watchlist Only: buy conditions are mostly aligned, but execution is blocked"
                    f"{reason_suffix}."
                )
        return {
            "text": text,
            "status": status,
            "buyChecks": buy_checks,
            "sellChecks": sell_checks,
            "exitChecks": exit_checks,
        }

    buy_checks = [
        verdict == "Bullish" and confidence >= 70,
        trend == "Bullish" and has_bullish_structure,
        bullish_alignment,
        not has_indecision_signal and not no_clean_trigger and has_bullish_trigger,
        buy_level in {"Watch", "Strong"} and exit_level in {"Low", "Medium"},
    ]
    sell_checks = [
        verdict == "Bearish" and confidence >= 70,
        trend == "Bearish" and has_bearish_structure,
        bearish_alignment,
        not has_indecision_signal and not no_clean_trigger and has_bearish_trigger,
        sell_level in {"Watch", "Strong"} and exit_level in {"Low", "Medium"},
    ]
    exit_checks = [
        verdict == "Neutral" or "mixed" in alignment.lower() or no_clean_trigger,
        exit_level == "High" or confidence < 60,
        structure_conflict,
        has_indecision_signal,
        buy_level == "Weak" and sell_level == "Weak",
    ]

    buy_passed = sum(1 for item in buy_checks if item)
    sell_passed = sum(1 for item in sell_checks if item)
    exit_passed = sum(1 for item in exit_checks if item)

    text = "Stand aside. Checklist does not support a clean trade yet."
    status = "wait"
    if buy_passed >= 4 and buy_passed > sell_passed and exit_passed < 3:
        text = "Safer to look for a buy. Most buy conditions are confirmed."
        status = "buy"
    elif sell_passed >= 4 and sell_passed > buy_passed and exit_passed < 3:
        text = "Safer to look for a sell. Most sell conditions are confirmed."
        status = "sell"
    elif exit_passed >= 2:
        text = "Safer to exit or stand aside. The checklist is flagging risk or indecision."
        status = "exit"

    return {
        "text": text,
        "status": status,
        "buyChecks": buy_checks,
        "sellChecks": sell_checks,
        "exitChecks": exit_checks,
    }


def _evaluate_execution_permission(decision_status, market_state):
    action_state = str((market_state or {}).get("action_state") or "WAIT")
    action = str((market_state or {}).get("action") or "hold").strip().lower()
    decision_kind = str((decision_status or {}).get("status") or "wait")
    decision_text = str((decision_status or {}).get("text") or "")
    tradeability = str((market_state or {}).get("tradeability") or "Low")
    regime = str((market_state or {}).get("regime") or "unstable")

    text = "No trade. Conditions are not clean enough yet."
    status = "no_trade"

    directional_entry_confirmed = (
        (action_state == "LONG_ACTIVE" and decision_kind == "buy")
        or (action_state == "SHORT_ACTIVE" and decision_kind == "sell")
    )

    if action_state == "EXIT_RISK" or action == "exit" or decision_kind == "exit":
        text = "Exit Recommended: active position quality is deteriorating."
        status = "exit_recommended"
    elif directional_entry_confirmed:
        text = (
            "Entry Allowed: buy conditions are confirmed."
            if action_state == "LONG_ACTIVE"
            else "Entry Allowed: sell conditions are confirmed."
        )
        status = "entry_allowed"
    elif action_state in {"LONG_ACTIVE", "SHORT_ACTIVE"} and decision_kind in {"buy", "sell"}:
        text = "No trade. Execution state and checklist direction are not aligned yet."
        status = "no_trade"
    elif action_state in {"SETUP_LONG", "SETUP_SHORT"}:
        text = "Watchlist Only: setup is forming, but execution is not confirmed yet."
        status = "watchlist_only"
    elif action_state == "WAIT" and decision_text.startswith("Watchlist Only:"):
        blockers = []
        if tradeability.lower() == "low":
            blockers.append("low tradeability")
        if regime.lower() in {"unstable", "range", "event-risk"}:
            blockers.append(f"{regime.lower()} regime")
        blocker_text = " and ".join(blockers)
        if blocker_text:
            text = f"Watchlist Only: entry is blocked by {blocker_text}."
        else:
            text = "Watchlist Only: setup is forming, but execution is not confirmed yet."
        status = "watchlist_only"

    return {
        "text": text,
        "status": status,
        "actionState": action_state,
        "decisionStatus": decision_kind,
    }


def _evaluate_trade_playbook(decision_status, execution_permission, market_state, regime_state, trade_guidance):
    decision_status = decision_status or {}
    execution_permission = execution_permission or {}
    market_state = market_state or {}
    regime_state = regime_state or {}
    trade_guidance = trade_guidance or {}

    warning_ladder = str(regime_state.get("warning_ladder") or "Normal")
    event_regime = str(regime_state.get("event_regime") or "normal")
    breakout_bias = str(regime_state.get("breakout_bias") or "Neutral")
    directional_bias = str(market_state.get("directional_bias") or "Neutral")
    action_state = str(market_state.get("action_state") or "WAIT")
    execution_status = str(execution_permission.get("status") or "no_trade")
    decision_kind = str(decision_status.get("status") or "wait")
    buy_level = str(trade_guidance.get("buyLevel") or "Weak")
    sell_level = str(trade_guidance.get("sellLevel") or "Weak")
    exit_level = str(trade_guidance.get("exitLevel") or "Low")

    bullish_alignment = breakout_bias == "Bullish" and directional_bias == "Bullish"
    bearish_alignment = breakout_bias == "Bearish" and directional_bias == "Bearish"
    directional_alignment = bullish_alignment or bearish_alignment
    directional_conflict = (
        breakout_bias in {"Bullish", "Bearish"}
        and directional_bias in {"Bullish", "Bearish"}
        and breakout_bias != directional_bias
    )
    active_long = action_state == "LONG_ACTIVE"
    active_short = action_state == "SHORT_ACTIVE"
    active_position = active_long or active_short
    active_position_aligned = (
        (active_long and breakout_bias == "Bullish")
        or (active_short and breakout_bias == "Bearish")
    )
    reversal_risk = event_regime == "panic_reversal" or (
        active_long and breakout_bias == "Bearish"
    ) or (
        active_short and breakout_bias == "Bullish"
    )

    stage = "stand_aside"
    title = "Stand Aside"
    text = "Let the normal directional engine do the work. No large-move setup needs action yet."
    why = []
    entry_readiness = "low"
    exit_urgency = "low"

    if execution_status == "exit_recommended" or reversal_risk:
        stage = "exit"
        title = "Exit / Reduce"
        text = "Momentum or reversal risk is now working against the current position. Protect capital first."
        why = [
            f"Event regime: {event_regime.replace('_', ' ')}",
            f"Breakout bias: {breakout_bias}",
            f"Execution: {execution_permission.get('text') or 'Exit recommended'}",
        ]
        entry_readiness = "blocked"
        exit_urgency = "high"
    elif active_position and warning_ladder == "Active Momentum Event" and active_position_aligned:
        stage = "hold"
        title = "Hold Winner"
        text = "Momentum is active and still aligned with the position. Favor holding over premature exits."
        why = [
            f"Warning ladder: {warning_ladder}",
            f"Breakout bias: {breakout_bias}",
            f"Action state: {action_state.replace('_', ' ')}",
        ]
        entry_readiness = "closed"
        exit_urgency = "low"
    elif warning_ladder == "Directional Expansion Likely":
        if directional_alignment and execution_status == "entry_allowed" and decision_kind in {"buy", "sell"}:
            stage = "enter"
            title = "Enter With Confirmation"
            text = "Large-move risk and the directional engine are aligned. This is the best early-entry state."
            why = [
                f"Breakout bias and directional bias align: {breakout_bias}",
                f"Execution: {execution_permission.get('text') or 'Entry allowed'}",
                f"Checklist status: {decision_status.get('text') or decision_kind}",
            ]
            entry_readiness = "high"
            exit_urgency = "low"
        elif directional_alignment:
            stage = "stalk_entry"
            title = "Stalk Entry"
            text = "Directional expansion looks likely, but execution confirmation is still incomplete. Prepare trigger levels."
            why = [
                f"Warning ladder: {warning_ladder}",
                f"Bias alignment: {breakout_bias}",
                f"Execution: {execution_permission.get('text') or 'Watchlist only'}",
            ]
            entry_readiness = "medium"
            exit_urgency = "low"
        else:
            stage = "prepare"
            title = "Prepare For Expansion"
            text = "Expansion risk is elevated, but direction is not aligned yet. Do not force an entry."
            why = [
                f"Warning ladder: {warning_ladder}",
                f"Breakout bias: {breakout_bias}",
                f"Directional bias: {directional_bias}",
            ]
            entry_readiness = "guarded"
            exit_urgency = "low"
    elif warning_ladder == "High Breakout Risk":
        stage = "stalk_entry" if directional_alignment else "prepare"
        title = "Stalk Breakout" if directional_alignment else "Prepare For Breakout"
        text = (
            "Move risk is elevated. Stalk the aligned side, but wait for the directional engine to fully confirm."
            if directional_alignment
            else "Move risk is elevated without a clean directional stack yet. Mark levels and stay patient."
        )
        why = [
            f"Warning ladder: {warning_ladder}",
            f"Breakout bias: {breakout_bias}",
            f"Directional bias: {directional_bias}",
        ]
        entry_readiness = "medium" if directional_alignment else "guarded"
        exit_urgency = "low"
    elif warning_ladder == "Expansion Watch":
        stage = "prepare"
        title = "Prepare"
        text = "Compression and regime context suggest a move may be forming. Prepare scenarios rather than entering early."
        why = [
            f"Event regime: {event_regime.replace('_', ' ')}",
            f"Breakout bias: {breakout_bias}",
        ]
        entry_readiness = "guarded"
        exit_urgency = "low"
    elif execution_status == "entry_allowed" and decision_kind in {"buy", "sell"}:
        stage = "enter"
        title = "Enter On Normal Flow"
        text = "The directional engine is already clear enough to trade, even without a major event regime."
        why = [
            f"Execution: {execution_permission.get('text') or 'Entry allowed'}",
            f"Buy/Sell setup: {buy_level}/{sell_level}",
            f"Exit warning: {exit_level}",
        ]
        entry_readiness = "high"
        exit_urgency = "low"
    elif execution_status == "watchlist_only":
        stage = "prepare"
        title = "Watchlist Only"
        text = "The setup is forming, but execution remains blocked by tradeability, trigger quality, or regime quality."
        why = [
            f"Execution: {execution_permission.get('text') or 'Watchlist only'}",
            f"Warning ladder: {warning_ladder}",
        ]
        entry_readiness = "guarded"
        exit_urgency = "low"

    return {
        "stage": stage,
        "title": title,
        "text": text,
        "why": why,
        "entryReadiness": entry_readiness,
        "exitUrgency": exit_urgency,
        "warningLadder": warning_ladder,
        "eventRegime": event_regime,
        "breakoutBias": breakout_bias,
        "directionalBias": directional_bias,
        "alignment": "conflicted" if directional_conflict else ("aligned" if directional_alignment else "mixed"),
    }


def _stabilize_trade_playbook(trade_playbook, execution_permission, decision_status):
    raw_stage = str((trade_playbook or {}).get("stage") or "stand_aside")
    raw_title = str((trade_playbook or {}).get("title") or "Stand Aside")
    raw_text = str((trade_playbook or {}).get("text") or "")
    raw_why = list((trade_playbook or {}).get("why") or [])
    raw_entry_readiness = str((trade_playbook or {}).get("entryReadiness") or "low")
    raw_exit_urgency = str((trade_playbook or {}).get("exitUrgency") or "low")
    raw_warning_ladder = str((trade_playbook or {}).get("warningLadder") or "Normal")
    raw_event_regime = str((trade_playbook or {}).get("eventRegime") or "normal")
    raw_breakout_bias = str((trade_playbook or {}).get("breakoutBias") or "Neutral")
    raw_directional_bias = str((trade_playbook or {}).get("directionalBias") or "Neutral")
    raw_alignment = str((trade_playbook or {}).get("alignment") or "mixed")
    now_ts = int(time.time())

    state = _load_json_file(
        PLAYBOOK_STATE_FILE,
        {
            "stable_stage": raw_stage,
            "stable_title": raw_title,
            "stable_text": raw_text,
            "stable_why": raw_why,
            "stable_entry_readiness": raw_entry_readiness,
            "stable_exit_urgency": raw_exit_urgency,
            "stable_warning_ladder": raw_warning_ladder,
            "stable_event_regime": raw_event_regime,
            "stable_breakout_bias": raw_breakout_bias,
            "stable_directional_bias": raw_directional_bias,
            "stable_alignment": raw_alignment,
            "last_raw_stage": raw_stage,
            "raw_streak": 1,
            "last_stable_change_ts": now_ts,
        },
    )

    last_raw_stage = str(state.get("last_raw_stage", raw_stage))
    raw_streak = int(state.get("raw_streak", 0) or 0)
    raw_streak = raw_streak + 1 if raw_stage == last_raw_stage else 1

    stable_stage = str(state.get("stable_stage", raw_stage))
    stable_title = str(state.get("stable_title", raw_title))
    stable_text = str(state.get("stable_text", raw_text))
    stable_why = list(state.get("stable_why", raw_why) or raw_why)
    stable_entry_readiness = str(state.get("stable_entry_readiness", raw_entry_readiness))
    stable_exit_urgency = str(state.get("stable_exit_urgency", raw_exit_urgency))
    stable_warning_ladder = str(state.get("stable_warning_ladder", raw_warning_ladder))
    stable_event_regime = str(state.get("stable_event_regime", raw_event_regime))
    stable_breakout_bias = str(state.get("stable_breakout_bias", raw_breakout_bias))
    stable_directional_bias = str(state.get("stable_directional_bias", raw_directional_bias))
    stable_alignment = str(state.get("stable_alignment", raw_alignment))
    last_stable_change_ts = int(state.get("last_stable_change_ts", now_ts) or now_ts)

    execution_status = str((execution_permission or {}).get("status") or "no_trade")
    decision_confirmed = bool((decision_status or {}).get("confirmed"))
    held_for_seconds = max(0, now_ts - last_stable_change_ts)

    if raw_stage == stable_stage:
        required_confirmation = 1
    elif raw_stage == "exit":
        required_confirmation = max(PLAYBOOK_CONFIRMATION_COUNT, EXIT_PLAYBOOK_CONFIRMATION_COUNT)
    elif raw_stage == "enter":
        required_confirmation = max(PLAYBOOK_CONFIRMATION_COUNT, ENTER_PLAYBOOK_CONFIRMATION_COUNT + 1)
    elif stable_stage in {"enter", "hold"} and raw_stage in {"prepare", "stalk_entry", "stand_aside"}:
        required_confirmation = max(PLAYBOOK_CONFIRMATION_COUNT, ENTER_PLAYBOOK_CONFIRMATION_COUNT + 2)
    elif stable_stage in {"prepare", "stalk_entry"} and raw_stage in {"prepare", "stalk_entry"}:
        required_confirmation = max(PLAYBOOK_CONFIRMATION_COUNT, 2)
    else:
        required_confirmation = PLAYBOOK_CONFIRMATION_COUNT

    can_flip = raw_streak >= required_confirmation
    if (
        stable_stage in {"enter", "hold"}
        and execution_status == "no_trade"
        and raw_stage in {"prepare", "stand_aside"}
        and not decision_confirmed
    ):
        required_confirmation = 1
        can_flip = True
    if can_flip and raw_stage != stable_stage and held_for_seconds < PLAYBOOK_FLIP_MIN_HOLD_SECONDS:
        if stable_stage in {"enter", "hold"} and raw_stage in {"prepare", "stalk_entry", "stand_aside"}:
            can_flip = False
        elif stable_stage == "exit" and raw_stage in {"prepare", "stalk_entry", "stand_aside"}:
            can_flip = False

    if (
        stable_stage in {"enter", "hold"}
        and execution_status == "no_trade"
        and raw_stage in {"prepare", "stand_aside"}
        and not decision_confirmed
    ):
        can_flip = True

    if (
        stable_stage == "enter"
        and raw_stage in {"prepare", "stalk_entry"}
        and execution_status == "entry_allowed"
        and decision_confirmed
        and held_for_seconds < PLAYBOOK_FLIP_MIN_HOLD_SECONDS
    ):
        can_flip = False

    # Protect against rapid enter -> prepare churn when move-risk sits on boundary.
    if (
        stable_stage in {"enter", "hold"}
        and raw_stage in {"prepare", "stalk_entry", "stand_aside"}
        and execution_status in {"no_trade", "watchlist_only"}
        and raw_warning_ladder in {"Expansion Watch", "High Breakout Risk"}
        and held_for_seconds < max(PLAYBOOK_FLIP_MIN_HOLD_SECONDS, ENTER_STAGE_PROTECT_SECONDS)
    ):
        can_flip = False

    preserve_context = (
        not can_flip
        and stable_stage in {"enter", "hold", "stalk_entry"}
        and held_for_seconds < PLAYBOOK_FLIP_MIN_HOLD_SECONDS
    )
    if preserve_context:
        if _warning_ladder_rank(raw_warning_ladder) < _warning_ladder_rank(stable_warning_ladder):
            raw_warning_ladder = stable_warning_ladder
        if _entry_readiness_rank(raw_entry_readiness) < _entry_readiness_rank(stable_entry_readiness):
            raw_entry_readiness = stable_entry_readiness
        if raw_breakout_bias == "Neutral" and stable_breakout_bias in {"Bullish", "Bearish"}:
            raw_breakout_bias = stable_breakout_bias
        if raw_event_regime == "normal" and stable_event_regime != "normal":
            raw_event_regime = stable_event_regime

    if can_flip:
        if raw_stage != stable_stage:
            last_stable_change_ts = now_ts
        stable_stage = raw_stage
        stable_title = raw_title
        stable_text = raw_text
        stable_why = raw_why
        stable_entry_readiness = raw_entry_readiness
        stable_exit_urgency = raw_exit_urgency
        stable_warning_ladder = raw_warning_ladder
        stable_event_regime = raw_event_regime
        stable_breakout_bias = raw_breakout_bias
        stable_directional_bias = raw_directional_bias
        stable_alignment = raw_alignment

    state.update(
        {
            "stable_stage": stable_stage,
            "stable_title": stable_title,
            "stable_text": stable_text,
            "stable_why": stable_why,
            "stable_entry_readiness": stable_entry_readiness,
            "stable_exit_urgency": stable_exit_urgency,
            "stable_warning_ladder": stable_warning_ladder,
            "stable_event_regime": stable_event_regime,
            "stable_breakout_bias": stable_breakout_bias,
            "stable_directional_bias": stable_directional_bias,
            "stable_alignment": stable_alignment,
            "last_raw_stage": raw_stage,
            "raw_streak": raw_streak,
            "last_stable_change_ts": last_stable_change_ts,
            "updated_at": now_ts,
        }
    )
    _save_json_file(PLAYBOOK_STATE_FILE, state)

    stabilized = dict(trade_playbook or {})
    stabilized["rawStage"] = raw_stage
    stabilized["stage"] = stable_stage
    stabilized["title"] = stable_title
    stabilized["text"] = stable_text
    stabilized["why"] = stable_why
    stabilized["entryReadiness"] = stable_entry_readiness
    stabilized["exitUrgency"] = stable_exit_urgency
    stabilized["warningLadder"] = stable_warning_ladder
    stabilized["eventRegime"] = stable_event_regime
    stabilized["breakoutBias"] = stable_breakout_bias
    stabilized["directionalBias"] = stable_directional_bias
    stabilized["alignment"] = stable_alignment
    stabilized["confirmationCount"] = raw_streak
    stabilized["requiredConfirmation"] = required_confirmation
    stabilized["heldForSeconds"] = max(0, now_ts - last_stable_change_ts)

    if (
        stabilized["breakoutBias"] in {"Bullish", "Bearish"}
        and stabilized["directionalBias"] in {"Bullish", "Bearish"}
        and stabilized["breakoutBias"] != stabilized["directionalBias"]
    ):
        stabilized["alignment"] = "conflicted"

    if stabilized["stage"] == "stand_aside":
        stabilized["title"] = "Stand Aside"
        stabilized["text"] = str((decision_status or {}).get("text") or (execution_permission or {}).get("text") or "No trade. Conditions are not clean enough yet.")
        stabilized["why"] = [
            f"Execution: {str((execution_permission or {}).get('text') or 'No trade')}",
            f"Decision: {str((decision_status or {}).get('text') or 'Stand aside.')}",
        ]
        stabilized["entryReadiness"] = "low"
    elif stabilized["stage"] in {"prepare", "stalk_entry"} and stabilized["alignment"] == "conflicted":
        stabilized["text"] = "Signals are building, but the directional stack is conflicted. Mark levels and stay patient."
        stabilized["why"] = [
            f"Warning ladder: {stabilized['warningLadder']}",
            f"Breakout bias: {stabilized['breakoutBias']}",
            f"Directional bias: {stabilized['directionalBias']}",
        ]

    return stabilized


def _stabilize_decision_status(decision_status):
    raw_status = str((decision_status or {}).get("status") or "wait")
    raw_text = str((decision_status or {}).get("text") or "Stand aside.")
    now_ts = int(time.time())
    state = _load_json_file(
        DECISION_STATE_FILE,
        {
            "stable_status": raw_status,
            "stable_text": raw_text,
            "last_raw_status": raw_status,
            "raw_streak": 1,
            "last_stable_change_ts": now_ts,
        },
    )

    last_raw_status = str(state.get("last_raw_status", raw_status))
    raw_streak = int(state.get("raw_streak", 0) or 0)
    if raw_status == last_raw_status:
        raw_streak += 1
    else:
        raw_streak = 1

    stable_status = str(state.get("stable_status", raw_status))
    stable_text = str(state.get("stable_text", raw_text))
    last_stable_change_ts = int(state.get("last_stable_change_ts", now_ts) or now_ts)

    if raw_status == stable_status:
        required_confirmation = 1
    elif raw_status in {"buy", "sell"} and stable_status == "wait":
        required_confirmation = max(DECISION_CONFIRMATION_COUNT, ENTER_DECISION_CONFIRMATION_COUNT)
    elif raw_status in {"buy", "sell"} and stable_status == "exit":
        required_confirmation = max(DECISION_CONFIRMATION_COUNT, ENTER_DECISION_CONFIRMATION_COUNT)
    elif raw_status == "exit" and stable_status in {"buy", "sell"}:
        required_confirmation = max(DECISION_CONFIRMATION_COUNT, EXIT_DECISION_CONFIRMATION_COUNT)
    elif raw_status == "wait" and stable_status in {"buy", "sell", "exit"}:
        required_confirmation = max(DECISION_CONFIRMATION_COUNT, WAIT_DECISION_CONFIRMATION_COUNT)
    elif raw_status in {"buy", "sell"} and stable_status in {"buy", "sell"} and raw_status != stable_status:
        required_confirmation = max(DECISION_CONFIRMATION_COUNT, DIRECTIONAL_REVERSAL_CONFIRMATION_COUNT)
    else:
        required_confirmation = DECISION_CONFIRMATION_COUNT

    if raw_streak >= required_confirmation:
        can_flip = True
        if raw_status != stable_status and (now_ts - last_stable_change_ts) < DECISION_FLIP_MIN_HOLD_SECONDS:
            if stable_status in {"buy", "sell"} and raw_status == "exit":
                can_flip = False
            elif stable_status == "exit" and raw_status in {"buy", "sell"}:
                can_flip = False
            elif stable_status in {"buy", "sell"} and raw_status in {"buy", "sell"}:
                can_flip = False
            elif stable_status in {"buy", "sell", "exit"} and raw_status == "wait":
                can_flip = False

        if can_flip:
            if raw_status != stable_status or raw_text != stable_text:
                last_stable_change_ts = now_ts
            stable_status = raw_status
            stable_text = raw_text

    state.update(
        {
            "stable_status": stable_status,
            "stable_text": stable_text,
            "last_raw_status": raw_status,
            "raw_streak": raw_streak,
            "last_stable_change_ts": last_stable_change_ts,
            "updated_at": now_ts,
        }
    )
    _save_json_file(DECISION_STATE_FILE, state)

    stabilized = dict(decision_status or {})
    stabilized["rawStatus"] = raw_status
    stabilized["rawText"] = raw_text
    stabilized["status"] = stable_status
    stabilized["text"] = stable_text
    stabilized["confirmationCount"] = raw_streak
    stabilized["requiredConfirmation"] = required_confirmation
    stabilized["confirmed"] = raw_streak >= required_confirmation and raw_status == stable_status
    stabilized["heldForSeconds"] = max(0, now_ts - last_stable_change_ts)
    return stabilized


def _stabilize_prediction(prediction, ta_data):
    raw_verdict = prediction.get("verdict", "Neutral")
    raw_confidence = int(prediction.get("confidence", 50) or 50)
    hard_no_trade_reason = prediction.get("noTradeReasonHard") or ""
    soft_no_trade_reason = prediction.get("noTradeReasonSoft") or ""
    has_hard_no_trade = bool(prediction.get("hasHardNoTrade")) or bool(hard_no_trade_reason)
    event_risk_active = bool((ta_data.get("event_risk") or {}).get("active"))
    state = _load_json_file(
        PREDICTION_STATE_FILE,
        {
            "stable_verdict": "Neutral",
            "last_raw_verdict": "Neutral",
            "raw_streak": 0,
        },
    )

    last_raw = state.get("last_raw_verdict", "Neutral")
    raw_streak = int(state.get("raw_streak", 0) or 0)
    if raw_verdict == last_raw:
        raw_streak += 1
    else:
        raw_streak = 1

    stable_verdict = state.get("stable_verdict", "Neutral")
    stable_confidence = int(state.get("stable_confidence", 50) or 50)

    if has_hard_no_trade or event_risk_active:
        stable_verdict = "Neutral"
        stable_confidence = min(raw_confidence, 60)
    elif raw_verdict == "Neutral":
        if raw_streak >= PREDICTION_CONFIRMATION_COUNT:
            stable_verdict = "Neutral"
            stable_confidence = min(raw_confidence, 60)
    elif raw_streak >= PREDICTION_CONFIRMATION_COUNT:
        stable_verdict = raw_verdict
        stable_confidence = raw_confidence
        if soft_no_trade_reason:
            # Keep directional read visible while acknowledging a soft execution blocker.
            stable_confidence = max(52, min(stable_confidence, 72))
    else:
        stable_confidence = min(raw_confidence, 58)

    state.update(
        {
            "stable_verdict": stable_verdict,
            "stable_confidence": stable_confidence,
            "last_raw_verdict": raw_verdict,
            "raw_streak": raw_streak,
            "updated_at": int(time.time()),
        }
    )
    _save_json_file(PREDICTION_STATE_FILE, state)
    stabilized = dict(prediction)
    stabilized["rawVerdict"] = raw_verdict
    stabilized["verdict"] = stable_verdict
    stabilized["confidence"] = stable_confidence
    stabilized["confirmationCount"] = raw_streak
    stabilized["confirmed"] = raw_streak >= PREDICTION_CONFIRMATION_COUNT and raw_verdict == stable_verdict
    return stabilized


def _extract_indicator_snapshot(payload):
    """Build a compact snapshot used to detect indicator changes."""
    ta_data = payload.get("TechnicalAnalysis", {}) if isinstance(payload, dict) else {}
    pa = ta_data.get("price_action", {}) if isinstance(ta_data, dict) else {}
    regime_state = payload.get("RegimeState", {}) if isinstance(payload, dict) else {}
    trade_playbook = payload.get("TradePlaybook", {}) if isinstance(payload, dict) else {}
    rr_signal = payload.get("RR200Signal", {}) if isinstance(payload, dict) else {}
    structure_context = ta_data.get("structure_context", {}) if isinstance(ta_data, dict) else {}
    try:
        micro_vwap_delta = round(float(structure_context.get("distSessionVwapPct") or 0.0), 3)
    except Exception:
        micro_vwap_delta = 0.0
    try:
        micro_orb_state = int(round(float(structure_context.get("openingRangeBreak") or 0.0)))
    except Exception:
        micro_orb_state = 0
    try:
        micro_sweep_state = int(round(float(structure_context.get("sweepReclaimSignal") or 0.0)))
    except Exception:
        micro_sweep_state = 0
    return {
        "trade_playbook_stage": (trade_playbook.get("stage") if isinstance(trade_playbook, dict) else None),
        "warning_ladder": (
            trade_playbook.get("warningLadder")
            if isinstance(trade_playbook, dict) and trade_playbook.get("warningLadder")
            else (regime_state.get("warning_ladder") if isinstance(regime_state, dict) else None)
        ),
        "event_regime": (
            trade_playbook.get("eventRegime")
            if isinstance(trade_playbook, dict) and trade_playbook.get("eventRegime")
            else (regime_state.get("event_regime") if isinstance(regime_state, dict) else None)
        ),
        "breakout_bias": (
            trade_playbook.get("breakoutBias")
            if isinstance(trade_playbook, dict) and trade_playbook.get("breakoutBias")
            else (regime_state.get("breakout_bias") if isinstance(regime_state, dict) else None)
        ),
        "market_structure": (pa.get("structure") if isinstance(pa, dict) else None),
        "verdict": payload.get("verdict"),
        "confidence_bucket": _confidence_bucket(payload.get("confidence")),
        "execution_permission": ((payload.get("ExecutionPermission") or {}).get("text")),
        "entry_readiness": (trade_playbook.get("entryReadiness") if isinstance(trade_playbook, dict) else None),
        "exit_urgency": (trade_playbook.get("exitUrgency") if isinstance(trade_playbook, dict) else None),
        "rr_signal_status": (rr_signal.get("status") if isinstance(rr_signal, dict) else None),
        "rr_signal_grade": (rr_signal.get("grade") if isinstance(rr_signal, dict) else None),
        "rr_signal_direction": _effective_rr_direction(rr_signal),
        "micro_vwap_delta_pct": micro_vwap_delta,
        "micro_vwap_band": _vwap_band_label(micro_vwap_delta),
        "micro_vwap_bias": _vwap_bias_label(micro_vwap_delta),
        "micro_orb_state": micro_orb_state,
        "micro_sweep_state": micro_sweep_state,
    }


def _diff_snapshot(prev_snapshot, cur_snapshot):
    """Return only fields that changed between snapshots."""
    changed = {}
    keys = set(prev_snapshot.keys()) | set(cur_snapshot.keys())
    for key in keys:
        if prev_snapshot.get(key) != cur_snapshot.get(key):
            changed[key] = {
                "previous": prev_snapshot.get(key),
                "current": cur_snapshot.get(key),
            }
    return changed


def _record_live_signal_outcome(payload, snapshot, changes, now_ts):
    ta_data = payload.get("TechnicalAnalysis") or {}
    forecast_state = payload.get("ForecastState") or {}
    execution_state = payload.get("ExecutionState") or {}
    price = float(ta_data.get("current_price") or 0.0)
    records = _load_json_file(LIVE_SIGNAL_OUTCOMES_FILE, {"records": []})
    record = {
        "id": f"{now_ts}:{snapshot.get('trade_playbook_stage')}:{payload.get('verdict')}:{price}",
        "ts": now_ts,
        "price": round(price, 2),
        "verdict": payload.get("verdict"),
        "confidence": payload.get("confidence"),
        "regimeBucket": forecast_state.get("regimeBucket"),
        "warningLadder": forecast_state.get("warningLadder"),
        "eventRegime": forecast_state.get("eventRegime"),
        "breakoutBias": forecast_state.get("breakoutBias"),
        "forecastConfidence": forecast_state.get("forecastConfidence"),
        "executionStatus": execution_state.get("status"),
        "entryAllowed": execution_state.get("entryAllowed"),
        "antiChopActive": execution_state.get("antiChopActive"),
        "snapshot": snapshot,
        "changes": changes,
        "outcomes": {},
    }
    current_records = list(records.get("records") or [])
    if current_records and current_records[-1].get("snapshot") == snapshot:
        return
    current_records.append(record)
    current_records = current_records[-500:]
    _save_json_file(LIVE_SIGNAL_OUTCOMES_FILE, {"records": current_records})


def _update_live_signal_outcomes(current_price, now_ts):
    records = _load_json_file(LIVE_SIGNAL_OUTCOMES_FILE, {"records": []})
    current_records = list(records.get("records") or [])
    updated = False
    summary = {
        "total_records": len(current_records),
        "resolved_15m": 0,
        "resolved_30m": 0,
        "resolved_60m": 0,
        "enter_hit_rate_30m": 0.0,
        "prepare_to_enter_candidates": 0,
    }
    enter_total = 0
    enter_hits = 0
    for record in current_records:
        base_price = float(record.get("price") or 0.0)
        if base_price <= 0:
            continue
        age_seconds = max(0, now_ts - int(record.get("ts", now_ts) or now_ts))
        outcomes = record.setdefault("outcomes", {})
        for label, seconds in (("15m", 900), ("30m", 1800), ("60m", 3600)):
            if age_seconds >= seconds and label not in outcomes:
                change_pct = ((float(current_price) - base_price) / base_price) * 100.0
                outcomes[label] = {
                    "return_pct": round(change_pct, 4),
                    "resolved_at": now_ts,
                }
                updated = True
            if label in outcomes:
                summary[f"resolved_{label}"] += 1
        if record.get("executionStatus") == "enter" and "30m" in outcomes:
            enter_total += 1
            direction = str(record.get("verdict") or "Neutral")
            realized = float((outcomes.get("30m") or {}).get("return_pct") or 0.0)
            if (direction == "Bullish" and realized > 0) or (direction == "Bearish" and realized < 0):
                enter_hits += 1
        if record.get("executionStatus") == "prepare":
            summary["prepare_to_enter_candidates"] += 1
    if enter_total:
        summary["enter_hit_rate_30m"] = round(enter_hits / enter_total, 4)
    if updated:
        _save_json_file(LIVE_SIGNAL_OUTCOMES_FILE, {"records": current_records})
    _save_json_file(LIVE_SIGNAL_SUMMARY_FILE, summary)


def _utc_day(ts):
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def _default_rr200_counter(now_ts):
    return {
        "current_day": _utc_day(now_ts),
        "today": {
            "total": 0,
            "tiers": {
                "A+ (Quant)": 0,
                "A (High Accuracy)": 0,
                "B (Qualified)": 0,
            },
        },
        "history": [],
        "expected_daily": {
            "total": 0.0,
            "tiers": {
                "A+ (Quant)": 0.0,
                "A (High Accuracy)": 0.0,
                "B (Qualified)": 0.0,
            },
            "rolling_days": 0,
        },
        "target_daily": {"min": 2, "max": 5},
        "last_delivery_ts": 0,
    }


def _recompute_rr200_expected(counter):
    history = counter.get("history") if isinstance(counter, dict) else []
    if not isinstance(history, list):
        history = []
    rolling = history[-14:]
    if not rolling:
        counter["expected_daily"] = _default_rr200_counter(int(time.time())).get("expected_daily", {})
        return counter

    total_sum = 0.0
    tier_sum = {"A+ (Quant)": 0.0, "A (High Accuracy)": 0.0, "B (Qualified)": 0.0}
    for item in rolling:
        if not isinstance(item, dict):
            continue
        total_sum += float(item.get("total", 0) or 0)
        tiers = item.get("tiers") if isinstance(item.get("tiers"), dict) else {}
        for key in tier_sum:
            tier_sum[key] += float(tiers.get(key, 0) or 0)
    days = max(1, len(rolling))
    counter["expected_daily"] = {
        "total": round(total_sum / days, 2),
        "tiers": {key: round(value / days, 2) for key, value in tier_sum.items()},
        "rolling_days": days,
    }
    return counter


def _load_rr200_counter(now_ts):
    counter = _load_json_file(RR200_SIGNAL_COUNTER_FILE, _default_rr200_counter(now_ts))
    if not isinstance(counter, dict):
        counter = _default_rr200_counter(now_ts)

    current_day = _utc_day(now_ts)
    stored_day = str(counter.get("current_day") or "")
    if stored_day != current_day:
        today = counter.get("today") if isinstance(counter.get("today"), dict) else {}
        if today and int(today.get("total", 0) or 0) > 0 and stored_day:
            history = counter.get("history") if isinstance(counter.get("history"), list) else []
            history.append(
                {
                    "day": stored_day,
                    "total": int(today.get("total", 0) or 0),
                    "tiers": dict(today.get("tiers") or {}),
                }
            )
            counter["history"] = history[-60:]
        counter["current_day"] = current_day
        counter["today"] = _default_rr200_counter(now_ts)["today"]
    return _recompute_rr200_expected(counter)


def _rr200_should_deliver(counter, now_ts):
    return True


def _rr200_delivery_allowed_for_payload(payload, now_ts):
    rr = payload.get("RR200Signal") if isinstance(payload, dict) else {}
    if not isinstance(rr, dict):
        return False
    grade = str(rr.get("grade") or "")
    if grade not in {"A+ (Quant)", "A (High Accuracy)", "B (Qualified)"}:
        return False
    counter = _load_rr200_counter(now_ts)
    return _rr200_should_deliver(counter, now_ts)


def _record_rr200_delivery(payload, now_ts):
    rr = payload.get("RR200Signal") if isinstance(payload, dict) else {}
    if not isinstance(rr, dict):
        return
    if str(rr.get("status") or "") != "ready":
        return
    grade = str(rr.get("grade") or "")
    if grade not in {"A+ (Quant)", "A (High Accuracy)", "B (Qualified)"}:
        return

    counter = _load_rr200_counter(now_ts)
    today = counter.get("today") if isinstance(counter.get("today"), dict) else {}
    tiers = today.get("tiers") if isinstance(today.get("tiers"), dict) else {}
    tiers[grade] = int(tiers.get(grade, 0) or 0) + 1
    today["tiers"] = tiers
    today["total"] = int(today.get("total", 0) or 0) + 1
    counter["today"] = today
    counter["last_delivery_ts"] = int(now_ts)
    counter = _recompute_rr200_expected(counter)
    _save_json_file(RR200_SIGNAL_COUNTER_FILE, counter)


def _rr200_delivery_allowed(now_ts):
    counter = _load_rr200_counter(now_ts)
    return _rr200_should_deliver(counter, now_ts)


def _build_prediction_response():
    """Central prediction builder used by both HTTP and websocket monitor."""
    ta_data = predict_gold.get_technical_analysis()
    if not isinstance(ta_data, dict):
        ta_data = {"error": "Technical analysis payload is invalid."}

    if ta_data.get("error"):
        return {
            "status": "error",
            "message": ta_data["error"],
            "verdict": "Neutral",
            "confidence": 50,
            "TechnicalAnalysis": ta_data,
        }, 502

    regime_memory = _load_json_file(
        REGIME_MEMORY_FILE,
        {
            "warning_ladder": "Normal",
            "event_regime": "normal",
            "breakout_bias": "Neutral",
            "raw_warning_ladder": "Normal",
            "raw_warning_streak": 0,
            "warning_dwell_bars": 0,
            "breakout_bias_dwell_bars": 0,
        },
    )
    ta_with_memory = dict(ta_data)
    ta_with_memory["_regime_memory"] = regime_memory

    raw_prediction = compute_prediction_from_ta(ta_with_memory)
    if isinstance(raw_prediction.get("_regime_memory"), dict):
        _save_json_file(REGIME_MEMORY_FILE, raw_prediction["_regime_memory"])

    prediction = _stabilize_prediction(
        raw_prediction,
        ta_data,
    )
    ta_data["_prediction_market_state"] = prediction.get("MarketState", {})
    decision_status = _evaluate_decision_status(
        verdict=prediction["verdict"],
        confidence=int(prediction["confidence"]),
        ta_data=ta_data,
        trade_guidance=prediction["TradeGuidance"],
    )
    decision_status = _stabilize_decision_status(decision_status)
    execution_permission = _evaluate_execution_permission(
        decision_status=decision_status,
        market_state=prediction.get("MarketState", {}),
    )
    trade_playbook = _evaluate_trade_playbook(
        decision_status=decision_status,
        execution_permission=execution_permission,
        market_state=prediction.get("MarketState", {}),
        regime_state=prediction.get("RegimeState", {}),
        trade_guidance=prediction.get("TradeGuidance", {}),
    )
    trade_playbook = _stabilize_trade_playbook(
        trade_playbook,
        execution_permission=execution_permission,
        decision_status=decision_status,
    )
    forecast_state = dict(prediction.get("ForecastState", {}) or {})
    forecast_state["warningLadder"] = trade_playbook.get("warningLadder") or forecast_state.get("warningLadder")
    forecast_state["eventRegime"] = trade_playbook.get("eventRegime") or forecast_state.get("eventRegime")
    forecast_state["breakoutBias"] = trade_playbook.get("breakoutBias") or forecast_state.get("breakoutBias")
    forecast_state["directionalBias"] = trade_playbook.get("directionalBias") or forecast_state.get("directionalBias")

    execution_state = dict(prediction.get("ExecutionState", {}) or {})
    stabilized_execution_status = (
        "exit"
        if execution_permission.get("status") == "exit_recommended"
        else ("enter" if trade_playbook.get("stage") in {"enter", "hold"} else ("prepare" if trade_playbook.get("stage") in {"prepare", "stalk_entry"} else "stand_aside"))
    )
    execution_state["status"] = stabilized_execution_status
    execution_state["title"] = trade_playbook.get("title") or _humanize_value(stabilized_execution_status or "stand_aside")
    execution_state["text"] = trade_playbook.get("text") or execution_permission.get("text") or ""
    execution_state["entryAllowed"] = execution_permission.get("status") == "entry_allowed"
    execution_state["exitRecommended"] = execution_permission.get("status") == "exit_recommended"
    execution_state["permissionStatus"] = execution_permission.get("status")
    execution_state["action"] = (
        "exit"
        if stabilized_execution_status == "exit"
        else ("hold" if trade_playbook.get("stage") == "hold" else ("enter" if stabilized_execution_status == "enter" else ("prepare" if stabilized_execution_status == "prepare" else "stand_aside")))
    )
    if stabilized_execution_status == "exit":
        execution_state["actionState"] = "EXIT_RISK"
    elif trade_playbook.get("stage") == "hold" and trade_playbook.get("directionalBias") == "Bullish":
        execution_state["actionState"] = "LONG_ACTIVE"
    elif trade_playbook.get("stage") == "hold" and trade_playbook.get("directionalBias") == "Bearish":
        execution_state["actionState"] = "SHORT_ACTIVE"
    elif stabilized_execution_status == "enter" and trade_playbook.get("directionalBias") == "Bullish":
        execution_state["actionState"] = "SETUP_LONG"
    elif stabilized_execution_status == "enter" and trade_playbook.get("directionalBias") == "Bearish":
        execution_state["actionState"] = "SETUP_SHORT"
    else:
        execution_state["actionState"] = "WAIT"

    response_payload = {
        "status": "success",
        "verdict": prediction["verdict"],
        "confidence": prediction["confidence"],
        "TechnicalAnalysis": ta_data,
        "TradeGuidance": prediction["TradeGuidance"],
        "MarketState": prediction.get("MarketState", {}),
        "RegimeState": prediction.get("RegimeState", {}),
        "ForecastState": forecast_state,
        "ExecutionState": execution_state,
        "RR200Signal": prediction.get("RR200Signal", {}),
        "RR200LiveCounter": _load_rr200_counter(int(time.time())),
        "DecisionStatus": decision_status,
        "ExecutionPermission": execution_permission,
        "TradePlaybook": trade_playbook,
    }
    response_payload["DashboardAction"] = _derive_dashboard_action(
        response_payload,
        ta_data=ta_data,
    )

    return response_payload, 200


def _indicator_monitor_loop():
    """Push websocket notifications whenever tracked indicator values change."""
    last_snapshot = None

    while True:
        try:
            # Skip polling only when there are no live viewers and no background push subscribers.
            if _monitor_state["clients"] <= 0 and not _has_background_alert_channels():
                socketio.sleep(2)
                continue

            payload, status_code = _build_prediction_response()
            if status_code == 200:
                current_snapshot = _extract_indicator_snapshot(payload)
                current_price = float(((payload.get("TechnicalAnalysis") or {}).get("current_price")) or 0.0)
                if current_price > 0:
                    _update_live_signal_outcomes(current_price, int(time.time()))
                if last_snapshot is not None:
                    changes = _diff_snapshot(last_snapshot, current_snapshot)
                    notification_changes = _filter_notification_changes(changes)
                    now_ts = int(time.time())
                    if (
                        notification_changes
                        and _is_material_change(notification_changes)
                    ):
                        alert_state = _load_json_file(
                            ALERT_STATE_FILE,
                            {
                                "last_trade_playbook_stage": "",
                                "last_execution_permission": "",
                                "last_market_structure": "",
                                "last_warning_ladder": "",
                                "last_event_regime": "",
                                "last_breakout_bias": "",
                                "last_verdict": "",
                                "last_confidence_bucket": "",
                                "last_entry_readiness": "",
                                "last_exit_urgency": "",
                                "last_rr_signal_status": "",
                                "last_rr_signal_grade": "",
                                "last_rr_signal_direction": "",
                                "last_alert_ts": 0,
                                "last_playbook_alert_ts": 0,
                                "last_context_alert_ts": 0,
                                "last_execution_alert_ts": 0,
                                "last_diagnostics_alert_ts": 0,
                                "last_price_action_alert_ts": 0,
                                "last_boundary_wobble_ts": 0,
                            },
                        )
                        decision_payload = payload.get("DecisionStatus") or {}
                        execution_permission_payload = payload.get("ExecutionPermission") or {}
                        trade_playbook_payload = payload.get("TradePlaybook") or {}
                        execution_permission = str(execution_permission_payload.get("text") or "")
                        permission_status = str(execution_permission_payload.get("status") or "no_trade")
                        trade_playbook_stage = str(trade_playbook_payload.get("stage") or "")
                        entry_readiness = str(trade_playbook_payload.get("entryReadiness") or "")
                        exit_urgency = str(trade_playbook_payload.get("exitUrgency") or "")
                        market_structure = str(((payload.get("TechnicalAnalysis") or {}).get("price_action") or {}).get("structure") or "")
                        warning_ladder = str(trade_playbook_payload.get("warningLadder") or (payload.get("RegimeState") or {}).get("warning_ladder") or "")
                        event_regime = str(trade_playbook_payload.get("eventRegime") or (payload.get("RegimeState") or {}).get("event_regime") or "")
                        breakout_bias = str(trade_playbook_payload.get("breakoutBias") or (payload.get("RegimeState") or {}).get("breakout_bias") or "")
                        rr_signal_payload = payload.get("RR200Signal") or {}
                        rr_signal_status = str(rr_signal_payload.get("status") or "")
                        rr_signal_grade = str(rr_signal_payload.get("grade") or "")
                        rr_signal_direction = _effective_rr_direction(rr_signal_payload)
                        rr_signal_actionable = _is_rr_signal_actionable(rr_signal_payload)
                        verdict = str(payload.get("verdict") or "")
                        confidence_bucket = _confidence_bucket(payload.get("confidence"))
                        previous_trade_playbook_stage = str(alert_state.get("last_trade_playbook_stage", ""))
                        previous_warning_ladder = str(alert_state.get("last_warning_ladder", ""))
                        previous_event_regime = str(alert_state.get("last_event_regime", ""))
                        previous_breakout_bias = str(alert_state.get("last_breakout_bias", ""))
                        previous_confidence_bucket = str(alert_state.get("last_confidence_bucket", ""))
                        previous_rr_signal_status = str(alert_state.get("last_rr_signal_status", ""))
                        previous_rr_signal_grade = str(alert_state.get("last_rr_signal_grade", ""))
                        previous_rr_signal_direction = str(alert_state.get("last_rr_signal_direction", ""))
                        playbook_changed = (
                            "trade_playbook_stage" in notification_changes
                            and bool(trade_playbook_stage)
                            and _is_major_playbook_transition(previous_trade_playbook_stage, trade_playbook_stage)
                        )
                        warning_tier_changed = (
                            _warning_tier_rank(previous_warning_ladder)
                            != _warning_tier_rank(warning_ladder)
                        )
                        warning_changed = (
                            "warning_ladder" in notification_changes
                            and bool(warning_ladder)
                            and warning_tier_changed
                            and (
                                _warning_alert_tier(warning_ladder) != "low"
                                or _warning_alert_tier(previous_warning_ladder) != "low"
                            )
                        )
                        event_regime_changed = (
                            "event_regime" in notification_changes
                            and bool(event_regime)
                            and not _is_context_flip_noise(
                                previous_warning_ladder,
                                warning_ladder,
                                previous_event_regime,
                                event_regime,
                            )
                        )
                        breakout_bias_changed = (
                            "breakout_bias" in notification_changes
                            and bool(breakout_bias)
                            and warning_ladder in {"High Breakout Risk", "Directional Expansion Likely", "Active Momentum Event"}
                            and (
                                breakout_bias != previous_breakout_bias
                                and (breakout_bias in {"Bullish", "Bearish"} or previous_breakout_bias in {"Bullish", "Bearish"})
                            )
                        )
                        market_structure_changed = (
                            "market_structure" in notification_changes
                            and bool(market_structure)
                        )
                        microstructure_changed = bool(
                            "micro_vwap_band" in notification_changes
                            or "micro_vwap_bias" in notification_changes
                            or "micro_orb_state" in notification_changes
                            or "micro_sweep_state" in notification_changes
                        )
                        verdict_changed = "verdict" in notification_changes and bool(verdict)
                        confidence_changed = (
                            "confidence_bucket" in notification_changes
                            and bool(confidence_bucket)
                            and confidence_bucket != previous_confidence_bucket
                            and confidence_bucket in {"High", "Very High", "Low"}
                        )
                        execution_permission_changed = "execution_permission" in notification_changes and bool(execution_permission)
                        rr_signal_status_changed = (
                            "rr_signal_status" in notification_changes
                            and bool(rr_signal_status)
                            and rr_signal_status != previous_rr_signal_status
                            and rr_signal_status in {"arming", "ready"}
                            and rr_signal_actionable
                        )
                        rr_signal_grade_changed = (
                            "rr_signal_grade" in notification_changes
                            and bool(rr_signal_grade)
                            and rr_signal_grade != previous_rr_signal_grade
                            and rr_signal_actionable
                        )
                        rr_signal_direction_changed = (
                            "rr_signal_direction" in notification_changes
                            and bool(rr_signal_direction)
                            and rr_signal_direction != previous_rr_signal_direction
                            and rr_signal_actionable
                        )
                        entry_readiness_changed = (
                            "entry_readiness" in notification_changes
                            and playbook_changed
                            and entry_readiness in {"medium", "high"}
                        )
                        exit_urgency_changed = (
                            "exit_urgency" in notification_changes
                            and exit_urgency == "high"
                        )
                        should_alert = bool(
                            market_structure_changed
                            or microstructure_changed
                            or warning_changed
                            or event_regime_changed
                            or breakout_bias_changed
                            or verdict_changed
                            or confidence_changed
                            or execution_permission_changed
                            or entry_readiness_changed
                            or exit_urgency_changed
                            or rr_signal_status_changed
                            or rr_signal_grade_changed
                            or rr_signal_direction_changed
                        )
                        signal_class = ""
                        rr_ready_event = bool(
                            rr_signal_status_changed
                            and rr_signal_status == "ready"
                            and rr_signal_grade in {"A+ (Quant)", "A (High Accuracy)", "B (Qualified)"}
                        )
                        if market_structure_changed or microstructure_changed:
                            signal_class = "price_action"
                        elif rr_signal_grade_changed or rr_signal_direction_changed:
                            signal_class = "execution"

                        if not should_alert:
                            last_snapshot = current_snapshot
                            continue

                        _record_live_signal_outcome(
                            payload,
                            current_snapshot,
                            notification_changes,
                            now_ts,
                        )

                        alert_notification = _build_signal_notification(
                            notification_changes,
                            rr_signal_payload,
                            market_structure,
                            ta_data=payload.get("TechnicalAnalysis"),
                            payload=payload,
                        )
                        alert_title = alert_notification.get("title") or "XAUUSD Direction / Grade Changed"
                        alert_message = alert_notification.get("body") or ""

                        if _monitor_state["clients"] > 0:
                            socketio.emit(
                                "indicator_change",
                                {
                                    "message": alert_message,
                                    "title": alert_title,
                                    "data": payload,
                                    "changes": notification_changes,
                                    "snapshot": current_snapshot,
                                    "verdict": payload.get("verdict"),
                                    "confidence": payload.get("confidence"),
                                    "decision_status": payload.get("DecisionStatus"),
                                    "execution_permission": payload.get("ExecutionPermission"),
                                    "timestamp": now_ts,
                                },
                            )

                        _send_web_push_notifications(
                            changes=notification_changes,
                            rr_signal=rr_signal_payload,
                            market_structure=market_structure,
                            ta_data=payload.get("TechnicalAnalysis"),
                            payload=payload,
                        )
                        _send_telegram_notification(
                            changes=notification_changes,
                            rr_signal=rr_signal_payload,
                            market_structure=market_structure,
                            ta_data=payload.get("TechnicalAnalysis"),
                            payload=payload,
                        )
                        if rr_ready_event:
                            _record_rr200_delivery(payload, now_ts)
                        _save_json_file(
                            ALERT_STATE_FILE,
                            {
                                "last_trade_playbook_stage": trade_playbook_stage,
                                "last_execution_permission": execution_permission,
                                "last_market_structure": market_structure,
                                "last_warning_ladder": warning_ladder,
                                "last_event_regime": event_regime,
                                "last_breakout_bias": breakout_bias,
                                "last_verdict": verdict,
                                "last_confidence_bucket": confidence_bucket,
                                "last_entry_readiness": entry_readiness,
                                "last_exit_urgency": exit_urgency,
                                "last_rr_signal_status": rr_signal_status,
                                "last_rr_signal_grade": rr_signal_grade,
                                "last_rr_signal_direction": rr_signal_direction,
                                "last_alert_ts": now_ts,
                                "last_playbook_alert_ts": (
                                    now_ts if signal_class == "playbook" else int(alert_state.get("last_playbook_alert_ts", 0) or 0)
                                ),
                                "last_context_alert_ts": (
                                    now_ts if signal_class == "context" else int(alert_state.get("last_context_alert_ts", 0) or 0)
                                ),
                                "last_execution_alert_ts": (
                                    now_ts if signal_class == "execution" else int(alert_state.get("last_execution_alert_ts", 0) or 0)
                                ),
                                "last_diagnostics_alert_ts": (
                                    now_ts if signal_class == "diagnostics" else int(alert_state.get("last_diagnostics_alert_ts", 0) or 0)
                                ),
                                "last_price_action_alert_ts": (
                                    now_ts if signal_class == "price_action" else int(alert_state.get("last_price_action_alert_ts", 0) or 0)
                                ),
                                "last_boundary_wobble_ts": int(alert_state.get("last_boundary_wobble_ts", 0) or 0),
                            },
                        )
                last_snapshot = current_snapshot
        except Exception:
            # Keep monitor alive if any external API call fails.
            pass

        socketio.sleep(MONITOR_INTERVAL_SECONDS)


def _ensure_monitor_started():
    with _monitor_lock:
        if _monitor_state["started"]:
            return
        socketio.start_background_task(_indicator_monitor_loop)
        _monitor_state["started"] = True


@socketio.on("connect")
def _on_socket_connect():
    with _monitor_lock:
        _monitor_state["clients"] += 1
    _ensure_monitor_started()


@socketio.on("disconnect")
def _on_socket_disconnect():
    with _monitor_lock:
        _monitor_state["clients"] = max(0, _monitor_state["clients"] - 1)

@app.after_request
def apply_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/sw.js')
def service_worker():
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')


@app.route('/api/push/public-key')
def get_push_public_key():
    if not VAPID_PUBLIC_KEY:
        return jsonify({"status": "error", "message": "Push key is not configured."}), 503
    return jsonify({"status": "success", "publicKey": VAPID_PUBLIC_KEY})


@app.route('/api/push/subscribe', methods=['POST'])
def subscribe_push():
    payload = request.get_json(silent=True) or {}
    sub = payload.get('subscription') if isinstance(payload, dict) else None
    if not isinstance(sub, dict):
        return jsonify({"status": "error", "message": "Invalid subscription payload."}), 400

    _upsert_subscription(sub)
    _ensure_monitor_started()
    return jsonify({"status": "success"})


@app.route('/api/push/unsubscribe', methods=['POST'])
def unsubscribe_push():
    payload = request.get_json(silent=True) or {}
    endpoint = payload.get('endpoint') if isinstance(payload, dict) else None
    if not endpoint:
        return jsonify({"status": "error", "message": "Missing endpoint."}), 400

    _remove_subscription_endpoint(endpoint)
    return jsonify({"status": "success"})


@app.route('/api/autoresearch/latest')
def get_autoresearch_latest():
    base_dir = Path(__file__).resolve().parent
    report_path = base_dir / "tools" / "reports" / "autoresearch_last.json"
    job_reports_path = base_dir / "tools" / "reports" / "autoresearch_jobs"
    active_snapshot_path = base_dir / "tools" / "reports" / "autoresearch_active.json"
    strategy_params_path = base_dir / "config" / "strategy_params.json"
    backtest_params_path = base_dir / "config" / "backtest_params.json"

    def _read_dict(path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _best_params_from(report):
        report = report or {}
        if isinstance(report.get("best_params"), dict):
            return report.get("best_params") or {}
        best = report.get("best") or {}
        if isinstance(best, dict) and isinstance(best.get("params"), dict):
            return best.get("params") or {}
        return {}

    def _summary_from(report):
        report = report or {}
        if isinstance(report.get("summary"), dict):
            return report.get("summary") or {}
        best = report.get("best") or {}
        if isinstance(best, dict) and isinstance(best.get("summary"), dict):
            return best.get("summary") or {}
        return {}

    def _top_results_from(report):
        report = report or {}
        top_results = report.get("top_results")
        if isinstance(top_results, list):
            return top_results
        top_five = report.get("top_5")
        if isinstance(top_five, list):
            return top_five
        return []

    def _parse_report_timestamp(value):
        if value in (None, ""):
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _report_timestamp(report, path):
        parsed = _parse_report_timestamp((report or {}).get("generated_at"))
        if parsed is not None:
            return parsed
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    def _latest_staged_job_report(jobs_root):
        if not jobs_root.exists():
            return {}, None

        latest_report = {}
        latest_path = None
        latest_timestamp = datetime.min.replace(tzinfo=timezone.utc)

        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir() or job_dir.name.startswith("smoke_"):
                continue

            final_report_path = job_dir / "final_report.json"
            if not final_report_path.exists():
                continue

            job_report = _read_dict(final_report_path)
            if not job_report:
                continue
            if str(job_report.get("search_mode") or "").lower() != "staged":
                continue

            job_name = str(job_report.get("job_name") or job_dir.name)
            if job_name.startswith("smoke_"):
                continue

            job_timestamp = _report_timestamp(job_report, final_report_path)
            if job_timestamp > latest_timestamp:
                latest_timestamp = job_timestamp
                latest_report = job_report
                latest_path = final_report_path

        return latest_report, latest_path

    report = _read_dict(report_path)
    selected_report_path = report_path if report else None
    staged_report, staged_report_path = _latest_staged_job_report(job_reports_path)
    if staged_report:
        report_timestamp = _report_timestamp(report, report_path) if report else datetime.min.replace(tzinfo=timezone.utc)
        staged_timestamp = _report_timestamp(staged_report, staged_report_path)
        if not report or staged_timestamp >= report_timestamp:
            report = staged_report
            selected_report_path = staged_report_path

    if not report:
        return jsonify({
            "status": "error",
            "message": "Autoresearch report not found. Run tools/autoresearch_loop.py first."
        }), 404

    active_snapshot = _read_dict(active_snapshot_path)
    active_strategy = active_snapshot.get("active_strategy") if isinstance(active_snapshot.get("active_strategy"), dict) else {}
    active_params = active_strategy.get("strategy_params") if isinstance(active_strategy.get("strategy_params"), dict) else {}
    if not active_params:
        active_params = _read_dict(strategy_params_path)
    active_backtest_params = active_strategy.get("backtest_params") if isinstance(active_strategy.get("backtest_params"), dict) else {}
    if not active_backtest_params:
        active_backtest_params = _read_dict(backtest_params_path)
    active_summary = active_strategy.get("summary") if isinstance(active_strategy.get("summary"), dict) else {}
    if not active_summary:
        active_summary = {
            "winning_ema": f"{active_params.get('ema_short', '-')}/{active_params.get('ema_long', '-')}",
            "winning_rsi": f"{active_params.get('rsi_overbought', '-')}/{active_params.get('rsi_oversold', '-')}",
            "winning_cmf": active_params.get("cmf_window"),
        }
    recommendation = active_snapshot.get("recommendation") if isinstance(active_snapshot.get("recommendation"), dict) else {}
    active_strategy_status = "Snapshot available" if active_snapshot else "Loaded from live config"
    active_strategy_reason = (
        f"Current live strategy comes from {strategy_params_path.name} and {backtest_params_path.name}."
    )

    best = report.get("best") or {}
    summary = _summary_from(report)
    best_params = _best_params_from(report)
    top_results = _top_results_from(report)
    roi = summary.get("roi") if isinstance(summary, dict) else None
    promoted = bool(report.get("promote", False))
    recommendation_status = "Promotion recommended" if promoted else "Hold current config"
    tracked_keys = ("ema_short", "ema_long", "rsi_overbought", "rsi_oversold", "cmf_window")
    active_matches_best_candidate = bool(recommendation.get("matches_active_strategy"))
    if best_params:
        active_matches_best_candidate = all(active_params.get(key) == best_params.get(key) for key in tracked_keys)

    return jsonify({
        "status": "success",
        "report_file": str(selected_report_path) if selected_report_path else str(report_path),
        "generated_at": report.get("generated_at"),
        "latest_generated_at": report.get("generated_at"),
        "promote": promoted,
        "latest_run_promote": promoted,
        "promotion_reason": report.get("promotion_reason", ""),
        "recommendation_status": recommendation_status,
        "recommendation_reason": report.get("promotion_reason", ""),
        "active_snapshot_generated_at": active_snapshot.get("generated_at"),
        "active_strategy_status": active_strategy_status,
        "active_strategy_reason": active_strategy_reason,
        "active_matches_best_candidate": active_matches_best_candidate,
        "active_params": active_params,
        "active_backtest_params": active_backtest_params,
        "active_summary": active_summary,
        "best_params": best_params,
        "roi": roi,
        "winning_ema": f"{best_params.get('ema_short', '-')}/{best_params.get('ema_long', '-')}",
        "median_score": best.get("median_score") if isinstance(best, dict) else None,
        "pass_rate": best.get("pass_rate") if isinstance(best, dict) else None,
        "summary": {
            **(summary if isinstance(summary, dict) else {}),
            "roi": roi,
            "top_ranked_candidates": len(top_results),
            "winning_ema": f"{best_params.get('ema_short', '-')}/{best_params.get('ema_long', '-')}",
            "winning_rsi": f"{best_params.get('rsi_overbought', '-')}/{best_params.get('rsi_oversold', '-')}",
            "winning_cmf": best_params.get("cmf_window"),
            "latest_run_roi": roi,
            "latest_run_promote": promoted,
        },
        "checks": {},
    })

@app.route('/api/predict')
def get_prediction():
    try:
        payload, status_code = _build_prediction_response()
        return jsonify(payload), status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/health')
def get_health():
    try:
        ta_status = {
            "last_refresh_ts": int(getattr(predict_gold, "LAST_TA_REFRESH_TS", 0) or 0),
            "cache_seconds": int(getattr(predict_gold, "TECHNICAL_ANALYSIS_CACHE_SECONDS", 20) or 20),
            "has_cached_snapshot": bool(getattr(predict_gold, "LAST_SUCCESSFUL_TA", None)),
        }
        cross_asset_status = {
            "last_refresh_ts": int(getattr(predict_gold, "LAST_CROSS_ASSET_TS", 0) or 0),
            "cache_seconds": int(getattr(predict_gold, "CROSS_ASSET_CACHE_SECONDS", 90) or 90),
            "has_cached_snapshot": isinstance(getattr(predict_gold, "LAST_CROSS_ASSET_CONTEXT", None), dict),
        }
        live_signal_summary = _load_json_file(LIVE_SIGNAL_SUMMARY_FILE, {})
        return jsonify({
            "status": "ok",
            "runtime": {
                "socket_async_mode": getattr(socketio, "async_mode", "unknown"),
                "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
                "notify_min_interval_seconds": NOTIFY_MIN_INTERVAL_SECONDS,
                "has_background_alert_channels": _has_background_alert_channels(),
            },
            "monitor": {
                "started": bool(_monitor_state.get("started")),
                "clients": int(_monitor_state.get("clients", 0) or 0),
            },
            "prediction_cache": ta_status,
            "cross_asset_cache": cross_asset_status,
            "live_signal_summary": live_signal_summary,
        }), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/outcomes/latest')
def get_latest_outcomes():
    try:
        return jsonify({
            "status": "success",
            "summary": _load_json_file(LIVE_SIGNAL_SUMMARY_FILE, {}),
            "records": (_load_json_file(LIVE_SIGNAL_OUTCOMES_FILE, {"records": []}).get("records") or [])[-50:],
        }), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


_ensure_monitor_started()

if __name__ == '__main__':
    # Run on 0.0.0.0 to allow access from other devices on the local network (like your phone)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
