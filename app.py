from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, join_room, leave_room
import copy
import hashlib
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import requests
from tools import predict_gold
from tools.research_runtime import create_research_brief, load_job_bundle, research_status
from tools.signal_engine import (
    compute_prediction_from_ta,
)
from tools.trade_brain import TradeBrainService

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
EXECUTION_QUALITY_ALERT_CONFIRMATION_COUNT = _read_int_env("EXECUTION_QUALITY_ALERT_CONFIRMATION_COUNT", 2, 1)
EXECUTION_QUALITY_CLEAR_CONFIRMATION_COUNT = _read_int_env("EXECUTION_QUALITY_CLEAR_CONFIRMATION_COUNT", 3, 1)
BOUNDARY_WOBBLE_COOLDOWN_SECONDS = _read_int_env("BOUNDARY_WOBBLE_COOLDOWN_SECONDS", 1200, 0)
ENTER_STAGE_PROTECT_SECONDS = _read_int_env("ENTER_STAGE_PROTECT_SECONDS", 900, 0)
STABLE_DECISION_BUFFER_BUCKET_SECONDS = _read_int_env("STABLE_DECISION_BUFFER_BUCKET_SECONDS", 30, 5)
STABLE_DECISION_BUFFER_MAX_BARS = _read_int_env("STABLE_DECISION_BUFFER_MAX_BARS", 5, 3)
STABLE_DECISION_BUFFER_WINDOW_SECONDS = _read_int_env("STABLE_DECISION_BUFFER_WINDOW_SECONDS", 150, 30)
STABLE_DECISION_REEVALUATION_BARS = _read_int_env("STABLE_DECISION_REEVALUATION_BARS", 3, 1)
STABLE_DECISION_CANDIDATE_SECONDS = _read_int_env("STABLE_DECISION_CANDIDATE_SECONDS", 90, 30)
STABLE_DECISION_INVALIDATION_SECONDS = _read_int_env("STABLE_DECISION_INVALIDATION_SECONDS", 60, 30)
STABLE_DECISION_LOCK_SECONDS = _read_int_env("STABLE_DECISION_LOCK_SECONDS", 300, 0)
STABLE_DECISION_FLIP_WINDOW_SECONDS = _read_int_env("STABLE_DECISION_FLIP_WINDOW_SECONDS", 600, 60)
STABLE_DECISION_REPEAT_SUPPRESS_SECONDS = _read_int_env("STABLE_DECISION_REPEAT_SUPPRESS_SECONDS", 300, 0)
STABLE_DECISION_OSCILLATION_SECONDS = _read_int_env("STABLE_DECISION_OSCILLATION_SECONDS", 180, 0)
STABLE_DECISION_CONFIDENCE_FLIP_PENALTY = _read_int_env("STABLE_DECISION_CONFIDENCE_FLIP_PENALTY", 15, 0)
STABLE_DECISION_HISTORY_LIMIT = _read_int_env("STABLE_DECISION_HISTORY_LIMIT", 10, 3)
DECISION_CHURN_LOG_LIMIT = _read_int_env("DECISION_CHURN_LOG_LIMIT", 200, 20)
NOTIFY_EXIT_READS = _read_bool_env("NOTIFY_EXIT_READS", True)
RR200_MAX_SIGNALS_PER_DAY = _read_int_env("RR200_MAX_SIGNALS_PER_DAY", 5, 1)
RR200_MIN_SIGNAL_SPACING_SECONDS = _read_int_env("RR200_MIN_SIGNAL_SPACING_SECONDS", 2700, 0)
VWAP_BIAS_ALERT_HYSTERESIS_PCT = 0.02
VWAP_BIAS_ALERT_MIN_MOVE_PCT = 0.03
VWAP_DELTA_ALERT_THRESHOLD_PCT = 0.10
NOTIFICATION_ALLOWED_FIELDS = {
    "market_structure",
    "micro_vwap_bias",
    "micro_vwap_delta_pct",
    "micro_orb_state",
    "micro_sweep_state",
    "execution_quality_signal",
}
PUSH_EXCLUDED_FIELDS = {
    "rsi_14",
    "ema_20",
    "ema_50",
    "adx_14",
    "atr_percent",
    "micro_vwap_band",
    "execution_state",
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
TRADE_BRAIN_STATE_FILE = BASE_DIR / "tools" / "reports" / "trade_brain_state.json"
STABLE_DECISION_STATE_FILE = BASE_DIR / "tools" / "reports" / "stable_decision_state.json"
DECISION_CHURN_LOG_FILE = BASE_DIR / "tools" / "reports" / "decision_churn_log.json"
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_CLAIMS_SUBJECT = os.getenv("VAPID_CLAIMS_SUBJECT", "mailto:alerts@example.com")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID", "").strip()
ENABLE_TELEGRAM_NOTIFICATIONS = _read_bool_env("GOLD_PREDICTOR_ENABLE_TELEGRAM", False)
CHANGE_SUMMARY_ORDER = [
    "market_structure",
    "execution_quality_signal",
    "micro_vwap_bias",
    "micro_vwap_delta_pct",
    "micro_orb_state",
    "micro_sweep_state",
]
trade_brain_service = TradeBrainService(TRADE_BRAIN_STATE_FILE)


def _trade_brain_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _trade_brain_first_float(*values, default=None):
    for value in values:
        parsed = _trade_brain_float(value, None)
        if parsed is not None:
            return parsed
    return default


def _get_request_user_id(default="anonymous"):
    raw = request.headers.get("x-user-id") or request.args.get("userId") or default
    text = str(raw or "").strip()
    return text or default


def _get_socket_user_id(payload=None, default="anonymous"):
    if isinstance(payload, dict):
        raw = payload.get("userId") or payload.get("user_id")
        text = str(raw or "").strip()
        if text:
            return text
    return default


def _trade_room(trade_id):
    return f"trade:{trade_id}"


def _build_trade_brain_market_data(ta_data, regime_state=None):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    regime_state = regime_state if isinstance(regime_state, dict) else {}
    price_action = ta_data.get("price_action") if isinstance(ta_data.get("price_action"), dict) else {}
    structure_context = ta_data.get("structure_context") if isinstance(ta_data.get("structure_context"), dict) else {}
    session_context = ta_data.get("session_context") if isinstance(ta_data.get("session_context"), dict) else {}
    current_session = session_context.get("current_session") if isinstance(session_context.get("current_session"), dict) else {}

    price = _trade_brain_first_float(
        ta_data.get("current_price"),
        ta_data.get("price"),
        ta_data.get("close"),
        default=None,
    )
    vwap = _trade_brain_first_float(
        ta_data.get("session_vwap"),
        ta_data.get("vwap"),
        ta_data.get("SESSION_VWAP"),
        structure_context.get("sessionVwap"),
        price,
    )
    structure = (
        price_action.get("structure")
        or ta_data.get("market_structure")
        or structure_context.get("marketStructure")
        or "neutral"
    )
    regime = (
        regime_state.get("event_regime")
        or regime_state.get("eventRegime")
        or regime_state.get("warning_ladder")
        or regime_state.get("warningLadder")
        or ta_data.get("volatility_regime")
        or session_context.get("marketStatus")
        or "normal"
    )
    return {
        "price": price,
        "adx": _trade_brain_first_float(ta_data.get("adx_14"), ta_data.get("ADX_14"), default=0.0),
        "vwap": vwap,
        "atrDollar": _trade_brain_first_float(ta_data.get("atr_14"), ta_data.get("ATR_14"), default=0.0),
        "atrPercent": _trade_brain_first_float(ta_data.get("atr_percent"), ta_data.get("ATR_PERCENT"), default=0.0),
        "session": session_context.get("currentLabel") or current_session.get("label") or "Unknown",
        "structure": structure,
        "regime": str(regime),
    }


def _trade_brain_direction_from_prediction(prediction):
    if not isinstance(prediction, dict):
        return None

    stable_decision = prediction.get("StableDecision") if isinstance(prediction.get("StableDecision"), dict) else {}
    stable_direction = str(stable_decision.get("direction") or "").strip().lower()
    if stable_direction == "long":
        return "LONG"
    if stable_direction == "short":
        return "SHORT"

    execution_quality = prediction.get("ExecutionQuality") if isinstance(prediction.get("ExecutionQuality"), dict) else {}
    execution_direction = str(execution_quality.get("direction") or "").strip().lower()
    if execution_direction == "long":
        return "LONG"
    if execution_direction == "short":
        return "SHORT"

    decision_status = prediction.get("DecisionStatus") if isinstance(prediction.get("DecisionStatus"), dict) else {}
    decision_direction = str(decision_status.get("status") or "").strip().lower()
    if decision_direction in {"buy", "long"}:
        return "LONG"
    if decision_direction in {"sell", "short"}:
        return "SHORT"

    verdict = str(prediction.get("verdict") or "").strip().lower()
    if verdict.startswith("bull"):
        return "LONG"
    if verdict.startswith("bear"):
        return "SHORT"

    guidance = prediction.get("TradeGuidance") if isinstance(prediction.get("TradeGuidance"), dict) else {}
    buy_level = str(guidance.get("buyLevel") or "").strip().lower()
    sell_level = str(guidance.get("sellLevel") or "").strip().lower()
    if buy_level == "strong" and sell_level != "strong":
        return "LONG"
    if sell_level == "strong" and buy_level != "strong":
        return "SHORT"
    return None


def _apply_trade_brain_learning_to_prediction(prediction, learning_adjustment):
    if not isinstance(prediction, dict):
        return prediction
    if not isinstance(learning_adjustment, dict) or not learning_adjustment.get("active"):
        return prediction

    adjusted = copy.deepcopy(prediction)
    delta = int(learning_adjustment.get("confidenceDelta") or 0)
    adjusted["confidence"] = max(0, min(100, int(adjusted.get("confidence") or 50) + delta))
    guidance = dict(adjusted.get("TradeGuidance", {}) or {})
    learning_note = str(learning_adjustment.get("message") or "").strip()
    top_setup = learning_adjustment.get("topSetup") if isinstance(learning_adjustment.get("topSetup"), dict) else {}
    sizing = learning_adjustment.get("sizing") if isinstance(learning_adjustment.get("sizing"), dict) else {}
    existing_summary = str(guidance.get("summary") or "").strip()
    guidance["learningNote"] = learning_note
    guidance["learningConfidenceDelta"] = delta

    summary_segments = [existing_summary] if existing_summary else []
    if learning_note:
        summary_segments.append(f"Reinforcement memory: {learning_note}")

    if top_setup.get("setup"):
        guidance["learningSetup"] = copy.deepcopy(top_setup)
        summary_segments.append(
            f"Preferred learned setup: {top_setup['setup']} from {int(top_setup.get('samples') or 0)} similar trades."
        )

    if sizing.get("active"):
        guidance["learningSizing"] = copy.deepcopy(sizing)
        size_note = str(sizing.get("summary") or "").strip()
        if size_note:
            summary_segments.append(size_note)

    guidance["summary"] = " ".join(segment for segment in summary_segments if segment)
    adjusted["TradeGuidance"] = guidance
    adjusted["TradeBrainLearning"] = copy.deepcopy(learning_adjustment)
    return adjusted


def _apply_trade_brain_learning_to_execution_quality(execution_quality, learning_adjustment):
    if not isinstance(execution_quality, dict):
        return execution_quality
    if not isinstance(learning_adjustment, dict) or not learning_adjustment.get("active"):
        return execution_quality

    adjusted = copy.deepcopy(execution_quality)
    delta = int(learning_adjustment.get("confidenceDelta") or 0)
    top_setup = learning_adjustment.get("topSetup") if isinstance(learning_adjustment.get("topSetup"), dict) else {}
    sizing = learning_adjustment.get("sizing") if isinstance(learning_adjustment.get("sizing"), dict) else {}
    score = _trade_brain_first_float(adjusted.get("score"), default=None)
    if score is not None:
        adjusted["score"] = max(0.0, min(100.0, round(score + (delta * 1.5), 1)))
    adjusted["learning"] = copy.deepcopy(learning_adjustment)
    learning_note = str(learning_adjustment.get("message") or "").strip()
    if sizing.get("summary"):
        existing_position_note = str(adjusted.get("positionSizeNote") or "").strip()
        sizing_note = str(sizing.get("summary") or "").strip()
        adjusted["positionSizeNote"] = (
            f"{existing_position_note} · {sizing_note}"
            if existing_position_note and sizing_note not in existing_position_note
            else sizing_note or existing_position_note
        )

    if top_setup.get("setup"):
        setup_name = str(top_setup.get("setup") or "").strip()
        setup_note = (
            f"Learned setup bias: {setup_name} from {int(top_setup.get('samples') or 0)} similar trades "
            f"averaging {float(top_setup.get('avgReward', 0.0) or 0.0):+.2f}R."
        )
        recommendation_score = float(top_setup.get("recommendationScore", 0.0) or 0.0)
        existing_setup = str(adjusted.get("setup") or "").strip()
        if recommendation_score > 0:
            if setup_name and setup_name.lower() not in existing_setup.lower():
                adjusted["setup"] = (
                    f"{existing_setup} · Prefer {setup_name}"
                    if existing_setup
                    else f"Prefer {setup_name}"
                )
            reasons = list(adjusted.get("reasons") or [])
            if setup_note not in reasons:
                reasons.insert(0, setup_note)
            adjusted["reasons"] = reasons[:6]
        elif recommendation_score < 0:
            if setup_name and setup_name.lower() not in existing_setup.lower():
                adjusted["setup"] = (
                    f"{existing_setup} · Avoid {setup_name}"
                    if existing_setup
                    else f"Avoid {setup_name}"
                )
            blockers = list(adjusted.get("blockers") or [])
            if setup_note not in blockers:
                blockers.insert(0, setup_note)
            adjusted["blockers"] = blockers[:6]

    if delta > 0:
        reasons = list(adjusted.get("reasons") or [])
        if learning_note and learning_note not in reasons:
            reasons.insert(0, learning_note)
        adjusted["reasons"] = reasons[:6]
    elif delta < 0:
        blockers = list(adjusted.get("blockers") or [])
        if learning_note and learning_note not in blockers:
            blockers.insert(0, learning_note)
        adjusted["blockers"] = blockers[:6]
    return adjusted


def _enrich_trade_create_payload(payload, user_id):
    payload = copy.deepcopy(payload if isinstance(payload, dict) else {})
    dashboard = trade_brain_service.get_dashboard_payload(user_id=user_id)
    market_data = dashboard.get("marketData") if isinstance(dashboard.get("marketData"), dict) else {}
    merged_context = dict(market_data)
    incoming_context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    merged_context.update(incoming_context)
    payload["context"] = merged_context
    if payload.get("price") in (None, "") and market_data.get("price") is not None:
        payload["price"] = market_data.get("price")
    return payload


def _emit_trade_brain_stats(dashboard, sid=None, user_id=None):
    if not isinstance(dashboard, dict):
        return
    payload = {
        "userId": str(user_id or "").strip() or None,
        "stats": dashboard.get("stats") or {},
        "dashboard": dashboard,
    }
    if sid:
        socketio.emit("stats:update", payload, to=sid)
        return
    socketio.emit("stats:update", payload)


def _emit_trade_brain_trade(event_name, trade, dashboard, sid=None, user_id=None):
    resolved_user_id = str(user_id or ((trade or {}).get("userId") if isinstance(trade, dict) else "") or "").strip() or None
    payload = {
        "userId": resolved_user_id,
        "trade": trade,
        "dashboard": dashboard,
    }
    if sid:
        socketio.emit(event_name, payload, to=sid)
        return
    socketio.emit(event_name, payload)


def _emit_trade_brain_events(events, trade=None, dashboard=None, sid=None, user_id=None):
    resolved_user_id = str(user_id or ((trade or {}).get("userId") if isinstance(trade, dict) else "") or "").strip() or None
    emitted = False
    for event in events or []:
        event_name = str(event.get("type") or "").strip()
        if not event_name:
            continue
        emitted = True
        if event_name in {"trade:created", "trade:updated", "trade:closed"}:
            _emit_trade_brain_trade(event_name, trade, dashboard, sid=sid, user_id=resolved_user_id)
            continue
        event_payload = dict(event) if isinstance(event, dict) else {"message": str(event)}
        if resolved_user_id and not event_payload.get("userId"):
            event_payload["userId"] = resolved_user_id
        if sid:
            socketio.emit(event_name, event_payload, to=sid)
        else:
            socketio.emit(event_name, event_payload)
    if emitted:
        _emit_trade_brain_stats(dashboard, sid=sid, user_id=resolved_user_id)
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
        if key in NOTIFICATION_ALLOWED_FIELDS and key not in PUSH_EXCLUDED_FIELDS
    }
    market_structure_delta = filtered.get("market_structure")
    if isinstance(market_structure_delta, dict) and not _is_alertable_market_structure_transition(
        market_structure_delta.get("previous"),
        market_structure_delta.get("current"),
    ):
        filtered.pop("market_structure", None)

    for micro_key in ("micro_orb_state", "micro_sweep_state"):
        micro_delta = filtered.get(micro_key)
        if isinstance(micro_delta, dict) and not _is_alertable_directional_micro_transition(
            micro_delta.get("previous"),
            micro_delta.get("current"),
        ):
            filtered.pop(micro_key, None)

    execution_quality_delta = filtered.get("execution_quality_signal")
    if isinstance(execution_quality_delta, dict) and not _is_material_execution_quality_transition(
        execution_quality_delta.get("previous"),
        execution_quality_delta.get("current"),
    ):
        filtered.pop("execution_quality_signal", None)

    vwap_bias_delta = filtered.get("micro_vwap_bias")
    if isinstance(vwap_bias_delta, dict):
        vwap_value_delta = filtered.get("micro_vwap_delta_pct")
        previous_value = vwap_value_delta.get("previous") if isinstance(vwap_value_delta, dict) else None
        current_value = vwap_value_delta.get("current") if isinstance(vwap_value_delta, dict) else None
        if not _is_alertable_vwap_bias_transition(
            vwap_bias_delta.get("previous"),
            vwap_bias_delta.get("current"),
            previous_value,
            current_value,
        ):
            filtered.pop("micro_vwap_bias", None)

    if "micro_vwap_bias" not in filtered:
        delta = filtered.get("micro_vwap_delta_pct")
        prev = delta.get("previous") if isinstance(delta, dict) else None
        cur = delta.get("current") if isinstance(delta, dict) else None
        if not (
            isinstance(prev, (int, float))
            and isinstance(cur, (int, float))
            and abs(float(cur) - float(prev)) >= VWAP_DELTA_ALERT_THRESHOLD_PCT
        ):
            filtered.pop("micro_vwap_delta_pct", None)
    return filtered


def _summarize_changes_for_push(changes):
    labels = {
        "market_structure": "Market Structure",
        "execution_quality_signal": "Execution Quality",
        "micro_vwap_bias": "VWAP Bias",
        "micro_vwap_delta_pct": "VWAP",
        "micro_orb_state": "ORB",
        "micro_sweep_state": "Sweep",
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
    return " | ".join(parts) if parts else "Signal state changed"


def _notification_fingerprint(changes):
    filtered = _filter_notification_changes(changes)
    normalized_changes = {}
    for key in sorted(filtered.keys()):
        value = filtered.get(key)
        if isinstance(value, dict):
            normalized_changes[key] = {
                "previous": _normalize_fingerprint_value(key, value.get("previous")),
                "current": _normalize_fingerprint_value(key, value.get("current")),
            }
        else:
            normalized_changes[key] = _normalize_fingerprint_value(key, value)
    raw_payload = json.dumps(
        {
            "title": _notification_title_for_changes(filtered),
            "changes": normalized_changes,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha1(raw_payload.encode("utf-8")).hexdigest()


def _normalize_fingerprint_value(key, value):
    if key == "micro_vwap_delta_pct":
        numeric = _coerce_float(value, None)
        return _round_display_value(numeric, 2, value) if numeric is not None else value
    if key in {"micro_orb_state", "micro_sweep_state"}:
        return _coerce_directional_state(value)
    if key == "execution_quality_signal":
        return _execution_quality_signal_core(value)
    return value


def _notification_tag_from_fingerprint(fingerprint):
    fingerprint = str(fingerprint or "").strip()
    if not fingerprint:
        return "xauusd-alert"
    return f"xauusd-alert-{fingerprint[:16]}"


def _coerce_directional_state(value):
    try:
        numeric = int(round(float(value)))
    except Exception:
        numeric = 0
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


def _is_alertable_directional_micro_transition(previous, current):
    previous_state = _coerce_directional_state(previous)
    current_state = _coerce_directional_state(current)
    if previous_state == current_state:
        return False
    # Fresh directional ORB/sweep events and direct side flips are actionable.
    # Clearing back to neutral/none is usually just signal expiry.
    return current_state != 0


def _market_structure_bias(value):
    text = str(value or "").strip()
    has_bullish = "Bullish" in text or "Higher Highs" in text or "Breakout" in text
    has_bearish = "Bearish" in text or "Lower Highs" in text or "Breakdown" in text
    if has_bullish and not has_bearish:
        return "Bullish"
    if has_bearish and not has_bullish:
        return "Bearish"
    return "Neutral"


def _market_structure_alert_tier(value):
    text = str(value or "").strip()
    if "Breakout" in text or "Breakdown" in text:
        return 3
    if "Higher Highs" in text or "Lower Highs" in text or "Structure" in text:
        return 2
    if "Drift" in text or "Pressure in Range" in text:
        return 1
    return 0


def _is_alertable_market_structure_transition(previous, current):
    previous_text = str(previous or "").strip()
    current_text = str(current or "").strip()
    if not current_text or previous_text == current_text:
        return False
    previous_bias = _market_structure_bias(previous_text)
    current_bias = _market_structure_bias(current_text)
    if current_bias == "Neutral":
        return False
    if previous_bias == "Neutral":
        return True
    if previous_bias != current_bias:
        return True
    return _market_structure_alert_tier(current_text) > _market_structure_alert_tier(previous_text)


def _alert_class_metadata(signal_class):
    signal_class = str(signal_class or "").strip().lower()
    mapping = {
        "context": (
            "last_context_alert_ts",
            "last_context_alert_fingerprint",
            ALERT_CONTEXT_COOLDOWN_SECONDS,
        ),
        "execution": (
            "last_execution_alert_ts",
            "last_execution_alert_fingerprint",
            ALERT_CLASS_COOLDOWN_SECONDS,
        ),
        "diagnostics": (
            "last_diagnostics_alert_ts",
            "last_diagnostics_alert_fingerprint",
            ALERT_CLASS_COOLDOWN_SECONDS,
        ),
        "price_action": (
            "last_price_action_alert_ts",
            "last_price_action_alert_fingerprint",
            PRICE_ACTION_ALERT_COOLDOWN_SECONDS,
        ),
    }
    return mapping.get(
        signal_class,
        ("last_alert_ts", "last_alert_fingerprint", ALERT_COOLDOWN_SECONDS),
    )


def _should_suppress_duplicate_alert(alert_state, fingerprint, signal_class, now_ts):
    fingerprint = str(fingerprint or "").strip()
    if not fingerprint:
        return False

    alert_state = alert_state if isinstance(alert_state, dict) else {}
    global_last_ts = int(alert_state.get("last_alert_ts", 0) or 0)
    global_last_fingerprint = str(alert_state.get("last_alert_fingerprint", "") or "")
    if (
        global_last_fingerprint == fingerprint
        and ALERT_COOLDOWN_SECONDS > 0
        and (now_ts - global_last_ts) < ALERT_COOLDOWN_SECONDS
    ):
        return True

    class_ts_key, class_fp_key, class_cooldown = _alert_class_metadata(signal_class)
    class_last_ts = int(alert_state.get(class_ts_key, 0) or 0)
    class_last_fingerprint = str(alert_state.get(class_fp_key, "") or "")
    return (
        class_last_fingerprint == fingerprint
        and class_cooldown > 0
        and (now_ts - class_last_ts) < class_cooldown
    )


def _notification_title_for_changes(changes):
    changed_keys = set(_filter_notification_changes(changes).keys())
    has_structure = "market_structure" in changed_keys
    has_execution_quality = "execution_quality_signal" in changed_keys
    has_micro_vwap = (
        "micro_vwap_bias" in changed_keys
        or "micro_vwap_delta_pct" in changed_keys
    )
    has_micro_orb = "micro_orb_state" in changed_keys
    has_micro_sweep = "micro_sweep_state" in changed_keys
    has_microstructure = has_micro_vwap or has_micro_orb or has_micro_sweep

    if has_structure and has_microstructure:
        return "XAUUSD Price Action Changed"
    if has_execution_quality and (has_structure or has_microstructure):
        return "XAUUSD Price Action Changed"
    if has_execution_quality:
        return "XAUUSD Execution Quality Changed"
    if has_microstructure:
        return "XAUUSD Microstructure Changed"
    if has_structure:
        return "XAUUSD Market Structure Changed"
    return "XAUUSD Signal Changed"


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


def _round_display_value(value, digits=2, default=0.0):
    try:
        quantizer = Decimal("1").scaleb(-int(digits))
        return float(
            Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP)
        )
    except Exception:
        return default


def _raw_vwap_bias_label(value):
    numeric = _round_display_value(value, 2, 0.0)
    if numeric >= 0.30:
        return "Bullish"
    if numeric >= 0.10:
        return "Mild Bullish"
    if numeric > -0.10:
        return "Neutral"
    if numeric > -0.30:
        return "Mild Bearish"
    return "Bearish"


def _vwap_bias_label(value):
    return _raw_vwap_bias_label(value)


def _is_alertable_vwap_bias_transition(previous_bias, current_bias, previous_value=None, current_value=None):
    previous_label = str(previous_bias or "").strip()
    current_label = str(current_bias or "").strip()
    if not previous_label or not current_label or previous_label == current_label:
        return False

    prev = _coerce_float(previous_value, None)
    cur = _coerce_float(current_value, None)
    if prev is not None and cur is not None:
        if round(prev, 2) == round(cur, 2):
            return False
        if abs(cur - prev) < VWAP_BIAS_ALERT_MIN_MOVE_PCT:
            return False

    return True


def _stabilize_vwap_bias_label(value, previous_label=None):
    raw_label = _raw_vwap_bias_label(value)
    previous_label = str(previous_label or "").strip()
    if not previous_label or previous_label == raw_label:
        return raw_label

    numeric = _round_display_value(value, 2, 0.0)

    hysteresis = float(VWAP_BIAS_ALERT_HYSTERESIS_PCT)
    neutral_lower = _round_display_value(-0.10 - hysteresis, 2, -0.12)
    neutral_upper = _round_display_value(0.10 + hysteresis, 2, 0.12)
    mild_bearish_lower = _round_display_value(-0.30 - hysteresis, 2, -0.32)
    mild_bearish_upper = _round_display_value(-0.10 + hysteresis, 2, -0.08)
    bearish_upper = _round_display_value(-0.30 + hysteresis, 2, -0.28)
    mild_bullish_lower = _round_display_value(0.10 - hysteresis, 2, 0.08)
    mild_bullish_upper = _round_display_value(0.30 + hysteresis, 2, 0.32)
    bullish_lower = _round_display_value(0.30 - hysteresis, 2, 0.28)

    if previous_label == "Neutral":
        if neutral_lower < numeric < neutral_upper:
            return "Neutral"
    elif previous_label == "Mild Bearish":
        if mild_bearish_lower < numeric <= mild_bearish_upper:
            return "Mild Bearish"
    elif previous_label == "Bearish":
        if numeric <= bearish_upper:
            return "Bearish"
    elif previous_label == "Mild Bullish":
        if mild_bullish_lower <= numeric < mild_bullish_upper:
            return "Mild Bullish"
    elif previous_label == "Bullish":
        if numeric >= bullish_lower:
            return "Bullish"

    return raw_label


def _format_vwap_microstructure_text(value, bias_label=None):
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    label = str(bias_label or "").strip() or _vwap_bias_label(numeric)
    return f"VWAP {numeric:.2f}% {label}"


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
    vwap_bias_delta = changes.get("micro_vwap_bias")
    vwap_value_delta = changes.get("micro_vwap_delta_pct")
    if isinstance(vwap_bias_delta, dict) or isinstance(vwap_value_delta, dict):
        prev_bias = str((vwap_bias_delta or {}).get("previous") or "")
        cur_bias = str((vwap_bias_delta or {}).get("current") or "")
        prev_val = (vwap_value_delta or {}).get("previous") if isinstance(vwap_value_delta, dict) else None
        cur_val = (vwap_value_delta or {}).get("current") if isinstance(vwap_value_delta, dict) else None
        value_suffix = ""
        if isinstance(prev_val, (int, float)) and isinstance(cur_val, (int, float)):
            value_suffix = f" ({float(prev_val):.2f}% -> {float(cur_val):.2f}%)"
        if (
            prev_bias
            and cur_bias
            and _is_alertable_vwap_bias_transition(prev_bias, cur_bias, prev_val, cur_val)
        ):
            strategy_note = _vwap_bias_strategy_note(cur_bias)
            strategy_suffix = f": {strategy_note}" if strategy_note else ""
            parts.append(f"VWAP {prev_bias} -> {cur_bias}{value_suffix}{strategy_suffix}")
        elif (
            isinstance(prev_val, (int, float))
            and isinstance(cur_val, (int, float))
            and abs(float(cur_val) - float(prev_val)) >= VWAP_DELTA_ALERT_THRESHOLD_PCT
        ):
            current_bias = cur_bias or prev_bias or _vwap_bias_label(cur_val)
            strength_word = (
                "strengthened"
                if abs(float(cur_val)) > abs(float(prev_val))
                else "eased"
            )
            strategy_note = _vwap_bias_strategy_note(current_bias)
            strategy_suffix = f": {strategy_note}" if strategy_note else ""
            parts.append(
                f"VWAP {current_bias} {strength_word}{value_suffix}{strategy_suffix}"
            )
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


def _format_execution_quality_change(changes):
    changes = changes if isinstance(changes, dict) else {}
    delta = changes.get("execution_quality_signal")
    if not isinstance(delta, dict):
        return ""
    previous = str(delta.get("previous") or "").strip() or "None"
    current = str(delta.get("current") or "").strip() or "None"
    if not _is_material_execution_quality_transition(previous, current):
        return ""
    previous_category = _execution_quality_signal_category(previous)
    current_category = _execution_quality_signal_category(current)
    if current_category in {"ready", "watchlist"} and previous_category in {"no_trade", "hard_blocked"}:
        return f"Setup confirmed: {current}"
    if previous_category in {"ready", "watchlist"} and current_category == "no_trade":
        return f"Setup invalidated: {previous} -> No Trade"
    if current_category == "hard_blocked":
        reason = current.replace("No Trade hard blocked:", "").strip()
        return f"Hard block active: {reason or current}"
    return f"{previous} -> {current}"


def _extract_market_session_state(ta_data):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    session = ta_data.get("session_context") if isinstance(ta_data.get("session_context"), dict) else {}
    bar_session = session.get("bar_session") if isinstance(session.get("bar_session"), dict) else {}
    current_session = session.get("current_session") if isinstance(session.get("current_session"), dict) else {}
    is_market_closed = bool(
        current_session.get("isMarketClosed")
        or session.get("isMarketClosed")
        or ta_data.get("market_closed")
    )
    return {
        "session": session,
        "bar_session": bar_session,
        "current_session": current_session,
        "is_market_closed": is_market_closed,
        "is_holiday_schedule": bool(
            current_session.get("isHolidaySchedule")
            or session.get("isHolidaySchedule")
            or ta_data.get("market_is_holiday_schedule")
        ),
        "market_status": str(
            current_session.get("marketStatus")
            or session.get("marketStatus")
            or ta_data.get("market_status")
            or ("closed" if is_market_closed else "open")
        ),
        "market_status_label": str(
            current_session.get("marketStatusLabel")
            or session.get("marketStatusLabel")
            or ta_data.get("market_status_label")
            or ("Market Closed" if is_market_closed else "Market Open")
        ),
        "current_label": str(
            session.get("currentLabel")
            or current_session.get("label")
            or (current_session.get("marketStatusLabel") if is_market_closed else "Off")
        ),
        "current_time": str(
            session.get("currentTimeDisplayUtc")
            or current_session.get("timeDisplayUtc")
            or datetime.now(timezone.utc).strftime("%H:%M UTC")
        ),
        "next_open": str(
            current_session.get("nextOpenTimeDisplayUtc")
            or session.get("nextOpenTimeDisplayUtc")
            or ta_data.get("market_next_open_display_utc")
            or current_session.get("nextOpenUtc")
            or ta_data.get("market_next_open_utc")
            or ""
        ).strip(),
        "closed_reason": str(
            current_session.get("closedReason")
            or session.get("closedReason")
            or ta_data.get("market_closed_reason")
            or "XAUUSD market is currently closed."
        ).strip(),
        "holiday_name": str(
            current_session.get("holidayName")
            or session.get("holidayName")
            or ta_data.get("market_holiday_name")
            or ""
        ).strip(),
        "holiday_note": str(
            current_session.get("holidayScheduleNote")
            or session.get("holidayScheduleNote")
            or ta_data.get("market_holiday_schedule_note")
            or ""
        ).strip(),
    }


def _format_bar_session_microstructure(ta_data, vwap_bias_override=None):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    market_session = _extract_market_session_state(ta_data)
    session = market_session["session"]
    bar_session = market_session["bar_session"]
    structure = ta_data.get("structure_context") if isinstance(ta_data.get("structure_context"), dict) else {}

    bar_label = str(session.get("label") or bar_session.get("label") or "Off")
    bar_time = str(session.get("barTimeDisplayUtc") or bar_session.get("timeDisplayUtc") or "N/A")
    current_label = market_session["current_label"]
    current_time = market_session["current_time"]
    same_session_label = bool(
        bar_label.strip()
        and current_label.strip()
        and bar_label.strip().lower() == current_label.strip().lower()
    )

    if market_session["is_market_closed"]:
        reopen_text = f" · Reopens {market_session['next_open']}" if market_session["next_open"] else ""
        if same_session_label:
            return (
                f"Session {bar_label} · Last bar {bar_time} · Now {current_time}"
                f"{reopen_text} · Microstructure is frozen until reopen"
            )
        return (
            f"Last bar {bar_label} {bar_time} · Now {current_label} {current_time}"
            f"{reopen_text} · Microstructure is frozen until reopen"
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

    vwap_text = _format_vwap_microstructure_text(vwap_delta, vwap_bias_override)
    orb_text = f"ORB {_orb_state_label(orb)}"
    sweep_text = "Sweep Bullish" if sweep > 0 else ("Sweep Bearish" if sweep < 0 else "No Sweep")

    if same_session_label:
        summary = (
            f"Session {bar_label} · Bar {bar_time} · Now {current_time} "
            f"· {vwap_text} · {orb_text} · {sweep_text}"
        )
    else:
        summary = (
            f"Bar {bar_label} {bar_time} · Now {current_label} {current_time} "
            f"· {vwap_text} · {orb_text} · {sweep_text}"
        )
    if market_session["is_holiday_schedule"] and market_session["holiday_name"]:
        summary += f" · Holiday schedule {market_session['holiday_name']}"
    return summary


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


def _round_level(value):
    numeric = _coerce_float(value, None)
    return round(numeric, 2) if numeric is not None else None


def _format_level_text(value):
    numeric = _round_level(value)
    return f"{numeric:.2f}" if numeric is not None else "N/A"


def _append_level(levels, label, price):
    numeric = _coerce_float(price, None)
    if numeric is None or numeric <= 0:
        return
    label_text = str(label or "Level").strip() or "Level"
    for item in levels:
        if abs(float(item["price"]) - numeric) <= 0.05:
            existing_label = str(item.get("label") or "")
            if label_text not in existing_label:
                item["label"] = f"{existing_label} / {label_text}".strip(" /")
            return
    levels.append({"label": label_text, "price": numeric})


def _nearest_above(levels, price, min_distance=0.0):
    current = _coerce_float(price, 0.0) or 0.0
    candidates = [
        item
        for item in levels
        if _coerce_float(item.get("price"), None) is not None
        and float(item["price"]) > current + min_distance
    ]
    return min(candidates, key=lambda item: float(item["price"])) if candidates else None


def _nearest_below(levels, price, min_distance=0.0):
    current = _coerce_float(price, 0.0) or 0.0
    candidates = [
        item
        for item in levels
        if _coerce_float(item.get("price"), None) is not None
        and float(item["price"]) < current - min_distance
    ]
    return max(candidates, key=lambda item: float(item["price"])) if candidates else None


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


def _execution_quality_grade(score, risk_reward, blockers):
    if blockers:
        return "No Trade"
    if score >= 82 and risk_reward >= 1.5:
        return "A"
    if score >= 70 and risk_reward >= 1.35:
        return "B"
    if score >= 58 and risk_reward >= 1.15:
        return "C"
    return "No Trade"


def _execution_quality_status(grade, blockers):
    if blockers or grade == "No Trade":
        return "blocked"
    if grade in {"A", "B"}:
        return "ready"
    return "watchlist"


def _is_hard_execution_quality_blocker(blocker):
    text = str(blocker or "").strip().lower()
    return bool(
        text.startswith("market is closed")
        or text.startswith("macro event window")
    )


def _execution_quality_signal_category(signal):
    text = str(signal or "").strip()
    if not text:
        return "no_trade"
    if text.startswith("No Trade hard blocked:"):
        return "hard_blocked"
    if text.startswith("No Trade"):
        return "no_trade"
    if text.startswith(("A ", "B ")):
        return "ready"
    if text.startswith("Watchlist "):
        return "watchlist"
    return "other"


def _execution_quality_signal_core(signal):
    text = str(signal or "").strip()
    if not text:
        return "No Trade"
    return re.sub(r"\s+(?:>=|<)\d+(?:\.\d+)?R$", "", text).strip() or text


def _is_material_execution_quality_transition(previous, current):
    previous_text = str(previous or "").strip()
    current_text = str(current or "").strip()
    if previous_text == current_text:
        return False
    previous_category = _execution_quality_signal_category(previous_text)
    current_category = _execution_quality_signal_category(current_text)
    if previous_category == "no_trade" and current_category == "no_trade":
        return False
    if previous_category == "hard_blocked" and current_category in {"hard_blocked", "no_trade"}:
        return False
    if (
        previous_category in {"ready", "watchlist"}
        and current_category in {"ready", "watchlist"}
        and _execution_quality_signal_core(previous_text) == _execution_quality_signal_core(current_text)
    ):
        return False
    return True


def _stabilize_execution_quality_alert_signal(raw_signal, previous_snapshot=None):
    previous_snapshot = previous_snapshot if isinstance(previous_snapshot, dict) else {}
    raw_signal = str(raw_signal or "No Trade").strip() or "No Trade"
    if not previous_snapshot:
        return raw_signal, raw_signal, 1

    previous_raw = str(previous_snapshot.get("execution_quality_raw_signal") or "").strip()
    previous_streak = int(previous_snapshot.get("execution_quality_raw_streak", 0) or 0)
    raw_streak = (
        previous_streak + 1
        if _execution_quality_signal_core(raw_signal) == _execution_quality_signal_core(previous_raw)
        else 1
    )
    previous_stable = (
        str(previous_snapshot.get("execution_quality_signal") or "").strip()
        or raw_signal
    )

    raw_category = _execution_quality_signal_category(raw_signal)
    stable_category = _execution_quality_signal_category(previous_stable)
    stable_signal = previous_stable

    if raw_signal == previous_stable:
        stable_signal = raw_signal
    elif raw_category == "hard_blocked":
        stable_signal = raw_signal
    elif raw_category in {"ready", "watchlist"}:
        if raw_streak >= EXECUTION_QUALITY_ALERT_CONFIRMATION_COUNT:
            stable_signal = raw_signal
    elif raw_category == "no_trade" and stable_category in {"ready", "watchlist"}:
        if raw_streak >= EXECUTION_QUALITY_CLEAR_CONFIRMATION_COUNT:
            stable_signal = raw_signal
    else:
        stable_signal = raw_signal

    return stable_signal, raw_signal, raw_streak


def _atr_status(atr_percent):
    value = _coerce_float(atr_percent, 0.0) or 0.0
    if value >= 0.55:
        return "Extreme"
    if value >= 0.30:
        return "Expanded"
    if value <= 0.12:
        return "Compressed"
    return "Normal"


def _execution_mode(adx, atr_percent, market_regime):
    adx_value = _coerce_float(adx, 0.0) or 0.0
    regime_text = str(market_regime or "")
    if adx_value >= 25.0 or "Trending" in regime_text:
        return "Trend Continuation"
    if adx_value <= 20.0 or "Range" in regime_text:
        return "Range Reversion"
    return "Transition"


def _execution_stop_profile(adx, atr_percent, mode):
    adx_value = _coerce_float(adx, 0.0) or 0.0
    atr_value = _coerce_float(atr_percent, 0.0) or 0.0
    mode_text = str(mode or "")

    if adx_value >= 35.0:
        return {
            "min_stop_atr": 1.5,
            "max_entry_stop_atr": 2.0,
            "management": "ADX > 35: trail winners with 2.0x ATR or VWAP +/- 0.5x ATR after TP1.",
        }
    if adx_value >= 25.0 or mode_text == "Trend Continuation":
        return {
            "min_stop_atr": 1.25,
            "max_entry_stop_atr": 1.5,
            "management": "ADX trend: use fixed targets; switch to trailing only if ADX clears 35.",
        }
    if atr_value >= 0.25:
        return {
            "min_stop_atr": 1.15,
            "max_entry_stop_atr": 1.5,
            "management": "Elevated ATR: avoid tight stops and reduce size instead of shrinking risk room.",
        }
    return {
        "min_stop_atr": 1.0,
        "max_entry_stop_atr": 1.5,
        "management": "Normal volatility: protect with about 1.0x ATR; fixed TP works until ADX > 35.",
    }


def _collect_execution_levels(ta_data):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    structure = ta_data.get("structure_context") if isinstance(ta_data.get("structure_context"), dict) else {}
    support_resistance = ta_data.get("support_resistance") if isinstance(ta_data.get("support_resistance"), dict) else {}
    pivot_levels = support_resistance.get("pivot_levels") if isinstance(support_resistance.get("pivot_levels"), dict) else {}

    levels = []
    for label, key, fallback_key in (
        ("VWAP", "sessionVwap", None),
        ("PP", "pivotPoint", "pp"),
        ("R1", "pivotResistance1", "r1"),
        ("S1", "pivotSupport1", "s1"),
        ("R2", "pivotResistance2", "r2"),
        ("S2", "pivotSupport2", "s2"),
    ):
        _append_level(
            levels,
            label,
            _first_non_empty(
                structure.get(key),
                pivot_levels.get(fallback_key) if fallback_key else None,
            ),
        )

    nearest_support = support_resistance.get("nearest_support")
    nearest_resistance = support_resistance.get("nearest_resistance")
    if isinstance(nearest_support, dict):
        _append_level(levels, nearest_support.get("label") or "Nearest Support", nearest_support.get("price"))
    if isinstance(nearest_resistance, dict):
        _append_level(levels, nearest_resistance.get("label") or "Nearest Resistance", nearest_resistance.get("price"))

    nearby_supports = (
        support_resistance.get("nearby_supports")
        if isinstance(support_resistance.get("nearby_supports"), list)
        else []
    )
    nearby_resistances = (
        support_resistance.get("nearby_resistances")
        if isinstance(support_resistance.get("nearby_resistances"), list)
        else []
    )
    for item in nearby_supports:
        if isinstance(item, dict):
            _append_level(levels, item.get("label") or item.get("family") or "Support", item.get("price"))
    for item in nearby_resistances:
        if isinstance(item, dict):
            _append_level(levels, item.get("label") or item.get("family") or "Resistance", item.get("price"))

    return sorted(levels, key=lambda item: float(item["price"]))


def _build_streamlined_execution_quality_plan(ta_data, regime_state, decision_status, execution_state):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    execution_state = execution_state if isinstance(execution_state, dict) else {}
    if str(execution_state.get("signalEngineMode") or "") != "streamlined_fixed":
        return None
    action_state = str(execution_state.get("actionState") or "")
    if action_state not in {"LONG_ACTIVE", "SHORT_ACTIVE"}:
        return None

    session = _extract_market_session_state(ta_data)
    event_risk = ta_data.get("event_risk") if isinstance(ta_data.get("event_risk"), dict) else {}
    if session["is_market_closed"] or bool(event_risk.get("active")):
        return None

    current_price = _coerce_float(ta_data.get("current_price"), 0.0) or 0.0
    if current_price <= 0:
        return None

    volatility = ta_data.get("volatility_regime") if isinstance(ta_data.get("volatility_regime"), dict) else {}
    structure_context = ta_data.get("structure_context") if isinstance(ta_data.get("structure_context"), dict) else {}
    price_action = ta_data.get("price_action") if isinstance(ta_data.get("price_action"), dict) else {}
    momentum_features = ta_data.get("momentum_features") if isinstance(ta_data.get("momentum_features"), dict) else {}

    adx = _coerce_float(volatility.get("adx_14"), 0.0) or 0.0
    atr_percent = _coerce_float(volatility.get("atr_percent"), 0.0) or 0.0
    atr_abs = _coerce_float(volatility.get("atr_14"), None)
    if atr_abs is None or atr_abs <= 0:
        atr_abs = (current_price * atr_percent / 100.0) if current_price > 0 and atr_percent > 0 else max(current_price * 0.001, 1.0)
    atr_label = _atr_status(atr_percent)

    direction = "Long" if action_state == "LONG_ACTIVE" else "Short"
    vwap_delta = _coerce_float(structure_context.get("distSessionVwapPct"), 0.0) or 0.0
    orb_state = int(round(_coerce_float(structure_context.get("openingRangeBreak"), 0.0) or 0.0))
    structure_text = str(price_action.get("structure") or "")
    structure_lower = structure_text.lower()
    vwap_threshold = 0.04
    vwap_aligned = (direction == "Long" and vwap_delta >= vwap_threshold) or (
        direction == "Short" and vwap_delta <= -vwap_threshold
    )
    orb_aligned = (direction == "Long" and orb_state > 0) or (direction == "Short" and orb_state < 0)
    structure_aligned = (
        direction == "Long" and any(marker in structure_lower for marker in ("bullish", "breakout", "support rejection"))
    ) or (
        direction == "Short" and any(marker in structure_lower for marker in ("bearish", "breakdown", "rejection"))
    )
    volume_spike = bool(int(round(_coerce_float(momentum_features.get("volumeSpike"), 0.0) or 0.0)))

    if vwap_aligned and orb_aligned:
        setup = f"VWAP / ORB Continuation {direction}"
    elif vwap_aligned:
        setup = f"VWAP Pullback {direction}"
    elif structure_aligned:
        setup = f"Structure Break {direction}"
    else:
        setup = f"Directional Continuation {direction}"

    stop_distance = atr_abs * 1.5
    target_one_distance = atr_abs * 3.0
    target_two_distance = atr_abs * 4.5
    entry_low = current_price - (atr_abs * 0.12)
    entry_high = current_price + (atr_abs * 0.12)
    if direction == "Long":
        stop = current_price - stop_distance
        target_one = current_price + target_one_distance
        target_two = current_price + target_two_distance
    else:
        stop = current_price + stop_distance
        target_one = current_price - target_one_distance
        target_two = current_price - target_two_distance

    score = 72.0
    if structure_aligned:
        score += 8.0
    if vwap_aligned:
        score += 6.0
    if orb_aligned:
        score += 6.0
    if adx >= 25.0:
        score += 6.0
    elif adx >= 20.0:
        score += 4.0
    if volume_spike:
        score += 4.0
    score = min(score, 92.0)
    grade = "A" if score >= 84.0 else "B"

    reasons = [
        f"Streamlined fixed-signal mode keeps confirmed {direction.lower()} setups actionable.",
        "ATR-defined risk model is used instead of nearby-level rejection gating.",
    ]
    if structure_aligned:
        reasons.append(f"{structure_text or direction + ' structure'} remains aligned.")
    if vwap_aligned:
        reasons.append(f"VWAP is aligned on the {direction.lower()} side.")
    if orb_aligned:
        reasons.append(f"ORB confirms the {direction.lower()} continuation.")
    if adx >= 20.0:
        reasons.append(f"ADX {adx:.1f} confirms directional momentum.")

    return {
        "status": "ready",
        "grade": grade,
        "score": round(score, 1),
        "mode": "Streamlined Fixed Signal",
        "atrStatus": atr_label,
        "direction": direction,
        "setup": setup,
        "entry": {
            "low": _round_level(entry_low),
            "high": _round_level(entry_high),
            "mid": _round_level(current_price),
            "text": f"{_format_level_text(entry_low)} - {_format_level_text(entry_high)}",
        },
        "stopLoss": {
            "price": _round_level(stop),
            "basis": "1.5x ATR fixed stop",
            "distance": round(stop_distance, 2),
            "atrMultiple": 1.5,
        },
        "targets": [
            {
                "label": "TP1",
                "price": _round_level(target_one),
                "basis": "3.0x ATR target",
                "rMultiple": 2.0,
            },
            {
                "label": "TP2",
                "price": _round_level(target_two),
                "basis": "4.5x ATR runner target",
                "rMultiple": 3.0,
            },
        ],
        "riskReward": 2.0,
        "positionSizeNote": (
            "Reduced risk / probe only" if atr_label in {"Compressed", "Extreme"} else "Normal planned risk only"
        ),
        "riskManagement": {
            "minStopAtr": 1.5,
            "maxEntryStopAtr": 1.5,
            "stopAtrMultiple": 1.5,
            "trailActive": bool(adx >= 35.0),
            "text": "Use fixed ATR-defined risk and trail only after directional follow-through strengthens.",
        },
        "components": {
            "regime": 24.0 if adx >= 20.0 else 18.0,
            "vwap": 20.0 if vwap_aligned else 12.0,
            "orb": 15.0 if orb_aligned else 7.0,
            "location": 18.0 if structure_aligned else 12.0,
            "risk": 20.0,
        },
        "reasons": _dedupe_preserve_order(reasons),
        "blockers": [],
        "invalidations": (
            [
                f"Long thesis fails below {_format_level_text(stop)} (1.5x ATR fixed stop).",
                "Exit if price loses VWAP and structure flips bearish.",
            ]
            if direction == "Long"
            else [
                f"Short thesis fails above {_format_level_text(stop)} (1.5x ATR fixed stop).",
                "Exit if price reclaims VWAP and structure flips bullish.",
            ]
        ),
    }


def _build_execution_quality_plan(ta_data, regime_state, decision_status, execution_state):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    regime_state = regime_state if isinstance(regime_state, dict) else {}
    decision_status = decision_status if isinstance(decision_status, dict) else {}
    execution_state = execution_state if isinstance(execution_state, dict) else {}

    streamlined_plan = _build_streamlined_execution_quality_plan(
        ta_data,
        regime_state,
        decision_status,
        execution_state,
    )
    if streamlined_plan is not None:
        return streamlined_plan

    current_price = _coerce_float(ta_data.get("current_price"), 0.0) or 0.0
    volatility = ta_data.get("volatility_regime") if isinstance(ta_data.get("volatility_regime"), dict) else {}
    structure = ta_data.get("structure_context") if isinstance(ta_data.get("structure_context"), dict) else {}
    price_action = ta_data.get("price_action") if isinstance(ta_data.get("price_action"), dict) else {}
    session = _extract_market_session_state(ta_data)
    event_risk = ta_data.get("event_risk") if isinstance(ta_data.get("event_risk"), dict) else {}

    adx = _coerce_float(volatility.get("adx_14"), 0.0) or 0.0
    atr_percent = _coerce_float(volatility.get("atr_percent"), 0.0) or 0.0
    atr_abs = _coerce_float(volatility.get("atr_14"), None)
    if atr_abs is None or atr_abs <= 0:
        atr_abs = (current_price * atr_percent / 100.0) if current_price > 0 and atr_percent > 0 else max(current_price * 0.001, 1.0)
    atr_label = _atr_status(atr_percent)
    mode = _execution_mode(adx, atr_percent, volatility.get("market_regime"))
    stop_profile = _execution_stop_profile(adx, atr_percent, mode)
    min_stop_atr = _coerce_float(stop_profile.get("min_stop_atr"), 1.0) or 1.0
    max_entry_stop_atr = _coerce_float(stop_profile.get("max_entry_stop_atr"), 1.5) or 1.5
    min_stop_distance = atr_abs * min_stop_atr
    max_entry_stop_distance = atr_abs * max_entry_stop_atr

    vwap_delta = _coerce_float(structure.get("distSessionVwapPct"), 0.0) or 0.0
    vwap_bias = _vwap_bias_label(vwap_delta)
    orb_state = int(round(_coerce_float(structure.get("openingRangeBreak"), 0.0) or 0.0))
    pivot_point = _coerce_float(structure.get("pivotPoint"), None)
    breakout_bias = str(regime_state.get("breakout_bias") or "Neutral")
    structure_text = str(price_action.get("structure") or "Consolidating")
    decision_kind = str(decision_status.get("status") or "wait")
    action_state = str(execution_state.get("actionState") or "WAIT")

    long_score = 0.0
    short_score = 0.0
    reasons = []
    blockers = []

    if vwap_bias == "Bullish":
        long_score += 3.0
    elif vwap_bias == "Mild Bullish":
        long_score += 2.0
    elif vwap_bias == "Bearish":
        short_score += 3.0
    elif vwap_bias == "Mild Bearish":
        short_score += 2.0

    if orb_state > 0:
        long_score += 2.5
    elif orb_state < 0:
        short_score += 2.5

    if pivot_point is not None and current_price > 0:
        if current_price >= pivot_point:
            long_score += 1.5
        else:
            short_score += 1.5

    if "Bullish" in structure_text:
        long_score += 2.0
    if "Bearish" in structure_text:
        short_score += 2.0
    if breakout_bias == "Bullish":
        long_score += 1.5
    elif breakout_bias == "Bearish":
        short_score += 1.5
    if decision_kind == "buy" or action_state == "LONG_ACTIVE":
        long_score += 2.0
    elif decision_kind == "sell" or action_state == "SHORT_ACTIVE":
        short_score += 2.0

    direction = "Neutral"
    if mode == "Range Reversion" and vwap_bias in {"Bullish", "Bearish"}:
        direction = "Short" if vwap_bias == "Bullish" else "Long"
        reasons.append("Range mode uses VWAP extension for mean reversion.")
    elif long_score >= short_score + 1.5:
        direction = "Long"
    elif short_score >= long_score + 1.5:
        direction = "Short"

    if session["is_market_closed"]:
        blockers.append("Market is closed; wait for a live reopen.")
    if bool(event_risk.get("active")):
        blockers.append("Macro event window is active; fresh entries are blocked.")
    if atr_label == "Extreme":
        blockers.append("ATR% is extreme; reduce risk or skip until volatility normalizes.")
    if direction == "Neutral":
        blockers.append("VWAP, ORB, pivots, and structure are not aligned enough.")

    if direction == "Long":
        if mode == "Trend Continuation":
            setup = "VWAP / ORB Continuation Long" if orb_state > 0 else "VWAP Pullback Long"
        elif mode == "Range Reversion":
            setup = "VWAP Stretch Reversion Long"
        else:
            setup = "Long Reclaim Setup"
    elif direction == "Short":
        if mode == "Trend Continuation":
            setup = "VWAP / ORB Continuation Short" if orb_state < 0 else "VWAP Pullback Short"
        elif mode == "Range Reversion":
            setup = "VWAP Stretch Reversion Short"
        else:
            setup = "Short Rejection Setup"
    else:
        setup = "No Clean Entry"

    levels = _collect_execution_levels(ta_data)
    vwap_target_price = _coerce_float(structure.get("sessionVwap"), None)
    buffer = max(atr_abs * 0.15, current_price * 0.00015 if current_price > 0 else 0.75)
    entry_mid = current_price
    entry_low = None
    entry_high = None
    stop = None
    target_one = None
    target_two = None
    stop_basis = ""
    target_one_basis = ""
    target_two_basis = ""

    if current_price > 0 and direction == "Long":
        support_anchor = _nearest_below(levels, current_price, min_distance=0.02)
        stop_anchor_price = float(support_anchor["price"]) if support_anchor else current_price - atr_abs * 0.55
        stop_basis = str(support_anchor.get("label") if support_anchor else "ATR fallback support")
        stop = stop_anchor_price - buffer
        raw_risk = max(entry_mid - stop, 0.0)
        if raw_risk < min_stop_distance:
            stop = entry_mid - min_stop_distance
        risk = max(entry_mid - stop, min_stop_distance)
        entry_low = max(stop + risk * 0.25, entry_mid - atr_abs * 0.12)
        entry_high = entry_mid + atr_abs * 0.05
        nearby_resistance = _nearest_above(levels, current_price, min_distance=0.02)
        if nearby_resistance and float(nearby_resistance["price"]) - entry_mid < risk * 0.8:
            blockers.append(f"{nearby_resistance['label']} is too close for a clean long target.")
        if mode == "Range Reversion" and vwap_target_price is not None and vwap_target_price > entry_mid + 0.02:
            target_one = vwap_target_price
            target_one_basis = "VWAP mean reversion target"
            target_two_level = _nearest_above(levels, target_one, min_distance=0.05)
        else:
            target_one_level = _nearest_above(levels, entry_mid + risk, min_distance=0.02)
            target_two_level = (
                _nearest_above(levels, float(target_one_level["price"]), min_distance=0.05)
                if target_one_level
                else _nearest_above(levels, entry_mid + risk * 1.8, min_distance=0.02)
            )
            target_one = float(target_one_level["price"]) if target_one_level else entry_mid + risk * 1.5
            target_one_basis = str(target_one_level.get("label") if target_one_level else "1.5R volatility target")
        target_two = float(target_two_level["price"]) if target_two_level else max(entry_mid + risk * 2.2, target_one + risk * 0.7)
        target_two_basis = str(target_two_level.get("label") if target_two_level else "2.2R runner target")
    elif current_price > 0 and direction == "Short":
        resistance_anchor = _nearest_above(levels, current_price, min_distance=0.02)
        stop_anchor_price = float(resistance_anchor["price"]) if resistance_anchor else current_price + atr_abs * 0.55
        stop_basis = str(resistance_anchor.get("label") if resistance_anchor else "ATR fallback resistance")
        stop = stop_anchor_price + buffer
        raw_risk = max(stop - entry_mid, 0.0)
        if raw_risk < min_stop_distance:
            stop = entry_mid + min_stop_distance
        risk = max(stop - entry_mid, min_stop_distance)
        entry_low = entry_mid - atr_abs * 0.05
        entry_high = min(stop - risk * 0.25, entry_mid + atr_abs * 0.12)
        nearby_support = _nearest_below(levels, current_price, min_distance=0.02)
        if nearby_support and entry_mid - float(nearby_support["price"]) < risk * 0.8:
            blockers.append(f"{nearby_support['label']} is too close for a clean short target.")
        if mode == "Range Reversion" and vwap_target_price is not None and vwap_target_price < entry_mid - 0.02:
            target_one = vwap_target_price
            target_one_basis = "VWAP mean reversion target"
            target_two_level = _nearest_below(levels, target_one, min_distance=0.05)
        else:
            target_one_level = _nearest_below(levels, entry_mid - risk, min_distance=0.02)
            target_two_level = (
                _nearest_below(levels, float(target_one_level["price"]), min_distance=0.05)
                if target_one_level
                else _nearest_below(levels, entry_mid - risk * 1.8, min_distance=0.02)
            )
            target_one = float(target_one_level["price"]) if target_one_level else entry_mid - risk * 1.5
            target_one_basis = str(target_one_level.get("label") if target_one_level else "1.5R volatility target")
        target_two = float(target_two_level["price"]) if target_two_level else min(entry_mid - risk * 2.2, target_one - risk * 0.7)
        target_two_basis = str(target_two_level.get("label") if target_two_level else "2.2R runner target")
    else:
        risk = 0.0

    if direction in {"Long", "Short"} and risk > max_entry_stop_distance:
        blockers.append(
            f"Planned stop is wider than {max_entry_stop_atr:.1f}x ATR; wait for a better pullback."
        )

    risk_reward = 0.0
    if stop is not None and target_one is not None and risk > 0:
        if direction == "Long":
            risk_reward = max((target_one - entry_mid) / risk, 0.0)
        elif direction == "Short":
            risk_reward = max((entry_mid - target_one) / risk, 0.0)
    if direction in {"Long", "Short"} and risk_reward < 1.15:
        blockers.append("Reward-to-risk is below 1.15R after nearby levels and ATR stop.")

    regime_component = 24.0 if mode == "Trend Continuation" and direction in {"Long", "Short"} else 18.0 if mode == "Range Reversion" and direction in {"Long", "Short"} else 10.0
    vwap_component = 20.0 if (direction == "Long" and vwap_bias == "Bullish") or (direction == "Short" and vwap_bias == "Bearish") else 15.0 if (direction == "Long" and "Bullish" in vwap_bias) or (direction == "Short" and "Bearish" in vwap_bias) else 8.0 if vwap_bias == "Neutral" else 2.0
    orb_component = 15.0 if (direction == "Long" and orb_state > 0) or (direction == "Short" and orb_state < 0) else 7.0 if orb_state == 0 else 1.0
    location_component = 20.0 if (direction == "Long" and pivot_point is not None and current_price >= pivot_point) or (direction == "Short" and pivot_point is not None and current_price <= pivot_point) else 11.0 if pivot_point is None else 4.0
    risk_component = min(max((risk_reward / 2.0) * 20.0, 0.0), 20.0)
    score = round(min(regime_component + vwap_component + orb_component + location_component + risk_component, 100.0), 1)
    grade = _execution_quality_grade(score, risk_reward, blockers)
    status = _execution_quality_status(grade, blockers)

    if adx >= 25:
        reasons.append(f"ADX {adx:.1f} supports trend-following mode.")
    elif adx <= 20:
        reasons.append(f"ADX {adx:.1f} favors range/reversion entries.")
    else:
        reasons.append(f"ADX {adx:.1f} is in the transition zone.")
    reasons.append(f"VWAP is {vwap_delta:.2f}% from price bias: {vwap_bias}.")
    reasons.append(f"ORB is {_orb_state_label(orb_state)} and price is {'above' if pivot_point is not None and current_price >= pivot_point else 'below' if pivot_point is not None else 'near'} PP.")
    if direction in {"Long", "Short"} and risk:
        reasons.append(
            f"Stop uses {risk / atr_abs:.2f}x ATR room; entry timing is poor above {max_entry_stop_atr:.1f}x ATR."
        )
    reasons.append(str(stop_profile.get("management") or ""))
    reasons = _dedupe_preserve_order(reasons)
    blockers = _dedupe_preserve_order(blockers)

    return {
        "status": status,
        "grade": grade,
        "score": score,
        "mode": mode,
        "atrStatus": atr_label,
        "direction": direction,
        "setup": setup,
        "entry": {
            "low": _round_level(entry_low),
            "high": _round_level(entry_high),
            "mid": _round_level(entry_mid) if current_price > 0 else None,
            "text": (
                f"{_format_level_text(entry_low)} - {_format_level_text(entry_high)}"
                if entry_low is not None and entry_high is not None
                else "Wait for a cleaner location"
            ),
        },
        "stopLoss": {
            "price": _round_level(stop),
            "basis": stop_basis or "No structural stop available",
            "distance": round(risk, 2) if risk else None,
            "atrMultiple": round(risk / atr_abs, 2) if risk and atr_abs else None,
        },
        "targets": [
            {
                "label": "TP1",
                "price": _round_level(target_one),
                "basis": target_one_basis or "No target available",
                "rMultiple": round(risk_reward, 2) if risk_reward else None,
            },
            {
                "label": "TP2",
                "price": _round_level(target_two),
                "basis": target_two_basis or "No runner target available",
                "rMultiple": round(abs((target_two or entry_mid) - entry_mid) / risk, 2) if target_two is not None and risk else None,
            },
        ],
        "riskReward": round(risk_reward, 2),
        "positionSizeNote": (
            "Normal planned risk only" if status == "ready" and atr_label in {"Normal", "Expanded"}
            else "Reduced risk / probe only" if status == "watchlist" or atr_label in {"Compressed", "Extreme"}
            else "Skip until blockers clear"
        ),
        "riskManagement": {
            "minStopAtr": round(min_stop_atr, 2),
            "maxEntryStopAtr": round(max_entry_stop_atr, 2),
            "stopAtrMultiple": round(risk / atr_abs, 2) if risk and atr_abs else None,
            "trailActive": bool(adx >= 35.0),
            "text": str(stop_profile.get("management") or ""),
        },
        "components": {
            "regime": round(regime_component, 1),
            "vwap": round(vwap_component, 1),
            "orb": round(orb_component, 1),
            "location": round(location_component, 1),
            "risk": round(risk_component, 1),
        },
        "reasons": reasons,
        "blockers": blockers,
        "invalidations": (
            [
                f"Long thesis fails below {_format_level_text(stop)} ({stop_basis}).",
                "Exit if price loses VWAP and ORB flips bearish.",
            ]
            if direction == "Long" and stop is not None
            else [
                f"Short thesis fails above {_format_level_text(stop)} ({stop_basis}).",
                "Exit if price reclaims VWAP and ORB flips bullish.",
            ]
            if direction == "Short" and stop is not None
            else ["No trade until VWAP, ORB, pivots, and ADX align."]
        ),
    }


def _execution_quality_alert_signal(execution_quality):
    quality = execution_quality if isinstance(execution_quality, dict) else {}
    grade = str(quality.get("grade") or "No Trade").strip() or "No Trade"
    status = str(quality.get("status") or "blocked").strip() or "blocked"
    direction = str(quality.get("direction") or "Neutral").strip() or "Neutral"
    setup = str(quality.get("setup") or "No Clean Entry").strip() or "No Clean Entry"
    blockers = quality.get("blockers") if isinstance(quality.get("blockers"), list) else []
    risk_reward = _coerce_float(quality.get("riskReward"), 0.0) or 0.0
    if risk_reward >= 2.5:
        rr_bucket = ">=2.5R"
    elif risk_reward >= 2.0:
        rr_bucket = ">=2.0R"
    elif risk_reward >= 1.5:
        rr_bucket = ">=1.5R"
    elif risk_reward >= 1.15:
        rr_bucket = ">=1.15R"
    else:
        rr_bucket = "<1.15R"
    setup_label = setup
    if direction in {"Long", "Short"}:
        direction_prefix = f"{direction} "
        direction_suffix = f" {direction}"
        if not (setup.startswith(direction_prefix) or setup.endswith(direction_suffix)):
            setup_label = f"{direction} {setup}"

    if status == "ready" and grade in {"A", "B"} and direction in {"Long", "Short"}:
        return f"{grade} {setup_label} {rr_bucket}"
    if status == "watchlist" and grade == "C" and direction in {"Long", "Short"}:
        return f"Watchlist {setup_label} {rr_bucket}"
    if blockers:
        hard_blocker = next(
            (str(item).strip() for item in blockers if _is_hard_execution_quality_blocker(item)),
            "",
        )
        if hard_blocker:
            return f"No Trade hard blocked: {hard_blocker[:90]}"
        return "No Trade"
    return "No Trade"


def _utc_iso_timestamp(ts):
    try:
        ts_value = int(float(ts or 0))
    except (TypeError, ValueError):
        ts_value = 0
    if ts_value <= 0:
        return None
    return (
        datetime.fromtimestamp(ts_value, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_utc_timestamp(value, default=0):
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip()
    if not text:
        return int(default or 0)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return int(datetime.fromisoformat(normalized).timestamp())
    except ValueError:
        return int(default or 0)


def _stable_decision_bucket_ts(now_ts):
    now_ts = int(now_ts or time.time())
    bucket = max(STABLE_DECISION_BUFFER_BUCKET_SECONDS, 1)
    return (now_ts // bucket) * bucket


def _stable_decision_state_rank(state_name):
    return {
        "SCANNING": 0,
        "INVALIDATED": 0,
        "EXITED": 0,
        "CANDIDATE": 1,
        "CONFIRMED": 2,
        "ACTIVE": 3,
    }.get(str(state_name or "SCANNING"), 0)


def _stable_decision_quality_status(decision_state):
    state_name = str(decision_state or "SCANNING")
    if state_name == "CANDIDATE":
        return "watchlist"
    if state_name in {"CONFIRMED", "ACTIVE"}:
        return "ready"
    return "blocked"


def _stable_decision_session_label(ta_data):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    session_context = ta_data.get("session_context") if isinstance(ta_data.get("session_context"), dict) else {}
    return str(
        session_context.get("currentLabel")
        or session_context.get("label")
        or session_context.get("bar_session")
        or "Unknown"
    )


def _stable_decision_bar_timestamp(ta_data, now_ts):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    session_context = ta_data.get("session_context") if isinstance(ta_data.get("session_context"), dict) else {}
    return str(
        ta_data.get("bar_timestamp_utc")
        or session_context.get("barTimestampUtc")
        or session_context.get("currentTimestampUtc")
        or _utc_iso_timestamp(now_ts)
        or ""
    )


def _stable_decision_signal_setup(execution_quality):
    execution_quality = execution_quality if isinstance(execution_quality, dict) else {}
    signal = _execution_quality_alert_signal(execution_quality)
    if signal.startswith("Watchlist "):
        return signal[len("Watchlist "):].strip() or "No Clean Entry"
    parts = signal.split(" ", 1)
    if parts and parts[0] in {"A", "B", "C"} and len(parts) == 2:
        return parts[1].strip() or "No Clean Entry"
    setup = str(execution_quality.get("setup") or "No Clean Entry").strip() or "No Clean Entry"
    risk_reward = _coerce_float(execution_quality.get("riskReward"), 0.0) or 0.0
    if signal.startswith("No Trade") and risk_reward >= 1.15:
        if risk_reward >= 2.5:
            rr_bucket = ">=2.5R"
        elif risk_reward >= 2.0:
            rr_bucket = ">=2.0R"
        elif risk_reward >= 1.5:
            rr_bucket = ">=1.5R"
        else:
            rr_bucket = ">=1.15R"
        if rr_bucket not in setup:
            setup = f"{setup} {rr_bucket}"
    return setup


def _stable_decision_direction(prediction, decision_status, execution_quality):
    execution_quality = execution_quality if isinstance(execution_quality, dict) else {}
    decision_status = decision_status if isinstance(decision_status, dict) else {}
    direction = str(execution_quality.get("direction") or "").strip()
    if direction in {"Long", "Short"}:
        return direction
    status = str(decision_status.get("status") or "").strip().lower()
    if status == "buy":
        return "Long"
    if status == "sell":
        return "Short"
    verdict = str((prediction or {}).get("verdict") or "").strip()
    if verdict == "Bullish":
        return "Long"
    if verdict == "Bearish":
        return "Short"
    return "Neutral"


def _stable_decision_structure_opposes(direction, market_structure):
    structure_text = str(market_structure or "").strip().lower()
    if direction == "Long":
        return any(
            token in structure_text
            for token in ("bearish", "breakdown", "rejection", "lower high", "distribution")
        )
    if direction == "Short":
        return any(
            token in structure_text
            for token in ("bullish", "breakout", "reclaim", "higher low", "accumulation")
        )
    return False


def _stable_decision_reason(previous_decision, raw_profile):
    previous_decision = previous_decision if isinstance(previous_decision, dict) else {}
    raw_profile = raw_profile if isinstance(raw_profile, dict) else {}
    previous_session = str(previous_decision.get("session") or "").strip()
    current_session = str(raw_profile.get("session") or "").strip()
    if (
        previous_session
        and previous_session != "Unknown"
        and current_session
        and previous_session != current_session
    ):
        return "SESSION_CHANGE"

    previous_direction = str(previous_decision.get("direction") or "Neutral")
    current_direction = str(raw_profile.get("direction") or "Neutral")
    if previous_direction in {"Long", "Short"} and current_direction in {"Long", "Short"}:
        if previous_direction != current_direction:
            return "STRUCTURE_BREAK"
    if _stable_decision_structure_opposes(previous_direction, raw_profile.get("market_structure")):
        return "STRUCTURE_BREAK"
    return "MOMENTUM_SHIFT"


def _stable_decision_target_state(previous_decision, raw_profile):
    previous_decision = previous_decision if isinstance(previous_decision, dict) else {}
    raw_profile = raw_profile if isinstance(raw_profile, dict) else {}
    previous_state = str(previous_decision.get("decision_state") or "SCANNING")
    direction = str(raw_profile.get("direction") or "Neutral")
    confidence = int(raw_profile.get("confidence") or 0)
    quality_grade = str(raw_profile.get("execution_quality") or "No Trade")
    execution_status = str(raw_profile.get("execution_status") or "stand_aside")
    action_state = str(raw_profile.get("action_state") or "WAIT")

    if action_state in {"LONG_ACTIVE", "SHORT_ACTIVE"} or execution_status == "hold":
        return "ACTIVE"

    actionable = direction in {"Long", "Short"} and quality_grade in {"A", "B", "C"}
    if not actionable or confidence < 60:
        if previous_state in {"CONFIRMED", "ACTIVE", "CANDIDATE"}:
            return "INVALIDATED"
        return "SCANNING"
    if confidence > 90:
        return "ACTIVE"
    if confidence >= 75:
        return "CONFIRMED"
    return "CANDIDATE"


def _stable_decision_default_payload(now_ts=None, session="Unknown", bar_timestamp=None):
    now_ts = int(now_ts or time.time())
    lock_until = _utc_iso_timestamp(now_ts)
    next_evaluation = _utc_iso_timestamp(now_ts + STABLE_DECISION_BUFFER_BUCKET_SECONDS)
    bar_value = str(bar_timestamp or _utc_iso_timestamp(now_ts) or "")
    return {
        "symbol": "XAUUSD",
        "decision_state": "SCANNING",
        "setup_type": "No Clean Entry",
        "execution_quality": "No Trade",
        "confidence": 0,
        "decision_locked_until": lock_until,
        "flip_count_10m": 0,
        "change_reason": "TIME_EXPIRED",
        "session": str(session or "Unknown"),
        "bar_timestamp": bar_value,
        "next_evaluation": next_evaluation,
        "signature": "Neutral|No Clean Entry|No Trade",
        "signal_signature": "Neutral|No Clean Entry|No Trade",
        "direction": "Neutral",
        "suppression_reason": "",
        "hold_reason": "",
        "locked": False,
    }


def _default_stable_decision_state(now_ts=None):
    now_ts = int(now_ts or time.time())
    return {
        "version": 1,
        "observation_buffer": [],
        "signal_buffer": [],
        "decision_ring": [],
        "flip_history": [],
        "decision_lock_until_ts": 0,
        "last_decision_time_ts": 0,
        "last_evaluation_bucket_ts": 0,
        "last_suppression_reason": "",
        "last_hold_reason": "",
        "stable_decision": _stable_decision_default_payload(now_ts=now_ts),
        "recent_churn": [],
    }


def _reset_stable_decision_state_if_stale(state, ta_data, now_ts=None):
    now_ts = int(now_ts or time.time())
    state = copy.deepcopy(state if isinstance(state, dict) else _default_stable_decision_state(now_ts))
    stable_decision = state.get("stable_decision") if isinstance(state.get("stable_decision"), dict) else {}
    last_decision_time_ts = int(state.get("last_decision_time_ts", 0) or 0)
    if last_decision_time_ts and (now_ts - last_decision_time_ts) > STABLE_DECISION_FLIP_WINDOW_SECONDS:
        return _default_stable_decision_state(now_ts)

    current_bar_ts = _parse_utc_timestamp(_stable_decision_bar_timestamp(ta_data, now_ts), default=0)
    stored_bar_ts = _parse_utc_timestamp(stable_decision.get("bar_timestamp"), default=0)
    if current_bar_ts and stored_bar_ts:
        stale_gap_seconds = max(STABLE_DECISION_BUFFER_WINDOW_SECONDS * 2, 600)
        if abs(current_bar_ts - stored_bar_ts) > stale_gap_seconds:
            return _default_stable_decision_state(now_ts)

    return state


def _stable_decision_observation(ta_data, regime_state, now_ts):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    structure_context = ta_data.get("structure_context") if isinstance(ta_data.get("structure_context"), dict) else {}
    volatility_regime = ta_data.get("volatility_regime") if isinstance(ta_data.get("volatility_regime"), dict) else {}
    price_action = ta_data.get("price_action") if isinstance(ta_data.get("price_action"), dict) else {}
    return {
        "bucket_ts": _stable_decision_bucket_ts(now_ts),
        "ts": int(now_ts),
        "bar_timestamp": _stable_decision_bar_timestamp(ta_data, now_ts),
        "session": _stable_decision_session_label(ta_data),
        "adx": round(_coerce_float(volatility_regime.get("adx_14"), 0.0) or 0.0, 4),
        "vwap_delta": round(_coerce_float(structure_context.get("distSessionVwapPct"), 0.0) or 0.0, 4),
        "orb_state": int(round(_coerce_float(structure_context.get("openingRangeBreak"), 0.0) or 0.0)),
        "market_structure": str(price_action.get("structure") or "Consolidating"),
        "event_regime": str((regime_state or {}).get("event_regime") or "normal"),
    }


def _stable_decision_prune_buffer(items, now_ts):
    now_ts = int(now_ts or time.time())
    items = list(items or [])
    minimum_ts = now_ts - STABLE_DECISION_BUFFER_WINDOW_SECONDS
    recent_items = [item for item in items if int(item.get("ts", now_ts) or now_ts) >= minimum_ts]
    return recent_items[-STABLE_DECISION_BUFFER_MAX_BARS:]


def _stable_decision_aggregate_observations(observation_buffer):
    observation_buffer = list(observation_buffer or [])
    if not observation_buffer:
        return {"adx": 0.0, "vwap_delta": 0.0, "orb_state": 0}
    count = max(len(observation_buffer), 1)
    adx = sum(_coerce_float(item.get("adx"), 0.0) or 0.0 for item in observation_buffer) / count
    vwap_delta = sum(_coerce_float(item.get("vwap_delta"), 0.0) or 0.0 for item in observation_buffer) / count
    orb_mean = sum(int(item.get("orb_state", 0) or 0) for item in observation_buffer) / count
    if orb_mean >= 0.4:
        orb_state = 1
    elif orb_mean <= -0.4:
        orb_state = -1
    else:
        orb_state = 0
    return {
        "adx": round(adx, 2),
        "vwap_delta": round(vwap_delta, 4),
        "orb_state": orb_state,
    }


def _update_stable_decision_buffer(state, ta_data, regime_state, now_ts=None):
    now_ts = int(now_ts or time.time())
    state = copy.deepcopy(state if isinstance(state, dict) else _default_stable_decision_state(now_ts))
    observation_buffer = list(state.get("observation_buffer") or [])
    observation = _stable_decision_observation(ta_data, regime_state, now_ts)
    if observation_buffer and int(observation_buffer[-1].get("bucket_ts", -1)) == observation["bucket_ts"]:
        observation_buffer[-1] = observation
    else:
        observation_buffer.append(observation)
    observation_buffer = _stable_decision_prune_buffer(observation_buffer, now_ts)
    state["observation_buffer"] = observation_buffer

    aggregated = _stable_decision_aggregate_observations(observation_buffer)
    buffered_ta = copy.deepcopy(ta_data if isinstance(ta_data, dict) else {})
    volatility_regime = dict(buffered_ta.get("volatility_regime") or {})
    structure_context = dict(buffered_ta.get("structure_context") or {})
    volatility_regime["adx_14"] = aggregated["adx"]
    structure_context["distSessionVwapPct"] = aggregated["vwap_delta"]
    structure_context["openingRangeBreak"] = aggregated["orb_state"]
    buffered_ta["volatility_regime"] = volatility_regime
    buffered_ta["structure_context"] = structure_context
    buffered_ta["decision_buffer"] = {
        "bars": len(observation_buffer),
        "window_seconds": STABLE_DECISION_BUFFER_WINDOW_SECONDS,
        "aggregated": aggregated,
    }
    return state, buffered_ta


def _build_stable_decision_raw_profile(
    prediction,
    ta_data,
    regime_state,
    decision_status,
    execution_state,
    execution_quality,
    recent_flip_count,
    now_ts,
):
    ta_data = ta_data if isinstance(ta_data, dict) else {}
    regime_state = regime_state if isinstance(regime_state, dict) else {}
    decision_status = decision_status if isinstance(decision_status, dict) else {}
    execution_quality = execution_quality if isinstance(execution_quality, dict) else {}
    direction = _stable_decision_direction(prediction, decision_status, execution_quality)
    setup_type = _stable_decision_signal_setup(execution_quality)
    market_structure = str(((ta_data.get("price_action") or {}).get("structure")) or "Consolidating")
    quality_grade = str(execution_quality.get("grade") or "No Trade").strip() or "No Trade"
    score = int(round(_coerce_float(execution_quality.get("score"), 0.0) or 0.0))
    adjusted_confidence = max(
        0,
        min(100, score - (recent_flip_count * STABLE_DECISION_CONFIDENCE_FLIP_PENALTY)),
    )
    raw_profile = {
        "direction": direction,
        "setup_type": setup_type,
        "execution_quality": quality_grade if quality_grade in {"A", "B", "C"} else "No Trade",
        "confidence": adjusted_confidence,
        "session": _stable_decision_session_label(ta_data),
        "bar_timestamp": _stable_decision_bar_timestamp(ta_data, now_ts),
        "market_structure": market_structure,
        "event_regime": str(regime_state.get("event_regime") or "normal"),
        "execution_status": str((execution_state or {}).get("status") or "stand_aside"),
        "action_state": str((execution_state or {}).get("actionState") or "WAIT"),
    }
    raw_profile["signal_signature"] = (
        f"{raw_profile['direction']}|{raw_profile['setup_type']}|{raw_profile['execution_quality']}"
    )
    raw_profile["target_state"] = _stable_decision_target_state({}, raw_profile)
    raw_profile["signature"] = f"{raw_profile['signal_signature']}|{raw_profile['target_state']}"
    return raw_profile


def _stable_decision_recent_flip_count(flip_history, now_ts):
    now_ts = int(now_ts or time.time())
    return len(
        [
            item
            for item in list(flip_history or [])
            if now_ts - int(item.get("ts", now_ts) or now_ts) <= STABLE_DECISION_FLIP_WINDOW_SECONDS
        ]
    )


def _stable_decision_update_signal_buffer(state, raw_profile, now_ts=None):
    now_ts = int(now_ts or time.time())
    state = copy.deepcopy(state if isinstance(state, dict) else _default_stable_decision_state(now_ts))
    signal_buffer = list(state.get("signal_buffer") or [])
    signal_entry = {
        "bucket_ts": _stable_decision_bucket_ts(now_ts),
        "ts": now_ts,
        "direction": raw_profile.get("direction"),
        "setup_type": raw_profile.get("setup_type"),
        "execution_quality": raw_profile.get("execution_quality"),
        "confidence": raw_profile.get("confidence"),
        "target_state": raw_profile.get("target_state"),
        "session": raw_profile.get("session"),
        "signal_signature": raw_profile.get("signal_signature"),
        "signature": raw_profile.get("signature"),
    }
    if signal_buffer and int(signal_buffer[-1].get("bucket_ts", -1)) == signal_entry["bucket_ts"]:
        signal_buffer[-1] = signal_entry
    else:
        signal_buffer.append(signal_entry)
    signal_buffer = _stable_decision_prune_buffer(signal_buffer, now_ts)
    state["signal_buffer"] = signal_buffer
    return state, signal_buffer


def _stable_decision_consecutive_matches(signal_buffer, signal_signature):
    signal_buffer = list(signal_buffer or [])
    count = 0
    first_ts = 0
    for item in reversed(signal_buffer):
        if str(item.get("signal_signature") or "") != str(signal_signature or ""):
            break
        count += 1
        first_ts = int(item.get("ts", 0) or 0)
    duration = 0
    if count and first_ts:
        duration = max(0, int(signal_buffer[-1].get("ts", first_ts) or first_ts) - first_ts)
    return count, duration


def _stable_decision_consecutive_opposition(signal_buffer, stable_decision):
    signal_buffer = list(signal_buffer or [])
    stable_decision = stable_decision if isinstance(stable_decision, dict) else {}
    stable_direction = str(stable_decision.get("direction") or "Neutral")
    stable_signature = str(stable_decision.get("signal_signature") or "")
    count = 0
    first_ts = 0
    for item in reversed(signal_buffer):
        direction = str(item.get("direction") or "Neutral")
        signal_signature = str(item.get("signal_signature") or "")
        target_state = str(item.get("target_state") or "SCANNING")
        if stable_direction == "Long":
            is_opposing = direction == "Short" or target_state in {"SCANNING", "INVALIDATED"}
        elif stable_direction == "Short":
            is_opposing = direction == "Long" or target_state in {"SCANNING", "INVALIDATED"}
        else:
            is_opposing = signal_signature != stable_signature and target_state in {"CANDIDATE", "CONFIRMED", "ACTIVE"}
        if not is_opposing:
            break
        count += 1
        first_ts = int(item.get("ts", 0) or 0)
    duration = 0
    if count and first_ts:
        duration = max(0, int(signal_buffer[-1].get("ts", first_ts) or first_ts) - first_ts)
    return count, duration


def _stable_decision_buffer_new_bars(state, now_ts=None):
    now_ts = int(now_ts or time.time())
    state = state if isinstance(state, dict) else {}
    last_bucket = int(state.get("last_evaluation_bucket_ts", 0) or 0)
    observation_buffer = list(state.get("observation_buffer") or [])
    return len(
        [
            item
            for item in observation_buffer
            if int(item.get("bucket_ts", now_ts) or now_ts) > last_bucket
        ]
    )


def _append_decision_churn_log(event):
    event = event if isinstance(event, dict) else {}
    payload = _load_json_file(DECISION_CHURN_LOG_FILE, {"events": []})
    events = list(payload.get("events") or [])
    events.append(event)
    events = events[-DECISION_CHURN_LOG_LIMIT:]
    _save_json_file(DECISION_CHURN_LOG_FILE, {"events": events})


def _record_stable_decision_churn(state, event, persist=False):
    state = state if isinstance(state, dict) else _default_stable_decision_state()
    event = event if isinstance(event, dict) else {}
    recent_churn = list(state.get("recent_churn") or [])
    recent_churn.append(event)
    state["recent_churn"] = recent_churn[-50:]
    state["last_suppression_reason"] = str(event.get("suppression_reason") or "")
    state["last_hold_reason"] = str(event.get("hold_reason") or "")
    if persist:
        _append_decision_churn_log(event)
    return state


def _stable_decision_with_timing_fields(stable_decision, state, now_ts=None):
    now_ts = int(now_ts or time.time())
    stable_decision = dict(stable_decision or _stable_decision_default_payload(now_ts=now_ts))
    lock_until_ts = int(state.get("decision_lock_until_ts", 0) or 0)
    new_bars = _stable_decision_buffer_new_bars(state, now_ts)
    remaining_bars = max(0, STABLE_DECISION_REEVALUATION_BARS - new_bars)
    next_bucket_ts = _stable_decision_bucket_ts(now_ts) + STABLE_DECISION_BUFFER_BUCKET_SECONDS
    if remaining_bars:
        next_eval_ts = next_bucket_ts + max(0, remaining_bars - 1) * STABLE_DECISION_BUFFER_BUCKET_SECONDS
    else:
        next_eval_ts = next_bucket_ts
    if lock_until_ts > now_ts:
        next_eval_ts = max(next_eval_ts, lock_until_ts)
    stable_decision["decision_locked_until"] = _utc_iso_timestamp(lock_until_ts) or stable_decision.get("decision_locked_until")
    stable_decision["next_evaluation"] = _utc_iso_timestamp(next_eval_ts) or stable_decision.get("next_evaluation")
    stable_decision["locked"] = lock_until_ts > now_ts
    stable_decision["suppression_reason"] = str(state.get("last_suppression_reason") or stable_decision.get("suppression_reason") or "")
    stable_decision["hold_reason"] = str(state.get("last_hold_reason") or stable_decision.get("hold_reason") or "")
    stable_decision["flip_count_10m"] = _stable_decision_recent_flip_count(state.get("flip_history"), now_ts)
    return stable_decision


def _clear_execution_quality_trade_plan(execution_quality):
    execution_quality = dict(execution_quality or {})
    execution_quality["entry"] = {
        "low": None,
        "high": None,
        "mid": None,
        "text": "Wait for a cleaner location",
    }
    execution_quality["stopLoss"] = {
        "price": None,
        "basis": "No structural stop available",
        "distance": None,
        "atrMultiple": None,
    }
    execution_quality["targets"] = [
        {
            "label": "TP1",
            "price": None,
            "basis": "No target available",
            "rMultiple": None,
        },
        {
            "label": "TP2",
            "price": None,
            "basis": "No runner target available",
            "rMultiple": None,
        },
    ]
    execution_quality["riskReward"] = 0.0
    execution_quality["positionSizeNote"] = "Skip until blockers clear"
    execution_quality["invalidations"] = [
        "No trade until VWAP, ORB, pivots, and ADX align.",
    ]
    return execution_quality


def _stable_decision_apply_payload_overrides(
    stable_decision,
    decision_status,
    execution_state,
    execution_quality,
):
    stable_decision = dict(stable_decision or {})
    decision_status = dict(decision_status or {})
    execution_state = dict(execution_state or {})
    execution_quality = dict(execution_quality or {})

    decision_state = str(stable_decision.get("decision_state") or "SCANNING")
    direction = str(stable_decision.get("direction") or "Neutral")
    quality_status = _stable_decision_quality_status(decision_state)
    raw_direction = str(execution_quality.get("direction") or "Neutral")

    execution_quality["rawScore"] = execution_quality.get("score")
    execution_quality["rawGrade"] = execution_quality.get("grade")
    execution_quality["rawDirection"] = raw_direction
    execution_quality["score"] = int(stable_decision.get("confidence") or 0)
    execution_quality["grade"] = str(stable_decision.get("execution_quality") or "No Trade")
    execution_quality["status"] = quality_status
    execution_quality["setup"] = str(stable_decision.get("setup_type") or execution_quality.get("setup") or "No Clean Entry")
    execution_quality["direction"] = direction
    execution_quality["decisionState"] = decision_state
    execution_quality["decisionLockedUntil"] = stable_decision.get("decision_locked_until")
    execution_quality["flipCount10m"] = stable_decision.get("flip_count_10m")
    execution_quality["changeReason"] = stable_decision.get("change_reason")
    execution_quality["nextEvaluation"] = stable_decision.get("next_evaluation")
    execution_quality["suppressionReason"] = stable_decision.get("suppression_reason")

    plan_direction_mismatch = (
        raw_direction in {"Long", "Short"}
        and direction in {"Long", "Short"}
        and raw_direction != direction
    )
    if (
        execution_quality["grade"] == "No Trade"
        or quality_status == "blocked"
        or plan_direction_mismatch
    ):
        execution_quality = _clear_execution_quality_trade_plan(execution_quality)

    if decision_state == "SCANNING":
        decision_status["status"] = "wait"
        decision_status["text"] = "Stable decision engine is scanning. Hold for cleaner confirmation."
        execution_state["status"] = "stand_aside"
        execution_state["title"] = "Scanning"
        execution_state["text"] = decision_status["text"]
        execution_state["action"] = "hold"
        execution_state["actionState"] = "WAIT"
        execution_state["entryAllowed"] = False
        execution_state["exitRecommended"] = False
    elif decision_state == "CANDIDATE":
        decision_status["status"] = "wait"
        decision_status["text"] = f"Watchlist only. {stable_decision.get('setup_type') or 'Setup'} is building but not yet confirmed."
        execution_state["status"] = "prepare"
        execution_state["title"] = "Candidate Setup"
        execution_state["text"] = decision_status["text"]
        execution_state["action"] = "prepare"
        execution_state["actionState"] = "SETUP_LONG" if direction == "Long" else "SETUP_SHORT" if direction == "Short" else "WAIT"
        execution_state["entryAllowed"] = False
        execution_state["exitRecommended"] = False
    elif decision_state in {"CONFIRMED", "ACTIVE"}:
        existing_status = str(execution_state.get("status") or "stand_aside")
        existing_action_state = str(execution_state.get("actionState") or "WAIT")
        existing_active_state = existing_action_state in {"LONG_ACTIVE", "SHORT_ACTIVE"} or existing_status == "hold"
        if existing_active_state:
            if direction in {"Long", "Short"}:
                decision_status["status"] = "buy" if direction == "Long" else "sell"
            if not str(decision_status.get("text") or "").strip():
                decision_status["text"] = (
                    "Safer to look for a buy. Long state is confirmed with acceptable tradeability."
                    if direction == "Long"
                    else "Safer to look for a sell. Short state is confirmed with acceptable tradeability."
                    if direction == "Short"
                    else "Active execution state is confirmed."
                )
            execution_state["status"] = existing_status or "hold"
            execution_state["title"] = str(
                execution_state.get("title")
                or ("Active Execution" if decision_state == "ACTIVE" else "Confirmed Setup")
            )
            execution_state["text"] = str(
                execution_state.get("text")
                or decision_status.get("text")
                or "Execution state is active and aligned."
            )
            execution_state["action"] = str(execution_state.get("action") or existing_status or "enter")
            execution_state["actionState"] = existing_action_state or (
                "LONG_ACTIVE" if direction == "Long" else "SHORT_ACTIVE" if direction == "Short" else "WAIT"
            )
            execution_state["entryAllowed"] = bool(execution_state.get("entryAllowed", True))
            execution_state["exitRecommended"] = bool(execution_state.get("exitRecommended", False))
        else:
            decision_status["status"] = "buy" if direction == "Long" else "sell" if direction == "Short" else "wait"
            state_label = "Active execution" if decision_state == "ACTIVE" else "Confirmed setup"
            decision_status["text"] = f"{state_label}. {stable_decision.get('setup_type') or 'Directional setup'} is stable enough to act on."
            execution_state["status"] = "enter"
            execution_state["title"] = "Active Execution" if decision_state == "ACTIVE" else "Confirmed Setup"
            execution_state["text"] = decision_status["text"]
            execution_state["action"] = "enter"
            execution_state["actionState"] = "LONG_ACTIVE" if direction == "Long" else "SHORT_ACTIVE" if direction == "Short" else "WAIT"
            execution_state["entryAllowed"] = True
            execution_state["exitRecommended"] = False
    else:
        decision_status["status"] = "exit"
        decision_status["text"] = f"Setup invalidated. {stable_decision.get('change_reason') or 'MOMENTUM_SHIFT'} broke the prior idea."
        execution_state["status"] = "stand_aside"
        execution_state["title"] = "Setup Invalidated"
        execution_state["text"] = decision_status["text"]
        execution_state["action"] = "hold"
        execution_state["actionState"] = "WAIT"
        execution_state["entryAllowed"] = False
        execution_state["exitRecommended"] = False
        execution_quality["grade"] = "No Trade"
        execution_quality["status"] = "blocked"

    decision_status["decisionState"] = decision_state
    decision_status["decisionLockedUntil"] = stable_decision.get("decision_locked_until")
    decision_status["changeReason"] = stable_decision.get("change_reason")
    decision_status["flipCount10m"] = stable_decision.get("flip_count_10m")
    decision_status["suppressionReason"] = stable_decision.get("suppression_reason")

    execution_state["decisionState"] = decision_state
    execution_state["decisionLockedUntil"] = stable_decision.get("decision_locked_until")
    execution_state["changeReason"] = stable_decision.get("change_reason")
    execution_state["flipCount10m"] = stable_decision.get("flip_count_10m")
    execution_state["antiFlipLockActive"] = bool(stable_decision.get("locked"))
    return stable_decision, decision_status, execution_state, execution_quality


def _apply_stable_decision_controls(
    state,
    prediction,
    ta_data,
    regime_state,
    decision_status,
    execution_state,
    execution_quality,
    now_ts=None,
    persist_churn=False,
):
    now_ts = int(now_ts or time.time())
    state = copy.deepcopy(state if isinstance(state, dict) else _default_stable_decision_state(now_ts))
    stable_decision = dict(state.get("stable_decision") or _stable_decision_default_payload(now_ts=now_ts))
    previous_decision = dict(stable_decision)

    flip_history = [
        item
        for item in list(state.get("flip_history") or [])
        if now_ts - int(item.get("ts", now_ts) or now_ts) <= STABLE_DECISION_FLIP_WINDOW_SECONDS
    ]
    state["flip_history"] = flip_history
    recent_flip_count = _stable_decision_recent_flip_count(flip_history, now_ts)

    raw_profile = _build_stable_decision_raw_profile(
        prediction,
        ta_data,
        regime_state,
        decision_status,
        execution_state,
        execution_quality,
        recent_flip_count,
        now_ts,
    )
    raw_profile["target_state"] = _stable_decision_target_state(previous_decision, raw_profile)
    raw_profile["signature"] = f"{raw_profile['signal_signature']}|{raw_profile['target_state']}"

    state, signal_buffer = _stable_decision_update_signal_buffer(state, raw_profile, now_ts)
    same_signal_count, same_signal_duration = _stable_decision_consecutive_matches(
        signal_buffer,
        raw_profile.get("signal_signature"),
    )
    opposing_count, opposing_duration = _stable_decision_consecutive_opposition(
        signal_buffer,
        previous_decision,
    )
    new_bars = _stable_decision_buffer_new_bars(state, now_ts)

    raw_execution_is_active = (
        str(raw_profile.get("action_state") or "WAIT") in {"LONG_ACTIVE", "SHORT_ACTIVE"}
        or str(raw_profile.get("execution_status") or "stand_aside") == "hold"
    )

    if raw_profile["target_state"] in {"CONFIRMED", "ACTIVE"} and not raw_execution_is_active:
        if same_signal_count < STABLE_DECISION_REEVALUATION_BARS or same_signal_duration < STABLE_DECISION_CANDIDATE_SECONDS:
            raw_profile["target_state"] = "CANDIDATE"
            raw_profile["signature"] = f"{raw_profile['signal_signature']}|{raw_profile['target_state']}"
    if previous_decision.get("decision_state") in {"CONFIRMED", "ACTIVE"}:
        if opposing_count < 2 and opposing_duration < STABLE_DECISION_INVALIDATION_SECONDS:
            if raw_profile["target_state"] == "INVALIDATED":
                raw_profile["target_state"] = str(previous_decision.get("decision_state") or "CONFIRMED")
                raw_profile["signature"] = f"{raw_profile['signal_signature']}|{raw_profile['target_state']}"

    allow_reevaluation = not state.get("last_evaluation_bucket_ts") or new_bars >= STABLE_DECISION_REEVALUATION_BARS
    if previous_decision.get("decision_state") in {"CONFIRMED", "ACTIVE"} and (
        opposing_count >= 2 or opposing_duration >= STABLE_DECISION_INVALIDATION_SECONDS
    ):
        allow_reevaluation = True

    if not allow_reevaluation:
        state["stable_decision"] = _stable_decision_with_timing_fields(previous_decision, state, now_ts)
        stable_decision, decision_status, execution_state, execution_quality = _stable_decision_apply_payload_overrides(
            state["stable_decision"],
            decision_status,
            execution_state,
            execution_quality,
        )
        state["stable_decision"] = stable_decision
        return stable_decision, decision_status, execution_state, execution_quality, state

    state["last_evaluation_bucket_ts"] = _stable_decision_bucket_ts(now_ts)
    state["last_suppression_reason"] = ""
    state["last_hold_reason"] = ""

    target_state = str(raw_profile.get("target_state") or "SCANNING")
    previous_state = str(previous_decision.get("decision_state") or "SCANNING")
    previous_signal_signature = str(previous_decision.get("signal_signature") or "")
    change_reason = _stable_decision_reason(previous_decision, raw_profile)

    lock_until_ts = int(state.get("decision_lock_until_ts", 0) or 0)
    lock_active = lock_until_ts > now_ts
    same_signal = raw_profile.get("signal_signature") == previous_signal_signature and previous_signal_signature
    upgrade = same_signal and _stable_decision_state_rank(target_state) > _stable_decision_state_rank(previous_state)
    downgrade = same_signal and _stable_decision_state_rank(target_state) < _stable_decision_state_rank(previous_state)
    changed_signal = raw_profile.get("signal_signature") != previous_signal_signature
    decision_changed = target_state != previous_state or changed_signal

    if downgrade and lock_active:
        target_state = previous_state
        decision_changed = False

    if decision_changed and lock_active and not upgrade:
        hold_reason = f"HOLD: {previous_decision.get('setup_type') or 'previous decision'}"
        churn_event = {
            "ts": now_ts,
            "suppression_reason": "LOCKED",
            "hold_reason": hold_reason,
            "from": previous_decision,
            "candidate": {
                "decision_state": raw_profile.get("target_state"),
                "setup_type": raw_profile.get("setup_type"),
                "execution_quality": raw_profile.get("execution_quality"),
                "confidence": raw_profile.get("confidence"),
                "change_reason": change_reason,
            },
        }
        state = _record_stable_decision_churn(state, churn_event, persist=persist_churn)
        state["stable_decision"] = _stable_decision_with_timing_fields(previous_decision, state, now_ts)
        stable_decision, decision_status, execution_state, execution_quality = _stable_decision_apply_payload_overrides(
            state["stable_decision"],
            decision_status,
            execution_state,
            execution_quality,
        )
        state["stable_decision"] = stable_decision
        return stable_decision, decision_status, execution_state, execution_quality, state

    decision_ring = list(state.get("decision_ring") or [])
    decision_ring = decision_ring[-STABLE_DECISION_HISTORY_LIMIT:]
    candidate_signature = f"{raw_profile['signal_signature']}|{target_state}"
    if decision_changed:
        recent_repeat = next(
            (
                item
                for item in reversed(decision_ring)
                if item.get("signature") == candidate_signature
                and now_ts - int(item.get("ts", now_ts) or now_ts) <= STABLE_DECISION_REPEAT_SUPPRESS_SECONDS
            ),
            None,
        )
        if recent_repeat and candidate_signature != str(previous_decision.get("signature") or ""):
            hold_reason = f"HOLD: {previous_decision.get('setup_type') or 'previous decision'}"
            churn_event = {
                "ts": now_ts,
                "suppression_reason": "REPEAT",
                "hold_reason": hold_reason,
                "from": previous_decision,
                "candidate": {
                    "decision_state": target_state,
                    "setup_type": raw_profile.get("setup_type"),
                    "execution_quality": raw_profile.get("execution_quality"),
                    "confidence": raw_profile.get("confidence"),
                    "change_reason": change_reason,
                },
            }
            state = _record_stable_decision_churn(state, churn_event, persist=persist_churn)
            state["stable_decision"] = _stable_decision_with_timing_fields(previous_decision, state, now_ts)
            stable_decision, decision_status, execution_state, execution_quality = _stable_decision_apply_payload_overrides(
                state["stable_decision"],
                decision_status,
                execution_state,
                execution_quality,
            )
            state["stable_decision"] = stable_decision
            return stable_decision, decision_status, execution_state, execution_quality, state

        if len(decision_ring) >= 2:
            immediate_previous = decision_ring[-1]
            prior_previous = decision_ring[-2]
            if (
                prior_previous.get("signature") == candidate_signature
                and immediate_previous.get("signature") != candidate_signature
                and now_ts - int(immediate_previous.get("ts", now_ts) or now_ts) <= STABLE_DECISION_OSCILLATION_SECONDS
            ):
                hold_reason = f"HOLD: {previous_decision.get('setup_type') or 'previous decision'}"
                churn_event = {
                    "ts": now_ts,
                    "suppression_reason": "OSCILLATION",
                    "hold_reason": hold_reason,
                    "from": previous_decision,
                    "candidate": {
                        "decision_state": target_state,
                        "setup_type": raw_profile.get("setup_type"),
                        "execution_quality": raw_profile.get("execution_quality"),
                        "confidence": raw_profile.get("confidence"),
                        "change_reason": change_reason,
                    },
                }
                state = _record_stable_decision_churn(state, churn_event, persist=persist_churn)
                state["stable_decision"] = _stable_decision_with_timing_fields(previous_decision, state, now_ts)
                stable_decision, decision_status, execution_state, execution_quality = _stable_decision_apply_payload_overrides(
                    state["stable_decision"],
                    decision_status,
                    execution_state,
                    execution_quality,
                )
                state["stable_decision"] = stable_decision
                return stable_decision, decision_status, execution_state, execution_quality, state

    output_setup = raw_profile.get("setup_type") or "No Clean Entry"
    output_grade = raw_profile.get("execution_quality") or "No Trade"
    output_confidence = int(raw_profile.get("confidence") or 0)
    if target_state == "SCANNING":
        output_grade = "No Trade"
        output_confidence = min(output_confidence, 59)
    elif target_state == "INVALIDATED":
        output_setup = str(previous_decision.get("setup_type") or output_setup)
        output_grade = "No Trade"
        output_confidence = min(output_confidence, 59)
    elif output_grade not in {"A", "B", "C"}:
        output_grade = "B" if target_state in {"CONFIRMED", "ACTIVE"} else "C"

    stable_decision = {
        "symbol": "XAUUSD",
        "decision_state": target_state,
        "setup_type": output_setup,
        "execution_quality": output_grade,
        "confidence": output_confidence,
        "decision_locked_until": _utc_iso_timestamp(now_ts + STABLE_DECISION_LOCK_SECONDS),
        "flip_count_10m": recent_flip_count,
        "change_reason": change_reason if decision_changed else str(previous_decision.get("change_reason") or change_reason),
        "session": str(raw_profile.get("session") or previous_decision.get("session") or "Unknown"),
        "bar_timestamp": str(raw_profile.get("bar_timestamp") or previous_decision.get("bar_timestamp") or _utc_iso_timestamp(now_ts) or ""),
        "next_evaluation": _utc_iso_timestamp(now_ts + STABLE_DECISION_BUFFER_BUCKET_SECONDS),
        "signal_signature": str(raw_profile.get("signal_signature") or "Neutral|No Clean Entry|No Trade"),
        "signature": candidate_signature,
        "direction": str(raw_profile.get("direction") or "Neutral"),
        "suppression_reason": "",
        "hold_reason": "",
        "locked": False,
    }

    previous_direction = str(previous_decision.get("direction") or "Neutral")
    current_signal_signature = str(raw_profile.get("signal_signature") or "")
    is_flip = bool(
        previous_signal_signature
        and current_signal_signature
        and previous_direction in {"Long", "Short"}
        and current_signal_signature != previous_signal_signature
    )

    if decision_changed:
        flip_event = {
            "ts": now_ts,
            "from": str(previous_decision.get("signature") or ""),
            "to": candidate_signature,
            "reason": stable_decision.get("change_reason"),
        }
        flip_history = list(state.get("flip_history") or [])
        if is_flip:
            flip_history.append(flip_event)
            flip_history = flip_history[-50:]
            state["flip_history"] = flip_history
        if is_flip and _stable_decision_recent_flip_count(flip_history, now_ts) > 3:
            state = _record_stable_decision_churn(
                state,
                {
                    "ts": now_ts,
                    "suppression_reason": "FLIP_ALERT",
                    "hold_reason": "Flip count exceeded 3 within 10 minutes.",
                    "from": previous_decision,
                    "candidate": stable_decision,
                },
                persist=persist_churn,
            )
        decision_ring.append(
            {
                "ts": now_ts,
                "signature": candidate_signature,
                "decision_state": stable_decision.get("decision_state"),
                "setup_type": stable_decision.get("setup_type"),
                "execution_quality": stable_decision.get("execution_quality"),
                "confidence": stable_decision.get("confidence"),
                "session": stable_decision.get("session"),
                "bar_timestamp": stable_decision.get("bar_timestamp"),
            }
        )
        decision_ring = decision_ring[-STABLE_DECISION_HISTORY_LIMIT:]
        state["decision_ring"] = decision_ring
        state["last_decision_time_ts"] = now_ts
        state["decision_lock_until_ts"] = now_ts + STABLE_DECISION_LOCK_SECONDS
    stable_decision = _stable_decision_with_timing_fields(stable_decision, state, now_ts)
    state["stable_decision"] = stable_decision
    stable_decision, decision_status, execution_state, execution_quality = _stable_decision_apply_payload_overrides(
        stable_decision,
        decision_status,
        execution_state,
        execution_quality,
    )
    state["stable_decision"] = stable_decision
    return stable_decision, decision_status, execution_state, execution_quality, state


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
    big_move_risk_bucket = _big_move_risk_bucket(big_move_risk)
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

    if (
        active_long
        and permission_status == "entry_allowed"
        and not bullish_major_conflicts
        and bullish_evidence
    ):
        variant_text = bullish_variant or "Bullish stack aligned"
        basis = _join_readable_list(bullish_evidence[:4])
        return {
            "label": "Buy",
            "tone": _dashboard_action_tone("Buy"),
            "reason": f"Active long remains aligned: {variant_text}. {basis}.",
            "variant": variant_text,
            "bullish_evidence": bullish_evidence,
            "bullish_major_conflicts": bullish_major_conflicts,
            "bullish_soft_conflicts": bullish_soft_conflicts,
        }
    if (
        active_short
        and permission_status == "entry_allowed"
        and not bearish_major_conflicts
        and bearish_evidence
    ):
        variant_text = bearish_variant or "Bearish stack aligned"
        basis = _join_readable_list(bearish_evidence[:4])
        return {
            "label": "Sell",
            "tone": _dashboard_action_tone("Sell"),
            "reason": f"Active short remains aligned: {variant_text}. {basis}.",
            "variant": variant_text,
            "bearish_evidence": bearish_evidence,
            "bearish_major_conflicts": bearish_major_conflicts,
            "bearish_soft_conflicts": bearish_soft_conflicts,
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
        big_move_risk_bucket in {"Elevated", "Extreme"}
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


def _align_dashboard_response_contract(payload):
    if not isinstance(payload, dict):
        return payload

    market_state = payload.get("MarketState")
    market_state = dict(market_state) if isinstance(market_state, dict) else {}
    execution_permission = payload.get("ExecutionPermission")
    execution_permission = (
        dict(execution_permission) if isinstance(execution_permission, dict) else {}
    )
    execution_state = payload.get("ExecutionState")
    execution_state = dict(execution_state) if isinstance(execution_state, dict) else {}
    trade_playbook = payload.get("TradePlaybook")
    trade_playbook = dict(trade_playbook) if isinstance(trade_playbook, dict) else {}
    dashboard_action = payload.get("DashboardAction")
    dashboard_action = dict(dashboard_action) if isinstance(dashboard_action, dict) else {}

    dashboard_label = str(dashboard_action.get("label") or "Stand Aside").strip()
    dashboard_reason = str(dashboard_action.get("reason") or "").strip()
    playbook_stage = str(trade_playbook.get("stage") or "stand_aside").strip()
    directional_bias = str(
        market_state.get("directional_bias")
        or trade_playbook.get("directionalBias")
        or "Neutral"
    ).strip()

    if dashboard_label == "Exit":
        exit_text = dashboard_reason or str(
            execution_permission.get("text")
            or trade_playbook.get("text")
            or "active position quality is deteriorating."
        ).strip()
        if exit_text and not exit_text.lower().startswith("exit recommended"):
            exit_text = f"Exit Recommended: {exit_text}"
        execution_permission["status"] = "exit_recommended"
        execution_permission["text"] = exit_text or "Exit Recommended: active position quality is deteriorating."
        execution_permission["actionState"] = "EXIT_RISK"
        if not execution_permission.get("decisionStatus"):
            execution_permission["decisionStatus"] = "exit"

        market_state["action"] = "exit"
        market_state["action_state"] = "EXIT_RISK"

        trade_playbook["stage"] = "exit"
        trade_playbook["title"] = "Exit / Reduce"
        trade_playbook["text"] = dashboard_reason or trade_playbook.get("text") or exit_text
        trade_playbook["entryReadiness"] = "blocked"
        trade_playbook["exitUrgency"] = "high"
        if not trade_playbook.get("why"):
            trade_playbook["why"] = [dashboard_reason or "Dashboard action mapped to exit."]
        if not trade_playbook.get("directionalBias") and directional_bias:
            trade_playbook["directionalBias"] = directional_bias

        execution_state["status"] = "exit"
        execution_state["title"] = trade_playbook.get("title") or "Exit / Reduce"
        execution_state["text"] = trade_playbook.get("text") or exit_text
        execution_state["entryAllowed"] = False
        execution_state["exitRecommended"] = True
        execution_state["permissionStatus"] = "exit_recommended"
        execution_state["action"] = "exit"
        execution_state["actionState"] = "EXIT_RISK"
    elif playbook_stage == "hold":
        execution_state["status"] = "hold"
        execution_state["title"] = trade_playbook.get("title") or execution_state.get("title") or "Hold"
        execution_state["text"] = (
            trade_playbook.get("text")
            or execution_state.get("text")
            or execution_permission.get("text")
            or ""
        )
        execution_state["permissionStatus"] = str(
            execution_permission.get("status") or execution_state.get("permissionStatus") or "no_trade"
        )
        execution_state["entryAllowed"] = execution_state["permissionStatus"] == "entry_allowed"
        execution_state["exitRecommended"] = execution_state["permissionStatus"] == "exit_recommended"
        execution_state["action"] = "hold"
        if directional_bias == "Bullish":
            execution_state["actionState"] = "LONG_ACTIVE"
        elif directional_bias == "Bearish":
            execution_state["actionState"] = "SHORT_ACTIVE"

    payload["MarketState"] = market_state
    payload["ExecutionPermission"] = execution_permission
    payload["ExecutionState"] = execution_state
    payload["TradePlaybook"] = trade_playbook
    return payload


def _sanitize_client_prediction_payload(payload):
    if not isinstance(payload, dict):
        return payload

    sanitized = copy.deepcopy(payload)
    for key in (
        "MarketState",
        "ExecutionPermission",
        "TradePlaybook",
        "DashboardAction",
        "RR200Signal",
        "TradeBrainRuntime",
    ):
        sanitized.pop(key, None)

    technical_analysis = sanitized.get("TechnicalAnalysis")
    if not isinstance(technical_analysis, dict):
        return sanitized

    technical_analysis.pop("rsi_14", None)
    technical_analysis.pop("rsi_signal", None)

    support_resistance = technical_analysis.get("support_resistance")
    if isinstance(support_resistance, dict):
        for key in ("round_numbers", "active_fvgs", "range_zone"):
            support_resistance.pop(key, None)

    structure_context = technical_analysis.get("structure_context")
    if isinstance(structure_context, dict):
        for key in (
            "roundNumberSupport",
            "roundNumberResistance",
            "majorRoundNumberSupport",
            "majorRoundNumberResistance",
            "roundNumberStep",
            "roundSupportDistancePct",
            "roundResistanceDistancePct",
            "bullishFvg",
            "bearishFvg",
            "bullishFvgDistancePct",
            "bearishFvgDistancePct",
            "inBullishFvg",
            "inBearishFvg",
            "rangeZone",
            "rangeZoneActive",
            "inRangeZone",
            "rangeZoneBreak",
            "rangeZonePosition",
            "rangeZoneWidthPct",
        ):
            structure_context.pop(key, None)

    return sanitized


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


def _vwap_bias_override_for_notification(changes=None, snapshot=None):
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    snapshot_bias = str(snapshot.get("micro_vwap_bias") or "").strip()
    if snapshot_bias:
        return snapshot_bias
    changes = changes if isinstance(changes, dict) else {}
    delta = changes.get("micro_vwap_bias")
    if isinstance(delta, dict):
        return str(delta.get("current") or "").strip() or None
    return None


def _build_signal_notification(changes, rr_signal, market_structure, ta_data=None, payload=None, snapshot=None):
    changes = _filter_notification_changes(changes)
    changed_keys = set((changes or {}).keys())
    title = _notification_title_for_changes(changes)
    fingerprint = _notification_fingerprint(changes)
    body_lines = []
    if "market_structure" in changed_keys:
        body_lines.append(f"Market Structure: {_format_market_structure_change(changes, market_structure)}")
    quality_change = _format_execution_quality_change(changes)
    if quality_change:
        body_lines.append(f"Execution Quality: {quality_change}")
    micro_change = _format_microstructure_change(changes)
    if micro_change:
        body_lines.append(f"Microstructure Change: {micro_change}")
    body_lines.append(
        "Bar Session / Microstructure: "
        f"{_format_bar_session_microstructure(ta_data, _vwap_bias_override_for_notification(changes, snapshot))}"
    )

    if not body_lines:
        body_lines.append(f"Market Structure: {_format_market_structure_change(changes, market_structure)}")

    body = "\n".join(body_lines)
    return {
        "title": title,
        "body": body,
        "notification_fingerprint": fingerprint,
        "notification_tag": _notification_tag_from_fingerprint(fingerprint),
        "dashboard_action": {},
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
        "event_regime",
        "breakout_bias",
        "execution_state",
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
    value = _round_display_value(score or 0, 2, 0.0)
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


def _send_telegram_notification(changes, rr_signal, market_structure, ta_data=None, payload=None, snapshot=None):
    if not _telegram_enabled():
        return

    notification = _build_signal_notification(
        changes,
        rr_signal,
        market_structure,
        ta_data=ta_data,
        payload=payload,
        snapshot=snapshot,
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


def _send_web_push_notifications(changes, rr_signal, market_structure, ta_data=None, payload=None, snapshot=None):
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
        snapshot=snapshot,
    )
    push_payload = {
        "title": notification.get("title"),
        "body": notification.get("body"),
        "tag": notification.get("notification_tag"),
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

    always_material = set()
    numeric_thresholds = {
        "micro_vwap_delta_pct": VWAP_DELTA_ALERT_THRESHOLD_PCT,
    }

    for key, delta in changes.items():
        if key == "market_structure":
            if not isinstance(delta, dict):
                continue
            if _is_alertable_market_structure_transition(
                delta.get("previous"),
                delta.get("current"),
            ):
                return True
            continue

        if key in {"micro_orb_state", "micro_sweep_state"}:
            if not isinstance(delta, dict):
                continue
            if _is_alertable_directional_micro_transition(
                delta.get("previous"),
                delta.get("current"),
            ):
                return True
            continue

        if key == "execution_quality_signal":
            if not isinstance(delta, dict):
                continue
            if _is_material_execution_quality_transition(
                delta.get("previous"),
                delta.get("current"),
            ):
                return True
            continue

        if key == "micro_vwap_bias":
            if not isinstance(delta, dict):
                continue
            vwap_value_delta = changes.get("micro_vwap_delta_pct")
            previous_value = vwap_value_delta.get("previous") if isinstance(vwap_value_delta, dict) else None
            current_value = vwap_value_delta.get("current") if isinstance(vwap_value_delta, dict) else None
            if _is_alertable_vwap_bias_transition(
                delta.get("previous"),
                delta.get("current"),
                previous_value,
                current_value,
            ):
                return True
            continue

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


def _evaluate_decision_status(
    verdict,
    confidence,
    ta_data,
    trade_guidance,
    execution_state=None,
    tradeability=None,
    regime=None,
):
    market_state = (ta_data or {}).get("_prediction_market_state") or {}
    market_session = _extract_market_session_state(ta_data)
    execution_state = execution_state if isinstance(execution_state, dict) else {}
    action_state = str(
        execution_state.get("actionState")
        or market_state.get("action_state")
        or ""
    )
    mtf = (ta_data or {}).get("multi_timeframe") or {}
    trend = str(mtf.get("h1_trend") or (ta_data or {}).get("ema_trend") or "Neutral")
    price_action = (ta_data or {}).get("price_action") or {}
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
    if market_session["is_market_closed"]:
        reopen_text = (
            f" Reopens {market_session['next_open']}."
            if market_session["next_open"]
            else ""
        )
        return {
            "text": f"Stand aside. {market_session['closed_reason']}{reopen_text}",
            "status": "wait",
            "buyChecks": [False, False, False, False, False],
            "sellChecks": [False, False, False, False, False],
            "exitChecks": [True, True, False, False, True],
            "marketClosed": True,
        }
    if action_state:
        text = "Stand aside. Checklist does not support a clean trade yet."
        status = "wait"
        tradeability = str(
            tradeability
            or execution_state.get("tradeability")
            or market_state.get("tradeability")
            or "Low"
        )
        regime = str(regime or market_state.get("regime") or "unstable")
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
                    "Sell bias is intact, but execution is blocked"
                    f"{reason_suffix}."
                )
            elif buy_passed == len(buy_checks) and buy_passed > sell_passed:
                reason_suffix = f" because {blocked_reason}" if blocked_reason else ""
                text = (
                    "Buy bias is intact, but execution is blocked"
                    f"{reason_suffix}."
                )
            elif sell_passed >= 3 and sell_passed > buy_passed:
                reason_suffix = f" because {blocked_reason}" if blocked_reason else ""
                text = (
                    "Sell conditions are mostly aligned, but execution is blocked"
                    f"{reason_suffix}."
                )
            elif buy_passed >= 3 and buy_passed > sell_passed:
                reason_suffix = f" because {blocked_reason}" if blocked_reason else ""
                text = (
                    "Buy conditions are mostly aligned, but execution is blocked"
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

    if bool((decision_status or {}).get("marketClosed")):
        return {
            "text": decision_text or "No trade. XAUUSD market is currently closed.",
            "status": "no_trade",
            "actionState": "WAIT",
            "decisionStatus": decision_kind,
            "marketClosed": True,
        }

    directional_entry_confirmed = (
        (action_state == "LONG_ACTIVE" and decision_kind == "buy")
        or (action_state == "SHORT_ACTIVE" and decision_kind == "sell")
    )

    if action_state == "EXIT_RISK" or action == "exit" or decision_kind == "exit":
        text = "Exit Recommended: active position quality is deteriorating."
        status = "exit_recommended"
    elif directional_entry_confirmed:
        text = (
            "Execution remains active on the buy side."
            if action_state == "LONG_ACTIVE"
            else "Execution remains active on the sell side."
        )
        status = "entry_allowed"
    elif action_state in {"LONG_ACTIVE", "SHORT_ACTIVE"} and decision_kind in {"buy", "sell"}:
        text = "No trade. Execution state and checklist direction are not aligned yet."
        status = "no_trade"
    elif action_state in {"SETUP_LONG", "SETUP_SHORT"}:
        text = "No trade yet. Directional bias is forming, but the trigger is not active."
        status = "no_trade"
    elif action_state == "WAIT" and decision_text.startswith("Watchlist Only:"):
        blockers = []
        if tradeability.lower() == "low":
            blockers.append("low tradeability")
        if regime.lower() in {"unstable", "range", "event-risk"}:
            blockers.append(f"{regime.lower()} regime")
        blocker_text = " and ".join(blockers)
        if blocker_text:
            text = f"No trade yet. Entry is blocked by {blocker_text}."
        else:
            text = "No trade yet. Directional bias is forming, but execution is not confirmed."
        status = "no_trade"

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
    if bool((decision_status or {}).get("marketClosed")):
        closed_text = str(
            decision_status.get("text")
            or execution_permission.get("text")
            or "XAUUSD market is currently closed."
        )
        return {
            "stage": "stand_aside",
            "title": "Market Closed",
            "text": closed_text,
            "why": [closed_text],
            "entryReadiness": "closed",
            "exitUrgency": "low",
            "warningLadder": warning_ladder,
            "eventRegime": event_regime,
            "breakoutBias": breakout_bias,
            "directionalBias": directional_bias,
            "alignment": "closed",
        }
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
        stabilized["text"] = str(
            (execution_permission or {}).get("text")
            or (decision_status or {}).get("text")
            or "No trade. Conditions are not clean enough yet."
        )
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


def _build_market_state_from_prediction(prediction):
    prediction = prediction if isinstance(prediction, dict) else {}
    return {
        "regime": str(prediction.get("regime") or "unstable"),
        "tradeability": str(prediction.get("tradeability") or "Low"),
        "directional_bias": str(prediction.get("directionalBias") or "Neutral"),
        "action_state": str(prediction.get("actionState") or "WAIT"),
        "action": str(prediction.get("action") or "hold"),
    }


def _directional_execution_action_state(direction, setup=False):
    direction = str(direction or "").strip()
    if direction == "Bullish":
        return "SETUP_LONG" if setup else "LONG_ACTIVE"
    if direction == "Bearish":
        return "SETUP_SHORT" if setup else "SHORT_ACTIVE"
    return "WAIT"


def _stabilize_execution_state(
    execution_state,
    trade_playbook,
    market_state,
    execution_permission,
    decision_status,
):
    raw_state = dict(execution_state) if isinstance(execution_state, dict) else {}
    trade_playbook = dict(trade_playbook) if isinstance(trade_playbook, dict) else {}
    market_state = dict(market_state) if isinstance(market_state, dict) else {}
    execution_permission = (
        dict(execution_permission) if isinstance(execution_permission, dict) else {}
    )
    decision_status = dict(decision_status) if isinstance(decision_status, dict) else {}

    stage = str(trade_playbook.get("stage") or "stand_aside").strip()
    directional_bias = str(
        market_state.get("directional_bias")
        or trade_playbook.get("directionalBias")
        or "Neutral"
    ).strip()

    stabilized = dict(raw_state)
    stabilized["tradeability"] = str(
        raw_state.get("tradeability")
        or market_state.get("tradeability")
        or "Low"
    )

    if stage == "exit":
        stabilized["status"] = "exit"
        stabilized["title"] = str(trade_playbook.get("title") or "Exit / Reduce")
        stabilized["text"] = str(
            trade_playbook.get("text")
            or raw_state.get("text")
            or execution_permission.get("text")
            or decision_status.get("text")
            or "Exit Recommended: active position quality is deteriorating."
        )
        stabilized["action"] = "exit"
        stabilized["actionState"] = "EXIT_RISK"
        stabilized["permissionStatus"] = "exit_recommended"
        stabilized["entryAllowed"] = False
        stabilized["exitRecommended"] = True
        return stabilized

    if stage == "hold":
        stabilized["status"] = "hold"
        stabilized["title"] = str(
            trade_playbook.get("title")
            or raw_state.get("title")
            or "Hold Winner"
        )
        stabilized["text"] = str(
            trade_playbook.get("text")
            or raw_state.get("text")
            or execution_permission.get("text")
            or decision_status.get("text")
            or "Momentum is active and still aligned with the position."
        )
        stabilized["action"] = "hold"
        stabilized["actionState"] = _directional_execution_action_state(
            directional_bias,
            setup=False,
        )
        stabilized["permissionStatus"] = "entry_allowed"
        stabilized["entryAllowed"] = True
        stabilized["exitRecommended"] = False
        return stabilized

    if stage == "enter":
        stabilized["status"] = "enter"
        stabilized["title"] = str(
            trade_playbook.get("title")
            or raw_state.get("title")
            or "Enter"
        )
        stabilized["text"] = str(
            trade_playbook.get("text")
            or raw_state.get("text")
            or decision_status.get("text")
            or execution_permission.get("text")
            or "The directional engine is clear enough to trade."
        )
        stabilized["action"] = (
            "buy" if directional_bias == "Bullish" else "sell" if directional_bias == "Bearish" else str(raw_state.get("action") or "hold")
        )
        stabilized["actionState"] = _directional_execution_action_state(
            directional_bias,
            setup=False,
        )
        stabilized["permissionStatus"] = "entry_allowed"
        stabilized["entryAllowed"] = True
        stabilized["exitRecommended"] = False
        return stabilized

    if stage in {"prepare", "stalk_entry"}:
        stabilized["status"] = "prepare"
        stabilized["title"] = str(
            trade_playbook.get("title")
            or raw_state.get("title")
            or "Prepare"
        )
        stabilized["text"] = str(
            trade_playbook.get("text")
            or raw_state.get("text")
            or execution_permission.get("text")
            or decision_status.get("text")
            or "The setup is forming, but execution is not confirmed yet."
        )
        stabilized["action"] = "hold"
        stabilized["actionState"] = _directional_execution_action_state(
            directional_bias,
            setup=True,
        )
        stabilized["permissionStatus"] = "watchlist_only"
        stabilized["entryAllowed"] = False
        stabilized["exitRecommended"] = False
        return stabilized

    stabilized["status"] = "stand_aside"
    stabilized["title"] = str(
        trade_playbook.get("title")
        or raw_state.get("title")
        or "Stand Aside"
    )
    stabilized["text"] = str(
        trade_playbook.get("text")
        or raw_state.get("text")
        or decision_status.get("text")
        or execution_permission.get("text")
        or "No trade. Conditions are not clean enough yet."
    )
    stabilized["action"] = "hold"
    stabilized["actionState"] = "WAIT"
    stabilized["permissionStatus"] = "no_trade"
    stabilized["entryAllowed"] = False
    stabilized["exitRecommended"] = False
    return stabilized


def _stabilize_decision_status(decision_status):
    raw_status = str((decision_status or {}).get("status") or "wait")
    raw_text = str((decision_status or {}).get("text") or "Stand aside.")
    raw_buy_checks = list((decision_status or {}).get("buyChecks") or [])
    raw_sell_checks = list((decision_status or {}).get("sellChecks") or [])
    raw_exit_checks = list((decision_status or {}).get("exitChecks") or [])
    raw_signature = json.dumps(
        {
            "text": raw_text,
            "buyChecks": raw_buy_checks,
            "sellChecks": raw_sell_checks,
            "exitChecks": raw_exit_checks,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    now_ts = int(time.time())
    state = _load_json_file(
        DECISION_STATE_FILE,
        {
            "stable_status": raw_status,
            "stable_text": raw_text,
            "stable_buy_checks": raw_buy_checks,
            "stable_sell_checks": raw_sell_checks,
            "stable_exit_checks": raw_exit_checks,
            "last_raw_status": raw_status,
            "raw_streak": 1,
            "last_raw_signature": raw_signature,
            "raw_signature_streak": 1,
            "last_stable_change_ts": now_ts,
        },
    )

    last_raw_status = str(state.get("last_raw_status", raw_status))
    raw_streak = int(state.get("raw_streak", 0) or 0)
    if raw_status == last_raw_status:
        raw_streak += 1
    else:
        raw_streak = 1

    last_raw_signature = str(state.get("last_raw_signature", raw_signature))
    raw_signature_streak = int(state.get("raw_signature_streak", 0) or 0)
    if raw_signature == last_raw_signature:
        raw_signature_streak += 1
    else:
        raw_signature_streak = 1

    stable_status = str(state.get("stable_status", raw_status))
    stable_text = str(state.get("stable_text", raw_text))
    stable_buy_checks = list(state.get("stable_buy_checks", raw_buy_checks) or raw_buy_checks)
    stable_sell_checks = list(state.get("stable_sell_checks", raw_sell_checks) or raw_sell_checks)
    stable_exit_checks = list(state.get("stable_exit_checks", raw_exit_checks) or raw_exit_checks)
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

    if raw_status != stable_status and raw_streak >= required_confirmation:
        can_flip = True
        if (now_ts - last_stable_change_ts) < DECISION_FLIP_MIN_HOLD_SECONDS:
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
            stable_buy_checks = raw_buy_checks
            stable_sell_checks = raw_sell_checks
            stable_exit_checks = raw_exit_checks
    elif raw_status == stable_status and raw_signature != json.dumps(
        {
            "text": stable_text,
            "buyChecks": stable_buy_checks,
            "sellChecks": stable_sell_checks,
            "exitChecks": stable_exit_checks,
        },
        ensure_ascii=True,
        sort_keys=True,
    ) and raw_signature_streak >= 2:
        stable_text = raw_text
        stable_buy_checks = raw_buy_checks
        stable_sell_checks = raw_sell_checks
        stable_exit_checks = raw_exit_checks

    state.update(
        {
            "stable_status": stable_status,
            "stable_text": stable_text,
            "stable_buy_checks": stable_buy_checks,
            "stable_sell_checks": stable_sell_checks,
            "stable_exit_checks": stable_exit_checks,
            "last_raw_status": raw_status,
            "raw_streak": raw_streak,
            "last_raw_signature": raw_signature,
            "raw_signature_streak": raw_signature_streak,
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
    stabilized["buyChecks"] = stable_buy_checks
    stabilized["sellChecks"] = stable_sell_checks
    stabilized["exitChecks"] = stable_exit_checks
    stabilized["confirmationCount"] = raw_streak
    stabilized["requiredConfirmation"] = required_confirmation
    stabilized["confirmed"] = raw_streak >= required_confirmation and raw_status == stable_status
    stabilized["heldForSeconds"] = max(0, now_ts - last_stable_change_ts)
    return stabilized


def _align_decision_status_with_execution_state(decision_status, execution_state):
    aligned = dict(decision_status or {})
    execution_state = execution_state if isinstance(execution_state, dict) else {}
    action_state = str(execution_state.get("actionState") or "")

    if action_state == "LONG_ACTIVE":
        aligned["status"] = "buy"
        aligned["text"] = "Safer to look for a buy. Long state is confirmed with acceptable tradeability."
        aligned["confirmed"] = True
    elif action_state == "SHORT_ACTIVE":
        aligned["status"] = "sell"
        aligned["text"] = "Safer to look for a sell. Short state is confirmed with acceptable tradeability."
        aligned["confirmed"] = True
    elif action_state == "EXIT_RISK":
        aligned["status"] = "exit"
        aligned["text"] = "Safer to exit or stand aside. Exit risk is confirmed against the active directional state."
        aligned["confirmed"] = True
    elif action_state == "SETUP_LONG":
        aligned["status"] = "wait"
        aligned["text"] = "Long setup forming. Bullish bias is developing, but trigger confirmation is still pending."
    elif action_state == "SETUP_SHORT":
        aligned["status"] = "wait"
        aligned["text"] = "Short setup forming. Bearish bias is developing, but trigger confirmation is still pending."

    return aligned


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


def _extract_indicator_snapshot(payload, previous_snapshot=None):
    """Build a compact snapshot used to detect indicator changes."""
    ta_data = payload.get("TechnicalAnalysis", {}) if isinstance(payload, dict) else {}
    pa = ta_data.get("price_action", {}) if isinstance(ta_data, dict) else {}
    regime_state = payload.get("RegimeState", {}) if isinstance(payload, dict) else {}
    execution_state = payload.get("ExecutionState", {}) if isinstance(payload, dict) else {}
    execution_quality = payload.get("ExecutionQuality", {}) if isinstance(payload, dict) else {}
    stable_decision = payload.get("StableDecision", {}) if isinstance(payload, dict) else {}
    structure_context = ta_data.get("structure_context", {}) if isinstance(ta_data, dict) else {}
    previous_snapshot = previous_snapshot if isinstance(previous_snapshot, dict) else {}
    micro_vwap_delta = _round_display_value(
        structure_context.get("distSessionVwapPct") or 0.0,
        2,
        0.0,
    )
    try:
        micro_orb_state = int(round(float(structure_context.get("openingRangeBreak") or 0.0)))
    except Exception:
        micro_orb_state = 0
    try:
        micro_sweep_state = int(round(float(structure_context.get("sweepReclaimSignal") or 0.0)))
    except Exception:
        micro_sweep_state = 0
    execution_quality_raw_signal = _execution_quality_alert_signal(execution_quality)
    (
        execution_quality_signal,
        execution_quality_raw_signal,
        execution_quality_raw_streak,
    ) = _stabilize_execution_quality_alert_signal(
        execution_quality_raw_signal,
        previous_snapshot,
    )
    stable_lock_until_ts = _parse_utc_timestamp(
        (stable_decision or {}).get("decision_locked_until"),
        default=0,
    )
    stable_lock_active = stable_lock_until_ts > int(time.time())
    if stable_lock_active:
        if previous_snapshot.get("market_structure") is not None:
            pa = dict(pa or {})
            pa["structure"] = previous_snapshot.get("market_structure")
        micro_vwap_delta = previous_snapshot.get("micro_vwap_delta_pct", micro_vwap_delta)
        execution_quality_signal = previous_snapshot.get("execution_quality_signal", execution_quality_signal)
        execution_quality_raw_signal = previous_snapshot.get("execution_quality_raw_signal", execution_quality_raw_signal)
        micro_orb_state = previous_snapshot.get("micro_orb_state", micro_orb_state)
        micro_sweep_state = previous_snapshot.get("micro_sweep_state", micro_sweep_state)
    return {
        "warning_ladder": (regime_state.get("warning_ladder") if isinstance(regime_state, dict) else None),
        "event_regime": (regime_state.get("event_regime") if isinstance(regime_state, dict) else None),
        "breakout_bias": (regime_state.get("breakout_bias") if isinstance(regime_state, dict) else None),
        "market_structure": (pa.get("structure") if isinstance(pa, dict) else None),
        "verdict": payload.get("verdict"),
        "confidence_bucket": _confidence_bucket(payload.get("confidence")),
        "execution_state": (
            execution_state.get("title")
            if isinstance(execution_state, dict) and execution_state.get("title")
            else (execution_state.get("status") if isinstance(execution_state, dict) else None)
        ),
        "execution_quality_signal": execution_quality_signal,
        "execution_quality_raw_signal": execution_quality_raw_signal,
        "execution_quality_raw_streak": execution_quality_raw_streak,
        "micro_vwap_delta_pct": micro_vwap_delta,
        "micro_vwap_band": _vwap_band_label(micro_vwap_delta),
        "micro_vwap_bias": _stabilize_vwap_bias_label(
            micro_vwap_delta,
            previous_snapshot.get("micro_vwap_bias"),
        ),
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
        "id": f"{now_ts}:{snapshot.get('execution_state')}:{payload.get('verdict')}:{price}",
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


def _build_prediction_response(user_id="anonymous"):
    """Central prediction builder used by both HTTP and websocket monitor."""
    now_ts = int(time.time())
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
    initial_regime_state = dict(prediction.get("RegimeState", {}) or {})
    trade_brain_market_data = _build_trade_brain_market_data(ta_data, initial_regime_state)
    raw_learning_direction = _trade_brain_direction_from_prediction(prediction)
    learning_adjustment = trade_brain_service.get_learning_adjustment(
        direction=raw_learning_direction,
        market_data=trade_brain_market_data,
        user_id=user_id,
    )
    prediction = _apply_trade_brain_learning_to_prediction(
        prediction,
        learning_adjustment,
    )
    market_state = _build_market_state_from_prediction(prediction)
    regime_state = dict(prediction.get("RegimeState", {}) or {})
    forecast_state = dict(prediction.get("ForecastState", {}) or {})
    raw_execution_state = dict(prediction.get("ExecutionState", {}) or {})
    decision_status = _evaluate_decision_status(
        verdict=prediction["verdict"],
        confidence=int(prediction["confidence"]),
        ta_data=ta_data,
        trade_guidance=prediction["TradeGuidance"],
        execution_state=raw_execution_state,
        tradeability=prediction.get("tradeability"),
        regime=prediction.get("regime"),
    )
    decision_status = _stabilize_decision_status(decision_status)
    execution_permission = _evaluate_execution_permission(
        decision_status,
        market_state,
    )
    trade_playbook = _evaluate_trade_playbook(
        decision_status,
        execution_permission,
        market_state,
        regime_state,
        prediction["TradeGuidance"],
    )
    trade_playbook = _stabilize_trade_playbook(
        trade_playbook,
        execution_permission,
        decision_status,
    )
    execution_state = _stabilize_execution_state(
        raw_execution_state,
        trade_playbook,
        market_state,
        execution_permission,
        decision_status,
    )
    decision_status = _align_decision_status_with_execution_state(
        decision_status,
        execution_state,
    )
    execution_status = str(execution_state.get("status") or "stand_aside")
    execution_state["status"] = execution_status
    execution_state["title"] = str(
        execution_state.get("title")
        or _humanize_value(execution_status or "stand_aside")
    )
    execution_state["text"] = str(
        execution_state.get("text")
        or decision_status.get("text")
        or "No trade. Conditions are not clean enough yet."
    )
    execution_state["action"] = str(
        execution_state.get("action") or execution_status
    )
    execution_state["actionState"] = str(
        execution_state.get("actionState") or "WAIT"
    )
    aligned_contract = _align_dashboard_response_contract(
        {
            "MarketState": market_state,
            "ExecutionPermission": execution_permission,
            "TradePlaybook": trade_playbook,
            "ExecutionState": execution_state,
        }
    )
    execution_state = dict(aligned_contract.get("ExecutionState") or execution_state)
    decision_status = _align_decision_status_with_execution_state(
        decision_status,
        execution_state,
    )

    stable_decision_state = _load_json_file(
        STABLE_DECISION_STATE_FILE,
        _default_stable_decision_state(now_ts),
    )
    stable_decision_state = _reset_stable_decision_state_if_stale(
        stable_decision_state,
        ta_data,
        now_ts,
    )
    stable_decision_state, buffered_ta_data = _update_stable_decision_buffer(
        stable_decision_state,
        ta_data,
        regime_state,
        now_ts,
    )
    execution_quality = _build_execution_quality_plan(
        buffered_ta_data,
        regime_state,
        decision_status,
        execution_state,
    )
    execution_quality = _apply_trade_brain_learning_to_execution_quality(
        execution_quality,
        learning_adjustment,
    )
    stable_decision, decision_status, execution_state, execution_quality, stable_decision_state = _apply_stable_decision_controls(
        stable_decision_state,
        prediction,
        ta_data,
        regime_state,
        decision_status,
        execution_state,
        execution_quality,
        now_ts=now_ts,
        persist_churn=True,
    )
    _save_json_file(STABLE_DECISION_STATE_FILE, stable_decision_state)
    trade_brain_result = None
    if trade_brain_market_data.get("price") is not None:
        trade_brain_result = trade_brain_service.evaluate_active_trade(
            trade_brain_market_data["price"],
            trade_brain_market_data,
            user_id=user_id,
        )
    final_trade_brain_direction = _trade_brain_direction_from_prediction(
        {
            **prediction,
            "StableDecision": stable_decision,
            "ExecutionQuality": execution_quality,
            "DecisionStatus": decision_status,
        }
    )
    trade_brain_dashboard = trade_brain_service.get_dashboard_payload(
        user_id=user_id,
        market_data=trade_brain_market_data,
        learning_direction=final_trade_brain_direction,
        learning_setup=execution_quality.get("setup"),
    )
    response_learning_adjustment = (
        trade_brain_dashboard.get("learning", {}).get("currentAdjustment")
        if isinstance(trade_brain_dashboard.get("learning"), dict)
        else {}
    )

    response_payload = {
        "status": "success",
        "verdict": prediction["verdict"],
        "confidence": prediction["confidence"],
        "TechnicalAnalysis": ta_data,
        "TradeGuidance": prediction["TradeGuidance"],
        "RegimeState": regime_state,
        "ForecastState": forecast_state,
        "ExecutionState": execution_state,
        "DecisionStatus": decision_status,
        "ExecutionQuality": execution_quality,
        "StableDecision": stable_decision,
        "TradeBrain": trade_brain_dashboard,
        "TradeBrainLearning": copy.deepcopy(response_learning_adjustment or {}),
        "TradeBrainRuntime": {
            "trade": trade_brain_result.get("trade") if isinstance(trade_brain_result, dict) else None,
            "events": trade_brain_result.get("events", []) if isinstance(trade_brain_result, dict) else [],
        },
    }

    return response_payload, 200


def _indicator_monitor_loop():
    """Push websocket notifications whenever tracked indicator values change."""
    last_snapshot = None

    while True:
        try:
            # Skip polling only when there are no live viewers and no background push subscribers.
            if (
                _monitor_state["clients"] <= 0
                and not _has_background_alert_channels()
                and not trade_brain_service.has_active_trades()
            ):
                socketio.sleep(2)
                continue

            payload, status_code = _build_prediction_response()
            if status_code == 200:
                trade_brain_runtime = payload.get("TradeBrainRuntime") if isinstance(payload, dict) else {}
                if isinstance(trade_brain_runtime, dict) and trade_brain_runtime.get("events"):
                    _emit_trade_brain_events(
                        trade_brain_runtime.get("events"),
                        trade=trade_brain_runtime.get("trade"),
                        dashboard=payload.get("TradeBrain"),
                    )
                current_snapshot = _extract_indicator_snapshot(
                    payload,
                    previous_snapshot=last_snapshot,
                )
                current_price = float(((payload.get("TechnicalAnalysis") or {}).get("current_price")) or 0.0)
                if current_price > 0:
                    _update_live_signal_outcomes(current_price, int(time.time()))
                trade_brain_market_data = _build_trade_brain_market_data(
                    payload.get("TechnicalAnalysis"),
                    payload.get("RegimeState"),
                )
                if trade_brain_market_data.get("price") is not None:
                    for result in trade_brain_service.evaluate_all_active_trades(
                        trade_brain_market_data["price"],
                        trade_brain_market_data,
                    ):
                        if result.get("events"):
                            _emit_trade_brain_events(
                                result.get("events"),
                                trade=result.get("trade"),
                                dashboard=result.get("dashboard"),
                            )
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
                                "last_market_structure": "",
                                "last_warning_ladder": "",
                                "last_event_regime": "",
                                "last_breakout_bias": "",
                                "last_verdict": "",
                                "last_confidence_bucket": "",
                                "last_execution_state": "",
                                "last_alert_ts": 0,
                                "last_alert_fingerprint": "",
                                "last_context_alert_ts": 0,
                                "last_context_alert_fingerprint": "",
                                "last_execution_alert_ts": 0,
                                "last_execution_alert_fingerprint": "",
                                "last_diagnostics_alert_ts": 0,
                                "last_diagnostics_alert_fingerprint": "",
                                "last_price_action_alert_ts": 0,
                                "last_price_action_alert_fingerprint": "",
                                "last_boundary_wobble_ts": 0,
                            },
                        )
                        execution_state_payload = payload.get("ExecutionState") or {}
                        market_structure = str(((payload.get("TechnicalAnalysis") or {}).get("price_action") or {}).get("structure") or "")
                        warning_ladder = str((payload.get("RegimeState") or {}).get("warning_ladder") or "")
                        event_regime = str((payload.get("RegimeState") or {}).get("event_regime") or "")
                        breakout_bias = str((payload.get("RegimeState") or {}).get("breakout_bias") or "")
                        verdict = str(payload.get("verdict") or "")
                        confidence_bucket = _confidence_bucket(payload.get("confidence"))
                        execution_state_label = str(
                            execution_state_payload.get("title")
                            or execution_state_payload.get("status")
                            or ""
                        )
                        market_structure_changed = (
                            "market_structure" in notification_changes
                            and bool(market_structure)
                        )
                        microstructure_changed = bool(
                            "micro_vwap_bias" in notification_changes
                            or "micro_vwap_delta_pct" in notification_changes
                            or "micro_orb_state" in notification_changes
                            or "micro_sweep_state" in notification_changes
                        )
                        execution_quality_changed = bool(
                            "execution_quality_signal" in notification_changes
                        )
                        should_alert = bool(
                            market_structure_changed
                            or microstructure_changed
                            or execution_quality_changed
                        )
                        signal_class = "price_action" if should_alert else ""

                        if not should_alert:
                            last_snapshot = current_snapshot
                            continue

                        alert_fingerprint = _notification_fingerprint(notification_changes)
                        if _should_suppress_duplicate_alert(
                            alert_state,
                            alert_fingerprint,
                            signal_class,
                            now_ts,
                        ):
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
                            {},
                            market_structure,
                            ta_data=payload.get("TechnicalAnalysis"),
                            payload=payload,
                            snapshot=current_snapshot,
                        )
                        alert_title = alert_notification.get("title") or "XAUUSD Direction / Grade Changed"
                        alert_message = alert_notification.get("body") or ""
                        client_payload = _sanitize_client_prediction_payload(payload)

                        if _monitor_state["clients"] > 0:
                            socketio.emit(
                                "indicator_change",
                                {
                                    "message": alert_message,
                                    "title": alert_title,
                                    "data": client_payload,
                                    "changes": notification_changes,
                                    "snapshot": current_snapshot,
                                    "verdict": payload.get("verdict"),
                                    "confidence": payload.get("confidence"),
                                    "decision_status": payload.get("DecisionStatus"),
                                    "notification_tag": alert_notification.get("notification_tag"),
                                    "timestamp": now_ts,
                                },
                            )

                        _send_web_push_notifications(
                            changes=notification_changes,
                            rr_signal={},
                            market_structure=market_structure,
                            ta_data=payload.get("TechnicalAnalysis"),
                            payload=payload,
                            snapshot=current_snapshot,
                        )
                        _send_telegram_notification(
                            changes=notification_changes,
                            rr_signal={},
                            market_structure=market_structure,
                            ta_data=payload.get("TechnicalAnalysis"),
                            payload=payload,
                            snapshot=current_snapshot,
                        )
                        _save_json_file(
                            ALERT_STATE_FILE,
                            {
                                "last_market_structure": market_structure,
                                "last_warning_ladder": warning_ladder,
                                "last_event_regime": event_regime,
                                "last_breakout_bias": breakout_bias,
                                "last_verdict": verdict,
                                "last_confidence_bucket": confidence_bucket,
                                "last_execution_state": execution_state_label,
                                "last_alert_ts": now_ts,
                                "last_alert_fingerprint": alert_fingerprint,
                                "last_context_alert_ts": (
                                    now_ts if signal_class == "context" else int(alert_state.get("last_context_alert_ts", 0) or 0)
                                ),
                                "last_context_alert_fingerprint": (
                                    alert_fingerprint if signal_class == "context" else str(alert_state.get("last_context_alert_fingerprint", "") or "")
                                ),
                                "last_execution_alert_ts": (
                                    now_ts if signal_class == "execution" else int(alert_state.get("last_execution_alert_ts", 0) or 0)
                                ),
                                "last_execution_alert_fingerprint": (
                                    alert_fingerprint if signal_class == "execution" else str(alert_state.get("last_execution_alert_fingerprint", "") or "")
                                ),
                                "last_diagnostics_alert_ts": (
                                    now_ts if signal_class == "diagnostics" else int(alert_state.get("last_diagnostics_alert_ts", 0) or 0)
                                ),
                                "last_diagnostics_alert_fingerprint": (
                                    alert_fingerprint if signal_class == "diagnostics" else str(alert_state.get("last_diagnostics_alert_fingerprint", "") or "")
                                ),
                                "last_price_action_alert_ts": (
                                    now_ts if signal_class == "price_action" else int(alert_state.get("last_price_action_alert_ts", 0) or 0)
                                ),
                                "last_price_action_alert_fingerprint": (
                                    alert_fingerprint if signal_class == "price_action" else str(alert_state.get("last_price_action_alert_fingerprint", "") or "")
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


@socketio.on("trade:subscribe")
def _on_trade_subscribe(payload):
    trade_id = str((payload or {}).get("tradeId") or "").strip() if isinstance(payload, dict) else ""
    if not trade_id:
        return
    join_room(_trade_room(trade_id))
    user_id = _get_socket_user_id(payload)
    trade = trade_brain_service.get_active_trade(user_id=user_id)
    if trade and trade.get("id") == trade_id:
        dashboard = trade_brain_service.get_dashboard_payload(user_id=user_id)
        _emit_trade_brain_trade("trade:updated", trade, dashboard, sid=request.sid)


@socketio.on("trade:unsubscribe")
def _on_trade_unsubscribe(payload):
    trade_id = str((payload or {}).get("tradeId") or "").strip() if isinstance(payload, dict) else ""
    if not trade_id:
        return
    leave_room(_trade_room(trade_id))


@socketio.on("trade:evaluate")
def _on_trade_evaluate(payload):
    payload = payload if isinstance(payload, dict) else {}
    user_id = _get_socket_user_id(payload)
    market_data = payload.get("marketData") if isinstance(payload.get("marketData"), dict) else {}
    price = _trade_brain_first_float(payload.get("price"), market_data.get("price"), default=None)
    if price is None:
        socketio.emit("trade:error", {"message": "Missing price."}, to=request.sid)
        return
    result = trade_brain_service.evaluate_active_trade(
        price,
        market_data,
        decision=payload.get("decision"),
        user_id=user_id,
    )
    if result is None:
        socketio.emit("trade:error", {"message": "No active trade to evaluate."}, to=request.sid)
        return
    if result.get("events"):
        _emit_trade_brain_trade("trade:updated", result.get("trade"), result.get("dashboard"), sid=request.sid)
        _emit_trade_brain_events(result.get("events"), trade=result.get("trade"), dashboard=result.get("dashboard"))
        return
    _emit_trade_brain_trade("trade:updated", result.get("trade"), result.get("dashboard"), sid=request.sid)
    _emit_trade_brain_stats(result.get("dashboard"), sid=request.sid)


@socketio.on("emotion:tag")
def _on_emotion_tag(payload):
    payload = payload if isinstance(payload, dict) else {}
    trade_id = str(payload.get("tradeId") or "").strip()
    if not trade_id:
        socketio.emit("trade:error", {"message": "Missing tradeId."}, to=request.sid)
        return
    user_id = _get_socket_user_id(payload)
    try:
        result = trade_brain_service.tag_emotion(
            trade_id,
            payload.get("emotion"),
            note=payload.get("note"),
            price=_trade_brain_float(payload.get("price"), None),
            unrealized_r=_trade_brain_float(payload.get("unrealizedR"), None),
            user_id=user_id,
        )
    except KeyError as exc:
        socketio.emit("trade:error", {"message": str(exc)}, to=request.sid)
        return
    dashboard = trade_brain_service.get_dashboard_payload(user_id=user_id)
    _emit_trade_brain_trade("trade:updated", result.get("trade"), dashboard)
    socketio.emit("emotion:updated", result.get("event"))
    _emit_trade_brain_stats(dashboard)


@socketio.on("stats:request")
def _on_stats_request(payload):
    user_id = _get_socket_user_id(payload)
    dashboard = trade_brain_service.get_dashboard_payload(user_id=user_id)
    _emit_trade_brain_stats(dashboard, sid=request.sid)

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


@app.route('/api/research/status')
def get_research_status():
    try:
        return jsonify(research_status()), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/research/jobs/<job_name>')
def get_research_job(job_name):
    try:
        payload = load_job_bundle(job_name)
        if payload is None:
            return jsonify({"status": "error", "message": f"Job '{job_name}' was not found."}), 404
        return jsonify({"status": "success", **payload}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/research/brief', methods=['POST'])
def post_research_brief():
    payload = request.get_json(silent=True) or {}
    hypothesis = str(payload.get("hypothesis") or "").strip()
    if not hypothesis:
        return jsonify({"status": "error", "message": "Missing hypothesis."}), 400
    job_name = str(payload.get("job_name") or payload.get("jobName") or "").strip()
    if not job_name:
        return jsonify({"status": "error", "message": "Missing job_name."}), 400

    try:
        brief = create_research_brief(
            job_name=job_name,
            hypothesis=hypothesis,
            focus=str(payload.get("focus") or "").strip(),
            constraints=str(payload.get("constraints") or "").strip(),
        )
        return jsonify({"status": "success", "brief": brief}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/trades', methods=['GET', 'POST'])
def handle_trades():
    user_id = _get_request_user_id()
    if request.method == 'POST':
        payload = _enrich_trade_create_payload(request.get_json(silent=True) or {}, user_id)
        try:
            trade = trade_brain_service.enter_trade(payload, user_id=user_id)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        dashboard = trade_brain_service.get_dashboard_payload(user_id=user_id)
        _emit_trade_brain_trade("trade:created", trade, dashboard, user_id=user_id)
        _emit_trade_brain_stats(dashboard, user_id=user_id)
        return jsonify({"status": "success", "trade": trade, "dashboard": dashboard}), 201

    page = int(request.args.get('page', 1) or 1)
    limit = int(request.args.get('limit', 50) or 50)
    status = request.args.get('status')
    listing = trade_brain_service.list_trades(user_id=user_id, status=status, page=page, limit=limit)
    return jsonify({"status": "success", **listing}), 200


@app.route('/api/trades/active')
def get_active_trade():
    user_id = _get_request_user_id()
    dashboard = trade_brain_service.get_dashboard_payload(user_id=user_id)
    return jsonify({"status": "success", "trade": dashboard.get("activeTrade"), "dashboard": dashboard}), 200


@app.route('/api/trades/<trade_id>', methods=['PATCH'])
def patch_trade(trade_id):
    user_id = _get_request_user_id()
    payload = request.get_json(silent=True) or {}
    try:
        trade = trade_brain_service.update_trade(trade_id, payload, user_id=user_id)
    except KeyError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    dashboard = trade_brain_service.get_dashboard_payload(user_id=user_id)
    _emit_trade_brain_trade("trade:updated", trade, dashboard, user_id=user_id)
    _emit_trade_brain_stats(dashboard, user_id=user_id)
    return jsonify({"status": "success", "trade": trade, "dashboard": dashboard}), 200


@app.route('/api/trades/<trade_id>/close', methods=['POST'])
def close_trade(trade_id):
    user_id = _get_request_user_id()
    payload = request.get_json(silent=True) or {}
    exit_price = _trade_brain_first_float(payload.get("exitPrice"), payload.get("price"), default=None)
    if exit_price is None:
        return jsonify({"status": "error", "message": "Missing exitPrice."}), 400
    try:
        result = trade_brain_service.close_trade(
            trade_id,
            exit_price=exit_price,
            reason=str(payload.get("reason") or "MANUAL_EXIT"),
            reasoning=str(payload.get("reasoning") or "Closed manually."),
            emotion=str(payload.get("emotion") or "calm"),
            final_pnl=_trade_brain_float(payload.get("finalPnL"), None),
            final_r=_trade_brain_float(payload.get("finalR"), None),
            user_id=user_id,
        )
    except KeyError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    dashboard = trade_brain_service.get_dashboard_payload(user_id=user_id)
    _emit_trade_brain_trade("trade:closed", result.get("trade"), dashboard, user_id=user_id)
    _emit_trade_brain_stats(dashboard, user_id=user_id)
    return jsonify({"status": "success", "trade": result.get("trade"), "dashboard": dashboard}), 200


@app.route('/api/trades/<trade_id>/review')
def get_trade_review(trade_id):
    user_id = _get_request_user_id()
    try:
        review = trade_brain_service.get_review(trade_id, user_id=user_id)
    except KeyError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    return jsonify({"status": "success", "review": review}), 200


@app.route('/api/trades/<trade_id>/snapshot', methods=['POST'])
def post_trade_snapshot(trade_id):
    user_id = _get_request_user_id()
    payload = request.get_json(silent=True) or {}
    try:
        snapshot = trade_brain_service.record_snapshot(trade_id, payload, user_id=user_id)
    except KeyError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    return jsonify({"status": "success", "snapshot": snapshot}), 200


@app.route('/api/stats')
def get_trade_brain_stats():
    user_id = _get_request_user_id()
    return jsonify({"status": "success", "stats": trade_brain_service.get_stats(user_id=user_id)}), 200


@app.route('/api/analytics/setups')
def get_trade_setup_analytics():
    user_id = _get_request_user_id()
    return jsonify({"status": "success", "setups": trade_brain_service.get_setup_analytics(user_id=user_id)}), 200


@app.route('/api/analytics/emotions')
def get_trade_emotion_analytics():
    user_id = _get_request_user_id()
    return jsonify({"status": "success", "emotions": trade_brain_service.get_emotion_analytics(user_id=user_id)}), 200


@app.route('/api/analytics/sessions')
def get_trade_session_analytics():
    user_id = _get_request_user_id()
    return jsonify({"status": "success", "sessions": trade_brain_service.get_session_analytics(user_id=user_id)}), 200


@app.route('/api/analytics/r-distribution')
def get_trade_r_distribution():
    user_id = _get_request_user_id()
    return jsonify({"status": "success", "distribution": trade_brain_service.get_r_distribution(user_id=user_id)}), 200


@app.route('/api/analytics/monthly')
def get_trade_monthly_analytics():
    user_id = _get_request_user_id()
    return jsonify({"status": "success", "monthly": trade_brain_service.get_monthly_analytics(user_id=user_id)}), 200

@app.route('/api/predict')
def get_prediction():
    try:
        payload, status_code = _build_prediction_response(user_id=_get_request_user_id())
        return jsonify(_sanitize_client_prediction_payload(payload)), status_code
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
        trade_brain_dashboard = trade_brain_service.get_dashboard_payload()
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
            "trade_brain": {
                "active_trade": trade_brain_dashboard.get("activeTrade"),
                "stats": trade_brain_dashboard.get("stats"),
                "notifications": trade_brain_dashboard.get("notifications"),
                "learning": trade_brain_dashboard.get("learning"),
            },
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
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
