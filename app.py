from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
import json
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
    return {
        key: val
        for key, val in changes.items()
        if key not in PUSH_EXCLUDED_FIELDS
    }


def _summarize_changes_for_push(changes):
    labels = {
        "trade_playbook_stage": "Trade Playbook",
        "warning_ladder": "Big Move Risk",
        "event_regime": "Event Regime",
        "breakout_bias": "Breakout Bias",
        "rr_signal_status": "RR200 Signal",
        "rr_signal_grade": "Quant Grade",
        "rr_signal_direction": "RR Direction",
        "market_structure": "Market Structure",
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
    has_verdict = "verdict" in changed_keys
    has_confidence = "confidence_bucket" in changed_keys
    has_permission = "execution_permission" in changed_keys
    has_entry_readiness = "entry_readiness" in changed_keys
    has_exit_urgency = "exit_urgency" in changed_keys

    if has_rr_status or has_rr_grade or has_rr_direction:
        return "XAUUSD RR200 Signal Changed"
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
    return "XAUUSD Execution Permission Changed"


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


def _send_telegram_notification(changes, verdict, confidence, trade_guidance, decision_status, execution_permission):
    if not _telegram_enabled():
        return

    title = _notification_title_for_changes(changes)

    change_summary = _summarize_changes_for_push(changes)
    trade_summary = ""
    if isinstance(trade_guidance, dict):
        trade_summary = trade_guidance.get("summary") or ""
    permission_text = ""
    if isinstance(execution_permission, dict):
        permission_text = str(execution_permission.get("text") or "").strip()
    has_permission_diff = isinstance(changes, dict) and "execution_permission" in changes

    lines = [
        title,
        change_summary,
    ]
    if permission_text and not has_permission_diff:
        lines.append(permission_text)
    else:
        lines.append(f"Verdict: {verdict} ({confidence}%)")
    if _should_append_trade_summary(permission_text, trade_summary):
        lines.append(trade_summary)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "\n".join(line for line in lines if line),
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


def _send_web_push_notifications(changes, verdict, confidence, trade_guidance, decision_status, execution_permission):
    if webpush is None or not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
        return

    subscriptions = _load_subscriptions()
    if not subscriptions:
        return

    title = _notification_title_for_changes(changes)

    change_summary = _summarize_changes_for_push(changes)
    trade_summary = ""
    if isinstance(trade_guidance, dict):
        trade_summary = trade_guidance.get("summary") or ""
    permission_text = ""
    if isinstance(execution_permission, dict):
        permission_text = str(execution_permission.get("text") or "").strip()
    has_permission_diff = isinstance(changes, dict) and "execution_permission" in changes
    body_parts = [change_summary]
    if permission_text and not has_permission_diff:
        body_parts.append(permission_text)
    else:
        body_parts.append(f"Verdict: {verdict} ({confidence}%)")
    if _should_append_trade_summary(permission_text, trade_summary):
        body_parts.append(trade_summary)
    body = " | ".join(part for part in body_parts if part)
    payload = {
        "title": title,
        "body": body,
        "verdict": verdict,
        "confidence": confidence,
        "trade_guidance": trade_guidance,
        "decision_status": decision_status,
        "execution_permission": execution_permission,
        "url": "/",
        "ts": int(time.time()),
    }

    dead_endpoints = []
    for sub in subscriptions:
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps(payload, ensure_ascii=True),
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
    }
    numeric_thresholds = {
        "rsi_14": 0.6,
        "ema_20": 0.25,
        "ema_50": 0.25,
        "adx_14": 0.4,
        "atr_percent": 0.02,
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
            text = "Long setup forming. Bias is constructive, but trigger confirmation is still pending."
            status = "wait"
        elif action_state == "LONG_ACTIVE":
            text = "Safer to look for a buy. Long state is confirmed with acceptable tradeability."
            status = "buy"
        elif action_state == "SETUP_SHORT":
            text = "Short setup forming. Bias is constructive, but trigger confirmation is still pending."
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
    decision_kind = str((decision_status or {}).get("status") or "wait")
    decision_text = str((decision_status or {}).get("text") or "")
    tradeability = str((market_state or {}).get("tradeability") or "Low")
    regime = str((market_state or {}).get("regime") or "unstable")

    text = "No trade. Conditions are not clean enough yet."
    status = "no_trade"

    if action_state in {"LONG_ACTIVE", "SHORT_ACTIVE"} and decision_kind in {"buy", "sell"}:
        text = (
            "Entry Allowed: buy conditions are confirmed."
            if action_state == "LONG_ACTIVE"
            else "Entry Allowed: sell conditions are confirmed."
        )
        status = "entry_allowed"
    elif action_state in {"SETUP_LONG", "SETUP_SHORT"}:
        text = "Watchlist Only: setup is forming, but execution is not confirmed yet."
        status = "watchlist_only"
    elif action_state == "EXIT_RISK" or decision_kind == "exit":
        text = "Exit Recommended: active position quality is deteriorating."
        status = "exit_recommended"
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
    no_trade_reason = prediction.get("noTradeReason") or ""
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

    if no_trade_reason or event_risk_active:
        stable_verdict = "Neutral"
        stable_confidence = min(raw_confidence, 60)
    elif raw_verdict == "Neutral":
        if raw_streak >= PREDICTION_CONFIRMATION_COUNT:
            stable_verdict = "Neutral"
            stable_confidence = min(raw_confidence, 60)
    elif raw_streak >= PREDICTION_CONFIRMATION_COUNT:
        stable_verdict = raw_verdict
        stable_confidence = raw_confidence
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
        "rr_signal_direction": (rr_signal.get("direction") if isinstance(rr_signal, dict) else None),
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
    today = counter.get("today") if isinstance(counter, dict) else {}
    today_total = int((today or {}).get("total", 0) or 0)
    last_delivery_ts = int(counter.get("last_delivery_ts", 0) or 0)
    if today_total >= RR200_MAX_SIGNALS_PER_DAY:
        return False
    if RR200_MIN_SIGNAL_SPACING_SECONDS > 0 and (now_ts - last_delivery_ts) < RR200_MIN_SIGNAL_SPACING_SECONDS:
        return False
    return True


def _rr200_delivery_allowed_for_payload(payload, now_ts):
    rr = payload.get("RR200Signal") if isinstance(payload, dict) else {}
    if not isinstance(rr, dict):
        return False
    grade = str(rr.get("grade") or "")
    if grade not in {"A+ (Quant)", "A (High Accuracy)", "B (Qualified)"}:
        return False
    counter = _load_rr200_counter(now_ts)
    if not _rr200_should_deliver(counter, now_ts):
        return False
    today = counter.get("today") if isinstance(counter.get("today"), dict) else {}
    today_total = int(today.get("total", 0) or 0)
    if grade in {"A+ (Quant)", "A (High Accuracy)"}:
        return True
    return today_total < 2


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
    if not _rr200_should_deliver(counter, now_ts):
        return
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

    return {
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
    }, 200


def _indicator_monitor_loop():
    """Push websocket notifications whenever tracked indicator values change."""
    last_snapshot = None
    last_emit_ts = 0

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
                        and (now_ts - last_emit_ts >= NOTIFY_MIN_INTERVAL_SECONDS)
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
                        rr_signal_direction = str(rr_signal_payload.get("direction") or "")
                        verdict = str(payload.get("verdict") or "")
                        confidence_bucket = _confidence_bucket(payload.get("confidence"))
                        decision_confirmed = bool(decision_payload.get("confirmed"))
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
                        boundary_wobble = _is_warning_boundary_wobble(
                            previous_warning_ladder,
                            warning_ladder,
                            previous_event_regime,
                            event_regime,
                            trade_playbook_stage,
                            permission_status,
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
                        if boundary_wobble:
                            warning_changed = False
                            event_regime_changed = False
                            breakout_bias_changed = False
                        market_structure_changed = "market_structure" in notification_changes
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
                        )
                        rr_signal_grade_changed = (
                            "rr_signal_grade" in notification_changes
                            and bool(rr_signal_grade)
                            and rr_signal_grade != previous_rr_signal_grade
                            and rr_signal_grade in {"A+ (Quant)", "A (High Accuracy)", "B (Qualified)"}
                        )
                        rr_signal_direction_changed = (
                            "rr_signal_direction" in notification_changes
                            and bool(rr_signal_direction)
                            and rr_signal_direction != previous_rr_signal_direction
                            and rr_signal_direction in {"Bullish", "Bearish"}
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
                            playbook_changed
                            or warning_changed
                            or event_regime_changed
                            or breakout_bias_changed
                            or rr_signal_status_changed
                            or rr_signal_grade_changed
                            or rr_signal_direction_changed
                            or entry_readiness_changed
                            or exit_urgency_changed
                            or
                            market_structure_changed
                            or verdict_changed
                            or confidence_changed
                            or (execution_permission_changed and decision_confirmed)
                        )
                        signal_class = ""
                        rr_ready_event = bool(
                            rr_signal_status_changed
                            and rr_signal_status == "ready"
                            and rr_signal_grade in {"A+ (Quant)", "A (High Accuracy)", "B (Qualified)"}
                        )
                        if market_structure_changed:
                            signal_class = "price_action"
                        elif execution_permission_changed and decision_confirmed:
                            signal_class = "execution"
                        elif rr_signal_status_changed or rr_signal_grade_changed or rr_signal_direction_changed:
                            signal_class = "execution"
                        elif playbook_changed:
                            signal_class = "playbook"
                        elif warning_changed or event_regime_changed or breakout_bias_changed:
                            signal_class = "context"
                        elif (
                            market_structure_changed
                            or verdict_changed
                            or confidence_changed
                            or entry_readiness_changed
                            or exit_urgency_changed
                        ):
                            signal_class = "diagnostics"

                        cooldown_suppressed = False

                        if should_alert and _is_forming_setup_wobble(
                            notification_changes,
                            trade_playbook_payload,
                            execution_permission_payload,
                            decision_payload,
                        ):
                            should_alert = False
                        if should_alert and permission_status == "exit_recommended":
                            should_alert = NOTIFY_EXIT_READS
                        if should_alert and rr_ready_event and not _rr200_delivery_allowed_for_payload(payload, now_ts):
                            should_alert = False
                            cooldown_suppressed = True
                        if should_alert:
                            last_trade_playbook_stage = str(alert_state.get("last_trade_playbook_stage", ""))
                            last_execution_permission = str(alert_state.get("last_execution_permission", ""))
                            last_market_structure = str(alert_state.get("last_market_structure", ""))
                            last_warning_ladder = str(alert_state.get("last_warning_ladder", ""))
                            last_event_regime = str(alert_state.get("last_event_regime", ""))
                            last_breakout_bias = str(alert_state.get("last_breakout_bias", ""))
                            last_verdict = str(alert_state.get("last_verdict", ""))
                            last_confidence_bucket = str(alert_state.get("last_confidence_bucket", ""))
                            last_entry_readiness = str(alert_state.get("last_entry_readiness", ""))
                            last_exit_urgency = str(alert_state.get("last_exit_urgency", ""))
                            last_rr_signal_status = str(alert_state.get("last_rr_signal_status", ""))
                            last_rr_signal_grade = str(alert_state.get("last_rr_signal_grade", ""))
                            last_rr_signal_direction = str(alert_state.get("last_rr_signal_direction", ""))
                            last_alert_ts = int(alert_state.get("last_alert_ts", 0) or 0)
                            last_playbook_alert_ts = int(alert_state.get("last_playbook_alert_ts", 0) or 0)
                            last_context_alert_ts = int(alert_state.get("last_context_alert_ts", 0) or 0)
                            last_execution_alert_ts = int(alert_state.get("last_execution_alert_ts", 0) or 0)
                            last_diagnostics_alert_ts = int(alert_state.get("last_diagnostics_alert_ts", 0) or 0)
                            last_price_action_alert_ts = int(alert_state.get("last_price_action_alert_ts", 0) or 0)
                            last_boundary_wobble_ts = int(alert_state.get("last_boundary_wobble_ts", 0) or 0)
                            duplicate_playbook = playbook_changed and trade_playbook_stage == last_trade_playbook_stage
                            duplicate_warning = warning_changed and warning_ladder == last_warning_ladder
                            duplicate_event_regime = event_regime_changed and event_regime == last_event_regime
                            duplicate_breakout_bias = breakout_bias_changed and breakout_bias == last_breakout_bias
                            duplicate_execution = execution_permission_changed and execution_permission == last_execution_permission
                            duplicate_structure = market_structure_changed and market_structure == last_market_structure
                            duplicate_verdict = verdict_changed and verdict == last_verdict
                            duplicate_confidence = confidence_changed and confidence_bucket == last_confidence_bucket
                            duplicate_entry_readiness = entry_readiness_changed and entry_readiness == last_entry_readiness
                            duplicate_exit_urgency = exit_urgency_changed and exit_urgency == last_exit_urgency
                            duplicate_rr_signal_status = rr_signal_status_changed and rr_signal_status == last_rr_signal_status
                            duplicate_rr_signal_grade = rr_signal_grade_changed and rr_signal_grade == last_rr_signal_grade
                            duplicate_rr_signal_direction = rr_signal_direction_changed and rr_signal_direction == last_rr_signal_direction
                            if (
                                duplicate_playbook
                                or duplicate_warning
                                or duplicate_event_regime
                                or duplicate_breakout_bias
                                or duplicate_execution
                                or duplicate_structure
                                or duplicate_verdict
                                or duplicate_confidence
                                or duplicate_entry_readiness
                                or duplicate_exit_urgency
                                or duplicate_rr_signal_status
                                or duplicate_rr_signal_grade
                                or duplicate_rr_signal_direction
                            ) and (now_ts - last_alert_ts) < ALERT_COOLDOWN_SECONDS:
                                cooldown_suppressed = True
                                should_alert = False
                            if should_alert and signal_class == "playbook" and (now_ts - last_playbook_alert_ts) < ALERT_CLASS_COOLDOWN_SECONDS:
                                cooldown_suppressed = True
                                should_alert = False
                            if should_alert and signal_class == "context" and (now_ts - last_context_alert_ts) < ALERT_CONTEXT_COOLDOWN_SECONDS:
                                cooldown_suppressed = True
                                should_alert = False
                            if should_alert and signal_class == "execution" and (now_ts - last_execution_alert_ts) < ALERT_CLASS_COOLDOWN_SECONDS:
                                cooldown_suppressed = True
                                should_alert = False
                            if should_alert and signal_class == "diagnostics" and (now_ts - last_diagnostics_alert_ts) < ALERT_CONTEXT_COOLDOWN_SECONDS:
                                cooldown_suppressed = True
                                should_alert = False
                            if should_alert and signal_class == "price_action" and (now_ts - last_price_action_alert_ts) < PRICE_ACTION_ALERT_COOLDOWN_SECONDS:
                                cooldown_suppressed = True
                                should_alert = False
                            if (
                                should_alert
                                and boundary_wobble
                                and not market_structure_changed
                                and (now_ts - last_boundary_wobble_ts) < BOUNDARY_WOBBLE_COOLDOWN_SECONDS
                            ):
                                should_alert = False
                        if (
                            should_alert
                            and boundary_wobble
                            and not market_structure_changed
                        ):
                            should_alert = False
                        if not should_alert:
                            if boundary_wobble:
                                alert_state["last_boundary_wobble_ts"] = now_ts
                                _save_json_file(ALERT_STATE_FILE, alert_state)
                            if not cooldown_suppressed:
                                last_snapshot = current_snapshot
                            continue

                        _record_live_signal_outcome(
                            payload,
                            current_snapshot,
                            notification_changes,
                            now_ts,
                        )

                        alert_title = _notification_title_for_changes(notification_changes)
                        alert_message = _summarize_changes_for_push(notification_changes)

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
                            verdict=payload.get("verdict"),
                            confidence=payload.get("confidence"),
                            trade_guidance=payload.get("TradeGuidance"),
                            decision_status=payload.get("DecisionStatus"),
                            execution_permission=payload.get("ExecutionPermission"),
                        )
                        _send_telegram_notification(
                            changes=notification_changes,
                            verdict=payload.get("verdict"),
                            confidence=payload.get("confidence"),
                            trade_guidance=payload.get("TradeGuidance"),
                            decision_status=payload.get("DecisionStatus"),
                            execution_permission=payload.get("ExecutionPermission"),
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
                                "last_boundary_wobble_ts": (
                                    now_ts if boundary_wobble else int(alert_state.get("last_boundary_wobble_ts", 0) or 0)
                                ),
                            },
                        )
                        last_emit_ts = now_ts
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

    if not report_path.exists():
        return jsonify({
            "status": "error",
            "message": "Autoresearch report not found. Run tools/autoresearch_loop.py first."
        }), 404

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to read autoresearch report: {e}"}), 500

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
    active_matches_best_candidate = bool(recommendation.get("matches_active_strategy"))
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

    return jsonify({
        "status": "success",
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
