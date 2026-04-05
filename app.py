from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
import json
import os
import re
import threading
import time
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


MONITOR_INTERVAL_SECONDS = _read_int_env("INDICATOR_MONITOR_INTERVAL_SECONDS", 3, 2)
NOTIFY_MIN_INTERVAL_SECONDS = _read_int_env("INDICATOR_NOTIFY_MIN_INTERVAL_SECONDS", 6, 1)
PREDICTION_CONFIRMATION_COUNT = _read_int_env("PREDICTION_CONFIRMATION_COUNT", 2, 1)
DECISION_CONFIRMATION_COUNT = _read_int_env("DECISION_CONFIRMATION_COUNT", 2, 1)
ENTER_DECISION_CONFIRMATION_COUNT = _read_int_env("ENTER_DECISION_CONFIRMATION_COUNT", 2, 1)
EXIT_DECISION_CONFIRMATION_COUNT = _read_int_env("EXIT_DECISION_CONFIRMATION_COUNT", 3, 1)
WAIT_DECISION_CONFIRMATION_COUNT = _read_int_env("WAIT_DECISION_CONFIRMATION_COUNT", 4, 1)
DIRECTIONAL_REVERSAL_CONFIRMATION_COUNT = _read_int_env("DIRECTIONAL_REVERSAL_CONFIRMATION_COUNT", 4, 1)
ALERT_COOLDOWN_SECONDS = _read_int_env("ALERT_COOLDOWN_SECONDS", 900, 0)
DECISION_FLIP_MIN_HOLD_SECONDS = _read_int_env("DECISION_FLIP_MIN_HOLD_SECONDS", 300, 0)
NOTIFY_EXIT_READS = _read_bool_env("NOTIFY_EXIT_READS", True)
PUSH_EXCLUDED_FIELDS = {
    "candle_pattern",
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
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_CLAIMS_SUBJECT = os.getenv("VAPID_CLAIMS_SUBJECT", "mailto:alerts@example.com")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID", "").strip()
CHANGE_SUMMARY_ORDER = [
    "market_structure",
    "execution_permission",
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
        "market_structure": "Market Structure",
        "execution_permission": "Execution Permission",
    }
    ordered_keys = [key for key in CHANGE_SUMMARY_ORDER if key in changes]
    ordered_keys.extend(key for key in changes if key not in ordered_keys)
    parts = []
    for key in ordered_keys[:3]:
        val = changes.get(key, {})
        prev = val.get("previous") if isinstance(val, dict) else None
        cur = val.get("current") if isinstance(val, dict) else None
        parts.append(f"{labels.get(key, key)}: {prev} -> {cur}")
    return " | ".join(parts) if parts else "Execution permission changed"


def _telegram_enabled():
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _has_background_alert_channels():
    return _has_push_subscribers() or _telegram_enabled()


def _send_telegram_notification(changes, verdict, confidence, trade_guidance, decision_status, execution_permission):
    if not _telegram_enabled():
        return

    title = "XAUUSD Execution Permission Changed"

    change_summary = _summarize_changes_for_push(changes)
    trade_summary = ""
    if isinstance(trade_guidance, dict):
        trade_summary = trade_guidance.get("summary") or ""
    permission_text = ""
    if isinstance(execution_permission, dict):
        permission_text = str(execution_permission.get("text") or "").strip()

    lines = [
        title,
        change_summary,
    ]
    if permission_text:
        lines.append(permission_text)
    else:
        lines.append(f"Verdict: {verdict} ({confidence}%)")
    if trade_summary:
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

    title = "XAUUSD Execution Permission Changed"

    change_summary = _summarize_changes_for_push(changes)
    trade_summary = ""
    if isinstance(trade_guidance, dict):
        trade_summary = trade_guidance.get("summary") or ""
    permission_text = ""
    if isinstance(execution_permission, dict):
        permission_text = str(execution_permission.get("text") or "").strip()
    body_parts = [change_summary]
    if permission_text:
        body_parts.append(permission_text)
    else:
        body_parts.append(f"Verdict: {verdict} ({confidence}%)")
    if trade_summary:
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
        "verdict",
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


def _evaluate_decision_status(verdict, confidence, ta_data, trade_guidance):
    market_state = (ta_data or {}).get("_prediction_market_state") or {}
    action_state = str(market_state.get("action_state") or "")
    trend = str((ta_data or {}).get("ema_trend") or "Neutral")
    price_action = (ta_data or {}).get("price_action") or {}
    mtf = (ta_data or {}).get("multi_timeframe") or {}
    structure = str(price_action.get("structure") or "")
    pattern = str(price_action.get("latest_candle_pattern") or "")
    alignment = str(mtf.get("alignment_label") or "Mixed / Low Alignment")
    summary = str((trade_guidance or {}).get("summary") or "")
    buy_level = str((trade_guidance or {}).get("buyLevel") or "Weak")
    sell_level = str((trade_guidance or {}).get("sellLevel") or "Weak")
    exit_level = str((trade_guidance or {}).get("exitLevel") or "Low")
    has_bullish_structure = bool(re.search(r"bullish|breakout|continuation", structure, re.IGNORECASE))
    has_bearish_structure = bool(re.search(r"bearish|breakdown|rejection", structure, re.IGNORECASE))
    has_doji_like_pattern = bool(re.search(r"doji|indecision|spinning top", pattern, re.IGNORECASE))
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
            not has_doji_like_pattern and not no_clean_trigger,
            buy_level in {"Watch", "Strong"} and exit_level in {"Low", "Medium"},
        ]
        sell_checks = [
            verdict == "Bearish" and confidence >= 70,
            trend == "Bearish" and has_bearish_structure,
            bearish_alignment,
            not has_doji_like_pattern and not no_clean_trigger,
            sell_level in {"Watch", "Strong"} and exit_level in {"Low", "Medium"},
        ]
        exit_checks = [
            verdict == "Neutral" or "mixed" in alignment.lower() or no_clean_trigger,
            exit_level == "High" or confidence < 60,
            structure_conflict,
            has_doji_like_pattern,
            buy_level == "Weak" and sell_level == "Weak",
        ]
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
        not has_doji_like_pattern and not no_clean_trigger,
        buy_level in {"Watch", "Strong"} and exit_level in {"Low", "Medium"},
    ]
    sell_checks = [
        verdict == "Bearish" and confidence >= 70,
        trend == "Bearish" and has_bearish_structure,
        bearish_alignment,
        not has_doji_like_pattern and not no_clean_trigger,
        sell_level in {"Watch", "Strong"} and exit_level in {"Low", "Medium"},
    ]
    exit_checks = [
        verdict == "Neutral" or "mixed" in alignment.lower() or no_clean_trigger,
        exit_level == "High" or confidence < 60,
        structure_conflict,
        has_doji_like_pattern,
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

    return {
        "text": text,
        "status": status,
        "actionState": action_state,
        "decisionStatus": decision_kind,
    }


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
    regime = ta_data.get("volatility_regime", {}) if isinstance(ta_data, dict) else {}
    mtf = ta_data.get("multi_timeframe", {}) if isinstance(ta_data, dict) else {}
    trade_guidance = payload.get("TradeGuidance", {}) if isinstance(payload, dict) else {}
    return {
        "market_structure": (pa.get("structure") if isinstance(pa, dict) else None),
        "execution_permission": ((payload.get("ExecutionPermission") or {}).get("text")),
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

    prediction = _stabilize_prediction(
        compute_prediction_from_ta(ta_data),
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

    return {
        "status": "success",
        "verdict": prediction["verdict"],
        "confidence": prediction["confidence"],
        "TechnicalAnalysis": ta_data,
        "TradeGuidance": prediction["TradeGuidance"],
        "MarketState": prediction.get("MarketState", {}),
        "DecisionStatus": decision_status,
        "ExecutionPermission": execution_permission,
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
                            {"last_execution_permission": "", "last_market_structure": "", "last_alert_ts": 0},
                        )
                        decision_payload = payload.get("DecisionStatus") or {}
                        execution_permission_payload = payload.get("ExecutionPermission") or {}
                        execution_permission = str(execution_permission_payload.get("text") or "")
                        permission_status = str(execution_permission_payload.get("status") or "no_trade")
                        market_structure = str(((payload.get("TechnicalAnalysis") or {}).get("price_action") or {}).get("structure") or "")
                        decision_confirmed = bool(decision_payload.get("confirmed"))
                        market_structure_changed = "market_structure" in notification_changes and bool(market_structure)
                        execution_permission_changed = "execution_permission" in notification_changes and bool(execution_permission)
                        should_alert = bool(
                            market_structure_changed
                            or (execution_permission_changed and decision_confirmed)
                        )
                        if should_alert and permission_status == "exit_recommended":
                            should_alert = NOTIFY_EXIT_READS
                        if should_alert:
                            last_execution_permission = str(alert_state.get("last_execution_permission", ""))
                            last_market_structure = str(alert_state.get("last_market_structure", ""))
                            last_alert_ts = int(alert_state.get("last_alert_ts", 0) or 0)
                            duplicate_execution = execution_permission_changed and execution_permission == last_execution_permission
                            duplicate_structure = market_structure_changed and market_structure == last_market_structure
                            if (duplicate_execution or duplicate_structure) and (now_ts - last_alert_ts) < ALERT_COOLDOWN_SECONDS:
                                should_alert = False
                        if not should_alert:
                            last_snapshot = current_snapshot
                            continue

                        if market_structure_changed and not execution_permission_changed:
                            alert_title = "Market Structure Changed"
                            alert_message = "Market structure changed"
                        elif execution_permission_changed and not market_structure_changed:
                            alert_title = "Execution Permission Changed"
                            alert_message = "Execution permission changed"
                        else:
                            alert_title = "Market Structure / Execution Changed"
                            alert_message = "Market structure or execution permission changed"

                        if _monitor_state["clients"] > 0:
                            socketio.emit(
                                "indicator_change",
                                {
                                    "message": alert_message,
                                    "title": alert_title,
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
                        _save_json_file(
                            ALERT_STATE_FILE,
                            {
                                "last_execution_permission": execution_permission,
                                "last_market_structure": market_structure,
                                "last_alert_ts": now_ts,
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
    promoted_report_path = base_dir / "data" / "swarm" / "promoted_result.json"
    swarm_report_path = base_dir / "data" / "swarm" / "latest_result.json"
    decision_report_path = base_dir / "data" / "swarm" / "promotion_decision.json"
    legacy_report_path = base_dir / "tools" / "reports" / "autoresearch_last.json"

    def _read_report(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read {path.name}: {e}") from e

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

    if promoted_report_path.exists() or swarm_report_path.exists():
        try:
            promoted_report = _read_report(promoted_report_path) if promoted_report_path.exists() else None
            latest_report = _read_report(swarm_report_path) if swarm_report_path.exists() else None
            decision_report = _read_report(decision_report_path) if decision_report_path.exists() else None
        except RuntimeError as e:
            return jsonify({"status": "error", "message": str(e)}), 500

        display_report = promoted_report or latest_report or {}
        best = _best_params_from(display_report)
        latest_best = _best_params_from(latest_report)
        latest_summary = _summary_from(latest_report)
        display_summary = _summary_from(display_report)
        top_results = _top_results_from(latest_report) or _top_results_from(display_report)
        top_result = top_results[0] if isinstance(top_results, list) and top_results else {}
        roi = best.get("roi") or display_summary.get("roi")
        latest_generated_at = (latest_report or {}).get("generated_at")
        latest_run_promote = bool((latest_report or {}).get("promote"))
        has_active_promoted_strategy = bool(promoted_report and _best_params_from(promoted_report))
        decision_reason = (decision_report or {}).get("promotion_reason") or (latest_report or {}).get("promotion_reason")
        active_strategy_reason = (
            "Active promoted strategy is available."
            if has_active_promoted_strategy
            else "No promoted strategy has been saved yet."
        )

        return jsonify({
            "status": "success" if display_report else "idle",
            "generated_at": display_report.get("generated_at"),
            "latest_generated_at": latest_generated_at,
            "promote": latest_run_promote,
            "latest_run_promote": latest_run_promote,
            "has_active_promoted_strategy": has_active_promoted_strategy,
            "promotion_reason": decision_reason or "Swarm has not produced a fresh result yet.",
            "active_strategy_reason": active_strategy_reason,
            "best_params": best,
            "latest_best_params": latest_best,
            "roi": roi,
            "winning_ema": f"{best.get('ema_short', '-')}/{best.get('ema_long', '-')}",
            "median_score": roi,
            "pass_rate": display_summary.get("pass_rate"),
            "summary": {
                "roi": roi,
                "top_ranked_candidates": len(top_results),
                "winning_ema": f"{best.get('ema_short', '-')}/{best.get('ema_long', '-')}",
                "winning_rsi": f"{best.get('rsi_overbought', '-')}/{best.get('rsi_oversold', '-')}",
                "winning_cmf": best.get("cmf_window"),
                "top_candidate_roi": top_result.get("roi") if isinstance(top_result, dict) else None,
                "trades": display_summary.get("trades"),
                "max_drawdown": display_summary.get("max_drawdown"),
                "profit_factor": display_summary.get("profit_factor"),
                "expectancy": display_summary.get("expectancy"),
                "latest_run_roi": latest_summary.get("roi"),
                "latest_run_promote": latest_run_promote,
                "has_active_promoted_strategy": has_active_promoted_strategy,
            },
            "checks": (decision_report or {}).get("checks") or (latest_report or {}).get("checks") or {},
        })

    if not legacy_report_path.exists():
        return jsonify({
            "status": "error",
            "message": "Autoresearch report not found. Run tools/swarm_optimize.py first."
        }), 404

    try:
        report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to read legacy autoresearch report: {e}"}), 500

    best = report.get("best") or {}
    summary = best.get("summary") if isinstance(best, dict) else {}
    best_params = _best_params_from(report)
    roi = summary.get("roi") if isinstance(summary, dict) else None
    promoted = bool(report.get("promote", False))

    return jsonify({
        "status": "success",
        "generated_at": report.get("generated_at"),
        "latest_generated_at": report.get("generated_at"),
        "promote": promoted,
        "latest_run_promote": promoted,
        "has_active_promoted_strategy": promoted,
        "promotion_reason": report.get("promotion_reason", ""),
        "active_strategy_reason": (
            "Active promoted strategy is available."
            if promoted
            else "No promoted strategy has been saved yet."
        ),
        "best_params": best_params,
        "roi": roi,
        "winning_ema": f"{best_params.get('ema_short', '-')}/{best_params.get('ema_long', '-')}",
        "median_score": best.get("median_score") if isinstance(best, dict) else None,
        "pass_rate": best.get("pass_rate") if isinstance(best, dict) else None,
        "summary": {
            **(summary if isinstance(summary, dict) else {}),
            "roi": roi,
            "winning_ema": f"{best_params.get('ema_short', '-')}/{best_params.get('ema_long', '-')}",
            "winning_rsi": f"{best_params.get('rsi_overbought', '-')}/{best_params.get('rsi_oversold', '-')}",
            "winning_cmf": best_params.get("cmf_window"),
            "latest_run_roi": roi,
            "latest_run_promote": promoted,
            "has_active_promoted_strategy": promoted,
        },
    })

@app.route('/api/predict')
def get_prediction():
    try:
        payload, status_code = _build_prediction_response()
        return jsonify(payload), status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


_ensure_monitor_started()

if __name__ == '__main__':
    # Run on 0.0.0.0 to allow access from other devices on the local network (like your phone)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
