from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
import json
import os
import threading
import time
from pathlib import Path
from tools import predict_gold

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

MONITOR_INTERVAL_SECONDS = max(3, int(os.getenv("INDICATOR_MONITOR_INTERVAL_SECONDS", "5")))
NOTIFY_MIN_INTERVAL_SECONDS = max(1, int(os.getenv("INDICATOR_NOTIFY_MIN_INTERVAL_SECONDS", "6")))
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
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_CLAIMS_SUBJECT = os.getenv("VAPID_CLAIMS_SUBJECT", "mailto:alerts@example.com")


def _load_subscriptions():
    if not SUBSCRIPTIONS_FILE.exists():
        return []
    try:
        data = json.loads(SUBSCRIPTIONS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


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
        "verdict": "Verdict",
        "confidence": "Confidence",
        "trend": "Trend",
        "market_structure": "Structure",
        "candle_pattern": "Pattern",
        "rsi_14": "RSI",
        "market_regime": "Regime",
        "alignment": "Alignment",
    }
    parts = []
    for key, val in list(changes.items())[:3]:
        prev = val.get("previous") if isinstance(val, dict) else None
        cur = val.get("current") if isinstance(val, dict) else None
        parts.append(f"{labels.get(key, key)}: {prev} -> {cur}")
    return " | ".join(parts) if parts else "Indicators changed"


def _send_web_push_notifications(changes, verdict, confidence):
    if webpush is None or not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
        return

    subscriptions = _load_subscriptions()
    if not subscriptions:
        return

    body = _summarize_changes_for_push(changes)
    payload = {
        "title": "XAUUSD Indicator Update",
        "body": body,
        "verdict": verdict,
        "confidence": confidence,
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
        "candle_pattern",
        "market_regime",
        "alignment",
    }
    numeric_thresholds = {
        "confidence": 2.0,
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


def _extract_indicator_snapshot(payload):
    """Build a compact snapshot used to detect indicator changes."""
    ta_data = payload.get("TechnicalAnalysis", {}) if isinstance(payload, dict) else {}
    pa = ta_data.get("price_action", {}) if isinstance(ta_data, dict) else {}
    regime = ta_data.get("volatility_regime", {}) if isinstance(ta_data, dict) else {}
    mtf = ta_data.get("multi_timeframe", {}) if isinstance(ta_data, dict) else {}

    return {
        "verdict": payload.get("verdict"),
        "confidence": payload.get("confidence"),
        "trend": ta_data.get("ema_trend"),
        "market_structure": pa.get("structure"),
        "candle_pattern": pa.get("latest_candle_pattern"),
        "rsi_14": ta_data.get("rsi_14"),
        "ema_20": ta_data.get("ema_20"),
        "ema_50": ta_data.get("ema_50"),
        "market_regime": regime.get("market_regime"),
        "alignment": mtf.get("alignment_label"),
        "adx_14": regime.get("adx_14"),
        "atr_percent": regime.get("atr_percent"),
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
    # Fetch data using the existing script logic
    ta_data = predict_gold.get_technical_analysis()
    sa_data = predict_gold.get_sentiment_analysis()

    if not isinstance(ta_data, dict):
        ta_data = {"error": "Technical analysis payload is invalid."}

    if not isinstance(sa_data, list):
        sa_data = []

    if ta_data.get("error"):
        return {
            "status": "error",
            "message": ta_data["error"],
            "verdict": "Neutral",
            "confidence": 50,
            "TechnicalAnalysis": ta_data,
            "SentimentalAnalysis": sa_data,
        }, 502

    # Calculate verdict using trend + multi-timeframe + regime + price action
    verdict = "Neutral"
    confidence = 50

    trend = ta_data.get("ema_trend", "Neutral")
    volume = ta_data.get("volume_analysis", {}).get("overall_volume_signal", "Neutral")
    regime = ta_data.get("volatility_regime", {}).get("market_regime", "Range-Bound")
    adx_14 = ta_data.get("volatility_regime", {}).get("adx_14", 0)
    mtf = ta_data.get("multi_timeframe", {})
    alignment_score = mtf.get("alignment_score", 0)
    alignment_label = mtf.get("alignment_label", "Mixed / Low Alignment")
    pa_struct = ta_data.get("price_action", {}).get("structure", "Consolidating")
    pa_pattern = ta_data.get("price_action", {}).get("latest_candle_pattern", "None")

    bull_score = 0.0
    bear_score = 0.0

    # 1. 1H trend baseline
    if trend == "Bullish":
        bull_score += 2.5
    elif trend == "Bearish":
        bear_score += 2.5

    # 2. Multi-timeframe alignment
    if alignment_score > 0:
        bull_score += min(alignment_score, 3) * 1.2
    elif alignment_score < 0:
        bear_score += min(abs(alignment_score), 3) * 1.2

    # 3. Regime filter (trending environments should have more confidence)
    if regime == "Trending" and adx_14 >= 22:
        if trend == "Bullish":
            bull_score += 1.0
        elif trend == "Bearish":
            bear_score += 1.0
    elif regime == "Weak Trend":
        if trend == "Bullish":
            bull_score += 0.4
        elif trend == "Bearish":
            bear_score += 0.4

    # 4. Volume & orderflow (if available)
    if "Accumulation" in volume:
        bull_score += 2.0
    elif "Buying Bias" in volume:
        bull_score += 1.0
    if "Distribution" in volume:
        bear_score += 2.0
    elif "Selling Bias" in volume:
        bear_score += 1.0

    # 5. Price action
    if "Bullish Breakout" in pa_struct:
        bull_score += 2.0
    elif "Bearish Breakdown" in pa_struct:
        bear_score += 2.0
    elif "Bullish Structure" in pa_struct or "Bullish Drift" in pa_struct or "Bullish Pressure" in pa_struct:
        bull_score += 1.0
    elif "Bearish Structure" in pa_struct or "Bearish Drift" in pa_struct or "Bearish Pressure" in pa_struct:
        bear_score += 1.0

    if "Bullish Engulfing" in pa_pattern:
        bull_score += 1.0
    elif "Bearish Engulfing" in pa_pattern:
        bear_score += 1.0
    elif "Bullish Hammer" in pa_pattern:
        bull_score += 0.6
    elif "Bearish Shooting Star" in pa_pattern:
        bear_score += 0.6

    # 6. RSI momentum
    rsi = ta_data.get("rsi_14", 50)
    if rsi < 30:
        bull_score += 1.5
    elif rsi < 40:
        bull_score += 0.6
    if rsi > 70:
        bear_score += 1.5
    elif rsi > 60:
        bear_score += 0.6

    # Verdict decision using directional margin
    score_diff = bull_score - bear_score
    score_margin = abs(score_diff)
    evidence_total = bull_score + bear_score

    if score_diff >= 1.2:
        verdict = "Bullish"
    elif score_diff <= -1.2:
        verdict = "Bearish"
    else:
        verdict = "Neutral"

    # Confidence calibration with penalties for lower-quality contexts
    base_conf = 50 + (score_margin * 8.0) + (evidence_total * 1.4)
    penalty = 0.0

    if regime == "Range-Bound":
        penalty += 8.0
    elif regime == "Weak Trend":
        penalty += 3.0

    if "N/A" in volume:
        penalty += 5.0

    if alignment_label == "Mixed / Low Alignment":
        penalty += 6.0

    confidence = base_conf - penalty
    if verdict == "Neutral":
        confidence = min(confidence, 63)

    # Ensure it stays within reasonable bounds
    confidence = round(min(max(confidence, 50), 95))

    return {
        "status": "success",
        "verdict": verdict,
        "confidence": confidence,
        "TechnicalAnalysis": ta_data,
        "SentimentalAnalysis": sa_data,
    }, 200


def _indicator_monitor_loop():
    """Push websocket notifications whenever tracked indicator values change."""
    last_snapshot = None
    last_emit_ts = 0

    while True:
        try:
            # Skip polling only when there are no live viewers and no background push subscribers.
            if _monitor_state["clients"] <= 0 and not _has_push_subscribers():
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
                        if _monitor_state["clients"] > 0:
                            socketio.emit(
                                "indicator_change",
                                {
                                    "message": "Indicators changed",
                                    "changes": notification_changes,
                                    "snapshot": current_snapshot,
                                    "verdict": payload.get("verdict"),
                                    "confidence": payload.get("confidence"),
                                    "timestamp": now_ts,
                                },
                            )

                        _send_web_push_notifications(
                            changes=notification_changes,
                            verdict=payload.get("verdict"),
                            confidence=payload.get("confidence"),
                        )
                        last_emit_ts = now_ts
                last_snapshot = current_snapshot
        except Exception:
            # Keep monitor alive if any external API call fails.
            pass

        socketio.sleep(MONITOR_INTERVAL_SECONDS)


@socketio.on("connect")
def _on_socket_connect():
    with _monitor_lock:
        _monitor_state["clients"] += 1
        if not _monitor_state["started"]:
            socketio.start_background_task(_indicator_monitor_loop)
            _monitor_state["started"] = True


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
    report_path = Path(__file__).resolve().parent / "tools" / "reports" / "autoresearch_last.json"

    if not report_path.exists():
        return jsonify({
            "status": "error",
            "message": "Autoresearch report not found. Run tools/autoresearch_loop.py first."
        }), 404

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to read autoresearch report: {e}"}), 500

    best = report.get("best") or {}
    summary = best.get("summary") if isinstance(best, dict) else {}

    return jsonify({
        "status": "success",
        "generated_at": report.get("generated_at"),
        "promote": report.get("promote", False),
        "promotion_reason": report.get("promotion_reason", ""),
        "best_params": best.get("params", {}) if isinstance(best, dict) else {},
        "median_score": best.get("median_score") if isinstance(best, dict) else None,
        "pass_rate": best.get("pass_rate") if isinstance(best, dict) else None,
        "summary": summary if isinstance(summary, dict) else {},
    })

@app.route('/api/predict')
def get_prediction():
    try:
        payload, status_code = _build_prediction_response()
        return jsonify(payload), status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to allow access from other devices on the local network (like your phone)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
