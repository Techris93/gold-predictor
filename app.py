from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
import json
import os
import threading
import time
from pathlib import Path
from tools import predict_gold
from tools.signal_engine import (
    classify_market_sentiment as shared_classify_market_sentiment,
    compute_prediction_from_ta,
    compute_trade_guidance as shared_compute_trade_guidance,
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


MONITOR_INTERVAL_SECONDS = _read_int_env("INDICATOR_MONITOR_INTERVAL_SECONDS", 5, 3)
NOTIFY_MIN_INTERVAL_SECONDS = _read_int_env("INDICATOR_NOTIFY_MIN_INTERVAL_SECONDS", 6, 1)
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
CHANGE_SUMMARY_ORDER = [
    "verdict",
    "confidence",
    "sentiment_label",
    "trend",
    "market_structure",
    "candle_pattern",
    "market_regime",
    "alignment",
    "trade_sell_setup",
    "trade_buy_setup",
    "trade_exit_warning",
    "rsi_14",
    "adx_14",
    "atr_percent",
]
SENTIMENT_WEIGHTED_SIGNALS = [
    {
        "terms": [
            "safe haven",
            "geopolitical",
            "war",
            "conflict",
            "middle east tension",
            "risk-off",
        ],
        "weight": 2,
        "side": "bull",
    },
    {
        "terms": [
            "gold rises",
            "gold rise",
            "higher",
            "rebound",
            "gains",
            "surges",
            "bullish",
            "buy",
            "demand",
            "record high",
            "inflows",
            "strong demand",
        ],
        "weight": 1,
        "side": "bull",
    },
    {
        "terms": [
            "dollar rises",
            "dollar rise",
            "stronger dollar",
            "dollar strength",
            "dollar index rises",
            "dxy rises",
            "yields rise",
            "yield rises",
            "treasury yields rise",
            "real yields rise",
            "hawkish fed",
            "hawkish",
            "rate hike",
            "rates stay high",
            "higher rates",
        ],
        "weight": 2,
        "side": "bear",
    },
    {
        "terms": [
            "gold lower",
            "gold falls",
            "gold fall",
            "drops",
            "drop",
            "slips",
            "bearish",
            "sell",
            "outflows",
            "profit-taking",
            "weaker",
        ],
        "weight": 1,
        "side": "bear",
    },
]


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
        "sentiment_label": "Sentiment",
        "market_structure": "Structure",
        "candle_pattern": "Pattern",
        "rsi_14": "RSI",
        "market_regime": "Regime",
        "alignment": "Alignment",
        "trade_sell_setup": "Sell Setup",
        "trade_buy_setup": "Buy Setup",
        "trade_exit_warning": "Exit Warning",
    }
    ordered_keys = [key for key in CHANGE_SUMMARY_ORDER if key in changes]
    ordered_keys.extend(key for key in changes if key not in ordered_keys)
    parts = []
    for key in ordered_keys[:3]:
        val = changes.get(key, {})
        prev = val.get("previous") if isinstance(val, dict) else None
        cur = val.get("current") if isinstance(val, dict) else None
        parts.append(f"{labels.get(key, key)}: {prev} -> {cur}")
    return " | ".join(parts) if parts else "Indicators changed"


def _classify_market_sentiment(news_list):
    return shared_classify_market_sentiment(news_list)


def _send_web_push_notifications(changes, verdict, confidence, trade_guidance):
    if webpush is None or not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
        return

    subscriptions = _load_subscriptions()
    if not subscriptions:
        return

    change_summary = _summarize_changes_for_push(changes)
    trade_summary = ""
    if isinstance(trade_guidance, dict):
        trade_summary = trade_guidance.get("summary") or ""
    body_parts = [change_summary, f"Verdict: {verdict} ({confidence}%)"]
    if trade_summary:
        body_parts.append(trade_summary)
    body = " | ".join(part for part in body_parts if part)
    payload = {
        "title": "XAUUSD Indicator Update",
        "body": body,
        "verdict": verdict,
        "confidence": confidence,
        "trade_guidance": trade_guidance,
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
        "confidence",
        "sentiment_label",
        "trend",
        "market_structure",
        "candle_pattern",
        "market_regime",
        "alignment",
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


def _compute_trade_guidance(ta_data, sentiment_label, confidence):
    return shared_compute_trade_guidance(ta_data, sentiment_label, confidence)


def _extract_indicator_snapshot(payload):
    """Build a compact snapshot used to detect indicator changes."""
    ta_data = payload.get("TechnicalAnalysis", {}) if isinstance(payload, dict) else {}
    pa = ta_data.get("price_action", {}) if isinstance(ta_data, dict) else {}
    regime = ta_data.get("volatility_regime", {}) if isinstance(ta_data, dict) else {}
    mtf = ta_data.get("multi_timeframe", {}) if isinstance(ta_data, dict) else {}
    trade_guidance = payload.get("TradeGuidance", {}) if isinstance(payload, dict) else {}
    sentiment_summary = payload.get("SentimentSummary", {}) if isinstance(payload, dict) else {}

    return {
        "verdict": payload.get("verdict"),
        "confidence": payload.get("confidence"),
        "sentiment_label": sentiment_summary.get("label"),
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
        "trade_sell_setup": trade_guidance.get("sellLevel"),
        "trade_buy_setup": trade_guidance.get("buyLevel"),
        "trade_exit_warning": trade_guidance.get("exitLevel"),
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
    sa_data = predict_gold.get_sentiment_analysis()

    if not isinstance(ta_data, dict):
        ta_data = {"error": "Technical analysis payload is invalid."}

    if not isinstance(sa_data, list):
        sa_data = []

    sentiment_summary = _classify_market_sentiment(sa_data)

    if ta_data.get("error"):
        return {
            "status": "error",
            "message": ta_data["error"],
            "verdict": "Neutral",
            "confidence": 50,
            "TechnicalAnalysis": ta_data,
            "SentimentalAnalysis": sa_data,
        }, 502

    prediction = compute_prediction_from_ta(ta_data, sentiment_summary)

    return {
        "status": "success",
        "verdict": prediction["verdict"],
        "confidence": prediction["confidence"],
        "TechnicalAnalysis": ta_data,
        "SentimentalAnalysis": sa_data,
        "SentimentSummary": sentiment_summary,
        "TradeGuidance": prediction["TradeGuidance"],
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
                            trade_guidance=payload.get("TradeGuidance"),
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

    if promoted_report_path.exists() or swarm_report_path.exists():
        try:
            promoted_report = _read_report(promoted_report_path) if promoted_report_path.exists() else None
            latest_report = _read_report(swarm_report_path) if swarm_report_path.exists() else None
            decision_report = _read_report(decision_report_path) if decision_report_path.exists() else None
        except RuntimeError as e:
            return jsonify({"status": "error", "message": str(e)}), 500

        display_report = promoted_report or latest_report or {}
        best = display_report.get("best_params") or {}
        latest_best = (latest_report or {}).get("best_params") or {}
        latest_summary = (latest_report or {}).get("summary") or {}
        display_summary = display_report.get("summary") or {}
        top_results = (latest_report or {}).get("top_results") or display_report.get("top_results") or []
        top_result = top_results[0] if isinstance(top_results, list) and top_results else {}
        roi = best.get("roi") or display_summary.get("roi")
        latest_generated_at = (latest_report or {}).get("generated_at")
        promote = bool(promoted_report or display_report.get("promote")) and bool(best)
        decision_reason = (decision_report or {}).get("promotion_reason") or (latest_report or {}).get("promotion_reason")

        return jsonify({
            "status": "success" if display_report else "idle",
            "generated_at": display_report.get("generated_at"),
            "latest_generated_at": latest_generated_at,
            "promote": promote,
            "promotion_reason": decision_reason or ("Loaded from promoted_result.json" if promote else "Swarm has not produced a fresh result yet."),
            "best_params": best,
            "latest_best_params": latest_best,
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
                "latest_run_promote": (latest_report or {}).get("promote"),
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


_ensure_monitor_started()

if __name__ == '__main__':
    # Run on 0.0.0.0 to allow access from other devices on the local network (like your phone)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
