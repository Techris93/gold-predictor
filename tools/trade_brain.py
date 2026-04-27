from __future__ import annotations

import copy
import json
import math
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "breakevenR": 1.0,
    "partialTpR": 1.5,
    "trailAdxThreshold": 35.0,
    "trailBufferMultiplier": 0.5,
    "breakevenBufferPercent": 0.2,
    "adxExhaustionDrop": 5.0,
    "learningEnabled": True,
    "learningMinSamples": 3,
    "learningAlpha": 0.35,
    "learningConfidenceScale": 4.0,
    "learningMaxConfidenceDelta": 10,
    "learningBaseRiskPercent": 1.0,
    "learningMinRiskMultiplier": 0.65,
    "learningMaxRiskMultiplier": 1.35,
    "learningRiskStep": 0.1,
    "maxNotifications": 30,
    "maxDecisionHistory": 600,
}

LEARNING_CONTEXT_WEIGHTS = {
    "setup_combo": 4.5,
    "combo": 3.5,
    "setup": 3.0,
    "structure": 2.0,
    "regime": 1.5,
    "session": 1.0,
    "direction": 0.75,
}

LEARNING_CONTEXT_LABELS = {
    "setup_combo": "Setup + Context",
    "combo": "Context",
    "setup": "Setup",
    "structure": "Structure",
    "regime": "Regime",
    "session": "Session",
    "direction": "Direction",
}

LONG_KEYWORDS = (
    "bull",
    "up",
    "support",
    "breakout",
    "reclaim",
    "higher low",
    "higher high",
)
SHORT_KEYWORDS = (
    "bear",
    "down",
    "resistance",
    "reject",
    "breakdown",
    "lower high",
    "lower low",
    "selloff",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_direction(direction: Any) -> str:
    raw = str(direction or "").strip().upper()
    if raw in {"BUY", "LONG"}:
        return "LONG"
    if raw in {"SELL", "SHORT"}:
        return "SHORT"
    raise ValueError("direction must be LONG/BUY or SHORT/SELL")


def _humanize_action(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ")
    return " ".join(piece.capitalize() for piece in text.split()) if text else "Update"


def _slug_text(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    normalized = "".join(char if char.isalnum() else "_" for char in text)
    condensed = "_".join(part for part in normalized.split("_") if part)
    return condensed or fallback


def _structure_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "neutral"
    if any(keyword in text for keyword in LONG_KEYWORDS):
        return "bullish"
    if any(keyword in text for keyword in SHORT_KEYWORDS):
        return "bearish"
    if any(keyword in text for keyword in ("range", "neutral", "consolid", "sideways", "drift")):
        return "range"
    return _slug_text(text, "neutral")


class TradeBrainService:
    def __init__(self, storage_path: str | Path, config: dict[str, Any] | None = None):
        self.storage_path = Path(storage_path)
        self.lock = threading.RLock()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.state = self._load_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": copy.deepcopy(self.config),
            "trades": [],
            "active_trade_id": None,
            "decision_history": [],
            "emotion_logs": [],
            "notifications": [],
            "last_market_data": None,
            "stats": self._empty_stats(),
        }

    def _empty_stats(self) -> dict[str, Any]:
        return {
            "totalTrades": 0,
            "closedTrades": 0,
            "activeTrades": 0,
            "wins": 0,
            "losses": 0,
            "totalR": 0.0,
            "avgR": 0.0,
            "winRate": 0.0,
            "bestSetup": None,
            "worstSetup": None,
        }

    def _load_state(self) -> dict[str, Any]:
        state = self._default_state()
        if not self.storage_path.exists():
            return state
        try:
            loaded = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return state
        if not isinstance(loaded, dict):
            return state
        state.update({key: value for key, value in loaded.items() if key in state})
        if not isinstance(state.get("trades"), list):
            state["trades"] = []
        if not isinstance(state.get("decision_history"), list):
            state["decision_history"] = []
        if not isinstance(state.get("emotion_logs"), list):
            state["emotion_logs"] = []
        if not isinstance(state.get("notifications"), list):
            state["notifications"] = []
        state["config"] = {**self.config, **(state.get("config") or {})}
        state["stats"] = self._compute_stats_locked(state["trades"])
        return state

    def _save_state_locked(self) -> None:
        self.state["config"] = copy.deepcopy(self.config)
        self.state["stats"] = self._compute_stats_locked(self.state["trades"])
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(self.state, ensure_ascii=True, indent=2), encoding="utf-8")

    def _trade_matches_user(self, trade: dict[str, Any], user_id: str | None) -> bool:
        if not user_id:
            return True
        return str(trade.get("userId") or "anonymous") == str(user_id)

    def _all_trades_locked(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return [trade for trade in self.state["trades"] if self._trade_matches_user(trade, user_id)]

    def _find_trade_locked(self, trade_id: str, user_id: str | None = None) -> tuple[int, dict[str, Any]]:
        for index, trade in enumerate(self.state["trades"]):
            if trade.get("id") == trade_id and self._trade_matches_user(trade, user_id):
                return index, trade
        raise KeyError(f"trade {trade_id} not found")

    def _active_trade_locked(self, user_id: str | None = None) -> dict[str, Any] | None:
        active_trade_id = self.state.get("active_trade_id")
        if not active_trade_id:
            return None
        for trade in self.state["trades"]:
            if trade.get("id") == active_trade_id and trade.get("status") == "ACTIVE":
                if self._trade_matches_user(trade, user_id):
                    return trade
        return None

    def _next_trade_id_locked(self) -> str:
        date_key = datetime.now(timezone.utc).strftime("%Y%m%d")
        sequence = len(self.state["trades"]) + 1
        return f"xau-{date_key}-{sequence:03d}"

    def _direction_sign(self, direction: str) -> int:
        return 1 if direction == "LONG" else -1

    def _one_r(self, trade: dict[str, Any]) -> float:
        return max(0.0001, abs(float(trade["entry"]["price"]) - float(trade["plan"]["initialStop"])))

    def _compute_unrealized_r(self, trade: dict[str, Any], current_price: float) -> float:
        sign = self._direction_sign(trade["direction"])
        one_r = self._one_r(trade)
        return round((sign * (current_price - float(trade["entry"]["price"]))) / one_r, 4)

    def _compute_unrealized_pnl(self, trade: dict[str, Any], current_price: float, unrealized_r: float) -> float:
        risk_dollar = _safe_float(trade["plan"].get("riskDollar"), None)
        if risk_dollar and abs(risk_dollar) > 0:
            return round(unrealized_r * risk_dollar, 2)
        position_size = _safe_float(trade["plan"].get("positionSize"), 0.0) or 0.0
        sign = self._direction_sign(trade["direction"])
        return round(sign * (current_price - float(trade["entry"]["price"])) * position_size, 2)

    def _reaches_price(self, direction: str, current_price: float, target_price: float | None) -> bool:
        if target_price is None:
            return False
        if direction == "LONG":
            return current_price >= target_price
        return current_price <= target_price

    def _stop_is_hit(self, direction: str, current_price: float, stop_price: float) -> bool:
        if direction == "LONG":
            return current_price <= stop_price
        return current_price >= stop_price

    def _structure_supports_direction(self, direction: str, structure: Any) -> bool:
        text = str(structure or "").lower()
        keywords = LONG_KEYWORDS if direction == "LONG" else SHORT_KEYWORDS
        return any(keyword in text for keyword in keywords)

    def _structure_breaks_direction(self, direction: str, structure: Any) -> bool:
        text = str(structure or "").lower()
        opposing = SHORT_KEYWORDS if direction == "LONG" else LONG_KEYWORDS
        return any(keyword in text for keyword in opposing)

    def _event(self, event_type: str, trade: dict[str, Any], message: str, price: float | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "type": event_type,
            "message": message,
            "tradeId": trade.get("id"),
            "price": price,
            "timestamp": _utc_now(),
        }
        if extra:
            payload.update(extra)
        return payload

    def _append_notification_locked(self, event: dict[str, Any]) -> None:
        self.state["notifications"].append(event)
        self.state["notifications"] = self.state["notifications"][-int(self.config["maxNotifications"]):]

    def _append_decision_locked(
        self,
        trade: dict[str, Any],
        action: str,
        price: float,
        reasoning: str,
        emotion: str = "calm",
        market_context: str | None = None,
    ) -> dict[str, Any]:
        decision = {
            "time": _utc_now(),
            "price": round(price, 4),
            "action": str(action).upper(),
            "reasoning": str(reasoning),
            "emotion": emotion,
            "marketContext": market_context or trade["entry"]["context"].get("regime") or "normal",
        }
        trade["live"]["decisions"].append(decision)
        self.state["decision_history"].append({"tradeId": trade["id"], **decision})
        self.state["decision_history"] = self.state["decision_history"][-int(self.config["maxDecisionHistory"]):]
        return decision

    def _move_stop_locked(
        self,
        trade: dict[str, Any],
        new_stop: float,
        reason: str,
        stop_type: str,
    ) -> bool:
        live_stop = trade["live"]["stopLoss"]
        current_stop = float(live_stop["current"])
        improves = new_stop > current_stop if trade["direction"] == "LONG" else new_stop < current_stop
        if not improves:
            return False
        rounded_stop = round(new_stop, 4)
        live_stop["current"] = rounded_stop
        live_stop["history"].append(
            {
                "price": rounded_stop,
                "time": _utc_now(),
                "reason": reason,
                "type": stop_type,
            }
        )
        return True

    def _derive_review_locked(self, trade: dict[str, Any]) -> dict[str, Any]:
        final_r = _safe_float(trade.get("exit", {}).get("finalR"), 0.0) or 0.0
        setup = trade["entry"].get("trigger") or "Manual entry"
        similar_closed = [
            item
            for item in self.state["trades"]
            if item.get("id") != trade.get("id")
            and item.get("status") == "CLOSED"
            and (item.get("entry", {}).get("trigger") or "Manual entry") == setup
        ]
        similar_rs = [_safe_float(item.get("exit", {}).get("finalR"), 0.0) or 0.0 for item in similar_closed]
        avg_similar_r = round(sum(similar_rs) / len(similar_rs), 2) if similar_rs else 0.0
        emotion = str(trade.get("exit", {}).get("emotion") or "calm").lower()

        if final_r >= 2.5 and emotion not in {"fear", "panic", "revenge"}:
            grade = "A"
        elif final_r >= 1.0:
            grade = "B"
        elif final_r >= 0:
            grade = "C"
        else:
            grade = "D"

        lessons = []
        if trade["live"]["takeProfit"]["tp1"].get("hit"):
            lessons.append("Partial profit logic engaged before final exit.")
        if trade["live"]["trailing"].get("active"):
            lessons.append("Trailing stop logic activated during the trade.")
        if final_r < 0:
            lessons.append("Losses should be reviewed against structure and invalidation quality.")
        elif final_r > 0:
            lessons.append("Risk stayed aligned with the plan long enough to extract positive R.")
        if emotion in {"fear", "panic", "revenge", "greed"}:
            lessons.append("Emotion tag suggests discipline review is needed around exits.")
        if not lessons:
            lessons.append("Trade followed the baseline process without major anomalies.")

        return {
            "trade": copy.deepcopy(trade),
            "grade": grade,
            "lessons": lessons,
            "comparison": {
                "yourR": round(final_r, 2),
                "avgSimilarR": avg_similar_r,
            },
        }

    def _close_trade_locked(
        self,
        trade: dict[str, Any],
        exit_price: float,
        reason: str,
        reasoning: str,
        emotion: str = "calm",
        final_pnl: float | None = None,
        final_r: float | None = None,
    ) -> dict[str, Any]:
        if trade.get("status") == "CLOSED":
            return self._event("trade:closed", trade, "Trade already closed.", exit_price)

        resolved_final_r = round(final_r if final_r is not None else self._compute_unrealized_r(trade, exit_price), 4)
        resolved_final_pnl = round(
            final_pnl if final_pnl is not None else self._compute_unrealized_pnl(trade, exit_price, resolved_final_r),
            2,
        )

        trade["status"] = "CLOSED"
        trade["live"]["status"] = "CLOSED"
        trade["live"]["currentPrice"] = round(exit_price, 4)
        trade["live"]["unrealizedR"] = resolved_final_r
        trade["live"]["unrealizedPnL"] = resolved_final_pnl
        trade["exit"] = {
            "price": round(exit_price, 4),
            "timestamp": _utc_now(),
            "reason": reason,
            "reasoning": reasoning,
            "finalPnL": resolved_final_pnl,
            "finalR": resolved_final_r,
            "emotion": emotion,
        }
        trade["live"]["exit"] = copy.deepcopy(trade["exit"])
        if self.state.get("active_trade_id") == trade.get("id"):
            self.state["active_trade_id"] = None
        self._append_decision_locked(trade, "EXIT", exit_price, reasoning, emotion=emotion)
        review = self._derive_review_locked(trade)
        trade["review"] = review
        event = self._event(
            "trade:closed",
            trade,
            f"{_humanize_action(reason)} at {round(exit_price, 2)} ({resolved_final_r:.2f}R)",
            exit_price,
            extra={"reason": reason, "finalR": resolved_final_r, "finalPnL": resolved_final_pnl},
        )
        self._append_notification_locked(event)
        return event

    def _compute_stats_locked(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        stats = self._empty_stats()
        stats["totalTrades"] = len(trades)
        stats["activeTrades"] = sum(1 for trade in trades if trade.get("status") == "ACTIVE")
        closed = [trade for trade in trades if trade.get("status") == "CLOSED"]
        stats["closedTrades"] = len(closed)
        wins = 0
        losses = 0
        total_r = 0.0
        setup_scores: dict[str, list[float]] = defaultdict(list)
        for trade in closed:
            final_r = _safe_float(trade.get("exit", {}).get("finalR"), 0.0) or 0.0
            total_r += final_r
            setup = trade.get("entry", {}).get("trigger") or "Manual entry"
            setup_scores[setup].append(final_r)
            if final_r > 0:
                wins += 1
            else:
                losses += 1
        stats["wins"] = wins
        stats["losses"] = losses
        stats["totalR"] = round(total_r, 2)
        stats["avgR"] = round(total_r / len(closed), 2) if closed else 0.0
        stats["winRate"] = round((wins / len(closed)) * 100, 2) if closed else 0.0
        if setup_scores:
            averages = {key: sum(values) / len(values) for key, values in setup_scores.items()}
            stats["bestSetup"] = max(averages, key=averages.get)
            stats["worstSetup"] = min(averages, key=averages.get)
        return stats

    def _closed_trades_locked(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return [trade for trade in self.state["trades"] if trade.get("status") == "CLOSED" and self._trade_matches_user(trade, user_id)]

    def _learning_context_keys(
        self,
        direction: str,
        session: Any,
        regime: Any,
        structure: Any,
        setup: Any | None = None,
    ) -> list[tuple[str, str]]:
        try:
            normalized_direction = _normalize_direction(direction)
        except ValueError:
            return []

        session_key = _slug_text(session or self.get_current_session(), "unknown")
        regime_key = _slug_text(regime, "normal")
        structure_key = _structure_bucket(structure)
        keys = [
            ("direction", f"direction:{normalized_direction}"),
            ("session", f"session:{normalized_direction}:{session_key}"),
            ("regime", f"regime:{normalized_direction}:{regime_key}"),
            ("structure", f"structure:{normalized_direction}:{structure_key}"),
            ("combo", f"combo:{normalized_direction}:{session_key}:{regime_key}:{structure_key}"),
        ]

        if setup not in (None, ""):
            setup_key = _slug_text(setup, "manual_entry")
            keys.append(("setup", f"setup:{normalized_direction}:{setup_key}"))
            keys.append(("setup_combo", f"setup_combo:{normalized_direction}:{setup_key}:{session_key}:{regime_key}:{structure_key}"))
        return keys

    def _build_learning_profiles_locked(self, user_id: str | None = None) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        alpha = float(self.config.get("learningAlpha", 0.35) or 0.35)
        for trade in self._closed_trades_locked(user_id):
            entry = trade.get("entry") if isinstance(trade.get("entry"), dict) else {}
            context = entry.get("context") if isinstance(entry.get("context"), dict) else {}
            reward = _safe_float(trade.get("exit", {}).get("finalR"), 0.0) or 0.0
            reward = max(-3.0, min(3.0, reward))
            direction = str(trade.get("direction") or "")
            session = entry.get("session") or context.get("session") or "Unknown"
            regime = context.get("regime") or "normal"
            structure = context.get("structure") or "neutral"
            setup = entry.get("trigger") or "Manual entry"

            for label, key in self._learning_context_keys(direction, session, regime, structure, setup):
                profile = profiles.get(key)
                if profile is None:
                    profile = {
                        "key": key,
                        "label": label,
                        "direction": direction,
                        "session": str(session),
                        "regime": str(regime),
                        "structure": _structure_bucket(structure),
                        "setup": str(setup),
                        "samples": 0,
                        "wins": 0,
                        "losses": 0,
                        "totalReward": 0.0,
                        "avgReward": 0.0,
                        "qValue": 0.0,
                        "lastReward": 0.0,
                    }
                    profiles[key] = profile

                profile["samples"] += 1
                profile["wins"] += 1 if reward > 0 else 0
                profile["losses"] += 1 if reward <= 0 else 0
                profile["totalReward"] = round(float(profile["totalReward"]) + reward, 4)
                profile["avgReward"] = round(float(profile["totalReward"]) / int(profile["samples"]), 4)
                profile["qValue"] = round(
                    reward if int(profile["samples"]) == 1 else ((1 - alpha) * float(profile["qValue"])) + (alpha * reward),
                    4,
                )
                profile["lastReward"] = round(reward, 4)

        return profiles

    def _profile_payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        samples = int(profile.get("samples", 0) or 0)
        wins = int(profile.get("wins", 0) or 0)
        descriptor_parts = [str(profile.get("direction") or "")]
        if profile.get("setup"):
            descriptor_parts.append(str(profile.get("setup") or ""))
        descriptor_parts.append(str(profile.get("session") or "Unknown"))
        descriptor_parts.append(str(profile.get("regime") or "normal"))
        descriptor_parts.append(str(profile.get("structure") or "neutral"))
        descriptor = " | ".join(part for part in descriptor_parts if part)
        return {
            "key": str(profile.get("key") or ""),
            "label": LEARNING_CONTEXT_LABELS.get(str(profile.get("label") or ""), _humanize_action(profile.get("label"))),
            "descriptor": descriptor,
            "samples": samples,
            "avgReward": round(float(profile.get("avgReward", 0.0) or 0.0), 2),
            "qValue": round(float(profile.get("qValue", 0.0) or 0.0), 2),
            "winRate": round((wins / samples) * 100, 2) if samples else 0.0,
        }

    def _neutral_sizing_guidance(self) -> dict[str, Any]:
        base_risk = float(self.config.get("learningBaseRiskPercent", 1.0) or 1.0)
        return {
            "active": False,
            "mode": "standard",
            "riskMultiplier": 1.0,
            "positionScale": 1.0,
            "suggestedRiskPercent": round(base_risk, 2),
            "summary": "Use base size until reinforcement memory has enough matching trades.",
        }

    def _build_sizing_guidance_locked(self, reward_score: float, active: bool) -> dict[str, Any]:
        if not active:
            return self._neutral_sizing_guidance()

        base_risk = float(self.config.get("learningBaseRiskPercent", 1.0) or 1.0)
        min_multiplier = float(self.config.get("learningMinRiskMultiplier", 0.65) or 0.65)
        max_multiplier = float(self.config.get("learningMaxRiskMultiplier", 1.35) or 1.35)
        step = float(self.config.get("learningRiskStep", 0.1) or 0.1)
        risk_multiplier = max(
            min_multiplier,
            min(max_multiplier, 1.0 + (float(reward_score) * step)),
        )
        risk_multiplier = round(risk_multiplier, 2)
        suggested_risk = round(base_risk * risk_multiplier, 2)

        if risk_multiplier >= 1.08:
            mode = "press"
            summary = (
                f"Lean into the edge with about {risk_multiplier:.2f}x base size "
                f"({suggested_risk:.2f}% risk if 1R equals 1% for you)."
            )
        elif risk_multiplier <= 0.92:
            mode = "trim"
            summary = (
                f"Trim exposure to about {risk_multiplier:.2f}x base size "
                f"({suggested_risk:.2f}% risk if 1R equals 1% for you)."
            )
        else:
            mode = "standard"
            summary = (
                f"Keep size near baseline at {risk_multiplier:.2f}x "
                f"({suggested_risk:.2f}% risk if 1R equals 1% for you)."
            )

        return {
            "active": True,
            "mode": mode,
            "riskMultiplier": risk_multiplier,
            "positionScale": risk_multiplier,
            "suggestedRiskPercent": suggested_risk,
            "summary": summary,
        }

    def _build_setup_rankings_locked(
        self,
        profiles: dict[str, dict[str, Any]],
        direction: str | None,
        market_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            normalized_direction = _normalize_direction(direction)
        except ValueError:
            return []

        market_snapshot = market_data if isinstance(market_data, dict) else {}
        target_session = _slug_text(market_snapshot.get("session") or self.get_current_session(), "unknown")
        target_regime = _slug_text(market_snapshot.get("regime"), "normal")
        target_structure = _structure_bucket(market_snapshot.get("structure"))
        min_samples = int(self.config.get("learningMinSamples", 3) or 3)

        rankings: dict[str, dict[str, Any]] = {}
        for profile in profiles.values():
            if str(profile.get("direction") or "") != normalized_direction:
                continue
            if str(profile.get("label") or "") not in {"setup", "setup_combo"}:
                continue

            setup_name = str(profile.get("setup") or "Manual entry")
            samples = int(profile.get("samples", 0) or 0)
            if samples <= 0:
                continue

            context_matches = 0
            if _slug_text(profile.get("session"), "unknown") == target_session:
                context_matches += 1
            if _slug_text(profile.get("regime"), "normal") == target_regime:
                context_matches += 1
            if _structure_bucket(profile.get("structure")) == target_structure:
                context_matches += 1

            context_weight = 1.0 + (0.12 * context_matches)
            if str(profile.get("label") or "") == "setup_combo":
                context_weight += 0.16
            sample_weight = min(1.0, samples / max(1, min_samples))
            recommendation_score = round(
                float(profile.get("qValue", 0.0) or 0.0) * context_weight * sample_weight,
                2,
            )

            payload = self._profile_payload(profile)
            payload.update(
                {
                    "setup": setup_name,
                    "contextMatches": context_matches,
                    "recommendationScore": recommendation_score,
                    "active": samples >= min_samples,
                }
            )
            existing = rankings.get(setup_name)
            if existing is None or float(payload["recommendationScore"]) > float(existing["recommendationScore"]):
                rankings[setup_name] = payload

        return sorted(
            rankings.values(),
            key=lambda item: (
                float(item.get("recommendationScore", 0.0) or 0.0),
                float(item.get("qValue", 0.0) or 0.0),
                int(item.get("samples", 0) or 0),
            ),
            reverse=True,
        )[:4]

    def _get_learning_adjustment_locked(
        self,
        direction: str | None,
        market_data: dict[str, Any] | None = None,
        setup: Any | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        min_samples = int(self.config.get("learningMinSamples", 3) or 3)
        max_delta = int(self.config.get("learningMaxConfidenceDelta", 10) or 10)
        closed_trades = self._closed_trades_locked(user_id)
        result = {
            "enabled": bool(self.config.get("learningEnabled", True)),
            "active": False,
            "direction": direction,
            "closedSamples": len(closed_trades),
            "minSamples": min_samples,
            "confidenceDelta": 0,
            "rewardScore": 0.0,
            "message": "Learning engine warming up.",
            "matchedContexts": [],
            "sizing": self._neutral_sizing_guidance(),
            "rankedSetups": [],
            "topSetup": None,
        }
        if not result["enabled"]:
            result["message"] = "Learning engine is disabled."
            return result
        if not closed_trades:
            result["message"] = "Close a few trades to let reinforcement memory calibrate future decisions."
            return result

        profiles = self._build_learning_profiles_locked(user_id)
        result["rankedSetups"] = self._build_setup_rankings_locked(profiles, direction, market_data)
        result["topSetup"] = copy.deepcopy(result["rankedSetups"][0]) if result["rankedSetups"] else None

        try:
            normalized_direction = _normalize_direction(direction)
        except ValueError:
            result["message"] = "Learning memory activates only when the model has a directional bias."
            return result

        market_snapshot = market_data if isinstance(market_data, dict) else {}
        session = market_snapshot.get("session") or self.get_current_session()
        regime = market_snapshot.get("regime") or "normal"
        structure = market_snapshot.get("structure") or "neutral"

        matches = []
        for label, key in self._learning_context_keys(normalized_direction, session, regime, structure, setup):
            profile = profiles.get(key)
            if not profile:
                continue
            payload = self._profile_payload(profile)
            payload["contextLabel"] = label
            payload["weight"] = float(LEARNING_CONTEXT_WEIGHTS.get(label, 1.0))
            matches.append(payload)

        matches.sort(key=lambda item: (item["weight"], item["samples"]), reverse=True)
        result["direction"] = normalized_direction
        result["matchedContexts"] = matches[:3]

        eligible = [item for item in matches if int(item.get("samples", 0) or 0) >= min_samples]
        if not eligible:
            strongest = matches[0] if matches else None
            if strongest:
                strongest_samples = int(strongest.get("samples", 0) or 0)
                result["message"] = (
                    f"Learning engine is warming up for {normalized_direction.lower()}. "
                    f"Strongest context has {strongest_samples}/{min_samples} matching trades."
                )
            else:
                result["message"] = f"No learned trade history yet for {normalized_direction.lower()} contexts."
            return result

        top_matches = eligible[:3]
        total_weight = sum(float(item.get("weight", 0.0) or 0.0) for item in top_matches)
        reward_score = (
            sum(float(item.get("qValue", 0.0) or 0.0) * float(item.get("weight", 0.0) or 0.0) for item in top_matches)
            / total_weight
            if total_weight
            else 0.0
        )
        delta = int(round(max(-max_delta, min(max_delta, reward_score * float(self.config.get("learningConfidenceScale", 4.0) or 4.0)))))
        strongest = top_matches[0]
        if delta > 0:
            message = (
                f"Reinforcement memory favors {normalized_direction.lower()} here: "
                f"{int(strongest['samples'])} similar trades averaged {float(strongest['avgReward']):+.2f}R."
            )
        elif delta < 0:
            message = (
                f"Reinforcement memory is cautious on {normalized_direction.lower()} here: "
                f"{int(strongest['samples'])} similar trades averaged {float(strongest['avgReward']):+.2f}R."
            )
        else:
            message = (
                f"Reinforcement memory is neutral after {int(strongest['samples'])} similar trades."
            )

        result.update(
            {
                "active": True,
                "confidenceDelta": delta,
                "rewardScore": round(reward_score, 2),
                "message": message,
                "sizing": self._build_sizing_guidance_locked(reward_score, active=True),
            }
        )
        return result

    def get_learning_adjustment(
        self,
        direction: str | None,
        market_data: dict[str, Any] | None = None,
        setup: Any | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(
                self._get_learning_adjustment_locked(
                    direction=direction,
                    market_data=market_data,
                    setup=setup,
                    user_id=user_id,
                )
            )

    def _get_learning_summary_locked(
        self,
        user_id: str | None = None,
        market_data: dict[str, Any] | None = None,
        direction: str | None = None,
        setup: Any | None = None,
    ) -> dict[str, Any]:
        closed_trades = self._closed_trades_locked(user_id)
        profiles = self._build_learning_profiles_locked(user_id)
        min_samples = int(self.config.get("learningMinSamples", 3) or 3)
        current_adjustment = self._get_learning_adjustment_locked(direction, market_data, setup, user_id)
        eligible_payloads = [
            self._profile_payload(profile)
            for profile in profiles.values()
            if int(profile.get("samples", 0) or 0) >= min_samples
        ]
        top_edges = sorted(
            [payload for payload in eligible_payloads if float(payload.get("qValue", 0.0) or 0.0) > 0],
            key=lambda item: (float(item.get("qValue", 0.0) or 0.0), int(item.get("samples", 0) or 0)),
            reverse=True,
        )[:3]
        top_risks = sorted(
            [payload for payload in eligible_payloads if float(payload.get("qValue", 0.0) or 0.0) < 0],
            key=lambda item: (float(item.get("qValue", 0.0) or 0.0), -int(item.get("samples", 0) or 0)),
        )[:3]
        return {
            "enabled": bool(self.config.get("learningEnabled", True)),
            "mode": "contextual_bandit",
            "closedSamples": len(closed_trades),
            "learnedContexts": len(eligible_payloads),
            "minSamples": min_samples,
            "currentAdjustment": current_adjustment,
            "sizing": copy.deepcopy(current_adjustment.get("sizing") or self._neutral_sizing_guidance()),
            "rankedSetups": copy.deepcopy(current_adjustment.get("rankedSetups") or []),
            "topSetup": copy.deepcopy(current_adjustment.get("topSetup")),
            "topEdges": top_edges,
            "topRisks": top_risks,
        }

    def _build_analytics_locked(self, user_id: str | None = None) -> dict[str, Any]:
        trades = self._all_trades_locked(user_id)
        closed = [trade for trade in trades if trade.get("status") == "CLOSED"]
        setups: dict[str, list[float]] = defaultdict(list)
        sessions: dict[str, list[float]] = defaultdict(list)
        monthly: dict[str, list[float]] = defaultdict(list)
        emotion_counts: dict[str, int] = defaultdict(int)
        r_distribution: dict[str, int] = defaultdict(int)

        for trade in closed:
            final_r = _safe_float(trade.get("exit", {}).get("finalR"), 0.0) or 0.0
            setup = trade.get("entry", {}).get("trigger") or "Manual entry"
            session = trade.get("entry", {}).get("session") or "Unknown"
            exit_ts = str(trade.get("exit", {}).get("timestamp") or trade.get("entry", {}).get("timestamp") or "")
            month_key = exit_ts[:7] if len(exit_ts) >= 7 else "Unknown"
            emotion = str(trade.get("exit", {}).get("emotion") or "calm").lower()
            bucket_value = math.floor(final_r)
            bucket_label = f"{bucket_value}R"

            setups[setup].append(final_r)
            sessions[session].append(final_r)
            monthly[month_key].append(final_r)
            emotion_counts[emotion] += 1
            r_distribution[bucket_label] += 1

        return {
            "setups": [
                {
                    "setup": key,
                    "trades": len(values),
                    "avgR": round(sum(values) / len(values), 2),
                    "winRate": round((sum(1 for item in values if item > 0) / len(values)) * 100, 2),
                }
                for key, values in sorted(setups.items())
            ],
            "emotions": [
                {"emotion": key, "count": value}
                for key, value in sorted(emotion_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "sessions": [
                {
                    "session": key,
                    "trades": len(values),
                    "avgR": round(sum(values) / len(values), 2),
                }
                for key, values in sorted(sessions.items())
            ],
            "distribution": [
                {"bucket": key, "count": value}
                for key, value in sorted(r_distribution.items(), key=lambda item: float(item[0].replace("R", "")))
            ],
            "monthly": [
                {
                    "month": key,
                    "trades": len(values),
                    "avgR": round(sum(values) / len(values), 2),
                    "totalR": round(sum(values), 2),
                }
                for key, values in sorted(monthly.items())
            ],
        }

    def enter_trade(self, payload: dict[str, Any], user_id: str = "anonymous") -> dict[str, Any]:
        with self.lock:
            if self._active_trade_locked(user_id) is not None:
                raise ValueError("an active trade already exists")

            direction = _normalize_direction(payload.get("direction"))
            price = _safe_float(payload.get("price") or payload.get("entryPrice"), None)
            stop_loss = _safe_float(payload.get("stopLoss") or payload.get("initialStop"), None)
            if price is None or stop_loss is None:
                raise ValueError("price and stopLoss are required")

            context = copy.deepcopy(payload.get("context") or {})
            one_r = max(0.0001, abs(price - stop_loss))
            risk_percent = _safe_float(payload.get("riskPercent"), 1.0) or 1.0
            risk_dollar = _safe_float(payload.get("riskDollar"), 0.0) or 0.0
            position_size = _safe_float(payload.get("positionSize"), 0.0) or 0.0
            tp1 = _safe_float(payload.get("takeProfit1"), None)
            tp2 = _safe_float(payload.get("takeProfit2"), None)
            if tp1 is None:
                tp1 = price + (one_r * float(self.config["partialTpR"]) * self._direction_sign(direction))
            if tp2 is None:
                tp2 = price + (one_r * 3.0 * self._direction_sign(direction))

            trade_id = str(payload.get("id") or self._next_trade_id_locked())
            session = context.get("session") or payload.get("session") or self.get_current_session()
            trade = {
                "id": trade_id,
                "userId": user_id or "anonymous",
                "symbol": str(payload.get("symbol") or "XAUUSD"),
                "direction": direction,
                "status": "ACTIVE",
                "createdAt": _utc_now(),
                "updatedAt": _utc_now(),
                "entry": {
                    "price": round(price, 4),
                    "timestamp": payload.get("timestamp") or _utc_now(),
                    "trigger": payload.get("trigger") or context.get("trigger") or "Manual entry",
                    "confidence": _safe_float(payload.get("confidence") or context.get("confidence"), 0.7) or 0.7,
                    "session": session,
                    "context": {
                        "adx": _safe_float(context.get("adx"), 0.0) or 0.0,
                        "atrPercent": _safe_float(context.get("atrPercent"), 0.0) or 0.0,
                        "atrDollar": _safe_float(context.get("atrDollar"), 0.0) or 0.0,
                        "vwap": _safe_float(context.get("vwap"), price) or price,
                        "structure": context.get("structure") or "neutral",
                        "regime": context.get("regime") or "normal",
                        "invalidation": _safe_float(context.get("invalidation"), stop_loss) or stop_loss,
                        **context,
                    },
                },
                "plan": {
                    "entryPrice": round(price, 4),
                    "initialStop": round(stop_loss, 4),
                    "invalidationPrice": round(_safe_float(context.get("invalidation"), stop_loss) or stop_loss, 4),
                    "tp1": round(tp1, 4),
                    "tp2": round(tp2, 4),
                    "riskPercent": risk_percent,
                    "riskDollar": round(risk_dollar, 2),
                    "positionSize": position_size,
                    "rrPlanned": round(abs(tp2 - price) / one_r, 2),
                },
                "live": {
                    "status": "ACTIVE",
                    "currentPrice": round(price, 4),
                    "unrealizedPnL": 0.0,
                    "unrealizedR": 0.0,
                    "highestPrice": round(price, 4),
                    "lowestPrice": round(price, 4),
                    "stopLoss": {
                        "current": round(stop_loss, 4),
                        "history": [
                            {
                                "price": round(stop_loss, 4),
                                "time": _utc_now(),
                                "reason": "Initial protective stop",
                                "type": "INITIAL",
                            }
                        ],
                    },
                    "takeProfit": {
                        "tp1": {"target": round(tp1, 4), "hit": False, "partialClosed": False, "percent": 50},
                        "tp2": {"target": round(tp2, 4), "hit": False, "partialClosed": False, "percent": 50},
                    },
                    "trailing": {
                        "active": False,
                        "method": None,
                        "adxAtTrigger": None,
                        "highestPriceSinceTrail": round(price, 4),
                        "lowestPriceSinceTrail": round(price, 4),
                        "buffer": 0.0,
                    },
                    "decisions": [],
                    "exit": None,
                    "lastEvaluationSignature": None,
                },
                "exit": None,
                "review": None,
                "snapshots": [],
            }

            self.state["trades"].append(trade)
            self.state["active_trade_id"] = trade_id
            self.state["last_market_data"] = {
                "price": round(price, 4),
                "adx": trade["entry"]["context"]["adx"],
                "vwap": trade["entry"]["context"]["vwap"],
                "atrDollar": trade["entry"]["context"]["atrDollar"],
                "atrPercent": trade["entry"]["context"]["atrPercent"],
                "structure": trade["entry"]["context"]["structure"],
                "regime": trade["entry"]["context"]["regime"],
            }
            self._append_decision_locked(
                trade,
                "ENTER",
                price,
                f"Enter {direction} at {round(price, 2)} with stop {round(stop_loss, 2)}.",
            )
            event = self._event("trade:created", trade, f"{direction} trade opened at {round(price, 2)}", price)
            self._append_notification_locked(event)
            self._save_state_locked()
            return copy.deepcopy(trade)

    def evaluate_active_trade(
        self,
        current_price: float,
        market_data: dict[str, Any] | None,
        decision: str | None = None,
        user_id: str = "anonymous",
    ) -> dict[str, Any] | None:
        with self.lock:
            trade = self._active_trade_locked(user_id)
            if trade is None:
                self.state["last_market_data"] = copy.deepcopy(market_data or {})
                self._save_state_locked()
                return None

            normalized_market = {
                "price": round(float(current_price), 4),
                "adx": _safe_float((market_data or {}).get("adx"), trade["entry"]["context"].get("adx", 0.0)) or 0.0,
                "vwap": _safe_float((market_data or {}).get("vwap"), current_price) or current_price,
                "atrDollar": _safe_float((market_data or {}).get("atrDollar"), trade["entry"]["context"].get("atrDollar", 0.0)) or 0.0,
                "atrPercent": _safe_float((market_data or {}).get("atrPercent"), trade["entry"]["context"].get("atrPercent", 0.0)) or 0.0,
                "structure": (market_data or {}).get("structure") or trade["entry"]["context"].get("structure") or "neutral",
                "regime": (market_data or {}).get("regime") or trade["entry"]["context"].get("regime") or "normal",
            }
            signature = json.dumps(
                {
                    "price": normalized_market["price"],
                    "adx": normalized_market["adx"],
                    "vwap": normalized_market["vwap"],
                    "atrDollar": normalized_market["atrDollar"],
                    "structure": normalized_market["structure"],
                    "regime": normalized_market["regime"],
                    "decision": decision,
                },
                sort_keys=True,
            )
            if trade["live"].get("lastEvaluationSignature") == signature:
                return {
                    "trade": copy.deepcopy(trade),
                    "events": [],
                    "dashboard": self.get_dashboard_payload(user_id=user_id, market_data=normalized_market),
                }

            trade["updatedAt"] = _utc_now()
            trade["live"]["lastEvaluationSignature"] = signature
            trade["live"]["currentPrice"] = normalized_market["price"]
            trade["live"]["highestPrice"] = round(max(float(trade["live"]["highestPrice"]), normalized_market["price"]), 4)
            trade["live"]["lowestPrice"] = round(min(float(trade["live"]["lowestPrice"]), normalized_market["price"]), 4)
            if trade["live"]["trailing"]["active"]:
                trade["live"]["trailing"]["highestPriceSinceTrail"] = round(
                    max(float(trade["live"]["trailing"]["highestPriceSinceTrail"]), normalized_market["price"]),
                    4,
                )
                trade["live"]["trailing"]["lowestPriceSinceTrail"] = round(
                    min(float(trade["live"]["trailing"]["lowestPriceSinceTrail"]), normalized_market["price"]),
                    4,
                )

            trade["entry"]["context"].update({key: value for key, value in normalized_market.items() if key != "price"})
            self.state["last_market_data"] = copy.deepcopy(normalized_market)

            unrealized_r = self._compute_unrealized_r(trade, normalized_market["price"])
            trade["live"]["unrealizedR"] = unrealized_r
            trade["live"]["unrealizedPnL"] = self._compute_unrealized_pnl(trade, normalized_market["price"], unrealized_r)

            events = []
            if decision:
                decision_action = str(decision).strip().upper()
                self._append_decision_locked(
                    trade,
                    decision_action,
                    normalized_market["price"],
                    f"Manual decision tag: {decision_action.lower()}.",
                    market_context=normalized_market["regime"],
                )
                event = self._event(
                    "trade:decision",
                    trade,
                    f"Decision tagged: {_humanize_action(decision_action)}",
                    normalized_market["price"],
                    extra={"decision": decision_action, "marketData": copy.deepcopy(normalized_market)},
                )
                self._append_notification_locked(event)
                events.append(event)

            direction = trade["direction"]
            current_stop = float(trade["live"]["stopLoss"]["current"])
            if self._stop_is_hit(direction, normalized_market["price"], current_stop):
                events.append(
                    self._close_trade_locked(
                        trade,
                        normalized_market["price"],
                        "STOP_OUT",
                        f"Price hit the active stop at {round(current_stop, 2)}.",
                    )
                )
                self._save_state_locked()
                return {
                    "trade": copy.deepcopy(trade),
                    "events": events,
                    "dashboard": self.get_dashboard_payload(user_id=user_id, market_data=normalized_market),
                }

            one_r = self._one_r(trade)
            be_buffer = one_r * float(self.config["breakevenBufferPercent"])
            if unrealized_r >= float(self.config["breakevenR"]):
                breakeven_stop = trade["entry"]["price"] + (be_buffer * self._direction_sign(direction))
                moved = self._move_stop_locked(
                    trade,
                    breakeven_stop,
                    "Moved stop to lock a small profit after 1R progress.",
                    "BREAKEVEN",
                )
                if moved:
                    self._append_decision_locked(
                        trade,
                        "MOVE_STOP",
                        normalized_market["price"],
                        "Breakeven protection activated after reaching 1R.",
                        market_context=normalized_market["regime"],
                    )
                    event = self._event(
                        "trade:updated",
                        trade,
                        f"Stop moved to breakeven at {round(float(trade['live']['stopLoss']['current']), 2)}",
                        normalized_market["price"],
                    )
                    self._append_notification_locked(event)
                    events.append(event)

            tp1_target = _safe_float(trade["live"]["takeProfit"]["tp1"].get("target"), None)
            if not trade["live"]["takeProfit"]["tp1"].get("hit") and self._reaches_price(direction, normalized_market["price"], tp1_target):
                trade["live"]["takeProfit"]["tp1"]["hit"] = True
                trade["live"]["takeProfit"]["tp1"]["partialClosed"] = True
                self._append_decision_locked(
                    trade,
                    "TAKE_PROFIT_1",
                    normalized_market["price"],
                    "TP1 reached. Partial profit was protected.",
                    market_context=normalized_market["regime"],
                )
                event = self._event(
                    "trade:alert",
                    trade,
                    f"TP1 reached at {round(normalized_market['price'], 2)}",
                    normalized_market["price"],
                    extra={"alertType": "tp1"},
                )
                self._append_notification_locked(event)
                events.append(event)

            tp2_target = _safe_float(trade["live"]["takeProfit"]["tp2"].get("target"), None)
            if self._reaches_price(direction, normalized_market["price"], tp2_target):
                trade["live"]["takeProfit"]["tp2"]["hit"] = True
                events.append(
                    self._close_trade_locked(
                        trade,
                        normalized_market["price"],
                        "TAKE_PROFIT_2",
                        "Final target reached.",
                    )
                )
                self._save_state_locked()
                return {
                    "trade": copy.deepcopy(trade),
                    "events": events,
                    "dashboard": self.get_dashboard_payload(user_id=user_id, market_data=normalized_market),
                }

            trailing = trade["live"]["trailing"]
            adx = normalized_market["adx"]
            structure = normalized_market["structure"]
            atr_dollar = normalized_market["atrDollar"]
            if adx >= float(self.config["trailAdxThreshold"]) and self._structure_supports_direction(direction, structure):
                buffer = max(one_r * 0.25, atr_dollar * float(self.config["trailBufferMultiplier"]))
                if not trailing["active"]:
                    trailing["active"] = True
                    trailing["method"] = "VWAP_ADX"
                    trailing["adxAtTrigger"] = adx
                    trailing["buffer"] = round(buffer, 4)
                    trailing["highestPriceSinceTrail"] = normalized_market["price"]
                    trailing["lowestPriceSinceTrail"] = normalized_market["price"]
                    self._append_decision_locked(
                        trade,
                        "TRAIL_START",
                        normalized_market["price"],
                        "ADX and structure aligned. Trailing stop activated.",
                        market_context=normalized_market["regime"],
                    )
                    event = self._event(
                        "trade:alert",
                        trade,
                        "Trailing stop activated.",
                        normalized_market["price"],
                        extra={"alertType": "trailing"},
                    )
                    self._append_notification_locked(event)
                    events.append(event)

                proposed_stop = normalized_market["vwap"] - buffer if direction == "LONG" else normalized_market["vwap"] + buffer
                if self._move_stop_locked(
                    trade,
                    proposed_stop,
                    "Trailing stop moved with VWAP and ADX support.",
                    "TRAILING",
                ):
                    self._append_decision_locked(
                        trade,
                        "TRAIL_UPDATE",
                        normalized_market["price"],
                        "VWAP trailing stop updated.",
                        market_context=normalized_market["regime"],
                    )

            if trailing["active"]:
                adx_at_trigger = _safe_float(trailing.get("adxAtTrigger"), adx) or adx
                if adx <= (adx_at_trigger - float(self.config["adxExhaustionDrop"])) and unrealized_r > 0:
                    events.append(
                        self._close_trade_locked(
                            trade,
                            normalized_market["price"],
                            "ADX_EXHAUSTION",
                            "Trend strength faded after trailing activation.",
                        )
                    )
                    self._save_state_locked()
                    return {
                        "trade": copy.deepcopy(trade),
                        "events": events,
                        "dashboard": self.get_dashboard_payload(user_id=user_id, market_data=normalized_market),
                    }

                if self._structure_breaks_direction(direction, structure) and unrealized_r > 0:
                    events.append(
                        self._close_trade_locked(
                            trade,
                            normalized_market["price"],
                            "STRUCTURE_BREAK",
                            "Market structure flipped against the trade.",
                        )
                    )
                    self._save_state_locked()
                    return {
                        "trade": copy.deepcopy(trade),
                        "events": events,
                        "dashboard": self.get_dashboard_payload(user_id=user_id, market_data=normalized_market),
                    }

            self._save_state_locked()
            return {
                "trade": copy.deepcopy(trade),
                "events": events,
                "dashboard": self.get_dashboard_payload(user_id=user_id, market_data=normalized_market),
            }

    def update_trade(self, trade_id: str, updates: dict[str, Any], user_id: str = "anonymous") -> dict[str, Any]:
        with self.lock:
            _, trade = self._find_trade_locked(trade_id, user_id)
            allowed_flat = {
                "status": (trade, "status"),
                "currentPrice": (trade["live"], "currentPrice"),
                "unrealizedPnL": (trade["live"], "unrealizedPnL"),
                "unrealizedR": (trade["live"], "unrealizedR"),
                "highestPrice": (trade["live"], "highestPrice"),
                "lowestPrice": (trade["live"], "lowestPrice"),
                "stopLossCurrent": (trade["live"]["stopLoss"], "current"),
                "stopLossHistory": (trade["live"]["stopLoss"], "history"),
                "decisions": (trade["live"], "decisions"),
            }
            for key, value in (updates or {}).items():
                if key in allowed_flat:
                    target, field = allowed_flat[key]
                    target[field] = copy.deepcopy(value)
                elif key in {"entry", "plan", "live"} and isinstance(value, dict):
                    trade[key].update(copy.deepcopy(value))
            trade["updatedAt"] = _utc_now()
            self._save_state_locked()
            return copy.deepcopy(trade)

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        reason: str,
        reasoning: str,
        emotion: str = "calm",
        final_pnl: float | None = None,
        final_r: float | None = None,
        user_id: str = "anonymous",
    ) -> dict[str, Any]:
        with self.lock:
            _, trade = self._find_trade_locked(trade_id, user_id)
            event = self._close_trade_locked(trade, float(exit_price), reason, reasoning, emotion, final_pnl, final_r)
            trade["updatedAt"] = _utc_now()
            self._save_state_locked()
            return {"trade": copy.deepcopy(trade), "event": event}

    def tag_emotion(
        self,
        trade_id: str,
        emotion: str,
        note: str | None = None,
        price: float | None = None,
        unrealized_r: float | None = None,
        user_id: str = "anonymous",
    ) -> dict[str, Any]:
        with self.lock:
            _, trade = self._find_trade_locked(trade_id, user_id)
            log = {
                "tradeId": trade_id,
                "emotion": str(emotion or "calm").lower(),
                "note": str(note or "").strip(),
                "timestamp": _utc_now(),
                "price": round(price if price is not None else float(trade["live"]["currentPrice"]), 4),
                "unrealizedR": round(
                    unrealized_r if unrealized_r is not None else float(trade["live"].get("unrealizedR") or 0.0),
                    4,
                ),
            }
            self.state["emotion_logs"].append(log)
            self._append_decision_locked(
                trade,
                "EMOTION_TAG",
                log["price"],
                log["note"] or f"Emotion tagged as {log['emotion']}.",
                emotion=log["emotion"],
            )
            event = self._event(
                "emotion:updated",
                trade,
                f"Emotion tagged: {log['emotion']}",
                log["price"],
                extra={"emotion": log["emotion"], "note": log["note"]},
            )
            self._append_notification_locked(event)
            self._save_state_locked()
            return {"log": log, "event": event, "trade": copy.deepcopy(trade)}

    def record_snapshot(self, trade_id: str, snapshot: dict[str, Any], user_id: str = "anonymous") -> dict[str, Any]:
        with self.lock:
            _, trade = self._find_trade_locked(trade_id, user_id)
            payload = {
                "timestamp": _utc_now(),
                "price": _safe_float(snapshot.get("price"), trade["live"].get("currentPrice")) or trade["live"].get("currentPrice"),
                "adx": _safe_float(snapshot.get("adx"), None),
                "vwap": _safe_float(snapshot.get("vwap"), None),
                "atrDollar": _safe_float(snapshot.get("atrDollar"), None),
                "structure": snapshot.get("structure") or "neutral",
                "regime": snapshot.get("regime") or "normal",
            }
            trade["snapshots"].append(payload)
            trade["updatedAt"] = _utc_now()
            self._save_state_locked()
            return copy.deepcopy(payload)

    def get_current_session(self) -> str:
        hour = datetime.now(timezone.utc).hour
        if 0 <= hour < 8:
            return "Asia"
        if 8 <= hour < 13:
            return "London"
        if 13 <= hour < 21:
            return "New York"
        return "Rollover"

    def list_trades(
        self,
        user_id: str | None = None,
        status: str | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        with self.lock:
            trades = list(reversed(self._all_trades_locked(user_id)))
            if status:
                normalized_status = str(status).upper()
                trades = [trade for trade in trades if str(trade.get("status") or "").upper() == normalized_status]
            page = max(1, int(page))
            limit = max(1, min(200, int(limit)))
            total = len(trades)
            start = (page - 1) * limit
            end = start + limit
            return {
                "trades": copy.deepcopy(trades[start:end]),
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "pages": max(1, math.ceil(total / limit)) if total else 1,
                },
            }

    def get_active_trade(self, user_id: str | None = None) -> dict[str, Any] | None:
        with self.lock:
            trade = self._active_trade_locked(user_id)
            return copy.deepcopy(trade) if trade else None

    def get_stats(self, user_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            trades = self._all_trades_locked(user_id)
            return copy.deepcopy(self._compute_stats_locked(trades))

    def get_review(self, trade_id: str, user_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            _, trade = self._find_trade_locked(trade_id, user_id)
            if trade.get("review"):
                return copy.deepcopy(trade["review"])
            return self._derive_review_locked(trade)

    def get_dashboard_payload(
        self,
        user_id: str | None = None,
        market_data: dict[str, Any] | None = None,
        learning_direction: str | None = None,
        learning_setup: Any | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            analytics = self._build_analytics_locked(user_id)
            trades = self._all_trades_locked(user_id)
            recent = list(reversed(trades))[:8]
            active_trade = copy.deepcopy(self._active_trade_locked(user_id))
            market_snapshot = copy.deepcopy(market_data or self.state.get("last_market_data"))
            if learning_direction is None and isinstance(active_trade, dict):
                learning_direction = active_trade.get("direction")
            if learning_setup is None and isinstance(active_trade, dict):
                learning_setup = (active_trade.get("entry") or {}).get("trigger")
            return {
                "activeTrade": active_trade,
                "recentTrades": copy.deepcopy(recent),
                "stats": copy.deepcopy(self._compute_stats_locked(trades)),
                "analytics": analytics,
                "notifications": copy.deepcopy(self.state["notifications"][-6:]),
                "marketData": market_snapshot,
                "emotionLogs": copy.deepcopy(
                    [entry for entry in self.state["emotion_logs"] if not user_id or entry.get("tradeId") in {trade.get("id") for trade in trades}][-20:]
                ),
                "learning": copy.deepcopy(
                    self._get_learning_summary_locked(
                        user_id=user_id,
                        market_data=market_snapshot,
                        direction=learning_direction,
                        setup=learning_setup,
                    )
                ),
            }

    def get_setup_analytics(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_dashboard_payload(user_id)["analytics"]["setups"]

    def get_emotion_analytics(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_dashboard_payload(user_id)["analytics"]["emotions"]

    def get_session_analytics(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_dashboard_payload(user_id)["analytics"]["sessions"]

    def get_r_distribution(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_dashboard_payload(user_id)["analytics"]["distribution"]

    def get_monthly_analytics(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_dashboard_payload(user_id)["analytics"]["monthly"]
