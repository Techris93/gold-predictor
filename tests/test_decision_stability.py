import unittest

import app as app_module


def _sample_ta_payload(direction="Short", session="London", bar_timestamp=None):
    bar_timestamp = bar_timestamp or "2026-04-27T12:00:00Z"
    if direction == "Long":
        structure = "Bullish Breakout"
        vwap_delta = 0.24
        orb_state = 1
    else:
        structure = "Bearish Breakdown"
        vwap_delta = -0.24
        orb_state = -1
    return {
        "session_context": {
            "currentLabel": session,
            "barTimestampUtc": bar_timestamp,
        },
        "price_action": {"structure": structure},
        "structure_context": {
            "distSessionVwapPct": vwap_delta,
            "openingRangeBreak": orb_state,
        },
        "volatility_regime": {"adx_14": 34.0},
    }


def _sample_execution_quality(direction="Short", score=84):
    return {
        "grade": "A",
        "score": score,
        "status": "ready",
        "direction": direction,
        "setup": f"VWAP / ORB Continuation {direction}",
        "riskReward": 1.8,
    }


def _sample_decision_status(direction="Short"):
    return {
        "status": "buy" if direction == "Long" else "sell",
        "text": f"{direction} bias is building.",
    }


def _sample_execution_state(direction="Short", active=False):
    if active:
        return {
            "status": "hold",
            "actionState": "LONG_ACTIVE" if direction == "Long" else "SHORT_ACTIVE",
            "title": f"{direction} Active",
        }
    return {
        "status": "prepare",
        "actionState": "SETUP_LONG" if direction == "Long" else "SETUP_SHORT",
        "title": f"{direction} Setup",
    }


def _step_state(state, now_ts, direction="Short", score=84, active=False):
    ta_payload = _sample_ta_payload(direction=direction)
    regime_state = {"event_regime": "breakout_watch"}
    prediction = {"verdict": "Bullish" if direction == "Long" else "Bearish"}
    decision_status = _sample_decision_status(direction=direction)
    execution_state = _sample_execution_state(direction=direction, active=active)
    execution_quality = _sample_execution_quality(direction=direction, score=score)
    state, _ = app_module._update_stable_decision_buffer(
        state,
        ta_payload,
        regime_state,
        now_ts,
    )
    return app_module._apply_stable_decision_controls(
        state,
        prediction,
        ta_payload,
        regime_state,
        decision_status,
        execution_state,
        execution_quality,
        now_ts=now_ts,
        persist_churn=False,
    )


class StableDecisionEngineTests(unittest.TestCase):
    def test_trade_brain_direction_prefers_final_execution_direction(self):
        prediction = {
            "verdict": "Bearish",
            "DecisionStatus": {"status": "buy"},
            "ExecutionQuality": {"direction": "Long"},
            "StableDecision": {"direction": "Long"},
        }

        direction = app_module._trade_brain_direction_from_prediction(prediction)

        self.assertEqual(direction, "LONG")

    def test_promotes_candidate_only_after_buffered_confirmation_window(self):
        base_ts = 1710000000
        state = app_module._default_stable_decision_state(base_ts)

        for offset in (0, 30, 60):
            stable, decision_status, execution_state, execution_quality, state = _step_state(
                state,
                base_ts + offset,
                direction="Short",
                score=84,
            )
            self.assertEqual(stable["decision_state"], "CANDIDATE")
            self.assertEqual(execution_quality["status"], "watchlist")
            self.assertEqual(decision_status["decisionState"], "CANDIDATE")
            self.assertEqual(execution_state["decisionState"], "CANDIDATE")

        stable, decision_status, execution_state, execution_quality, state = _step_state(
            state,
            base_ts + 90,
            direction="Short",
            score=84,
        )

        self.assertEqual(stable["decision_state"], "CONFIRMED")
        self.assertEqual(stable["execution_quality"], "A")
        self.assertEqual(execution_quality["status"], "ready")
        self.assertEqual(decision_status["decisionState"], "CONFIRMED")
        self.assertEqual(execution_state["decisionState"], "CONFIRMED")

    def test_opposing_flip_is_suppressed_while_decision_lock_is_active(self):
        base_ts = 1710000000
        state = app_module._default_stable_decision_state(base_ts)

        for offset in (0, 30, 60, 90):
            stable, _, _, _, state = _step_state(
                state,
                base_ts + offset,
                direction="Short",
                score=84,
            )

        self.assertEqual(stable["decision_state"], "CONFIRMED")

        stable, _, _, _, state = _step_state(
            state,
            base_ts + 120,
            direction="Long",
            score=84,
        )
        self.assertEqual(stable["decision_state"], "CONFIRMED")

        stable, _, _, execution_quality, state = _step_state(
            state,
            base_ts + 150,
            direction="Long",
            score=84,
        )

        self.assertEqual(stable["decision_state"], "CONFIRMED")
        self.assertEqual(stable["suppression_reason"], "LOCKED")
        self.assertEqual(state["last_suppression_reason"], "LOCKED")
        self.assertEqual(execution_quality["status"], "ready")

    def test_oscillation_holds_current_candidate_instead_of_flipping_back(self):
        now_ts = 1710000500
        state = app_module._default_stable_decision_state(now_ts)
        long_setup = app_module._stable_decision_signal_setup(
            _sample_execution_quality(direction="Long", score=84)
        )
        short_setup = app_module._stable_decision_signal_setup(
            _sample_execution_quality(direction="Short", score=84)
        )
        long_signal = f"Long|{long_setup}|A"
        short_signal = f"Short|{short_setup}|A"

        state["stable_decision"].update(
            {
                "decision_state": "CANDIDATE",
                "setup_type": short_setup,
                "execution_quality": "A",
                "confidence": 84,
                "signal_signature": short_signal,
                "signature": f"{short_signal}|CANDIDATE",
                "direction": "Short",
                "session": "London",
                "change_reason": "MOMENTUM_SHIFT",
            }
        )
        state["decision_lock_until_ts"] = now_ts - 1
        state["decision_ring"] = [
            {
                "ts": now_ts - 400,
                "signature": f"{long_signal}|CANDIDATE",
                "decision_state": "CANDIDATE",
                "setup_type": long_setup,
                "execution_quality": "A",
                "confidence": 84,
            },
            {
                "ts": now_ts - 60,
                "signature": f"{short_signal}|CANDIDATE",
                "decision_state": "CANDIDATE",
                "setup_type": short_setup,
                "execution_quality": "A",
                "confidence": 84,
            },
        ]

        stable, _, _, execution_quality, state = _step_state(
            state,
            now_ts,
            direction="Long",
            score=84,
        )

        self.assertEqual(stable["decision_state"], "CANDIDATE")
        self.assertEqual(stable["direction"], "Short")
        self.assertEqual(stable["suppression_reason"], "OSCILLATION")
        self.assertEqual(state["last_suppression_reason"], "OSCILLATION")
        self.assertEqual(execution_quality["status"], "watchlist")

    def test_blocked_stable_override_clears_stale_trade_plan_fields(self):
        stable_decision = {
            "decision_state": "SCANNING",
            "direction": "Short",
            "execution_quality": "No Trade",
            "confidence": 59,
            "decision_locked_until": "2026-04-27T01:01:36Z",
            "flip_count_10m": 0,
            "change_reason": "MOMENTUM_SHIFT",
            "next_evaluation": "2026-04-27T01:02:30Z",
            "suppression_reason": "LOCKED",
        }
        decision_status = {"status": "buy", "text": "Raw long setup is ready."}
        execution_state = {"status": "enter", "actionState": "SETUP_LONG"}
        execution_quality = {
            "grade": "B",
            "score": 86,
            "status": "ready",
            "direction": "Long",
            "setup": "VWAP Pullback Long",
            "entry": {
                "low": 4696.44,
                "high": 4697.29,
                "mid": 4697.04,
                "text": "4696.44 - 4697.29",
            },
            "stopLoss": {
                "price": 4688.35,
                "basis": "PP / Daily Pivot Point",
                "distance": 8.69,
                "atrMultiple": 1.74,
            },
            "targets": [
                {
                    "label": "TP1",
                    "price": 4722.12,
                    "basis": "R2",
                    "rMultiple": 2.89,
                },
                {
                    "label": "TP2",
                    "price": 4728.2,
                    "basis": "2.2R runner target",
                    "rMultiple": 3.59,
                },
            ],
            "riskReward": 2.89,
            "positionSizeNote": "Normal planned risk only",
            "invalidations": [
                "Long thesis fails below 4688.35 (PP / Daily Pivot Point).",
                "Exit if price loses VWAP and ORB flips bearish.",
            ],
        }

        _, decision_status, execution_state, adjusted_quality = app_module._stable_decision_apply_payload_overrides(
            stable_decision,
            decision_status,
            execution_state,
            execution_quality,
        )

        self.assertEqual(decision_status["status"], "wait")
        self.assertEqual(execution_state["status"], "stand_aside")
        self.assertEqual(adjusted_quality["rawScore"], 86)
        self.assertEqual(adjusted_quality["rawGrade"], "B")
        self.assertEqual(adjusted_quality["rawDirection"], "Long")
        self.assertEqual(adjusted_quality["direction"], "Short")
        self.assertEqual(adjusted_quality["grade"], "No Trade")
        self.assertEqual(adjusted_quality["status"], "blocked")
        self.assertEqual(adjusted_quality["entry"]["text"], "Wait for a cleaner location")
        self.assertIsNone(adjusted_quality["entry"]["mid"])
        self.assertIsNone(adjusted_quality["stopLoss"]["price"])
        self.assertEqual(adjusted_quality["riskReward"], 0.0)
        self.assertEqual(adjusted_quality["positionSizeNote"], "Skip until blockers clear")
        self.assertEqual(
            adjusted_quality["invalidations"],
            ["No trade until VWAP, ORB, pivots, and ADX align."],
        )
        self.assertTrue(all(target["price"] is None for target in adjusted_quality["targets"]))


if __name__ == "__main__":
    unittest.main()