import unittest

import app as app_module
from tools.signal_engine import compute_prediction_from_ta


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


def _streamlined_bearish_ta_payload():
    return {
        "current_price": 3320.0,
        "ema_20": 3326.0,
        "ema_50": 3334.0,
        "price_action": {
            "structure": "Bearish Breakdown",
            "latest_candle_pattern": "None",
        },
        "support_resistance": {
            "reaction": "Bearish Resistance Rejection",
            "support_distance_pct": 0.08,
            "resistance_distance_pct": 0.32,
            "nearest_support": {"label": "Recent Swing Low", "price": 3312.0},
            "nearest_resistance": {"label": "Recent Swing High", "price": 3332.0},
            "nearby_supports": [{"label": "Recent Swing Low", "price": 3312.0}],
            "nearby_resistances": [{"label": "Recent Swing High", "price": 3332.0}],
            "pivot_levels": {
                "pp": 3324.0,
                "r1": 3336.0,
                "s1": 3310.0,
                "r2": 3346.0,
                "s2": 3300.0,
            },
        },
        "structure_context": {
            "distSessionVwapPct": -0.12,
            "sessionVwap": 3325.0,
            "openingRangeBreak": -1,
            "sweepReclaimSignal": 0,
            "recentSwingHigh": 3330.0,
            "recentSwingLow": 3321.0,
            "pivotPoint": 3324.0,
            "pivotResistance1": 3336.0,
            "pivotSupport1": 3310.0,
            "pivotResistance2": 3346.0,
            "pivotSupport2": 3300.0,
        },
        "volatility_regime": {
            "market_regime": "Weak Trend",
            "adx_14": 24.0,
            "atr_14": 8.0,
            "atr_percent": 0.24,
        },
        "multi_timeframe": {
            "m15_trend": "Bearish",
            "h1_trend": "Bearish",
            "h4_trend": "Neutral",
            "alignment_score": -2,
            "alignment_label": "Strong Bearish Alignment",
        },
        "momentum_features": {
            "macdHistogramSlope": -0.08,
            "volumeSpike": 1,
        },
        "volume_analysis": {
            "overall_volume_signal": "Strong Selling Pressure (Distribution)",
        },
        "session_context": {
            "isMarketClosed": False,
            "current_session": {"isMarketClosed": False},
            "bar_session": {"marketStatus": "open"},
        },
        "event_risk": {"active": False},
        "market_regime_scores": {
            "trend_probability": 0.62,
            "expansion_probability": 0.58,
            "chop_probability": 0.18,
        },
        "active_strategy_params": {},
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
    def test_streamlined_bearish_signal_stays_actionable(self):
        ta_data = _streamlined_bearish_ta_payload()

        prediction = compute_prediction_from_ta(ta_data)
        market_state = app_module._build_market_state_from_prediction(prediction)
        execution_state = dict(prediction.get("ExecutionState") or {})
        decision_status = app_module._evaluate_decision_status(
            verdict=prediction["verdict"],
            confidence=int(prediction["confidence"]),
            ta_data=ta_data,
            trade_guidance=prediction["TradeGuidance"],
            execution_state=execution_state,
            tradeability=prediction.get("tradeability"),
            regime=prediction.get("regime"),
        )
        execution_permission = app_module._evaluate_execution_permission(
            decision_status,
            market_state,
        )
        execution_quality = app_module._build_execution_quality_plan(
            ta_data,
            prediction.get("RegimeState") or {},
            decision_status,
            execution_state,
        )

        self.assertEqual(prediction["verdict"], "Bearish")
        self.assertEqual(prediction["actionState"], "SHORT_ACTIVE")
        self.assertEqual(prediction.get("signalEngineMode"), "streamlined_fixed")
        self.assertEqual(prediction.get("tradeability"), "High")
        self.assertEqual(decision_status["status"], "sell")
        self.assertNotIn("Watchlist", decision_status["text"])
        self.assertEqual(execution_permission["status"], "entry_allowed")
        self.assertEqual(execution_quality["status"], "ready")
        self.assertEqual(execution_quality["grade"], "A")
        self.assertEqual(execution_quality["setup"], "VWAP / ORB Continuation Short")
        self.assertEqual(execution_quality["riskReward"], 2.0)

    def test_trade_brain_direction_prefers_final_execution_direction(self):
        prediction = {
            "verdict": "Bearish",
            "DecisionStatus": {"status": "buy"},
            "ExecutionQuality": {"direction": "Long"},
            "StableDecision": {"direction": "Long"},
        }

        direction = app_module._trade_brain_direction_from_prediction(prediction)

        self.assertEqual(direction, "LONG")

    def test_micro_vwap_notification_snapshot_ignores_hidden_third_decimal_boundary_wobble(self):
        previous_snapshot = {"micro_vwap_bias": "Mild Bullish", "micro_vwap_delta_pct": 0.11}
        dip_payload = {
            "TechnicalAnalysis": {
                "structure_context": {
                    "distSessionVwapPct": 0.079,
                }
            }
        }

        dip_snapshot = app_module._extract_indicator_snapshot(
            dip_payload,
            previous_snapshot=previous_snapshot,
        )

        self.assertEqual(dip_snapshot["micro_vwap_delta_pct"], 0.08)
        self.assertEqual(dip_snapshot["micro_vwap_bias"], "Mild Bullish")
        self.assertFalse(
            app_module._is_material_change(
                {
                    "micro_vwap_bias": {
                        "previous": previous_snapshot["micro_vwap_bias"],
                        "current": dip_snapshot["micro_vwap_bias"],
                    },
                    "micro_vwap_delta_pct": {
                        "previous": previous_snapshot["micro_vwap_delta_pct"],
                        "current": dip_snapshot["micro_vwap_delta_pct"],
                    },
                }
            )
        )

        rebound_payload = {
            "TechnicalAnalysis": {
                "structure_context": {
                    "distSessionVwapPct": 0.141,
                }
            }
        }

        rebound_snapshot = app_module._extract_indicator_snapshot(
            rebound_payload,
            previous_snapshot=dip_snapshot,
        )

        self.assertEqual(rebound_snapshot["micro_vwap_delta_pct"], 0.14)
        self.assertEqual(rebound_snapshot["micro_vwap_bias"], "Mild Bullish")
        self.assertFalse(
            app_module._is_material_change(
                {
                    "micro_vwap_bias": {
                        "previous": dip_snapshot["micro_vwap_bias"],
                        "current": rebound_snapshot["micro_vwap_bias"],
                    },
                    "micro_vwap_delta_pct": {
                        "previous": dip_snapshot["micro_vwap_delta_pct"],
                        "current": rebound_snapshot["micro_vwap_delta_pct"],
                    },
                }
            )
        )

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