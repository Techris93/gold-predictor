import unittest
from unittest.mock import patch

import pandas as pd

import app as app_module
from tools import predict_gold
from tools.predict_gold import _normalize_ta_payload_schema
from tools.signal_engine import build_ta_payload_from_row


def _sample_row():
    return pd.Series(
        {
            "Close": 4833.25,
            "Open": 4831.10,
            "EMA_TREND": "Bearish",
            "EMA_20": 4838.91,
            "EMA_50": 4842.37,
            "ADX_14": 31.4,
            "ATR_14": 8.55,
            "ATR_PERCENT": 0.177,
            "M15_TREND": "Bearish",
            "H1_TREND": "Bearish",
            "H4_TREND": "Neutral",
            "ALIGNMENT_SCORE": -2,
            "ALIGNMENT_LABEL": "Strong Bearish Alignment",
            "PA_STRUCTURE": "Bearish Drift",
            "CANDLE_PATTERN": "None",
            "CMF_14": -0.08,
            "OBV": 120500.0,
            "OBV_PREV": 121000.0,
            "VOLUME_SIGNAL": "Slight Selling Bias",
            "MACD_LINE": -1.12,
            "MACD_SIGNAL": -0.95,
            "MACD_HIST": -0.17,
            "MACD_HIST_SLOPE": -0.03,
            "VOLUME_ZSCORE": 1.4,
            "VOLUME_SPIKE": 1,
            "OPENING_RANGE_BREAK": -1,
            "SWEEP_RECLAIM_SIGNAL": 0,
            "SWEEP_RECLAIM_QUALITY": 0.0,
            "SESSION_VWAP": 4836.12,
            "DIST_SESSION_VWAP_PCT": -0.06,
            "RECENT_SWING_HIGH_24": 4846.40,
            "RECENT_SWING_LOW_24": 4826.85,
            "RECENT_SWING_HIGH_96": 4860.20,
            "RECENT_SWING_LOW_96": 4798.70,
            "PIVOT_POINT": 4834.06,
            "PIVOT_R1": 4834.24,
            "PIVOT_S1": 4833.91,
            "PIVOT_R2": 4834.39,
            "PIVOT_S2": 4833.73,
            "ROUND_NUMBER_SUPPORT": 4830.0,
            "ROUND_NUMBER_RESISTANCE": 4835.0,
            "MAJOR_ROUND_NUMBER_SUPPORT": 4830.0,
            "MAJOR_ROUND_NUMBER_RESISTANCE": 4840.0,
            "ROUND_NUMBER_STEP": 5.0,
            "ROUND_SUPPORT_DISTANCE_PCT": 0.07,
            "ROUND_RESISTANCE_DISTANCE_PCT": 0.04,
            "BULLISH_FVG_LOW": 4828.20,
            "BULLISH_FVG_HIGH": 4830.60,
            "BULLISH_FVG_DISTANCE_PCT": 0.06,
            "BEARISH_FVG_LOW": 4836.25,
            "BEARISH_FVG_HIGH": 4844.72,
            "BEARISH_FVG_DISTANCE_PCT": 0.06,
            "IN_BULLISH_FVG": 0,
            "IN_BEARISH_FVG": 0,
            "RANGE_ZONE_LOW": 4829.50,
            "RANGE_ZONE_HIGH": 4837.50,
            "RANGE_ZONE_ACTIVE": 1,
            "IN_RANGE_ZONE": 1,
            "RANGE_ZONE_BREAK": -1,
            "RANGE_ZONE_POSITION": 0.46,
            "RANGE_ZONE_WIDTH_PCT": 0.17,
            "REALIZED_VOL_8": 0.19,
            "REALIZED_VOL_32": 0.14,
            "REALIZED_VOL_RATIO": 1.23,
            "REALIZED_VOL_PERCENTILE": 72.0,
            "ATR_PERCENTILE": 68.0,
            "TREND_FOLLOW_THROUGH": 1.08,
            "TREND_PROBABILITY": 0.79,
            "EXPANSION_PROBABILITY": 0.66,
            "CHOP_PROBABILITY": 0.17,
            "EVENT_ACTIVE": 0,
        },
        name=pd.Timestamp("2026-04-20T12:00:00Z"),
    )


def _sample_ta_payload():
    payload = build_ta_payload_from_row(_sample_row())
    payload["support_resistance"]["nearest_support"] = {
        "label": "Previous Day Low",
        "price": 4833.88,
    }
    payload["support_resistance"]["nearest_resistance"] = {
        "label": "Previous Day High",
        "price": 4834.21,
    }
    return payload


class DashboardPayloadContractTests(unittest.TestCase):
    def test_decision_status_uses_execution_state_for_active_short(self):
        ta = _sample_ta_payload()
        trade_guidance = {
            "summary": "Short setup confirmed with acceptable tradeability.",
            "buyLevel": "Weak",
            "sellLevel": "Strong",
            "exitLevel": "Low",
        }

        decision = app_module._evaluate_decision_status(
            verdict="Bearish",
            confidence=66,
            ta_data=ta,
            trade_guidance=trade_guidance,
            execution_state={"actionState": "SHORT_ACTIVE", "status": "enter"},
        )

        self.assertEqual(decision["status"], "sell")
        self.assertEqual(
            decision["text"],
            "Safer to look for a sell. Short state is confirmed with acceptable tradeability.",
        )

    def test_decision_status_uses_h1_trend_for_generic_checklist(self):
        ta = _sample_ta_payload()
        ta["ema_trend"] = "Bullish"
        ta["multi_timeframe"]["h1_trend"] = "Bearish"
        ta["multi_timeframe"]["alignment_label"] = "Strong Bearish Alignment"
        ta["price_action"]["structure"] = "Bearish Drift"
        trade_guidance = {
            "summary": "Sell bias is favored; wait for bearish continuation or rejection confirmation.",
            "buyLevel": "Weak",
            "sellLevel": "Strong",
            "exitLevel": "Low",
        }

        decision = app_module._evaluate_decision_status(
            verdict="Bearish",
            confidence=66,
            ta_data=ta,
            trade_guidance=trade_guidance,
            execution_state={},
        )

        self.assertEqual(decision["status"], "sell")
        self.assertEqual(
            decision["text"],
            "Safer to look for a sell. Most sell conditions are confirmed.",
        )

    def test_decision_status_uses_surviving_tradeability_and_regime_for_wait_blockers(self):
        ta = _sample_ta_payload()
        trade_guidance = {
            "summary": "Sell bias is favored, but conditions are not clean enough for execution yet.",
            "buyLevel": "Weak",
            "sellLevel": "Strong",
            "exitLevel": "Low",
        }

        decision = app_module._evaluate_decision_status(
            verdict="Bearish",
            confidence=51,
            ta_data=ta,
            trade_guidance=trade_guidance,
            execution_state={"actionState": "WAIT", "status": "stand_aside", "tradeability": "High"},
            tradeability="High",
            regime="range",
        )

        self.assertEqual(decision["status"], "wait")
        self.assertIn("execution is blocked because the market regime is range", decision["text"])
        self.assertNotIn("tradeability is still low", decision["text"])

    def test_build_ta_payload_from_row_keeps_core_context(self):
        payload = build_ta_payload_from_row(_sample_row())

        self.assertIn("pivot_levels", payload["support_resistance"])
        self.assertIn("openingRangeBreak", payload["structure_context"])
        self.assertIn("distSessionVwapPct", payload["structure_context"])
        self.assertIn("pivotPoint", payload["structure_context"])
        self.assertNotIn("rsi_14", payload)
        self.assertNotIn("rsi_signal", payload)
        self.assertNotIn("round_numbers", payload["support_resistance"])
        self.assertNotIn("active_fvgs", payload["support_resistance"])
        self.assertNotIn("range_zone", payload["support_resistance"])
        self.assertNotIn("roundNumberSupport", payload["structure_context"])
        self.assertNotIn("bullishFvg", payload["structure_context"])
        self.assertNotIn("rangeZone", payload["structure_context"])

    def test_market_structure_uses_last_closed_15m_bar(self):
        candles = []
        for idx in range(28):
            base = 100.0 + (idx * 0.1)
            candles.append(
                {
                    "Open": base,
                    "High": base + 0.6,
                    "Low": base - 0.6,
                    "Close": base + 0.2,
                    "Volume": 1000,
                }
            )
        candles.extend(
            [
                {"Open": 101.5, "High": 102.0, "Low": 101.0, "Close": 101.8, "Volume": 1000},
                {"Open": 101.8, "High": 102.6, "Low": 101.4, "Close": 102.2, "Volume": 1000},
                {"Open": 102.2, "High": 103.2, "Low": 101.9, "Close": 102.9, "Volume": 1000},
                {"Open": 102.9, "High": 103.0, "Low": 96.5, "Close": 97.2, "Volume": 1000},
            ]
        )
        frame = pd.DataFrame(
            candles,
            index=pd.date_range("2026-04-20T09:00:00Z", periods=len(candles), freq="15min", tz="UTC"),
        )

        with patch.object(
            predict_gold,
            "_fetch_mtf_trends",
            return_value={
                "m15_trend": "Bullish",
                "h1_trend": "Bullish",
                "h4_trend": "Neutral",
                "alignment_score": 2,
                "alignment_label": "Strong Bullish Alignment",
                "data_points": {"m15": 200, "h1": 200, "h4": 200},
                "sources": {"m15": "test", "h1": "test", "h4": "test"},
            },
        ), patch.object(
            predict_gold,
            "_get_cached_cross_asset_context",
            return_value={"bias": "Neutral", "score": 0.0, "display": "Neutral"},
        ):
            payload = predict_gold._build_technical_analysis_from_frame(
                frame,
                td_symbol="XAU/USD",
                now_ts=1713600000,
                data_source="unit-test",
            )

        self.assertEqual(payload["price_action"]["basis"], "last_closed_15m_bar")
        self.assertEqual(payload["price_action"]["bar_timestamp_utc"], "2026-04-20T16:30:00+00:00")
        self.assertEqual(payload["price_action"]["structure"], "Higher Highs / Higher Lows (Bullish Structure)")

    def test_normalize_ta_payload_schema_keeps_pivots_and_microstructure(self):
        normalized = _normalize_ta_payload_schema(_sample_ta_payload())

        self.assertIn("pivot_levels", normalized["support_resistance"])
        self.assertIn("openingRangeBreak", normalized["structure_context"])
        self.assertIn("distSessionVwapPct", normalized["structure_context"])
        self.assertIn("pivotPoint", normalized["structure_context"])
        self.assertNotIn("round_numbers", normalized["support_resistance"])
        self.assertNotIn("active_fvgs", normalized["support_resistance"])
        self.assertNotIn("range_zone", normalized["support_resistance"])
        self.assertNotIn("roundNumberSupport", normalized["structure_context"])
        self.assertNotIn("bullishFvg", normalized["structure_context"])
        self.assertNotIn("rangeZone", normalized["structure_context"])

    def test_sanitize_client_payload_strips_removed_dashboard_fields(self):
        payload = {
            "status": "success",
            "verdict": "Bearish",
            "confidence": 66,
            "TechnicalAnalysis": _sample_ta_payload(),
            "TradeGuidance": {
                "summary": "Short setup confirmed.",
                "buyLevel": "Weak",
                "sellLevel": "Strong",
                "exitLevel": "Low",
            },
            "MarketState": {"regime": "trend", "action_state": "SHORT_ACTIVE"},
            "RegimeState": {"breakout_bias": "Bearish", "event_regime": "range_expansion"},
            "ForecastState": {"regimeBucket": "active_momentum"},
            "ExecutionState": {"status": "hold", "title": "Hold Winner"},
            "RR200Signal": {"status": "arming"},
            "DecisionStatus": {"status": "sell", "text": "Safer to look for a sell."},
            "ExecutionPermission": {"status": "entry_allowed"},
            "TradePlaybook": {"stage": "hold", "title": "Hold Winner"},
            "DashboardAction": {"label": "Sell", "reason": "Active short remains aligned."},
        }

        body = app_module._sanitize_client_prediction_payload(payload)

        for key in (
            "MarketState",
            "ExecutionPermission",
            "TradePlaybook",
            "DashboardAction",
            "RR200Signal",
        ):
            self.assertNotIn(key, body)

        self.assertNotIn("rsi_14", body["TechnicalAnalysis"])
        self.assertNotIn("rsi_signal", body["TechnicalAnalysis"])
        self.assertNotIn("round_numbers", body["TechnicalAnalysis"]["support_resistance"])
        self.assertNotIn("active_fvgs", body["TechnicalAnalysis"]["support_resistance"])
        self.assertNotIn("range_zone", body["TechnicalAnalysis"]["support_resistance"])
        self.assertNotIn("roundNumberSupport", body["TechnicalAnalysis"]["structure_context"])
        self.assertNotIn("bullishFvg", body["TechnicalAnalysis"]["structure_context"])
        self.assertNotIn("rangeZone", body["TechnicalAnalysis"]["structure_context"])
        self.assertIn("pivot_levels", body["TechnicalAnalysis"]["support_resistance"])
        self.assertIn("pivotPoint", body["TechnicalAnalysis"]["structure_context"])
        self.assertIn("RegimeState", body)
        self.assertIn("ExecutionState", body)

    def test_predict_route_returns_lean_client_contract(self):
        payload = {
            "status": "success",
            "verdict": "Bearish",
            "confidence": 66,
            "TechnicalAnalysis": _sample_ta_payload(),
            "TradeGuidance": {
                "summary": "Short setup confirmed.",
                "buyLevel": "Weak",
                "sellLevel": "Strong",
                "exitLevel": "Low",
            },
            "MarketState": {"regime": "trend", "action_state": "SHORT_ACTIVE"},
            "RegimeState": {"breakout_bias": "Bearish", "event_regime": "range_expansion"},
            "ForecastState": {"regimeBucket": "active_momentum"},
            "ExecutionState": {"status": "hold", "title": "Hold Winner"},
            "RR200Signal": {"status": "arming"},
            "DecisionStatus": {"status": "sell", "text": "Safer to look for a sell."},
            "ExecutionPermission": {"status": "entry_allowed"},
            "TradePlaybook": {"stage": "hold", "title": "Hold Winner"},
            "DashboardAction": {"label": "Sell", "reason": "Active short remains aligned."},
        }

        with patch.object(app_module, "_build_prediction_response", return_value=(payload, 200)):
            client = app_module.app.test_client()
            response = client.get("/api/predict")

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        for key in (
            "TradeGuidance",
            "RegimeState",
            "ForecastState",
            "ExecutionState",
            "DecisionStatus",
        ):
            self.assertIn(key, body)
        for key in (
            "MarketState",
            "ExecutionPermission",
            "TradePlaybook",
            "DashboardAction",
            "RR200Signal",
        ):
            self.assertNotIn(key, body)
        self.assertNotIn("rsi_14", body["TechnicalAnalysis"])
        self.assertNotIn("rsi_signal", body["TechnicalAnalysis"])
        self.assertNotIn("round_numbers", body["TechnicalAnalysis"]["support_resistance"])
        self.assertNotIn("active_fvgs", body["TechnicalAnalysis"]["support_resistance"])
        self.assertNotIn("range_zone", body["TechnicalAnalysis"]["support_resistance"])
        self.assertIn("pivot_levels", body["TechnicalAnalysis"]["support_resistance"])

    def test_prediction_builder_returns_lean_backend_contract(self):
        ta_payload = _sample_ta_payload()
        prediction_payload = {
            "verdict": "Bearish",
            "confidence": 66,
            "TradeGuidance": {
                "summary": "Short setup confirmed.",
                "buyLevel": "Weak",
                "sellLevel": "Strong",
                "exitLevel": "Low",
            },
            "RegimeState": {"breakout_bias": "Bearish", "event_regime": "range_expansion"},
            "ForecastState": {"regimeBucket": "active_momentum"},
            "ExecutionState": {"status": "hold", "title": "Hold Winner", "text": "Manage the active short."},
            "DecisionStatus": {"status": "sell", "text": "Safer to look for a sell."},
            "MarketState": {"regime": "trend", "action_state": "SHORT_ACTIVE"},
            "ExecutionPermission": {"status": "entry_allowed"},
            "TradePlaybook": {"stage": "hold", "title": "Hold Winner"},
            "DashboardAction": {"label": "Sell"},
            "RR200Signal": {"status": "arming"},
        }

        with patch.object(app_module.predict_gold, "get_technical_analysis", return_value=ta_payload), patch.object(
            app_module, "compute_prediction_from_ta", return_value=prediction_payload
        ), patch.object(
            app_module, "_stabilize_prediction", side_effect=lambda prediction, _: prediction
        ), patch.object(
            app_module,
            "_evaluate_decision_status",
            return_value={"status": "sell", "text": "Safer to look for a sell."},
        ), patch.object(
            app_module,
            "_stabilize_decision_status",
            side_effect=lambda decision: decision,
        ):
            body, status_code = app_module._build_prediction_response()

        self.assertEqual(status_code, 200)
        self.assertIn("TradeGuidance", body)
        self.assertIn("RegimeState", body)
        self.assertIn("ForecastState", body)
        self.assertIn("ExecutionState", body)
        self.assertIn("DecisionStatus", body)
        for key in (
            "MarketState",
            "ExecutionPermission",
            "TradePlaybook",
            "DashboardAction",
            "RR200Signal",
            "RR200LiveCounter",
        ):
            self.assertNotIn(key, body)

    def test_prediction_builder_aligns_decision_status_with_active_execution_state(self):
        ta_payload = _sample_ta_payload()

        with patch.object(app_module.predict_gold, "get_technical_analysis", return_value=ta_payload):
            body, status_code = app_module._build_prediction_response()

        self.assertEqual(status_code, 200)
        self.assertEqual(body["ExecutionState"]["actionState"], "SHORT_ACTIVE")
        self.assertEqual(body["DecisionStatus"]["status"], "sell")
        self.assertEqual(
            body["DecisionStatus"]["text"],
            "Safer to look for a sell. Short state is confirmed with acceptable tradeability.",
        )

    def test_predict_route_returns_error_payload_when_builder_fails(self):
        with patch.object(app_module, "_build_prediction_response", side_effect=RuntimeError("boom")):
            client = app_module.app.test_client()
            response = client.get("/api/predict")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"status": "error", "message": "boom"})


if __name__ == "__main__":
    unittest.main()
