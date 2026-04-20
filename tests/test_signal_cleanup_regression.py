import copy
import unittest
from unittest.mock import patch

import pandas as pd

import app as app_module
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
    payload["rsi_14"] = 68.2
    payload["rsi_signal"] = "Overbought (Bearish bias)"
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
    def test_build_ta_payload_from_row_preserves_dashboard_fields(self):
        payload = build_ta_payload_from_row(_sample_row())

        self.assertIn("pivot_levels", payload["support_resistance"])
        self.assertIn("round_numbers", payload["support_resistance"])
        self.assertIn("active_fvgs", payload["support_resistance"])
        self.assertIn("range_zone", payload["support_resistance"])
        self.assertIn("openingRangeBreak", payload["structure_context"])
        self.assertIn("distSessionVwapPct", payload["structure_context"])
        self.assertIn("roundNumberSupport", payload["structure_context"])
        self.assertIn("bullishFvg", payload["structure_context"])
        self.assertIn("rangeZone", payload["structure_context"])

    def test_normalize_ta_payload_schema_preserves_rsi_and_sr_context(self):
        normalized = _normalize_ta_payload_schema(_sample_ta_payload())

        self.assertEqual(normalized["rsi_14"], 68.2)
        self.assertEqual(normalized["rsi_signal"], "Overbought (Bearish bias)")
        self.assertIn("round_numbers", normalized["support_resistance"])
        self.assertIn("active_fvgs", normalized["support_resistance"])
        self.assertIn("range_zone", normalized["support_resistance"])
        self.assertIn("roundNumberSupport", normalized["structure_context"])
        self.assertIn("bearishFvg", normalized["structure_context"])
        self.assertIn("rangeZoneBreak", normalized["structure_context"])

    def test_predict_route_returns_full_dashboard_contract(self):
        payload = {
            "status": "success",
            "verdict": "Bearish",
            "confidence": 66,
            "TechnicalAnalysis": _sample_ta_payload(),
            "TradeGuidance": {"summary": "Short setup confirmed.", "buyLevel": "Weak", "sellLevel": "Strong", "exitLevel": "Low"},
            "MarketState": {"regime": "trend", "action_state": "SHORT_ACTIVE", "action": "sell"},
            "RegimeState": {"breakout_bias": "Bearish", "event_regime": "range_expansion"},
            "ForecastState": {"regimeBucket": "active_momentum"},
            "ExecutionState": {"status": "hold", "title": "Hold Winner", "action": "hold"},
            "RR200Signal": {"status": "arming"},
            "DecisionStatus": {"status": "sell", "text": "Safer to look for a sell."},
            "ExecutionPermission": {"status": "entry_allowed", "text": "Execution remains active on the sell side."},
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
            "MarketState",
            "RegimeState",
            "ExecutionPermission",
            "TradePlaybook",
            "ForecastState",
            "ExecutionState",
            "DashboardAction",
            "RR200Signal",
        ):
            self.assertIn(key, body)
        self.assertIn("rsi_14", body["TechnicalAnalysis"])
        self.assertIn("rsi_signal", body["TechnicalAnalysis"])
        self.assertIn("round_numbers", body["TechnicalAnalysis"]["support_resistance"])
        self.assertIn("active_fvgs", body["TechnicalAnalysis"]["support_resistance"])

    def test_predict_route_returns_error_payload_when_builder_fails(self):
        with patch.object(app_module, "_build_prediction_response", side_effect=RuntimeError("boom")):
            client = app_module.app.test_client()
            response = client.get("/api/predict")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"status": "error", "message": "boom"})


if __name__ == "__main__":
    unittest.main()
