import copy
import unittest
from unittest.mock import patch

import app as app_module
import numpy as np
import pandas as pd

from app import _sanitize_client_prediction_payload
from tools.event_regime import compute_event_regime_snapshot
from tools.predict_gold import _normalize_ta_payload_schema
from tools.signal_engine import (
    build_ta_payload_from_row,
    compute_prediction_from_ta,
    prepare_historical_features,
)


ALLOWED_STRUCTURE_CONTEXT_KEYS = {
    "openingRangeBreak",
    "sweepReclaimSignal",
    "sweepReclaimQuality",
    "sessionVwap",
    "distSessionVwapPct",
    "recentSwingHigh",
    "recentSwingLow",
    "pivotPoint",
    "pivotResistance1",
    "pivotSupport1",
    "pivotResistance2",
    "pivotSupport2",
}


def _make_feature_input_frame():
    index = pd.date_range("2024-01-01 00:00", periods=60, freq="1h", tz="UTC")
    steps = np.arange(len(index), dtype=float)
    close = 100.0 + (steps * 0.035) + (np.sin(steps / 4.0) * 0.18)
    open_price = close + (np.cos(steps / 3.0) * 0.05)
    high = np.maximum(open_price, close) + 0.22
    low = np.minimum(open_price, close) - 0.22
    volume = 1000 + (steps * 12)
    return pd.DataFrame(
        {
            "Open": np.round(open_price, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume.astype(int),
            "COMPRESSION_RATIO": np.full(len(index), 0.82),
            "SQUEEZE_ON": np.zeros(len(index), dtype=int),
        },
        index=index,
    )


def _prepared_feature_row():
    features = prepare_historical_features(_make_feature_input_frame())
    return features, features.iloc[-1]


def _base_prediction_input():
    _, row = _prepared_feature_row()
    ta_data = build_ta_payload_from_row(row)
    ta_data["current_price"] = 101.0
    ta_data["ema_trend"] = "Bullish"
    ta_data["execution_trend"] = "Bullish"
    ta_data["volatility_regime"] = {
        "market_regime": "Trending",
        "adx_14": 28.0,
        "atr_14": 0.65,
        "atr_percent": 0.42,
    }
    ta_data["multi_timeframe"] = {
        "m15_trend": "Bullish",
        "h1_trend": "Bullish",
        "h4_trend": "Bullish",
        "alignment_score": 3,
        "alignment_label": "Strong Bullish Alignment",
        "sources": {
            "m15": "derived_proxy_or_base",
            "h1": "derived_resample",
            "h4": "derived_resample",
        },
    }
    ta_data["price_action"] = {
        "structure": "Bullish Drift",
        "latest_candle_pattern": "None",
    }
    ta_data["volume_analysis"] = {
        "cmf_14": 0.02,
        "obv_trend": "Rising",
        "overall_volume_signal": "Neutral",
    }
    ta_data["momentum_features"] = {
        "macdLine": 0.02,
        "macdSignal": 0.01,
        "macdHistogram": 0.01,
        "macdHistogramSlope": 0.0,
        "volumeZScore": 0.2,
        "volumeSpike": 0,
    }
    ta_data["market_regime_scores"] = {
        "trend_probability": 0.68,
        "expansion_probability": 0.52,
        "chop_probability": 0.20,
    }
    ta_data["event_regime"] = {
        "breakout_bias": "Neutral",
        "cross_asset_bias": "Neutral",
        "big_move_risk": 36.0,
        "expansion_probability_30m": 54.0,
        "expansion_probability_60m": 59.0,
        "warning_ladder": "Normal",
        "event_regime": "normal",
        "components": {},
    }
    ta_data["event_risk"] = {
        "active": False,
    }
    ta_data["_regime_memory"] = {}
    ta_data["structure_context"] = {
        "openingRangeBreak": 0,
        "sweepReclaimSignal": 0,
        "sweepReclaimQuality": 0.0,
        "sessionVwap": 100.72,
        "distSessionVwapPct": 0.12,
        "recentSwingHigh": 101.25,
        "recentSwingLow": 100.48,
        "pivotPoint": 100.80,
        "pivotResistance1": 101.45,
        "pivotSupport1": 100.62,
        "pivotResistance2": 101.88,
        "pivotSupport2": 100.24,
    }
    ta_data["support_resistance"] = {
        "pivot_levels": {
            "pp": 100.80,
            "r1": 101.45,
            "s1": 100.62,
            "r2": 101.88,
            "s2": 100.24,
        }
    }
    return ta_data


def _legacy_ta_payload():
    return {
        "current_price": 101.0,
        "rsi_14": 68.2,
        "rsi_signal": "Overbought (Bearish bias)",
        "momentum_features": {
            "macdHistogram": 0.01,
            "macdHistogramSlope": 0.0,
            "volumeZScore": 0.2,
            "volumeSpike": 0,
            "rsiBullishDivergence": 1,
            "rsiBearishDivergence": 0,
            "rsiDivergenceStrength": 0.7,
        },
        "support_resistance": {
            "pivotLevels": {
                "pp": 100.8,
                "r1": 101.4,
                "s1": 100.6,
                "r2": 101.9,
                "s2": 100.2,
            },
            "nearby_supports": [{"label": "Legacy Support"}],
            "nearby_resistances": [{"label": "Legacy Resistance"}],
            "round_numbers": {
                "support": 100.0,
                "resistance": 101.0,
            },
            "active_fvgs": {
                "bullish": {"low": 100.1, "high": 100.4},
            },
            "range_zone": {"low": 100.2, "high": 100.8},
        },
        "structure_context": {
            "openingRangeBreak": 1,
            "sweepReclaimSignal": 1,
            "sweepReclaimQuality": 0.8,
            "sessionVwap": 100.7,
            "distSessionVwapPct": 0.18,
            "recentSwingHigh": 101.2,
            "recentSwingLow": 100.5,
            "pivotPoint": 100.8,
            "pivotResistance1": 101.4,
            "pivotSupport1": 100.6,
            "pivotResistance2": 101.9,
            "pivotSupport2": 100.2,
            "roundNumberSupport": 100.0,
            "bullishFvg": {"low": 100.1, "high": 100.4},
            "rangeZone": {"low": 100.2, "high": 100.8},
        },
        "_prediction_market_state": {"internal": True},
    }


def _base_event_regime_row():
    return {
        "COMPRESSION_RATIO": 0.70,
        "ATR_EXPANSION_RATIO": 1.08,
        "VOL_OF_VOL": 0.02,
        "GAP_PCT": 0.0,
        "BAR_VELOCITY": 0.55,
        "MICRO_RETURN_BURST": 1.02,
        "VELOCITY_DECAY": 1.0,
        "WICK_ASYMMETRY_PERSISTENCE": 0.0,
        "ASIAN_RANGE_POSITION": 0.50,
        "RANGE_POSITION_24": 0.55,
        "ATR_14": 1.0,
        "SESSION_REOPEN": 0,
        "SQUEEZE_ON": 0,
        "REALIZED_VOL_PERCENTILE": 58.0,
        "ATR_PERCENTILE": 57.0,
        "TREND_FOLLOW_THROUGH": 1.0,
        "DIST_SESSION_VWAP_PCT": 0.0,
        "OPENING_RANGE_BREAK": 0,
        "SWEEP_RECLAIM_SIGNAL": 0,
        "SWEEP_RECLAIM_QUALITY": 0.0,
        "IS_LONDON_OPEN": 1,
        "IS_NEW_YORK_OPEN": 0,
        "IS_COMEX_OPEN": 0,
        "IS_FIX_WINDOW": 0,
        "IS_LONDON_NY_OVERLAP": 0,
        "DIST_PRIOR_DAY_HIGH_PCT": 0.18,
        "DIST_PRIOR_DAY_LOW_PCT": 0.22,
        "DIST_PRIOR_WEEK_HIGH_PCT": 0.30,
        "DIST_PRIOR_WEEK_LOW_PCT": 0.32,
        "MACD_HIST": 0.0,
        "MACD_HIST_SLOPE": 0.0,
        "RSI_BULLISH_DIVERGENCE": 0,
        "RSI_BEARISH_DIVERGENCE": 0,
        "VOLUME_ZSCORE": 0.0,
        "VOLUME_SPIKE": 0,
        "EVENT_ACTIVE": 0,
        "MINUTES_TO_NEXT_EVENT": 180.0,
        "WICKINESS": 0.18,
    }


def _make_bearish_prediction_input():
    ta_data = _base_prediction_input()
    ta_data["current_price"] = 100.4
    ta_data["ema_trend"] = "Bearish"
    ta_data["execution_trend"] = "Bearish"
    ta_data["multi_timeframe"] = {
        "m15_trend": "Bearish",
        "h1_trend": "Bearish",
        "h4_trend": "Bearish",
        "alignment_score": -3,
        "alignment_label": "Strong Bearish Alignment",
        "sources": {
            "m15": "derived_proxy_or_base",
            "h1": "derived_resample",
            "h4": "derived_resample",
        },
    }
    ta_data["price_action"] = {
        "structure": "Bearish Drift",
        "latest_candle_pattern": "None",
    }
    ta_data["volume_analysis"] = {
        "cmf_14": -0.02,
        "obv_trend": "Falling",
        "overall_volume_signal": "Neutral",
    }
    ta_data["momentum_features"] = {
        "macdLine": -0.02,
        "macdSignal": -0.01,
        "macdHistogram": -0.01,
        "macdHistogramSlope": 0.0,
        "volumeZScore": 0.2,
        "volumeSpike": 0,
    }
    ta_data["event_regime"] = {
        "breakout_bias": "Neutral",
        "cross_asset_bias": "Neutral",
        "big_move_risk": 36.0,
        "expansion_probability_30m": 54.0,
        "expansion_probability_60m": 59.0,
        "warning_ladder": "Normal",
        "event_regime": "normal",
        "components": {},
    }
    ta_data["structure_context"] = {
        "openingRangeBreak": 0,
        "sweepReclaimSignal": 0,
        "sweepReclaimQuality": 0.0,
        "sessionVwap": 100.72,
        "distSessionVwapPct": -0.12,
        "recentSwingHigh": 101.25,
        "recentSwingLow": 100.48,
        "pivotPoint": 100.80,
        "pivotResistance1": 101.45,
        "pivotSupport1": 100.62,
        "pivotResistance2": 101.88,
        "pivotSupport2": 100.24,
    }
    return ta_data


def _action_rank(action_state):
    return {
        "WAIT": 0,
        "SETUP_LONG": 1,
        "SETUP_SHORT": 1,
        "LONG_ACTIVE": 2,
        "SHORT_ACTIVE": 2,
        "EXIT_RISK": 1,
    }.get(str(action_state), -1)


class SignalCleanupRegressionTests(unittest.TestCase):
    def test_feature_frame_excludes_removed_columns_but_keeps_live_dependencies(self):
        features = prepare_historical_features(_make_feature_input_frame())
        column_names = set(features.columns)

        self.assertNotIn("RSI_14", column_names)
        self.assertNotIn("RSI_DIVERGENCE_STRENGTH", column_names)
        self.assertFalse(any("FVG" in name for name in column_names))
        self.assertFalse(any(name.startswith("RANGE_ZONE") for name in column_names))
        self.assertFalse(any("ROUND_NUMBER" in name for name in column_names))

        self.assertIn("RSI_BULLISH_DIVERGENCE", column_names)
        self.assertIn("RSI_BEARISH_DIVERGENCE", column_names)
        self.assertIn("PIVOT_POINT", column_names)
        self.assertIn("PIVOT_R1", column_names)
        self.assertIn("PIVOT_S1", column_names)

    def test_payload_contract_is_compact_pivot_and_microstructure_only(self):
        _, row = _prepared_feature_row()
        payload = build_ta_payload_from_row(row)

        self.assertEqual(set(payload["support_resistance"].keys()), {"pivot_levels"})
        self.assertEqual(
            set(payload["support_resistance"]["pivot_levels"].keys()),
            {"pp", "r1", "s1", "r2", "s2"},
        )
        self.assertEqual(set(payload["structure_context"].keys()), ALLOWED_STRUCTURE_CONTEXT_KEYS)

        for removed_key in (
            "nearby_supports",
            "nearby_resistances",
            "round_numbers",
            "active_fvgs",
            "range_zone",
        ):
            self.assertNotIn(removed_key, payload["support_resistance"])

        for removed_key in (
            "roundNumberSupport",
            "roundNumberResistance",
            "bullishFvg",
            "bearishFvg",
            "rangeZone",
            "rangeZoneActive",
            "rangeZonePosition",
            "rangeZoneBreak",
        ):
            self.assertNotIn(removed_key, payload["structure_context"])

    def test_normalize_ta_payload_schema_strips_legacy_rsi_and_support_shape(self):
        normalized = _normalize_ta_payload_schema(_legacy_ta_payload())

        self.assertNotIn("rsi_14", normalized)
        self.assertNotIn("rsi_signal", normalized)
        self.assertEqual(set(normalized["support_resistance"].keys()), {"pivot_levels"})
        self.assertEqual(set(normalized["structure_context"].keys()), ALLOWED_STRUCTURE_CONTEXT_KEYS)
        self.assertNotIn("rsiBullishDivergence", normalized["momentum_features"])
        self.assertNotIn("rsiBearishDivergence", normalized["momentum_features"])
        self.assertNotIn("rsiDivergenceStrength", normalized["momentum_features"])

    def test_sanitize_client_prediction_payload_strips_legacy_and_internal_fields(self):
        payload = {
            "TechnicalAnalysis": _legacy_ta_payload(),
            "TradeGuidance": {"summary": "internal"},
            "MarketState": {"regime": "trend"},
            "RR200Signal": {"status": "ready"},
            "DecisionStatus": {"text": "internal"},
            "ExecutionPermission": {"status": "entry_allowed"},
            "TradePlaybook": {"stage": "enter"},
            "ForecastState": {"regimeBucket": "active_momentum"},
            "ExecutionState": {"status": "enter"},
            "DashboardAction": {"label": "Buy"},
        }

        sanitized = _sanitize_client_prediction_payload(payload)

        for removed_key in (
            "TradeGuidance",
            "MarketState",
            "RR200Signal",
            "DecisionStatus",
            "ExecutionPermission",
            "TradePlaybook",
            "ForecastState",
            "ExecutionState",
            "DashboardAction",
        ):
            self.assertNotIn(removed_key, sanitized)

        ta_data = sanitized["TechnicalAnalysis"]
        self.assertNotIn("rsi_14", ta_data)
        self.assertNotIn("rsi_signal", ta_data)
        self.assertNotIn("_prediction_market_state", ta_data)
        self.assertEqual(set(ta_data["support_resistance"].keys()), {"pivot_levels"})
        self.assertEqual(set(ta_data["structure_context"].keys()), ALLOWED_STRUCTURE_CONTEXT_KEYS)

    def test_prediction_ignores_removed_legacy_signal_fields(self):
        baseline_input = _base_prediction_input()
        legacy_input = copy.deepcopy(baseline_input)
        legacy_input["rsi_14"] = 4.0
        legacy_input["rsi_signal"] = "Oversold (Bullish bias)"
        legacy_input["support_resistance"] = {
            "pivot_levels": baseline_input["support_resistance"]["pivot_levels"],
            "reaction": "Bearish Resistance Rejection",
            "nearby_supports": [{"label": "Legacy S", "family": "range", "distance_pct": 0.01}],
            "nearby_resistances": [{"label": "Legacy R", "family": "fvg", "distance_pct": 0.01}],
            "round_numbers": {"support": 100.0, "resistance": 102.0},
            "active_fvgs": {"bullish": {"low": 100.2, "high": 100.4}},
            "range_zone": {"low": 100.1, "high": 100.9},
        }
        legacy_input["structure_context"] = dict(baseline_input["structure_context"])
        legacy_input["structure_context"].update(
            {
                "roundNumberSupport": 100.0,
                "roundNumberResistance": 102.0,
                "bullishFvg": {"low": 100.2, "high": 100.4},
                "bearishFvg": {"low": 101.6, "high": 101.8},
                "rangeZone": {"low": 100.1, "high": 100.9},
                "rangeZoneActive": True,
                "rangeZonePosition": 0.15,
                "rangeZoneBreak": 1,
            }
        )
        legacy_input["momentum_features"] = dict(baseline_input["momentum_features"])
        legacy_input["momentum_features"].update(
            {
                "rsiBullishDivergence": 1,
                "rsiBearishDivergence": 0,
                "rsiDivergenceStrength": 0.9,
            }
        )

        baseline = compute_prediction_from_ta(baseline_input)
        legacy = compute_prediction_from_ta(legacy_input)

        self.assertEqual(legacy["verdict"], baseline["verdict"])
        self.assertEqual(legacy["directionalBias"], baseline["directionalBias"])
        self.assertEqual(legacy["actionState"], baseline["actionState"])
        self.assertEqual(legacy["action"], baseline["action"])
        self.assertEqual(legacy["setupScore"], baseline["setupScore"])
        self.assertEqual(legacy["triggerScore"], baseline["triggerScore"])

    def test_opening_range_break_improves_bullish_execution_readiness(self):
        baseline_input = _base_prediction_input()
        orb_input = copy.deepcopy(baseline_input)
        orb_input["price_action"] = {
            "structure": "Bullish Breakout",
            "latest_candle_pattern": "None",
        }
        orb_input["structure_context"] = dict(baseline_input["structure_context"])
        orb_input["structure_context"]["openingRangeBreak"] = 1
        orb_input["momentum_features"] = dict(baseline_input["momentum_features"])
        orb_input["momentum_features"].update(
            {
                "macdHistogram": 0.05,
                "macdHistogramSlope": 0.012,
                "volumeZScore": 2.0,
                "volumeSpike": 1,
            }
        )
        orb_input["volume_analysis"] = dict(baseline_input["volume_analysis"])
        orb_input["volume_analysis"]["overall_volume_signal"] = "Strong Buying Pressure (Accumulation)"
        orb_input["event_regime"] = {
            "breakout_bias": "Bullish",
            "cross_asset_bias": "Bullish",
            "big_move_risk": 68.0,
            "expansion_probability_30m": 72.0,
            "expansion_probability_60m": 81.0,
            "warning_ladder": "High Breakout Risk",
            "event_regime": "breakout_watch",
            "components": {},
        }

        baseline = compute_prediction_from_ta(baseline_input)
        orb = compute_prediction_from_ta(orb_input)

        self.assertEqual(orb["verdict"], "Bullish")
        self.assertGreater(orb["triggerScore"], baseline["triggerScore"])
        self.assertGreaterEqual(_action_rank(orb["actionState"]), _action_rank(baseline["actionState"]))

    def test_sweep_reclaim_improves_bullish_trigger_score(self):
        baseline_input = _base_prediction_input()
        sweep_input = copy.deepcopy(baseline_input)
        sweep_input["structure_context"] = dict(baseline_input["structure_context"])
        sweep_input["structure_context"].update(
            {
                "sweepReclaimSignal": 1,
                "sweepReclaimQuality": 1.0,
            }
        )

        baseline = compute_prediction_from_ta(baseline_input)
        sweep = compute_prediction_from_ta(sweep_input)

        self.assertEqual(sweep["verdict"], "Bullish")
        self.assertGreater(sweep["triggerScore"], baseline["triggerScore"])

    def test_bearish_opening_range_break_improves_execution_readiness(self):
        baseline_input = _make_bearish_prediction_input()
        orb_input = copy.deepcopy(baseline_input)
        orb_input["price_action"] = {
            "structure": "Bearish Breakdown",
            "latest_candle_pattern": "None",
        }
        orb_input["structure_context"] = dict(baseline_input["structure_context"])
        orb_input["structure_context"]["openingRangeBreak"] = -1
        orb_input["momentum_features"] = dict(baseline_input["momentum_features"])
        orb_input["momentum_features"].update(
            {
                "macdHistogram": -0.05,
                "macdHistogramSlope": -0.012,
                "volumeZScore": 2.0,
                "volumeSpike": 1,
            }
        )
        orb_input["volume_analysis"] = dict(baseline_input["volume_analysis"])
        orb_input["volume_analysis"]["overall_volume_signal"] = "Strong Selling Pressure (Distribution)"
        orb_input["event_regime"] = {
            "breakout_bias": "Bearish",
            "cross_asset_bias": "Bearish",
            "big_move_risk": 68.0,
            "expansion_probability_30m": 72.0,
            "expansion_probability_60m": 81.0,
            "warning_ladder": "High Breakout Risk",
            "event_regime": "breakout_watch",
            "components": {},
        }

        baseline = compute_prediction_from_ta(baseline_input)
        orb = compute_prediction_from_ta(orb_input)

        self.assertEqual(orb["verdict"], "Bearish")
        self.assertGreater(orb["triggerScore"], baseline["triggerScore"])
        self.assertGreaterEqual(_action_rank(orb["actionState"]), _action_rank(baseline["actionState"]))

    def test_bearish_sweep_reclaim_improves_trigger_score(self):
        baseline_input = _make_bearish_prediction_input()
        sweep_input = copy.deepcopy(baseline_input)
        sweep_input["structure_context"] = dict(baseline_input["structure_context"])
        sweep_input["structure_context"].update(
            {
                "sweepReclaimSignal": -1,
                "sweepReclaimQuality": 1.0,
            }
        )

        baseline = compute_prediction_from_ta(baseline_input)
        sweep = compute_prediction_from_ta(sweep_input)

        self.assertEqual(sweep["verdict"], "Bearish")
        self.assertGreater(sweep["triggerScore"], baseline["triggerScore"])

    def test_event_risk_blocks_execution_even_when_direction_is_aligned(self):
        event_input = _base_prediction_input()
        event_input["price_action"] = {
            "structure": "Bullish Breakout",
            "latest_candle_pattern": "None",
        }
        event_input["structure_context"] = dict(event_input["structure_context"])
        event_input["structure_context"]["openingRangeBreak"] = 1
        event_input["event_regime"] = {
            "breakout_bias": "Bullish",
            "cross_asset_bias": "Bullish",
            "big_move_risk": 82.0,
            "expansion_probability_30m": 84.0,
            "expansion_probability_60m": 90.0,
            "warning_ladder": "Active Momentum Event",
            "event_regime": "range_expansion",
            "components": {},
        }
        event_input["event_risk"] = {
            "active": True,
        }

        result = compute_prediction_from_ta(event_input)
        combined_reasons = " ".join(
            part for part in [result["noTradeReason"], result["noTradeReasonSoft"], result["noTradeReasonHard"]] if part
        )

        self.assertIn("Major macro event window is active.", combined_reasons)
        self.assertEqual(result["actionState"], "WAIT")
        self.assertEqual(result["action"], "hold")

    def test_event_regime_live_feature_hits_cover_remaining_signal_inputs(self):
        cases = [
            ("opening_range_break", {"OPENING_RANGE_BREAK": 1}, "opening_range_break"),
            (
                "sweep_reclaim",
                {"SWEEP_RECLAIM_SIGNAL": 1, "SWEEP_RECLAIM_QUALITY": 1.0},
                "sweep_reclaim",
            ),
            ("session_vwap_dislocation", {"DIST_SESSION_VWAP_PCT": 0.35}, "session_vwap_dislocation"),
            ("rsi_bullish_divergence", {"RSI_BULLISH_DIVERGENCE": 1}, "rsi_bullish_divergence"),
            ("volume_spike", {"VOLUME_SPIKE": 1, "VOLUME_ZSCORE": 2.1}, "volume_spike"),
        ]

        for case_name, updates, expected_flag in cases:
            with self.subTest(case=case_name):
                row = _base_event_regime_row()
                row.update(updates)
                snapshot = compute_event_regime_snapshot(
                    row,
                    trend="Bullish",
                    alignment_label="Strong Bullish Alignment",
                    market_structure="Bullish Drift",
                    candle_pattern="None",
                    event_risk={"active": False, "minutes_to_next_release": 180.0},
                    cross_asset_context={},
                )
                self.assertTrue(snapshot["feature_hits"][expected_flag])

                if case_name == "opening_range_break":
                    self.assertEqual(snapshot["breakout_bias"], "Bullish")
                if case_name == "rsi_bullish_divergence":
                    self.assertGreaterEqual(snapshot["components"]["momentum_signal_score"], 4.0)

    def test_predict_route_returns_sanitized_payload(self):
        payload = {
            "status": "success",
            "verdict": "Bullish",
            "TradeGuidance": {"summary": "internal"},
            "MarketState": {"regime": "trend"},
            "DecisionStatus": {"text": "internal"},
            "ExecutionPermission": {"status": "entry_allowed"},
            "TradePlaybook": {"stage": "enter"},
            "ForecastState": {"regimeBucket": "active_momentum"},
            "ExecutionState": {"status": "enter"},
            "DashboardAction": {"label": "Buy"},
            "TechnicalAnalysis": _legacy_ta_payload(),
        }

        with patch.object(app_module, "_build_prediction_response", return_value=(payload, 200)):
            client = app_module.app.test_client()
            response = client.get("/api/predict")

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertNotIn("TradeGuidance", body)
        self.assertNotIn("MarketState", body)
        self.assertNotIn("DecisionStatus", body)
        self.assertNotIn("ExecutionPermission", body)
        self.assertNotIn("TradePlaybook", body)
        self.assertNotIn("ForecastState", body)
        self.assertNotIn("ExecutionState", body)
        self.assertNotIn("DashboardAction", body)
        self.assertEqual(set(body["TechnicalAnalysis"]["support_resistance"].keys()), {"pivot_levels"})
        self.assertEqual(set(body["TechnicalAnalysis"]["structure_context"].keys()), ALLOWED_STRUCTURE_CONTEXT_KEYS)
        self.assertNotIn("rsi_14", body["TechnicalAnalysis"])
        self.assertNotIn("rsi_signal", body["TechnicalAnalysis"])

    def test_health_route_returns_runtime_and_cache_status(self):
        with (
            patch.object(app_module, "MONITOR_INTERVAL_SECONDS", 17),
            patch.object(app_module, "NOTIFY_MIN_INTERVAL_SECONDS", 29),
            patch.object(app_module.socketio, "async_mode", "threading", create=True),
            patch.object(app_module, "_has_background_alert_channels", return_value=True),
            patch.object(app_module, "_load_json_file", return_value={"wins": 3}),
            patch.object(app_module.predict_gold, "LAST_TA_REFRESH_TS", 1700000001, create=True),
            patch.object(app_module.predict_gold, "TECHNICAL_ANALYSIS_CACHE_SECONDS", 21, create=True),
            patch.object(app_module.predict_gold, "LAST_SUCCESSFUL_TA", {"cached": True}, create=True),
            patch.object(app_module.predict_gold, "LAST_CROSS_ASSET_TS", 1700000002, create=True),
            patch.object(app_module.predict_gold, "CROSS_ASSET_CACHE_SECONDS", 91, create=True),
            patch.object(app_module.predict_gold, "LAST_CROSS_ASSET_CONTEXT", {"bias": "Neutral"}, create=True),
            patch.dict(getattr(app_module, "_monitor_state"), {"started": True, "clients": 2}, clear=False),
        ):
            client = app_module.app.test_client()
            response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(
            body["runtime"],
            {
                "socket_async_mode": "threading",
                "monitor_interval_seconds": 17,
                "notify_min_interval_seconds": 29,
                "has_background_alert_channels": True,
            },
        )
        self.assertEqual(body["monitor"], {"started": True, "clients": 2})
        self.assertEqual(
            body["prediction_cache"],
            {
                "last_refresh_ts": 1700000001,
                "cache_seconds": 21,
                "has_cached_snapshot": True,
            },
        )
        self.assertEqual(
            body["cross_asset_cache"],
            {
                "last_refresh_ts": 1700000002,
                "cache_seconds": 91,
                "has_cached_snapshot": True,
            },
        )
        self.assertEqual(body["live_signal_summary"], {"wins": 3})

    def test_health_route_returns_error_payload_when_status_build_fails(self):
        with patch.object(app_module, "_load_json_file", side_effect=RuntimeError("health boom")):
            client = app_module.app.test_client()
            response = client.get("/api/health")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"status": "error", "message": "health boom"})

    def test_predict_route_returns_error_payload_when_builder_fails(self):
        with patch.object(app_module, "_build_prediction_response", side_effect=RuntimeError("boom")):
            client = app_module.app.test_client()
            response = client.get("/api/predict")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"status": "error", "message": "boom"})


if __name__ == "__main__":
    unittest.main()