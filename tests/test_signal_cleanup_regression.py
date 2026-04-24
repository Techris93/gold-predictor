import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

import app as app_module
from scripts import update_event_risk_windows
from tools import predict_gold
from tools.event_regime import compute_event_regime_snapshot
from tools.predict_gold import _normalize_ta_payload_schema
from tools.signal_engine import (
    _build_session_context_from_datetime,
    build_ta_payload_from_row,
)


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
    current_session = _build_session_context_from_datetime(
        pd.Timestamp("2026-04-21T12:00:00Z")
    )
    _apply_current_session(payload, current_session)
    payload["support_resistance"]["nearest_support"] = {
        "label": "Previous Day Low",
        "price": 4833.88,
    }
    payload["support_resistance"]["nearest_resistance"] = {
        "label": "Previous Day High",
        "price": 4834.21,
    }
    return payload


def _apply_current_session(payload, current_session):
    payload["session_context"]["current_session"] = current_session
    payload["session_context"]["currentLabel"] = current_session["label"]
    payload["session_context"]["currentTimestampUtc"] = current_session["timestampUtc"]
    payload["session_context"]["currentTimeDisplayUtc"] = current_session["timeDisplayUtc"]
    payload["session_context"]["marketStatus"] = current_session["marketStatus"]
    payload["session_context"]["marketStatusLabel"] = current_session["marketStatusLabel"]
    payload["session_context"]["isMarketClosed"] = current_session["isMarketClosed"]
    payload["session_context"]["isWeekendClosed"] = current_session["isWeekendClosed"]
    payload["session_context"]["isDailyRolloverPause"] = current_session["isDailyRolloverPause"]
    payload["session_context"]["isHolidayClosed"] = current_session["isHolidayClosed"]
    payload["session_context"]["isHolidaySchedule"] = current_session["isHolidaySchedule"]
    payload["session_context"]["closedReason"] = current_session["closedReason"]
    payload["session_context"]["nextOpenUtc"] = current_session["nextOpenUtc"]
    payload["session_context"]["nextOpenTimeDisplayUtc"] = current_session["nextOpenTimeDisplayUtc"]
    payload["session_context"]["lastCloseUtc"] = current_session["lastCloseUtc"]
    payload["session_context"]["lastCloseTimeDisplayUtc"] = current_session["lastCloseTimeDisplayUtc"]
    payload["session_context"]["weeklyScheduleUtc"] = current_session["weeklyScheduleUtc"]
    payload["session_context"]["holidayName"] = current_session["holidayName"]
    payload["session_context"]["holidayScheduleNote"] = current_session["holidayScheduleNote"]
    payload["session_context"]["holidayScheduleSource"] = current_session["holidayScheduleSource"]
    return payload


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class DashboardPayloadContractTests(unittest.TestCase):
    def test_session_context_marks_weekend_closed_and_next_open(self):
        weekend_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-04-25T12:00:00Z")
        )

        self.assertTrue(weekend_session["isMarketClosed"])
        self.assertTrue(weekend_session["isWeekendClosed"])
        self.assertEqual(weekend_session["label"], "Weekend Closed")
        self.assertEqual(weekend_session["nextOpenTimeDisplayUtc"], "Sun 22:00 UTC")
        self.assertEqual(weekend_session["marketStatus"], "weekend_closed")

    def test_session_context_marks_rollover_pause_and_next_open(self):
        rollover_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-04-21T21:30:00Z")
        )

        self.assertTrue(rollover_session["isMarketClosed"])
        self.assertTrue(rollover_session["isDailyRolloverPause"])
        self.assertEqual(rollover_session["label"], "Daily Rollover Pause")
        self.assertEqual(rollover_session["nextOpenTimeDisplayUtc"], "Tue 22:00 UTC")
        self.assertEqual(rollover_session["marketStatus"], "rollover_pause")

    def test_session_context_marks_good_friday_holiday_close(self):
        holiday_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-04-03T12:00:00Z")
        )

        self.assertTrue(holiday_session["isMarketClosed"])
        self.assertTrue(holiday_session["isHolidayClosed"])
        self.assertTrue(holiday_session["isHolidaySchedule"])
        self.assertEqual(holiday_session["holidayName"], "Good Friday")
        self.assertEqual(holiday_session["marketStatus"], "holiday_closed")
        self.assertEqual(holiday_session["nextOpenTimeDisplayUtc"], "Sun 22:00 UTC")

    def test_session_context_marks_memorial_day_holiday_schedule(self):
        holiday_schedule_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-05-25T14:00:00Z")
        )

        self.assertFalse(holiday_schedule_session["isMarketClosed"])
        self.assertTrue(holiday_schedule_session["isHolidaySchedule"])
        self.assertEqual(holiday_schedule_session["holidayName"], "Memorial Day")
        self.assertEqual(holiday_schedule_session["marketStatus"], "holiday_schedule")

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

    def test_decision_status_stands_aside_when_market_is_weekend_closed(self):
        ta = _sample_ta_payload()
        closed_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-04-25T12:00:00Z")
        )
        _apply_current_session(ta, closed_session)

        decision = app_module._evaluate_decision_status(
            verdict="Bearish",
            confidence=66,
            ta_data=ta,
            trade_guidance={
                "summary": "Short setup confirmed with acceptable tradeability.",
                "buyLevel": "Weak",
                "sellLevel": "Strong",
                "exitLevel": "Low",
            },
            execution_state={},
        )

        self.assertEqual(decision["status"], "wait")
        self.assertTrue(decision["marketClosed"])
        self.assertIn("closed for the weekend", decision["text"])
        self.assertIn("Sun 22:00 UTC", decision["text"])

    def test_decision_status_stands_aside_when_market_is_rollover_paused(self):
        ta = _sample_ta_payload()
        rollover_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-04-21T21:30:00Z")
        )
        _apply_current_session(ta, rollover_session)

        decision = app_module._evaluate_decision_status(
            verdict="Bearish",
            confidence=66,
            ta_data=ta,
            trade_guidance={
                "summary": "Short setup confirmed with acceptable tradeability.",
                "buyLevel": "Weak",
                "sellLevel": "Strong",
                "exitLevel": "Low",
            },
            execution_state={},
        )

        self.assertEqual(decision["status"], "wait")
        self.assertTrue(decision["marketClosed"])
        self.assertIn("daily rollover pause", decision["text"].lower())
        self.assertIn("Tue 22:00 UTC", decision["text"])

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
        self.assertEqual(payload["structure_context"]["openingRangeBreak"], 1)

    def test_sweep_reclaim_uses_last_closed_15m_bar(self):
        candles = []
        for idx in range(60):
            base = 100.0 + (idx * 0.08)
            candles.append(
                {
                    "Open": base,
                    "High": base + 0.45,
                    "Low": base - 0.35,
                    "Close": base + 0.18,
                    "Volume": 1000 + idx,
                }
            )
        frame = pd.DataFrame(
            candles,
            index=pd.date_range("2026-04-20T09:00:00Z", periods=len(candles), freq="15min", tz="UTC"),
        )
        enriched = pd.DataFrame(
            {
                "PA_STRUCTURE": ["Bearish Drift"] * len(frame),
                "CANDLE_PATTERN": ["None"] * len(frame),
                "OPENING_RANGE_BREAK": [0] * (len(frame) - 1) + [1],
                "SWEEP_RECLAIM_SIGNAL": [0] * (len(frame) - 2) + [1, 0],
                "SWEEP_RECLAIM_QUALITY": [0.0] * (len(frame) - 2) + [0.86, 0.0],
            },
            index=frame.index,
        )

        def _fake_build_ta_payload_from_row(row, *_args, **_kwargs):
            return {
                "structure_context": {
                    "openingRangeBreak": int(row.get("OPENING_RANGE_BREAK") or 0),
                    "sweepReclaimSignal": int(row.get("SWEEP_RECLAIM_SIGNAL") or 0),
                    "sweepReclaimQuality": float(row.get("SWEEP_RECLAIM_QUALITY") or 0.0),
                }
            }

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
        ), patch.object(
            predict_gold,
            "classify_price_action",
            return_value=("Bearish Drift", "None"),
        ), patch.object(
            predict_gold,
            "prepare_historical_features",
            return_value=enriched,
        ), patch.object(
            predict_gold,
            "build_ta_payload_from_row",
            side_effect=_fake_build_ta_payload_from_row,
        ):
            payload = predict_gold._build_technical_analysis_from_frame(
                frame,
                td_symbol="XAU/USD",
                now_ts=1713600000,
                data_source="unit-test",
            )

        self.assertEqual(payload["price_action"]["basis"], "last_closed_15m_bar")
        self.assertEqual(payload["structure_context"]["sweepReclaimSignal"], 1)
        self.assertAlmostEqual(payload["structure_context"]["sweepReclaimQuality"], 0.86)
        self.assertEqual(payload["structure_context"]["openingRangeBreak"], 0)

    def test_signal_notification_omits_execution_and_decision_text(self):
        notification = app_module._build_signal_notification(
            {
                "market_structure": {
                    "previous": "Consolidating",
                    "current": "Higher Highs / Higher Lows (Bullish Structure)",
                },
                "micro_orb_state": {
                    "previous": -1,
                    "current": 1,
                },
            },
            {},
            "Higher Highs / Higher Lows (Bullish Structure)",
            ta_data=_sample_ta_payload(),
            payload={
                "ExecutionState": {"title": "Stand Aside"},
                "DecisionStatus": {
                    "text": "Stand aside. Checklist does not support a clean trade yet."
                },
            },
        )

        self.assertIn("Market Structure:", notification["body"])
        self.assertIn("Microstructure Change: ORB Bearish -> Bullish", notification["body"])
        self.assertNotIn("Execution State:", notification["body"])
        self.assertNotIn("Decision:", notification["body"])
        self.assertNotIn("Checklist does not support a clean trade yet.", notification["body"])

    def test_notification_filter_excludes_execution_state(self):
        changes = {
            "execution_state": {
                "previous": "Watchlist Only",
                "current": "No Trade",
            },
            "market_structure": {
                "previous": "Bearish Drift",
                "current": "Higher Highs / Higher Lows (Bullish Structure)",
            },
        }

        filtered = app_module._filter_notification_changes(changes)

        self.assertNotIn("execution_state", filtered)
        self.assertIn("market_structure", filtered)

    def test_notification_filter_allows_execution_quality_signal_only(self):
        filtered = app_module._filter_notification_changes(
            {
                "execution_quality_signal": {
                    "previous": "No Trade",
                    "current": "A VWAP / ORB Continuation Long >=1.5R",
                },
                "verdict": {"previous": "Neutral", "current": "Bullish"},
                "confidence_bucket": {"previous": "Low", "current": "High"},
            }
        )

        self.assertEqual(
            filtered,
            {
                "execution_quality_signal": {
                    "previous": "No Trade",
                    "current": "A VWAP / ORB Continuation Long >=1.5R",
                }
            },
        )

    def test_execution_quality_signal_body_uses_price_action_title(self):
        notification = app_module._build_signal_notification(
            {
                "execution_quality_signal": {
                    "previous": "Watchlist Long Watchlist Reclaim >=1.5R",
                    "current": "No Trade blocked: Recent Swing High is too close for a clean long target.",
                }
            },
            {},
            "Bullish Drift",
            ta_data=_sample_ta_payload(),
        )

        self.assertEqual(notification["title"], "XAUUSD Execution Quality Changed")
        self.assertIn(
            "Execution Quality: Watchlist Long Watchlist Reclaim >=1.5R -> No Trade blocked",
            notification["body"],
        )
        self.assertIn("Bar Session / Microstructure:", notification["body"])

    def test_notification_filter_suppresses_non_price_action_categories(self):
        changes = {
            "warning_ladder": {"previous": "Normal", "current": "High Breakout Risk"},
            "event_regime": {"previous": "normal", "current": "event_risk"},
            "breakout_bias": {"previous": "Neutral", "current": "Bearish"},
            "verdict": {"previous": "Neutral", "current": "Bearish"},
            "confidence_bucket": {"previous": "Guarded", "current": "High"},
            "execution_state": {"previous": "Watchlist Only", "current": "No Trade"},
            "market_structure": {
                "previous": "Bearish Drift",
                "current": "Higher Highs / Higher Lows (Bullish Structure)",
            },
            "micro_vwap_delta_pct": {"previous": 0.31, "current": 0.56},
        }

        filtered = app_module._filter_notification_changes(changes)

        self.assertEqual(
            set(filtered.keys()),
            {"market_structure", "micro_vwap_delta_pct"},
        )

    def test_execution_state_change_cannot_restore_old_alert_title(self):
        notification = app_module._build_signal_notification(
            {
                "execution_state": {
                    "previous": "Watchlist Only",
                    "current": "No Trade",
                },
            },
            {},
            "Bearish Drift",
            ta_data=_sample_ta_payload(),
            payload={},
        )

        self.assertEqual(notification["title"], "XAUUSD Signal Changed")
        self.assertNotEqual(notification["title"], "XAUUSD Execution State Changed")

    def test_suppressed_notification_categories_cannot_restore_old_titles(self):
        title = app_module._notification_title_for_changes(
            {
                "warning_ladder": {
                    "previous": "Normal",
                    "current": "High Breakout Risk",
                },
                "verdict": {
                    "previous": "Neutral",
                    "current": "Bearish",
                },
                "confidence_bucket": {
                    "previous": "Guarded",
                    "current": "High",
                },
            }
        )

        self.assertEqual(title, "XAUUSD Signal Changed")
        self.assertNotEqual(title, "XAUUSD Trade Context Changed")
        self.assertNotEqual(title, "XAUUSD Verdict Changed")
        self.assertNotEqual(title, "XAUUSD Confidence Changed")
        self.assertNotEqual(title, "XAUUSD State Changed")

    def test_notification_keeps_material_vwap_move_without_bias_flip(self):
        notification = app_module._build_signal_notification(
            {
                "micro_vwap_delta_pct": {
                    "previous": 0.31,
                    "current": 0.56,
                },
            },
            {},
            "Bearish Drift",
            ta_data=_sample_ta_payload(),
            payload={},
        )

        self.assertEqual(notification["title"], "XAUUSD Microstructure Changed")
        self.assertIn(
            "VWAP Bullish strengthened (0.31% -> 0.56%)",
            notification["body"],
        )

    def test_notification_drops_small_vwap_move_without_bias_flip(self):
        filtered = app_module._filter_notification_changes(
            {
                "micro_vwap_delta_pct": {
                    "previous": 0.31,
                    "current": 0.33,
                },
            }
        )

        self.assertEqual(filtered, {})

    def test_notification_fingerprint_ignores_body_clock_drift(self):
        changes = {
            "execution_state": {
                "previous": "Watchlist Only",
                "current": "No Trade",
            },
        }

        first = app_module._build_signal_notification(
            changes,
            {},
            "Bearish Drift",
            ta_data=_sample_ta_payload(),
            payload={},
        )
        second = app_module._build_signal_notification(
            changes,
            {},
            "Bearish Drift",
            ta_data=_sample_ta_payload(),
            payload={},
        )

        self.assertEqual(
            first["notification_fingerprint"],
            second["notification_fingerprint"],
        )
        self.assertEqual(first["notification_tag"], second["notification_tag"])

    def test_duplicate_alert_suppression_uses_fingerprint_and_class_cooldown(self):
        fingerprint = app_module._notification_fingerprint(
            {
                "warning_ladder": {
                    "previous": "Expansion Watch",
                    "current": "High Breakout Risk",
                },
            }
        )
        alert_state = {
            "last_alert_ts": 1000,
            "last_alert_fingerprint": fingerprint,
            "last_context_alert_ts": 1000,
            "last_context_alert_fingerprint": fingerprint,
        }

        self.assertTrue(
            app_module._should_suppress_duplicate_alert(
                alert_state,
                fingerprint,
                "context",
                1000 + min(app_module.ALERT_COOLDOWN_SECONDS, app_module.ALERT_CONTEXT_COOLDOWN_SECONDS) - 1,
            )
        )
        self.assertFalse(
            app_module._should_suppress_duplicate_alert(
                alert_state,
                "different-fingerprint",
                "context",
                1001,
            )
        )

    def test_stabilize_vwap_bias_label_holds_neutral_through_boundary_noise(self):
        self.assertEqual(
            app_module._stabilize_vwap_bias_label(-0.10, "Neutral"),
            "Neutral",
        )
        self.assertEqual(
            app_module._stabilize_vwap_bias_label(-0.101, "Neutral"),
            "Neutral",
        )

    def test_stabilize_vwap_bias_label_breaks_only_after_real_move(self):
        self.assertEqual(
            app_module._stabilize_vwap_bias_label(-0.121, "Neutral"),
            "Mild Bearish",
        )
        self.assertEqual(
            app_module._stabilize_vwap_bias_label(-0.281, "Bearish"),
            "Bearish",
        )

    def test_extract_indicator_snapshot_uses_stable_vwap_bias_label(self):
        payload = {
            "TechnicalAnalysis": {
                "price_action": {"structure": "Bearish Drift"},
                "structure_context": {"distSessionVwapPct": -0.10},
            },
            "RegimeState": {},
            "ExecutionState": {},
            "verdict": "Neutral",
            "confidence": 58,
        }

        snapshot = app_module._extract_indicator_snapshot(
            payload,
            previous_snapshot={"micro_vwap_bias": "Neutral"},
        )

        self.assertEqual(snapshot["micro_vwap_bias"], "Neutral")
        self.assertEqual(snapshot["micro_vwap_band"], "-0.1% to +0.0%")

    def test_bar_session_formatter_reports_weekend_closure(self):
        ta = _sample_ta_payload()
        closed_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-04-25T12:00:00Z")
        )
        _apply_current_session(ta, closed_session)

        summary = app_module._format_bar_session_microstructure(ta)

        self.assertIn("Weekend Closed", summary)
        self.assertIn("Reopens Sun 22:00 UTC", summary)
        self.assertIn("Microstructure is frozen until reopen", summary)

    def test_bar_session_formatter_reports_rollover_pause(self):
        ta = _sample_ta_payload()
        rollover_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-04-21T21:30:00Z")
        )
        _apply_current_session(ta, rollover_session)

        summary = app_module._format_bar_session_microstructure(ta)

        self.assertIn("Daily Rollover Pause", summary)
        self.assertIn("Reopens Tue 22:00 UTC", summary)
        self.assertIn("Microstructure is frozen until reopen", summary)

    def test_bar_session_formatter_appends_holiday_schedule_when_open(self):
        ta = _sample_ta_payload()
        holiday_schedule_session = _build_session_context_from_datetime(
            pd.Timestamp("2026-05-25T14:00:00Z")
        )
        _apply_current_session(ta, holiday_schedule_session)

        summary = app_module._format_bar_session_microstructure(ta)

        self.assertIn("Holiday schedule Memorial Day", summary)
        self.assertNotIn("frozen until reopen", summary)

    def test_bar_session_formatter_collapses_matching_session_labels(self):
        summary = app_module._format_bar_session_microstructure(_sample_ta_payload())

        self.assertIn("Session London / New York Overlap · Bar 12:00 UTC · Now 12:00 UTC", summary)
        self.assertNotIn(
            "Bar London / New York Overlap 12:00 UTC · Now London / New York Overlap 12:00 UTC",
            summary,
        )

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

    def test_event_regime_dedupes_next_event_from_near_events(self):
        snapshot = compute_event_regime_snapshot(
            _sample_row(),
            trend="Bearish",
            alignment_label="Strong Bearish Alignment",
            market_structure="Bearish Drift",
            candle_pattern="None",
            event_risk={
                "active": False,
                "minutes_to_next_release": 1909.0,
                "next_release_event": {"name": "Retail Sales"},
                "near_releases": [
                    {"name": "Retail Sales", "minutes": 1909.0},
                    {"name": "Retail Sales", "minutes": 1909.0},
                    {"name": "CPI", "minutes": 2600.0},
                ],
            },
            cross_asset_context={},
        )

        self.assertEqual(snapshot["next_event_name"], "Retail Sales")
        self.assertEqual(snapshot["near_events"], [{"name": "CPI", "minutes": 2600.0}])

    def test_event_regime_omits_later_occurrence_of_same_next_event_name(self):
        snapshot = compute_event_regime_snapshot(
            _sample_row(),
            trend="Bearish",
            alignment_label="Strong Bearish Alignment",
            market_structure="Bearish Drift",
            candle_pattern="None",
            event_risk={
                "active": False,
                "minutes_to_next_release": 131.0,
                "next_release_event": {"name": "Initial Jobless Claims"},
                "near_releases": [
                    {"name": "Initial Jobless Claims", "minutes": 131.0},
                    {"name": "Initial Jobless Claims", "minutes": 1571.0},
                ],
            },
            cross_asset_context={},
        )

        self.assertEqual(snapshot["next_event_name"], "Initial Jobless Claims")
        self.assertEqual(snapshot["near_events"], [])

    def test_event_risk_context_counts_down_from_release_timestamp(self):
        window = {
            "name": "Initial Jobless Claims",
            "start": "2026-04-23T12:10:00Z",
            "end": "2026-04-23T13:30:00Z",
            "release": "2026-04-23T12:30:00Z",
            "reason": "Synthetic test window.",
        }

        with patch.object(predict_gold, "_load_event_risk_windows", return_value=[window]):
            context = predict_gold._event_risk_context(
                int(pd.Timestamp("2026-04-23T12:04:00Z").timestamp())
            )

        self.assertEqual(context["next_release_event"]["name"], "Initial Jobless Claims")
        self.assertAlmostEqual(context["minutes_to_next_release"], 26.0, places=1)
        self.assertEqual(context["near_releases"][0]["release"], "2026-04-23T12:30:00Z")
        self.assertEqual(context["now_utc"], "2026-04-23T12:04:00+00:00")

    def test_near_macro_calendar_ignores_similarly_named_secondary_releases(self):
        release_options = [
            (469, "State Unemployment Insurance Weekly Claims Report"),
            (180, "Unemployment Insurance Weekly Claims Report"),
            (477, "Monthly State Retail Sales"),
            (92, "Selected Real Retail Sales Series"),
            (263, "Debt to Gross Domestic Product Ratios"),
            (53, "Gross Domestic Product"),
            (397, "Gross Domestic Product by County"),
            (331, "Gross Domestic Product by Industry"),
            (140, "Gross Domestic Product by State"),
        ]
        fetched_source_names = []

        def fake_fetch_release_datetimes(*, release_id, release_name, target_year, default_release_time_et):
            fetched_source_names.append(release_name)
            return [pd.Timestamp("2026-04-30T08:30:00-04:00").to_pydatetime()]

        with (
            patch.object(update_event_risk_windows, "_load_fred_release_options", return_value=release_options),
            patch.object(
                update_event_risk_windows,
                "_fetch_release_datetimes",
                side_effect=fake_fetch_release_datetimes,
            ),
        ):
            windows = update_event_risk_windows.fetch_near_macro_windows(
                now_et=pd.Timestamp("2026-04-23T12:00:00-04:00").to_pydatetime(),
                horizon_days=20,
            )

        self.assertIn("Unemployment Insurance Weekly Claims Report", fetched_source_names)
        self.assertIn("Selected Real Retail Sales Series", fetched_source_names)
        self.assertIn("Gross Domestic Product", fetched_source_names)
        self.assertNotIn("State Unemployment Insurance Weekly Claims Report", fetched_source_names)
        self.assertNotIn("Monthly State Retail Sales", fetched_source_names)
        self.assertNotIn("Debt to Gross Domestic Product Ratios", fetched_source_names)
        self.assertNotIn("Gross Domestic Product by County", fetched_source_names)
        self.assertNotIn("Gross Domestic Product by Industry", fetched_source_names)
        self.assertNotIn("Gross Domestic Product by State", fetched_source_names)
        reasons = "\n".join(str(item.get("reason") or "") for item in windows)
        self.assertNotIn("State Unemployment", reasons)
        self.assertNotIn("Monthly State Retail Sales", reasons)
        self.assertNotIn("Gross Domestic Product by", reasons)

    def test_sp_global_us_pmi_calendar_adds_flash_and_standard_us_windows(self):
        sample_calendar_html = """
        <html><body>
        <div>Calendar</div>
        <div>Upcoming</div>
        <div>2026</div>
        <div>April 23</div>
        <div>13:45 UTC S&P Global Flash US PMI</div>
        <div>May 01</div>
        <div>13:45 UTC S&P Global US Manufacturing PMI</div>
        <div>May 05</div>
        <div>13:45 UTC S&P Global US Services PMI</div>
        </body></html>
        """

        with patch.object(
            update_event_risk_windows.requests,
            "get",
            return_value=_FakeResponse(sample_calendar_html),
        ):
            windows = update_event_risk_windows.fetch_sp_global_us_pmi_windows(
                now_utc=pd.Timestamp("2026-04-23T12:00:00Z").to_pydatetime(),
                horizon_days=20,
            )

        names = {(item["name"], item["release"]) for item in windows}
        self.assertIn(
            ("S&P Global Flash US Manufacturing PMI", "2026-04-23T13:45:00Z"),
            names,
        )
        self.assertIn(
            ("S&P Global Flash US Services PMI", "2026-04-23T13:45:00Z"),
            names,
        )
        self.assertIn(
            ("S&P Global US Manufacturing PMI", "2026-05-01T13:45:00Z"),
            names,
        )
        self.assertIn(
            ("S&P Global US Services PMI", "2026-05-05T13:45:00Z"),
            names,
        )

    def test_ism_calendar_adds_manufacturing_and_services_windows(self):
        sample_ism_html = """
        <html><body>
        <h3>2026 ISM PMI Reports Release Dates</h3>
        <table><tbody>
        <tr><th scope="row">April 2026</th><td>1</td><td>6</td></tr>
        <tr><th scope="row">May 2026</th><td>1</td><td>5</td></tr>
        </tbody></table>
        </body></html>
        """

        with patch.object(
            update_event_risk_windows.requests,
            "get",
            return_value=_FakeResponse(sample_ism_html),
        ):
            windows = update_event_risk_windows.fetch_ism_pmi_windows(
                now_et=pd.Timestamp("2026-04-01T13:00:00-04:00").to_pydatetime(),
                horizon_days=40,
            )

        names = {(item["name"], item["release"]) for item in windows}
        self.assertIn(("ISM Services PMI", "2026-04-06T14:00:00Z"), names)
        self.assertIn(("ISM Manufacturing PMI", "2026-05-01T14:00:00Z"), names)
        self.assertIn(("ISM Services PMI", "2026-05-05T14:00:00Z"), names)

    def test_static_jolts_schedule_adds_known_release_dates(self):
        windows = update_event_risk_windows.fetch_jolts_windows(
            now_et=pd.Timestamp("2026-04-23T12:00:00-04:00").to_pydatetime(),
            horizon_days=80,
        )

        names = {(item["name"], item["release"]) for item in windows}
        self.assertIn(("JOLTS", "2026-05-05T14:00:00Z"), names)
        self.assertIn(("JOLTS", "2026-06-02T14:00:00Z"), names)
        self.assertIn(("JOLTS", "2026-06-30T14:00:00Z"), names)

    def test_michigan_sentiment_schedule_adds_prelim_and_final_windows(self):
        windows = update_event_risk_windows.fetch_michigan_sentiment_windows(
            now_et=pd.Timestamp("2026-04-23T12:00:00-04:00").to_pydatetime(),
            horizon_days=80,
        )

        names = {(item["name"], item["release"]) for item in windows}
        self.assertIn(
            ("Michigan Consumer Sentiment (Prelim)", "2026-05-08T14:00:00Z"),
            names,
        )
        self.assertIn(
            ("Michigan Consumer Sentiment (Final)", "2026-05-22T14:00:00Z"),
            names,
        )
        self.assertIn(
            ("Michigan Consumer Sentiment (Prelim)", "2026-06-12T14:00:00Z"),
            names,
        )

    def test_conference_board_consumer_confidence_uses_last_tuesday_rule(self):
        windows = update_event_risk_windows.fetch_conference_board_consumer_confidence_windows(
            now_et=pd.Timestamp("2026-04-23T12:00:00-04:00").to_pydatetime(),
            horizon_days=80,
        )

        names = {(item["name"], item["release"]) for item in windows}
        self.assertIn(
            ("Conference Board Consumer Confidence", "2026-04-28T14:00:00Z"),
            names,
        )
        self.assertIn(
            ("Conference Board Consumer Confidence", "2026-05-26T14:00:00Z"),
            names,
        )
        self.assertIn(
            ("Conference Board Consumer Confidence", "2026-06-30T14:00:00Z"),
            names,
        )

    def test_fomc_minutes_windows_follow_three_week_rule(self):
        sample_fomc_html = """
        <html><body>
        <div>2026 FOMC Meetings</div>
        <div>January</div>
        <div>27-28</div>
        <div>March</div>
        <div>17-18*</div>
        <div>2027 FOMC Meetings</div>
        </body></html>
        """

        with patch.object(
            update_event_risk_windows.requests,
            "get",
            return_value=_FakeResponse(sample_fomc_html),
        ):
            windows = update_event_risk_windows.fetch_fomc_minutes_windows(
                now_et=pd.Timestamp("2026-02-01T12:00:00-05:00").to_pydatetime(),
                horizon_days=90,
            )

        names = {(item["name"], item["release"]) for item in windows}
        self.assertIn(("FOMC Minutes", "2026-02-18T19:00:00Z"), names)
        self.assertIn(("FOMC Minutes", "2026-04-08T18:00:00Z"), names)

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
        self.assertIn("ExecutionQuality", body)
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

        with patch.object(app_module.predict_gold, "get_technical_analysis", return_value=ta_payload), patch.object(
            app_module,
            "_evaluate_trade_playbook",
            return_value={
                "stage": "hold",
                "title": "Hold Winner",
                "text": "Momentum is active and still aligned with the short position.",
                "directionalBias": "Bearish",
            },
        ), patch.object(
            app_module,
            "_stabilize_trade_playbook",
            side_effect=lambda playbook, *_: playbook,
        ):
            body, status_code = app_module._build_prediction_response()

        self.assertEqual(status_code, 200)
        self.assertEqual(body["ExecutionState"]["actionState"], "SHORT_ACTIVE")
        self.assertEqual(body["DecisionStatus"]["status"], "sell")
        self.assertEqual(
            body["DecisionStatus"]["text"],
            "Safer to look for a sell. Short state is confirmed with acceptable tradeability.",
        )

    def test_prediction_builder_uses_stable_playbook_for_execution_state(self):
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
            "ExecutionState": {"status": "stand_aside", "title": "Stand Aside", "text": "Raw execution state."},
            "directionalBias": "Bearish",
            "tradeability": "High",
            "regime": "trend",
            "actionState": "SHORT_ACTIVE",
            "action": "sell",
        }

        with patch.object(app_module.predict_gold, "get_technical_analysis", return_value=ta_payload), patch.object(
            app_module, "compute_prediction_from_ta", return_value=prediction_payload
        ), patch.object(
            app_module, "_stabilize_prediction", side_effect=lambda prediction, _: prediction
        ), patch.object(
            app_module,
            "_evaluate_decision_status",
            return_value={"status": "sell", "text": "Safer to look for a sell.", "buyChecks": [], "sellChecks": [True], "exitChecks": []},
        ), patch.object(
            app_module,
            "_stabilize_decision_status",
            side_effect=lambda decision: decision,
        ), patch.object(
            app_module,
            "_evaluate_trade_playbook",
            return_value={
                "stage": "hold",
                "title": "Hold Winner",
                "text": "Momentum is active and still aligned with the short position.",
                "directionalBias": "Bearish",
            },
        ), patch.object(
            app_module,
            "_stabilize_trade_playbook",
            side_effect=lambda playbook, *_: playbook,
        ):
            body, status_code = app_module._build_prediction_response()

        self.assertEqual(status_code, 200)
        self.assertEqual(body["ExecutionState"]["status"], "hold")
        self.assertEqual(body["ExecutionState"]["title"], "Hold Winner")
        self.assertEqual(body["ExecutionState"]["actionState"], "SHORT_ACTIVE")
        self.assertEqual(
            body["ExecutionState"]["text"],
            "Momentum is active and still aligned with the short position.",
        )

    def test_execution_quality_builds_institutional_vwap_orb_long_plan(self):
        ta_payload = _sample_ta_payload()
        ta_payload["current_price"] = 4800.0
        ta_payload["price_action"]["structure"] = "Bullish Drift"
        ta_payload["volatility_regime"].update(
            {
                "market_regime": "Trending",
                "adx_14": 28.0,
                "atr_14": 8.0,
                "atr_percent": 0.32,
            }
        )
        ta_payload["structure_context"].update(
            {
                "openingRangeBreak": 1,
                "sessionVwap": 4792.0,
                "distSessionVwapPct": 0.17,
                "pivotPoint": 4794.0,
                "pivotResistance1": 4824.0,
                "pivotSupport1": 4782.0,
                "pivotResistance2": 4848.0,
                "pivotSupport2": 4768.0,
            }
        )
        ta_payload["support_resistance"].update(
            {
                "nearest_support": {"label": "PP", "price": 4794.0},
                "nearest_resistance": {"label": "R1", "price": 4824.0},
                "pivot_levels": {
                    "pp": 4794.0,
                    "r1": 4824.0,
                    "s1": 4782.0,
                    "r2": 4848.0,
                    "s2": 4768.0,
                },
            }
        )

        plan = app_module._build_execution_quality_plan(
            ta_payload,
            {"breakout_bias": "Bullish"},
            {"status": "buy", "text": "Safer to look for a buy."},
            {"actionState": "LONG_ACTIVE", "status": "enter"},
        )

        self.assertEqual(plan["direction"], "Long")
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["grade"], "A")
        self.assertIn("Long", plan["setup"])
        self.assertGreaterEqual(plan["riskReward"], 1.5)
        self.assertEqual(plan["stopLoss"]["basis"], "PP")
        self.assertEqual(plan["targets"][0]["basis"], "R1")

    def test_execution_quality_blocks_low_reward_to_risk_location(self):
        ta_payload = _sample_ta_payload()
        ta_payload["current_price"] = 4800.0
        ta_payload["price_action"]["structure"] = "Bearish Drift"
        ta_payload["volatility_regime"].update(
            {
                "market_regime": "Trending",
                "adx_14": 31.0,
                "atr_14": 8.0,
                "atr_percent": 0.34,
            }
        )
        ta_payload["structure_context"].update(
            {
                "openingRangeBreak": -1,
                "sessionVwap": 4812.0,
                "distSessionVwapPct": -0.25,
                "pivotPoint": 4810.0,
                "pivotResistance1": 4812.0,
                "pivotSupport1": 4796.0,
                "pivotResistance2": 4826.0,
                "pivotSupport2": 4782.0,
            }
        )
        ta_payload["support_resistance"].update(
            {
                "nearest_support": {"label": "S1", "price": 4796.0},
                "nearest_resistance": {"label": "PP", "price": 4810.0},
                "pivot_levels": {
                    "pp": 4810.0,
                    "r1": 4812.0,
                    "s1": 4796.0,
                    "r2": 4826.0,
                    "s2": 4782.0,
                },
            }
        )

        plan = app_module._build_execution_quality_plan(
            ta_payload,
            {"breakout_bias": "Bearish"},
            {"status": "sell", "text": "Safer to look for a sell."},
            {"actionState": "SHORT_ACTIVE", "status": "enter"},
        )

        self.assertEqual(plan["direction"], "Short")
        self.assertEqual(plan["status"], "blocked")
        self.assertEqual(plan["grade"], "No Trade")
        self.assertTrue(any("too close" in item for item in plan["blockers"]))
        self.assertLess(plan["targets"][1]["price"], plan["targets"][0]["price"])

    def test_execution_quality_alert_signal_emphasizes_blocked_no_trade(self):
        signal = app_module._execution_quality_alert_signal(
            {
                "status": "blocked",
                "grade": "No Trade",
                "score": 52,
                "direction": "Long",
                "setup": "Long Watchlist Reclaim",
                "riskReward": 1.57,
                "blockers": ["Recent Swing High is too close for a clean long target."],
            }
        )

        self.assertEqual(
            signal,
            "No Trade blocked: Recent Swing High is too close for a clean long target.",
        )

    def test_execution_quality_alert_signal_avoids_duplicate_direction(self):
        ready_signal = app_module._execution_quality_alert_signal(
            {
                "status": "ready",
                "grade": "A",
                "direction": "Long",
                "setup": "VWAP / ORB Continuation Long",
                "riskReward": 1.8,
                "blockers": [],
            }
        )
        watchlist_signal = app_module._execution_quality_alert_signal(
            {
                "status": "watchlist",
                "grade": "C",
                "direction": "Long",
                "setup": "Long Watchlist Reclaim",
                "riskReward": 1.6,
                "blockers": [],
            }
        )

        self.assertEqual(ready_signal, "A VWAP / ORB Continuation Long >=1.5R")
        self.assertEqual(watchlist_signal, "Watchlist Long Watchlist Reclaim >=1.5R")

    def test_stand_aside_playbook_prefers_no_trade_permission_text(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            app_module,
            "PLAYBOOK_STATE_FILE",
            Path(tmpdir) / "playbook_state.json",
        ), patch.object(app_module.time, "time", return_value=1_700_000_000):
            playbook = app_module._stabilize_trade_playbook(
                {
                    "stage": "stand_aside",
                    "title": "Stand Aside",
                    "text": "Let the normal directional engine do the work.",
                    "why": [],
                    "entryReadiness": "low",
                    "exitUrgency": "low",
                    "warningLadder": "Normal",
                    "eventRegime": "normal",
                    "breakoutBias": "Neutral",
                    "directionalBias": "Bearish",
                    "alignment": "mixed",
                },
                {
                    "status": "no_trade",
                    "text": "No trade. Conditions are not clean enough yet.",
                },
                {
                    "status": "sell",
                    "text": "Safer to look for a sell. Short state is confirmed with acceptable tradeability.",
                },
            )

        self.assertEqual(playbook["title"], "Stand Aside")
        self.assertEqual(playbook["text"], "No trade. Conditions are not clean enough yet.")
        self.assertNotIn("Safer to look for a sell", playbook["text"])

    def test_stabilize_decision_status_requires_repeat_for_same_status_detail_changes(self):
        first_payload = {
            "status": "wait",
            "text": "Watchlist Only: buy conditions are mostly aligned, but execution is blocked.",
            "buyChecks": [True, True, False],
            "sellChecks": [False, False, False],
            "exitChecks": [False, False, False],
        }
        second_payload = {
            "status": "wait",
            "text": "Watchlist Only: sell conditions are mostly aligned, but execution is blocked.",
            "buyChecks": [False, False, False],
            "sellChecks": [True, True, False],
            "exitChecks": [False, False, False],
        }

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            app_module,
            "DECISION_STATE_FILE",
            Path(tmpdir) / "decision_state.json",
        ), patch.object(app_module.time, "time", return_value=1_700_000_000):
            first = app_module._stabilize_decision_status(first_payload)
            second = app_module._stabilize_decision_status(second_payload)
            third = app_module._stabilize_decision_status(second_payload)

        self.assertEqual(first["text"], first_payload["text"])
        self.assertEqual(second["text"], first_payload["text"])
        self.assertEqual(second["buyChecks"], first_payload["buyChecks"])
        self.assertEqual(second["sellChecks"], first_payload["sellChecks"])
        self.assertEqual(third["text"], second_payload["text"])
        self.assertEqual(third["buyChecks"], second_payload["buyChecks"])
        self.assertEqual(third["sellChecks"], second_payload["sellChecks"])

    def test_predict_route_returns_error_payload_when_builder_fails(self):
        with patch.object(app_module, "_build_prediction_response", side_effect=RuntimeError("boom")):
            client = app_module.app.test_client()
            response = client.get("/api/predict")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"status": "error", "message": "boom"})


if __name__ == "__main__":
    unittest.main()
