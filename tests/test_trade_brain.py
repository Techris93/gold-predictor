from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from tools.trade_brain import TradeBrainService


class TradeBrainServiceTests(unittest.TestCase):
    def test_enter_trade_sets_defaults_and_tracks_active_trade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TradeBrainService(Path(temp_dir) / "trade_brain_state.json")

            trade = service.enter_trade(
                {
                    "direction": "long",
                    "price": 100.0,
                    "stopLoss": 95.0,
                    "riskDollar": 100.0,
                    "positionSize": 10.0,
                    "context": {
                        "adx": 21.0,
                        "atrPercent": 0.4,
                        "atrDollar": 2.5,
                        "vwap": 99.5,
                        "structure": "bullish continuation",
                        "regime": "trend",
                    },
                }
            )

            self.assertEqual(trade["status"], "ACTIVE")
            self.assertEqual(trade["direction"], "LONG")
            self.assertEqual(trade["plan"]["tp1"], 107.5)
            self.assertEqual(trade["plan"]["tp2"], 115.0)
            self.assertIsNotNone(service.get_active_trade())
            self.assertEqual(service.get_stats()["activeTrades"], 1)

    def test_evaluate_trade_moves_to_breakeven_then_closes_at_tp2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TradeBrainService(Path(temp_dir) / "trade_brain_state.json")
            trade = service.enter_trade(
                {
                    "direction": "LONG",
                    "price": 100.0,
                    "stopLoss": 95.0,
                    "riskDollar": 100.0,
                    "positionSize": 10.0,
                    "context": {
                        "adx": 22.0,
                        "atrPercent": 0.3,
                        "atrDollar": 2.0,
                        "vwap": 99.8,
                        "structure": "bullish continuation",
                        "regime": "trend",
                    },
                }
            )

            first_eval = service.evaluate_active_trade(
                105.0,
                {
                    "adx": 38.0,
                    "vwap": 103.5,
                    "atrDollar": 2.0,
                    "atrPercent": 0.3,
                    "structure": "bullish continuation",
                    "regime": "trend",
                },
            )
            self.assertIsNotNone(first_eval)
            active_trade = service.get_active_trade()
            self.assertIsNotNone(active_trade)
            self.assertGreater(active_trade["live"]["stopLoss"]["current"], 100.0)
            self.assertTrue(active_trade["live"]["trailing"]["active"])

            second_eval = service.evaluate_active_trade(
                115.0,
                {
                    "adx": 40.0,
                    "vwap": 111.0,
                    "atrDollar": 2.0,
                    "atrPercent": 0.3,
                    "structure": "bullish continuation",
                    "regime": "trend",
                },
            )
            self.assertIsNotNone(second_eval)
            closed_trade = second_eval["trade"]
            self.assertEqual(closed_trade["status"], "CLOSED")
            self.assertEqual(closed_trade["exit"]["reason"], "TAKE_PROFIT_2")
            self.assertAlmostEqual(closed_trade["exit"]["finalR"], 3.0)
            stats = service.get_stats()
            self.assertEqual(stats["wins"], 1)
            self.assertEqual(stats["activeTrades"], 0)
            self.assertEqual(stats["totalTrades"], 1)

    def test_tag_emotion_and_review_are_available_after_close(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TradeBrainService(Path(temp_dir) / "trade_brain_state.json")
            trade = service.enter_trade(
                {
                    "direction": "SHORT",
                    "price": 100.0,
                    "stopLoss": 105.0,
                    "riskDollar": 100.0,
                    "context": {
                        "adx": 28.0,
                        "atrPercent": 0.3,
                        "atrDollar": 2.0,
                        "vwap": 100.5,
                        "structure": "bearish continuation",
                        "regime": "trend",
                    },
                }
            )

            service.tag_emotion(trade["id"], "focused", "Sticking to the plan")
            service.close_trade(
                trade["id"],
                exit_price=94.0,
                reason="MANUAL_EXIT",
                reasoning="Momentum faded near the session close.",
                emotion="calm",
            )

            review = service.get_review(trade["id"])
            dashboard = service.get_dashboard_payload()

            self.assertIn(review["grade"], {"A", "B", "C", "D"})
            self.assertGreaterEqual(len(review["lessons"]), 1)
            self.assertEqual(dashboard["analytics"]["emotions"][0]["emotion"], "calm")
            self.assertEqual(dashboard["stats"]["closedTrades"], 1)

    def test_learning_adjustment_uses_closed_trade_rewards_for_matching_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TradeBrainService(Path(temp_dir) / "trade_brain_state.json")

            for offset in (6.0, 5.5, 5.0):
                trade = service.enter_trade(
                    {
                        "direction": "LONG",
                        "price": 100.0,
                        "stopLoss": 95.0,
                        "riskDollar": 100.0,
                        "trigger": "VWAP reclaim",
                        "session": "London",
                        "context": {
                            "adx": 31.0,
                            "atrPercent": 0.25,
                            "atrDollar": 2.0,
                            "vwap": 99.8,
                            "structure": "bullish continuation",
                            "regime": "trend",
                            "session": "London",
                        },
                    }
                )
                service.close_trade(
                    trade["id"],
                    exit_price=100.0 + offset,
                    reason="MANUAL_EXIT",
                    reasoning="Captured the expected continuation move.",
                    emotion="calm",
                )

            adjustment = service.get_learning_adjustment(
                direction="LONG",
                market_data={
                    "session": "London",
                    "regime": "trend",
                    "structure": "bullish continuation",
                },
                setup="VWAP reclaim",
            )
            dashboard = service.get_dashboard_payload(
                market_data={
                    "session": "London",
                    "regime": "trend",
                    "structure": "bullish continuation",
                },
                learning_direction="LONG",
                learning_setup="VWAP reclaim",
            )

            self.assertTrue(adjustment["active"])
            self.assertGreater(adjustment["confidenceDelta"], 0)
            self.assertGreaterEqual(adjustment["closedSamples"], 3)
            self.assertEqual(adjustment["topSetup"]["setup"], "VWAP reclaim")
            self.assertGreater(adjustment["sizing"]["riskMultiplier"], 1.0)
            self.assertGreaterEqual(len(adjustment["rankedSetups"]), 1)
            self.assertTrue(dashboard["learning"]["currentAdjustment"]["active"])
            self.assertEqual(dashboard["learning"]["rankedSetups"][0]["setup"], "VWAP reclaim")
            self.assertGreaterEqual(len(dashboard["learning"]["topEdges"]), 1)

    def test_evaluate_all_active_trades_handles_multiple_user_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TradeBrainService(Path(temp_dir) / "trade_brain_state.json")
            payload = {
                "direction": "LONG",
                "price": 100.0,
                "stopLoss": 95.0,
                "riskDollar": 100.0,
                "context": {
                    "adx": 22.0,
                    "atrPercent": 0.3,
                    "atrDollar": 2.0,
                    "vwap": 99.5,
                    "structure": "bullish continuation",
                    "regime": "trend",
                },
            }

            user_a_trade = service.enter_trade(payload, user_id="user-a")
            user_b_trade = service.enter_trade(payload, user_id="user-b")

            self.assertEqual(service.get_active_trade(user_id="user-a")["id"], user_a_trade["id"])
            self.assertEqual(service.get_active_trade(user_id="user-b")["id"], user_b_trade["id"])

            results = service.evaluate_all_active_trades(
                94.0,
                {
                    "adx": 18.0,
                    "vwap": 97.0,
                    "atrDollar": 2.0,
                    "atrPercent": 0.3,
                    "structure": "bearish reversal",
                    "regime": "trend",
                },
            )

            self.assertEqual({result["userId"] for result in results}, {"user-a", "user-b"})
            self.assertTrue(all(result["trade"]["status"] == "CLOSED" for result in results))
            self.assertEqual(service.get_stats(user_id="user-a")["activeTrades"], 0)
            self.assertEqual(service.get_stats(user_id="user-b")["activeTrades"], 0)

    def test_dashboard_backfills_closed_trades_from_historical_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "live_signal_outcomes.json").write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "1777244605:Enter With Confirmation:Bearish:4681.43",
                                "ts": 1777244605,
                                "price": 4681.43,
                                "verdict": "Bearish",
                                "confidence": 66,
                                "breakoutBias": "Bearish",
                                "warningLadder": "Directional Expansion Likely",
                                "eventRegime": "breakout_watch",
                                "executionStatus": "enter",
                                "entryAllowed": True,
                                "snapshot": {
                                    "market_structure": "Bearish Drift",
                                    "event_regime": "breakout_watch",
                                },
                                "outcomes": {
                                    "30m": {
                                        "return_pct": -0.0606,
                                        "resolved_at": 1777246479,
                                    }
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            service = TradeBrainService(temp_path / "trade_brain_state.json")

            first_dashboard = service.get_dashboard_payload(user_id="tb-user")
            second_dashboard = service.get_dashboard_payload(user_id="tb-user")

            self.assertEqual(first_dashboard["stats"]["closedTrades"], 1)
            self.assertEqual(second_dashboard["stats"]["closedTrades"], 1)
            self.assertEqual(first_dashboard["recentTrades"][0]["direction"], "SHORT")
            self.assertEqual(first_dashboard["recentTrades"][0]["entry"]["trigger"], "Enter With Confirmation")
            self.assertEqual(first_dashboard["recentTrades"][0]["backfill"]["recordId"], "1777244605:Enter With Confirmation:Bearish:4681.43")


class TradeBrainPredictionRouteTests(unittest.TestCase):
    def test_predict_route_passes_trade_brain_user_header(self) -> None:
        with app_module.app.test_client() as client, patch.object(
            app_module,
            "_build_prediction_response",
            return_value=({"status": "success", "TradeBrain": {}}, 200),
        ) as build_prediction:
            response = client.get("/api/predict", headers={"x-user-id": "tb-user-123"})

        self.assertEqual(response.status_code, 200)
        build_prediction.assert_called_once_with(user_id="tb-user-123")


class TradeBrainMonitorTests(unittest.TestCase):
    def test_indicator_monitor_evaluates_active_trades_without_clients(self) -> None:
        result = {
            "userId": "tb-user",
            "trade": {"id": "trade-1", "status": "CLOSED"},
            "events": [{"type": "trade:closed", "tradeId": "trade-1"}],
            "dashboard": {"stats": {"closedTrades": 1}},
        }
        payload = {
            "TradeBrainRuntime": {"events": []},
            "TechnicalAnalysis": {"current_price": 100.0},
            "RegimeState": {},
        }

        with patch.dict(app_module._monitor_state, {"clients": 0}, clear=False), patch.object(
            app_module,
            "_has_background_alert_channels",
            return_value=False,
        ), patch.object(
            app_module.trade_brain_service,
            "has_active_trades",
            return_value=True,
        ), patch.object(
            app_module,
            "_build_prediction_response",
            return_value=(payload, 200),
        ), patch.object(
            app_module,
            "_extract_indicator_snapshot",
            return_value={},
        ), patch.object(
            app_module,
            "_update_live_signal_outcomes",
        ), patch.object(
            app_module.trade_brain_service,
            "evaluate_all_active_trades",
            return_value=[result],
        ) as evaluate_all, patch.object(
            app_module,
            "_emit_trade_brain_events",
        ) as emit_events, patch.object(
            app_module.socketio,
            "sleep",
            side_effect=RuntimeError("stop-monitor"),
        ):
            with self.assertRaises(RuntimeError):
                app_module._indicator_monitor_loop()

        evaluate_all.assert_called_once()
        self.assertEqual(evaluate_all.call_args.args[0], 100.0)
        emit_events.assert_called_once_with(
            result["events"],
            trade=result["trade"],
            dashboard=result["dashboard"],
        )


if __name__ == "__main__":
    unittest.main()