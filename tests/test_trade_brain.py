from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()