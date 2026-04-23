from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.research_runtime import create_research_brief, load_job_bundle, research_status


class ResearchRuntimeTests(unittest.TestCase):
    def test_create_research_brief_persists_brief_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            payload = create_research_brief(
                job_name="xau-smoke",
                hypothesis="Mean reversion works better around session resets.",
                focus="session structure",
                constraints="keep latency low",
                base_dir=base_dir,
            )

            self.assertTrue(Path(payload["brief_json"]).exists())
            self.assertTrue(Path(payload["brief_markdown"]).exists())
            bundle = load_job_bundle("xau-smoke", base_dir=base_dir)
            self.assertIsNotNone(bundle)
            self.assertEqual(bundle["brief"]["hypothesis"], payload["hypothesis"])

    def test_research_status_surfaces_latest_job_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            job_dir = base_dir / "tools" / "reports" / "autoresearch_jobs" / "xau-live"
            job_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / "tools" / "reports").mkdir(parents=True, exist_ok=True)
            (base_dir / "tools" / "reports" / "autoresearch_last.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-04-23T09:00:00Z",
                        "promote": True,
                        "promotion_reason": "Stronger ranking score.",
                        "market_data_source": "Yahoo Finance",
                        "search_mode": "staged",
                        "final_stage": "confirm",
                        "best": {"params": {"ema_short": 9}, "median_score": 1.2, "ranking_score": 1.3, "pass_rate": 0.82},
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (job_dir / "heartbeat.json").write_text(json.dumps({"stage": "confirm"}), encoding="utf-8")
            (job_dir / "final_report.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-04-23T09:00:00Z",
                        "promote": True,
                        "promotion_reason": "Stronger ranking score.",
                        "market_data_source": "Yahoo Finance",
                        "search_mode": "staged",
                        "final_stage": "confirm",
                        "best": {"params": {"ema_short": 9}, "median_score": 1.2, "ranking_score": 1.3, "pass_rate": 0.82},
                    }
                ),
                encoding="utf-8",
            )

            payload = research_status(base_dir=base_dir)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["latest_job_name"], "xau-live")
            self.assertTrue(payload["promote"])
            self.assertEqual(payload["job_state"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
