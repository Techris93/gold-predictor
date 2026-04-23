from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tools.autoresearch_job import AutoResearchJobPaths


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "tools" / "reports"
LATEST_REPORT_FILE = REPORTS_DIR / "autoresearch_last.json"
ACTIVE_SNAPSHOT_FILE = REPORTS_DIR / "autoresearch_active.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _latest_job_name(base_dir: Path) -> Optional[str]:
    jobs_root = base_dir / "tools" / "reports" / "autoresearch_jobs"
    if not jobs_root.exists():
        return None
    latest_dir: Optional[Path] = None
    for candidate in jobs_root.iterdir():
        if not candidate.is_dir() or candidate.name.startswith("smoke_"):
            continue
        if latest_dir is None or candidate.stat().st_mtime > latest_dir.stat().st_mtime:
            latest_dir = candidate
    return latest_dir.name if latest_dir else None


def research_status(base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    latest_report = _read_json(base_dir / "tools" / "reports" / "autoresearch_last.json")
    active_snapshot = _read_json(base_dir / "tools" / "reports" / "autoresearch_active.json")
    latest_job_name = _latest_job_name(base_dir)
    latest_job = load_job_bundle(latest_job_name, base_dir=base_dir) if latest_job_name else None
    final_report = (latest_job or {}).get("finalReport") or latest_report
    best = final_report.get("best") if isinstance(final_report.get("best"), dict) else {}
    return {
        "status": "ok",
        "generated_at": _now(),
        "latest_job_name": latest_job_name,
        "latest_report_generated_at": final_report.get("generated_at"),
        "latest_report_path": str(base_dir / "tools" / "reports" / "autoresearch_last.json"),
        "promote": bool(final_report.get("promote")),
        "promotion_reason": final_report.get("promotion_reason"),
        "market_data_source": final_report.get("market_data_source"),
        "search_mode": final_report.get("search_mode"),
        "final_stage": final_report.get("final_stage"),
        "best_candidate": {
            "params": best.get("params") if isinstance(best, dict) else {},
            "median_score": best.get("median_score") if isinstance(best, dict) else None,
            "ranking_score": best.get("ranking_score") if isinstance(best, dict) else None,
            "pass_rate": best.get("pass_rate") if isinstance(best, dict) else None,
        },
        "active_snapshot": {
            "generated_at": active_snapshot.get("generated_at"),
            "matches_candidate": (active_snapshot.get("recommendation") or {}).get("matches_active_strategy"),
        },
        "job_state": (latest_job or {}).get("state") or {},
        "heartbeat": (latest_job or {}).get("heartbeat") or {},
    }


def create_research_brief(
    *,
    job_name: str,
    hypothesis: str,
    focus: str = "",
    constraints: str = "",
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    paths = AutoResearchJobPaths(base_dir, job_name)
    paths.ensure()
    brief = {
        "job_name": job_name,
        "created_at": _now(),
        "hypothesis": hypothesis.strip(),
        "focus": focus.strip(),
        "constraints": constraints.strip(),
        "job_dir": str(paths.job_dir),
    }
    brief_json_path = paths.job_dir / "brief.json"
    brief_md_path = paths.job_dir / "brief.md"
    brief_json_path.write_text(json.dumps(brief, indent=2), encoding="utf-8")
    brief_md_path.write_text(
        "\n".join(
            [
                f"# Autoresearch Brief: {job_name}",
                "",
                f"- Created at: {brief['created_at']}",
                f"- Hypothesis: {brief['hypothesis']}",
                f"- Focus: {brief['focus'] or 'none'}",
                f"- Constraints: {brief['constraints'] or 'none'}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    brief["brief_json"] = str(brief_json_path)
    brief["brief_markdown"] = str(brief_md_path)
    return brief


def load_job_bundle(job_name: Optional[str], *, base_dir: Path = BASE_DIR) -> Optional[Dict[str, Any]]:
    if not job_name:
        return None
    paths = AutoResearchJobPaths(base_dir, job_name)
    if not paths.job_dir.exists():
        return None
    return {
        "jobName": job_name,
        "jobDir": str(paths.job_dir),
        "brief": _read_json(paths.job_dir / "brief.json"),
        "plan": _read_json(paths.plan_file),
        "state": _read_json(paths.state_file),
        "heartbeat": _read_json(paths.heartbeat_file),
        "leaderboard": _read_json(paths.leaderboard_file),
        "finalReport": _read_json(paths.final_report_file),
    }
