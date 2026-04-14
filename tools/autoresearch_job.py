from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return default


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                items.append(payload)
    return items


def params_signature(params: Mapping[str, Any]) -> str:
    normalized = json.dumps(dict(sorted((str(key), params[key]) for key in params)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def candidate_key(stage_name: str, params: Mapping[str, Any]) -> str:
    return f"{stage_name}:{params_signature(params)}"


@dataclass(frozen=True)
class AutoResearchJobPaths:
    base_dir: Path
    job_name: str

    @property
    def job_dir(self) -> Path:
        return self.base_dir / "tools" / "reports" / "autoresearch_jobs" / self.job_name

    @property
    def plan_file(self) -> Path:
        return self.job_dir / "plan.json"

    @property
    def state_file(self) -> Path:
        return self.job_dir / "state.json"

    @property
    def leaderboard_file(self) -> Path:
        return self.job_dir / "leaderboard.json"

    @property
    def heartbeat_file(self) -> Path:
        return self.job_dir / "heartbeat.json"

    @property
    def final_report_file(self) -> Path:
        return self.job_dir / "final_report.json"

    def stage_results_file(self, stage_name: str) -> Path:
        return self.job_dir / f"results_{stage_name}.jsonl"

    def ensure(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)