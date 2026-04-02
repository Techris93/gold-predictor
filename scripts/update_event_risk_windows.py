#!/usr/bin/env python3
"""
Refresh event-risk suppression windows from official BLS and Federal Reserve
release calendars.

Output format matches config/event_risk_windows.json:
{
  "windows": [
    {"name": "...", "start": "...Z", "end": "...Z", "reason": "..."}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
ET = ZoneInfo("America/New_York")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "event_risk_windows.json"


@dataclass(frozen=True)
class WindowConfig:
    name: str
    summary_match: str
    release_time_et: tuple[int, int]
    start_pad_minutes: int
    end_pad_minutes: int
    reason_template: str


BLS_WINDOWS = [
    WindowConfig(
        name="NFP / Employment Situation",
        summary_match="Employment Situation",
        release_time_et=(8, 30),
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template="BLS Employment Situation release scheduled 08:30 ET on {date_label}.",
    ),
    WindowConfig(
        name="CPI",
        summary_match="Consumer Price Index",
        release_time_et=(8, 30),
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template="BLS Consumer Price Index release scheduled 08:30 ET on {date_label}.",
    ),
    WindowConfig(
        name="PPI",
        summary_match="Producer Price Index",
        release_time_et=(8, 30),
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template="BLS Producer Price Index release scheduled 08:30 ET on {date_label}.",
    ),
]


def _strip_html(raw_html: str) -> list[str]:
    text = re.sub(r"(?is)<script.*?>.*?</script>", "\n", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", "\n", text)
    text = unescape(text)
    lines = []
    for line in text.splitlines():
        normalized = " ".join(line.replace("\xa0", " ").split())
        if normalized:
            lines.append(normalized)
    return lines


def _parse_ics_datetime(value: str) -> datetime:
    value = value.strip()
    if "T" in value:
        value = value.rstrip("Z")
        dt = datetime.strptime(value, "%Y%m%dT%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    dt = datetime.strptime(value, "%Y%m%d")
    return dt.replace(tzinfo=ET)


def _parse_ics_events(raw_ics: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    current_key: str | None = None

    for raw_line in raw_ics.splitlines():
        line = raw_line.rstrip("\r")
        if line == "BEGIN:VEVENT":
            current = {}
            current_key = None
            continue
        if line == "END:VEVENT":
            if current:
                events.append(current)
            current = None
            current_key = None
            continue
        if current is None:
            continue
        if line.startswith((" ", "\t")) and current_key:
            current[current_key] += line[1:]
            continue

        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_key = key.split(";", 1)[0]
        current[current_key] = value

    return events


def _format_window(start_et: datetime, config: WindowConfig, reason: str) -> dict[str, str]:
    start_utc = (start_et - timedelta(minutes=config.start_pad_minutes)).astimezone(timezone.utc)
    end_utc = (start_et + timedelta(minutes=config.end_pad_minutes)).astimezone(timezone.utc)
    return {
        "name": config.name,
        "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": reason,
    }


def fetch_bls_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    response = requests.get(BLS_ICS_URL, timeout=30)
    response.raise_for_status()
    events = _parse_ics_events(response.text)
    end_cutoff = now_et + timedelta(days=horizon_days)

    windows: list[dict[str, str]] = []
    for event in events:
        summary = event.get("SUMMARY", "")
        dt_start_raw = event.get("DTSTART")
        if not summary or not dt_start_raw:
            continue

        parsed_start = _parse_ics_datetime(dt_start_raw)
        if parsed_start.tzinfo != ET:
            parsed_start = parsed_start.astimezone(ET)

        for config in BLS_WINDOWS:
            if config.summary_match not in summary:
                continue

            release_dt = parsed_start.replace(
                hour=config.release_time_et[0],
                minute=config.release_time_et[1],
                second=0,
                microsecond=0,
            )
            if release_dt < now_et or release_dt > end_cutoff:
                continue

            windows.append(
                _format_window(
                    start_et=release_dt,
                    config=config,
                    reason=config.reason_template.format(date_label=release_dt.strftime("%B %-d, %Y")),
                )
            )
            break

    return windows


def _month_number(month_label: str) -> int:
    cleaned = month_label.replace(".", "").strip()
    try:
        return datetime.strptime(cleaned, "%B").month
    except ValueError:
        return datetime.strptime(cleaned, "%b").month


def _extract_fomc_statement_dates(raw_html: str, target_year: int) -> list[datetime]:
    lines = _strip_html(raw_html)
    header = f"{target_year} FOMC Meetings"
    next_header_pattern = re.compile(r"^\d{4} FOMC Meetings$")

    try:
        start_index = lines.index(header) + 1
    except ValueError as exc:
        raise RuntimeError(f"Could not locate {header} section in FOMC calendar.") from exc

    section: list[str] = []
    for line in lines[start_index:]:
        if next_header_pattern.match(line):
            break
        section.append(line)

    statement_dates: list[datetime] = []
    month_pattern = re.compile(r"^[A-Za-z]+(?:/[A-Za-z]+)?$")
    range_pattern = re.compile(r"^(\d{1,2})-(\d{1,2})(\*)?$")

    idx = 0
    while idx < len(section):
        month_label = section[idx]
        if not month_pattern.match(month_label):
            idx += 1
            continue

        if idx + 1 >= len(section):
            break

        range_line = section[idx + 1]
        match = range_pattern.match(range_line)
        if not match:
            idx += 1
            continue

        month_parts = month_label.split("/")
        end_month_label = month_parts[-1]
        statement_day = int(match.group(2))
        statement_month = _month_number(end_month_label)
        statement_dates.append(datetime(target_year, statement_month, statement_day, 14, 0, tzinfo=ET))
        idx += 2

    return statement_dates


def fetch_fomc_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    response = requests.get(FOMC_CALENDAR_URL, timeout=30)
    response.raise_for_status()
    end_cutoff = now_et + timedelta(days=horizon_days)

    statement_dates: list[datetime] = []
    for year in {now_et.year, end_cutoff.year}:
        statement_dates.extend(_extract_fomc_statement_dates(response.text, year))

    windows: list[dict[str, str]] = []
    for statement_dt in sorted(statement_dates):
        if statement_dt < now_et or statement_dt > end_cutoff:
            continue
        start_utc = (statement_dt - timedelta(minutes=45)).astimezone(timezone.utc)
        end_utc = (statement_dt + timedelta(minutes=90)).astimezone(timezone.utc)
        windows.append(
            {
                "name": "FOMC Statement / Press Conference",
                "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reason": (
                    "Federal Reserve FOMC statement and press conference window "
                    f"for {statement_dt.strftime('%B %-d, %Y')}."
                ),
            }
        )

    return windows


def build_windows(horizon_days: int) -> dict[str, list[dict[str, str]]]:
    now_et = datetime.now(tz=ET)
    windows = fetch_bls_windows(now_et=now_et, horizon_days=horizon_days)
    windows.extend(fetch_fomc_windows(now_et=now_et, horizon_days=horizon_days))
    windows.sort(key=lambda item: item["start"])
    return {"windows": windows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh macro event-risk windows from official schedules.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for JSON file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=180,
        help="Number of future days to include in the generated window list.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated JSON to stdout instead of writing a file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_windows(horizon_days=args.horizon_days)

    if args.stdout:
        print(json.dumps(payload, indent=2))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {args.output} with {len(payload['windows'])} future windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
