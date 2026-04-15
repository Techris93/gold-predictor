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
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "event_risk_windows.json"


@dataclass(frozen=True)
class WindowConfig:
    name: str
    source_name: str
    fred_release_id: int
    start_pad_minutes: int
    end_pad_minutes: int
    reason_template: str


BLS_WINDOWS = [
    WindowConfig(
        name="NFP / Employment Situation",
        source_name="Employment Situation",
        fred_release_id=50,
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Employment Situation release date from the St. Louis Fed FRED release calendar "
            "for {date_label}; FRED notes dates are published by the source."
        ),
    ),
    WindowConfig(
        name="CPI",
        source_name="Consumer Price Index",
        fred_release_id=10,
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Consumer Price Index release date from the St. Louis Fed FRED release calendar "
            "for {date_label}; FRED notes dates are published by the source."
        ),
    ),
    WindowConfig(
        name="PPI",
        source_name="Producer Price Index",
        fred_release_id=46,
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Producer Price Index release date from the St. Louis Fed FRED release calendar "
            "for {date_label}; FRED notes dates are published by the source."
        ),
    ),
]


@dataclass(frozen=True)
class MacroReleaseConfig:
    name: str
    match_terms: tuple[str, ...]
    start_pad_minutes: int
    end_pad_minutes: int
    reason_template: str


NEAR_RELEASE_WINDOWS = [
    MacroReleaseConfig(
        name="Initial Jobless Claims",
        match_terms=("unemployment insurance weekly claims",),
        start_pad_minutes=20,
        end_pad_minutes=60,
        reason_template=(
            "Release date from the St. Louis Fed FRED all-releases calendar for {date_label} "
            "({source_label})."
        ),
    ),
    MacroReleaseConfig(
        name="ADP Employment",
        match_terms=("adp", "employment"),
        start_pad_minutes=20,
        end_pad_minutes=60,
        reason_template=(
            "Release date from the St. Louis Fed FRED all-releases calendar for {date_label} "
            "({source_label})."
        ),
    ),
    MacroReleaseConfig(
        name="Retail Sales",
        match_terms=("retail sales",),
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Release date from the St. Louis Fed FRED all-releases calendar for {date_label} "
            "({source_label})."
        ),
    ),
    MacroReleaseConfig(
        name="Personal Income and Outlays",
        match_terms=("personal income and outlays",),
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Release date from the St. Louis Fed FRED all-releases calendar for {date_label} "
            "({source_label})."
        ),
    ),
    MacroReleaseConfig(
        name="Gross Domestic Product",
        match_terms=("gross domestic product",),
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Release date from the St. Louis Fed FRED all-releases calendar for {date_label} "
            "({source_label})."
        ),
    ),
]

ALL_RELEASES_CALENDAR_URL = "https://fred.stlouisfed.org/releases/calendar?y={year}"


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


def _format_window(start_et: datetime, config: WindowConfig, reason: str) -> dict[str, str]:
    start_utc = (start_et - timedelta(minutes=config.start_pad_minutes)).astimezone(timezone.utc)
    end_utc = (start_et + timedelta(minutes=config.end_pad_minutes)).astimezone(timezone.utc)
    return {
        "name": config.name,
        "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": reason,
    }


def _format_custom_window(
    start_et: datetime,
    name: str,
    start_pad_minutes: int,
    end_pad_minutes: int,
    reason: str,
) -> dict[str, str]:
    start_utc = (start_et - timedelta(minutes=start_pad_minutes)).astimezone(timezone.utc)
    end_utc = (start_et + timedelta(minutes=end_pad_minutes)).astimezone(timezone.utc)
    return {
        "name": name,
        "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": reason,
    }


def _fred_release_url(release_id: int, year: int) -> str:
    return f"https://fred.stlouisfed.org/releases/calendar?rid={release_id}&y={year}"


def _extract_fred_release_datetimes(raw_html: str, release_name: str, target_year: int) -> list[datetime]:
    lines = _strip_html(raw_html)
    year_marker = str(target_year)
    date_pattern = re.compile(
        rf"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) "
        rf"([A-Za-z]+) (\d{{1,2}}), {target_year}(?: Updated)?$"
    )
    time_pattern = re.compile(r"^(\d{1,2}:\d{2})\s*(am|pm)$", re.IGNORECASE)

    if year_marker not in lines:
        return []

    release_datetimes: list[datetime] = []
    for idx, line in enumerate(lines):
        date_match = date_pattern.match(line)
        if not date_match:
            continue

        cursor = idx + 1
        while cursor < len(lines) and str(lines[cursor]).lower() == "updated":
            cursor += 1
        if cursor >= len(lines):
            continue

        time_match = time_pattern.match(lines[cursor])
        if not time_match:
            continue

        release_line = lines[cursor + 1] if cursor + 1 < len(lines) else ""
        if release_name.lower() not in release_line.lower():
            continue

        month_name = date_match.group(2)
        day = int(date_match.group(3))
        time_label = f"{time_match.group(1)} {time_match.group(2).lower()}"
        release_dt_ct = datetime.strptime(
            f"{target_year} {month_name} {day} {time_label}",
            "%Y %B %d %I:%M %p",
        ).replace(tzinfo=CT)
        release_datetimes.append(release_dt_ct.astimezone(ET))

    return release_datetimes


def _load_fred_release_options(year: int) -> list[tuple[int, str]]:
    response = requests.get(ALL_RELEASES_CALENDAR_URL.format(year=year), timeout=30)
    response.raise_for_status()
    options: list[tuple[int, str]] = []
    for rid_raw, name_raw in re.findall(r'<option\s+value="(\d+)"[^>]*>([^<]+)</option>', response.text, flags=re.IGNORECASE):
        try:
            rid = int(rid_raw)
        except ValueError:
            continue
        cleaned_name = " ".join(unescape(name_raw).replace("\xa0", " ").split())
        if cleaned_name:
            options.append((rid, cleaned_name))
    return options


def fetch_bls_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    end_cutoff = now_et + timedelta(days=horizon_days)
    years = range(now_et.year, end_cutoff.year + 1)
    windows: list[dict[str, str]] = []

    for config in BLS_WINDOWS:
        for year in years:
            response = requests.get(_fred_release_url(config.fred_release_id, year), timeout=30)
            response.raise_for_status()
            release_datetimes = _extract_fred_release_datetimes(
                raw_html=response.text,
                release_name=config.source_name,
                target_year=year,
            )
            for release_dt in release_datetimes:
                if release_dt < now_et or release_dt > end_cutoff:
                    continue
                windows.append(
                    _format_window(
                        start_et=release_dt,
                        config=config,
                        reason=config.reason_template.format(date_label=release_dt.strftime("%B %-d, %Y")),
                    )
                )

    return windows


def fetch_near_macro_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    end_cutoff = now_et + timedelta(days=horizon_days)
    years = range(now_et.year, end_cutoff.year + 1)
    windows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    release_options = _load_fred_release_options(now_et.year)
    for config in NEAR_RELEASE_WINDOWS:
        matched_options = [
            (rid, source_name)
            for rid, source_name in release_options
            if all(term in source_name.lower() for term in config.match_terms)
        ]
        for rid, source_name in matched_options:
            for year in years:
                response = requests.get(_fred_release_url(rid, year), timeout=30)
                response.raise_for_status()
                release_datetimes = _extract_fred_release_datetimes(
                    raw_html=response.text,
                    release_name=source_name,
                    target_year=year,
                )
                for release_dt in release_datetimes:
                    if release_dt < now_et or release_dt > end_cutoff:
                        continue
                    dedupe_key = (config.name, release_dt.isoformat())
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    windows.append(
                        _format_custom_window(
                            start_et=release_dt,
                            name=config.name,
                            start_pad_minutes=config.start_pad_minutes,
                            end_pad_minutes=config.end_pad_minutes,
                            reason=config.reason_template.format(
                                date_label=release_dt.strftime("%B %-d, %Y"),
                                source_label=source_name,
                            ),
                        )
                    )

    return windows


def load_existing_bls_windows(output_path: Path, now_utc: datetime, horizon_days: int) -> list[dict[str, str]]:
    if not output_path.exists():
        return []

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    end_cutoff = now_utc + timedelta(days=horizon_days)
    known_names = {config.name for config in BLS_WINDOWS}
    preserved: list[dict[str, str]] = []

    for item in payload.get("windows", []):
        if item.get("name") not in known_names:
            continue
        try:
            start_dt = datetime.strptime(item["start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if now_utc <= start_dt <= end_cutoff:
            preserved.append(item)

    return preserved


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


def build_windows(horizon_days: int, output_path: Path) -> dict[str, list[dict[str, str]]]:
    now_et = datetime.now(tz=ET)
    now_utc = now_et.astimezone(timezone.utc)

    try:
        windows = fetch_bls_windows(now_et=now_et, horizon_days=horizon_days)
    except Exception as exc:
        windows = load_existing_bls_windows(output_path=output_path, now_utc=now_utc, horizon_days=horizon_days)
        warning = (
            "Warning: failed to refresh BLS windows from the FRED release calendar; "
            f"preserving existing future BLS windows instead. Error: {exc}"
        )
        print(warning, file=sys.stderr)

    try:
        windows.extend(fetch_near_macro_windows(now_et=now_et, horizon_days=horizon_days))
    except Exception as exc:
        warning = (
            "Warning: failed to refresh near macro windows from the FRED all-releases calendar; "
            f"continuing without additional near windows. Error: {exc}"
        )
        print(warning, file=sys.stderr)

    windows.extend(fetch_fomc_windows(now_et=now_et, horizon_days=horizon_days))
    deduped = {}
    for item in windows:
        deduped[(item.get("name"), item.get("start"), item.get("end"))] = item
    windows = list(deduped.values())
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
    payload = build_windows(horizon_days=args.horizon_days, output_path=args.output)

    if args.stdout:
        print(json.dumps(payload, indent=2))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {args.output} with {len(payload['windows'])} future windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
