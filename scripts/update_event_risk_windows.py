#!/usr/bin/env python3
"""
Refresh event-risk suppression windows from official macro release calendars.

Output format matches config/event_risk_windows.json:
{
  "windows": [
    {
      "name": "...",
      "start": "...Z",
      "end": "...Z",
      "release": "...Z",
      "reason": "..."
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FRED_RELEASES_API_URL = "https://api.stlouisfed.org/fred/releases"
FRED_RELEASE_DATES_API_URL = "https://api.stlouisfed.org/fred/release/dates"
ISM_RELEASE_CALENDAR_URL = "https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar/"
SP_GLOBAL_PMI_CALENDAR_URL = "https://www.pmi.spglobal.com/Public/Release/ReleaseDates?language=en&os=os"
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FRED_API_TIMEOUT_SECONDS = max(5, int(os.getenv("FRED_API_TIMEOUT_SECONDS", "30")))
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
    default_release_time_et: str
    start_pad_minutes: int
    end_pad_minutes: int
    reason_template: str


BLS_WINDOWS = [
    WindowConfig(
        name="NFP / Employment Situation",
        source_name="Employment Situation",
        fred_release_id=50,
        default_release_time_et="08:30",
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Employment Situation release date from the St. Louis Fed FRED release schedule "
            "for {date_label}; FRED notes dates are published by the source."
        ),
    ),
    WindowConfig(
        name="CPI",
        source_name="Consumer Price Index",
        fred_release_id=10,
        default_release_time_et="08:30",
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Consumer Price Index release date from the St. Louis Fed FRED release schedule "
            "for {date_label}; FRED notes dates are published by the source."
        ),
    ),
    WindowConfig(
        name="PPI",
        source_name="Producer Price Index",
        fred_release_id=46,
        default_release_time_et="08:30",
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Producer Price Index release date from the St. Louis Fed FRED release schedule "
            "for {date_label}; FRED notes dates are published by the source."
        ),
    ),
]


@dataclass(frozen=True)
class MacroReleaseConfig:
    name: str
    match_terms: tuple[str, ...]
    default_release_time_et: str
    start_pad_minutes: int
    end_pad_minutes: int
    reason_template: str
    source_name_equals: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class PmiReleaseConfig:
    source_label: str
    event_names: tuple[str, ...]
    start_pad_minutes: int
    end_pad_minutes: int
    reason_template: str


@dataclass(frozen=True)
class StaticReleaseConfig:
    name: str
    release_dates: tuple[str, ...]
    release_time_et: str
    start_pad_minutes: int
    end_pad_minutes: int
    reason_template: str


NEAR_RELEASE_WINDOWS = [
    MacroReleaseConfig(
        name="Initial Jobless Claims",
        match_terms=("unemployment insurance weekly claims",),
        default_release_time_et="08:30",
        start_pad_minutes=20,
        end_pad_minutes=60,
        reason_template=(
            "Release date from the St. Louis Fed FRED release schedule for {date_label} "
            "({source_label})."
        ),
        source_name_equals=("Unemployment Insurance Weekly Claims Report",),
    ),
    MacroReleaseConfig(
        name="ADP Employment",
        match_terms=("adp", "employment"),
        default_release_time_et="08:15",
        start_pad_minutes=20,
        end_pad_minutes=60,
        reason_template=(
            "Release date from the St. Louis Fed FRED release schedule for {date_label} "
            "({source_label})."
        ),
        source_name_equals=("ADP National Employment Report",),
    ),
    MacroReleaseConfig(
        name="Retail Sales",
        match_terms=("retail sales",),
        default_release_time_et="08:30",
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Release date from the St. Louis Fed FRED release schedule for {date_label} "
            "({source_label})."
        ),
        source_name_equals=("Selected Real Retail Sales Series",),
    ),
    MacroReleaseConfig(
        name="Personal Income and Outlays",
        match_terms=("personal income and outlays",),
        default_release_time_et="08:30",
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Release date from the St. Louis Fed FRED release schedule for {date_label} "
            "({source_label})."
        ),
        source_name_equals=("Personal Income and Outlays",),
    ),
    MacroReleaseConfig(
        name="Gross Domestic Product",
        match_terms=("gross domestic product",),
        default_release_time_et="08:30",
        start_pad_minutes=30,
        end_pad_minutes=75,
        reason_template=(
            "Release date from the St. Louis Fed FRED release schedule for {date_label} "
            "({source_label})."
        ),
        source_name_equals=("Gross Domestic Product",),
    ),
]

US_PMI_WINDOWS = [
    PmiReleaseConfig(
        source_label="S&P Global Flash US PMI",
        event_names=(
            "S&P Global Flash US Manufacturing PMI",
            "S&P Global Flash US Services PMI",
        ),
        start_pad_minutes=15,
        end_pad_minutes=45,
        reason_template=(
            "Official S&P Global PMI calendar lists {source_label} for {date_label} "
            "at {release_label} UTC; this publication bundle covers the paired US flash "
            "manufacturing and services PMI releases."
        ),
    ),
    PmiReleaseConfig(
        source_label="S&P Global US Manufacturing PMI",
        event_names=("S&P Global US Manufacturing PMI",),
        start_pad_minutes=15,
        end_pad_minutes=45,
        reason_template=(
            "Official S&P Global PMI calendar lists {source_label} for {date_label} "
            "at {release_label} UTC."
        ),
    ),
    PmiReleaseConfig(
        source_label="S&P Global US Services PMI",
        event_names=("S&P Global US Services PMI",),
        start_pad_minutes=15,
        end_pad_minutes=45,
        reason_template=(
            "Official S&P Global PMI calendar lists {source_label} for {date_label} "
            "at {release_label} UTC."
        ),
    ),
]

ISM_WINDOW_CONFIGS = [
    {
        "name": "ISM Manufacturing PMI",
        "column_key": "manufacturing",
        "start_pad_minutes": 15,
        "end_pad_minutes": 45,
        "reason_template": (
            "Official ISM PMI release calendar lists the {report_label} for {date_label} "
            "at 10:00 ET."
        ),
        "report_label": "ISM Manufacturing PMI",
    },
    {
        "name": "ISM Services PMI",
        "column_key": "services",
        "start_pad_minutes": 15,
        "end_pad_minutes": 45,
        "reason_template": (
            "Official ISM PMI release calendar lists the {report_label} for {date_label} "
            "at 10:00 ET."
        ),
        "report_label": "ISM Services PMI",
    },
]

JOLTS_WINDOWS = [
    StaticReleaseConfig(
        name="JOLTS",
        release_dates=(
            "2026-03-13",
            "2026-03-31",
            "2026-05-05",
            "2026-06-02",
            "2026-06-30",
            "2026-08-04",
            "2026-09-01",
            "2026-09-29",
            "2026-11-03",
            "2026-12-01",
        ),
        release_time_et="10:00",
        start_pad_minutes=15,
        end_pad_minutes=45,
        reason_template=(
            "Official BLS JOLTS release schedule lists {name} for {date_label} "
            "at 10:00 ET."
        ),
    ),
]

MICHIGAN_SENTIMENT_WINDOWS = [
    StaticReleaseConfig(
        name="Michigan Consumer Sentiment (Prelim)",
        release_dates=(
            "2026-01-09",
            "2026-02-06",
            "2026-03-13",
            "2026-04-10",
            "2026-05-08",
            "2026-06-12",
            "2026-07-17",
            "2026-08-14",
            "2026-09-11",
            "2026-10-09",
            "2026-11-06",
            "2026-12-04",
        ),
        release_time_et="10:00",
        start_pad_minutes=15,
        end_pad_minutes=45,
        reason_template=(
            "Official University of Michigan Surveys of Consumers 2026 release dates list "
            "{name} for {date_label} at 10:00 ET."
        ),
    ),
    StaticReleaseConfig(
        name="Michigan Consumer Sentiment (Final)",
        release_dates=(
            "2026-01-23",
            "2026-02-20",
            "2026-03-27",
            "2026-04-24",
            "2026-05-22",
            "2026-06-26",
            "2026-07-31",
            "2026-08-28",
            "2026-09-25",
            "2026-10-23",
            "2026-11-20",
            "2026-12-18",
        ),
        release_time_et="10:00",
        start_pad_minutes=15,
        end_pad_minutes=45,
        reason_template=(
            "Official University of Michigan Surveys of Consumers 2026 release dates list "
            "{name} for {date_label} at 10:00 ET."
        ),
    ),
]

CONFERENCE_BOARD_NAME = "Conference Board Consumer Confidence"
FOMC_MINUTES_NAME = "FOMC Minutes"

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


def _build_window(
    *,
    release_dt: datetime,
    name: str,
    start_pad_minutes: int,
    end_pad_minutes: int,
    reason: str,
) -> dict[str, str]:
    release_utc = release_dt.astimezone(timezone.utc)
    start_utc = (release_dt - timedelta(minutes=start_pad_minutes)).astimezone(timezone.utc)
    end_utc = (release_dt + timedelta(minutes=end_pad_minutes)).astimezone(timezone.utc)
    return {
        "name": name,
        "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "release": release_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": reason,
    }


def _format_window(start_et: datetime, config: WindowConfig, reason: str) -> dict[str, str]:
    return _build_window(
        release_dt=start_et,
        name=config.name,
        start_pad_minutes=config.start_pad_minutes,
        end_pad_minutes=config.end_pad_minutes,
        reason=reason,
    )


def _format_custom_window(
    start_et: datetime,
    name: str,
    start_pad_minutes: int,
    end_pad_minutes: int,
    reason: str,
) -> dict[str, str]:
    return _build_window(
        release_dt=start_et,
        name=name,
        start_pad_minutes=start_pad_minutes,
        end_pad_minutes=end_pad_minutes,
        reason=reason,
    )


def _fetch_text(url: str, *, timeout: int = 30, allow_curl_fallback: bool = False) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text
    if allow_curl_fallback and "captcha_form" in text.lower():
        result = subprocess.run(
            ["curl", "-sL", url],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return str(result.stdout or "")
    return text


def _release_datetime_from_day(year: int, month: int, day: int, release_time_label: str) -> datetime:
    parsed_time = datetime.strptime(str(release_time_label), "%H:%M")
    return datetime(year, month, day, parsed_time.hour, parsed_time.minute, tzinfo=ET)


def _iter_month_starts(start_dt: datetime, end_dt: datetime) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year = start_dt.year
    month = start_dt.month
    while True:
        months.append((year, month))
        if year == end_dt.year and month == end_dt.month:
            break
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def _last_weekday_of_month(year: int, month: int, weekday: int) -> datetime:
    if month == 12:
        cursor = datetime(year + 1, 1, 1, tzinfo=ET) - timedelta(days=1)
    else:
        cursor = datetime(year, month + 1, 1, tzinfo=ET) - timedelta(days=1)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def _fred_release_url(release_id: int, year: int) -> str:
    return f"https://fred.stlouisfed.org/releases/calendar?rid={release_id}&y={year}"


def _fred_api_get_json(url: str, params: dict) -> dict:
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is not configured.")
    payload = dict(params)
    payload["api_key"] = FRED_API_KEY
    payload["file_type"] = "json"
    response = requests.get(url, params=payload, timeout=FRED_API_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _release_datetime_from_date(date_label: str, release_time_label: str) -> datetime:
    release_day = datetime.strptime(str(date_label), "%Y-%m-%d")
    hour = 8
    minute = 30
    try:
        parsed_time = datetime.strptime(str(release_time_label), "%H:%M")
        hour = int(parsed_time.hour)
        minute = int(parsed_time.minute)
    except Exception:
        pass
    return datetime(release_day.year, release_day.month, release_day.day, hour, minute, tzinfo=ET)


def _fetch_release_datetimes_via_api(release_id: int, target_year: int, default_release_time_et: str) -> list[datetime]:
    payload = _fred_api_get_json(
        FRED_RELEASE_DATES_API_URL,
        {
            "release_id": int(release_id),
            "include_release_dates_with_no_data": "true",
            "sort_order": "asc",
            "limit": 10000,
        },
    )
    items = payload.get("release_dates") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []

    release_datetimes: list[datetime] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        date_value = item.get("date")
        if not date_value:
            continue
        try:
            release_dt = _release_datetime_from_date(str(date_value), default_release_time_et)
        except Exception:
            continue
        if release_dt.year == int(target_year):
            release_datetimes.append(release_dt)

    return release_datetimes


def _fetch_release_datetimes(
    release_id: int,
    release_name: str,
    target_year: int,
    default_release_time_et: str,
) -> list[datetime]:
    if FRED_API_KEY:
        try:
            release_datetimes = _fetch_release_datetimes_via_api(
                release_id=release_id,
                target_year=target_year,
                default_release_time_et=default_release_time_et,
            )
            if release_datetimes:
                return release_datetimes
        except Exception as exc:
            print(
                (
                    "Warning: failed to read release dates from FRED API for "
                    f"{release_name} ({target_year}); falling back to release calendar page. Error: {exc}"
                ),
                file=sys.stderr,
            )

    response = requests.get(_fred_release_url(release_id, target_year), timeout=30)
    response.raise_for_status()
    return _extract_fred_release_datetimes(
        raw_html=response.text,
        release_name=release_name,
        target_year=target_year,
    )


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
    if FRED_API_KEY:
        try:
            payload = _fred_api_get_json(
                FRED_RELEASES_API_URL,
                {
                    "limit": 1000,
                    "offset": 0,
                    "order_by": "release_id",
                    "sort_order": "asc",
                },
            )
            entries = payload.get("releases") if isinstance(payload, dict) else None
            if isinstance(entries, list):
                options_from_api: list[tuple[int, str]] = []
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    rid = item.get("id")
                    name = " ".join(str(item.get("name") or "").replace("\xa0", " ").split())
                    if not name:
                        continue
                    try:
                        options_from_api.append((int(rid), name))
                    except Exception:
                        continue
                if options_from_api:
                    return options_from_api
        except Exception as exc:
            print(
                (
                    "Warning: failed to load release list from FRED API; "
                    f"falling back to releases page parser. Error: {exc}"
                ),
                file=sys.stderr,
            )

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
            release_datetimes = _fetch_release_datetimes(
                release_id=config.fred_release_id,
                release_name=config.source_name,
                target_year=year,
                default_release_time_et=config.default_release_time_et,
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
        exact_names = {name.lower() for name in config.source_name_equals}
        if exact_names:
            matched_options = [
                (rid, source_name)
                for rid, source_name in release_options
                if source_name.lower() in exact_names
            ]
        else:
            matched_options = [
                (rid, source_name)
                for rid, source_name in release_options
                if all(term in source_name.lower() for term in config.match_terms)
                and not any(term in source_name.lower() for term in config.exclude_terms)
            ]
        for rid, source_name in matched_options:
            for year in years:
                release_datetimes = _fetch_release_datetimes(
                    release_id=rid,
                    release_name=source_name,
                    target_year=year,
                    default_release_time_et=config.default_release_time_et,
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


def _load_existing_windows(
    output_path: Path,
    now_utc: datetime,
    horizon_days: int,
    names: set[str],
) -> list[dict[str, str]]:
    if not output_path.exists():
        return []

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    end_cutoff = now_utc + timedelta(days=horizon_days)
    preserved: list[dict[str, str]] = []

    for item in payload.get("windows", []):
        if item.get("name") not in names:
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


def _extract_sp_global_calendar_entries(raw_html: str) -> list[tuple[datetime, str]]:
    lines = _strip_html(raw_html)
    year_pattern = re.compile(r"^\d{4}$")
    date_pattern = re.compile(r"^([A-Za-z]+)\s+(\d{1,2})$")
    time_pattern = re.compile(r"^(\d{1,2}:\d{2})\s*UTC(?:\s+(.+))?$")

    current_year: int | None = None
    current_date: datetime | None = None
    entries: list[tuple[datetime, str]] = []
    parsing_calendar = False

    index = 0
    while index < len(lines):
        normalized = str(lines[index]).strip()
        if normalized == "Upcoming":
            parsing_calendar = True
            index += 1
            continue
        if not parsing_calendar:
            index += 1
            continue
        if year_pattern.match(normalized):
            current_year = int(normalized)
            current_date = None
            index += 1
            continue
        date_match = date_pattern.match(normalized)
        if date_match and current_year is not None:
            month = _month_number(date_match.group(1))
            day = int(date_match.group(2))
            current_date = datetime(current_year, month, day, tzinfo=timezone.utc)
            index += 1
            continue
        time_match = time_pattern.match(normalized)
        if time_match and current_date is not None:
            release_time = datetime.strptime(time_match.group(1), "%H:%M")
            release_label = str(time_match.group(2) or "").strip()
            if not release_label and index + 1 < len(lines):
                release_label = str(lines[index + 1]).strip()
                if (
                    not release_label
                    or year_pattern.match(release_label)
                    or date_pattern.match(release_label)
                    or time_pattern.match(release_label)
                    or release_label == "Upcoming"
                ):
                    release_label = ""
            release_dt = current_date.replace(
                hour=release_time.hour,
                minute=release_time.minute,
            )
            if release_label:
                entries.append((release_dt, release_label))
        index += 1

    return entries


def fetch_sp_global_us_pmi_windows(now_utc: datetime, horizon_days: int) -> list[dict[str, str]]:
    response = requests.get(SP_GLOBAL_PMI_CALENDAR_URL, timeout=30)
    response.raise_for_status()

    end_cutoff = now_utc + timedelta(days=horizon_days)
    windows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    config_by_source = {config.source_label: config for config in US_PMI_WINDOWS}

    for release_dt_utc, source_label in _extract_sp_global_calendar_entries(response.text):
        if release_dt_utc < now_utc or release_dt_utc > end_cutoff:
            continue
        config = config_by_source.get(source_label)
        if config is None:
            continue
        release_dt_et = release_dt_utc.astimezone(ET)
        reason = config.reason_template.format(
            source_label=source_label,
            date_label=release_dt_utc.strftime("%B %-d, %Y"),
            release_label=release_dt_utc.strftime("%H:%M"),
        )
        for event_name in config.event_names:
            dedupe_key = (event_name, release_dt_utc.isoformat())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            windows.append(
                _format_custom_window(
                    start_et=release_dt_et,
                    name=event_name,
                    start_pad_minutes=config.start_pad_minutes,
                    end_pad_minutes=config.end_pad_minutes,
                    reason=reason,
                )
            )

    return windows


def _extract_ism_release_dates(raw_html: str, target_year: int) -> list[dict[str, datetime]]:
    section_match = re.search(
        rf"<h3>\s*{target_year}\s+ISM PMI.*?</h3>\s*<table.*?<tbody>(.*?)</tbody>",
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not section_match:
        return []

    rows: list[dict[str, datetime]] = []
    for month_label, manufacturing_day, services_day in re.findall(
        r"<tr>\s*<th[^>]*scope=\"row\"[^>]*>(.*?)</th>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*</tr>",
        section_match.group(1),
        flags=re.IGNORECASE | re.DOTALL,
    ):
        cleaned_month = " ".join(re.sub(r"(?s)<[^>]+>", " ", month_label).split())
        if not cleaned_month.endswith(str(target_year)):
            continue
        month_text, _ = cleaned_month.rsplit(" ", 1)
        month_number = _month_number(month_text)
        manufacturing_day_value = int(re.sub(r"[^0-9]", "", manufacturing_day))
        services_day_value = int(re.sub(r"[^0-9]", "", services_day))
        rows.append(
            {
                "manufacturing": _release_datetime_from_day(
                    target_year,
                    month_number,
                    manufacturing_day_value,
                    "10:00",
                ),
                "services": _release_datetime_from_day(
                    target_year,
                    month_number,
                    services_day_value,
                    "10:00",
                ),
            }
        )

    return rows


def fetch_ism_pmi_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    raw_html = _fetch_text(
        ISM_RELEASE_CALENDAR_URL,
        timeout=30,
        allow_curl_fallback=True,
    )

    end_cutoff = now_et + timedelta(days=horizon_days)
    windows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for year in {now_et.year, end_cutoff.year}:
        for release_row in _extract_ism_release_dates(raw_html, year):
            for config in ISM_WINDOW_CONFIGS:
                release_dt = release_row[config["column_key"]]
                if release_dt < now_et or release_dt > end_cutoff:
                    continue
                dedupe_key = (config["name"], release_dt.isoformat())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                windows.append(
                    _format_custom_window(
                        start_et=release_dt,
                        name=config["name"],
                        start_pad_minutes=config["start_pad_minutes"],
                        end_pad_minutes=config["end_pad_minutes"],
                        reason=config["reason_template"].format(
                            report_label=config["report_label"],
                            date_label=release_dt.strftime("%B %-d, %Y"),
                        ),
                    )
                )

    return windows


def _build_windows_from_static_configs(
    configs: list[StaticReleaseConfig],
    now_et: datetime,
    horizon_days: int,
) -> list[dict[str, str]]:
    end_cutoff = now_et + timedelta(days=horizon_days)
    windows: list[dict[str, str]] = []

    for config in configs:
        for release_date in config.release_dates:
            release_dt = _release_datetime_from_date(release_date, config.release_time_et)
            if release_dt < now_et or release_dt > end_cutoff:
                continue
            windows.append(
                _format_custom_window(
                    start_et=release_dt,
                    name=config.name,
                    start_pad_minutes=config.start_pad_minutes,
                    end_pad_minutes=config.end_pad_minutes,
                    reason=config.reason_template.format(
                        name=config.name,
                        date_label=release_dt.strftime("%B %-d, %Y"),
                    ),
                )
            )

    return windows


def fetch_jolts_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    return _build_windows_from_static_configs(JOLTS_WINDOWS, now_et=now_et, horizon_days=horizon_days)


def fetch_michigan_sentiment_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    return _build_windows_from_static_configs(
        MICHIGAN_SENTIMENT_WINDOWS,
        now_et=now_et,
        horizon_days=horizon_days,
    )


def fetch_conference_board_consumer_confidence_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    end_cutoff = now_et + timedelta(days=horizon_days)
    windows: list[dict[str, str]] = []
    for year, month in _iter_month_starts(now_et, end_cutoff):
        release_day = _last_weekday_of_month(year, month, 1)
        release_dt = release_day.replace(hour=10, minute=0)
        if release_dt < now_et or release_dt > end_cutoff:
            continue
        windows.append(
            _format_custom_window(
                start_et=release_dt,
                name=CONFERENCE_BOARD_NAME,
                start_pad_minutes=15,
                end_pad_minutes=45,
                reason=(
                    "The Conference Board publishes the Consumer Confidence Index at "
                    "10:00 ET on the last Tuesday of each month."
                ),
            )
        )
    return windows


def _fetch_fomc_statement_dates(now_et: datetime, horizon_days: int) -> list[datetime]:
    response = requests.get(FOMC_CALENDAR_URL, timeout=30)
    response.raise_for_status()
    end_cutoff = now_et + timedelta(days=horizon_days)

    statement_dates: list[datetime] = []
    for year in {now_et.year, end_cutoff.year}:
        statement_dates.extend(_extract_fomc_statement_dates(response.text, year))

    return sorted(statement_dates)


def fetch_fomc_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    end_cutoff = now_et + timedelta(days=horizon_days)
    windows: list[dict[str, str]] = []
    for statement_dt in _fetch_fomc_statement_dates(now_et=now_et, horizon_days=horizon_days):
        if statement_dt < now_et or statement_dt > end_cutoff:
            continue
        windows.append(
            _build_window(
                release_dt=statement_dt,
                name="FOMC Statement / Press Conference",
                start_pad_minutes=45,
                end_pad_minutes=90,
                reason=(
                    "Federal Reserve FOMC statement and press conference window "
                    f"for {statement_dt.strftime('%B %-d, %Y')}."
                ),
            )
        )

    return windows


def fetch_fomc_minutes_windows(now_et: datetime, horizon_days: int) -> list[dict[str, str]]:
    end_cutoff = now_et + timedelta(days=horizon_days)
    windows: list[dict[str, str]] = []
    for statement_dt in _fetch_fomc_statement_dates(now_et=now_et, horizon_days=horizon_days):
        minutes_dt = statement_dt + timedelta(days=21)
        if minutes_dt < now_et or minutes_dt > end_cutoff:
            continue
        windows.append(
            _build_window(
                release_dt=minutes_dt,
                name=FOMC_MINUTES_NAME,
                start_pad_minutes=15,
                end_pad_minutes=45,
                reason=(
                    "Federal Reserve FOMC calendars note minutes are released three weeks "
                    "after the policy decision; this window tracks that scheduled minutes release."
                ),
            )
        )

    return windows


def build_windows(horizon_days: int, output_path: Path) -> dict[str, list[dict[str, str]]]:
    now_et = datetime.now(tz=ET)
    now_utc = now_et.astimezone(timezone.utc)

    try:
        windows = fetch_bls_windows(now_et=now_et, horizon_days=horizon_days)
    except Exception as exc:
        windows = _load_existing_windows(
            output_path=output_path,
            now_utc=now_utc,
            horizon_days=horizon_days,
            names={config.name for config in BLS_WINDOWS},
        )
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

    try:
        windows.extend(fetch_sp_global_us_pmi_windows(now_utc=now_utc, horizon_days=horizon_days))
    except Exception as exc:
        windows.extend(
            _load_existing_windows(
                output_path=output_path,
                now_utc=now_utc,
                horizon_days=horizon_days,
                names={
                    event_name
                    for config in US_PMI_WINDOWS
                    for event_name in config.event_names
                },
            )
        )
        warning = (
            "Warning: failed to refresh S&P Global PMI windows from the official PMI calendar; "
            f"preserving existing future US PMI windows instead. Error: {exc}"
        )
        print(warning, file=sys.stderr)

    try:
        windows.extend(fetch_ism_pmi_windows(now_et=now_et, horizon_days=horizon_days))
    except Exception as exc:
        windows.extend(
            _load_existing_windows(
                output_path=output_path,
                now_utc=now_utc,
                horizon_days=horizon_days,
                names={str(config["name"]) for config in ISM_WINDOW_CONFIGS},
            )
        )
        warning = (
            "Warning: failed to refresh ISM PMI windows from the official ISM calendar; "
            f"preserving existing future ISM PMI windows instead. Error: {exc}"
        )
        print(warning, file=sys.stderr)

    windows.extend(fetch_jolts_windows(now_et=now_et, horizon_days=horizon_days))
    windows.extend(fetch_michigan_sentiment_windows(now_et=now_et, horizon_days=horizon_days))
    windows.extend(
        fetch_conference_board_consumer_confidence_windows(
            now_et=now_et,
            horizon_days=horizon_days,
        )
    )

    try:
        windows.extend(fetch_fomc_windows(now_et=now_et, horizon_days=horizon_days))
        windows.extend(fetch_fomc_minutes_windows(now_et=now_et, horizon_days=horizon_days))
    except Exception as exc:
        windows.extend(
            _load_existing_windows(
                output_path=output_path,
                now_utc=now_utc,
                horizon_days=horizon_days,
                names={FOMC_MINUTES_NAME, "FOMC Statement / Press Conference"},
            )
        )
        warning = (
            "Warning: failed to refresh Federal Reserve meeting windows; "
            f"preserving existing future FOMC statement and minutes windows instead. Error: {exc}"
        )
        print(warning, file=sys.stderr)

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
