"""Timezone utilities for consistent timestamp handling.

Design principles:
- Store all timestamps in UTC in the database
- Display all timestamps in user's local time
- Use timezone-aware datetime objects internally

Database storage: UTC (stored as ISO 8601 text without 'Z' suffix for SQLite compatibility)
Internal representation: timezone-aware datetime (UTC)
Display: converted to local time
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def utc_now() -> datetime:
    """Get current time as timezone-aware UTC datetime.

    Use this instead of datetime.now() for all timestamp creation.
    """
    return datetime.now(timezone.utc)


def to_utc_string(dt: datetime) -> str:
    """Convert datetime to UTC ISO string for database storage.

    Args:
        dt: A datetime object (timezone-aware or naive).
            If naive, assumes it's already UTC.

    Returns:
        ISO 8601 string without timezone suffix (for SQLite compatibility).
    """
    if dt.tzinfo is not None:
        # Convert to UTC if timezone-aware
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_utc_timestamp(timestamp_str: Optional[str]) -> Optional[datetime]:
    """Parse a UTC timestamp string from the database.

    Args:
        timestamp_str: ISO 8601 format string from database (assumed UTC).

    Returns:
        Timezone-aware datetime in UTC, or None if input is None.
    """
    if timestamp_str is None:
        return None

    # Parse the timestamp
    try:
        # Handle both with and without microseconds
        if "." in timestamp_str:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        else:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Try fromisoformat as fallback
        dt = datetime.fromisoformat(timestamp_str)

    # Make it timezone-aware (UTC)
    return dt.replace(tzinfo=timezone.utc)


def to_local_time(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a UTC datetime to local time.

    Args:
        dt: Timezone-aware datetime in UTC.

    Returns:
        Timezone-aware datetime in local timezone, or None if input is None.
    """
    if dt is None:
        return None

    # Ensure it's UTC if timezone-aware, or assume UTC if naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to local time
    return dt.astimezone()


def format_local_time(dt: Optional[datetime], fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a UTC datetime as local time string for display.

    Args:
        dt: Timezone-aware datetime (assumed UTC).
        fmt: strftime format string.

    Returns:
        Formatted local time string, or '--' if dt is None.
    """
    if dt is None:
        return "--"

    local_dt = to_local_time(dt)
    return local_dt.strftime(fmt)


def format_local_iso(dt: Optional[datetime]) -> Optional[str]:
    """Format a UTC datetime as local ISO string for API responses.

    Args:
        dt: Timezone-aware datetime (assumed UTC).

    Returns:
        ISO 8601 string with timezone offset, or None if dt is None.
    """
    if dt is None:
        return None

    local_dt = to_local_time(dt)
    return local_dt.isoformat()


def extract_recording_time(file_path: str) -> Optional[datetime]:
    """Extract recording time from filename or file modification time.

    Tries to extract timestamp from filename patterns like:
    - recording-20240126-143045.mp3 (from recording daemon)
    - 20240126_143045_something.mp3
    - something_20240126-143045.mp3

    Falls back to file modification time if no pattern matches.

    Args:
        file_path: Path to the audio file.

    Returns:
        Timezone-aware datetime in UTC, or None if extraction fails.
    """
    path = Path(file_path)
    filename = path.stem  # filename without extension

    # Pattern 1: recording-YYYYMMDD-HHMMSS
    match = re.search(r'(\d{8})-(\d{6})', filename)
    if match:
        try:
            date_str = match.group(1)
            time_str = match.group(2)
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            # Assume local time, convert to UTC
            local_tz = datetime.now().astimezone().tzinfo
            dt = dt.replace(tzinfo=local_tz)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    # Pattern 2: YYYYMMDD_HHMMSS
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        try:
            date_str = match.group(1)
            time_str = match.group(2)
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            local_tz = datetime.now().astimezone().tzinfo
            dt = dt.replace(tzinfo=local_tz)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    # Pattern 3: YYYY-MM-DD_HH-MM-SS or similar with separators
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})[_T](\d{2})-(\d{2})-(\d{2})', filename)
    if match:
        try:
            dt_str = f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}{match.group(5)}{match.group(6)}"
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
            local_tz = datetime.now().astimezone().tzinfo
            dt = dt.replace(tzinfo=local_tz)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    # Fall back to file modification time
    try:
        if path.exists():
            mtime = os.path.getmtime(path)
            return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except (OSError, ValueError):
        pass

    return None
