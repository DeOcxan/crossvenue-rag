"""The package's clock: the timestamps it writes and reads, and the seconds in a day.

Every timestamp it writes is UTC to the second.
"""

from __future__ import annotations

from datetime import UTC, datetime

SECONDS_PER_DAY = 86_400


def utc_now() -> str:
    """The time of a record: `2026-09-25T01:23:14Z`."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_time(value: str) -> datetime:
    """The time a record's stamp names, `2026-09-25T01:23:14Z` or with an offset."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def utc_now_with_offset() -> str:
    """The time of a measurement with its UTC offset: `2026-09-25T22:58:50+00:00`.

    The form every committed machine.json holds.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_id() -> str:
    """The name of a run directory under `results/`: UTC time to the second, `20260925T012314Z`."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
