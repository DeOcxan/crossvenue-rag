"""Tests for `crossvenue_rag.clock`."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from crossvenue_rag.clock import utc_now


def test_the_stamp_is_utc_to_the_second_with_a_z():
    stamp = utc_now()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamp)
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert abs((datetime.now(UTC) - parsed).total_seconds()) < 5


def test_the_run_id_is_utc_to_the_second_in_the_results_directory_form():
    from crossvenue_rag.clock import run_id

    stamp = run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    assert abs((datetime.now(UTC) - parsed).total_seconds()) < 5


def test_a_measurement_time_is_utc_to_the_second_with_its_offset():
    from crossvenue_rag.clock import utc_now_with_offset

    stamp = utc_now_with_offset()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", stamp)
    assert abs((datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds()) < 5
