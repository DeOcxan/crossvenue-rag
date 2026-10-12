"""Tests for `crossvenue_rag.confined_paths`."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossvenue_rag.confined_paths import confined
from crossvenue_rag.errors import UserError


@pytest.mark.parametrize("part", ["", "  ", "..", ".", "a/..", "a/../..", "../sibling/file.md"])
def test_a_part_that_leads_to_the_root_itself_or_above_it_is_refused(tmp_path, part):
    with pytest.raises(UserError, match="venue") as refused:
        confined(tmp_path / "root", part, what="venue", hint="fix the venue")
    assert repr(part) in str(refused.value)
    assert str(tmp_path / "root") in str(refused.value)
    assert refused.value.hint == "fix the venue"


def test_an_absolute_part_is_refused(tmp_path):
    outside = tmp_path / "outside"
    with pytest.raises(UserError, match="is absolute"):
        confined(tmp_path / "root", str(outside), what="venue", hint="")


def test_an_absolute_part_inside_the_root_is_refused_and_the_error_names_the_root(tmp_path):
    root = tmp_path / "root"
    with pytest.raises(UserError, match="is absolute") as refused:
        confined(root, str(root / "binance"), what="venue", hint="fix the venue")
    assert str(root) in str(refused.value).removeprefix(f"venue {str(root / 'binance')!r}")


def test_a_symlink_under_the_root_that_points_outside_it_is_refused(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UserError, match=r"'linked/page\.md' names no path inside"):
        confined(root, "linked/page.md", what="page path", hint="")


def test_a_path_inside_the_root_is_returned_as_joined_not_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("data/raw")
    joined = confined(root, "binance/docs/./page.md", what="page path", hint="")
    assert joined == root / "binance/docs/./page.md"
    assert not joined.is_absolute()
    assert joined.relative_to(root).as_posix() == "binance/docs/page.md"


def test_a_control_character_is_refused_as_a_user_error(tmp_path: Path):
    from crossvenue_rag.confined_paths import plain_name_problem

    with pytest.raises(UserError, match="control character") as refused:
        confined(tmp_path, "a\x00b", what="venue", hint="h")
    assert str(tmp_path) in str(refused.value)
    assert plain_name_problem("a\x00b") == "holds a control character"
    assert plain_name_problem("x/y") == "holds a path separator"
    assert plain_name_problem("opus") is None


def test_a_directory_path_under_a_file_is_found(tmp_path: Path):
    from crossvenue_rag.confined_paths import file_on_path

    (tmp_path / "f.txt").write_text("x")
    assert file_on_path(tmp_path / "f.txt" / "runs") == tmp_path / "f.txt"
    assert file_on_path(tmp_path / "runs") is None
