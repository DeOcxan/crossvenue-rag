"""Tests for `crossvenue_rag.corpus.git_source`."""

from __future__ import annotations

import pytest

from crossvenue_rag.corpus.git_source import matches


def test_globs_match_zero_directories_and_exclusions_win():
    """`**/` has to match zero directories too: the Binance repo's Chinese
    translations are at the top level, where a literal `**/*_CN.md` misses them."""
    assert matches("docs/v5/order/create.md", ["docs/**/*.md"], []) is True
    assert matches("docs/assets/diagram.png", ["docs/**/*.md"], []) is False
    assert matches("rest-api_CN.md", ["*.md"], ["**/*_CN.md"]) is False
    assert matches("testnet/rest-api_CN.md", ["*.md"], ["**/*_CN.md"]) is False
    # An empty include means "everything not excluded"; exclude always wins.
    assert matches("anything.md", [], []) is True
    assert matches("anything.md", [], ["*.md"]) is False


def test_a_failed_clone_is_a_user_error_naming_the_git_command(tmp_path):
    import pytest

    from crossvenue_rag.corpus.git_source import clone_or_update
    from crossvenue_rag.corpus.sources import SourceSpec
    from crossvenue_rag.errors import UserError

    spec = SourceSpec(
        venue="bybit",
        venue_type="cex",
        kind="git",
        repo=(tmp_path / "no-such-repo").as_uri(),
        license="MIT",
    )
    with pytest.raises(UserError, match="git clone failed with exit status") as failed:
        clone_or_update(spec, tmp_path / "raw" / "bybit")
    assert "`paths.sources`" in failed.value.hint


def _raw_with_a_sibling(tmp_path):
    """data/raw holding another venue's file, next to data/outside holding a file."""
    raw, outside = tmp_path / "data" / "raw", tmp_path / "data" / "outside"
    (raw / "binance").mkdir(parents=True)
    (raw / "binance" / "keep.md").write_text("binance", encoding="utf-8")
    outside.mkdir()
    (outside / "keep.md").write_text("outside", encoding="utf-8")
    (raw / "linked").symlink_to(outside, target_is_directory=True)
    return raw, outside


# Each venue the sources file refuses, and a symlink it does not check, as `fetch` would join them;
# ABSOLUTE stands for the absolute path of data/outside.
UNSAFE_VENUES = ["", "   ", "..", "a/../..", "ABSOLUTE", "linked"]


@pytest.mark.parametrize("venue", UNSAFE_VENUES)
def test_a_venue_outside_the_raw_directory_is_refused_before_anything_is_deleted(tmp_path, venue):
    from crossvenue_rag.corpus.git_source import fetch_git_source
    from crossvenue_rag.corpus.sources import SourceSpec
    from crossvenue_rag.errors import UserError

    raw, outside = _raw_with_a_sibling(tmp_path)
    spec = SourceSpec(
        venue="bybit",
        venue_type="cex",
        kind="git",
        repo=(tmp_path / "no-such-repo").as_uri(),
        license="MIT",
    )
    # model_copy skips validation, as a spec that got past the sources file would.
    unsafe = spec.model_copy(update={"venue": str(outside) if venue == "ABSOLUTE" else venue})
    for commit in (None, "0" * 40):
        with pytest.raises(UserError, match="names no path") as refused:
            fetch_git_source(unsafe, raw, commit)
        assert "`paths.sources`" in refused.value.hint
    assert (raw / "binance" / "keep.md").read_text(encoding="utf-8") == "binance"
    assert (outside / "keep.md").read_text(encoding="utf-8") == "outside"
