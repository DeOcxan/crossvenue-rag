"""Tests for `crossvenue_rag.hashes`: the sha256 of a text."""

from __future__ import annotations

import hashlib

from crossvenue_rag.hashes import text_sha256


def test_the_hash_is_the_hex_sha256_of_the_utf8_bytes():
    assert text_sha256("\u00fc") == hashlib.sha256("\u00fc".encode()).hexdigest()
    assert len(text_sha256("")) == 64
