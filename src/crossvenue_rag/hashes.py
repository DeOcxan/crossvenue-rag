"""The sha256 of a text: chunk ids, the index version, deduplication and the model cache."""

from __future__ import annotations

import hashlib


def text_sha256(text: str) -> str:
    """The hex sha256 of `text` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
