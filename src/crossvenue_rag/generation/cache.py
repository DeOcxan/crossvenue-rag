"""On-disk cache of model responses, keyed by what determines the response.

The key is a hash of the provider, the exact model id, the parameters, the
prompt, the output schema and a replicate number, which makes a deliberate repeat
(the judge's self-consistency run) a different entry. A re-run of the same
configuration reads the cache and costs nothing for every call that succeeded; a reply
that failed validation is not cached. Every result records the key of the entry it came
from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

from ..hashes import text_sha256
from ..run_files import read_json_model
from .records import CallRecord


def cache_key(request: dict[str, Any]) -> str:
    """sha256 of the request, canonically serialized. No secret ever enters it."""
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return text_sha256(canonical)


# Entries are spread over directories named by the first hex digits of their key.
SHARD_DIGITS = 2


class CacheEntry(BaseModel):
    """A cached model response: the reply's JSON object and the record of its call, marked cached.

    The entry's `request` is saved for a reader of the file and not read back.
    """

    model_config = ConfigDict(frozen=True)

    content: dict[str, JsonValue]
    record: CallRecord

    @field_validator("record", mode="before")
    @classmethod
    def _cached(cls, saved: object) -> object:
        return CallRecord.from_cache(saved) if isinstance(saved, dict) else saved


class DiskCache:
    """Model responses stored as one JSON file per cache key under a directory."""

    def __init__(self, directory: str | Path) -> None:
        """A cache rooted at `directory`, created on the first write."""
        self.directory = Path(directory)

    def path(self, key: str) -> Path:
        """The file that holds the entry for `key`, whether or not it exists."""
        return self.directory / key[:SHARD_DIGITS] / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        """The cached entry for `key`, or None on a miss."""
        path = self.path(key)
        if not path.exists():
            return None
        return read_entry(path)

    def put(self, key: str, entry: dict[str, Any]) -> Path:
        """Writes `entry` for `key` atomically and returns its file."""
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        return path


def read_entry(path: Path) -> CacheEntry:
    """The cache entry stored in `path` (`run_files.read_json_model`)."""
    return read_json_model(path, CacheEntry, hint=f"delete {path} and re-run")
