"""What the decision records (docs/adr/NNNN-*.md) claim, one module per record."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from ...check import Doc
from ...sources import Sources
from . import (
    adr_0001,
    adr_0002,
    adr_0003,
    adr_0004,
    adr_0005,
    adr_0006,
    adr_0007,
    adr_0008,
    adr_0009,
    adr_0010,
    adr_0011,
    adr_0012,
    adr_0013,
)

RECORDS: dict[str, Callable[[Doc, Sources], None]] = {
    "0001": adr_0001.no_framework_record,
    "0002": adr_0002.structural_chunking_record,
    "0003": adr_0003.exact_deduplication_record,
    "0004": adr_0004.index_version_record,
    "0005": adr_0005.labeling_pool_record,
    "0006": adr_0006.single_annotator_record,
    "0007": adr_0007.venue_split_record,
    "0008": adr_0008.package_structure_record,
    "0009": adr_0009.generators_record,
    "0010": adr_0010.judge_record,
    "0011": adr_0011.preregistration_record,
    "0012": adr_0012.deferred_corpus_changes_record,
    "0013": adr_0013.generator_prompt_v2_record,
}


def record_number(path: str) -> str | None:
    """The four-digit number a record's file name starts with (`0005`); None if none."""
    number = re.match(r"(\d{4})-", Path(path).name)
    return number.group(1) if number else None


def adr(doc: Doc, src: Sources) -> None:
    """docs/adr/NNNN-*.md: the record's number in its heading, then the claims of its module.

    The module is the one `RECORDS` holds for that number; a record without one has only
    its number checked.
    """
    number = record_number(doc.path)
    if number is None:
        return
    doc.definition(f"# {number}.", reason="the record's number")
    record = RECORDS.get(number)
    if record is not None:
        record(doc, src)
