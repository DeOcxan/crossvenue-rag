"""The literature evals/methodology.md cites, as docs/references.yaml records it."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from ...yaml_files import read_model
from ..sources import Sources
from .memo import per_sources

REFERENCES_FILE = "docs/references.yaml"
# The heading of methodology.md's reference list, and the heading after its last section.
REFERENCES_HEADING = "## References"
NOT_REFERENCES_HEADING = "### Not verified / left open"


class Reference(BaseModel):
    """One item of docs/references.yaml: the list item as written, under its section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section: str
    entry: str


class ReferenceList(BaseModel):
    """docs/references.yaml: the items of methodology.md's reference list, in order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    references: list[Reference]


@per_sources
def references(src: Sources) -> dict[str, list[str]]:
    """The entries of docs/references.yaml by section, both in the file's order."""
    hint = "write docs/references.yaml as a `references` list of `section` and `entry` items"
    by_section: dict[str, list[str]] = {}
    for item in read_model(src.path(REFERENCES_FILE), ReferenceList, hint=hint).references:
        by_section.setdefault(item.section, []).append(item.entry)
    return by_section


def written_sections(text: str) -> list[str]:
    """The section headings of a document's reference list, in order, without `### `.

    Raises:
        ValueError: `text` has no `REFERENCES_HEADING` line.
    """
    start = text.index(f"\n{REFERENCES_HEADING}\n")
    end = text.find(f"\n{NOT_REFERENCES_HEADING}\n", start)
    return re.findall(r"^### (.+)$", text[start : end if end >= 0 else None], flags=re.M)
