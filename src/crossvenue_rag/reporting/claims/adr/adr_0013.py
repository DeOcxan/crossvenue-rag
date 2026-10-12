"""What docs/adr/0013-generator-prompt-v2.md claims: Gemini's refusals under prompt v1."""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import answers
from ...sources import Sources
from ..comparisons import comparison_questions


def generator_prompt_v2_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0013-generator-prompt-v2.md."""
    doc.check(
        "Gemini with venue_split refused {} of the {} comparison questions, and in {} of them",
        formatting.count(answers.comparison_refusals(src, answers.refusing_config(src))),
        comparison_questions(src),
        formatting.count(len(answers.refusals_with_every_venue(src))),
    )
