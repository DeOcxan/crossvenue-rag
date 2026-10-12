"""Mutation test of the documents' checks: a changed number must not pass.

A covered number token is changed in a copy of its document's text, and the
document's claims and audit must then report a problem the unchanged text does
not have. `make test` runs a fixed sample stratified by the kind of check that
covers the token; the `slow` test, which `make check` and `make reproduce` run,
changes 500 sampled tokens, every run reference and every number of
methodology.md's reference list.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.patterns import REF, TOKEN
from crossvenue_rag.reporting.reproduce import METHODOLOGY, document_specs
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import needs_local_corpus

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260927
FULL_SAMPLE = 500
PER_KIND = 8
REFERENCE_SAMPLE = 5
KINDS = ("check", "table", "claim", "given", "definition", "ref", "unsaved")
TESTED_KINDS = ("check", "table", "claim", "given", "definition")
# A list or section number ("3. ", "## 9a. "): numbering, which a pattern matches and no
# claim computes.
NUMBERING = re.compile(r"\s*(?:#{2,3} )?\d+[a-z]?\. ")
WORD_SWAP = {"two": "three", "three": "four", "four": "five", "five": "six", "six": "seven",
             "seven": "eight", "eight": "nine", "nine": "ten", "ten": "eleven",
             "eleven": "twelve", "twelve": "two", "half": "twice", "twice": "half",
             "tenfold": "twice"}  # fmt: skip

# The test runs on the committed runs and the local corpus and index.
pytestmark = needs_local_corpus

Mutation = tuple[str, int, int, str, str]  # document, start, end, text before, text after


def stand_in_count(*, every: bool) -> int:
    """The README's test counter here: counting the tests would run pytest inside pytest."""
    return 0


def changed(token: str) -> str:
    """The token with its last digit moved on by one, or its number word swapped."""
    if token.lower() in WORD_SWAP:
        word = WORD_SWAP[token.lower()]
        return word.capitalize() if token[0].isupper() else word
    i = max(k for k, ch in enumerate(token) if ch.isdigit())
    return token[:i] + str((int(token[i]) + 1) % 10) + token[i + 1 :]


def is_numbering(text: str, start: int) -> bool:
    """Whether the token at `start` is its line's list or section number."""
    numbering = NUMBERING.match(text, text.rfind("\n", 0, start) + 1)
    return numbering is not None and start < numbering.end()


def covering_kinds(path: str, text: str, src: Sources) -> list[tuple[str, int, int]]:
    """(kind, start, end) of every span a check of the document covers, in call order."""
    spans: list[tuple[str, int, int]] = []
    originals = {kind: getattr(Doc, kind) for kind in KINDS}

    def recording(kind: str) -> Callable[..., object]:
        def method(self: Doc, *args: object, **kwargs: object) -> object:
            covered, refs = len(self.covered_spans()), len(self.ref_spans())
            out = originals[kind](self, *args, **kwargs)
            added = self.covered_spans()[covered:] + self.ref_spans()[refs:]
            spans.extend((kind, a, b) for a, b in added)
            return out

        return method

    with pytest.MonkeyPatch.context() as patch:
        for kind in KINDS:
            patch.setattr(Doc, kind, recording(kind))
        document_specs(ROOT, stand_in_count)[path](Doc.from_text(path, text), src)
    return spans


def tokens_by_kind(path: str, text: str, src: Sources) -> Iterator[tuple[str, int, int, str]]:
    """(kind, start, end, token) of every covered number, the kind of the first check
    covering it; run references as kind `ref`."""
    spans = covering_kinds(path, text, src)
    masked = Doc.from_text(path, text)
    for m in TOKEN.finditer(text):
        if masked.is_masked(m.start()) or is_numbering(text, m.start()):
            continue
        kind = next((k for k, a, b in spans if a <= m.start() < b), None)
        if kind is not None:
            yield kind, m.start(), m.end(), m.group()
    for m in REF.finditer(text):
        yield "ref", m.start(), m.end(), m.group()


def reference_list_tokens(text: str) -> list[tuple[int, int, str]]:
    """Every number of methodology.md's reference list."""
    first = text.index("\n## References")
    end = text.index("### Not verified", first)
    return [(m.start(), m.end(), m.group()) for m in TOKEN.finditer(text, first, end)]


def changed_reference(token: str) -> str:
    """A run reference with one character of its identifier changed."""
    start = token.rfind("/") + 1
    i = next(k for k in range(start, len(token)) if token[k] in "0123456789abcdef")
    return token[:i] + ("1" if token[i] == "0" else "0") + token[i + 1 :]


def problems(path: str, text: str, src: Sources) -> set[str]:
    """The problems the document's claims and audit report on `text`."""
    doc = Doc.from_text(path, text)
    document_specs(ROOT, stand_in_count)[path](doc, src)
    doc.audit()
    return {str(p) for p in doc.problems}


def passing(mutations: list[Mutation], texts: dict[str, str], src: Sources) -> list[str]:
    """The mutations that no check reports: no problem beyond the unchanged text's.

    With the stand-in test count the README's counts differ in every run, which
    is why a mutation must add a problem rather than cause any.
    """
    unchanged = {path: problems(path, text, src) for path, text in texts.items()}
    missed = []
    for path, start, end, old, new in mutations:
        text = texts[path]
        if not problems(path, text[:start] + new + text[end:], src) - unchanged[path]:
            missed.append(f"{path}:{text.count(chr(10), 0, start) + 1}: {old!r} -> {new!r}")
    return missed


def candidates(src: Sources) -> tuple[dict[str, str], list[tuple[str, str, int, int, str]]]:
    """The documents' texts, and (kind, document, start, end, token) of every mutable token.

    List and section numbers are left out, as a pattern matches any numbering. So are the
    numbers a document declares unsaved: no run holds them, so any number is accepted.
    """
    texts = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in document_specs(ROOT, stand_in_count)
    }
    found = [
        (kind, path, start, end, token)
        for path, text in texts.items()
        for kind, start, end, token in tokens_by_kind(path, text, src)
        if kind != "unsaved"
    ]
    return texts, found


def mutation(kind: str, path: str, start: int, end: int, token: str) -> Mutation:
    new = changed_reference(token) if kind == "ref" else changed(token)
    return path, start, end, token, new


def test_a_sample_of_changed_numbers_fails_for_every_kind_of_check():
    src = Sources(ROOT)
    texts, found = candidates(src)
    rng = random.Random(SEED)
    chosen = []
    for kind in (*TESTED_KINDS, "ref"):
        of_kind = [c for c in found if c[0] == kind]
        chosen += rng.sample(of_kind, min(PER_KIND, len(of_kind)))
    references = reference_list_tokens(texts[METHODOLOGY])
    chosen += [("definition", METHODOLOGY, *t) for t in rng.sample(references, REFERENCE_SAMPLE)]
    assert {c[0] for c in chosen} == {*TESTED_KINDS, "ref"}
    assert passing([mutation(*c) for c in chosen], texts, src) == []


@pytest.mark.slow
def test_every_sampled_change_fails():
    src = Sources(ROOT)
    texts, found = candidates(src)
    tokens = [c for c in found if c[0] != "ref"]
    chosen = random.Random(SEED).sample(tokens, FULL_SAMPLE) + [c for c in found if c[0] == "ref"]
    chosen += [("definition", METHODOLOGY, *t) for t in reference_list_tokens(texts[METHODOLOGY])]
    missed = passing([mutation(*c) for c in chosen], texts, src)
    assert missed == [], f"{len(chosen) - len(missed)} of {len(chosen)} caught"
