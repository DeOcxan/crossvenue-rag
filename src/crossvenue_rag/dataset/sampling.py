"""The round-2 subsample for intra-rater agreement, stratified by category.

The sample (evals/methodology.md decision 7) is drawn from a seed and written to a
file before round 2, so the list the annotator works through is fixed in advance.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .. import formatting
from ..errors import UserError, require_file
from ..models import Question

ROUND2_SIZE = 25
ROUND2_SEED = 20260924
# A question is cited by its id and the first six words of its text.
CITATION_WORDS = 6


def stratified_quotas(sizes: dict[str, int], n: int, item: str = "question") -> dict[str, int]:
    """Items per category: proportional to size, at least one each.

    Largest-remainder rounding; ties go to the larger category, then to the
    category name, so the result depends on nothing but the input. `item` names
    what is sampled in the error a sample too large gives.

    Raises:
        UserError: `n` is below the number of non-empty categories, or above their total.
    """
    sizes = {c: s for c, s in sizes.items() if s > 0}
    total = sum(sizes.values())
    if n < len(sizes):
        raise UserError(
            f"a sample of {formatting.count(n)} cannot hold one of each of "
            f"{formatting.plural(len(sizes), 'category', 'categories')}",
            hint=f"ask for at least {formatting.count(len(sizes))}",
        )
    if n > total:
        raise UserError(
            f"a sample of {formatting.count(n)} exceeds the "
            f"{formatting.plural(total, f'eligible {item}')}",
            hint=f"ask for at most {formatting.count(total)}",
        )
    exact = {c: n * s / total for c, s in sizes.items()}
    quotas = {c: min(s, max(1, math.floor(exact[c]))) for c, s in sizes.items()}

    def by_remainder(c: str) -> tuple[float, int, str]:
        return (-(exact[c] - math.floor(exact[c])), -sizes[c], c)

    while sum(quotas.values()) < n:
        for c in sorted((c for c in sizes if quotas[c] < sizes[c]), key=by_remainder):
            if sum(quotas.values()) == n:
                break
            quotas[c] += 1
    while sum(quotas.values()) > n:
        # Only possible when the minimum of one pushed the total over n: the
        # category with the most to spare relative to its share gives one back.
        spare = [c for c in sizes if quotas[c] > 1]
        quotas[max(spare, key=lambda c: (quotas[c] - exact[c], sizes[c], c))] -= 1
    return quotas


def stratified_sample(
    questions: Sequence[Question], n: int = ROUND2_SIZE, seed: int = ROUND2_SEED
) -> list[Question]:
    """`n` judged questions, stratified by category; deterministic for a seed.

    Only questions round 1 judged are eligible (gold chunks, or recorded as
    "nothing in the pool answers this"): re-judging a question never judged
    measures nothing. Returned in the question file's order, which is the order
    they are labeled.
    """
    eligible = [q for q in questions if q.is_judged]
    by_category: dict[str, list[Question]] = defaultdict(list)
    for q in eligible:
        by_category[q.effective_category].append(q)
    quotas = stratified_quotas({c: len(qs) for c, qs in by_category.items()}, n)
    rng = random.Random(seed)
    chosen: set[str] = set()
    for category in sorted(by_category):
        ids = sorted(q.id for q in by_category[category])
        chosen.update(rng.sample(ids, quotas[category]))
    return [q for q in eligible if q.id in chosen]


def cited_words(question: Question) -> str:
    """The first `CITATION_WORDS` words of a question's text, as a citation of it quotes them."""
    return " ".join(question.question.split()[:CITATION_WORDS])


def write_sample_file(path: str | Path, sample: Sequence[Question], seed: int) -> None:
    """One question per line: id, category, first six words. First token is the id."""
    lines = [
        f"# Round-2 subsample: {formatting.plural(len(sample), 'question')}, "
        f"stratified by category, seed {formatting.own_form(seed)}.",
        "# Written by `crossvenue round-sample`; read by `crossvenue label --sample-file`.",
    ]
    for q in sample:
        lines.append(f'{q.id}  {q.effective_category:<18} "{cited_words(q)}"')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_sample_file(path: str | Path) -> list[str]:
    """Question ids: the first token of each non-blank line; `#` starts a comment.

    Raises:
        UserError: Nothing is at `path`, or a directory is. The file is not UTF-8.
    """
    hint = "pass a list of question ids, one per line"
    try:
        text = require_file(path, hint=hint).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UserError(f"{path} is not UTF-8 text: {exc.reason}", hint=hint) from exc
    ids: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids
