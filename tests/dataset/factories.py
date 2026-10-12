"""Builders the dataset tests share, and other folders' tests of dataset records and queues."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crossvenue_rag import jsonl
from crossvenue_rag.dataset.records import QuestionRecord
from crossvenue_rag.models import Question, QuestionCategory

# v0.2.1-gold's category sizes, all 114 judged in round 1.
GOLD_SIZES: dict[QuestionCategory, int] = {
    "comparison": 50,
    "unanswerable": 20,
    "adversarial": 13,
    "fact": 8,
    "ambiguous": 6,
    "noisy": 6,
    "version_sensitive": 6,
    "multihop": 5,
}
# The same sizes keyed by plain category name, as `stratified_quotas` takes them.
SIZES: dict[str, int] = dict(GOLD_SIZES.items())


def question(i: int, category: QuestionCategory, *, judged: bool = True) -> Question:
    """Question `i` of `category`, judged in round 1 unless `judged` is False."""
    return Question(
        id=f"cv-{i:04d}",
        question=f"question number {i} about {category} and some more words",
        tier="gold",
        venues=["binance"],
        qtype="fact",
        category=category,
        gold_chunk_ids=["c1"] if judged and category != "unanswerable" else [],
        annotated_at="2026-09-24T12:00:00Z" if judged else "",
    )


def gold_like() -> list[Question]:
    """Questions of `GOLD_SIZES`, every one judged in round 1."""
    out, i = [], 1
    for category, size in GOLD_SIZES.items():
        for _ in range(size):
            out.append(question(i, category))
            i += 1
    return out


def record(**patch: Any) -> QuestionRecord:
    """A dataset record of one answerable bybit fact question, with `patch` applied."""
    base = {
        "id": "q1",
        "question": "How does Bybit report the remaining rate limit?",
        "tier": "gold",
        "venues": ["bybit"],
        "qtype": "fact",
        "category": "fact",
        "difficulty": "easy",
        "is_answerable": True,
        "expected_behavior": "answer",
    }
    return QuestionRecord(**{**base, **patch})


def save_records(path: str | Path, records: Sequence[QuestionRecord]) -> None:
    """Writes `records` as a dataset file, omitting an empty `venue_coverage`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in records:
        data = item.model_dump()
        if not data["venue_coverage"]:
            del data["venue_coverage"]  # as `models.Question.to_dict` does
        rows.append(data)
    jsonl.write(path, rows)
