"""Choosing 105 questions out of a candidate pool against the target composition.

Every decision is recorded with a reason in `evals/selection/selection.jsonl`, so
the rejections are part of the output.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import formatting, jsonl
from ..clock import utc_now
from ..errors import UserError
from .questions import MIN_QUESTIONS_PER_CATEGORY
from .records import QuestionRecord

# evals/methodology.md decision 2. Sums to 105: the three constraints (comparison 50,
# unanswerable 15, nothing below `MIN_QUESTIONS_PER_CATEGORY`) give 95, and the chosen
# adversarial 12 and fact 8 add 10 above that minimum.
TARGETS: dict[str, int] = {
    "comparison": 50,
    "unanswerable": 15,
    "adversarial": 12,
    "fact": 8,
    "multihop": MIN_QUESTIONS_PER_CATEGORY,
    "ambiguous": MIN_QUESTIONS_PER_CATEGORY,
    "version_sensitive": MIN_QUESTIONS_PER_CATEGORY,
    "noisy": MIN_QUESTIONS_PER_CATEGORY,
}

# Thin categories first, then comparison, then the deep ones: the order follows the
# risk of running out. ambiguous, noisy and version_sensitive hold barely more
# candidates than their targets, so whether the pool can fill them is answered in
# the first twenty minutes rather than after an hour of comparison work.
# comparison, adversarial, fact and multihop all have double-digit headroom.
CATEGORY_ORDER: list[str] = [
    # thin: decided first, because the answer might be "generate more"
    "ambiguous",
    "noisy",
    "version_sensitive",
    "unanswerable",
    # the headline subgroup, while attention is still good
    "comparison",
    # deep pools: genuine selection, no risk of exhaustion
    "adversarial",
    "multihop",
    "fact",
]

REJECTION_REASONS: dict[str, str] = {
    "g": "not grounded",
    "p": "paraphrased heading",
    "d": "duplicate information need",
    "u": "unrealistic phrasing",
    "c": "wrong category",
    "o": "other",
}

SELECTION_FILENAME = "selection/selection.jsonl"


def target_size() -> int:
    """Questions the category targets add up to (`TARGETS`): the registered set size."""
    return sum(TARGETS.values())


@dataclass(slots=True)
class Decision:
    """One accept/reject, appended as it is made so Ctrl-C loses nothing."""

    id: str
    decision: str  # "accept" | "reject" | "skip"
    reason: str
    ts: str

    @classmethod
    def now(cls, question_id: str, decision: str, reason: str = "") -> Decision:
        """A decision on the question, stamped with `utc_now()`."""
        return cls(
            id=question_id,
            decision=decision,
            reason=reason,
            ts=utc_now(),
        )


def selection_path(questions_path: str | Path) -> Path:
    """Sidecar next to the question file this selection will produce."""
    return Path(questions_path).parent / SELECTION_FILENAME


def load_decisions(path: str | Path) -> dict[str, Decision]:
    """The last decision recorded per question, replayed from the log.

    The log is append-only and the last line per question wins, so a changed
    decision is another line and the earlier one stays in the log.

    Raises:
        UserError: A line is not UTF-8, not JSON, or not a decision record (`id`, `decision`,
            `reason`, `ts`).
    """
    path = Path(path)
    if not path.exists():
        return {}
    decisions: dict[str, Decision] = {}
    for number, row in jsonl.iter_rows(path):
        try:
            decision = Decision(**row)
        except TypeError as exc:
            raise UserError(
                f"{formatting.location(path, number)}: not a decision record ({exc})",
                hint="restore the selection log from its last good copy",
            ) from exc
        decisions[decision.id] = decision
    return decisions


def append_decision(path: str | Path, decision: Decision) -> None:
    """Appends one decision to the selection log, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonl.append(path, [asdict(decision)])


@dataclass(slots=True)
class Progress:
    """Where selection stands against the targets.

    `remaining` is everything available to accept (undecided plus skipped).
    `unseen` is the subset never shown, which is the one that answers
    "is the headroom running out while I watch".
    """

    accepted: Counter[str]
    remaining: Counter[str]
    unseen: Counter[str]
    answerable_comparison: int

    def shortfall(self, category: str) -> int:
        """How many more accepts the category needs to reach its target."""
        return max(0, TARGETS.get(category, 0) - self.accepted[category])

    def reachable(self, category: str) -> bool:
        """Whether the accepted and remaining candidates can reach the target."""
        return self.accepted[category] + self.remaining[category] >= TARGETS.get(category, 0)

    @property
    def total_accepted(self) -> int:
        """Questions accepted across all categories."""
        return sum(self.accepted.values())

    @property
    def total_target(self) -> int:
        """The number of questions the targets add up to, `target_size`."""
        return target_size()


def progress(candidates: Sequence[QuestionRecord], decisions: dict[str, Decision]) -> Progress:
    """Where selection stands against the targets, per category."""
    accepted: Counter[str] = Counter()
    remaining: Counter[str] = Counter()
    unseen: Counter[str] = Counter()
    answerable_comparison = 0

    for record in candidates:
        state = decisions.get(record.id)
        if state is None:
            remaining[record.category] += 1
            unseen[record.category] += 1
        elif state.decision == "skip":
            remaining[record.category] += 1
        elif state.decision == "accept":
            accepted[record.category] += 1
            if record.is_comparison and record.is_answerable:
                answerable_comparison += 1
    return Progress(accepted, remaining, unseen, answerable_comparison)


def queue(
    candidates: Sequence[QuestionRecord],
    decisions: dict[str, Decision],
    categories: Sequence[str] | None = None,
    *,
    include_skipped: bool = False,
) -> Iterator[QuestionRecord]:
    """Undecided candidates, grouped by category in `CATEGORY_ORDER`.

    Skipped questions are held back by default: a skip defers the question, and
    re-offering it immediately turns the skip key into a no-op.
    """
    wanted = list(categories) if categories else CATEGORY_ORDER
    by_category: dict[str, list[QuestionRecord]] = {}
    for record in candidates:
        by_category.setdefault(record.category, []).append(record)

    for category in wanted:
        for record in by_category.get(category, []):
            state = decisions.get(record.id)
            if state is None or (include_skipped and state.decision == "skip"):
                yield record


@dataclass(slots=True)
class CompositionReport:
    """How the accepted questions compare with the target composition."""

    accepted: Counter[str]
    missing: dict[str, int]
    answerable_comparison: int
    unexpected: list[str]

    @property
    def ok(self) -> bool:
        """Whether every target is met and every accepted question is a candidate."""
        return (
            not self.missing
            and not self.unexpected
            and self.answerable_comparison >= TARGETS["comparison"]
        )

    def problems(self) -> list[str]:
        """One line per unmet target or unexpected question, empty when `ok`."""
        out = [
            f"{category}: {formatting.count(self.accepted[category])} accepted, target "
            f"{formatting.count(TARGETS[category])} (short by {formatting.count(short)})"
            for category, short in sorted(self.missing.items())
        ]
        if self.answerable_comparison < TARGETS["comparison"]:
            out.append(
                f"comparison: only {formatting.count(self.answerable_comparison)} of the accepted "
                f"items are answerable, target {formatting.count(TARGETS['comparison'])}. "
                "Unanswerable questions are "
                f"excluded from every recall average, so they do not count toward the "
                f"subgroup that carries the headline claim"
            )
        out += [f"accepted question not in the candidate pool: {i}" for i in self.unexpected]
        return out


def check_composition(
    candidates: Sequence[QuestionRecord], decisions: dict[str, Decision]
) -> CompositionReport:
    """The accepted set against `TARGETS` (evals/methodology.md decision 2).

    Unanswerable questions carry no gold chunks and are excluded from every recall
    average, so only answerable comparison items count toward the comparison target.
    """
    by_id = {r.id: r for r in candidates}
    accepted: Counter[str] = Counter()
    answerable_comparison = 0
    unexpected: list[str] = []

    for question_id, state in decisions.items():
        if state.decision != "accept":
            continue
        record = by_id.get(question_id)
        if record is None:
            unexpected.append(question_id)
            continue
        accepted[record.category] += 1
        if record.is_comparison and record.is_answerable:
            answerable_comparison += 1

    missing = {
        category: target - accepted[category]
        for category, target in TARGETS.items()
        if accepted[category] < target
    }
    return CompositionReport(accepted, missing, answerable_comparison, sorted(unexpected))


def accepted_records(
    candidates: Sequence[QuestionRecord], decisions: dict[str, Decision]
) -> list[QuestionRecord]:
    """Accepted candidates, in candidate-file order so the output is stable."""
    keep = {i for i, d in decisions.items() if d.decision == "accept"}
    return [r for r in candidates if r.id in keep]


def rejection_summary(
    candidates: Sequence[QuestionRecord], decisions: dict[str, Decision]
) -> Counter[str]:
    """Reject reasons, for the CHANGELOG entry of the resulting gold set."""
    known = {r.id for r in candidates}
    return Counter(
        d.reason or "(unspecified)"
        for i, d in decisions.items()
        if d.decision == "reject" and i in known
    )
