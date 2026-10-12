"""Prompt v1 against prompt v2 on the same questions (methodology section 9a).

`compare_prompts` tests H-v2 on judged records; `side_effects` checks the
partial answers from the records alone, with no judge. `prompt_run` writes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ConfigDict

from ..errors import UserError
from ..evaluation.paired import compare_pair, holm_by_family, pair_tests
from ..metric_direction import higher_is_better
from ..run_files import (
    CsvRow,
    CsvTable,
    OptionalFloat,
    read_table,
)
from .records import AnswerRecord
from .schemas import is_absence_claim, outcome

COMPARISON_FILE = "comparison.csv"


class PromptComparison(CsvRow):
    """One row of comparison.csv: prompt v1 against v2 on one metric in one configuration.

    `low` and `high` bound the 95% interval of `diff`; `holm_p` is set once Holm
    runs across configurations within the metric. `read_comparison` reads every
    format of the file:

    - the prompts run 20260926T004851Z_prompts: without `higher_is_better`,
      which reads as True;
    - every other run: every column.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: str
    metric: str
    n: int
    v1: float
    v2: float
    diff: float  # v2 - v1
    low: float
    high: float
    better: int  # questions where v2 is better than v1, in the metric's direction
    worse: int
    p: float  # McNemar for wrong refusals, randomization otherwise
    test: str
    holm_p: OptionalFloat = None
    higher_is_better: bool = True


def read_comparison(run_dir: Path) -> CsvTable[PromptComparison]:
    """comparison.csv of a prompts run, in any format `PromptComparison` lists."""
    return read_table(run_dir / COMPARISON_FILE, PromptComparison)


@dataclass(frozen=True)
class SideEffect:
    """One line of side_effects.jsonl: a partial answer's claims that contradict it.

    Claims are numbered from 1.
    """

    answer_id: str
    config: str
    category: str
    missing: list[str]  # the venues the answer lists as missing, lower case
    claims: int
    claims_on_missing_venue: list[int]
    absence_claims_on_missing_venue: list[int]
    claims_citing_only_other_venues: list[int]


def _by_key(records: Sequence[AnswerRecord]) -> dict[tuple[str, str], AnswerRecord]:
    return {(r.config, r.question_id): r for r in records if r.generation is not None}


def compare_prompts(
    v1: Sequence[AnswerRecord],
    v2: Sequence[AnswerRecord],
    configs: Sequence[str],
    resamples: int,
    seed: int,
) -> list[PromptComparison]:
    """H-v2 per configuration: v2 against v1, paired on questions.

    - wrong refusals: `refused` (under the outcome rule) on a question whose
      `expected_behavior` is answer or ask_clarification; exact McNemar.
    - venue_attribution_error: the judge's metric, on questions where both
      versions' answers have claims; 95% paired bootstrap and a two-sided
      paired randomization test.

    Each metric is Holm-adjusted across the configurations. Faithfulness is not
    compared: its judge dimension failed calibration (methodology section 9).

    Raises:
        UserError: v1 and v2 share no answered question of a configuration that expects
            an answer or a clarifying question; the message names the configuration.
    """
    a, b = _by_key(v1), _by_key(v2)
    rows: list[PromptComparison] = []
    for config in configs:
        pairs = _config_pairs(a, b, config)
        xs, ys = _wrong_refusals(pairs)
        if not xs:
            raise UserError(
                f"{config}: v1 and v2 share no answered question that expects an answer or "
                "a clarifying question, so its wrong refusals cannot be compared",
                hint="compare two judged runs of the same questions and configurations",
            )
        rows.append(_comparison(config, "wrong_refusal", xs, ys, resamples, seed))
        xs, ys = _attribution_errors(pairs)
        if xs:
            rows.append(_comparison(config, "venue_attribution_error", xs, ys, resamples, seed))
    return _adjust(rows)


Pairs = Sequence[tuple[AnswerRecord, AnswerRecord]]


def _config_pairs(
    a: Mapping[tuple[str, str], AnswerRecord],
    b: Mapping[tuple[str, str], AnswerRecord],
    config: str,
) -> Pairs:
    """The (v1, v2) records of one configuration on the questions both answered."""
    return [(r1, b[key]) for key, r1 in sorted(a.items()) if key[0] == config and key in b]


def _adjust(rows: Sequence[PromptComparison]) -> list[PromptComparison]:
    """The rows with Holm within each metric, across the configurations."""
    adjusted = holm_by_family(rows, lambda r: r.metric, lambda r: r.p)
    return [
        row.model_copy(update={"holm_p": holm_p})
        for row, holm_p in zip(rows, adjusted, strict=True)
    ]


def _wrong_refusals(pairs: Pairs) -> tuple[list[float], list[float]]:
    """Refused (1) or not (0) under v1 and v2, over the questions that expect no refusal."""
    xs, ys = [], []
    for r1, r2 in pairs:
        if r1.expected_behavior == "refuse":
            continue
        xs.append(1.0 if outcome(r1.generated().answer) == "refused" else 0.0)
        ys.append(1.0 if outcome(r2.generated().answer) == "refused" else 0.0)
    return xs, ys


def _attribution_errors(pairs: Pairs) -> tuple[list[float], list[float]]:
    """venue_attribution_error under v1 and v2 where both versions' answers carry claims."""
    xs, ys = [], []
    for r1, r2 in pairs:
        if r1.judge is None or r2.judge is None:
            continue
        v1_err = r1.judge.metrics.venue_attribution_error
        v2_err = r2.judge.metrics.venue_attribution_error
        if v1_err is not None and v2_err is not None:
            xs.append(v1_err)
            ys.append(v2_err)
    return xs, ys


def _comparison(
    config: str, metric: str, xs: list[float], ys: list[float], resamples: int, seed: int
) -> PromptComparison:
    """v2 minus v1 on one metric: exact McNemar for wrong refusals, else randomization."""
    up = higher_is_better(metric)
    binary = metric == "wrong_refusal"
    # v2 minus v1: v2 is the pair's a.
    test = compare_pair(
        ys,
        xs,
        resamples=resamples,
        seed=seed,
        higher_is_better=up,
        tests=pair_tests(randomisation=not binary, mcnemar=binary),
    )
    p = test.mcnemar_p if binary else test.randomisation_p
    if p is None:
        raise ValueError(f"{config} {metric}: the test asked for returned no p-value")
    return PromptComparison(
        config=config,
        metric=metric,
        n=test.n,
        v1=test.mean_b,
        v2=test.mean_a,
        diff=test.boot.mean_diff,
        low=test.boot.low,
        high=test.boot.high,
        better=test.boot.better,
        worse=test.boot.worse,
        p=p,
        test="exact McNemar" if binary else "paired randomisation",
        higher_is_better=up,
    )


def side_effects(records: Sequence[AnswerRecord], venue_of: Mapping[str, str]) -> list[SideEffect]:
    """Per answer flagged partial, the claims that contradict it, from the records alone.

    A claim about a venue the answer lists in `missing` states facts the answer
    says the chunks lack; a claim whose cited chunks all belong to other venues
    is flagged as well. Each claim cites numbers of its record's `chunk_ids`, as
    `read_answers` ensures.

    Raises:
        ValueError: A partial answer cites a chunk `venue_of` lacks; the message names it.
    """
    out = []
    for r in records:
        gen = r.generation
        if gen is None or outcome(gen.answer) != "partial":
            continue
        answer = gen.answer
        missing = {m.venue.lower() for m in answer.missing}
        on_missing, foreign = [], []
        numbered = gen.numbered_chunk_ids()
        for i, claim in enumerate(answer.claims, start=1):
            venue = claim.venue.lower()
            ids = [numbered[n] for n in claim.chunks]
            if lacking := [c for c in ids if c not in venue_of]:
                raise ValueError(f"{r.answer_id} cites {lacking[0]}, which venue_of lacks")
            cited = {venue_of[c] for c in ids}
            if venue in missing:
                on_missing.append(i)
            if cited and venue not in cited:
                foreign.append(i)
        out.append(
            SideEffect(
                answer_id=r.answer_id,
                config=r.config,
                category=r.category,
                missing=sorted(missing),
                claims=len(answer.claims),
                claims_on_missing_venue=on_missing,
                absence_claims_on_missing_venue=[
                    i for i in on_missing if is_absence_claim(answer.claims[i - 1].text)
                ],
                claims_citing_only_other_venues=foreign,
            )
        )
    return out
