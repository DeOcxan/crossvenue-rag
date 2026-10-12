"""Judge selection from the calibration agreements (methodology section 9)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import formatting
from ..config import Config
from ..config_generation import CalibrationDimension
from ..errors import UserError
from ..stats import mean
from .calibration_file import (
    STATISTICS,
    Calibration,
    Candidate,
    DimensionStatistics,
    human_comparison,
    judge_comparison,
    replicate_comparison,
)
from .grader_agreement import agreement_from_pairs, agreement_intervals, paired_verdicts
from .judge_sample import RATINGS_PATH, human_ratings, judgements_path, load_judgements
from .records import ANSWERS_FILE, read_answers, read_judgements, read_ratings
from .runs import require_answer_run
from .schemas import Judgement

# The agreement statistic of each calibrated dimension; its keys are `CalibrationDimension`.
STATISTIC: dict[CalibrationDimension, str] = {
    "supported": "kappa",
    "venue_correct": "kappa",
    "behaviour_correct": "kappa",
    "coverage": "spearman",
}


def agreement_record(
    first: Mapping[str, Judgement],
    second: Mapping[str, Judgement],
    resamples: int,
    seed: int,
    exclude_behaviour: Sequence[str] = (),
) -> dict[str, DimensionStatistics]:
    """Per dimension: n, point estimates and 95% bootstrap intervals over answers."""
    pairs = paired_verdicts(first, second, exclude_behaviour)
    intervals = agreement_intervals(pairs, resamples, seed)
    out: dict[str, DimensionStatistics] = {}
    for d in agreement_from_pairs(pairs):
        row: dict[str, object] = {"n": d.n, "answers": len(pairs)}
        for statistic in STATISTICS:
            value = getattr(d, statistic)
            if value is None:
                continue
            row[statistic] = value
            ci = intervals.get(d.dimension, {}).get(statistic)
            if ci is not None:
                # No interval when no resample defines the statistic.
                if ci.low is not None and ci.high is not None:
                    row[f"{statistic}_ci"] = (ci.low, ci.high)
                row[f"{statistic}_undefined_resamples"] = ci.undefined
        out[d.dimension] = DimensionStatistics.model_validate(row)
    return out


@dataclass(frozen=True)
class Selection:
    """The judge section 9 selects, why, and every candidate's passes and fails."""

    candidates: dict[str, Candidate]
    selected: str
    reason: str


def _at_least(dims: Mapping[str, DimensionStatistics], dimension: str, statistic: str) -> float:
    """The dimension's `statistic`, or minus infinity where it is undefined."""
    value = dims[dimension].value(statistic)
    return float("-inf") if value is None else value


def select_judge(
    versus_human: Mapping[str, Mapping[str, DimensionStatistics]],
    thresholds: Mapping[CalibrationDimension, float],
    cost_per_answer: Mapping[str, float],
) -> Selection:
    """The judge methodology section 9 selects, with every candidate's passes and fails.

    The judge has the most dimensions at or above their threshold; ties are
    broken by agreement on venue_correct, then on coverage, then by lower cost.

    Raises:
        ValueError: `versus_human` names no candidate; `calibration_report` refuses an
            empty `generation.judges` before this is reached. Or a candidate has no
            statistics on a dimension of `thresholds`, venue_correct or coverage, or no
            cost in `cost_per_answer`; the message names it.
    """
    if not versus_human:
        raise ValueError("select_judge needs at least one candidate")
    needed = (*thresholds, "venue_correct", "coverage")
    for name, dims in versus_human.items():
        if lacking := [d for d in needed if d not in dims]:
            raise ValueError(f"{name} has no statistics on {lacking[0]}")
        if name not in cost_per_answer:
            raise ValueError(f"{name} has no cost per answer")
    candidates: dict[str, Candidate] = {}
    for name, dims in versus_human.items():
        passes = [d for d, t in thresholds.items() if _at_least(dims, d, STATISTIC[d]) >= t]
        candidates[name] = Candidate(
            passes=passes,
            fails=[d for d in thresholds if d not in passes],
            cost_per_answer_usd=cost_per_answer[name],
        )

    def key(name: str) -> tuple[float, ...]:
        dims = versus_human[name]
        return (
            len(candidates[name].passes),
            _at_least(dims, "venue_correct", STATISTIC["venue_correct"]),
            _at_least(dims, "coverage", STATISTIC["coverage"]),
            -cost_per_answer[name],
        )

    ranked = sorted(candidates, key=key, reverse=True)
    best = ranked[0]
    most = len(candidates[best].passes)
    tied = [j for j in ranked if len(candidates[j].passes) == most]
    if len(tied) == 1:
        reason = (
            f"{best} meets the threshold on {formatting.plural(most, 'dimension')}, "
            "more than any other candidate"
        )
    else:
        reason = (
            f"{', '.join(tied)} meet the threshold on {formatting.plural(most, 'dimension')} each; "
            f"{best} is chosen by the tie-break (venue_correct kappa, coverage Spearman, cost)"
        )
    return Selection(candidates, best, reason)


def _require_inputs(cfg: Config, run_dir: Path, ratings_path: str | Path, *, rated: bool) -> None:
    """Refuses a calibration with nothing to calibrate.

    Raises:
        UserError: `generation.judges` names no candidate,
            `generation.calibration_exclude_behaviour` names an answer the run does not
            have, or the ratings file holds no human rating (`rated` is False).
    """
    require_answer_run(run_dir)
    if not cfg.generation.judges:
        raise UserError(
            "generation.judges names no judge candidate",
            hint="add the candidates to generation.judges in the config, as "
            "configs/default.yaml does",
        )
    answer_ids = {r.answer_id for r in read_answers(run_dir / ANSWERS_FILE)}
    unknown = sorted(set(cfg.generation.calibration_exclude_behaviour) - answer_ids)
    if unknown:
        raise UserError(
            "generation.calibration_exclude_behaviour names answers the run does not "
            f"have: {', '.join(unknown)}",
            hint="list answer ids (configuration:question) of the run, or remove them",
        )
    if not rated:
        raise UserError(
            f"no human rating in {ratings_path}",
            hint="rate the calibration sample with `crossvenue rate` first",
        )


def calibration_report(cfg: Config, run_dir: str | Path) -> Calibration:
    """Every agreement the selection rests on, and the selection itself.

    The human ratings are those in `judge_sample.RATINGS_PATH`. The judges,
    thresholds and behavior exclusions come from `generation`
    (`calibration_exclude_behaviour`: rated answers left out of
    `behaviour_correct` against the human, methodology section 9 records why);
    the bootstrap from `eval`.

    Raises:
        UserError: `generation.judges` names no candidate, or a candidate with no judgments in
            the run or that graded none of the rated answers, or
            `generation.calibration_exclude_behaviour` names an answer the run does not have.
    """
    gen = cfg.generation
    judges = list(gen.judges)
    ratings_path = RATINGS_PATH
    ratings = read_ratings(ratings_path)
    _require_inputs(cfg, Path(run_dir), ratings_path, rated=bool(ratings))
    thresholds = gen.calibration_thresholds
    resamples, seed = cfg.eval.bootstrap_resamples, cfg.eval.bootstrap_seed
    exclude_behaviour = gen.calibration_exclude_behaviour
    humans = human_ratings(ratings_path)
    # Each judge's file is read once: its judgments and what they cost.
    rows = {j: read_judgements(p) for j in judges if (p := judgements_path(run_dir, j)).exists()}
    graded = {
        j: {r.answer_id: r.judge.judgement for r in rows.get(j, []) if r.judge} for j in judges
    }
    for j, grading in graded.items():
        if not grading:
            raise UserError(
                f"no judgments by {j} in {run_dir}",
                hint=f"run `crossvenue judge-sample --judge {j}` first, "
                "or leave it out of generation.judges",
            )
        if not grading.keys() & humans.keys():
            raise UserError(
                f"{j} graded no answer rated in {ratings_path}, so nothing compares "
                "it with the human",
                hint=f"run `crossvenue judge-sample --judge {j}` on the sample that was rated",
            )
    comparisons: dict[str, dict[str, DimensionStatistics]] = {}
    for j in judges:
        comparisons[human_comparison(j)] = agreement_record(
            graded[j], humans, resamples, seed, exclude_behaviour
        )
    for i, a in enumerate(judges):
        for b in judges[i + 1 :]:
            comparisons[judge_comparison(a, b)] = agreement_record(
                graded[a], graded[b], resamples, seed
            )
    for j in judges:
        repeat = load_judgements(judgements_path(run_dir, j, 1))
        if repeat:
            comparisons[replicate_comparison(j, 1)] = agreement_record(
                graded[j], repeat, resamples, seed
            )
    costs = {j: mean([r.judge.call.paid_usd() for r in rows[j] if r.judge]) for j in judges}
    selection = select_judge(
        {j: comparisons[human_comparison(j)] for j in judges}, thresholds, costs
    )
    return Calibration(
        run=str(run_dir),
        ratings=str(ratings_path),
        raters=sorted({row.rater for row in ratings}),
        answers_rated=len(humans),
        resamples=resamples,
        seed=seed,
        thresholds=dict(thresholds),
        excluded_from_behaviour_correct=list(exclude_behaviour),
        statistic=STATISTIC,
        comparisons=comparisons,
        candidates=selection.candidates,
        selected=selection.selected,
        reason=selection.reason,
    )
