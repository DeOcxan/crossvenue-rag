"""Dataset commands: validate-evals, kappa, select, round-sample."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from .. import formatting
from ..config_loading import DEFAULT_CONFIG
from ..dataset.agreement import (
    KAPPA_TARGET,
    MIN_RETEST_DAYS,
    Bootstrap,
    KappaRequest,
    kappa_study,
)
from ..dataset.finalize import finalize_selection
from ..dataset.questions import load_questions
from ..dataset.records import DATASETS_DIR, load_records, newest_candidate_pool
from ..dataset.sampling import (
    ROUND2_SEED,
    ROUND2_SIZE,
    cited_words,
    stratified_sample,
    write_sample_file,
)
from ..dataset.selection import selection_path
from ..dataset.validate import validate_dataset
from ..errors import require_file
from ..labeling.select_session import SelectionPlan, check_categories, run
from . import render
from .app import ConfigOpt, QuestionsOpt, SetOpt, console, load, say

# Round-2 questions whose gold the answer-rating screen showed during the quiet
# period (methodology section 5); `kappa` reports agreement without them too.
EXPOSED_FILE = Path("evals/selection/round2_exposed.txt")


def validate_evals(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    dataset: Annotated[
        list[Path] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset file; repeatable. Default: every evals/datasets/*.jsonl.",
        ),
    ] = None,
    skip_near_duplicates: Annotated[
        bool,
        typer.Option(
            "--skip-near-duplicates",
            help="No near-duplicate check, the one that needs the encoder.",
        ),
    ] = False,
) -> None:
    """Validates evaluation datasets against the current corpus."""
    cfg = load(config, set_)
    paths = list(dataset) if dataset else sorted(DATASETS_DIR.glob("*.jsonl"))
    if not paths:
        say(console, f"no dataset files under {DATASETS_DIR}", style="yellow")
        return

    failed = False
    for path in paths:
        report = validate_dataset(path, cfg, check_near_duplicates=not skip_near_duplicates)
        status = ("ok", "green") if report.ok else ("failed", "red")
        say(console, Text.assemble("\n", (str(path), "bold"), "  ", status))
        if report.stats:
            say(console, render.validation_stats(report.stats))
        for line in render.validation_messages(report):
            say(console, line)
        failed = failed or not report.ok

    if failed:
        raise typer.Exit(1)


def round_sample(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    n: Annotated[int, typer.Option("--n", help="Questions in the subsample.")] = ROUND2_SIZE,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = ROUND2_SEED,
    out: Annotated[Path, typer.Option("--out", "-o", help="Sample file to write.")] = Path(
        "evals/selection/round2_sample.txt"
    ),
) -> None:
    """Draws the round-2 subsample, stratified by category, and writes it to a file.

    `crossvenue label --round 2 --sample-file FILE` then labels exactly that list.
    """
    cfg = load(config, set_)
    path = str(questions or cfg.eval.questions)
    sample = stratified_sample(load_questions(path), n, seed)
    write_sample_file(out, sample, seed)
    counts = Counter(q.effective_category for q in sample)
    say(
        console,
        f"{formatting.plural(len(sample), 'question')}, seed {formatting.own_form(seed)} -> {out}",
    )
    for category, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        say(console, f"  {category:<18} {formatting.count(count)}")
    for q in sample:
        say(console, f'{q.id}  {q.effective_category:<18} "{cited_words(q)}"')


def kappa(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    first: Annotated[int, typer.Option("--first", help="First round to compare.")] = 1,
    second: Annotated[int, typer.Option("--second", help="Second round to compare.")] = 2,
    exposed: Annotated[
        Path | None,
        typer.Option(
            "--exposed",
            help="Questions whose gold was shown before the second round (one id per line); "
            f"default {EXPOSED_FILE}, when it exists.",
        ),
    ] = None,
    short_gap: Annotated[
        bool,
        typer.Option(
            "--short-gap",
            help="A kappa reported although a question was re-judged within "
            f"{formatting.count(MIN_RETEST_DAYS)} "
            "days; the output says so.",
        ),
    ] = False,
) -> None:
    """Cohen's kappa between two annotation rounds (intra-rater reliability).

    Prints the registered reading (round 1 as corrected, all sampled questions),
    the range of the retest gap and the agreement per question category. Then come
    the three sensitivity readings of methodology section 5: round 1 as first
    judged, and the questions not listed in `--exposed`. After them come each
    reading's bootstrap interval over questions (`eval.bootstrap_resamples`,
    `eval.bootstrap_seed`), the direction of each disagreement, and the
    verdicts on the source chunks round 1 marked. Refuses when any question's
    two judgments are closer than the minimum retest gap, unless `--short-gap`.
    """
    if exposed is not None:
        require_file(exposed, hint="pass --exposed a list of question ids, one per line")
    cfg = load(config, set_)
    bootstrap = Bootstrap(cfg.eval.bootstrap_resamples, cfg.eval.bootstrap_seed)
    request = KappaRequest(
        str(questions or cfg.eval.questions),
        first,
        second,
        EXPOSED_FILE if exposed is None else exposed,
        short_gap=short_gap,
        bootstrap=bootstrap,
    )
    study = kappa_study(request)
    say(console, render.rounds_agreement_table(study.report, first, second))
    if study.gaps:
        short = (
            f" ({formatting.count(len(study.short))} under "
            f"{formatting.count(MIN_RETEST_DAYS)} days: not the registered measurement)"
        )
        say(
            console,
            Text.assemble(
                f"retest gap: {formatting.number(min(study.gaps.values()), 1)} to "
                f"{formatting.number(max(study.gaps.values()), 1)} days",
                (short if study.short else "", "yellow"),
            ),
        )
    say(console, render.kappa_by_category_table(study.by_category))
    say(console, render.agreement_variants_table(study.variants, first, second))
    say(
        console,
        Text(
            f"\nBelow the target of {formatting.number(KAPPA_TARGET, 2)}, methodology section 5 "
            "reports kappa as a limitation of the version, names the categories with the most "
            "disagreement and refines the criterion of section 5a where they show it ambiguous.",
            style="dim",
        ),
    )
    say(
        console,
        f"\nbootstrap over questions: {formatting.count(bootstrap.resamples)} resamples, "
        f"seed {formatting.own_form(bootstrap.seed)}",
    )
    say(console, render.kappa_intervals_table(study.variants, first, second))
    say(
        console,
        render.disagreement_direction_table(
            study.direction, study.direction_by_category, first, second
        ),
    )
    say(console, render.source_chunks_table(study.sources, first, second))


def select(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Candidate pool; default: the newest evals/datasets/*-candidates.jsonl.",
        ),
    ] = None,
    questions: QuestionsOpt = None,
    category: Annotated[
        list[str] | None,
        typer.Option("--category", help="The categories to review; default: all."),
    ] = None,
    include_skipped: Annotated[
        bool, typer.Option("--include-skipped", help="Skipped questions, shown again.")
    ] = False,
    finalize: Annotated[
        bool,
        typer.Option("--finalize", help="The question file, written from the accepted candidates."),
    ] = False,
) -> None:
    """Chooses the evaluation set from a candidate pool, recording every decision."""
    cfg = load(config, set_)
    path = dataset or newest_candidate_pool(DATASETS_DIR)
    candidates = load_records(path)
    questions_path = str(questions or cfg.eval.questions)
    selection_file = str(selection_path(questions_path))
    check_categories(category)
    if not finalize:
        say(
            console,
            f"pool: {path} ({formatting.plural(len(candidates), 'candidate')})",
            style="dim",
        )
        plan = SelectionPlan(questions_path, selection_file, category, include_skipped)
        run(cfg, candidates, plan)
        return

    finalized = finalize_selection(cfg, candidates, questions_path, selection_file)
    say(console, render.composition(finalized.composition))
    for problem in finalized.composition.problems():
        say(console, Text.assemble("  ", ("short", "red"), f" {problem}"))
    finalized.require_composition()
    written = formatting.plural(finalized.written or 0, "question")
    say(console, Text.assemble("\n", ("wrote", "green"), f" {written} -> {questions_path}"))
    if finalized.reasons:
        say(console, render.rejection_reasons(finalized.reasons))
    validation = finalized.validation
    if validation is None:
        return
    status = ("ok", "green") if validation.ok else ("failed", "red")
    say(console, Text.assemble("\n", ("validation", "bold"), " ", status))
    for line in render.validation_messages(validation):
        say(console, line)
    if not validation.ok:
        raise typer.Exit(1)
