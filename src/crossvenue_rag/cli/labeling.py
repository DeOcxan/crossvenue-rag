"""Labeling commands: label (and `--review`, `--augment`), augment-candidates, derive-coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from .. import formatting
from ..config_loading import DEFAULT_CONFIG
from ..corpus.artifacts import load_chunk_venues
from ..dataset.journal import load_rounds
from ..dataset.questions import load_questions, require_label_versions
from ..dataset.sampling import read_sample_file
from ..errors import UserError
from ..index_version import require_current_index
from ..labeling.candidates import augment_candidates, read_run_rankings, save_candidates
from ..labeling.coverage import RULE, derive_coverage
from ..labeling.modes import LabelRequest, run_label
from ..labeling.pool import pool_depth_of
from .app import ConfigOpt, QuestionsOpt, SetOpt, console, load, say


def label(
    ids: Annotated[
        list[str] | None,
        typer.Argument(
            help="Question ids to review (with `--review`) or relabel (with `--relabel`).",
            show_default=False,
        ),
    ] = None,
    review: Annotated[
        bool,
        typer.Option(
            "--review",
            help="The listed labeled questions, re-opened with their current marks; changes "
            "are recorded as corrections to round 1.",
        ),
    ] = False,
    review_file: Annotated[
        Path | None,
        typer.Option("--review-file", help="A file of ids to review (first token per line)."),
    ] = None,
    augment: Annotated[
        bool,
        typer.Option(
            "--augment",
            help="Only the never-judged chunks listed in `--question-file`, judged "
            "(written by `crossvenue augment-candidates`); extends round 1.",
        ),
    ] = False,
    question_file: Annotated[
        Path | None,
        typer.Option("--question-file", help="Candidate list for `--augment`."),
    ] = None,
    adjudicate: Annotated[
        bool,
        typer.Option(
            "--adjudicate",
            help="A decision on each (question, chunk) pair round 1 and round 2 judged "
            "differently; recorded under 'adjudicated', rounds 1 and 2 untouched.",
        ),
    ] = False,
    relabel: Annotated[
        bool,
        typer.Option(
            "--relabel",
            help="The given questions, judged again on their recorded round-1 candidates; "
            "recorded under 'relabeled', the question file untouched.",
        ),
    ] = False,
    add_chunks: Annotated[
        Path | None,
        typer.Option(
            "--add-chunks",
            help="With `--relabel` or `--round 3`: chunks to show beside the recorded "
            "candidates, one JSON object per line with the question's `id` and its `chunk_ids`.",
        ),
    ] = None,
    metadata_from: Annotated[
        Path | None,
        typer.Option(
            "--metadata-from",
            help="With `--round 3`: the dataset the questions are shown from; the journal "
            "stays the one of the question file.",
        ),
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    all_: Annotated[
        bool, typer.Option("--all", help="Questions that already have labels, shown again.")
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", help="Candidates shown per question (default: `eval.pool_display`)."
        ),
    ] = None,
    pool_depth: Annotated[
        int | None,
        typer.Option(
            "--pool-depth",
            help="Depth taken from each method (default: `eval.pool_depth_per_method`).",
        ),
    ] = None,
    round_: Annotated[
        int,
        typer.Option(
            "--round",
            help="Annotation round. 1 writes gold labels; 2 re-annotates a subsample "
            "for intra-rater agreement; 3 relabels the whole set, blind. 2 and 3 leave "
            "round 1 untouched.",
        ),
    ] = 1,
    annotator: Annotated[str, typer.Option("--annotator", help="Recorded on every judgment.")] = "",
    sample: Annotated[
        int | None,
        typer.Option(
            "--sample",
            help="N questions to label: at random in round 1, stratified by category in round 2.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for `--sample`.")] = 0,
    sample_file: Annotated[
        Path | None,
        typer.Option(
            "--sample-file",
            help="Round 2: exactly the questions listed here, to label "
            "(written by `crossvenue round-sample`).",
        ),
    ] = None,
) -> None:
    """Marks gold chunks for eval questions, in the terminal.

    `crossvenue label --review cv-0043 cv-0046` or `--review-file FILE` re-opens
    labeled questions for correction instead. `--adjudicate` decides the pairs the
    two labeling rounds disagree on, `--relabel ID ...` judges named questions
    again, and `--round 3` judges every question again, blind; all three only add
    entries to the journal.
    """
    cfg = load(config, set_)
    request = LabelRequest(
        questions_path=str(questions or cfg.eval.questions),
        ids=tuple(ids or ()),
        review=review,
        review_file=review_file,
        augment=augment,
        question_file=question_file,
        revisit=all_,
        limit=limit,
        pool_depth=pool_depth,
        annotation_round=round_,
        annotator=annotator,
        sample=sample,
        seed=seed,
        sample_file=sample_file,
        adjudicate=adjudicate,
        relabel=relabel,
        add_chunks=add_chunks,
        metadata_from=metadata_from,
    )
    run_label(cfg, request)


def augment_candidates_command(
    run: Annotated[
        Path, typer.Option("--run", help="An eval or pool run: results/<ts> or results/<ts>_pool.")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Candidate list to write (JSONL).")],
    method: Annotated[
        list[str],
        typer.Option("--method", "-m", help="Method whose top k to judge; repeatable."),
    ],
    category: Annotated[
        str | None, typer.Option("--category", help="Only questions of this category.")
    ] = None,
    k: Annotated[
        int | None,
        typer.Option(
            "--k", help="Depth of each method's list (default: `eval.pool_depth_per_method`)."
        ),
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
) -> None:
    """Lists the chunks a run's methods retrieved that no round-1 judgment showed."""
    cfg = load(config, set_)
    path = str(questions or cfg.eval.questions)
    if out.exists():
        raise UserError(f"{out} exists", hint="pick another file; the list records one run")
    depth = pool_depth_of(cfg, k)
    rankings = read_run_rankings(run)
    entries = augment_candidates(
        rankings, load_questions(path), load_rounds(path), method, category, depth
    )
    save_candidates(out, entries)
    total = sum(len(e.chunks) for e in entries)
    say(
        console,
        f"{formatting.plural(total, 'unjudged chunk')} across "
        f"{formatting.plural(len(entries), 'question')} -> {out}\n"
        f"judge them with: crossvenue label --augment --question-file {out}",
    )


def derive_coverage_command(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    exclude_file: Annotated[
        Path | None,
        typer.Option("--exclude-file", help="Ids to leave unchanged (first token per line)."),
    ] = None,
) -> None:
    """Fills venue_coverage values that follow from recorded data, journaled as derived."""
    cfg = load(config, set_)
    path = str(questions or cfg.eval.questions)
    # The venues come first: without chunks.jsonl the error names it and `ingest`.
    venues = load_chunk_venues(cfg)
    version = require_current_index(cfg)
    require_label_versions(path, version, all_rounds=True)
    exclude = set(read_sample_file(exclude_file)) if exclude_file else set()
    changed = derive_coverage(path, venues, version, exclude)
    say(console, f"rule: {RULE}")
    for item in changed:
        note = f"  undecided: {', '.join(item.undecided)}" if item.undecided else ""
        coverage = ", ".join(f"{venue} {value}" for venue, value in item.coverage.items())
        say(console, Text.assemble(f"{item.question_id}: {coverage}", (note, "yellow")))
    say(
        console,
        Text.assemble(
            (formatting.plural(len(changed), "question"), "green"),
            f" updated, {formatting.count(len(exclude))} excluded",
        ),
    )
