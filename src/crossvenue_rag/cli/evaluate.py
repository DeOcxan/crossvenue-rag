"""The commands that retrieve or evaluate: retrieve, eval, pool-run, bench-rerank, power."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from .. import formatting, listing
from ..config import Config
from ..config_loading import DEFAULT_CONFIG
from ..config_pipeline import RetrievalMethod
from ..corpus.sources import venue_types
from ..dataset.journal import load_judged_sets
from ..dataset.questions import index_check, load_questions
from ..errors import UserError, not_blank, require_file
from ..evaluation.bench import BENCH_REPORT_FILE, BenchRequest, bench_rerank
from ..evaluation.jobs import EvalJob, run_evaluation
from ..evaluation.power import (
    POWER_PERMUTATIONS,
    POWER_STUDIES,
    PowerRequest,
    power_run,
    run_questions_path,
)
from ..labeling.candidates import PoolSpec, pool_run
from ..labeling.pool import pool_depth_of
from ..retrieval.filters import filters_from_options
from ..retrieval.methods import method_named
from ..retrieval.retriever import checked_retriever, compare_methods
from ..run_files import REPORT_FILE, run_path
from . import render
from .app import (
    ConfigOpt,
    QuestionsOpt,
    SetOpt,
    at_least,
    console,
    load,
    methods_from,
    question_category,
    say,
)


def retrieve(
    query: Annotated[str, typer.Argument(help="The question to search for.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    method: Annotated[
        str,
        typer.Option(
            "--method", "-m", help="One method, or 'all' for every method except the oracle."
        ),
    ] = "hybrid",
    k: Annotated[
        int | None,
        typer.Option(
            "--k",
            help="Results per method, 1 or more and at most what the method can return "
            "(default: `retrieval.k`).",
        ),
    ] = None,
    venue: Annotated[
        list[str] | None, typer.Option("--venue", help="Venues to search; repeatable.")
    ] = None,
    venue_type: Annotated[
        str | None, typer.Option("--venue-type", help="The venue type to search, cex or dex.")
    ] = None,
    filter_: Annotated[
        list[str] | None,
        typer.Option(
            "--filter",
            "-f",
            help="Metadata filter FIELD=VALUE (`environment`, `api_version`, `product`, "
            "`doc_type`); a repeated field ORs its values, 'none' matches unassigned.",
        ),
    ] = None,
) -> None:
    """Runs one query and prints the top k results."""
    not_blank("the query", query)
    at_least("--k", k, 1)
    cfg = load(config, set_)
    depth = cfg.retrieval.k if k is None else k
    methods = methods_from(method)
    _require_depth_served(cfg, methods, depth)
    filters = filters_from_options(
        venue or [], venue_type, filter_ or [], venue_types(cfg.paths.sources)
    )
    for run in compare_methods(cfg, query, methods, depth, filters):
        say(console, render.retrieval_results(run.method, depth, run.results, run.timings))


def _require_depth_served(cfg: Config, methods: list[RetrievalMethod], depth: int) -> None:
    """Refuses a `--k` deeper than a method's list can be, as the config refuses a depth.

    The refusal names every bound `depth` exceeds, so following its hint once is enough.

    Raises:
        UserError: `depth` exceeds a key of `Config.depth_bounds` for one of `methods`.
    """
    exceeded: dict[str, int] = {}
    short: list[str] = []
    for method in methods:
        over = {key: bound for key, bound in cfg.depth_bounds(method).items() if depth > bound}
        exceeded.update(over)
        if over:
            short.append(method)
    if not exceeded:
        return
    bounds = listing.prose(
        [f"{key} ({formatting.own_form(bound)})" for key, bound in exceeded.items()]
    )
    raises = " ".join(f"--set {key}={formatting.own_form(depth)}" for key in exceeded)
    raise UserError(
        f"--k {formatting.own_form(depth)} is deeper than {bounds}, so "
        f"{listing.prose(short)} would return fewer results than asked for",
        hint=f"pass --k {formatting.own_form(min(exceeded.values()))} or less, or raise "
        f"{listing.prose(list(exceeded))}: `{raises}`",
    )


def eval_command(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    venue_filter: Annotated[
        bool,
        typer.Option(
            "--venue-filter",
            help="Each query restricted to the question's labeled venues (reads the labels).",
        ),
    ] = False,
) -> None:
    """Runs the six-method ablation and writes results/<timestamp>/."""
    cfg = load(config, set_)
    path = Path(questions or cfg.eval.questions)
    question_list = load_questions(path)
    check = index_check(path, question_list)
    job = EvalJob(path, question_list, load_judged_sets(path), check, venue_filter=venue_filter)
    run_dir = run_evaluation(cfg, job)
    # Printed as text: a report can hold brackets (a --note) that rich would read as markup.
    say(console, (run_dir / REPORT_FILE).read_text(encoding="utf-8"))
    say(console, Text("results:", style="green"), str(run_dir))


def pool_run_command(
    method: Annotated[
        list[str], typer.Option("--method", "-m", help="Method to retrieve with; repeatable.")
    ],
    k: Annotated[
        int | None,
        typer.Option(
            "--k",
            help="Retrieval depth, above `max(eval.k_values)` "
            "(default: `eval.pool_depth_per_method`).",
        ),
    ] = None,
    category: Annotated[
        str | None, typer.Option("--category", help="Only questions of this category.")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
) -> None:
    """Runs retrieval only: its top k becomes chunks to judge, and it computes no metric.

    Any method may run, the oracle included: augmentation judges the never-judged top
    k of a method not in the labeling pool (methodology section 1, rule 3). Writes
    results/<timestamp>_pool/ for `crossvenue augment-candidates --run`.
    """
    cfg = load(config, set_)
    path = require_file(
        questions or cfg.eval.questions,
        hint="check the path given by --questions or eval.questions",
    )
    methods = [method_named(m) for m in method]
    question_category("--category", category)
    depth = pool_depth_of(cfg, k, "--k")
    run_dir = run_path(cfg.paths.results, "_pool")
    with checked_retriever(cfg, index_check(path)) as (retriever, version):
        rows = pool_run(cfg, retriever, PoolSpec(path, methods, depth, category), version, run_dir)
    say(
        console,
        Text.assemble(
            f"{formatting.plural(rows, 'retrieval')}, k={formatting.own_form(depth)} "
            f"-> {run_dir}  ",
            ("(pool construction, no metrics)", "dim"),
        ),
    )


def bench_rerank_command(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    rounds: Annotated[
        int,
        typer.Option("--rounds", help="Passes over the questions, 1 or more; order alternates."),
    ] = 1,
    force: Annotated[
        bool, typer.Option("--force", help="Timing with no wait for a quiet machine.")
    ] = False,
) -> None:
    """Times venue_split with one cross-encoder call per venue against one for all venues.

    Runs the two alternately in one process; writes results/<timestamp>_bench/.
    """
    at_least("--rounds", rounds, 1)
    cfg = load(config, set_)
    request = BenchRequest(Path(questions or cfg.eval.questions), rounds, force=force)
    run_dir = bench_rerank(
        cfg,
        request,
        load_questions(request.questions_path),
        notify=lambda reasons: say(
            console, Text("waiting for a quiet machine:", style="yellow"), reasons
        ),
    )
    say(console, (run_dir / BENCH_REPORT_FILE).read_text(encoding="utf-8"))
    say(console, Text("results:", style="green"), str(run_dir))


def power_command(
    run: Annotated[Path, typer.Option("--run", help="A retrieval run, results/<ts>.")],
    category: Annotated[str, typer.Option("--category", help="Question category.")] = "comparison",
    k: Annotated[
        int | None,
        typer.Option("--k", help="recall@k, 1 or more (default: the largest of `eval.k_values`)."),
    ] = None,
    studies: Annotated[
        int, typer.Option("--studies", help="Simulated studies per delta, 1 or more.")
    ] = POWER_STUDIES,
    permutations: Annotated[
        int, typer.Option("--permutations", help="Sign flips per randomization test, 1 or more.")
    ] = POWER_PERMUTATIONS,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
) -> None:
    """Simulates the power of the Holm-adjusted paired randomization test on recall@k.

    Uses the discordance and difference sizes the run's comparison family shows;
    writes results/<timestamp>_power/.
    """
    question_category("--category", category)
    at_least("--k", k, 1)
    at_least("--studies", studies, 1)
    at_least("--permutations", permutations, 1)
    cfg = load(config, set_)
    depth = max(cfg.eval.k_values) if k is None else k
    request = PowerRequest(run, category, depth, studies, permutations)
    power = power_run(cfg, request, load_questions(run_questions_path(run)))
    # The lines name the --category given, so they print as literal text.
    for line in render.power_lines(power.study):
        say(console, line)
    say(console, Text("results:", style="green"), str(power.run_dir))
