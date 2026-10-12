"""Answer commands: generating, rating and judging answers, and comparing prompts.

ask, answer-eval, rate-sample, rate, judge-sample, judge-agreement, select-judge, judge-run
and compare-prompts.
"""

from __future__ import annotations

import getpass
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from .. import formatting
from ..config_loading import DEFAULT_CONFIG
from ..confined_paths import file_on_path
from ..errors import UserError
from ..generation.ask import DEFAULT_GENERATOR, DEFAULT_RETRIEVER, ask
from ..generation.calibration_file import CALIBRATION_FILE
from ..generation.calibration_report import CALIBRATION_MARKDOWN, select_judge_files
from ..generation.calibration_sample import draw_calibration_sample
from ..generation.jobs import (
    AnswerEvalJob,
    JudgeRunJob,
    JudgeSampleJob,
    answer_eval,
    compare_prompt_runs,
    estimate_answer_eval,
    judge_run,
    judge_sample,
)
from ..generation.judge_sample import RATINGS_PATH, judge_agreements
from ..labeling.rate import rate_calibration_sample
from ..reporting import run_ids
from ..run_files import REPORT_FILE
from . import render_answers
from .app import ConfigOpt, QuestionsOpt, SetOpt, console, load, positive_budget, say

# The width of the stratum column `rate-sample` prints.
STRATUM_WIDTH = 45

RunOpt = Annotated[Path, typer.Option("--run", help="An answer run, results/<ts>_answers.")]
JudgeOpt = Annotated[
    str, typer.Option("--judge", help="Judge candidate, for example sonnet or opus.")
]
CALIBRATION_SAMPLE = Path(run_ids.CALIBRATION_SAMPLE)


def ask_command(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    retriever: Annotated[
        str, typer.Option("--retriever", help="Retrieval method.")
    ] = DEFAULT_RETRIEVER,
    generator: Annotated[
        str, typer.Option("--generator", help="Generator name from `generation.generators`.")
    ] = DEFAULT_GENERATOR,
) -> None:
    """Answers one question: cited claims per venue, or a refusal or a clarifying question."""
    result = ask(load(config, set_), question, retriever, generator)
    render_answers.print_answer(console, result)


def answer_eval_command(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    question_id: Annotated[
        list[str] | None,
        typer.Option("--question-id", help="Only these questions (repeatable)."),
    ] = None,
    judge: Annotated[
        str | None, typer.Option("--judge", help="A judge candidate to grade the answers too.")
    ] = None,
    budget: Annotated[
        float | None, typer.Option("--budget", help="Run budget in USD (default: config).")
    ] = None,
    estimate: Annotated[
        bool, typer.Option("--estimate", help="A cost estimate alone, with no model call.")
    ] = False,
    note: Annotated[str, typer.Option("--note", help=f"A line written into {REPORT_FILE}.")] = "",
) -> None:
    """Generates every configuration of `generation.matrix`; writes results/<ts>_answers/."""
    positive_budget(budget)
    cfg = load(config, set_)
    job = AnswerEvalJob(
        questions_path=Path(questions or cfg.eval.questions),
        question_ids=tuple(question_id or ()),
        judge=judge,
        budget=budget,
        note=note,
    )
    if estimate:
        say(console, render_answers.estimate_table(estimate_answer_eval(cfg, job)))
        return
    run_dir = answer_eval(cfg, job, progress=lambda answer_id: say(console, answer_id, style="dim"))
    # Printed as text: a report can hold brackets (a --note) that rich would read as markup.
    say(console, (run_dir / REPORT_FILE).read_text(encoding="utf-8"))
    say(console, Text("results:", style="green"), str(run_dir))


def rate_sample_command(
    run: RunOpt,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Sample file.")] = CALIBRATION_SAMPLE,
) -> None:
    """Draws and pins the judge-calibration sample, stratified by configuration and category."""
    if out.is_dir():
        raise UserError(
            f"--out {out} is a directory", hint="pass --out the path of the sample file"
        )
    blocker = file_on_path(out.parent)
    if blocker is not None:
        raise UserError(
            f"--out {out} cannot be written: {blocker} is a file",
            hint="pass --out a path whose directories are directories",
        )
    strata = draw_calibration_sample(load(config, set_), run, out)
    say(console, f"{formatting.plural(sum(strata.values()), 'answer')} -> {out}")
    for name, count in sorted(strata.items()):
        say(console, f"  {name:<{STRATUM_WIDTH}} {formatting.count(count)}")


def rate_command(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    sample_file: Annotated[
        Path, typer.Option("--sample-file", help="Written by rate-sample.")
    ] = CALIBRATION_SAMPLE,
    rater: Annotated[
        str,
        typer.Option("--rater", help="Recorded on every rating (default: the login name)."),
    ] = "",
) -> None:
    """Grades the calibration sample by hand; no judge's grade is ever shown."""
    cfg = load(config, set_)
    rater = rater.strip() or getpass.getuser()
    rated = rate_calibration_sample(
        cfg, sample_file, questions or cfg.eval.questions, rater, partial(say, console)
    )
    say(console, f"{formatting.plural(rated, 'rating')} written to {RATINGS_PATH}")


def judge_sample_command(
    judge: JudgeOpt,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    sample_file: Annotated[
        Path, typer.Option("--sample-file", help="Written by rate-sample.")
    ] = CALIBRATION_SAMPLE,
    replicate: Annotated[
        int, typer.Option("--replicate", help="0 = first grading; 1 = self-consistency repeat.")
    ] = 0,
    budget: Annotated[
        float | None, typer.Option("--budget", help="Run budget in USD (default: config).")
    ] = None,
) -> None:
    """Has a judge candidate grade the calibration sample.

    Writes `judgements_<judge>.jsonl`, or `judgements_<judge>_r1.jsonl` with `--replicate 1`.
    """
    positive_budget(budget)
    if replicate not in (0, 1):
        raise UserError(
            f"--replicate must be 0 or 1, got {formatting.count(replicate)}",
            hint="pass --replicate 0 to grade the sample, or 1 to repeat that grading for "
            "self-consistency",
        )
    cfg = load(config, set_)
    job = JudgeSampleJob(
        judge=judge,
        sample_file=sample_file,
        questions_path=Path(questions or cfg.eval.questions),
        replicate=replicate,
        budget=budget,
    )
    outcome = judge_sample(cfg, job)
    say(
        console,
        f"{formatting.count(outcome.judged)} of "
        f"{formatting.plural(outcome.sampled, 'answer')} judged, "
        f"{formatting.plural(outcome.errors, 'judge error')}, "
        f"{formatting.cost(outcome.spent_usd)} spent -> {outcome.path}",
    )
    if outcome.stopped:
        say(console, Text("stopped:", style="red"), str(outcome.stopped))


def judge_agreement_command(
    run: RunOpt,
    judge: JudgeOpt,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
) -> None:
    """Reports how a judge candidate agrees with the human ratings and with its own repeat."""
    cfg = load(config, set_)
    judge = cfg.generation.chosen_judge(judge)
    agreements = judge_agreements(cfg, run, judge)
    if agreements.with_humans is not None:
        title = f"{judge} vs human"
        say(console, render_answers.agreement_table(title, agreements.with_humans))
    else:
        console.print("[yellow]no human ratings[/yellow] (run `crossvenue rate` first)")
    if agreements.with_repeat is not None:
        title = f"{judge} vs {judge} (replicate 1)"
        say(console, render_answers.agreement_table(title, agreements.with_repeat))


def select_judge_command(
    run: RunOpt,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
) -> None:
    """Compares every judge candidate with the human ratings and with each other.

    Reports the judge methodology section 9 selects; writes calibration.json and
    calibration.md.
    """
    cfg = load(config, set_)
    text = select_judge_files(cfg, run)
    say(console, text)
    say(
        console,
        Text("written:", style="green"),
        f"{run / CALIBRATION_FILE}, {CALIBRATION_MARKDOWN}",
    )


def judge_run_command(
    run: RunOpt,
    judge: Annotated[
        str | None, typer.Option("--judge", help="Judge name (default: `generation.judge`).")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    questions: QuestionsOpt = None,
    budget: Annotated[
        float | None, typer.Option("--budget", help="Run budget in USD (default: config).")
    ] = None,
    note: Annotated[str, typer.Option("--note", help=f"A line written into {REPORT_FILE}.")] = "",
) -> None:
    """Judges a saved answer run without generating again; writes results/<ts>_judged/."""
    positive_budget(budget)
    cfg = load(config, set_)
    job = JudgeRunJob(
        run=run,
        questions_path=Path(questions or cfg.eval.questions),
        judge=judge,
        budget=budget,
        note=note,
    )
    outcome = judge_run(cfg, job, progress=lambda answer_id: say(console, answer_id))
    say(
        console,
        f"{formatting.count(outcome.judged)} of {formatting.plural(outcome.answers, 'answer')}, "
        f"{formatting.plural(outcome.failed, 'judge error')}, "
        f"{formatting.cost(outcome.spent_usd)} spent -> {outcome.run_dir}",
    )
    if outcome.stopped:
        say(console, Text("stopped:", style="red"), str(outcome.stopped))


def compare_prompts_command(
    v1: Annotated[Path, typer.Option("--v1", help="Judged run of prompt v1.")],
    v2: Annotated[Path, typer.Option("--v2", help="Judged run of prompt v2.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
) -> None:
    """Tests H-v2 (methodology section 9a): prompt v1 against v2 per configuration.

    Also runs the side-effect checks on v2's partial answers; writes
    results/<ts>_prompts/.
    """
    run_dir = compare_prompt_runs(load(config, set_), v1, v2)
    report = (run_dir / REPORT_FILE).read_text(encoding="utf-8")
    say(console, report)
    say(console, Text("results:", style="green"), str(run_dir))
