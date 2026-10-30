"""The Typer app of `crossvenue`: the options, checks and `say` its command modules share."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Annotated, get_args

import typer
from rich.console import Console, RenderableType

from .. import formatting
from ..config import Config
from ..config_loading import load_config
from ..config_pipeline import RetrievalMethod
from ..errors import UserError
from ..models import QuestionCategory
from ..retrieval.methods import AD_HOC_METHODS, ad_hoc_method

# The top-level `crossvenue --help` text.
HELP = """The crossvenue-rag command line.

    crossvenue fetch               # sources at the manifest's commits -> data/raw
    crossvenue ingest              # data/raw -> chunks.jsonl
    crossvenue stats               # corpus statistics
    crossvenue index               # chunks -> pgvector + BM25
    crossvenue retrieve            # one method, or all five (not the oracle)
    crossvenue eval                # six-method ablation -> results/<ts>/
    crossvenue label               # marks gold chunks; `--review`, `--augment`
    crossvenue pool-run            # retrieval only, to build a label pool
    crossvenue bench-rerank        # one cross-encoder call against one per venue
    crossvenue power               # power of the recall comparison, by simulation
    crossvenue augment-candidates  # never-judged chunks from a run
    crossvenue derive-coverage     # venue_coverage implied by the record
    crossvenue validate-evals      # checks the versioned datasets
    crossvenue round-sample        # fixed, stratified round-2 subsample
    crossvenue kappa               # agreement between annotation rounds
    crossvenue round-agreement     # rounds 1 and 3 across a criterion revision
    crossvenue round3-gold         # the gold set built from round 3 alone
    crossvenue select              # chooses the eval set from candidates
    crossvenue count-term          # chunks containing a term
    crossvenue id-map              # old chunk id -> canonical chunk id
    crossvenue remap-ids           # carries chunk-id pointers through a map
    crossvenue ask                 # answers one question, with citations
    crossvenue answer-eval         # generator x retriever matrix; `--estimate` prices it
    crossvenue rate-sample         # fixed, stratified judge-calibration sample
    crossvenue rate                # grades answers by hand, judge hidden
    crossvenue judge-sample        # a judge candidate grades the sample
    crossvenue judge-agreement     # judge vs human, judge vs itself
    crossvenue select-judge        # the judge section 9 selects -> calibration.json
    crossvenue judge-run           # judges a saved answer run -> results/<ts>_judged/
    crossvenue compare-prompts     # prompt v1 against v2 -> results/<ts>_prompts/
    crossvenue reproduce           # rechecks the numbers the checked documents state
    crossvenue runs-manifest       # file hashes and `index_version` of committed runs
    crossvenue check               # config and what is built; read-only

Every command except `reproduce` and `runs-manifest` takes `--config` and repeatable
`--set section.field=value`, so an ablation never requires editing code.
"""

app = typer.Typer(add_completion=False, help=HELP, no_args_is_help=True)


console = Console()


def say(
    out: Console, *objects: RenderableType, style: str | None = None, soft_wrap: bool = False
) -> None:
    """Prints `objects` with markup, highlighting and emoji codes off, so a value prints as is."""
    out.print(
        *objects, style=style, soft_wrap=soft_wrap, markup=False, highlight=False, emoji=False
    )


ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to a YAML config.")]


SetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="A config field set to a value, for example `retrieval.k=20`; repeatable.",
    ),
]


QuestionsOpt = Annotated[
    Path | None,
    typer.Option("--questions", "-q", help="Question file in place of `eval.questions`."),
]


def load(config: Path, overrides: list[str] | None) -> Config:
    """The config with `--set` overrides applied; INFO logging is set up first.

    Raises:
        UserError: An override of `overrides` is not `section.field=value` with a YAML value
            under a section, or the config schema rejects a value it sets.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    return load_config(config, overrides)


def methods_from(method: str) -> list[RetrievalMethod]:
    """The methods `--method` names, a typo refused before any model is loaded.

    Raises:
        UserError: `method` is neither `all` nor a method an ad-hoc question can use.
    """
    if method == "all":
        return list(AD_HOC_METHODS)
    return [ad_hoc_method(method, "--method")]


def at_least(option: str, value: int | None, low: int) -> None:
    """Refuses `option` below `low`; an option left out (None) passes.

    Raises:
        UserError: `value` is below `low`.
    """
    if value is not None and value < low:
        raise UserError(
            f"{option} must be at least {formatting.count(low)}, got {formatting.count(value)}",
            hint=f"pass {option} {formatting.count(low)} or more, or leave it out",
        )


def positive_budget(budget: float | None) -> None:
    """Refuses a `--budget` that is not a finite amount above 0; left out (None) passes.

    Raises:
        UserError: `budget` is 0, negative, infinite or not a number.
    """
    if budget is not None and not (math.isfinite(budget) and budget > 0):
        raise UserError(
            f"--budget must be an amount of USD above 0, got {formatting.cost(budget)}",
            hint="pass --budget 0.5, for example, or leave it out for generation.budget_usd",
        )


def question_category(option: str, value: str | None) -> None:
    """Refuses a category that no question can have; left out (None) passes.

    Raises:
        UserError: `value` is not a `QuestionCategory`.
    """
    categories = get_args(QuestionCategory)
    if value is not None and value not in categories:
        raise UserError(
            f"{option} {value!r} is not a question category",
            hint=f"pass {option} one of: {', '.join(categories)}",
        )
