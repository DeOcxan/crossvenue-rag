"""What docs/reference/cli.md claims.

One function per section of the document after the common options, in document order.
"""

from __future__ import annotations

from ... import formatting
from ...dataset.agreement import MIN_RETEST_DAYS
from ...dataset.selection import TARGETS
from ...dataset.validate import LEAKAGE_NGRAM
from ...evaluation import machine
from ...evaluation.power import TARGET_POWER_PERCENT
from ...generation.schemas import FIRST_SCHEMA, PARTIAL_SCHEMA
from ...models import MULTI_VENUE
from ..check import Doc
from ..compute import cli_options
from ..outcomes import Param
from ..sources import Sources
from . import comparisons
from .reasons import INTERVAL_LEVEL, SERVE_ADDRESS, STATUS_CODE

COMMAND_RULE = "a rule the command applies"
BUDGET = "the domain `cli.app.positive_budget` checks"
REPLICATES = "the replicates `judge-sample` accepts"


def cli_reference(doc: Doc, src: Sources) -> None:
    """States the claims of docs/reference/cli.md."""
    doc.definition("takes these two options", reason="the two rows of the table that follows")
    _retrieval(doc, src)
    _question_set(doc, src)
    _generation(doc, src)
    _service(doc)
    _reporting(doc)


def _default(src: Sources, command: str, option: str) -> str:
    """The default of `option`, a config value in its own form (docs/style.md rule 11)."""
    return formatting.own_form(cli_options.option_default(src, command, option))


def _least(src: Sources, command: str, option: str) -> str:
    return formatting.count(cli_options.least_value(src, command, option))


def _retrieval(doc: Doc, src: Sources) -> None:
    """`retrieve`, `eval`, `power` and `bench-rerank`."""
    doc.check("-k` | int >= {}, at most the candidate depths", _least(src, "retrieve", "--k"))
    doc.check(
        "Runs all {} methods over the question file",
        formatting.count(len(src.default_config.eval.methods)),
    )
    doc.check("the smallest \u03b4 reaching {}% power", formatting.count(TARGET_POWER_PERCENT))
    doc.check("-k` | int >= {} | max(`eval.k_values`) | recall@k", _least(src, "power", "--k"))
    for option in ("--studies", "--permutations"):
        doc.check(
            f"{option[1:]}` | int >= {{}} | {{}} |",
            _least(src, "power", option),
            _default(src, "power", option),
        )
    doc.check("whose text names {} or more venues", formatting.count(MULTI_VENUE))
    doc.check(
        "the 1-minute load average exceeds {} the CPU count",
        formatting.fixed(machine.BUSY_LOAD_PER_CPU, 1),
    )
    doc.check(
        "process uses at least {}% CPU. Other processes at or above {}% CPU",
        formatting.fixed(machine.COMPETING_CPU_PERCENT, 0),
        formatting.fixed(machine.BUSY_CPU_PERCENT, 0),
    )
    doc.check(
        "-rounds` | int >= {} | {} |",
        _least(src, "bench-rerank", "--rounds"),
        _default(src, "bench-rerank", "--rounds"),
    )


def _question_set(doc: Doc, src: Sources) -> None:
    """`select`, `label`, `round-sample`, `kappa` and `validate-evals`."""
    doc.check(
        "fewer than {} answerable comparison questions", formatting.count(TARGETS["comparison"])
    )
    doc.check("-round` | int | {} |", _default(src, "label", "--round"))
    doc.definition("1 writes gold labels, 2 and later", reason="the round numbers `label` reads")
    doc.check("-seed` | int | {} | Seed for", _default(src, "label", "--seed"))
    doc.definition("Two checks run before", reason="the two checks the paragraph describes")
    doc.check("-n` | int | {} |", _default(src, "round-sample", "--n"))
    doc.check("-seed` | int | {} | Sampling seed", _default(src, "round-sample", "--seed"))
    doc.definitions(
        (
            "between two annotation rounds",
            "the three sensitivity readings",
            "one or both of two choices",
        ),
        reason="the rounds kappa compares, the readings methodology \u00a75 registers and the "
        "choices they vary",
    )
    days = formatting.count(MIN_RETEST_DAYS)
    doc.check("A gap under {} days (methodology \u00a75)", days)
    doc.check("re-judged within {} days of round 1", days)
    doc.check("-first` | int | {} |", _default(src, "kappa", "--first"))
    doc.check("-second` | int | {} |", _default(src, "kappa", "--second"))
    doc.check("sharing {} or more consecutive words", formatting.count(LEAKAGE_NGRAM))


def _generation(doc: Doc, src: Sources) -> None:
    """The answer-generation commands."""
    doc.definition("| finite float > 0 |", reason=BUDGET)
    first, repeat = Param("0", REPLICATES), Param("1", REPLICATES)
    doc.check(
        "-replicate` | {} or {} | {} | {} for the first grading, {} for the repeat |",
        first,
        repeat,
        _default(src, "judge-sample", "--replicate"),
        first,
        repeat,
    )
    doc.given("each with a {}% bootstrap interval", comparisons.level(), reason=INTERVAL_LEVEL)
    doc.check(
        "of the {} dimensions in `generation.calibration_thresholds`",
        formatting.count(len(src.default_config.generation.calibration_thresholds)),
    )
    doc.definition("An answer shared by two configurations", reason=COMMAND_RULE)
    # The `--v2` row names `--v1` before its schema; a template types `--` as an en dash.
    for lead, schema in (("of prompt v1", FIRST_SCHEMA), ("v1`", PARTIAL_SCHEMA)):
        doc.given(
            lead + ": its answers are of answer schema {}",
            formatting.own_form(schema),
            reason="the schema each prompt pairs with",
        )


def _service(doc: Doc) -> None:
    doc.definition("on 127.0.0.1:8000", reason=SERVE_ADDRESS)
    doc.definitions(("is a 422", "is a 502", "is a 503"), reason=STATUS_CODE)


def _reporting(doc: Doc) -> None:
    doc.definition("exits 1 if", reason="the exit status `cli.reporting` sets")
