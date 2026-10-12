"""What docs/reference/config.md claims.

The tables come first, in document order, then one function per section for the rest of its
text, in document order.
"""

from __future__ import annotations

from ... import formatting
from ...config_eval import TOP_ONE_K
from ...config_generation import PRIMARY_PAIR_SIZE
from ...config_pipeline import EF_CONSTRUCTION_PER_M
from ...generation.schemas import FIRST_SCHEMA, PARTIAL_SCHEMA
from ..check import Doc
from ..compute import config_reference as schema
from ..outcomes import SKIP
from ..sources import Sources
from .comparisons import retried_status_parts
from .corpus_datasheet import ENCODER_TEXT_TOKENS
from .methodology import protocol
from .reasons import BGE_SPECIAL_TOKENS, BGE_TOKENIZER, BGE_WINDOW, CITED

# The sections whose tables hold a number, in document order.
SECTIONS = (
    "chunking",
    "embedding",
    "reranker",
    "retrieval",
    "store",
    "fetch",
    "eval",
    "generation",
)
SCHEMA_RULE = "a rule the schema checks"


def config_reference(doc: Doc, src: Sources) -> None:
    """States the claims of docs/reference/config.md."""
    for section in SECTIONS:
        _table(doc, src, section)
    _chunking(doc)
    _retrieval(doc)
    _store(doc)
    _fetch(doc)
    _eval(doc, src)
    _generation(doc, src)
    _validation(doc)


def _table(doc: Doc, src: Sources, section: str) -> None:
    """The table of `section`: one row per key, its domain and its number compared."""
    model = schema.section_model(section)
    rows: dict[str, list[object]] = {}
    for key in model.model_fields:
        domain = schema.field_domain(section, key)
        value = schema.number_value(src, section, key)
        rows[f"`{key}`"] = [
            SKIP if domain is None else domain,
            SKIP if value is None else value,
            SKIP,
        ]
    doc.table(f"## `{section}`", rows)


def _chunking(doc: Doc) -> None:
    doc.given(
        "({} for `bge-base-en-v1.5`, {} with its {} special tokens)",
        formatting.count(ENCODER_TEXT_TOKENS),
        formatting.count(BGE_WINDOW),
        formatting.count(BGE_SPECIAL_TOKENS),
        reason=BGE_TOKENIZER,
    )


def _store(doc: Doc) -> None:
    doc.check("at least {} `hnsw_m`", formatting.count(EF_CONSTRUCTION_PER_M))


def _retrieval(doc: Doc) -> None:
    doc.definition("(Cormack et al. 2009;", reason=CITED)


def _fetch(doc: Doc) -> None:
    doc.definition("so pacing can be 0", reason="the least value of `request_delay_seconds`")


def _eval(doc: Doc, src: Sources) -> None:
    doc.definition("no id twice", reason=SCHEMA_RULE)
    doc.check("all {}: `bm25`", formatting.count(schema.yaml_length(src, "eval", "methods")))
    doc.check(
        "(schema default: the first {})",
        formatting.count(schema.schema_default_length("eval", "methods")),
    )
    doc.check("| all {} | Slices", formatting.count(schema.yaml_length(src, "eval", "breakdowns")))
    doc.definition("distinct int > 0", reason=SCHEMA_RULE)
    doc.check("that includes {} |", formatting.count(TOP_ONE_K))
    doc.check("not positive, without {} or", formatting.count(TOP_ONE_K))
    doc.definition("one of that metric's two", reason="the two scopes the parenthesis lists")


def _generation(doc: Doc, src: Sources) -> None:
    holes, statuses = retried_status_parts()
    doc.check("on the statuses " + holes + ", or on", *statuses)
    doc.check("{} lowercase hex digits", formatting.count(schema.sha256_digits()))
    for text, key in (
        ("`{}` (full value in `configs/default.yaml`) | A call", "generator_prompt_sha256"),
        ("`{}` (full value in `configs/default.yaml`) | A judge call", "judge_prompt_sha256"),
        ("`{}` (prompt v1; full value", "primary_prompt_sha256"),
    ):
        pin = str(schema.yaml_value(src, "generation", key))
        doc.ref(text, formatting.short_hash(pin))
    pairs = "the answer schema each prompt version pairs with"
    doc.given("Answer schema: {} adds", formatting.own_form(PARTIAL_SCHEMA), reason=pairs)
    doc.given(
        "prompt v1 needs {}, prompt v2 needs {}",
        formatting.own_form(FIRST_SCHEMA),
        formatting.own_form(PARTIAL_SCHEMA),
        reason=pairs,
    )
    doc.check(
        "| {} configurations | What",
        formatting.count(schema.yaml_length(src, "generation", "matrix")),
    )
    doc.check("| empty, or {} different `matrix` names", formatting.count(PRIMARY_PAIR_SIZE))
    doc.check(
        "the {} trial questions",
        formatting.count(schema.yaml_length(src, "generation", "calibration_exclude")),
    )
    # The kappa dimensions written with one value: a second value fails the comparison.
    doc.check(
        "`behaviour_correct` {} (Cohen's \u03ba); `coverage` {} (Spearman)",
        *protocol.judge_thresholds(src),
    )
    doc.definition(
        "[\u22121, 1]", reason="the range of Cohen's \u03ba and of Spearman's correlation"
    )
    run = str(schema.yaml_value(src, "generation", "calibration_run"))
    doc.ref("| `results/{}` | Answer run", run.removeprefix("results/"))


def _validation(doc: Doc) -> None:
    doc.check("< {} \u00d7 `store.hnsw_m`", formatting.count(EF_CONSTRUCTION_PER_M))
    doc.definitions(
        (
            "with a cut-off twice",
            "with a method twice",
            "a result counted twice",
            "naming a pair twice",
            "one Holm counts twice",
            "would take as 1",
            "not two different configurations",
            "Two configurations would write",
        ),
        reason=SCHEMA_RULE,
    )
