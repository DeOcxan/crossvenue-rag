"""Answer runs: their config, records, outcomes and paired comparisons.

`src.v1` and `src.v2` are the answer runs of generator prompts v1 and v2
(`run_ids.ANSWERS_V1`, `run_ids.ANSWERS_V2`); `src.judged_v1` and
`src.judged_v2` their judged runs (`run_ids.JUDGED_V1`, `run_ids.JUDGED_V2`). A
function that takes `prompt` reads the run of that prompt version, 1 or 2, and raises
ValueError for another.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from ...corpus.artifacts import CHUNKS_FILENAME
from ...errors import UserError
from ...evaluation.paired import holm
from ...evaluation.power import SIGNIFICANCE_LEVEL
from ...generation.answer_rows import SCOPES, in_scope
from ...generation.answering import split_answer_id
from ...generation.attribution import AttributionSummary, attribution_summary
from ...generation.metrics import METRICS
from ...generation.records import AnswerRecord
from ...generation.schemas import GeneratedAnswer, Outcome, outcome
from ...models import COMPARISON, MULTI_VENUE
from ...retries import RETRY_STATUSES
from ...run_files import CONFIG_FILE, PAIRED_FILE
from ..sources import AnswerRun, Sources
from .estimates import Estimate
from .lookup import entry
from .memo import per_sources

# A citation written in an answer's text: `[1]`, `[ 2`.
CITATION = re.compile(r"\[\s*\d")
# The generator prompt versions, each with an answer run and a judged run.
PROMPT_V1, PROMPT_V2 = 1, 2
PROMPTS = (PROMPT_V1, PROMPT_V2)


def answer_run(src: Sources, prompt: int) -> AnswerRun:
    """The answer run of prompt version `prompt`; ValueError unless it is 1 or 2.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return entry({PROMPT_V1: src.v1, PROMPT_V2: src.v2}, prompt, "the answer runs by prompt")


def judged_run(src: Sources, prompt: int) -> AnswerRun:
    """The judged run of prompt version `prompt`; ValueError unless it is 1 or 2.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    runs = {PROMPT_V1: src.judged_v1, PROMPT_V2: src.judged_v2}
    return entry(runs, prompt, "the judged runs by prompt")


def retried_statuses() -> list[int]:
    """The HTTP statuses the model client retries, ascending (`retries.RETRY_STATUSES`)."""
    return sorted(RETRY_STATUSES)


def generator_temperature(src: Sources) -> float:
    """The generators' temperature in the prompt v1 answer run's config.

    Raises:
        ValueError: the config sets no `generation.generator_params.temperature`.
    """
    temperature = src.v1.config.generation.generator_params.temperature
    if temperature is None:
        raise ValueError(f"{src.v1.dir} sets no generation.generator_params.temperature")
    return temperature


def generator_max_tokens(src: Sources) -> int:
    """The generators' output token limit in the prompt v1 answer run's config."""
    return src.v1.config.generation.generator_params.max_tokens


def max_invalid_retries(src: Sources) -> int:
    """Retries of a response that fails validation, in the prompt v1 answer run's config."""
    return src.v1.config.generation.max_invalid_retries


def primary_metrics(src: Sources) -> tuple[str, ...]:
    """The primary metrics of the prompt v1 judged run, which Holm adjusts within."""
    return tuple(src.judged_v1.config.generation.primary_metrics)


def answer_top_k(src: Sources) -> int:
    """The chunks each answer of the prompt v1 run is generated from (`generation.top_k`)."""
    return src.v1.config.generation.top_k


def configurations(src: Sources, run: AnswerRun | None = None) -> list[str]:
    """The answer configurations of `run`, by default the prompt v1 answer run.

    In the order of the run's `generation.matrix`.
    """
    return [entry.name for entry in (run or src.v1).config.generation.matrix]


def generators(src: Sources) -> list[str]:
    """The generators of `configurations`, each once, in the order the matrix names them."""
    return list(dict.fromkeys(entry.generator for entry in src.v1.config.generation.matrix))


def model_families(src: Sources) -> int:
    """The model families of `generators`: the vendors their model ids name."""
    named = src.v1.config.generation.generators
    return len({named[g].model.split("/")[0] for g in generators(src)})


def primary_pair(src: Sources) -> tuple[str, str]:
    """The pre-registered primary pair of answer configurations (`generation.primary_pair`).

    As the prompt v1 judged run's config sets it.

    Raises:
        UserError: config.yaml of the prompt v1 judged run sets no `generation.primary_pair`.
    """
    pair = src.judged_v1.config.generation.primary_pair
    if not pair:
        raise UserError(
            f"{src.judged_v1.dir / CONFIG_FILE} sets no generation.primary_pair",
            hint="point run_ids.JUDGED_V1 at a judged run whose config sets "
            "generation.primary_pair",
        )
    a, b = pair
    return a, b


def refusing_config(src: Sources) -> str:
    """The configuration whose comparison refusals the documents break down.

    The first of the pre-registered primary pair (`primary_pair`).
    """
    return primary_pair(src)[0]


def judge_is(src: Sources, judge: str, model: str) -> bool:
    """Whether the judged run's judge is `judge` with the model id `model`."""
    generation = src.judged_v1.config.generation
    return generation.judge == judge and generation.judges[judge].model == model


def judged_means(
    src: Sources, pair: tuple[str, str], metric: str, scope: str
) -> tuple[float, float]:
    """(mean of a, mean of b) on `metric` over `scope` in the prompt v1 judged run."""
    row = entry(src.judged_v1.paired, (*pair, metric, scope), PAIRED_FILE)
    return row.mean_a, row.mean_b


def judged_interval(
    src: Sources, prompt: int, pair: tuple[str, str], metric: str, scope: str
) -> Estimate:
    """A - b in a judged run, with its interval.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return Estimate.of(entry(judged_run(src, prompt).paired, (*pair, metric, scope), PAIRED_FILE))


def judged_includes_zero(
    src: Sources, prompt: int, pair: tuple[str, str], metric: str, scope: str
) -> bool:
    """Whether the interval of a - b in a judged run includes 0.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return judged_interval(src, prompt, pair, metric, scope).includes_zero()


def judged_p(
    src: Sources, prompt: int, pair: tuple[str, str], metric: str, scope: str, column: str
) -> float:
    """The p-value `column` (`randomisation_p`, `holm_p`) of a - b in a judged run.

    Raises:
        ValueError: `prompt` is not 1 or 2, or `column` is no column of paired.csv.
    """
    return entry(judged_run(src, prompt).paired, (*pair, metric, scope), PAIRED_FILE).value(column)


def judged_role(src: Sources, prompt: int, pair: tuple[str, str], metric: str, scope: str) -> str:
    """The role (`primary`, `secondary`) of a - b in a judged run.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return entry(judged_run(src, prompt).paired, (*pair, metric, scope), PAIRED_FILE).role


def judged_questions(
    src: Sources, prompt: int, pair: tuple[str, str], metric: str, scope: str
) -> int:
    """Questions paired in a - b on `metric` over `scope` in a judged run.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return entry(judged_run(src, prompt).paired, (*pair, metric, scope), PAIRED_FILE).n


def judged_holm(
    src: Sources, prompt: int, pair: tuple[str, str], metrics: Sequence[str], scope: str
) -> list[float]:
    """p_Holm of each of `metrics` in a judged run.

    Holm-adjusted across `metrics` from their randomization p-values.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return holm([judged_p(src, prompt, pair, m, scope, "randomisation_p") for m in metrics])


def primary_holm(
    src: Sources, prompt: int, pair: tuple[str, str], metric: str, scope: str
) -> float:
    """p_Holm of `metric`, a `generation.primary_metrics` of the judged run, within them.

    Raises:
        ValueError: `prompt` is not 1 or 2, or `metric` is not a primary metric of the run.
    """
    family = [str(m) for m in judged_run(src, prompt).config.generation.primary_metrics]
    if metric not in family:
        raise ValueError(f"{metric} is not a primary metric of the prompt v{prompt} judged run")
    return judged_holm(src, prompt, pair, family, scope)[family.index(metric)]


@per_sources
def attribution(src: Sources, prompt: int) -> dict[str, AttributionSummary]:
    """Misattributed claims by kind per configuration of a judged run.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    run = judged_run(src, prompt)
    return attribution_summary(run.records, src.venue_of, src.doc_path_of, configurations(src, run))


def attributed_claims(src: Sources, prompt: int) -> int:
    """The claims the attribution table of a judged run counts, over every configuration.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return sum(summary.claims for summary in attribution(src, prompt).values())


def misattributed_count(src: Sources) -> int:
    """Misattributed claims of every kind, over both judged runs and all configurations."""
    return sum(
        sum(summary.misattributed.values())
        for prompt in PROMPTS
        for summary in attribution(src, prompt).values()
    )


def misattributions_are_wrappers(src: Sources) -> bool:
    """Whether every misattributed claim of both judged runs is of the wrapper kind.

    The wrapper kind cites a CCXT page for an exchange; no claim may be a
    cross-exchange or other misattribution.
    """
    return all(
        x.misattributed["cross_exchange"] == 0 and x.misattributed["other"] == 0
        for prompt in PROMPTS
        for x in attribution(src, prompt).values()
    )


def answer_count(src: Sources) -> int:
    """Answers of the prompt v1 answer run."""
    return len(src.v1.records)


def answered_question_count(src: Sources) -> int:
    """Questions the prompt v1 answer run answers."""
    return len({r.question_id for r in src.v1.records})


def configuration_pairs(src: Sources) -> int:
    """Unordered pairs of the answer configurations."""
    return len(list(combinations(configurations(src), 2)))


def paired_comparison_count(src: Sources) -> int:
    """Paired answer comparisons: every configuration pair on every metric and scope."""
    return configuration_pairs(src) * len(METRICS) * len(SCOPES)


def expected_false_positives(src: Sources) -> int:
    """Comparisons expected significant by chance when every null holds, rounded."""
    return round(paired_comparison_count(src) * SIGNIFICANCE_LEVEL)


@per_sources
def outcomes(src: Sources, prompt: int) -> dict[tuple[str, str], Outcome]:
    """(config, question id) -> outcome under the rule of methodology section 9a.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return {
        (r.config, r.question_id): outcome(r.generation.answer)
        for r in answer_run(src, prompt).records
        if r.generation is not None
    }


def outcome_count(
    src: Sources, config: str, category: str | None, kind: Outcome, prompt: int
) -> int:
    """Answers of `config` with outcome `kind` in the answer run of `prompt`.

    On the questions of `category`, or on every question when it is None.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return sum(
        1
        for (c, q), o in outcomes(src, prompt).items()
        if c == config
        and o == kind
        and (
            category is None
            or entry(src.questions, q, "the gold set").effective_category == category
        )
    )


def comparison_refusals(src: Sources, config: str) -> int:
    """Refusals of comparison questions by `config` in the prompt v1 answer run."""
    return outcome_count(src, config, COMPARISON, "refused", PROMPT_V1)


def wrong_refusals(src: Sources, config: str, prompt: int = PROMPT_V1) -> int:
    """Refusals of `config` of questions that do not expect one, in the run of `prompt`.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return sum(
        1
        for (c, q), o in outcomes(src, prompt).items()
        if c == config
        and o == "refused"
        and not entry(src.questions, q, "the gold set").expects_refusal
    )


def _record(src: Sources, config: str, question_id: str) -> AnswerRecord:
    for r in src.v1.records:
        if r.config == config and r.question_id == question_id:
            return r
    raise ValueError(f"the prompt v1 answer run has no answer of {config} to {question_id}")


def answer(src: Sources, config: str, question_id: str) -> GeneratedAnswer:
    """`config`'s generated answer to one question in the prompt v1 run.

    Raises:
        ValueError: the run has no answer of `config` to `question_id`.
    """
    return _record(src, config, question_id).generated().answer


def claim_count(src: Sources, answer_id: str) -> int:
    """Claims of one answer of the prompt v1 answer run (`config:question id`)."""
    return len(answer(src, *split_answer_id(answer_id)).claims)


def claims_mention(src: Sources, answer_id: str, values: Sequence[str]) -> bool:
    """Whether the claims of one prompt v1 answer, joined, contain each of `values`."""
    text = " ".join(c.text for c in answer(src, *split_answer_id(answer_id)).claims)
    return all(v in text for v in values)


@per_sources
def refusal_groups(src: Sources) -> dict[str, list[str]]:
    """`refusing_config`'s prompt v1 comparison refusals by the gold in its context.

    Group `a`: a gold chunk of every named venue in the context; `b`: of some;
    `c`: of none.
    """
    groups: dict[str, list[str]] = {"a": [], "b": [], "c": []}
    for r in src.v1.records:
        if r.config != refusing_config(src) or not in_scope(r, COMPARISON):
            continue
        if outcome(r.generated().answer) != "refused":
            continue
        venues = entry(src.questions, r.question_id, "the gold set").venues
        have = _gold_venues_seen(src, r)
        key = "a" if len(have) == len(venues) else "b" if have else "c"
        groups[key].append(r.question_id)
    return groups


def _gold_venues_seen(src: Sources, record: AnswerRecord) -> list[str]:
    """The named venues of which a gold chunk is in an answer record's context."""
    q = entry(src.questions, record.question_id, "the gold set")
    gold = set(q.gold_chunk_ids)
    seen = record.generated().chunk_ids
    return [
        v
        for v in q.venues
        if any(c in gold and entry(src.venue_of, c, CHUNKS_FILENAME) == v for c in seen)
    ]


def refusal_ids(src: Sources) -> list[str]:
    """`refusing_config`'s prompt v1 comparison refusals, group a, then b, then c."""
    return [q for group in refusal_groups(src).values() for q in group]


def context_venues(src: Sources, question_id: str) -> Counter[str]:
    """Chunks per venue in the context of `refusing_config`'s prompt v1 answer (`answer`).

    Raises:
        ValueError: the prompt v1 answer run has no answer of `refusing_config` to `question_id`.
    """
    record = _record(src, refusing_config(src), question_id)
    return Counter(entry(src.venue_of, c, CHUNKS_FILENAME) for c in record.generated().chunk_ids)


def refusals_with_every_venue(src: Sources) -> list[str]:
    """Of `refusal_ids`, those whose context held a chunk of every named venue."""
    return [
        q
        for q in refusal_ids(src)
        if all(context_venues(src, q)[v] for v in src.questions[q].venues)
    ]


def even_context_chunks(src: Sources) -> int | None:
    """The chunks per venue when every context of `refusals_with_every_venue` splits evenly.

    The top k split equally between the `MULTI_VENUE` venues of a comparison; None when one
    of them does not split so.
    """
    half = answer_top_k(src) // MULTI_VENUE
    even = all(
        set(context_venues(src, q).values()) == {half} for q in refusals_with_every_venue(src)
    )
    return half if even else None


def refusal_group_sizes(src: Sources) -> list[tuple[str, int]]:
    """(group, refusals) of each of `refusal_groups`, a to c."""
    return [(g, len(ids)) for g, ids in refusal_groups(src).items()]


def gold_less_venue(src: Sources, question_id: str) -> bool:
    """Whether a venue the question names has no gold chunk; ValueError if the gold set lacks it.

    Raises:
        ValueError: the gold set has no question `question_id`.
    """
    q = entry(src.questions, question_id, "the gold set")
    return any(not any(src.venue_of.get(c) == v for c in q.gold_chunk_ids) for v in q.venues)


def gold_less_refusals(src: Sources, group: str) -> int:
    """Refusals of one of `refusal_groups` that name a venue with no gold chunk.

    Raises:
        ValueError: `group` is not `a`, `b` or `c`.
    """
    return sum(
        gold_less_venue(src, q) for q in entry(refusal_groups(src), group, "the refusal groups")
    )


def claimless_partial_configs(src: Sources) -> list[str]:
    """The configuration of each prompt v2 answer flagged partial that states no claim."""
    return [
        r.config
        for r in src.v2.records
        if r.generated().answer.partial and not r.generated().answer.claims
    ]


def claimless_partials_all_from(src: Sources, config: str) -> bool:
    """Whether every claim-less partial prompt v2 answer is `config`'s."""
    return set(claimless_partial_configs(src)) == {config}
