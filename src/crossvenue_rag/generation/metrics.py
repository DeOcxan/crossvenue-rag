"""Answer metrics derived from a judgment."""

from __future__ import annotations

from typing import get_args

from pydantic import BaseModel, ConfigDict

from ..config_generation import AnswerMetric
from ..stats import mean
from .schemas import Judgement

# The metrics in table order; the config's `generation.primary_metrics` names from these.
METRICS: tuple[AnswerMetric, ...] = get_args(AnswerMetric)


class AnswerMetrics(BaseModel):
    """The metrics of one judged answer, as its judge record saves them.

    A metric that does not apply to the answer is None.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    faithfulness: float | None
    venue_attribution_error: float | None
    coverage: float | None
    behaviour_correct: float

    def value(self, metric: str) -> float | None:
        """The metric named `metric`, one of `METRICS`.

        Raises:
            ValueError: `metric` is not one of `METRICS`.
        """
        if metric not in METRICS:
            raise ValueError(f"{metric!r} is not an answer metric, one of {', '.join(METRICS)}")
        value: float | None = getattr(self, metric)
        return value


def answer_metrics(judgement: Judgement) -> AnswerMetrics:
    """The metrics a judgment gives its answer.

    - `faithfulness`: share of claims `supported` by the chunks they cite.
    - `venue_attribution_error`: share of claims attributed to the wrong venue.
    - `coverage`: per venue with gold facts, `covered_facts` / `gold_facts`,
      averaged over those venues, so a comparison answer that covers one side
      scores 0.5. A venue the judge finds no gold fact for is left out.
    - `behaviour_correct`: 1.0 when the answer answered, refused or asked, as
      the question expects; else 0.0.

    An answer with no claims (a refusal) has no faithfulness and no attribution
    error. A question with no gold facts has no coverage: None, never 0, so an
    average is taken over the answers the metric applies to.
    """
    claims = judgement.claims
    coverage = [c for c in judgement.coverage if c.gold_facts > 0]
    return AnswerMetrics(
        faithfulness=mean([c.supported for c in claims]) if claims else None,
        venue_attribution_error=mean([not c.venue_correct for c in claims]) if claims else None,
        coverage=mean([c.covered_facts / c.gold_facts for c in coverage]) if coverage else None,
        behaviour_correct=1.0 if judgement.behaviour_correct else 0.0,
    )
