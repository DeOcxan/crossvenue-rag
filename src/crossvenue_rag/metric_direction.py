"""Which way each metric improves, of retrieval or of answers.

A paired comparison counts a question as better for a when a moves in the
metric's direction, so an off-venue share or an attribution error that falls is
an improvement.
"""

from __future__ import annotations

HIGHER_IS_BETTER: dict[str, bool] = {
    # Retrieval metrics, by family (`recall@10` is family `recall`).
    "mrr": True,
    "recall": True,
    "precision": True,
    "ndcg": True,
    "ndcg_prime": True,
    "venue_coverage": True,
    "unjudged": False,
    "off_venue": False,
    "via_ccxt": False,
    # Answer metrics. faithfulness is the share of claims supported;
    # wrong_refusal is the prompt comparison's refused-when-an-answer-was-expected
    # indicator.
    "faithfulness": True,
    "venue_attribution_error": False,
    "coverage": True,
    "behaviour_correct": True,
    "wrong_refusal": False,
}


def higher_is_better(metric: str) -> bool:
    """Whether a larger value of `metric` is better.

    Args:
        metric: A metric name with or without its cutoff: `recall@10`, `off_venue@1`,
            `coverage`.

    Raises:
        ValueError: No direction is recorded for the metric's family.
    """
    family = metric.split("@", 1)[0]
    if family not in HIGHER_IS_BETTER:
        raise ValueError(f"no direction recorded for metric {metric!r}")
    return HIGHER_IS_BETTER[family]
