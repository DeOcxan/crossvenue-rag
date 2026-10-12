"""The names of the retrieval methods: every method, the labeling pool and the oracle.

Kept apart from `retriever`, so a module that only names methods loads none of the
retriever's indexes or their libraries.
"""

from __future__ import annotations

from typing import get_args

from ..config_pipeline import RetrievalMethod
from ..errors import UserError

METHODS: tuple[RetrievalMethod, ...] = get_args(RetrievalMethod)
# The methods whose results make up a labeling pool. Changing it leaves the pools
# recorded in the journal as they are, but a method added here must first have its
# unjudged results on labeled questions judged by augmentation
# (evals/methodology.md, section 1). The oracle can never be one: it reads the labels.
POOL_METHODS: tuple[RetrievalMethod, ...] = (
    "bm25",
    "dense",
    "hybrid",
    "hybrid_rerank",
    "venue_split",
)
ORACLE_METHODS: tuple[RetrievalMethod, ...] = ("venue_split_oracle",)
# The methods an ad-hoc question can use: an oracle reads a question's labeled venues,
# which an ad-hoc question does not have.
AD_HOC_METHODS: tuple[RetrievalMethod, ...] = tuple(m for m in METHODS if m not in ORACLE_METHODS)


def _named(name: str, option: str, methods: tuple[RetrievalMethod, ...]) -> RetrievalMethod:
    for method in methods:
        if method == name:
            return method
    raise UserError(
        f"{option} {name!r} is not a retrieval method",
        hint=f"pass {option} one of: {', '.join(methods)}",
    )


def method_named(name: str) -> RetrievalMethod:
    """`name`, the value of `--method`, as a retrieval method.

    Raises:
        UserError: `name` is no method; the message names `--method` and the value.
    """
    return _named(name, "--method", METHODS)


def ad_hoc_method(name: str, option: str) -> RetrievalMethod:
    """`name`, the value of `option`, as a method an ad-hoc question can use.

    Raises:
        UserError: `name` is an oracle, which reads a question's labeled venues, or no method;
            the message names `option` and the hint the methods it takes.
    """
    if name in ORACLE_METHODS:
        raise UserError(
            f"{option} {name}: it reads a question's labeled venues, which an ad-hoc question "
            "does not have",
            hint=f"pass {option} one of: {', '.join(AD_HOC_METHODS)}; the oracle runs inside "
            "`crossvenue eval`",
        )
    return _named(name, option, AD_HOC_METHODS)
