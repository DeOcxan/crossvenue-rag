"""Which candidates the annotator judges: the union of every pool method's top hits.

A pool from one retriever would bake its blind spots into the gold and make it unbeatable by
construction; the union narrows pooling bias without removing it (docs/how-to/label.md).
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from .. import formatting
from ..config import Config
from ..dataset.journal import Judgment
from ..errors import UserError
from ..models import Chunk, Question, Result
from ..retrieval.fusion import reciprocal_rank_fusion
from ..retrieval.methods import POOL_METHODS
from ..retrieval.protocols import LabelRetriever

# How far each method's depth may grow, as a multiple of the configured depth,
# when the methods overlap so much that the fused list is short of `limit`.
_MAX_DEPTH_MULTIPLIER = 4


def gather_candidates(
    retriever: LabelRetriever, question: Question, per_method: int, limit: int
) -> list[Result]:
    """Top candidates pooled across the five pool methods, ranked by RRF.

    If the fused pool holds fewer than `limit` distinct chunks (the methods
    overlapping heavily), each method is asked for more, up to
    `_MAX_DEPTH_MULTIPLIER` times the configured depth. The annotator then judges
    `limit` distinct passages unless that cap comes first. After deduplication,
    distinct ids are distinct texts.
    """
    depth = per_method
    while True:
        by_method: dict[str, list[str]] = {}
        seen: dict[str, Result] = {}
        # The pool is built from POOL_METHODS only: never an oracle, and never
        # a method whose unjudged results on older questions were not first
        # judged by augmentation (evals/methodology.md, section 1).
        for method in POOL_METHODS:
            results = retriever.retrieve(question.question, method=method, k=depth)
            by_method[method] = [r.chunk_id for r in results]
            for result in results:
                seen.setdefault(result.chunk_id, result)
        if len(seen) >= limit or depth >= per_method * _MAX_DEPTH_MULTIPLIER:
            break
        depth += per_method

    fused = reciprocal_rank_fusion(by_method, k=retriever.cfg.retrieval.rrf_k)
    pooled: list[Result] = []
    for item in fused[:limit]:
        result = seen[item.chunk_id]
        result.score = item.score
        result.component_ranks = item.ranks
        pooled.append(result)
    return pooled


def review_pool(
    retriever: LabelRetriever,
    question: Question,
    entry: Judgment,
    per_method: int,
    limit: int,
) -> list[Result]:
    """The exact pool round 1 recorded, in its order, with fresh method ranks.

    Re-gathering could return a different pool (depth top-up, a different
    reranker build); reviewing on the recorded one keeps the judged set, which
    agreement is computed over, unchanged.
    """
    fresh = {r.chunk_id: r for r in gather_candidates(retriever, question, per_method, limit)}
    pool: list[Result] = []
    for chunk_id in entry.pool:
        result = fresh.get(chunk_id)
        if result is None:
            result = Result.from_chunk(retriever.by_id[chunk_id], 0.0)
        pool.append(result)
    return pool


# Seed of the round-2 candidate order; recorded in every round-2 journal entry.
RETEST_SHUFFLE_SEED = 20260924


def retest_candidates(
    chunks: Mapping[str, Chunk], question_id: str, round1_entry: Judgment, seed: int
) -> list[Result]:
    """Round 2's candidates: exactly what round 1 judged, in a shuffled order.

    The set is round 1's recorded pool plus every chunk its augmentations showed
    (`Judgment.judged_ids`), so kappa compares the two rounds on the same
    candidates. The order is shuffled, seeded per question, so position in the
    list is no cue to the first verdict. No method ranks are attached: which
    retriever found a chunk is not part of the criterion.

    Raises:
        UserError: A chunk round 1 judged is not in `chunks`.
    """
    ids = round1_entry.judged_ids()
    missing = [chunk_id for chunk_id in ids if chunk_id not in chunks]
    if missing:
        raise UserError(
            f"{question_id}: {formatting.plural(len(missing), 'round-1 chunk')} "
            "not in the current index",
            hint="round 1 was judged against another index; carry ids with `remap-ids`",
        )
    return [
        Result.from_chunk(chunks[chunk_id], 0.0)
        for chunk_id in retest_order(question_id, round1_entry, seed)
    ]


def retest_order(question_id: str, round1_entry: Judgment, seed: int) -> list[str]:
    """The chunks round 1 judged, in the order round 2 showed them (seeded per question)."""
    order = round1_entry.judged_ids()
    random.Random(f"{seed}:{question_id}").shuffle(order)
    return order


def pool_depth_of(cfg: Config, override: int | None, option: str | None = None) -> int:
    """The pool depth: the config's, or an override held to the config's bound.

    A pool is never only as deep as the ranks the metrics score (ADR 0005). `option` is the
    command-line option the override came from, which the message names.

    Raises:
        UserError: `override` is not deeper than the deepest rank the metrics score.
    """
    if override is None:
        return cfg.eval.pool_depth_per_method
    deepest = max(cfg.eval.k_values)
    if override <= deepest:
        given = f"{option} {formatting.own_form(override)}: " if option else ""
        raise UserError(
            f"{given}a pool depth of {formatting.count(override)} does not exceed "
            f"max(eval.k_values) ({formatting.count(deepest)})",
            hint=f"give a depth above {formatting.count(deepest)}, so the judged set reaches past "
            "the scored ranks (ADR 0005)",
        )
    return override
