"""Which candidates the annotator judges: the top of the fused union of the pool methods' hits.

A gold from one retriever's pool would hold only the relevant chunks that retriever finds and
make it unbeatable by construction; the union narrows pooling bias without removing it
(evals/methodology.md, section 1).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from .. import formatting
from ..config import Config
from ..dataset.journal import Judgment, candidate_ids
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
        # a method whose results on older questions that round 1 has not judged were
        # not first judged by augmentation (evals/methodology.md, section 1).
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
    """What round 1 judged (`candidate_ids`), in its order, with fresh method ranks.

    That is the recorded pool, then every chunk its augmentations showed: the set round 2
    shows. Re-gathering could return a different
    pool (depth top-up, a different reranker build); reviewing on the judged set, which
    agreement is computed over, keeps it unchanged.
    """
    fresh = {r.chunk_id: r for r in gather_candidates(retriever, question, per_method, limit)}
    pool: list[Result] = []
    for chunk_id in candidate_ids(entry):
        result = fresh.get(chunk_id)
        if result is None:
            result = Result.from_chunk(retriever.by_id[chunk_id], 0.0)
        pool.append(result)
    return pool


# Seed of the round-2 candidate order; recorded in every round-2 journal entry.
RETEST_SHUFFLE_SEED = 20260924
# Seed of the round-3 candidate order (methodology section 5b). Round 2's seed would repeat
# round 2's order on every sampled question whose candidates did not change.
ROUND3_SHUFFLE_SEED = 20261031


def retest_candidates(
    chunks: Mapping[str, Chunk],
    question_id: str,
    round1_entry: Judgment,
    seed: int | None,
    added: Sequence[str] = (),
) -> list[Result]:
    """A later pass's candidates: what round 1 judged and any `added` chunks.

    The set is round 1's recorded pool plus every chunk its augmentations showed
    (`Judgment.judged_ids`), so `kappa` compares the two rounds on the same
    candidates. Round 3 adds the `added` chunks, shuffled in among them. With a `seed`, the
    order is shuffled per question, so position in the list is no cue to the first
    verdict. With `seed` None, the order is the recorded one, then the `added` chunks, as
    `--relabel` shows them. No method ranks are attached: which retriever found a chunk
    is not part of the criterion.

    Raises:
        UserError: A chunk to show is not in `chunks`.
    """
    ids = candidate_ids(round1_entry, added)
    missing = [chunk_id for chunk_id in ids if chunk_id not in chunks]
    if missing:
        raise UserError(
            f"{question_id}: {formatting.plural(len(missing), 'chunk')} not in the current index",
            hint="round 1 was judged against another index, or the added-chunks file names a "
            "chunk it lacks; carry ids with `remap-ids`",
        )
    order = ids if seed is None else retest_order(question_id, round1_entry, seed, added)
    return [Result.from_chunk(chunks[chunk_id], 0.0) for chunk_id in order]


def retest_order(
    question_id: str, round1_entry: Judgment, seed: int, added: Sequence[str]
) -> list[str]:
    """The chunks a later round shows, in the order it shows them (seeded per question)."""
    order = candidate_ids(round1_entry, added)
    random.Random(f"{seed}:{question_id}").shuffle(order)
    return order


def pool_depth_of(cfg: Config, override: int | None, option: str) -> int:
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
        raise UserError(
            f"{option} {formatting.own_form(override)}: a pool depth of "
            f"{formatting.count(override)} does not exceed "
            f"max(eval.k_values) ({formatting.count(deepest)})",
            hint=f"give a depth above {formatting.count(deepest)}, so the judged set reaches past "
            "the scored ranks (ADR 0005)",
        )
    return override
