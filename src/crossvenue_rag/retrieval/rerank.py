"""Cross-encoder reranking of the top-N candidates.

A cross-encoder reads the query and the chunk together, where a bi-encoder
compresses the chunk into one vector before it sees the query. It is the most
expensive stage (docs/results/findings.md, the cross-encoder's share of
hybrid_rerank's p50 latency), so it scores the top N candidates only;
sentence_transformers is imported when the model is first loaded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config_pipeline import RerankerConfig
from ..errors import UserError
from ..formatting import revision_text
from ..index.embeddings import MODEL_HINT, resolve_device
from ..models import Result

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)


class Reranker:
    """The configured cross-encoder, loaded on first use, rescoring retrieval candidates."""

    def __init__(
        self, cfg: RerankerConfig, device: str | None, model: CrossEncoder | None = None
    ) -> None:
        """A reranker for `cfg` on `device` (None picks one).

        The model loads on first use unless `model` is given.
        """
        self.cfg = cfg
        self.device = device
        self._model = model

    @property
    def model(self) -> CrossEncoder:
        """The cross-encoder, loaded at the configured revision on first access.

        Raises:
            UserError: The model or revision cannot be loaded.
        """
        if self._model is None:
            resolved = resolve_device(self.device)
            log.info("loading reranker %s on %s", self.cfg.model_name, resolved)
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.cfg.model_name, device=resolved, revision=self.cfg.revision
                )
            except (OSError, ValueError, ImportError) as exc:
                raise UserError(
                    f"cannot load reranker.model_name {self.cfg.model_name}"
                    f"{revision_text(self.cfg.revision)}: {exc}",
                    hint=MODEL_HINT,
                ) from exc
        return self._model

    def rerank(self, query: str, results: list[Result], k: int) -> list[Result]:
        """The best k of the top-N candidates, rescored.

        Scores are replaced, not blended: mixing a fused RRF score with a
        cross-encoder logit would produce a number with no meaning. The fusion
        ranks stay on each Result for inspection. The caller's Result objects
        are rescored in place (`score`), so a list that shares them holds
        the cross-encoder's scores too.
        """
        return self.rerank_many(query, [results], k)[0]

    def rerank_many(self, query: str, groups: list[list[Result]], k: int) -> list[list[Result]]:
        """`rerank` for several candidate lists at once, in one cross-encoder call.

        `venue_split` reranks one list per named venue. Scoring all their
        (query, chunk) pairs in a single `predict` call, then splitting the
        scores back per list, takes one pass over the model instead of one per
        venue. Each list is still truncated to top-N and sorted on its own, and its
        Result objects are rescored in place, as in `rerank`.
        """
        candidates = [group[: self.cfg.top_n] for group in groups]
        pairs = [(query, r.text) for group in candidates for r in group]
        if not pairs:
            return [[] for _ in groups]
        scores = self.model.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False)

        out: list[list[Result]] = []
        offset = 0
        for group in candidates:
            for result, score in zip(group, scores[offset : offset + len(group)], strict=True):
                result.score = float(score)
            offset += len(group)
            group.sort(key=lambda r: (-r.score, r.chunk_id))
            out.append(group[:k])
        return out
