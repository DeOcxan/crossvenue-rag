"""What callers need of a retriever, one protocol per contract.

`retriever.Retriever` satisfies all of them; tests pass fakes that satisfy only
the one a function takes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from ..config import Config
from ..config_pipeline import RetrievalMethod
from ..models import Chunk, Result, Timings
from .filters import Filters

# How venue_split calls the cross-encoder: once per venue, or once for all venues. The
# bench (`evaluation.bench.MODES`) times and reports them in this order.
RerankCalls = Literal["per_venue", "one_call"]


class Ranks(Protocol):
    """A ranked list per method: what answering, pooling and estimating read."""

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        *,
        k: int,
        venues_hint: Sequence[str] | None = None,
    ) -> list[Result]:
        """The top `k` of `method` for `query`; the oracle reads `venues_hint`."""
        ...


class TimedRanks(Protocol):
    """A ranked list per method, under filters, with its step timings: what evaluation reads."""

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        *,
        k: int,
        filters: Filters | None,
        timings: Timings | None,
        venues_hint: Sequence[str] | None = None,
    ) -> list[Result]:
        """The top `k` of `method` for `query`, restricted by `filters`, timed into `timings`."""
        ...


class WarmTimedRanks(TimedRanks, Protocol):
    """Timed ranks that load their models first: what `retriever.compare_methods` reads."""

    def warmup(self, *, load_reranker: bool) -> None:
        """Loads the models, the reranker only when `load_reranker`, before anything is timed."""
        ...


class ChunkLookup(Protocol):
    """Chunks by id: what augmentation and the judge's chunk list read."""

    @property
    def by_id(self) -> Mapping[str, Chunk]:
        """Every chunk of the index by its id."""
        ...


class DetectsVenues(Protocol):
    """Names the venues a text mentions."""

    def detect(self, text: str) -> list[str]:
        """The venues `text` names, in the order it names them."""
        ...


class DetectingRanks(Ranks, Protocol):
    """Ranks, and names the venues a question mentions: what an ad-hoc answer reads."""

    @property
    def venue_detector(self) -> DetectsVenues:
        """The detector venue_split uses."""
        ...


class AskRetriever(DetectingRanks, Protocol):
    """Ranks and detects venues on a current index: what `crossvenue ask` opens."""

    def require_current_index(self) -> str:
        """The `index_version` both indexes carry.

        Raises UserError unless the two indexes agree with each other and with the config.
        """
        ...


class TimesVenueSplit(Protocol):
    """Names the venues a question mentions and times venue_split: what bench-rerank reads."""

    @property
    def venue_detector(self) -> DetectsVenues:
        """The detector venue_split uses."""
        ...

    def venue_split_timed(
        self,
        query: str,
        rerank_calls: RerankCalls,
        timings: Timings,
        pairs_per_call: list[int],
    ) -> list[Result]:
        """venue_split over the venues detected in `query`, reranked as `rerank_calls` says."""
        ...


class BenchRetriever(TimesVenueSplit, Protocol):
    """Times venue_split on a current index, models loaded first: what `bench-rerank` opens."""

    def require_current_index(self) -> str:
        """The `index_version` both indexes carry.

        Raises UserError unless the two indexes agree with each other and with the config.
        """
        ...

    def warmup(self, *, load_reranker: bool = True) -> None:
        """Loads the models, the reranker only when `load_reranker`, before anything is timed."""
        ...


class LabelRetriever(ChunkLookup, Ranks, Protocol):
    """What labeling reads: ranks, chunks by id, and the retriever's config."""

    @property
    def cfg(self) -> Config:
        """The config the retriever was built from."""
        ...
