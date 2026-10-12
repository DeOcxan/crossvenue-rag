"""Shared fixtures.

Two rules hold across the suite:

- no model weights are downloaded: the embedder and reranker are stubbed and
  token counting is a word count;
- no database is required: dense retrieval is exercised through a fake store.

Anything that needs real weights or a real Postgres is not a unit test.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.config_loading import load_config
from crossvenue_rag.corpus.ingest import ingest
from crossvenue_rag.corpus.manifest import ManifestRow, write_manifest
from crossvenue_rag.evaluation.power import PowerStudy
from crossvenue_rag.generation.calibration_file import Calibration, DimensionStatistics
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.models import (
    Chunk,
    ChunkSource,
    Document,
    Result,
    Timings,
    VenueType,
    make_chunk_id,
)
from crossvenue_rag.retrieval.methods import METHODS, ORACLE_METHODS
from crossvenue_rag.retrieval.retriever import Retriever
from crossvenue_rag.run_files import Registry, RegistryEntry, file_sha256, registry_json


def default_config(root: Path) -> Config:
    """configs/default.yaml with every data path redirected under `root`."""
    config = load_config(REPO / "configs/default.yaml")
    config.paths.raw = root / "raw"
    config.paths.processed = root / "processed"
    config.paths.index = root / "index"
    config.paths.results = root / "results"
    config.paths.manifest = root / "manifest.csv"
    return config


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Default config with all paths redirected into a tmp directory."""
    return default_config(tmp_path)


@pytest.fixture
def doc() -> Document:
    return Document(
        venue="bybit",
        venue_type="cex",
        doc_path="v5/order/create.md",
        source_url="https://example.invalid/create.md",
        text=(FIXTURES / "sample_docusaurus.md").read_text(encoding="utf-8"),
    )


def make_chunk(chunk_id: str, venue: str, text: str, venue_type: VenueType = "cex") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        venue=venue,
        venue_type=venue_type,
        doc_path=f"{venue}/doc.md",
        heading_path=["API", "Rate Limits"],
        source_url=f"https://example.invalid/{venue}",
        text=text,
        token_count=len(text.split()),
        position=0,
    )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture(scope="module")
def bm25() -> BM25Index:
    return BM25Index.build(VENUE_CHUNKS)


def occurrence(
    venue: str, doc_path: str, text: str, position: int = 0, env: str | None = None
) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(venue, text),
        venue=venue,
        venue_type="cex",
        doc_path=doc_path,
        heading_path=["Limits"],
        source_url=f"https://example.invalid/{doc_path}",
        text=text,
        token_count=len(text.split()),
        position=position,
        sources=[
            ChunkSource(
                venue,
                doc_path,
                ["Limits"],
                f"https://example.invalid/{doc_path}",
                position,
                environment=env,
            )
        ],
        environment=env,
    )


def build_corpus(
    cfg: Config, extra: dict[str, tuple[str, VenueType, str]] | None = None
) -> list[Chunk]:
    rows: list[ManifestRow] = []
    for local_path, (venue, venue_type, text) in {**CORPUS, **(extra or {})}.items():
        path = cfg.paths.raw / local_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        rows.append(
            ManifestRow(
                venue=venue,
                venue_type=venue_type,
                source_url=f"https://example.invalid/{local_path}",
                local_path=local_path,
                fetched_at="2026-01-01T00:00:00Z",
                commit_sha="0" * 40,
                sha256=file_sha256(path),
            )
        )
    write_manifest(cfg.paths.manifest, rows)

    cfg.chunking.max_tokens = 80
    cfg.chunking.overlap_tokens = 10
    cfg.chunking.min_tokens = 10
    return ingest(cfg, count_tokens=count_words)


FIXTURES = Path(__file__).parent / "fixtures"


def count_words(text: str) -> int:
    """The token counter tests chunk with: whitespace words, so no tokenizer is loaded.

    Production never falls back to it: `tokens.token_counter_for` refuses rather than
    switch counters, since another counter chunks differently under the same `index_version`.
    """
    return len(text.split())


# Near-duplicate text on purpose: the same sentence, four venues.
VENUE_CHUNKS = [
    make_chunk("c-binance", "binance", "Rate limits are expressed as request weight per minute."),
    make_chunk("c-bybit", "bybit", "Rate limits are expressed as requests per second per UID."),
    make_chunk(
        "c-uniswap",
        "uniswap",
        "Uniswap v3 supports fee tiers of 0.01%, 0.05%, 0.30% and 1.00%.",
        venue_type="dex",
    ),
    make_chunk(
        "c-pancakeswap",
        "pancakeswap",
        "PancakeSwap v3 supports fee tiers of 0.01%, 0.05%, 0.25% and 1.00%.",
        venue_type="dex",
    ),
]


# Four pages that say nearly the same thing in nearly the same words: the
# corpus in miniature.
CORPUS: dict[str, tuple[str, VenueType, str]] = {
    "binance/rest-api.md": (
        "binance",
        "cex",
        """---
title: Rate Limits
---

# Rate Limits

Every request has a weight. The weight is counted against a per-minute budget of
1200 and reported in the `X-MBX-USED-WEIGHT-1M` response header.

## Order Rate Limits

Orders are limited to 50 per ten seconds and 160000 per day.

```http
GET /api/v3/exchangeInfo
```
""",
    ),
    "bybit/rate-limit.md": (
        "bybit",
        "cex",
        """---
title: Rate Limit
---

# Rate Limit

:::info
Rate limits are applied per UID, per endpoint, per second.
:::

The default for order creation is 10 requests per second. The response header
`X-Bapi-Limit-Status` reports how much of the window is left.
""",
    ),
    "uniswap/fees.mdx": (
        "uniswap",
        "dex",
        """import { FeeTable } from '../components/FeeTable'

# Fee Tiers

Uniswap v3 pools can be created with one of four fee tiers: 0.01%, 0.05%,
0.30% and 1.00%. The fee is set at pool creation and cannot be changed.
""",
    ),
    "pancakeswap/fees.md": (
        "pancakeswap",
        "dex",
        """# Fee Tiers

{% hint style="info" %}
PancakeSwap v3 is a fork of Uniswap v3.
{% endhint %}

PancakeSwap v3 pools can be created with one of four fee tiers: 0.01%, 0.05%,
0.25% and 1.00%. The fee is set at pool creation and cannot be changed.
""",
    ),
}


# The libraries docs/code-style.md rule 30 lets a module defer, by import name
# (scikit-learn imports as sklearn): commands that do not need them must not load them.
HEAVY_MODULES = (
    "psycopg", "pgvector", "bm25s", "Stemmer", "torch", "sentence_transformers",
    "transformers", "scipy", "sklearn",
)  # fmt: skip


# Oracle methods need a question's labeled venues; ad-hoc queries have none.
AD_HOC_METHODS = [m for m in METHODS if m not in ORACLE_METHODS]


class FakeEmbedder:
    """Deterministic hashed bag-of-words encoder.

    Real enough for the pipeline to run end to end (similar texts get similar
    vectors), and it needs no network, no GPU and no 400 MB download.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in text.lower().split():
            vector[hash(word) % self.dimension] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_documents(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def model_fingerprint(self) -> str:
        return "fake@dim64"


class FakeReranker:
    """Scores by word overlap, so 'reranking changed the order' is testable."""

    def rerank(self, query: str, results: list, k: int) -> list:
        query_words = set(query.lower().split())
        for result in results:
            overlap = len(query_words & set(result.text.lower().split()))
            result.score = float(overlap)
        results.sort(key=lambda r: (-r.score, r.chunk_id))
        return results[:k]

    def rerank_many(self, query: str, groups: list, k: int) -> list:
        return [self.rerank(query, group, k) for group in groups]


class InMemoryRetriever(Retriever):
    """Dense search against a numpy matrix instead of pgvector.

    Overriding the one method that needs a database keeps the production code
    free of a test-only seam, and every other stage (fusion, reranking,
    filtering, timing) runs exactly as it does in production.
    """

    def __init__(self, cfg: Config, chunks: list[Chunk], bm25: BM25Index) -> None:
        # Retriever takes the concrete Reranker; the fake scores without weights.
        super().__init__(cfg, chunks=chunks, bm25=bm25, reranker=FakeReranker())  # type: ignore[arg-type]
        self._fake = FakeEmbedder()
        self._matrix = self._fake.encode_documents([c.text for c in chunks])

    def _dense_search(
        self,
        query: str,
        k: int,
        venues: list[str] | None,
        timings: Timings,
        allowed_ids: set[str] | None = None,
    ) -> list[Result]:
        start = time.perf_counter()
        vector = self._fake.encode_query(query)
        timings.encode_ms += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        similarities = self._matrix @ vector
        order = np.argsort(-similarities)
        results: list[Result] = []
        for index in order:
            chunk = self.chunks[int(index)]
            if venues and chunk.venue not in venues:
                continue
            if allowed_ids is not None and chunk.chunk_id not in allowed_ids:
                continue
            results.append(Result.from_chunk(chunk, float(similarities[index])))
            if len(results) == k:
                break
        timings.search_ms += (time.perf_counter() - start) * 1000
        return results


# Output defects docs/code-style.md rule 25 forbids in text a person reads.
# Code spans are removed first: an identifier or quoted JSON keeps its own form.
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_OUTPUT_DEFECTS = {
    "ASCII hyphen as a minus sign": re.compile(r"(?<![\w.\-/:])-\d"),
    '"(s)" plural': re.compile(r"\w\(s\)"),
    '"--" as a dash': re.compile(r" -- "),
    # A list of three or more numbers (two are an interval, rule 10). A list that opens
    # with a quoted string. A dict or set that opens with a quoted string or a number key.
    "list or dict repr": re.compile(
        r"\[(\s*[+\-]?\d[\d.]*\s*,){2,}"
        r"|\[\s*(['\"])[^'\"\n]*\2\s*[,\]]"
        r"|\{\s*['\"]"
        r"|\{\s*[+\-]?\d[\d.]*\s*:"
    ),
    "four or more decimals": re.compile(r"\d\.\d{4,}"),
    # `behaviour_correct` is a schema field (rule 26), so an underscore ends the stem.
    "UK spelling": re.compile(
        r"(?<![\w])(labell\w*|judgement\w*|randomis\w*|favour\w*|behaviours?)(?![\w])",
        re.IGNORECASE,
    ),
    "time word": re.compile(r"\b(yet|now)\b", re.IGNORECASE),
}


def output_defects(text: str) -> list[str]:
    """The kinds of output defect found in `text`, with the first offending line of each."""
    found: list[str] = []
    for line in text.splitlines():
        plain = _CODE_SPAN.sub("", line)
        for kind, pattern in _OUTPUT_DEFECTS.items():
            if pattern.search(plain) and not any(f.startswith(kind) for f in found):
                found.append(f"{kind}: {line.strip()[:120]}")
    return found


def power_study_of(**fields: Any) -> PowerStudy:
    """A power.json study: `fields` over a small two-comparison study with no points."""
    base: dict[str, Any] = {
        "run": "results/run", "metric": "recall@10", "category": "comparison", "questions": 50,
        "family": [["a", "b"], ["a", "c"]], "fractional_values": 0, "values": 150,
        "discordant_pairs": 40, "question_pairs": 150, "psi": 0.25, "mean_magnitude": 0.5,
        "ceiling": 0.125, "alpha": 0.05, "studies": 10, "permutations": 10, "seed": 1,
        "points": [], "delta_80_holm_first_step": None, "delta_80_unadjusted": None,
    }  # fmt: skip
    return PowerStudy.model_validate({**base, **fields})


def dimension_of(**fields: Any) -> DimensionStatistics:
    """One dimension of a calibration comparison: `fields` over no answers and no statistics."""
    return DimensionStatistics.model_validate({"n": 0, "answers": 0, **fields})


def calibration_of(**fields: Any) -> Calibration:
    """A calibration.json: `fields` over an empty one.

    A candidate or a dimension may be given as a partial dict; the rest is filled in.
    """
    data: dict[str, Any] = {
        "run": "results/run", "ratings": "ratings.jsonl", "raters": ["tester"],
        "answers_rated": 0, "resamples": 10, "seed": 1, "thresholds": {},
        "excluded_from_behaviour_correct": [], "statistic": {}, "comparisons": {},
        "candidates": {}, "selected": "", "reason": "",
    }  # fmt: skip
    data.update(fields)
    data["candidates"] = {
        name: {"passes": [], "fails": [], "cost_per_answer_usd": 0.0, **candidate}
        for name, candidate in data["candidates"].items()
    }
    data["comparisons"] = {
        name: {d: dimension_of(**row).model_dump() for d, row in dims.items()}
        for name, dims in data["comparisons"].items()
    }
    return Calibration.model_validate(data)


def registry_json_of(runs: dict[str, dict[str, Any]]) -> str:
    """results/runs.json listing `runs`, each entry's fields over an unversioned run's."""
    empty: dict[str, Any] = {
        "index_version": None,
        "index_version_from": None,
        "notes": [],
        "files": {},
    }
    entries = {name: RegistryEntry(**{**empty, **entry}) for name, entry in runs.items()}
    return registry_json(Registry(runs=entries))


REPO = Path(__file__).resolve().parent.parent
# A test that checks a document against the local corpus and index (`make fetch ingest
# index`), which are not committed, so it is skipped where they are absent (CI).
needs_local_corpus = pytest.mark.skipif(
    not all(
        (REPO / path).exists()
        for path in ("data/processed/chunks.jsonl", "data/index/index_meta.json")
    ),
    reason="needs the local corpus and index",
)


def committed_with(filename: str) -> list[Path]:
    """The committed runs (`results/runs.json`) whose directory holds `filename`."""
    from crossvenue_rag.reporting.runs import committed_runs

    runs = [REPO / "results" / name for name in committed_runs(REPO)]
    return [run for run in runs if (run / filename).exists()]


def committed_of_kind(filename: str, *suffixes: str) -> list[Path]:
    """`committed_with(filename)` of the runs whose name ends in one of `suffixes`;
    with none, the retrieval runs, whose names have no suffix."""
    runs = committed_with(filename)
    if not suffixes:
        return [run for run in runs if "_" not in run.name]
    return [run for run in runs if run.name.endswith(suffixes)]


def assert_round_trip(run: Path, filename: str, table: Any, tmp_path: Path) -> None:
    """`table`, read from `run/filename`, writes back byte for byte."""
    from crossvenue_rag.run_files import write_table

    write_table(tmp_path / filename, table)
    assert (tmp_path / filename).read_bytes() == (run / filename).read_bytes()


def write_journal(questions_path: str | Path, data: dict[str, Any]) -> None:
    """Saves a journal written as plain JSON beside `questions_path`, through its model."""
    from crossvenue_rag.dataset.journal import Journal, save_rounds

    save_rounds(questions_path, Journal.from_json(data))


def journal_dict(questions_path: str | Path) -> dict[str, Any]:
    """The journal beside `questions_path` as plain JSON, as its file holds it."""
    from crossvenue_rag.dataset.journal import load_rounds

    data: dict[str, Any] = load_rounds(questions_path).model_dump()
    return data


def augmentation(shown: list[str], relevant: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """An augmentation of a journal's round 1 as plain JSON: `shown` judged, `relevant` marked."""
    return {
        "at": "2026-09-25T00:00:00Z",
        "annotator": "me",
        "index_version": "v",
        "source": "candidates.jsonl",
        "shown": list(shown),
        "relevant": list(relevant),
        "coverage_changes": [],
    }


def correction(previous_relevant: list[str], relevant: list[str] | None = None) -> dict[str, Any]:
    """A review correction of a journal's round 1 as plain JSON."""
    return {
        "at": "2026-09-24T23:00:00Z",
        "annotator": "me",
        "index_version": "v",
        "previous_relevant": list(previous_relevant),
        "previous_venue_coverage": {},
        "relevant": list(previous_relevant if relevant is None else relevant),
        "venue_coverage": {},
    }


def adjudication(decisions: dict[str, bool]) -> dict[str, Any]:
    """An adjudication entry of a journal as plain JSON: chunk id to its decided verdict."""
    return {
        "annotator": "me",
        "criterion": "2026-10-10",
        "index_version": "v",
        "decisions": {
            chunk: {"relevant": verdict, "annotated_at": "2026-10-10T20:00:00Z"}
            for chunk, verdict in decisions.items()
        },
        "annotated_at": "2026-10-10T20:00:00Z",
    }


def relabel(pool: list[str], relevant: list[str]) -> dict[str, Any]:
    """A relabel entry of a journal as plain JSON."""
    return {
        "annotator": "me",
        "annotated_at": "2026-10-10T20:00:00Z",
        "criterion": "2026-10-10",
        "index_version": "v",
        "pool": list(pool),
        "relevant": list(relevant),
    }
