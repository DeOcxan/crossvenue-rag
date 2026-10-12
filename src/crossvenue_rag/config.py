"""The whole pipeline configuration: its sections, the checks across them and its defaults.

`config_loading` reads the YAML file into it. A saved config holds no secret:
`Config.to_yaml` leaves `store.dsn` out, a DSN with a password comes from
`CROSSVENUE_PG_DSN`, and the OpenRouter key is read by `generation.llm`.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import model_validator

from . import formatting
from .config_eval import EvalConfig
from .config_generation import GenerationConfig
from .config_pipeline import (
    FUSED_METHODS,
    RERANKED_METHODS,
    ChunkingConfig,
    EmbeddingConfig,
    FetchConfig,
    PathsConfig,
    RerankerConfig,
    RetrievalConfig,
    RetrievalMethod,
    StoreConfig,
)
from .config_section import Section


class Config(Section):
    """The whole pipeline configuration, read from the YAML file and `--set` only.

    `CROSSVENUE_PG_DSN` in the environment overrides `store.dsn`
    (`config_loading.load_config`); no other environment variable is read.
    """

    paths: PathsConfig = PathsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    store: StoreConfig = StoreConfig()
    fetch: FetchConfig = FetchConfig()
    eval: EvalConfig = EvalConfig()
    generation: GenerationConfig = GenerationConfig()

    @model_validator(mode="after")
    def _rerank_depth(self) -> Config:
        # `eval` retrieves at max(k_values), `retrieve` at retrieval.k, the labeling pool
        # at eval.pool_depth_per_method and the generator at generation.top_k.
        deepest = max(
            self.retrieval.k,
            *self.eval.k_values,
            self.eval.pool_depth_per_method,
            self.generation.top_k,
        )
        if self.reranker.top_n < deepest:
            raise ValueError(
                f"reranker.top_n ({formatting.own_form(self.reranker.top_n)}) is below the "
                f"deepest list asked for ({formatting.own_form(deepest)}, from retrieval.k, "
                "eval.k_values, eval.pool_depth_per_method and generation.top_k); reranking "
                "would return fewer results than asked for"
            )
        # Each side of a hybrid retrieval gives at least as many candidates as a list holds.
        deepest_k = max(self.retrieval.k, *self.eval.k_values)
        for name in ("bm25_candidates", "dense_candidates"):
            candidates = getattr(self.retrieval, name)
            if candidates < deepest_k:
                raise ValueError(
                    f"retrieval.{name} ({formatting.own_form(candidates)}) is below the "
                    f"deepest k asked for ({formatting.own_form(deepest_k)}, from retrieval.k "
                    "and eval.k_values); a hybrid list would come back shorter"
                )
        return self

    def depth_bounds(self, method: RetrievalMethod) -> dict[str, int]:
        """The config keys that bound how many results `method` returns, with their values.

        A fused method returns at most as many results as each candidate side gives,
        the bound `_rerank_depth` applies to the configured depths, and a reranking
        method at most `reranker.top_n`. BM25 and dense alone have no bound.
        """
        sides = ("bm25_candidates", "dense_candidates") if method in FUSED_METHODS else ()
        bounds: dict[str, int] = {f"retrieval.{s}": getattr(self.retrieval, s) for s in sides}
        if method in RERANKED_METHODS:
            bounds["reranker.top_n"] = self.reranker.top_n
        return bounds

    def with_questions(self, questions: Path) -> Config:
        """A copy of this config that reads `questions`; this one is left as it is."""
        return self.model_copy(
            update={"eval": self.eval.model_copy(update={"questions": questions})}
        )

    def to_yaml(self) -> str:
        """The resolved config as YAML, so a results directory records what ran.

        Every key is written except `store.dsn`, which can hold a password.
        """
        text: str = yaml.safe_dump(
            self.model_dump(mode="json"), sort_keys=False, default_flow_style=False
        )
        return text


# The config schema's defaults (`Config()`), read without configs/default.yaml. The reports, the run
# registry and the commands that read the repository's saved files take its default paths
# from it, the corpus artifacts, `paths.results` and `eval.questions` among them.
SCHEMA_DEFAULTS = Config()
