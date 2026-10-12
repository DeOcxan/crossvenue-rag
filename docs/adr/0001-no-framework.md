# 0001. Plain Python pipeline, no RAG framework

- Date: 2026-09-20
- Status: Accepted

## Context

The project's output is a claim about where retrieval breaks and why on near-identical documentation from different venues. Checking such a claim requires every stage (normalization, chunking, fusion, reranking, scoring) to be individually observable and individually measurable.

RAG frameworks (LangChain, LlamaIndex) package exactly those stages behind abstractions: the chunker behind a `TextSplitter`, fusion behind an `EnsembleRetriever`, evaluation behind callbacks. The parts under investigation would be the parts hardest to see.

## Decision

The pipeline is plain Python over a small set of single-purpose libraries: `bm25s` for the sparse index, `sentence-transformers` for the bi-encoder and cross-encoder, `psycopg` + `pgvector` for dense search, `pydantic` for configuration and schemas, `typer` and `rich` for the CLI. Chunking, fusion, filtering, metrics, the evaluation harness and the labeling tool are written in the repository.

## Consequences

- Every number in `docs/results/findings.md` traces to a specific function, and every stage has its own tests.
- The repository carries code a framework would have supplied; its size is in [design](../explanation/design.md#no-framework).
- Integrations a framework offers for free (other vector stores, other model providers, agents) do not exist and are added only when a measurement needs them.
