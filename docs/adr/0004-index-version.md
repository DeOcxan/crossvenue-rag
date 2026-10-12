# 0004. `index_version` and the gates on labels and results

- Date: 2026-09-24
- Status: Accepted

## Context

A gold label is a list of chunk ids. A chunk id means something only against the corpus, chunker and schema that produced it, and a vector only against the model that encoded it. A label made against one index and scored against another does not fail: it points at different text, or at nothing, and the metric is wrong without looking wrong. Exact deduplication (ADR 0003) changed every chunk id at once, which made the risk concrete.

## Decision

- `index_version` is a 16-hex hash of the manifest (sha256 of `data/manifest.csv`), the chunking config, the embedding model (name, dimension, document prefix, normalization) and the chunk schema version.
- It is written to `data/index/index_meta.json`, onto every Postgres row, into every `results/<timestamp>/index.json` and `report.md`, onto every judgment in the annotation journal, and into every frozen gold dataset's `.meta.json`.
- `eval` and `label` refuse to run when the index on disk, Postgres and the config disagree, or when any judgment was made against a different version. The message names the fix: rebuild, carry ids across with `crossvenue id-map` and `crossvenue remap-ids` when only ids changed, or re-judge.
- `crossvenue remap-ids` rewrites only id fields and verifies byte for byte that every other field of every record is unchanged.
- `crossvenue index` is the only command that issues DDL; `crossvenue check` opens a read-only connection.

## Consequences

- Any change to the manifest, chunking, embedding model or chunk schema forces `make ingest index`, and labels made before it are refused until migrated or re-judged.
- Results, labels and indexes are comparable only within one `index_version`; every report states it.
- The index every published result uses is named in [design](../explanation/design.md#index_version).
- The version hashes the recorded inputs and configuration of the corpus. It does not hash the code that turns them into chunks, or the model revision. A change to normalization or chunking code that the config does not express changes chunk text and ids under the same index_version. The embedding and reranker models are pinned to Hugging Face commits (`embedding.revision`, `reranker.revision`) and recorded in `index_meta.json`, outside the hashed components, because adding them to the hash would move the published version.
- Two checks cover that gap for this corpus version: `ingest` refuses a raw file whose sha256 differs from the manifest, and `eval` refuses a gold id that the chunks do not contain. A model revision change is caught by neither.
- The next corpus version also hashes the model revisions and a normalizer and chunker version constant that any change to that code increments. A test fails when the chunks of a fixture corpus change and the constant does not.
