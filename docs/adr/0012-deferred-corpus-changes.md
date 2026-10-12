# 0012. Corpus corrections deferred to the next corpus version

- Date: 2026-09-25
- Status: Accepted

## Context

A review of how the corpus is built found five defects. Each fix changes `data/processed/chunks.jsonl`, and so chunk ids, the index_version or both (ADR 0004). Every published number, gold label and saved run refers to index_version `2825a7160c4d09bf`; a fix now would invalidate them, and the gold set would need `remap-ids` and partial re-judging.

## Decision

The five corrections wait for the next corpus version and are recorded as limitations in the [corpus datasheet](../reference/corpus.md#composition). Counts are chunks in the current index.

| Defect | Fix | What changes |
|---|---|---|
| The breadcrumb is written only into chunks that do not start with a heading: 4,522 of the 8,820 heading-led chunks miss a parent heading | write the full breadcrumb into every chunk | text and id of 4,522 chunks; token counts, so some chunks split |
| fnmatch's `*` crosses `/` in the Binance include globs | glob with path-aware matching and name the testnet reference pages explicitly | removes `testnet/CHANGELOG.md` (121 chunks), `demo-mode/` (13) and the terms pages (3); gold labels that point into them, such as `cv-0148 "do i still need a listenKey"`, are re-judged |
| GitBook's "Querying This Documentation" section survives normalization | strip it in the GitBook normalizer | removes 270 PancakeSwap chunks (61,955 tokens) |
| Normalization collapses blank-line runs inside 19 fenced code blocks | apply the collapse to prose segments only | text and id of the chunks holding those blocks |
| The path mapping drops PancakeSwap's first URL segment | drop a slug only where `sources.yaml` names one | every PancakeSwap `local_path`; `doc_path` and metadata of the 206 chunks from 31 `archive/` pages |

## Consequences

- The published results describe the corpus with these defects; the datasheet states each one with its size.
- The next corpus version applies all five together with the index_version change listed in the [Consequences of ADR 0004](0004-index-version.md#consequences), so the labels migrate once.
