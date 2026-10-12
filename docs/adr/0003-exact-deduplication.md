# 0003. Exact deduplication with provenance, never across venues

- Date: 2026-09-24
- Status: Accepted

## Context

Of the 12,783 chunks the corpus produced, 1,385 (10.8%) belonged to 536 groups of byte-identical text; 433 of the groups were Binance mainnet pages against their `testnet/` copies, and no group spanned two venues.

The copies were first kept in the index, and scoring expanded each gold set with every byte-identical twin (`eval.score_duplicates_as_equivalent`). That fixed the metric but not retrieval. Measured over the 114 selected questions and each method's top 10 (`results/20260924T200525Z`), a copy of a text already ranked above it held a mean of 0.447–0.579 slots per question in each method's top 10. The means were bm25 0.518, dense 0.447, hybrid 0.526 and hybrid_rerank 0.579. This happened on 29–35 questions per method, with up to 7 slots on one question. The labeling pool had the same problem: `cv-0001 "whats the largest recvWindow binance will"` was judged on 25 candidates holding 23 distinct texts.

The evaluation methodology had treated the duplication as part of the phenomenon under test. Measured, the mainnet/testnet confusion is in near-identical text, which differs; identical text has nothing in it that tells the environments apart.

## Decision

- A chunk id is `sha256(venue + "\x1f" + text)[:16]`, a content hash. Chunks with the same id are stored once.
- Every occurrence is kept in `Chunk.sources` (venue, doc path, heading path, URL, position, structural metadata).
- The citation fields describe one primary occurrence: non-testnet before testnet, then doc path lexicographically, then position.
- Identical text on two venues is never merged: the venue is in the hash.
- Near-duplicates are never merged: only exact equality of the chunk text.
- `eval.score_duplicates_as_equivalent` is removed.
- Pointers into the old ids are carried across by a verified id map (`evals/migrations/2026-09-24-exact-dedup.idmap.json`, all 12,783 old ids mapped); `v0.1.5-candidates` is the migrated candidate pool.

## Consequences

- The index holds 11,934 chunks (−849) and 2,484,877 tokens (−152,233).
- The lexicographic tie-break is blind to meaning: the Bybit IP-limit paragraph shared by `docs/p2p/rate-limit.mdx` and `docs/v5/rate-limit/rate-limit.mdx` cites the P2P page, although `cv-0004 "bybit sends 403 access too frequent,"` asks about the trading API. Retrieval and scoring work on chunk ids and are unaffected; answer generation cites the occurrence that matches the question.
- `cv-0091 "on binance spot testnet how long"` and `cv-0128 "is the ip ban on binance"` were written for a trap made of two identical chunks; their notes describe the merged chunk from `v0.1.6-candidates` on.
- Measurements over the corpus change with it: every corpus measurement in findings, such as PancakeSwap's best-match cosine to Uniswap, is taken on the deduplicated corpus ([findings](../results/findings.md#pancakeswaps-median-best-match-cosine-to-uniswap-is-0143)).
