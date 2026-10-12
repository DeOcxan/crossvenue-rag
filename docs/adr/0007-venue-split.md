# 0007. `venue_split` retrieval and dictionary venue detection

- Date: 2026-09-24
- Status: Accepted

## Context

On the first labeled set (`v0.2-gold`, `results/20260924T234623Z`), none of the four methods put a chunk from every named venue into the top 10 of more than 35 of the 50 comparison questions. venue_coverage@10 was 0.70 (35/50) for bm25 and hybrid_rerank, 0.60 (30/50) for hybrid and 0.52 (26/50) for dense. The annotator recorded 22 comparison venues as `missing_from_pool`. A single ranked list over all venues lets the venue with the more similar wording take every slot.

## Decision

- `venue_split` reads the venues a question names, runs hybrid_rerank once per venue under a venue filter, and interleaves the per-venue lists round-robin up to k, in the order the venues are named. With fewer than two venues named it is exactly hybrid_rerank.
- Venues are detected by a dictionary: regex aliases per venue in `data/sources.yaml`, matched case-insensitively as whole words (`uni v3`, `hl`, `pancake`). `bnb` is not an alias: "BNB Chain" occurs in 101 PancakeSwap chunks, 2 Uniswap chunks and no Binance chunk (chunks containing the term, case-insensitive, `crossvenue count-term "BNB Chain"`). The dictionary is not tuned to the evaluation set.
- `venue_split_oracle` does the same with the question's labeled venues in place of the detected ones, which a deployed system does not have. It bounds what better venue detection can add to venue coverage, and it is not an upper bound on recall, precision or nDCG. It is labeled "(oracle)" in every report and is never a pool method (ADR 0005).
- An LLM venue extractor is not used: the dictionary is deterministic, auditable as data and adds no model call. How close it comes to the oracle is in [findings](../results/findings.md#venue-detection-finds-every-named-venue-on-47-of-50-comparison-questions).

## Consequences

- Each detected venue's first chunk enters the top 10, so venue coverage is guaranteed whenever detection finds every named venue. The coverage result therefore measures the detector and the split, and says nothing about whether the covered chunks are relevant. Coverage and recall against hybrid_rerank are in [findings](../results/findings.md#splitting-retrieval-by-venue-raises-comparison-coverage-from-070-to-098).
- Each named venue adds a hybrid_rerank pass, so latency grows with the number of venues a question names. Scoring all venues' candidates in one cross-encoder call (`Reranker.rerank_many`) returns the same retrievals with one call instead of one per venue; the measured cost of both is in [findings](../results/findings.md#the-cross-encoder-takes-2505-of-hybrid_reranks-2531-ms-at-p50).
- A question whose venues the dictionary does not find is answered as plain hybrid_rerank. How often that happens, and how close detection comes to the oracle, is in [findings](../results/findings.md#venue-detection-finds-every-named-venue-on-47-of-50-comparison-questions).
