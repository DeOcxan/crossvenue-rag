# 0005. The labeling pool: methods, depth 20, 25 shown, augmentation

- Date: 2026-09-24
- Status: Accepted

## Context

Gold labels cannot be made by reading 11,934 chunks per question. The Cranfield/TREC answer is pooling: judge the union of several systems' top results and treat everything outside as not relevant ([methodology §1](../../evals/methodology.md#1-test-collection-construction-and-pooling)). Labeling from one retriever makes that retriever unbeatable by construction. A pool no deeper than the evaluation depth judges nothing the metric does not already see. A method added after labeling is scored against judgments it never contributed to, so every relevant chunk only it finds counts as a miss.

`venue_split` (ADR 0007) was added after round-1 labeling, so part of its top 10 on comparison questions had never been judged; the measured share is deviation D4 in [methodology](../../evals/methodology.md#deviations-from-protocol).

## Decision

- The pool is the union of each `retrieval.methods.POOL_METHODS` method's top 20 (`eval.pool_depth_per_method`), fused with RRF, and the top 25 distinct candidates are shown (`eval.pool_display`). If the methods overlap so much that fewer than 25 remain, each method is asked for more, up to four times the depth. A config validator rejects a pool depth that does not exceed `max(k_values)`.
- `POOL_METHODS` are bm25, dense, hybrid, hybrid_rerank and venue_split. `venue_split_oracle` is never a pool method: it reads the question's labeled venues, and a pool built from the labels it is judged against is circular.
- A method added to evaluation after questions are labeled has its unjudged top 20 on those questions judged by augmentation before its recall is compared. A retrieval-only pool-construction run (`crossvenue pool-run`) lists the never-judged chunks (`crossvenue augment-candidates`). `crossvenue label --augment` shows only those chunks, records every one shown, appends the relevant ones to gold and removes nothing.
- Every report states each method's `unjudged@k` and nDCG′@k (condensed list, Sakai 2007, [doi:10.1145/1277741.1277756](https://doi.org/10.1145/1277741.1277756); full citation in the [methodology references](../../evals/methodology.md#references)).

## Consequences

- Round 1 of `v0.2.1-gold` was pooled from the four methods that existed then; the comparison questions were augmented with venue_split's and venue_split_oracle's never-judged top 20 (`results/20260925T003721Z_pool`). What the augmentation found and how much it moved recall is in [findings](../results/findings.md#adding-17-gold-chunks-changes-every-methods-comparison-recall10-by-0034-to-0092).
- Recall is recall within the pool, comparable only within one gold version. For a method whose top 10 is fully judged, the drop that augmentation causes is a lower bound on the pool's inflation.
- The display cap, not the depth, sets how much of each pool method's top 10 is judged, so part of it can stay unjudged; every run reports that share as `unjudged@k` ([findings](../results/findings.md#adding-17-gold-chunks-changes-every-methods-comparison-recall10-by-0034-to-0092)).
- Augmentation covered comparison questions only, so recall on other categories carries more unjudged chunks for the augmented methods ([findings](../results/findings.md#three-limits-on-the-retrieval-results)).
