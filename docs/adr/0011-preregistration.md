# 0011. Answer comparisons registered before the run

- Date: 2026-09-25
- Status: Accepted

## Context

The answer matrix yields 6 pairs of the 4 configurations × 4 metrics × 2 scopes, 48 paired comparisons. Reading them after the run and reporting whichever came out significant would find something by chance: at α = 0.05, about two of 48 null comparisons. The retrieval results faced the same problem and resolved it with one declared primary comparison and Holm-adjusted secondary ones.

## Decision

- Registered in [methodology §9](../../evals/methodology.md#9-answer-generation-and-judging-pre-registered) on 2026-09-25, before any answer was generated for evaluation.
- Primary: gemini@venue_split − gemini@hybrid_rerank on venue_attribution_error and on coverage, over the 50 comparison questions; the two form one family, Holm-adjusted. The generator is fixed, so the comparison isolates retrieval.
- Secondary: every other pair, metric and scope, Holm-adjusted within each metric and scope. Reported, not used to change the primary conclusion.
- `generation.primary_pair`, `primary_metrics` and `primary_scope` in the config hold the same declaration; `answer-eval` marks those rows `primary` in `paired.csv`.
- A judge dimension that fails its calibration threshold makes the metric built on it uninterpretable; thresholds are not changed after ratings are seen.

## Consequences

- A result outside the primary comparison is secondary, however striking.
- A change to the primary comparison after the run is a deviation, recorded in [methodology, Deviations from protocol](../../evals/methodology.md#deviations-from-protocol).
- The measured comparisons are in [findings](../results/findings.md#under-prompt-v1-splitting-retrieval-by-venue-raises-answer-coverage).
