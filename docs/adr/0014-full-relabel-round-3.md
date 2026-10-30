# 0014. Full relabel as annotation round 3

- Date: 2026-10-30
- Status: Accepted

## Context

Deviation D25 leaves part of the set with labels made under the round-1 criterion until a full relabel, so one gold version mixes criteria. [Methodology §5](../../evals/methodology.md#5-annotator-agreement-with-one-annotator) names re-labeling against the refined criterion as the next step.

## Decision

- Registered in [methodology §5b](../../evals/methodology.md#5b-annotation-round-3-pre-registered) on 2026-10-30, before the first judgment of the round, following [ADR 0011](0011-preregistration.md).
- Every question is judged once more on a blind screen, on the candidates the earlier rounds recorded, with nothing retrieved again.
- `v0.3-gold` takes its labels from round 3 alone, so every label follows one criterion. Earlier entries stay in the journal as history.
- There is no round 4. The κ of round 2 stays the only test-retest figure and belongs to the round-1 criterion.

## Consequences

- Reliability under the amended criterion is not measured. Agreement of round 3 with round 1 is descriptive: counts and raw agreement, with no κ and no threshold.
- Registered results stay on `v0.2.1-gold`, and results on `v0.3-gold` are reported beside them.
- The judge reads gold, so existing answers are judged again on `v0.3-gold`, and the judge calibration is not repeated.
- A departure from §5b during the round is a deviation in [methodology, Deviations from protocol](../../evals/methodology.md#deviations-from-protocol).
