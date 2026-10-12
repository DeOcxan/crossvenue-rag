# 0006. Human labeling by one annotator, reliability by test-retest κ

- Date: 2026-09-24
- Status: Accepted

## Context

The gold set is the reference against which retrieval methods are ranked and against which an LLM answer judge will later be validated. Four ways to produce it were available: an LLM labeling pooled candidates, crowd workers, several expert annotators, or one expert annotator. Only one person with the domain knowledge to judge exchange API documentation is available.

## Decision

- One human annotator judges every pooled candidate against a written criterion (`evals/methodology.md` §5a): a chunk is relevant if a correct answer cannot be assembled without it.
- Reliability is intra-rater: a 25-question subsample, stratified by category and fixed in advance (`evals/selection/round2_sample.txt`), is re-judged blind at least 14 days later (round 2). Cohen's κ is computed over candidate decisions judged in both rounds, with raw agreement alongside. Target κ ≥ 0.61.
- Below 0.61, κ is reported as a limitation of this version of the set, with the categories whose decisions disagree most named. The criterion in `evals/methodology.md` §5a is refined where those disagreements show it ambiguous. Re-labeling the set against the refined criterion is recorded as the next step, outside this version. Re-labeling 114 questions (2,850 decisions) does not fit the project's schedule.
- LLM labeling is not used. The questions are drafted by an LLM and the answer judge is one. Gold labels produced by a model of the same kind would measure agreement between models rather than correctness, and the synthetic-evaluation literature reports exactly this bias toward the generating model (van Elburg et al. 2025, [arXiv:2508.11758](https://arxiv.org/abs/2508.11758); Rahmani et al. 2025, [arXiv:2506.10301](https://arxiv.org/abs/2506.10301); full citations in the [methodology references](../../evals/methodology.md#references)). The human gold set is what makes a later LLM judge checkable.

## Consequences

- κ is test-retest reliability of one person's criterion, not inter-rater agreement, and is reported as such.
- Labeling is the throughput bottleneck: one person makes every decision, 25 candidates per question. How fast round 1 went is in the [datasheet](../../evals/datasheet.md#labeling), beside the agreement round 2 measured ([agreement](../../evals/agreement.md)).
- Corrections after round 1 (`--review`, `--augment`) are recorded as corrections within round 1, with the previous set kept, never as a round 2.
