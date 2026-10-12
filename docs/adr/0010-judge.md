# 0010. Two judge candidates, chosen by agreement with a human

- Date: 2026-09-25
- Status: Accepted

## Context

Answers are graded per claim (is it stated by its cited chunk, is it attributed to the right venue) and as a whole (coverage of the gold facts, expected behavior). 114 questions across four configurations make 456 answers; grading them by hand is out of reach for one annotator (ADR 0006). An LLM judge can grade them, but its grades are only as good as its agreement with a human. A judge from the same family as a generator may favor that generator (Rahmani et al. 2025, [arXiv:2506.10301](https://arxiv.org/abs/2506.10301); full citation in the [methodology references](../../evals/methodology.md#references)).

Which judge agrees best with a human is an empirical question, and a stronger model costs more: `anthropic/claude-opus-5.5` is $4 / $20 per million input / output tokens on OpenRouter, `anthropic/claude-sonnet-5` $2 / $10 (2026-09-25).

In the trial run, judge prompt v1 wrote verdicts before their rationale and once contradicted it, and once returned coverage for a venue with no gold chunk. Prompt v2 asks for the rationale first, names the venues with gold chunks, and separates behavior from coverage; a judgment whose coverage venues differ from the gold venues is rejected.

## Decision

- Two candidates, `anthropic/claude-sonnet-5` and `anthropic/claude-opus-5.5`, both through OpenRouter, pinned to Anthropic as provider with fallback disabled. No generator comes from Anthropic, so neither judge grades a model of its own family.
- Both get the same prompt (`prompts/judge.md` v2, its sha256 in every judgment), the same JSON schema, effort medium and at most 8,000 output tokens. No temperature is sent.
- Calibration: 40 answers of the full generation run, stratified by configuration and category and fixed in advance, are graded by a human with `crossvenue rate`, which never shows a judge's grade. The five trial questions are excluded, since the judge prompt was revised on them. Each candidate grades the same 40 answers (`crossvenue judge-sample`) and once more for self-consistency; `crossvenue judge-agreement` reports agreement per dimension.
- The judge used on the full matrix is the candidate that meets the thresholds of [methodology §9](../../evals/methodology.md#9-answer-generation-and-judging-pre-registered) on more dimensions. On a tie, the higher agreement on the primary comparison's dimensions (venue_correct and coverage) decides, then the cheaper one. If neither meets the threshold on a dimension, that dimension enters no conclusion.
- The candidates' agreement on the calibration sample, the chosen judge and the dimensions that enter no conclusion are in [findings](../results/findings.md#the-judge-meets-three-of-four-thresholds-on-the-point-estimate-and-one-on-the-interval).

## Consequences

- The judge's grades are trusted as far as their agreement with one human, measured on 40 answers, reaches.
- Calibrating two candidates costs two gradings of 40 answers plus two repeats, before the full matrix is judged once.
- A judge's output varies between calls at the default temperature; the self-consistency run measures how much.
