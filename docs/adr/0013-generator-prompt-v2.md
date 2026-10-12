# 0013. A second generator prompt, analyzed as exploratory

- Date: 2026-09-25
- Status: Accepted

## Context

Rule 5 of generator prompt v1 allows a full answer or a refusal and nothing between: "If the chunks do not contain the answer, refuse". Under v1, Gemini with venue_split refused 25 of the 50 comparison questions, and in 24 of them the context held chunks of every named venue ([findings](../results/findings.md#under-prompt-v1-geminivenue_split-refuses-25-of-the-50-comparison-questions)). Under the rule, a comparison with one side missing falls under not containing the answer, and "give no claims" leaves no claim for the side that is there. The primary comparison had been registered for v1 before any answer was generated ([ADR 0011](0011-preregistration.md)).

## Decision

- Prompt v2 (`prompts/generator.md`) allows a third outcome, a partial answer: claims for the venues the chunks cover and a `missing` entry for each venue they do not. It also lets an absence from a list the chunk enumerates be a claim. Prompt v1 stays as `prompts/generator.v1.md`.
- v2 was written after v1's answers were read, so its analysis is exploratory. Hypothesis H-v2 and a secondary venue_split − hybrid_rerank comparison under v2 are registered in [methodology §9a](../../evals/methodology.md#9a-generator-prompt-v2-exploratory) before any v2 answer was generated, and the change is deviation D10 ([deviations](../../evals/methodology.md#deviations-from-protocol)).
- The primary comparison stays on v1.

## Consequences

- No v2 result changes a conclusion of methodology §9; every v2 result is labeled exploratory.
- The judge was calibrated on v1 answers, which hold no partial answer and no absence claim, so the calibration does not cover the forms v2 adds.
- The v1-to-v2 comparisons are in [findings](../results/findings.md#prompt-v2-lowers-wrong-refusals-in-every-configuration-exploratory).
