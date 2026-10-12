# Annotator agreement

Test-retest agreement of the one annotator on the gold set `v0.2.1-gold`, as [methodology §5](methodology.md#5-annotator-agreement-with-one-annotator) registers it. Round 1 is the labeling that produced `v0.2.1-gold`; round 2 re-judged 25 of its questions blind on 2026-10-10. Apart from the retrieval check, which names its runs, and the retest minimum and fallback of methodology §5, every number below is from `crossvenue kappa` over `evals/annotations/questions.rounds.json`. Its round-1 entries for the 25 questions are identical to those frozen in `evals/datasets/annotations/v0.2.1-gold.rounds.json`. The unit is the candidate-level relevant or not relevant decision: 750 (question, chunk) pairs over 25 questions. Intervals are 95% percentile bootstrap intervals over questions: each of 10,000 resamples (seed `20260924`) draws 25 questions with replacement and computes κ on the pooled pairs of the drawn questions.

## Registered estimate: κ 0.470 [0.317, 0.645], below the 0.61 target

Cohen's κ between round 1 as corrected (the gold of `v0.2.1-gold`) and round 2, from the journal `evals/annotations/questions.rounds.json`, over all 25 sampled questions, 750 pairs, with a 95% percentile bootstrap interval over questions:

| Quantity | Value |
|---|---|
| both relevant | 33 |
| both not relevant | 654 |
| disagreements | 63 |
| raw agreement | 0.916 |
| expected agreement by chance | 0.842 |
| Cohen's κ | 0.470 [0.317, 0.645] |
| resamples with undefined κ | 0 of 10,000 |

κ 0.470 is "moderate" on the Landis and Koch bands that [methodology §5](methodology.md#5-annotator-agreement-with-one-annotator) cites, and the target registered there is κ ≥ 0.61 ("substantial"). The point estimate is below the target, and the interval includes it. Under methodology §5, κ is therefore reported as a limitation of `v0.2.1-gold`: the relevance labels of this version are reproducible by their own annotator at κ 0.470 [0.317, 0.645]. The categories with the most disagreement are comparison and ambiguous (next section). Refining the criterion in methodology §5a where they show it ambiguous, and re-labeling the set against it, are outside this version. The κ is computed on the unresolved pair of rounds; no adjudication pass preceded it. The disagreements were adjudicated after it, under the criterion amended after round 2, and the resolved labels are `v0.2.2-gold` ([CHANGELOG](datasets/CHANGELOG.md#v022-gold-2026-10-10)). An exploratory check, decided after round 2, ran the six retrieval methods on `v0.2.2-gold` (`results/20261010T214504Z`) and on `v0.2.1-gold` (`results/20261010T222553Z`), over the 50 comparison questions, the 59 multi-venue ones and the 92 and 90 labeled ones. No comparison changes sign, significance or whether its interval excludes 0 ([findings](../docs/results/findings.md#on-v022-gold-no-retrieval-comparison-changes-sign-significance-or-whether-its-interval-excludes-0)).

## The three sensitivity readings lie within 0.013 of the registered κ

Methodology §5 registers four readings. Round 1 is read as corrected (registered) or as first judged, over all 25 questions or over the 16 unexposed questions, whose gold chunks the answer-rating screen never showed during the quiet period. `evals/selection/round2_exposed.txt` lists the 9 others. Each reading from the journal `evals/annotations/questions.rounds.json` on `v0.2.1-gold`, with its questions and pairs and a 95% percentile bootstrap interval over questions:

| Reading | Questions | Pairs | Raw agreement | κ | Undefined resamples |
|---|---|---|---|---|---|
| corrected, all questions (registered) | 25 | 750 | 0.916 | 0.470 [0.317, 0.645] | 0 of 10,000 |
| corrected, unexposed questions | 16 | 470 | 0.921 | 0.457 [0.283, 0.684] | 0 of 10,000 |
| first judged, all questions | 25 | 750 | 0.913 | 0.460 [0.318, 0.629] | 0 of 10,000 |
| first judged, unexposed questions | 16 | 470 | 0.921 | 0.457 [0.298, 0.683] | 0 of 10,000 |

Each reading's interval includes the registered point estimate.

## Comparison holds 41 of the 63 disagreements, and ambiguous has the lowest κ

Per category, from the journal `evals/annotations/questions.rounds.json` on `v0.2.1-gold`, 750 pairs over 25 questions, most disagreement first; a category of one question is a single question's agreement:

| Category | Questions | Pairs | Disagreements | Raw agreement | κ |
|---|---|---|---|---|---|
| comparison | 11 | 400 | 41 | 0.897 | 0.435 |
| ambiguous | 1 | 25 | 9 | 0.640 | −0.190 |
| fact | 2 | 50 | 5 | 0.900 | 0.390 |
| adversarial | 3 | 75 | 4 | 0.947 | 0.639 |
| version_sensitive | 1 | 25 | 2 | 0.920 | 0.706 |
| multihop | 1 | 25 | 1 | 0.960 | 0.779 |
| noisy | 1 | 25 | 1 | 0.960 | 0.834 |
| unanswerable | 5 | 125 | 0 | 1.000 | - |

The unanswerable questions' 125 pairs are not relevant in both rounds, so κ is undefined there (`crossvenue kappa` prints 1.000 for this case).

## Round 2 marks more pairs relevant: 50 of the 63 disagreements

Round 1 marked 46 of the 750 pairs relevant and round 2 marked 83. Of the 63 disagreements, 13 are relevant in round 1 only and 50 in round 2 only. Per category, from the journal `evals/annotations/questions.rounds.json` on `v0.2.1-gold`, 750 pairs over 25 questions:

| Category | Disagreements | Relevant in round 1 only | Relevant in round 2 only |
|---|---|---|---|
| comparison | 41 | 4 | 37 |
| ambiguous | 9 | 3 | 6 |
| fact | 5 | 3 | 2 |
| adversarial | 4 | 3 | 1 |
| version_sensitive | 2 | 0 | 2 |
| multihop | 1 | 0 | 1 |
| noisy | 1 | 0 | 1 |
| unanswerable | 0 | 0 | 0 |
| total | 63 | 13 | 50 |

Cohen's κ computes chance agreement from each round's own share of relevant verdicts, so this shift between rounds does not lower κ by itself (methodology §5, Agreement with a single annotator).

## Source chunks, marked only on round 1's screen, are 3 of the 13 round-1-only pairs

The round-1 screen marked the chunk a question was drafted from; the round-2 screen hid the marker. The 25 questions list 35 source chunks (`source_chunk_ids`). 18 of them were among the candidates both rounds judged; the other 17 were in neither round's candidates. Verdicts on the 18 source chunks, from the journal `evals/annotations/questions.rounds.json` on `v0.2.1-gold`:

| Verdicts | Source chunks |
|---|---|
| relevant in both rounds | 15 |
| relevant in round 1 only | 3 |
| relevant in round 2 only | 0 |
| not relevant in either round | 0 |

Round 1 marked all 18 source chunks relevant and round 2 marked 15 of them. The 3 source chunks relevant in round 1 only are 3 of the 13 round-1-only disagreements. The other 10 round-1-only pairs and all 50 round-2-only pairs are chunks the marker never flagged. No run separates the marker's effect from other disagreement; methodology §5 states it as a possible cause.

## The retest gap is 15.7 to 15.9 days

Each question's round-1 and round-2 judgments are 15.7 to 15.9 days apart. The gap runs from the question's last round-1 record, its `annotated_at` or its last augmentation, which methodology §5 counts as round 1, to the `annotated_at` of its round-2 judgment. That meets the registered minimum of 14 days, so the 7-day fallback of methodology §5 does not apply.
