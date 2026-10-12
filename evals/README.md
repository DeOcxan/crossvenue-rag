# Evaluation set

The question set, its labels and the record of how both were produced. How to work with them, and the conventions every question set follows, are in [`docs/how-to/label.md`](../docs/how-to/label.md).

| Path | Contents |
|---|---|
| [`methodology.md`](methodology.md) | Evaluation protocol, what the set can prove, deviations from the protocol |
| [`agreement.md`](agreement.md) | Test-retest agreement of the annotator between labeling rounds 1 and 2: κ with its interval, per category, direction of disagreement, source-chunk verdicts |
| [`datasheet.md`](datasheet.md) | Datasheet of the gold set `v0.2.1-gold`, including the record format, and of what `v0.2.2-gold` changes |
| [`domain-research.md`](domain-research.md) | The developer questions and near-duplicate traps the questions are drawn from |
| `questions.jsonl` | Working question set that `eval` and `label` read (`eval.questions`) |
| `questions.example.jsonl` | Five hand-written example questions in the format the harness reads |
| `datasets/` | Immutable versions (`*-candidates`, `*-gold`) with their `.meta.json`, and [`CHANGELOG.md`](datasets/CHANGELOG.md) |
| `datasets/annotations/` | Judgment journal of each gold version, frozen with it |
| `annotations/` | Judgment journal of the working set (per round, the pool shown, the chunks marked, the `index_version`), and the human answer ratings of the judge calibration, `judge_ratings.jsonl` |
| `selection/` | Working lists: `selection.jsonl` (every selection decision), `review_round1.txt` (ids re-opened in review), `augment_round1.jsonl` (chunks judged by augmentation), `round2_sample.txt` (the 25 questions round 2 re-judged, stratified by category and fixed before round 2), `round2_exposed.txt` (the round-2 questions whose gold chunks the answer-rating screen showed, reported separately in the agreement figure), `judge_calibration_sample.txt` (the 40 answers the human rated to calibrate the judge), `d23_answer_chunks.jsonl` (the answer chunks of the questions of deviation D23 that their round-1 candidates missed, which `make relabel-d23` shows) |
| `migrations/` | Chunk-id maps from `crossvenue id-map`, used by `crossvenue remap-ids` |
