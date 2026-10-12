"""Every run or dataset the documents report, by name.

Each constant names a saved artifact under the repository root: a run directory
under `results/`, a dataset under `evals/datasets/`, or a file under `evals/migrations/`
or `evals/selection/`. The claims and the compute functions read these names; a claim
template and scripts/figures.py's labels write a dataset's version (`v0.2.1-gold`) as text.
"""

# Retrieval runs.
RETRIEVAL_RUN = "20260925T012314Z"
NDCG_PRIME_RUN = "20260925T015715Z"
PVALUE_RUN = "20260925T132613Z"
# venue_split with one cross-encoder call for all venues.
ONE_CALL_RUN = "20260925T144223Z"
# The evaluation run after the package restructuring, and the one before it.
AFTER_RESTRUCTURE_RUN = "20260925T024237Z"
BEFORE_RESTRUCTURE_RUN = "20260925T015715Z"
POOL_RUN = "20260925T003721Z_pool"
BENCH_RUN = "20260925T225850Z_bench"
JUDGED_V1 = "20260926T000001Z_judged"
JUDGED_V2 = "20260926T004836Z_judged"
PROMPTS_RUN = "20260926T004851Z_prompts"
POWER_RUN = "20260926T015648Z_power"
ANSWERS_V1 = "20260925T175128Z_answers"
ANSWERS_V2 = "20260925T202827Z_answers"
GOLD = "evals/datasets/v0.2.1-gold.jsonl"
GOLD_BEFORE_AUGMENTATION = "evals/datasets/v0.2-gold.jsonl"
# The gold set with the round-2 disagreements adjudicated and the D23 questions relabeled.
GOLD_ADJUDICATED = "evals/datasets/v0.2.2-gold.jsonl"
# The run made before exact deduplication; the id map of the deduplication migration
# maps its chunk ids to those of the deduplicated corpus.
PRE_DEDUP_RUN = "20260924T200525Z"
DEDUP_IDMAP = "evals/migrations/2026-09-24-exact-dedup.idmap.json"
# The first retrieval run on v0.2-gold.
FIRST_GOLD_RUN = "20260924T234623Z"
ROUND2_SAMPLE = "evals/selection/round2_sample.txt"
# The round-2 questions whose gold chunks the answer-rating screen showed.
ROUND2_EXPOSED = "evals/selection/round2_exposed.txt"
# The sensitivity run on `v0.2.2-gold`, and its baseline on `v0.2.1-gold` with the same
# config; both made with embedding.device=cpu.
SENSITIVITY_RUN = "20261010T214504Z"
SENSITIVITY_BASE_RUN = "20261010T222553Z"
# The working question set, whose journal holds round 2 beside round 1.
WORKING_QUESTIONS = "evals/questions.jsonl"
# The questions of deviation D23, with the answer chunks their round-1 pools missed.
D23_ANSWER_CHUNKS = "evals/selection/d23_answer_chunks.jsonl"
# The answers of the judge calibration.
CALIBRATION_SAMPLE = "evals/selection/judge_calibration_sample.txt"
# The run on `v0.2-gold` that measured the coverage failure.
COVERAGE_FAILURE_RUN = "20260925T002151Z"
# The last candidate set before selection, and the one holding the comparison supplement.
CANDIDATES_FINAL = "evals/datasets/v0.1.6-candidates.jsonl"
COMPARISON_SUPPLEMENT = "evals/datasets/v0.1.2-candidates.jsonl"
