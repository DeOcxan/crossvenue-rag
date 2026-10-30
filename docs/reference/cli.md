# Command-line reference

Every command of `crossvenue`, with its arguments and options as `crossvenue COMMAND --help` prints them. Make targets that wrap a command are listed next to it.

## Common options

Every command except `reproduce` and `runs-manifest` takes these two options.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--config`, `-c` | path of a file | `configs/default.yaml` | YAML config to load: a mapping of sections ([configuration](config.md)) |
| `--set`, `-s` | `section.field=value` | - | One config field set to a value; repeatable, for example `--set retrieval.k=20`. The value is read as YAML; an override without `=`, with an empty key, with a key under a value or with a value that is not valid YAML is refused |

Commands that read `eval.questions` also take `--questions`, `-q` (path of a question file, [question files](config.md#question-files)), which overrides it. `power` reads the question file its retrieval run recorded.

Every option below is checked where it enters, before any model, index or database is opened, unless the check needs one of them. A value outside its stated domain stops the command with a message naming the option and a hint naming the fix. The exception is the VALUE of `retrieve --filter`, as its row says.

With `CROSSVENUE_DEBUG` set to any value but empty, `0`, `false` or `no` (for example `CROSSVENUE_DEBUG=1`), an expected error prints its full traceback instead of the message and hint.

## Corpus

### `fetch`

Downloads the documentation sources as `data/manifest.csv` records them: each git source at its recorded commit, each file checked against its recorded sha256, each GitBook page compared with its recorded hash. A difference stops the command with an error that counts, per venue, the files changed, missing and not in the manifest; the manifest is not written. Make target: `make fetch`.

| Option | Type | Meaning |
|---|---|---|
| `--venue` | str | Venues to fetch |
| `--follow-branch` | flag | Each git source's branch head and every GitBook page as the site serves it, with the manifest rewritten; starts another corpus version |
| `--allow-missing` | flag | Continuing past a GitBook page that fails to download, with the recorded copy kept if there is one |

### `ingest`

Normalizes and chunks everything in the manifest, deduplicates, and writes `data/processed/chunks.jsonl` and the corpus statistics. Make target: `make ingest`. No options beyond the common ones.

### `stats`

Prints corpus statistics from the last ingestion: per-venue counts, chunk-length distribution, exact deduplication, metadata coverage. Make target: `make stats`. No options beyond the common ones.

### `count-term TERM`

Counts chunks containing `TERM`, case-insensitive, over `data/processed/chunks.jsonl`: the project's one counting convention.

| Argument or option | Type | Meaning |
|---|---|---|
| `TERM` | str, required, not blank | Substring to look for |
| `--venue` | str | The one venue of the corpus to count in |

## Index

### `index`

Builds the pgvector and BM25 indexes from `chunks.jsonl` and writes `data/index/index_meta.json`; refuses an empty `chunks.jsonl`. Make target: `make index` (starts Postgres first).

| Option | Meaning |
|---|---|
| `--skip-dense` | The BM25 index alone, with no dense index |

### `id-map OLD_CHUNKS`

Maps every chunk id of an earlier `chunks.jsonl` onto the current chunk with the same venue and text.

| Argument or option | Type | Meaning |
|---|---|---|
| `OLD_CHUNKS` | path, required | `chunks.jsonl` from before the rebuild: each line a JSON object with a string `chunk_id`, `venue` and `text` |
| `--out`, `-o` | path, required | Where to write the map; not a directory. The map names the built index, so one must be built |

### `remap-ids SOURCE [TARGET]`

Carries chunk-id pointers through an id map, and refuses to write if anything other than the ids would change.

| Argument or option | Type | Meaning |
|---|---|---|
| `SOURCE` | path, required | JSONL dataset whose lines round-trip through `json.dumps`, or a `.rounds.json` journal |
| `TARGET` | path | Output path for JSONL, which must not exist, in an existing directory; refused for a journal, which is rewritten in place |
| `--map` | path, required | Map written by `id-map`, for the built index |
| `--field` | str | Id-list fields to rewrite in a JSONL dataset, by default `source_chunk_ids` and `gold_chunk_ids`. Each record that is rewritten must have each as a list of chunk ids. With `--only`, only the listed records are checked and rewritten; without it, every record is. Refused for a journal, whose every chunk-id list is always rewritten: pools and verdicts, corrections, augmentations, adjudicated decisions and the relabel, each keeping what it held before under `migrated_from` |
| `--only` | str | The record ids to rewrite, each the id of a record in `SOURCE`; the others are kept as they are |

## Retrieval and evaluation

### `retrieve QUERY`

Runs one query and prints the top k results. Make target: `make retrieve Q="..." M=all`.

| Argument or option | Type | Default | Meaning |
|---|---|---|---|
| `QUERY` | str, required, not blank | - | The question to search for |
| `--method`, `-m` | str | `hybrid` | One method, or `all` for every method except `venue_split_oracle` |
| `--k` | int ≥ 1, at most the candidate depths (`retrieval.bm25_candidates`, `retrieval.dense_candidates`) for the fused methods and `reranker.top_n` for the reranked ones | `retrieval.k` | Results to show per method |
| `--venue` | str | - | Venues of `data/sources.yaml` to search |
| `--venue-type` | `cex` or `dex` | - | The one venue type to search |
| `--filter`, `-f` | `FIELD=VALUE` | - | Metadata filter; FIELD is one of `environment`, `api_version`, `product` or `doc_type` (venues take `--venue` and `--venue-type`), VALUE one the corpus has. VALUE is checked against the chunks when the search runs, with a message naming the field. A repeated field ORs its values; `none` matches unassigned |

### `eval`

Runs all six methods over the question file and writes `results/<timestamp>/`: `per_query.csv`, `summary.csv`, the per-slice CSVs, `paired.csv`, `off_venue.csv`, `report.md`, `summary.json`, `index.json` and the resolved config. Make target: `make eval`.

`summary.csv` holds each method's metrics, the p50 and p95 of the total latency (`p50_ms`, `p95_ms`), and the nearest-rank p50 and p95 of each retrieval step and of the steps' sum at full precision: `encode_p50_ms`, `encode_p95_ms`, `search_*`, `fuse_*`, `rerank_*` and `total_*`. Runs written before these step columns existed have the per-step table in `report.md` only.

In `paired.csv` and `off_venue.csv`, `better`, `worse` and `ties` count questions in the metric's direction, which the `higher_is_better` column states: for `off_venue@k` and `via_ccxt@k` a smaller share is better. A `paired.csv` without the `higher_is_better` column counts a question as `better` when a > b; every metric such a file holds (`recall@k`, `venue_coverage@k`) is higher-is-better, so the counts are the same.

| Option | Type | Meaning |
|---|---|---|
| `--questions`, `-q` | path of a question file | Question file in place of `eval.questions` |
| `--venue-filter` | flag | Each query restricted to the question's labeled `venues`, so it reads the labels, as venue_split_oracle does; no committed run uses it |

### `pool-run`

Runs retrieval only, for building a labeling pool: writes `results/<timestamp>_pool/`, computes no metrics, and is never quoted as an evaluation.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--method`, `-m` | str, required | - | Method(s) to retrieve with, from `bm25`, `dense`, `hybrid`, `hybrid_rerank`, `venue_split` and `venue_split_oracle` (the oracle's top k is judged in augmentation too); repeatable |
| `--k` | int > max(`eval.k_values`) | `eval.pool_depth_per_method` | Retrieval depth; a pool no deeper than the scored ranks adds no unjudged chunk ([ADR 0005](../adr/0005-labeling-pool.md)) |
| `--category` | question category | - | Only questions of this category |

### `power`

Simulates the power of the Holm-adjusted paired randomization test on per-question recall@k over one question category. The discordance and difference sizes are those of the run's comparison family (the secondary `paired.csv` rows of recall@k over the scope `labelled <category>`). Writes `results/<timestamp>_power/`: `power.json` (the model's parameters, power at α/family and at α for each δ, and the smallest δ reaching 80% power) and `index.json` (the source run's `index_version` and question file). No model call.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--run` | path, required | - | A retrieval run with `index.json`, `per_query.csv` and `paired.csv`, whose question file is still there with the sha256 the run recorded |
| `--category` | str | comparison | Question category; the run must compare recall@k on it |
| `--k` | int ≥ 1 | max(`eval.k_values`) | recall@k; the run must compare it on the category (the message lists the pairs it does compare) |
| `--studies` | int ≥ 1 | 2000 | Simulated studies per δ |
| `--permutations` | int ≥ 1 | 2000 | Sign flips per randomization test |

### `bench-rerank`

Times venue_split with one cross-encoder call per named venue against one call for all venues, on the questions whose text names two or more venues. Both modes run back to back for each question, in alternating order, after the models are loaded and warmed up. The machine counts as busy when the 1-minute load average exceeds half the CPU count or another `crossvenue`, `pytest`, `uvicorn` or `sentence_transformers` process uses at least 5% CPU. Other processes at or above 50% CPU are recorded and do not block. Before each question's pair of timings the run waits for a quiet machine, and a pair after which the machine is busy is discarded and repeated. A wait or a run of discarded pairs past the limits in `evaluation/bench.py` stops the run with an error that names `--force`. Writes `results/<timestamp>_bench/`: `raw.csv` (one row per question, round and mode: total and rerank ms, calls, pairs per call, whether both modes retrieved the same ids, the load average before the pair), `summary.csv` (per mode: nearest-rank p50 and p95 and the mean of total and rerank ms, calls per query, pairs per call), `machine.json` (load average and the number of competing and busy processes before and after, never their command lines; the discarded pairs; the load during timing), `config.yaml`, `index.json` (`index_version`, question file and its sha256) and `BENCH.md`. No model call beyond the local encoders.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--rounds` | int ≥ 1 | 1 | Passes over the questions; the order alternates between rounds too |
| `--force` | flag | off | Timing with no wait for a quiet machine; the state is still recorded in `machine.json` |

## Question set

### `select`

Chooses the evaluation set from a candidate file, recording every decision in `evals/selection/selection.jsonl`. Make targets: `make select`, `make select-finalize`.

| Option | Type | Meaning |
|---|---|---|
| `--dataset`, `-d` | path | Candidate file; default the newest `evals/datasets/*-candidates.jsonl` |
| `--category` | str | The categories to review; default all |
| `--include-skipped` | flag | Skipped questions, shown again |
| `--finalize` | flag | The question file, written from the accepted candidates |

Categories come in order of risk of running out: ambiguous, noisy, version_sensitive, unanswerable, then comparison, adversarial, multihop, fact. Each question is shown with its metadata and a preview of every chunk it is grounded in. The progress panel shows `unseen` (left to look at) and `available` (could still be accepted), and flags `no headroom` when the next rejection would make a target unreachable. Decisions are written after every keystroke. `--finalize` refuses a set that misses a category target or has fewer than 50 answerable comparison questions.

### `label [IDS]...`

Marks gold chunks in the terminal. Make targets: `make label` (round 1), `make relabel` (round 2, from `evals/selection/round2_sample.txt`), `make adjudicate` (`--adjudicate`), `make round3` (`--round 3`) and `make relabel-d23` (`--relabel` on the questions of deviation D23).

| Argument or option | Type | Default | Meaning |
|---|---|---|---|
| `IDS` | str | - | Question ids to review, with `--review`, or to relabel, with `--relabel` |
| `--review` | flag | off | The listed labeled questions, re-opened with their current marks; changes are recorded as corrections to round 1 |
| `--review-file` | path | - | A file of ids to review, first token per line |
| `--augment` | flag | off | Only the never-judged chunks listed in `--question-file`, judged; extends round 1 |
| `--question-file` | path | - | Candidate list for `--augment`, written by `augment-candidates` |
| `--adjudicate` | flag | off | A decision on each (question, chunk) pair that round 1 and round 2 judged differently; recorded under `adjudicated` in the journal |
| `--relabel` | flag | off | The listed questions, judged again on the candidates round 1 judged; recorded under `relabeled` in the journal |
| `--add-chunks` | path | - | With `--relabel`: chunks shown after the recorded candidates. With `--round 3`: chunks shuffled in among them. One `{"id": ..., "chunk_ids": [...]}` object per line |
| `--metadata-from` | path | - | With `--round 3`: the dataset the questions are shown from; the journal stays the one of the question file |
| `--all` | flag | off | Questions that already have labels, shown again |
| `--limit` | int | `eval.pool_display` | Candidates shown per question |
| `--pool-depth` | int | `eval.pool_depth_per_method` | Depth taken from each method; must exceed max(`eval.k_values`) |
| `--round` | int | 1 | Annotation round; 1 writes gold labels, 2 re-annotates a subsample, 3 relabels the whole set, blind; 2 and 3 leave round 1 untouched |
| `--annotator` | str | - | Recorded on every judgment |
| `--sample` | int | - | N questions to label: at random in round 1, stratified by category in round 2 |
| `--seed` | int | 0 | Seed for `--sample` |
| `--sample-file` | path | - | Round 2: exactly the questions listed here, to label, written by `round-sample` |

Round 1 pools every pool method's top `--pool-depth` with RRF and shows `--limit` distinct candidates. The screen is compact: number, venue, heading path, which methods found it at which rank, the first line. `v<n>` shows candidate *n* in full with every occurrence of its text and each occurrence's `environment` / `api_version` / `product` / `doc_type`; `f` toggles full text for the table. `s` after a number marks one of the question's `source_chunk_ids`.

Two checks run before a judgment is saved. Comparison coverage: if no marked chunk comes from a venue the question names, the tool says which. It then asks for `b` back, `a` the answer is absent from that venue's docs, or `m` no suitable candidate of that venue in the pool. The answer is stored per venue in `venue_coverage`. Version and environment: for `version_sensitive` and `adversarial` questions the notes and every marked chunk's `environment` / `api_version` are shown; Enter saves, `b` goes back.

`label` and `eval` refuse to run if any judgment on record carries another `index_version`; the message says whether to carry ids across with `id-map` and `remap-ids` or to re-judge.

`--review` re-opens labeled questions on the pool round 1 recorded, with current marks starred. A change replaces the verdict and is appended to `corrections` in the journal with the previous set, and an unchanged question writes nothing. `--augment` shows only the listed chunks, with the current gold as context, records every chunk shown and appends relevant ones.

Round 2 (`--round 2 --sample-file`, `make relabel`) shows each listed question with exactly the candidates round 1 judged (its recorded pool, plus the augmentation chunks of comparison questions). The candidates are shuffled with a fixed seed that is written to the journal. The screen has no marks, no "found by" column and no source-chunk markers. Nothing is retrieved again, so the index must be the one round 1 was judged against, and `label` refuses to start otherwise. Round 2 writes only to the journal, never to gold.

`--adjudicate` walks every pair that round 1, as corrected, and round 2 judged differently, grouped by question in question id order and, within a question, in round-2 screen order. For each question it shows the question, category, venues and the criterion of methodology §5a as amended on 2026-10-10. For each pair it shows the chunk in full with its venue and heading and both earlier verdicts, labeled round 1 and round 2. `v<n>` shows another pair of the question in full, and `f` lists the question's pairs, compact or full. `y` records relevant, `n` not relevant, and `b` goes back one pair. Each decision is written to the journal under `adjudicated` as it is made, with its timestamp and the criterion date. Ctrl-C therefore loses nothing, and the next run resumes at the first pair without a decision. `--relabel ID ...` shows each listed question on the round-1 screen with the candidates round 1 judged followed by those `--add-chunks` names, and journals the verdict under `relabeled`. Both modes read chunks from the corpus file, check the built index against every recorded judgment, and write neither the question file nor the round-1 and round-2 entries, so `kappa` is unchanged by them.

Round 3 (`--round 3 --add-chunks FILE --metadata-from DATASET`, `make round3`) walks every question without a round-3 entry, on the round-2 screen, from the chunks file and with no database. Above the candidate table it shows the criterion of methodology §5a as amended. The candidates are what round 1 judged plus the added chunks, shuffled per question with seed 20261031, recorded as `shuffle_seed`. Each verdict is written when it is given, under key `3` of the journal, with the `criterion` date and `relevant` in candidate order; nothing else is written. `u` re-opens the previous question of the session without its marks. An unanswerable question refuses numbers and takes `n`, or `x`, which records `flagged: true`. Ctrl-C stops, and the next run resumes at the first question without an entry. `--round 3` refuses ids and the other modes' options, and a later round is refused.

### `augment-candidates`

Lists the chunks a run's methods retrieved that no round-1 judgment showed.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--run` | path, required | - | A run directory, `results/<ts>` or `results/<ts>_pool` |
| `--out`, `-o` | path, required | - | Candidate list to write (JSONL) |
| `--method`, `-m` | str, required | - | Method(s) whose top k to judge; repeatable |
| `--category` | str | - | Only questions of this category |
| `--k` | int | `eval.pool_depth_per_method` | Depth of each method's list; must exceed max(`eval.k_values`) |

### `derive-coverage`

Fills the `venue_coverage` values that follow from recorded data, journaled as derived.

| Option | Type | Meaning |
|---|---|---|
| `--exclude-file` | path | Ids to leave unchanged, first token per line |

### `round-sample`

Draws the round-2 subsample, stratified by category (proportional to size, at least one per category), from the questions round 1 judged, prints it, and writes it to a file that `label --sample-file` reads. Make target: `make round2-sample`.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--n` | int | 25 | Questions in the subsample |
| `--seed` | int | 20260924 | Sampling seed |
| `--out`, `-o` | path | `evals/selection/round2_sample.txt` | Sample file to write |

It prints the questions and their category counts; the same seed always gives the same list.

### `kappa`

Cohen's kappa between two annotation rounds over the candidates judged in both. Make target: `make kappa`. It prints the registered reading first: round 1 as corrected, over all sampled questions. Then come the range of the retest gap in days, the agreement per question category with the most disagreements first, and the three sensitivity readings of methodology §5. Each sensitivity reading changes one or both of two choices. One is round 1 as first judged (each entry's `previous_relevant` before its earliest correction, plus augmentation) instead of as corrected. The other is only the questions not listed in `--exposed` instead of all. Last come each reading's percentile bootstrap interval over questions (`eval.bootstrap_resamples` resamples, seed `eval.bootstrap_seed`) and the disagreements split by the round that judged the pair relevant, in total and per category. The final table holds the verdicts both rounds gave the questions' source chunks, which the round-1 screen marked and round 2 hid. A gap under 14 days (methodology §5) is refused unless `--short-gap` is set.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--first` | int | 1 | First round |
| `--second` | int | 2 | Second round |
| `--exposed` | path | `evals/selection/round2_exposed.txt`, when it exists | Questions whose gold was shown before the second round; a path given that does not exist is refused |
| `--short-gap` | flag | off | A kappa reported although a question was re-judged within 14 days of round 1; the output marks it as not the registered measurement |

### `round-agreement`

The agreement of round 3 with round 1 as corrected, over the pairs both judged, across a criterion revision. Make target: `make round3-agreement`. It prints the verdict counts, the raw agreement and, by category, the direction of disagreement, with no κ and no target: reliability under the amended criterion was not measured. It refuses until round 3 has judged every question.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--metadata-from` | path | `evals/datasets/v0.2.2-gold.jsonl` | The dataset that gives each question's category |

### `round3-gold`

Writes the gold set built from round 3 alone, then lists the flagged questions. Make target: `make round3-gold`. Each question keeps every field of `--base` except `gold_chunk_ids`, `venue_coverage`, `annotator`, `annotated_at` and `annotation_round`. It refuses an existing `--out`, an unfinished round, and a journal in which any entry besides round 3 differs from the journal frozen with `--base`.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--base` | path | `evals/datasets/v0.2.2-gold.jsonl` | The version every other field comes from |
| `--out`, `-o` | path | `evals/datasets/v0.3-gold.jsonl` | Dataset file to write |

### `validate-evals`

Validates datasets against the current corpus. Make target: `make validate-evals` (full check).

| Option | Type | Meaning |
|---|---|---|
| `--dataset`, `-d` | path | Dataset file(s); default every `evals/datasets/*.jsonl` |
| `--skip-near-duplicates` | flag | No near-duplicate check, the one that needs the encoder, as in CI |

Errors: a file with no questions, schema and category / answerability / expected-behavior consistency, duplicate ids, duplicate question text, venues not in the corpus, stale `gold_chunk_ids`, a `notes` reference to a question that does not exist, a labeled answerable comparison without `venue_coverage`, and a `venue_coverage` that disagrees with the question's venues or its gold chunks. Warnings: stale `source_chunk_ids`, an answerable question that names no venue, a `notes` reference to the question itself or to a question sharing no venue, near-duplicate questions (cosine ≥ `eval.duplicate_threshold`), a question sharing 6 or more consecutive words with one of its gold chunks, quoted spans excluded (methodology §6), lopsided category or venue balance, no metadata file beside the dataset, and a check skipped because `chunks.jsonl` is missing, no encoder is available or the manifest is empty. It checks each version's `.meta.json` against the file (the counts), the corpus and the journal in `evals/datasets/annotations/` (every gold list equals what its journal entries resolve to). The corpus check compares the manifest sha256 and `index_version`, and a gold version made against another index is an error. Where the `.meta.json` records them (`dataset_sha256`, `journal_sha256`), it also checks the sha256 of the file and of the journal.

## Answer generation

These commands call models through OpenRouter, which is paid. The key is `OPENROUTER_API_KEY`, from the environment or the repository's dotenv file (template: `.env.example`), and is never printed. Responses that pass validation are cached in `generation.cache_dir`, so repeating a command with the same configuration costs nothing for every call that succeeded. A call whose responses all failed validation is paid again unless any run has cached a valid response to the same request. Each run stops before a call whose expected cost, the most expensive known call of that model, would pass its budget (`generation.budget_usd`, or `--budget`). The first call of a model with no cached call is not bounded.

### `ask QUESTION`

Answers one question with the configured retriever and generator. It prints the answer, each claim with its venue, and each citation with its chunk's venue, its document path and the URL of the occurrence. It also prints the outcome under the rule of methodology §9a: refused (a partial answer without a claim counts as one), asks for clarification, partial with the venues the documentation lacks, or full. Make target: `make ask Q="..."`.

| Argument or option | Type | Default | Meaning |
|---|---|---|---|
| `QUESTION` | str, required, not blank | - | The question |
| `--retriever` | str | `venue_split` | Retrieval method: `bm25`, `dense`, `hybrid`, `hybrid_rerank` or `venue_split`; not `venue_split_oracle` |
| `--generator` | str | `gemini` | A key of `generation.generators` |

### `answer-eval`

Generates an answer for every configuration of `generation.matrix` on every question and optionally has a judge grade each one. It writes `results/<timestamp>_answers/`: `answers.jsonl`, `per_answer.csv`, `summary.csv`, `paired.csv`, `report.md`, the resolved config and `index.json` (`index_version`, question file and its sha256). Refuses a prompt file whose sha256 differs from `generation.generator_prompt_sha256` or whose version line does not match `answer_schema_version`. Metrics whose judge dimension failed calibration (`generation.calibration_run`) are marked uncalibrated in `report.md` and by the `calibrated` column of `paired.csv`. `paired.csv` counts `better`, `worse` and `ties` in each metric's direction (`higher_is_better`): lower is better for venue_attribution_error, higher for faithfulness (the share of claims supported), coverage and `behaviour_correct`. A `paired.csv` without these columns records no per-question counts. The primary comparison is primary only under the prompt it was registered for (`generation.primary_prompt_sha256`). Refuses records that more than one judge graded, naming the judges, before it writes the run directory.

| Option | Type | Meaning |
|---|---|---|
| `--questions`, `-q` | path of a question file | Question file in place of `eval.questions` |
| `--question-id` | str | Only these questions, each an id of the question file; repeatable |
| `--judge` | str | A judge candidate to grade the answers too, a key of `generation.judges` (`sonnet` or `opus`) |
| `--budget` | finite float > 0 | Run budget in USD; default `generation.budget_usd` |
| `--estimate` | flag | A cost estimate alone, with no model call: every generator request is built, the cached ones are counted and the rest are priced at each model's mean cached cost. Every judge call is priced at the judge's mean, cached or not, since a judge request holds an answer not yet generated |
| `--note` | str | A line written into `report.md` |

### `rate-sample`

Draws and pins the judge-calibration sample: `generation.calibration_size` answers of a run, stratified by configuration and category, seeded, with the questions of `generation.calibration_exclude` left out, written to a file before any rating.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--run` | path, required | - | An answer run, `results/<ts>_answers`, made against the current index; `generation.calibration_exclude` may name only questions it answered |
| `--out`, `-o` | path, not a directory | `evals/selection/judge_calibration_sample.txt` | Sample file |

### `rate`

Grades the calibration sample by hand, in the judge's schema; ratings are appended to `evals/annotations/judge_ratings.jsonl`. No judge's grade is ever shown.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--sample-file` | path | `evals/selection/judge_calibration_sample.txt` | Written by `rate-sample`: names its run, an answer run at the current index, and answers of that run |
| `--rater` | str | login name | Recorded on every rating; blank means the login name |

### `judge-sample`

Has one judge candidate grade the calibration sample and writes `judgements_<judge>.jsonl` into the run (`judgements_<judge>_r1.jsonl` for the self-consistency repeat). Prints the number of sampled answers judged, the number of judge errors, the spend and the output path. A judge error is a judgment still invalid after the retries or a call that failed after its retries; each is recorded as an `error` row and left out of the agreement. When the budget stops the run, it also prints `stopped:` with the reason. The verdicts are written only to the file, so the terminal shows none before the human has rated the sample.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--judge` | str, required | - | Judge candidate, a key of `generation.judges` (`sonnet` or `opus`) |
| `--sample-file` | path | `evals/selection/judge_calibration_sample.txt` | Written by `rate-sample`, as for `rate` |
| `--replicate` | 0 or 1 | 0 | 0 for the first grading, 1 for the repeat |
| `--budget` | finite float > 0 | `generation.budget_usd` | Run budget in USD |

### `judge-agreement`

Reports how one judge candidate agrees with the human ratings and with its own repeat, per dimension: agreement and Cohen's κ for `supported`, `venue_correct` and `behaviour_correct`; Pearson and Spearman correlation for coverage.

| Option | Type | Meaning |
|---|---|---|
| `--run` | path, required | An answer run, with an `answers.jsonl` |
| `--judge` | str, required | Judge candidate, a key of `generation.judges` |

### `select-judge`

Compares every candidate in `generation.judges` with the human ratings, the candidates with each other, and each candidate with its self-consistency repeat where one exists. It reports agreement and Cohen's κ for the binary dimensions and Spearman for coverage, each with a 95% bootstrap interval over answers (seed and resamples from `eval`). It also reports the number of resamples in which κ is undefined. Applies the rule of methodology §9: of the 4 dimensions in `generation.calibration_thresholds`, the candidate that meets more thresholds is selected; ties go by κ on `venue_correct`, then Spearman on coverage, then cost. It writes `calibration.json` and `calibration.md` into the run. No model call.

| Option | Type | Meaning |
|---|---|---|
| `--run` | path, required | The answer run the calibration sample was drawn from; `generation.judges` must name a candidate, the ratings file hold a rating, and `generation.calibration_exclude_behaviour` name only answers of the run |

### `judge-run`

Judges a saved answer run without generating again: each record of `answers.jsonl` is copied unchanged with the judge's grading added. The run is written to `results/<timestamp>_judged/` with the same files as `answer-eval` (`answers.jsonl`, `per_answer.csv`, `summary.csv`, `paired.csv`, `report.md`, `config.yaml`, `index.json`) plus `source.json` (source run, judge, spend). An answer shared by two configurations is one cached judge call. A judgment that stays invalid after the retries, or whose call fails after its retries, is recorded in the record's `judge_error`. Refuses a run made against another `index_version` than the current one (the run's `index.json`, or its entry in `results/runs.json` for runs saved before runs recorded it). `judge-sample`, `rate`, `rate-sample` and `compare-prompts` apply the same check, and `rate` writes the version into every rating.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--run` | path, required | - | An answer run made against the current index |
| `--judge` | str | `generation.judge` | Judge name, a key of `generation.judges` |
| `--budget` | finite float > 0 | `generation.budget_usd` | Run budget in USD |
| `--note` | str | - | A line written into `report.md` |

### `compare-prompts`

Tests H-v2 of methodology §9a: prompt v1 against prompt v2 per configuration, paired on questions. Wrong refusals by exact McNemar over the questions that expect an answer or a clarifying question; venue_attribution_error by paired bootstrap and randomization test; Holm across configurations within each metric. Also checks v2's partial answers for claims about a venue they list as missing and for claims whose cited chunks all belong to other venues. Writes `results/<timestamp>_prompts/` (`comparison.csv`, `side_effects.jsonl`, `report.md`, `source.json`, `index.json`). No model call. In `comparison.csv`, `diff` is v2 − v1, and `better` and `worse` count questions where v2 is better or worse than v1 in the metric's direction: fewer wrong refusals, a lower venue_attribution_error. A `comparison.csv` without the `higher_is_better` column counts the same way.

Refuses a configuration that has no question both runs answered and that expects an answer or a clarifying question: wrong refusals are H-v2's test for every configuration. A configuration where no question has claims in both versions' answers gets no venue_attribution_error row; `report.md` names it as not computable.

| Option | Type | Meaning |
|---|---|---|
| `--v1` | path, required | Judged run of prompt v1: its answers are of answer schema 1 |
| `--v2` | path, required | Judged run of prompt v2, another run than `--v1`: its answers are of answer schema 2 |

## Service

`make serve` runs the FastAPI app (`crossvenue_rag.service.app`) on 127.0.0.1:8000 with the config in `CROSSVENUE_CONFIG`, the path of a config file (default `configs/default.yaml`).

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | - | `status`, `answering` (`available`, or why not), the generator names, `top_k`, `budget_usd`, `spent_usd` |
| `POST /answer` | `{"question": str, "retriever": "venue_split", "generator": "gemini"}` | What `ask` prints, as JSON |

`retriever` is one of `bm25`, `dense`, `hybrid`, `hybrid_rerank`, `venue_split`. An empty question, an unknown generator or retriever, or a field the body does not have is a 422. A failed model call is a 502, and a key not set at startup or a spent budget is a 503. The key is read once, at startup, and one budget (`generation.budget_usd`) covers every request until the process restarts.

## Reporting

Both commands run from the repository root, which holds `.gitignore` and `results/`; from another directory they stop.

### `reproduce`

Checks the committed runs, then the numbers of the documents `reporting/reproduce.py` lists against their sources, as [design](../explanation/design.md#checking-the-documents-against-their-sources) describes. Prints each discrepancy as `file:line: message` and exits 1 if there is one. Make target: `make reproduce`. Not run in CI.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--verbose`, `-v` | flag | off | Details of the check: the values it lists, with their kind; the unsaved ones always print |

### `runs-manifest`

Writes `results/runs.json`: every run committed to the repository, the sha256 of each of its files, and the `index_version` it was made against. The `index_version` comes from the run's `index.json`. For answer runs saved before runs recorded it, the version is derived: every answer's chunks equal the top k of a retrieval run at that version. A prompts or power run inherits it from the runs it was computed from. It is run once after another run is committed; `make reproduce` then fails if a listed file changes.

## Diagnostics

### `check`

Validates the config and reports which artifacts exist; a damaged `index_meta.json` is reported on the index line. Read-only. Make target: `make check-config`. No options beyond the common ones.

## Make targets without a command

- `make help` lists every target with its one-line description.
- `make install` runs `uv sync --locked --group dev`: the environment from `uv.lock` with the dev group.
- `make db` runs `docker compose up -d` and waits until Postgres accepts connections.
- `make db-down` runs `docker compose down`: it stops Postgres and keeps its volume.
- `make test` runs `pytest`, which leaves out the slow full mutation test.
- `make lint` runs `lint-docs` and `lint-links`, then `ruff check`, `ruff format --check`, `mypy`, `lint-imports`, `tests/test_repo_hygiene.py` and `tests/test_code_hygiene.py`.
- `make lint-docs` runs Vale over `README.md`, `docs/` and `evals/`, and fails on any alert at warning level or above.
- `make lint-links` runs lychee offline over the same files, checking every relative link and heading anchor.
- `make check` runs `lint`, then every test (`pytest -m ""`) and `crossvenue reproduce`.
- `make figures` runs `scripts/figures.py` with the `figures` extra: every figure of findings, as SVG and PNG, from the saved runs.
- `make fmt` runs `ruff format .` and `ruff check --fix .`.
- `make clean` removes `data/processed` and `data/index`; `data/raw`, the manifest and every run in `results/` stay.
