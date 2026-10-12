# Configuration reference

Every config key the pipeline reads, with the value `configs/default.yaml` sets, and the settings kept in code: `fetch.user_agent`, which the YAML does not set, and the pool methods, a code constant. The schema is `src/crossvenue_rag/config.py` (pydantic), with its sections in `config_pipeline.py`, `config_eval.py` and `config_generation.py` beside it. `src/crossvenue_rag/config_loading.py` reads the file and applies the overrides. The Value column is what the YAML sets; where the schema default, which applies when a key is absent, differs from it, the Meaning column says so. Any field can be overridden from the command line with `--set section.field=value`, so an ablation is a config change, never a code change. The value is read as YAML (`--set eval.methods=[bm25]` gives a list). An override without `=`, with an empty key, with a key under a value that is not a section, or with a value that is not valid YAML is refused. The config file must be a UTF-8 file holding a YAML mapping of sections, and every section refuses a key it does not have and a YAML boolean (`yes`, `on`, `true`) for a number. No environment variable sets a key, except `CROSSVENUE_PG_DSN` for `store.dsn`. Every `eval` run copies the resolved config into `results/<timestamp>/config.yaml`.

## `paths`

| Key | Type | Value | Meaning |
|---|---|---|---|
| `sources` | path | `data/sources.yaml` | Source registry: repositories, globs, licenses, metadata rules, venue aliases |
| `manifest` | path | `data/manifest.csv` | Per-document provenance written by `fetch` |
| `raw` | path to a directory, created if absent | `data/raw` | Fetched documents; `fetch` refuses a path that is an existing file |
| `processed` | path | `data/processed` | `chunks.jsonl` and the corpus statistics |
| `index` | path | `data/index` | BM25 index and `index_meta.json` |
| `results` | path | `results` | Run directories |

## `chunking`

| Key | Type | Value | Meaning |
|---|---|---|---|
| `max_tokens` | int > 0 | 450 | Token budget of a rendered chunk, breadcrumb included, counted with the embedding model's tokenizer. At most the tokens of text that tokenizer states the model reads (510 for `bge-base-en-v1.5`, 512 with its two special tokens), checked at `ingest` |
| `overlap_tokens` | int ≥ 0 | 60 | Most tokens a chunk closed at `max_tokens` carries into the next chunk, as whole trailing blocks after its last code block; a chunk closed at a heading carries none; must be below `max_tokens` |
| `min_tokens` | int ≥ 0 | 40 | A heading section shorter than this merges into the next one; must not exceed `max_tokens` |
| `keep_code_blocks_intact` | bool | true | Whether a fenced code block stays whole up to `max_code_block_tokens` |
| `max_code_block_tokens` | int > 0 | 1200 | A code block up to this size may exceed `max_tokens`; above it, the block is split |

## `embedding`

| Key | Type | Value | Meaning |
|---|---|---|---|
| `model_name` | str, not blank | `BAAI/bge-base-en-v1.5` | Bi-encoder for dense retrieval and the near-duplicate question check; a model that cannot be loaded is refused when first used |
| `dimension` | int > 0 | 768 | Vector dimension; must match the model |
| `query_prefix` | str | `Represent this sentence for searching relevant passages: ` | Prepended to queries; bge models are trained with it (schema default: empty) |
| `document_prefix` | str | empty | Prepended to documents |
| `normalize` | bool | true | Whether vectors are L2-normalized |
| `batch_size` | int > 0 | 32 | Encoding batch size |
| `device` | `cpu`, `cuda`, `mps` or `xpu`, optionally `:N`, or null | null | null picks mps, cuda or cpu; a device this machine does not have is refused when the model loads |
| `revision` | non-empty str or null | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | Hugging Face commit of the weights and of the tokenizer that counts the chunk budget; recorded in `index_meta.json` (schema default: null, the head of the model's default branch) |

`model_name`, `dimension`, `document_prefix` and `normalize` enter the `index_version`; `revision` does not, a gap [ADR 0004](../adr/0004-index-version.md#consequences) records.

## `reranker`

| Key | Type | Value | Meaning |
|---|---|---|---|
| `model_name` | str, not blank | `BAAI/bge-reranker-base` | Cross-encoder for `hybrid_rerank` and `venue_split`; a model that cannot be loaded is refused when first used |
| `top_n` | int > 0 | 50 | Fused candidates passed to the cross-encoder; must be at least `retrieval.k`, the largest `eval.k_values`, `eval.pool_depth_per_method` and `generation.top_k` |
| `batch_size` | int > 0 | 16 | Cross-encoder batch size |
| `revision` | non-empty str or null | `2cfc18c9415c912f9d8155881c133215df768a70` | Hugging Face commit of the cross-encoder; recorded in `index_meta.json` (schema default: null, the head of the model's default branch) |

## `retrieval`

| Key | Type | Value | Meaning |
|---|---|---|---|
| `k` | int > 0 | 10 | Results returned per query |
| `dense_candidates` | int > 0 | 50 | Dense candidates before fusion; at least `k` and the largest `eval.k_values` |
| `bm25_candidates` | int > 0 | 50 | BM25 candidates before fusion; at least `k` and the largest `eval.k_values` |
| `rrf_k` | int > 0 | 60 | Reciprocal rank fusion constant (Cormack et al. 2009; full citation in the [methodology references](../../evals/methodology.md#references)) |

## `store`

| Key | Type | Value | Meaning |
|---|---|---|---|
| `dsn` | libpq URI (`postgresql://` or `postgres://`) or `key=value` pairs | `postgresql://crossvenue:crossvenue@localhost:5433/crossvenue` | Postgres connection; a non-empty `CROSSVENUE_PG_DSN` in the environment takes precedence and is checked the same way. `Config.to_yaml` leaves the key out of a run's saved `config.yaml`, since it can hold a password; the committed runs under `results/` were saved before it did and hold the default DSN |
| `table` | str, at most 49 characters | `chunks` | pgvector table; ASCII letters, digits and `_`, because it is spliced into DDL, and short enough that its index names stay within Postgres's 63-byte identifier limit |
| `hnsw_m` | int in [2, 100] | 16 | HNSW graph degree, within pgvector's bounds |
| `hnsw_ef_construction` | int in [4, 1000] | 64 | HNSW build-time candidate list; at least twice `hnsw_m`, as pgvector requires |
| `hnsw_ef_search` | int in [1, 1000] | 100 | HNSW query-time candidate list, set per transaction; a search deeper than this, or filtered by more than one venue or by metadata, is exact instead ([design](../explanation/design.md#retrieval)) |

## `fetch`

Applies to `kind: gitbook` sources; git sources ignore it.

| Key | Type | Value | Meaning |
|---|---|---|---|
| `request_delay_seconds` | finite float ≥ 0 | 0.5 | Pause between requests |
| `timeout_seconds` | finite float > 0 | 30 | Per-request timeout |
| `user_agent` | str, not blank | schema default `crossvenue-rag/<version> (research corpus builder)` | Sent with every request; `<version>` is the installed package version (`importlib.metadata`). Not set in `configs/default.yaml` |
| `max_retries` | int in [0, 10] | 3 | Retries per request; the wait doubles on each |
| `backoff_seconds` | finite float > 0 | 0.5 | First retry delay; doubles on each retry, or follows `Retry-After` when longer. Apart from `request_delay_seconds`, so pacing can be 0 |

## `eval`

| Key | Type | Value | Meaning |
|---|---|---|---|
| `questions` | path of a file | `evals/questions.jsonl` | Question file `eval`, `label` and `select` read; each line a JSON object whose fields have the types of a question, and no id twice ([Question files](#question-files)) |
| `methods` | non-empty list of methods, each once | all six: `bm25`, `dense`, `hybrid`, `hybrid_rerank`, `venue_split`, `venue_split_oracle` | Methods `eval` runs (schema default: the first four) |
| `k_values` | list of distinct int > 0 that includes 1 | `[1, 5, 10]` | Cut-offs for every metric; sorted on load; the top-1 off-venue comparison reads off_venue@1 |
| `breakdowns` | list of `venue`, `qtype`, `category` | all three | Slices reported next to the aggregate (schema default: `venue`, `qtype`) |
| `duplicate_threshold` | float in [0, 1] | 0.92 | Question-embedding cosine at which `validate-evals` flags a near-duplicate pair ([methodology §6](../../evals/methodology.md#6-decontamination-and-deduplication-of-questions)) |
| `pool_depth_per_method` | int > 0 | 20 | Candidates each pool method contributes to a labeling pool; must exceed `max(k_values)` |
| `pool_display` | int > 0 | 25 | Fused candidates shown to the annotator; must be at least `pool_depth_per_method` |
| `paired_comparisons` | list of `[a, b]` | `[hybrid_rerank, hybrid]`, `[venue_split, hybrid_rerank]`, `[venue_split_oracle, hybrid_rerank]`, `[venue_split_oracle, venue_split]`, `[hybrid_rerank, dense]`, `[hybrid_rerank, bm25]` | Pairs `report.md` compares as a minus b; a method is never paired with itself and a pair appears once; a pair whose methods `methods` does not run is skipped (schema default: none) |
| `primary_comparison` | `[a, b, metric, scope]` | `[venue_split, hybrid_rerank, venue_coverage@10, comparison]` | The one comparison the study is designed around; every other pair is secondary and Holm-adjusted within its metric and scope. `(a, b)` must be in `paired_comparisons`, the metric `venue_coverage@K` or `recall@K` with K the largest `k_values`, and the scope one of that metric's two (`multi-venue` or `comparison`; `labelled` or `labelled comparison`) (schema default: none) |
| `bootstrap_resamples` | int > 0 | 10000 | Paired bootstrap resamples, and permutations of the paired randomization test |
| `bootstrap_seed` | int ≥ 0 | 20260924 | Seed of the bootstrap and of the randomization test |

The pool methods themselves are a code constant, `retrieval.methods.POOL_METHODS`, which no config key sets.

## `generation`

Every model is called through OpenRouter ([ADR 0009](../adr/0009-generators.md)).

| Key | Type | Value | Meaning |
|---|---|---|---|
| `top_k` | int > 0 | 8 | Chunks of the chosen retriever given to the generator |
| `cache_dir` | path | `data/cache/llm` | Response cache, keyed by a hash of the request and a replicate number |
| `max_retries` | int in [0, 10] | 5 | HTTP retries of a call on the statuses 408, 429, 500, 502, 503, 504 and 529, or on a network error; any other 4xx fails at once |
| `backoff_seconds` | finite float > 0 | 2.0 | First retry delay; doubles on each retry, or follows `Retry-After` when longer |
| `max_invalid_retries` | int ≥ 0 | 2 | Times a response that is not valid JSON or fails its schema is asked for again |
| `timeout_seconds` | finite float > 0 | 180 | Per-request timeout |
| `budget_usd` | finite float > 0 | 3.0 | Most one run may spend, from the cost OpenRouter reports; the run stops before a call whose expected cost, the most expensive call of that model in this run or the cache, could pass it. The first call of a model with no cached call is not bounded. `make serve` holds one budget for the life of the process |
| `generator_prompt` | path of a file | `prompts/generator.md` | Generator system prompt (v2); its sha256 is written into every answer. v1 is `prompts/generator.v1.md` |
| `generator_prompt_sha256` | 64 lowercase hex digits, or null | `826c1cf9…` (full value in `configs/default.yaml`) | A call is refused when `generator_prompt` has another sha256 (schema default: null, no check) |
| `answer_schema_version` | 1 or 2 | 2 | Answer schema: 2 adds `partial` and `missing`. Must match the version line of `generator_prompt`: prompt v1 needs 1, prompt v2 needs 2 |
| `judge_prompt` | path of a file | `prompts/judge.md` | Judge system prompt (v2); its sha256 is written into every judgment |
| `judge_prompt_sha256` | 64 lowercase hex digits, or null | `4c7f312d…` (full value in `configs/default.yaml`) | A judge call is refused when `judge_prompt` has another sha256 (schema default: null) |
| `generators` | map of name to model | `gemini`, `qwen`, `mercury` | Each an exact `model` id (an id containing `latest` is refused), optional `provider_order` (pins OpenRouter providers, fallback disabled; each named once, none blank) and `temperature_supported` (false: no temperature is sent). Schema default: none |
| `generator_params` | object | `reasoning_effort: minimal`, `max_tokens: 4000`, `temperature: 0.0` | One rule for every generator |
| `judges` | map of name to model | `sonnet` (`anthropic/claude-sonnet-5`), `opus` (`anthropic/claude-opus-5.5`), both pinned to Anthropic | Judge candidates. Schema default: none |
| `judge` | str or null | `opus` | The candidate in use, chosen by `select-judge` against the human ratings ([ADR 0010](../adr/0010-judge.md)). Schema default: null, no judge |
| `judge_params` | object | `effort: medium`, `max_tokens: 8000` | The same for both candidates; no temperature is sent |
| `matrix` | list of `{name, generator, retriever}` | four configurations | What `answer-eval` runs; each `name` non-empty and used once, `generator` a key of `generators`. Schema default: empty |
| `primary_pair` | empty, or two different `matrix` names | `[gemini@venue_split, gemini@hybrid_rerank]` | Pre-registered primary comparison (methodology §9). Schema default: empty |
| `primary_metrics` | list of `faithfulness`, `venue_attribution_error`, `coverage`, `behaviour_correct`, each once | `[venue_attribution_error, coverage]` | Its metrics. Schema default: empty |
| `primary_scope` | `comparison` or `all` | `comparison` | Its questions |
| `primary_prompt_sha256` | 64 lowercase hex digits, or null | `519bd2c2…` (prompt v1; full value in `configs/default.yaml`) | The generator prompt the primary comparison was registered for; under another prompt the pair is secondary, Holm-adjusted within its own metrics (methodology §9a; schema default: null, every prompt) |
| `calibration_size` | int > 0 | 40 | Answers in the judge-calibration sample |
| `calibration_seed` | int | 20260924 | Its seed |
| `calibration_exclude` | list of question ids | the five trial questions | Never drawn into the calibration sample; `rate-sample` refuses an id the run did not answer. Schema default: empty |
| `calibration_thresholds` | map of `supported`, `venue_correct`, `behaviour_correct` or `coverage` to a float in [−1, 1] | `supported`, `venue_correct`, `behaviour_correct` 0.61 (Cohen's κ); `coverage` 0.70 (Spearman) | Agreement with the human a judge dimension needs to enter a conclusion (methodology §9) |
| `calibration_run` | path or null | `results/20260925T175128Z_answers` | Answer run whose `calibration.json` says which judge dimensions passed. `report.md` and `paired.csv` of a run mark the metrics of the others uncalibrated (schema default: null, every metric uncalibrated). The committed judged runs predate the mark. A run without `calibration.json` is refused when a report reads it |
| `calibration_exclude_behaviour` | list of answer ids | `qwen@venue_split:cv-0138` | Rated answers left out of `behaviour_correct` against the human; the reason is in methodology §9. `select-judge` refuses an id the run does not have. Schema default: empty |

## Validation

Loading a config fails, with the reason, when:

| Condition | Why it is refused |
|---|---|
| `overlap_tokens` ≥ `max_tokens` | Chunking never advances |
| `min_tokens` > `max_tokens` | No chunk can satisfy both |
| `reranker.top_n` < `retrieval.k`, `max(eval.k_values)`, `eval.pool_depth_per_method` or `generation.top_k` | Reranking returns fewer results than asked for |
| `retrieval.bm25_candidates` or `dense_candidates` < `retrieval.k` or `max(eval.k_values)` | A hybrid list would come back shorter than asked for |
| `store.hnsw_ef_construction` < 2 × `store.hnsw_m`, or an HNSW value outside pgvector's range | pgvector refuses to build or search the index |
| `eval.primary_comparison` with a metric or scope the report does not compute | Every comparison would silently become secondary |
| `pool_depth_per_method` ≤ `max(k_values)` | Only ranks the metric already scores are judged |
| `pool_display` < `pool_depth_per_method` | Fetched candidates would be hidden from the annotator |
| `k_values` empty, not positive, without 1 or with a cut-off twice; `eval.methods` empty or with a method twice | No metric to compute, no off_venue@1 for the top-1 off-venue comparison, or a result counted twice |
| `eval.paired_comparisons` pairing a method with itself, or naming a pair twice | An empty comparison, or one Holm counts twice |
| `store.table` not ASCII letters, digits and `_` | It is spliced into DDL |
| a number key given a YAML boolean | YAML reads `yes`, `on` and `true` as true, which a number field would take as 1 |
| a `generation` model id containing `latest` | An alias can move under a running study |
| `generation.judge` not one of `generation.judges` | The run would name a judge that is not configured |
| `generation.matrix` naming an unknown generator, or `primary_pair` an unknown configuration | The run or the pre-registration would refer to nothing |
| `generation.matrix` with an empty or repeated name, or `primary_pair` not two different configurations | Two configurations would write the same answer ids, or the primary comparison would match nothing |
| `generation.primary_metrics` or `primary_scope` outside the metrics and scopes the comparisons compute | The pre-registered comparison would silently be secondary |
| `generation.calibration_thresholds` naming an unknown dimension, or a value outside [−1, 1] | A dimension no judge is scored on, or a threshold no agreement can meet |
| a prompt sha256 that is not 64 lowercase hex digits | A call would be refused for a malformed pin rather than a changed prompt |
| a non-finite float (`.inf`, `.nan`) where a bound is stated, a blank model name or `user_agent`, an empty `revision` | A timeout that never fires, a budget that never stops a run, or a request the server refuses |

## Question files

A question file (`eval.questions`, `--questions`) holds one JSON object per line. `id`, `question`, `tier`, `venues` and `qtype` are required; every field a question has must have its type: strings, `venues`, `gold_chunk_ids` and `source_chunk_ids` lists of strings, `is_answerable` a boolean, `annotation_round` an integer, and `tier`, `qtype`, `category`, `difficulty`, `expected_behavior` and the values of `venue_coverage` one of their listed values. A field a question does not have is ignored, since a dataset record carries more. An id appears once. A file that is missing, a directory, a line that is not a JSON object, a field of the wrong type and a repeated id are refused with the file and the line.

## Other files

| File | Meaning |
|---|---|
| `.env.example` | Template for the dotenv file: `OPENROUTER_API_KEY` and, commented out, `CROSSVENUE_PG_DSN` |
