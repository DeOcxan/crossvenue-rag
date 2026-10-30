# Design

How the pipeline is built and why each stage is shaped the way it is. The decisions behind it, with the alternatives considered, are recorded in [`docs/adr/`](../adr/).

```
data/sources.yaml ──fetch──> data/raw/ + data/manifest.csv
                                   │
                             normalize      strip Docusaurus/MDX/GitBook markup;
                                   │        fenced code kept but for blank-line runs (ADR 0012)
                              chunking      heading-aware, token-budgeted,
                                   │        code blocks kept intact
                              metadata      environment / api_version / product /
                                   │        doc_type, by rules in sources.yaml
                                dedup       one chunk per identical text per venue,
                                   │        every occurrence kept in `sources`
                                   ▼
                      data/processed/chunks.jsonl        index_version stamps
                          │                    │         index, rows, results, labels
                          │                    │
                    BM25 (bm25s)      bge-base-en-v1.5 ──> pgvector
                          │                    │
                          └────── RRF ─────────┘
                                   │
                        bge-reranker-base (top-N)
```

## No framework

There is no LangChain or LlamaIndex here. The project's output is a claim about where retrieval breaks and why, which requires every stage (normalization, chunking, fusion, reranking, scoring) to be individually legible and individually measurable. A framework that wraps the chunker in a `TextSplitter` and fusion in an `EnsembleRetriever` makes the parts under investigation unobservable.

The cost is about 42,000 lines under `src/`, counting every line of its `.py` files, blank and comment lines included, as `find src -name '*.py' | xargs wc -l` gives it; a framework would have supplied part of them. The size of one module, which [code-style rule 2](../code-style.md) bounds at about 400 lines of code, counts its lines of code only: non-blank, non-comment lines, docstrings excluded. The benefit is that every number in [findings](../results/findings.md) can be traced to a specific function ([ADR 0001](../adr/0001-no-framework.md)).

## Module layout

Ten subpackages, one per domain, and a package root holding only what at least two layers need: `config`, `config_section`, `config_pipeline`, `config_eval`, `config_generation`, `config_loading`, `models`, `errors`, `stats`, `index_version`, `clock`, `jsonl`, `metric_direction`, `formatting`, `listing`, `hashes`, `run_files`, `retries`, `yaml_files`, `confined_paths` and `rich_text`. The import graph runs one way (corpus, index, retrieval, then evaluation or dataset, then generation, then labeling or reporting, then cli or service). `lint-imports` enforces it in `make lint` and CI with four contracts, all in `pyproject.toml`: a layers contract, a contract that the root modules import no subpackage, a no-cycles contract inside every subpackage and inside `reporting.claims` and `reporting.compute`, and a contract that `reporting.compute` never imports `reporting.claims`. `evaluation` and `dataset` are independent siblings ([ADR 0008](../adr/0008-package-structure.md)). Classes exist only where there is state (`Retriever`, `ChunkStore`, `BM25Index`, `Embedder`, `Reranker`, `Chunker`, `MetadataRules`, `VenueDetector`, `Evaluator`, `LLM`, `DiskCache`, `Sources`) or where a record has a known shape (pydantic models and dataclasses such as `Chunk`, `RunIndex` and the `CsvRow` rows of every saved CSV table); everything else is functions.

```
crossvenue_rag/
  config.py            the whole config schema: its sections, the checks across them and its defaults
  config_section.py    the base of every section: an unknown key or a YAML boolean is refused
  config_pipeline.py   the sections of the corpus and retrieval steps, fetcher to vector store
  config_eval.py       the section of the retrieval evaluation and its comparison scopes
  config_generation.py the sections of answer generation and judging
  config_loading.py    reading the config file, `--set` overrides and CROSSVENUE_PG_DSN
  models.py            data passed between stages: Chunk, ChunkSource, Result, Question, Timings
  errors.py            the expected failures: UserError for a fixable problem, ExternalFailure for a failure outside the program
  stats.py             mean, mean of the present values, nearest-rank percentile, Cohen's kappa
  index_version.py     the hash naming an index; needed by ingest, index, retrieval and label gates
  clock.py             record timestamps and run-directory names
  jsonl.py             reading, writing and appending JSON Lines files
  metric_direction.py  which way every retrieval and answer metric improves
  formatting.py        text a person reads for a value: metrics, differences, p-values, ms, USD, counts, model revisions
  listing.py           lists of items written as prose: `a, b and c`, `a, b, c, d, e, ... 3 more`
  hashes.py            the sha256 of a text: chunk ids, the index version, deduplication, the model cache
  run_files.py         the files every run kind shares: index.json, config.yaml, the
                       results/runs.json registry, typed CSV rows and tables
  retries.py           which HTTP statuses are retried and the backoff between attempts
  yaml_files.py        reading a YAML file whose top level is a mapping
  confined_paths.py    checks on a path or name from config, the command line or a download
  rich_text.py         rich table rows built from values, so brackets in a value are never markup
  corpus/              documentation in, chunks out
    sources.py           the registry and the metadata-rule schema (data/sources.yaml)
    manifest.py          data/manifest.csv, and the fingerprint datasets record
    fetch.py             acquisition: which sources, and the comparison with the manifest
    git_source.py        git sources: blobless shallow sparse checkout, include globs
    gitbook_source.py    GitBook sources: the pages llms.txt lists, with retries
    normalize.py         per-stack markup stripping; code blocks kept as written, blank-line runs aside
    tokens.py            token counting with the embedding model's tokenizer
    chunking.py          heading-aware, token-budgeted chunker
    doc_metadata.py      evaluating the metadata rules
    dedup.py             exact deduplication and the primary-source rule
    ingest.py            raw pages -> chunks.jsonl + corpus_stats.json
    artifacts.py         chunks.jsonl and corpus_stats.json: the statistics models, their
                         reader and writer; count-term
  index/               chunks into searchable indexes
    embeddings.py        the bi-encoder
    store.py             pgvector table, schema and search
    bm25_index.py        the sparse index
    migrate.py           carrying chunk-id pointers across an id-changing rebuild
    build.py             the index command: the version check, then both indexes
    status.py            what `crossvenue check` reports of the sources and the built index
  retrieval/           one retrieve() over six methods
    methods.py           the method names: every method, the labeling pool, the oracle
    retriever.py         Retriever: bm25, dense, hybrid, hybrid_rerank, venue_split(_oracle)
    filters.py           venue and metadata filters, resolved once for both indexes
    fusion.py            reciprocal rank fusion
    rerank.py            the cross-encoder
    venue_detect.py      which venues a question names
    protocols.py         what callers need of a retriever, one protocol per contract
  evaluation/          running methods over a question set and reporting it
    metrics.py           recall, precision, nDCG, nDCG', MRR, venue_coverage, off_venue and
                         via_ccxt (methodology 9b)
    evaluator.py         the ablation loop, per-method / per-slice summaries, off_venue@k and
                         via_ccxt@k for every question naming a venue
    paired.py            exact McNemar and paired bootstrap; better/worse in the metric's direction;
                         PairedEstimate, the fields every paired row starts with
    bench.py             bench-rerank: one cross-encoder call against one per venue; the
                         raw.csv and summary.csv rows
    machine.py           the machine a timing runs on: load, competing processes, machine.json
    power.py             simulated power of the Holm-adjusted randomization test; power.json
    mcnemar_power.py     analytic power of the exact McNemar test (methodology 7)
    t_test_size.py       the paired t-test sample size the methodology compares it with
    comparisons.py       which pairs and scopes report.md compares; off_venue_comparisons;
                         PairedRow, a row of paired.csv, and its reader
    results.py           results/<ts>/: CSVs (off_venue.csv among them), summary.json, config
                         and index record; the per_query, summary and by_<dimension> rows
                         and their readers
    report.py            results/<ts>/report.md
    labels.py            how methods and missing values are shown, and the Markdown tables of a run report
    jobs.py              eval as the command starts it: the checks, warm-up, run and results
  dataset/             the question set as data
    records.py           dataset schema, versioned files, newest candidate pool
    questions.py         the working question file, checked against the index of its labels
    journal.py           the judgment journal as typed records: rounds, adjudication, relabel
    round3_gold.py       the gold set built from round 3 alone (round3-gold)
    remap.py             remap-ids for a journal or a dataset
    validate.py          validate-evals
    selection.py         choosing the set: decisions and composition checks
    finalize.py          select --finalize: the question file from the accepted candidates
    sampling.py          the stratified round-2 subsample (round-sample)
    added_chunks.py      the --add-chunks file of label --relabel and --round 3
    agreement.py         agreement between annotation rounds (kappa from stats.py)
  labeling/            interactive judgment at the keyboard
    screen.py            what a candidate row shows: compact view, highlighting
    pool.py              which candidates are judged
    judge.py             the choice prompt and the pre-save checks (coverage, versions)
    session.py           a labeling round
    modes.py             the label command's modes, chosen from its options
    review.py            corrections to round 1 (--review)
    candidates.py        never-judged chunks from a run; the pool-construction run
    augment.py           judging those chunks (--augment)
    coverage.py          venue_coverage values that follow from recorded data
    adjudicate.py        deciding the pairs rounds 1 and 2 disagree on (--adjudicate)
    relabel.py           judging named questions again under the amended criterion (--relabel)
    round3.py            the blind relabel of the whole set (--round 3)
    select_session.py    the selection loop
    rate.py              grading answers by hand, the judge's grade hidden
  generation/          answers from retrieved chunks, and their judging
    openrouter.py        OpenRouter's protocol: request body, retried POST, response models
    api_key.py           the OpenRouter key, from the environment or the dotenv file
    llm.py               the model client: validation, cache, budget, call records
    cache.py             responses cached on disk by request hash
    context.py           the chunk block of the prompt; citations to occurrences
    schemas.py           answer and judgment JSON schemas and their parsing
    records.py           answer, judgment and rating records: one reader and writer per file
    answer.py            one answer, one judgment
    metrics.py           faithfulness, venue attribution, coverage, behavior
    grader_agreement.py  agreement between two graders per dimension, with intervals
    calibration_sample.py  the calibration sample: stratified answers, the raters' file
    judge_sample.py      a judge candidate's judgments of the sample, set beside the human ratings
    calibration.py       judge selection from the calibration agreements
    calibration_report.py  calibration.json and calibration.md in the answer run
    calibration_file.py  calibration.json: its model, reader and writer
    answering.py         the configuration matrix: one record per configuration and question
    judging.py           saved answers judged again, the generation kept
    answer_rows.py       per-answer and per-configuration rows
    answer_pairs.py      paired comparisons of configurations, Holm families
    answer_report.py     an answer run's report.md
    answer_run.py        writing results/<ts>_answers/
    estimate.py          the cost of an answer run, from the cache
    jobs.py              answer-eval, judge-run, judge-sample and compare-prompts as the
                         commands start them
    prompt_compare.py    prompt v1 against v2 (H-v2) and the side-effect checks
    prompt_run.py        writing results/<ts>_prompts/ and its report.md
    attribution.py       venue attribution errors by kind: cross-exchange, through a wrapper
    ask.py               one ad-hoc question answered, for ask and POST /answer
    runs.py              an answer run's index_version, pinned prompts, uncalibrated metrics
  reporting/           documents against their sources
    check.py             Doc: templates, tables and the audit of uncovered numbers
    text_spans.py        offsets in the whitespace-collapsed text, and sets of spans
    patterns.py          typography, numbers, number words and references in a document
    outcomes.py          what a check reports: problems and the numbers it lists
    sources.py           loaders of the saved files
    run_ids.py           the runs and datasets the documents report
    compute/             the values the claims recompute
    claims/              what each document claims
    runs.py              committed runs: file hashes and index_version
    rederive.py          a committed retrieval run's tables re-derived from per_query.csv
    reproduce.py         make reproduce: run the claims, collect problems
  service/             the HTTP interface
    app.py               FastAPI: POST /answer, GET /health, each with its response model
  cli/                 argument parsing and a call into the library
    __init__.py          registers every command, in the order --help lists them
    __main__.py          python -m crossvenue_rag.cli
    app.py               the Typer app, shared options and `say`, which prints a value with markup off
    corpus.py, index.py, evaluate.py, dataset.py, labeling.py, generation.py, reporting.py   one module per command group
    render.py            rich tables
    render_answers.py    rich output of the answer commands: an answer, a cost estimate, an agreement
    errors.py            the friendly-error boundary
```

Why this split, one sentence each. `corpus` is everything true of the documents before any index exists. `index` is everything that depends on a built index. `retrieval` is the query-time path. `evaluation` measures retrieval and holds no code that chooses or judges questions. `dataset` is the question set as files and rules, with no keyboard. `generation` answers from retrieved chunks and grades the answers, using dataset records and evaluation's paired statistics. `labeling` is the only interactive code besides the CLI. `reporting` checks the documents' numbers against their sources and records the committed runs' file hashes. `service` exposes the same answer function as `crossvenue ask` over HTTP. `cli` is parsing and presentation. The root holds only what at least two layers need and nothing that imports upward.

## Tests

`tests/` mirrors the package: one folder per subpackage and one test file per module, as [code-style rule 27](../code-style.md) requires (`tests/labeling/test_judge.py` tests `labeling/judge.py`). The root modules have root test files (`test_config.py`, `test_config_section.py`, `test_config_pipeline.py`, `test_config_eval.py`, `test_config_generation.py`, `test_config_loading.py`, `test_models.py`, `test_stats.py`, `test_index_version.py`, `test_errors.py`, `test_clock.py`, `test_jsonl.py`, `test_metric_direction.py`, `test_formatting.py`, `test_listing.py`, `test_hashes.py`, `test_run_files.py`, `test_retries.py`, `test_yaml_files.py`, `test_confined_paths.py`, `test_rich_text.py`). Six files test no single module and are listed as suite-level in `test_repo_hygiene.py`:

- `test_pipeline_smoke.py`: one end-to-end run over a four-document fixture corpus
- `reporting/test_mutations.py`: the mutation test of the documents' checks
- `reporting/test_claims_adr.py`: the dispatch of each decision record to its claims
- `test_input_domains.py`: the files, environment variables and library preconditions, each fed the values outside its domain
- `test_repo_hygiene.py`: the checks of the published files and of the test layout that the Checks tables of [`style.md`](../style.md#checks) and [`code-style.md`](../code-style.md#checks) name
- `test_code_hygiene.py`: the checks of the Python code that read its syntax tree (`tests/python_code.py`)

Every other module has a test file of its name. The exceptions are the modules under `reporting/claims/`, which `make reproduce` and the mutation test check against the documents, and `reporting/run_ids.py` and `retrieval/protocols.py`, which hold only constants or Protocol classes.

Fixtures shared across folders (the in-memory retriever, the fixture corpus, fake embedder and reranker) are in `tests/conftest.py`, and ones used inside a single folder are in that folder's `conftest.py`. Builders a folder's tests import are in its `factories.py` (`tests/generation/factories.py`, with the offline setup of the generation tests in `tests/generation/offline.py`). No test loads model weights or needs a database.

## Normalization

The corpus comes from four publishing stacks, and each adds non-content syntax: Docusaurus MDX imports, `<Tabs>` and `:::note` admonitions (Bybit); JSX components (Uniswap, dYdX); GitBook `{% hint %}` blocks and HTML entities (Hyperliquid, PancakeSwap); docsify `{docsify-ignore}` directives (CCXT).

It is stripped before anything is indexed. Markup tokens left in are expected to match across every venue at once, the failure this project measures; retrieval on an index built without normalization, against the default index, would test that.

Fenced code blocks are kept as written, except that normalization collapses runs of blank lines inside 19 of them, a defect deferred in [ADR 0012](../adr/0012-deferred-corpus-changes.md). Every other transformation runs only on the prose between fences. Inline code spans are masked too, because `` `POST /v5/order/create?symbol=<SYMBOL>` `` looks exactly like an HTML tag to a regex. Stripped as a tag, `<SYMBOL>` disappears and the chunk documents a request without its required parameter.

## Chunking

API documentation is a tree of headings whose leaves are parameter tables and code samples, so a fixed-window splitter cuts a parameter table away from the endpoint it belongs to. The chunker therefore:

- breaks at headings, but only once the running chunk has at least `min_tokens`; otherwise `## Parameters` becomes a two-word chunk;
- treats a fenced code block as atomic, letting it exceed `max_tokens` rather than splitting a signature, up to `max_code_block_tokens`;
- carries the heading breadcrumb into the chunk text, so the third chunk of a long section still says which endpoint it describes;
- counts tokens with the embedding model's own tokenizer, so a "450-token" chunk is 450 tokens to the model that encodes it.

Token counting is injected (`TokenCounter`): production uses the HF tokenizer at the revision `embedding.revision` pins, and refuses to run without it. Tests inject a word count, so the test suite downloads no tokenizer and needs no network.

The budget is measured on the rendered chunk, breadcrumb included: tokenizers are not additive, so summing separately-counted blocks undershoots the joined text by the separators. Over the 450-token budget are 18 prose chunks of 11,934, plus 191 code blocks kept whole on purpose (`data/processed/chunks.jsonl`, `token_count > 450`, split on a code fence). Parameters and the tests that pin the budget: [ADR 0002](../adr/0002-structural-chunking.md).

`chunk_id` is `sha256(venue + "\x1f" + text)[:16]`, a content hash, so byte-identical text within a venue has one id wherever it occurs, which is what deduplication collapses on. The venue is in the hash so that identical text on two venues can never share an id. It is stable across re-ingestion of an unchanged corpus, so gold labels stay valid.

## Exact deduplication

Byte-identical chunks within a venue are stored once. Identical text is the same string in every occurrence, so it holds no mainnet/testnet trap; the trap is in the chunks whose text differs, and those are untouched. Stored more than once, a text takes top-k slots and labeling-pool places that another passage could hold ([ADR 0003](../adr/0003-exact-deduplication.md); the duplicate groups are in [findings](../results/findings.md#108-of-chunk-occurrences-are-byte-identical)).

What is deliberately not collapsed:

- Near-duplicates. Only exact equality of the chunk text (normalized markdown with its heading breadcrumb) merges. Binance's REST and WebSocket "Timing security" sections are near-identical (difflib ratio 0.985) and stay two chunks.
- Identical text on two venues: which venue says something is the measurement. The corpus has none (0 groups); `make stats` reports the count on every ingest.

`corpus.dedup.deduplicate` runs after the whole corpus is chunked. The 12,783 chunk occurrences are stored as 11,934 chunks: 536 groups merged, 433 of them Binance mainnet/testnet mirrors (`make stats`, "exact deduplication").

## Provenance and the primary source

Every occurrence of a merged text is kept in `Chunk.sources` (`venue`, `doc_path`, `heading_path`, `source_url`, `position`, and its own structural metadata). That a passage appears on both the mainnet and the testnet page is itself evidence for an answer, so the merge keeps it.

The chunk's top-level citation fields describe one of them, the primary, chosen by `corpus.dedup.primary_key`: non-testnet before testnet, then `doc_path` lexicographically, then position. It is deterministic and makes the 433 Binance mirrors cite the production page. The lexicographic tie-break is blind to meaning: Bybit's shared IP-limit paragraph cites `docs/p2p/rate-limit.mdx` before `docs/v5/rate-limit/rate-limit.mdx`. That is why the other occurrences are shown to the annotator.

Known limitation: the primary can be the wrong page for the question. In groups whose occurrences differ in product rather than environment, the lexicographic tie-break picks whichever path sorts first. `cv-0004 "bybit sends 403 access too frequent,"` and `cv-0094 "bybit p2p says 600 per 5"`, a candidate dropped at selection (in `v0.1.6-candidates`, not in `v0.2.1-gold`), both are answered by that Bybit IP-limit paragraph, and its primary is the P2P page although `cv-0004` asks about the trading API. The rule is left as it is: it is deterministic, and the primary only decides what the chunk's top-level fields say; a citation resolves to the matching occurrence in `sources`. An answer cites the occurrence in `sources` that matches the question (venue, product, version, environment), falling back to the primary only when no occurrence is more specific (`generation.context.resolve_citation`). Retrieval and scoring are unaffected, since both work on chunk ids.

## Structural metadata

`environment` (mainnet/testnet), `api_version` (v5, v4, infinity, …), `product` (spot/perp) and `doc_type` (reference/guide/faq/changelog) are read off each document's path and headings by rules in `data/sources.yaml`, which hold no code. They are facts about each vendor's layout and change with the vendor. Per field, the venue's rules are tried in order, then a small global block (environment from headings such as "Testnet Deployments"), and the first rule that matches sets the field. `value: null` is a stop rule, used for Binance's `demo-mode/`, which is a third environment rather than either of the two. No match leaves the field null: "the layout does not say" is recorded as such, never guessed.

No retrieval method reads these fields. They are for filters and for analyzing version confusion. Coverage, as the share of chunks whose primary occurrence got a value (`make stats`): `doc_type` 88.5%, `api_version` 44.8%, `product` 38.4%, `environment` 10.9%. `data/sources.yaml` sets an environment only from a Binance path, a dYdX faucet page or a heading that names a network. Null environments per venue: bybit 3,145, pancakeswap 2,012, uniswap 1,706, ccxt 1,429, dydx 1,034, hyperliquid 790, bingx 507 and binance 13. Binance's 13 are `demo-mode/`, a third environment; the other venues' pages match none of those rules.

## `index_version`

One 16-hex hash names the index a label or result belongs to. A gold label is a list of chunk ids, and an id from one index scored against another silently points at different text. It hashes the manifest, the chunking config, the embedding model (name, dimension, document prefix, normalization) and the chunk schema version. `crossvenue index` writes it to `data/index/index_meta.json` and onto every Postgres row; `eval` writes it into `results/<timestamp>/index.json` and `report.md`; `label` stamps it on every judgment in the judgment journal. Every published result and label uses `index_version` `2825a7160c4d09bf` (`data/index/index_meta.json`), except `results/20260924T200525Z`, which was made before exact deduplication and records no `index_version`. `index_version` identifies the corpus, the chunking and the model; the HNSW graph and the vectors are outside it. A rebuild under the same `index_version` encodes every chunk again, on an accelerator when one is present, and pgvector assigns graph levels at random. One such rebuild changed 35 of the 570 top-10 lists of the dense-dependent methods on `v0.2.1-gold` while BM25 stayed identical. The compute device changes them too: on the same index, the CPU run `results/20261010T222553Z` and the MPS run `results/20260925T012314Z` differ in 25 of 684 retrieval lists, none of them BM25's. On the 50 comparison questions only nDCG and MRR differ, in the fourth decimal, and dense's unjudged@10, 0.086 against 0.078. In the other categories recall and precision differ too, most in version_sensitive hybrid recall@10, 0.717 against 0.750 (the device note in `results/runs.json`). A rerun of `eval` therefore reproduces BM25 exactly and the dense-dependent methods up to what a rebuild or the device changes. `eval` and `label` refuse to run if the index on disk, Postgres and the config disagree, or if any existing judgment carries another version. The message names the fix: rebuild, or carry ids across with `crossvenue id-map` / `remap-ids` when only ids changed, or re-judge ([ADR 0004](../adr/0004-index-version.md)).

## Retrieval

Six methods behind one signature:

```python
retrieve(query, method, k, filters, timings, venues_hint) -> list[Result]
#         method ∈ {bm25, dense, hybrid, hybrid_rerank,
#                   venue_split, venue_split_oracle}
#         filters: {"venue": [...]} | {"venue_type": "dex"}
#                  | {"environment": "testnet", "api_version": ["v5"], ...}
```

One signature rather than six code paths, so an ablation is a loop over strings.

- BM25 (`bm25s`, English stopwords, Snowball stemming). It matches the literal tokens that distinguish venues (`/v5/order/create`, `X-MBX-APIKEY`, `1200`). On the 50 comparison questions of `v0.2.1-gold` its recall@10 is 0.465, against 0.353 for dense and 0.620 for hybrid_rerank (`results/20260925T012314Z`, [findings](../results/findings.md#splitting-retrieval-by-venue-raises-comparison-coverage-from-070-to-098)), scored against comparison gold augmented after results were seen ([methodology D4](../../evals/methodology.md#deviations-from-protocol)).
- Dense (`BAAI/bge-base-en-v1.5` vectors in pgvector, cosine, HNSW). The bge query prefix is applied, as the model is trained with it.
- Hybrid: reciprocal rank fusion.
- hybrid_rerank: `BAAI/bge-reranker-base` over the fused top-N.
- venue_split: reads the venues a question names (regex aliases in `data/sources.yaml`, `venue_detect`) and fuses BM25 and dense candidates once per venue under a venue filter. It scores every venue's candidates in one cross-encoder call and interleaves the per-venue lists round-robin; with fewer than two venues named it is hybrid_rerank. venue_split_oracle runs the same with the question's labeled `venues` (`venues_hint`) in place of the detected ones, which a deployed system does not have. It bounds what better venue detection can add to venue coverage; it is not an upper bound on recall, precision or nDCG. It is marked "(oracle)" in every report ([ADR 0007](../adr/0007-venue-split.md)). `venue_split` is one of `POOL_METHODS`, the methods labeling pools are built from; the oracle is not, because it reads the labels. `venue_coverage@k` needs no gold labels.

`unjudged@k`, in each retrieval report that has the column, measures how much of each method's top k no judgment covers, and `crossvenue label --augment` judges those chunks ([ADR 0005](../adr/0005-labeling-pool.md)).

Fusion uses ranks rather than normalized scores. BM25 scores are unbounded and corpus-dependent; cosine similarities are in [−1, 1]. Normalizing one onto the other assumes a relation between their distributions; ranks assume none. The constant `k` (60, from Cormack et al. 2009, [doi:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114); full citation in the [methodology references](../../evals/methodology.md#references)) damps the top of each list so one method cannot dominate on its first hit alone.

The venue filter restricts a search to the venues it names; venue_split and the oracle apply it once per venue. `eval --venue-filter` restricts every query to the question's labeled `venues`, so it reads the labels, as the oracle does. No committed run used it: every committed retrieval run has chunks from unnamed venues in the top k of its unfiltered methods. The harness therefore reports no filtered-against-unfiltered gap; per question, `per_query.csv` records the venues of each method's top k (`retrieved_venues`) and `venue_coverage@k`. `off_venue@k` and `via_ccxt@k` give the share of the top k from venues the question does not name ([findings](../results/findings.md#unnamed-venues-fill-0036-of-hybrid_reranks-comparison-top-10-and-0214-of-bm25s-exploratory)). The measurement of the gap, and of which unnamed venue fills the top k, is proposed in [directions](directions.md#venue-filter-as-a-control). The filter is resolved to an explicit venue allow-list before it reaches either index, so BM25 and Postgres apply byte-identical filters. An unknown venue raises rather than silently matching nothing, and so does a venue and venue type that no venue satisfies together.

Metadata filters (`environment`, `api_version`, `product`, `doc_type`) resolve, once, to a chunk-id allow-list handed to both indexes. A merged chunk matches if any of its occurrences does: text shared by the mainnet and testnet pages is testnet documentation too. A filtered BM25 search ranks the whole corpus before dropping disallowed hits. A selective filter therefore returns fewer than k only when fewer than k allowed chunks share a query token, since a chunk that shares none is no hit.

Postgres holds both the vectors and the chunk metadata, so a filter is a `WHERE` clause in the same query as the vector search. The HNSW graph is built with random level assignment, so builds of the same index can return different approximate top k ([`index_version`](#index_version)). The HNSW index filters after its scan, which visits about `store.hnsw_ef_search` rows, so a filter on it could return fewer than k. A single-venue filter, the only one venue_split and the oracle use, is left to the Postgres planner. On the built index `EXPLAIN` shows it served from the venue btree index for each of the eight venues at k of 20 and 50, an exact scan; the planner's choice follows the table statistics. Any other filter, and any k above `hnsw_ef_search`, is an exact scan of the allowed rows. Every row stores the embedding model's fingerprint and the `index_version`, so swapping the model without re-indexing is detectable, and `crossvenue index` drops and recreates the table so a schema change cannot leave half-migrated rows.

## Configuration and ablation

Every setting the pipeline reads is in [`configs/default.yaml`](../../configs/default.yaml), validated by pydantic, except the schema defaults the [config reference](../reference/config.md) lists and the code constants it names (`POOL_METHODS`, [ADR 0005](../adr/0005-labeling-pool.md)). An ablation is a different config or a field override, never a code change:

```bash
crossvenue ingest --set chunking.max_tokens=256 --set chunking.overlap_tokens=32
crossvenue eval   --set retrieval.rrf_k=10 --set reranker.top_n=100
```

Validation catches the combinations that would silently produce a meaningless run. Among the checks of sizes and depths: overlap ≥ chunk size is refused (chunking never advances), and so is rerank depth below `retrieval.k` or the largest `eval.k_values` (fewer results than asked for). A pool depth no deeper than max k is refused, since the annotator would judge only ranks the metric already scores. So is a primary comparison whose pair, metric or scope the report does not compute, which would make every row secondary. A table name that is not ASCII letters, digits and `_` is refused because it is spliced into DDL.

Secrets never appear in YAML. `OPENROUTER_API_KEY` is the key for every model call, also read from the repository's dotenv file. `CROSSVENUE_PG_DSN` overrides `store.dsn`, so a DSN can carry a password without it reaching a file. `.env.example` lists both.

Every evaluation run writes `results/<timestamp>/` containing `per_query.csv`, `summary.csv`, the per-slice breakdowns, `report.md` and a copy of the resolved config, so every number traces to the settings that produced it.

## Measuring latency

Per-step timings (`encode`, `search`, `fuse`, `rerank`) are recorded by passing a `Timings` object into `retrieve`; without one, `retrieve` times into a `Timings` of its own and discards it. The retriever keeps no timing state between calls. The per-query latency the reports give is the sum of the four measured steps (`total_ms` in the CSV files), which is shorter than the wall time of the call. Filter resolution and venue detection fall outside every step; building the result objects of a fused method is inside `fuse`. venue_split encodes the query once per named venue, so its `encode` step on a split question is the sum of those encodings.

Two things the harness does to keep those numbers free of setup costs:

- Models are loaded in `Retriever.warmup()` before anything is timed. Otherwise the weight loading is timed as part of the first query's `encode` step.
- The ablation loops methods on the outside, so the cross-encoder's accelerator work cannot run into the next question's `encode` and `search` timings of another method.

Measured numbers are in [findings](../results/findings.md#the-cross-encoder-takes-2505-of-hybrid_reranks-2531-ms-at-p50).

## Answer generation

`generation/` is above dataset and evaluation, and below labeling and reporting, in the layering ([ADR 0008](../adr/0008-package-structure.md)). It turns a retriever's top-k chunks into a cited answer, grades the answer with a judge, and reports a matrix of configurations with the same paired statistics retrieval uses. The choices behind it are ADRs [0009](../adr/0009-generators.md) (generators), [0010](../adr/0010-judge.md) (judge candidates and their calibration) and [0011](../adr/0011-preregistration.md) (pre-registered comparisons).

Every model call goes through one thin client for OpenRouter's OpenAI-compatible API (`generation.llm`, with the protocol in `generation.openrouter`), with no SDK (ADR 0001). It uses one key, `OPENROUTER_API_KEY`, read from the environment or the repository's dotenv file and sent only in the request header. The client:

- pins providers where the config says so, with fallback disabled and every parameter required, so a request is never served by an endpoint that drops the JSON schema or a parameter;
- retries the HTTP statuses 408, 429, 500, 502, 503, 504 and 529, and network errors, with exponential backoff, and sends the request again, at most twice, for a response that is not valid JSON or fails its schema;
- keeps a per-run budget (`generation.budget_usd`). Before each uncached call it adds the most expensive recorded call of that model, from this run or the cache, to what the run has spent. It stops the run rather than pass the budget. The first call of a model with no cached call has no known cost and is not bounded;
- produces a `CallRecord`: model id requested, model id and provider the response reports, input, reasoning and answer tokens, the cost OpenRouter reports, latency, HTTP retries, invalid responses and the parameters sent.

A response is cached on disk (`data/cache/llm/`) under a hash of the request and a replicate number once it passes validation. A re-run therefore costs nothing for every call that succeeded, and a judge's deliberate repeat is a separate entry. A call whose responses all fail validation writes no entry: its paid attempts are counted in the run's spend. A re-run calls the model again for it unless the cache holds a valid response to the same request from any run, since the key is the request body and the replicate number. Results name the cache key of every call, and `answer-eval --estimate` prices a run from the cache before any call.

The generator sees the question and the numbered chunks, each with its venue and the environment, API version and product of its occurrences (`generation.context`). The prompt (`prompts/generator.md`, its sha256 in every answer) requires every claim to cite chunks and name its venue, and forbids moving a fact from one venue to another. It allows four outcomes: a full answer, a partial answer that lists in `missing` what the chunks lack per venue, a refusal, and a question asking which venue is meant. Prompt v1 (`prompts/generator.v1.md`, the run §9 is registered on) allowed the same outcomes except the partial answer ([methodology §9a](../../evals/methodology.md#9a-generator-prompt-v2-exploratory)). A citation names a chunk; `resolve_citation` turns it into the occurrence whose environment, product and version match the question, falling back to the primary occurrence, as ADR 0003 decided for merged chunks.

The judge grades each claim (`supported`, `venue_correct`) and the answer (`coverage` of the gold facts per venue, `behaviour_correct`); the metrics follow from those verdicts (`generation.metrics`). Its agreement with a human is measured on a fixed, stratified sample graded with `crossvenue rate`, whose screen is built from a view that holds no judge field.

`crossvenue answer-eval` runs `generation.matrix`: each question's top-k is retrieved once per retrieval method and shared by every generator on it, so generators differ only in the model. A failed call is recorded with its error and the run continues; a budget stop ends the run and writes what it has.

## Interface

Two thin entry points over the same function, `generation.ask.answer_question`, which the CLI reaches through `generation.ask.ask`: the configured retriever (default venue_split) and generator (default gemini) answer one question. The result lists the answer, each claim with its venue, and each citation with its chunk's venue, its document path and the URL of the occurrence. It also gives the outcome under the rule every metric uses (full, partial, refused, asked), with the venues a partial answer lacks. A claim whose venue differs from its citations' is visible as such.

- `crossvenue ask "question"` prints it; `--retriever` and `--generator` choose the configuration.
- `service/app.py` is a FastAPI app with `POST /answer` (the same result as JSON) and `GET /health`, each declaring its response model (`AskResult`, `HealthResponse`), so the OpenAPI schema describes both bodies; `make serve` runs it with uvicorn on 127.0.0.1:8000. It has no UI and no HTML. The retriever and the client are built once at startup, and the OpenRouter key is read then; `create_app` takes them ready-made, so the tests run with fakes and no network. A request the service cannot answer as asked (an empty question, an unknown generator or retriever) is a 422. A model call that fails is a 502, and a missing key or a spent budget is a 503. The service is one run for the budget (`generation.budget_usd`): every request draws on it until a restart, and `GET /health` reports what is spent.

## Error handling

Every command is a pipeline step with prerequisites. Library code raises `errors.UserError` for a problem the user can fix and `errors.ExternalFailure` for a failure outside the program (a failed model call, an invalid model reply). `cli.errors.friendly_errors` turns a `UserError` (a missing prerequisite, an invalid config file named by path, a mistyped method or venue), an `ExternalFailure` and an unreachable database into its message, a hint and exit 1. It also prints the line naming the debug switch. Anything else keeps its traceback, a `ValueError` or a pydantic `ValidationError` included, because that one is a bug. `CROSSVENUE_DEBUG=1` re-raises everything.

## Checking the documents against their sources

`crossvenue reproduce`, which `make reproduce` runs, checks the numbers of the text, as this section defines them, in the documents `reporting/reproduce.py` lists against their sources; [reproduce](../how-to/reproduce.md#check-the-documents-numbers-against-their-sources) says how to run it.

It first checks the committed runs themselves. `results/runs.json` must list exactly the runs committed to the repository (the `!results/<run>/` lines of `.gitignore`), and every file's sha256 and every run's `index_version` must equal what it records. Every retrieval run's metric tables (`summary.csv`, `by_*.csv`, `paired.csv`) must equal what the evaluation code re-derives from the retrievals in the run's `per_query.csv`, cell for cell; latency is not re-derived, because `per_query.csv` rounds each timing. A run that records no question file, or whose question file has changed since it was made (`results/20260924T200525Z` read the working file), is not re-derived.

It runs no retrieval, so it compares against the committed runs exactly, whichever graph the local index holds. Then it checks the numbers of those documents against the files it reads. The committed files it reads include:

- the runs under `results/` and their registry, `results/runs.json`;
- the datasets under `evals/datasets/` and their journals;
- the working question file `evals/questions.jsonl`, with its journal and the human ratings under `evals/annotations/`;
- the lists under `evals/selection/` and the chunk-id map under `evals/migrations/`;
- `configs/default.yaml`, `data/manifest.csv`, `data/sources.yaml` and `docs/references.yaml`;
- the checked documents and `docs/explanation/directions.md`, whose own text some numbers count;
- the code, tests and scripts of the repository, `pyproject.toml` and `.gitignore`.

It also reads the chunks `make ingest` writes to `data/processed/chunks.jsonl` and the `index_version` `make index` records in `data/index/index_meta.json`, which are not committed. It runs `pytest --collect-only` for the test counts `README.md` states. Some numbers are recomputed from raw records: corpus counts, duplicate groups, venue detection, recall against both gold versions, refusal outcomes, attribution kinds, the paired bench statistics, the TF-IDF check, and the off-venue measures and their paired comparisons, re-scored from the retrieval run's `per_query.csv`. Other retrieval metrics and paired statistics are read from the aggregate files the run wrote, which the first step has re-derived for the retrieval runs. Other runs' outputs are read as saved, such as `comparison.csv`, `calibration.json`, `power.json` and the answer and bench runs' `summary.csv`. It makes no model call and needs no database.

Numbers inside code, link targets, identifiers (`recall@10`, `cv-0045`, `v0.2.1`), dates, references (`ADR 0007`, `§9a`, `rule 5`) and numbers joined to a word by a hyphen (`50-question`, `top-1`) are not numbers of the text. Numbers written as words (two to twelve, half, twice, tenfold), numbers in headings and numbers with a unit glued on (`12ms`, `10x`) are. "One" is left out, because it is also a pronoun, "each one's", and a definition, "one call per venue". The fee tiers are checked against the corpus with the `count-term` convention.

The kinds of value are these:

- compared: matched with the value its check recomputes from those files or takes from the code (a constant, a default, a command's source);
- given: matched like a compared value, so a different value is still a problem, and listed with its reason instead of counted;
- unsaved: its sentence is matched, its value is compared with nothing, and it is listed with its reason on every run;
- identifier: a run directory, `index_version` or hash prefix, compared with the one its check reads, and counted apart.

The total `make reproduce` prints counts compared values; a value is a hole of a checked sentence or a table cell that a check compares, whatever the cell holds. `crossvenue reproduce -v` prints the details: the values the checks list, with their kind and their template, table cell or reason.

Under `src/crossvenue_rag/reporting/`, `claims/` states what each document claims: `readme.py`, `design.py`, `findings/`, `agreement.py`, `methodology/`, `adr/`, `corpus_datasheet.py`, `datasheet.py` with `datasheet_versions.py` and `datasheet_limits.py`, `domain_research.py`, `cli_reference.py` and `config_reference.py`, with one module per decision record. `compute/` holds the values those claims recompute, as named, tested functions; its package docstring (`compute/__init__.py`) says how its modules divide them. The runs the documents report are named once, in `run_ids.py`. Editing a number in these documents means editing its run or its check, never the text alone. A claim is matched with every run of whitespace collapsed to one space, in the claim and in the document, so re-wrapping a paragraph changes no check; tables and lists are read line by line. The methodology's reference list must equal `docs/references.yaml` section by section: the same items in the same order, none missing and none extra. After the documents, `make reproduce` runs the full mutation test of these checks (`pytest -m slow tests/reporting/test_mutations.py`). In a copy of the documents it changes 500 sampled numbers, every run reference and every number of the reference list, one at a time, and fails if any change goes unnoticed. `make test` runs a fixed sample of 53 of these mutations, covering every kind of check. `make reproduce` is not part of CI: CI has no `data/processed/`.
