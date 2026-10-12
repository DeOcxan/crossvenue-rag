# Reproduce the documented numbers

How to check the numbers of the documents against their sources, and, in the sections after the first, how to rerun each measurement. The documents are `README.md`, [findings](../results/findings.md), the [agreement](../../evals/agreement.md), the methodology, the design document, the decision records, the three pages of `docs/reference/`, the [gold-set datasheet](../../evals/datasheet.md) and the [domain research](../../evals/domain-research.md). Every snippet runs from the repository root. The snippets read only saved files, except `make figures`, which writes the figures; the corpus build, which writes `data/`; the drift measurement, which rewrites `data/manifest.csv`; `eval`, `bench-rerank` and `power`, which each write another run; and the steps under [Judge the answers](#judge-the-answers), which write runs and ratings and make paid model calls. The runs the documents cite are committed under `results/`.

## Check the documents' numbers against their sources

1. Run the check: `make reproduce`, which runs `uv run crossvenue reproduce` (`-v` prints the details) and then the full mutation test.
2. Fix each problem it prints, in the run or in its check, never in the text alone. [Design](../explanation/design.md#checking-the-documents-against-their-sources) says what the check reads and how each kind of value is treated.

## Draw the figures

1. Draw them: `make figures` (`uv run --extra figures python scripts/figures.py`), which writes `docs/figures/<name>.svg`, which the documents embed, and an uncommitted `<name>.png` print copy. Every value a run reports is compared with it before it is drawn, and the bootstrap intervals of the per-method means in the venue-coverage and recall figures, which no run reports, are drawn unchecked.

2. Find each figure's source:

   | Figure | Source |
   |---|---|
   | `venue_coverage_comparison` | `per_query.csv` of the retrieval run, bootstrap interval with the run's seed and resamples |
   | `recall_ndcg_comparison` | the same, recall@10 and nDCG@10 recomputed per question against `v0.2.1-gold` |
   | `recall_gold_versions` | the same retrievals scored against `v0.2-gold` and `v0.2.1-gold`, the first checked against `results/20260925T002151Z` and the second against the retrieval run |
   | `latency_steps` | the per-step p50 of `summary.csv` (`encode_p50_ms`, `search_p50_ms`, `fuse_p50_ms`, `rerank_p50_ms`), or for a run written before those columns the per-step p50 table of its `report.md`; total p50 from `summary.csv` |
   | `refusals_partial` | `answers.jsonl` of the v1 and v2 answer runs, under the outcome rule of methodology §9a |
   | `judge_calibration` | `calibration.json` of the v1 answer run |
   | `wrong_refusals` | `comparison.csv` of the prompts run, checked against the answer runs |
   | `off_venue` | `retrieved_venues` of the retrieval run's `per_query.csv`, each mean checked against the value findings uses (methodology §9b) |

## Build the corpus and the index

1. Install, start Postgres and fetch the sources: `make install db fetch`. `make fetch` checks out the commits `data/manifest.csv` records and stops when a file's sha256 differs, such as that of a GitBook page (Hyperliquid, PancakeSwap) that changed after the manifest was written.
2. Chunk, deduplicate and index: `make ingest index stats`. `make stats` prints `index_version` `2825a7160c4d09bf` and 11,934 chunks when the sources match `data/manifest.csv`; `eval` refuses to run on any other `index_version`.
3. Check the config and the artifacts: `uv run crossvenue check` (config valid, artifacts present, `index_version` consistent).

## Measure drift in the sources

1. Keep a copy of the manifest: `cp data/manifest.csv /tmp/before`.
2. Follow the branches: `uv run crossvenue fetch --follow-branch`, which rewrites the manifest and keeps `fetched_at` for every file whose hash and commit are unchanged.
3. Compare the two: `diff /tmp/before data/manifest.csv`. Which rows a moved git source or a changed GitBook page changes is in [corpus datasheet, maintenance](../reference/corpus.md#maintenance).

## Run the retrieval evaluation on v0.2.1-gold

1. Run it: `uv run crossvenue eval -q evals/datasets/v0.2.1-gold.jsonl`, which writes `results/<timestamp>/`. BM25 metrics are reproduced exactly, and the dense-dependent methods up to what a rebuild or the compute device changes ([design, `index_version`](../explanation/design.md#index_version)).
2. Find each number of findings in the run:

   | Findings section | File | Rows and columns |
   |---|---|---|
   | Comparison table (recall, nDCG, nDCG′, venue coverage) | `by_category.csv` | `category=comparison`; `recall@10`, `ndcg@10`, `ndcg_prime@10`, `venue_coverage@10` |
   | venue_split − hybrid_rerank, hybrid_rerank − hybrid, − dense, − bm25, oracle − venue_split | `paired.csv` | `a`, `b`, `metric`, `scope`; `diff`, `low`, `high`, `better`, `worse`, `ties`, `mcnemar_p`, `randomisation_p`, `role` (primary, secondary, or descriptive for a scope under 50 questions), `holm_p` (secondary rows: Holm within metric and scope) |
   | `unjudged@10` and nDCG′@10 − nDCG@10 on the 90 labeled questions | `summary.csv` | `unjudged@10`, `ndcg@10`, `ndcg_prime@10` |
   | Cross-encoder latency | `report.md` | per-step p50 table; totals also in `summary.csv` as `p50_ms`, `p95_ms`, and in runs that have the per-step columns, each step's p50 and p95 as `<step>_p50_ms`, `<step>_p95_ms` |
   | Small slices | `by_category.csv` | `labelled_questions` per category |

### Compare venue detection with the labeled venues

1. Run the detector over the questions:

   ```bash
   uv run python - <<'PY'
   import json, collections
   from crossvenue_rag.retrieval.venue_detect import VenueDetector, detection_outcome
   det = VenueDetector.from_file('data/sources.yaml')
   qs = [json.loads(l) for l in open('evals/datasets/v0.2.1-gold.jsonl')]
   slices = {'comparison': [q for q in qs if q['category'] == 'comparison'],
             'naming >= 2 venues': [q for q in qs if len(q['venues']) >= 2],
             'all': qs}
   for name, group in slices.items():
       c = collections.Counter(detection_outcome(det.detect(q['question']), q['venues']) for q in group)
       print(name, len(group), dict(c))
   PY
   ```

### Score the same retrievals against v0.2-gold and v0.2.1-gold

1. Score the saved retrievals against both gold versions:

   ```bash
   uv run python - <<'PY'
   import csv, json, collections
   from crossvenue_rag.evaluation.metrics import recall_at_k
   run = 'results/20260925T012314Z'
   def gold(path):
       return {q['id']: q['gold_chunk_ids'] for q in map(json.loads, open(path))
               if q['category'] == 'comparison' and q['gold_chunk_ids']}
   old, new = gold('evals/datasets/v0.2-gold.jsonl'), gold('evals/datasets/v0.2.1-gold.jsonl')
   scores = collections.defaultdict(lambda: ([], []))
   for r in csv.DictReader(open(f'{run}/per_query.csv')):
       if r['question_id'] in new:
           ids = r['retrieved_ids'].split()
           scores[r['method']][0].append(recall_at_k(ids, old[r['question_id']], 10))
           scores[r['method']][1].append(recall_at_k(ids, new[r['question_id']], 10))
   for m, (a, b) in scores.items():
       print(f"{m:20} {sum(a)/len(a):.3f} {sum(b)/len(b):.3f} {sum(b)/len(b)-sum(a)/len(a):+.3f}")
   PY
   ```

2. Read the augmentation counts (534 chunks shown, 17 relevant, 13 at ranks 11–20) in `evals/datasets/CHANGELOG.md` under `v0.2.1-gold` and in the `augmentations` entries of `evals/datasets/annotations/v0.2.1-gold.rounds.json`.

### Recompute the off-venue shares from the saved run

1. Re-score the `retrieved_venues` the run saved; no retrieval runs again ([methodology §9b](../../evals/methodology.md#9b-off-venue-retrieval-exploratory)):

   ```bash
   uv run python - <<'PY'
   from crossvenue_rag.reporting.compute.retrieval_off_venue import off_venue_means, off_venue_paired_rows
   from crossvenue_rag.reporting.sources import Sources
   src = Sources(".")
   for category in ("comparison", "adversarial"):
       for method in ("bm25", "dense", "hybrid", "hybrid_rerank", "venue_split", "venue_split_oracle"):
           n, m = off_venue_means(src, method, category)
           print(f"{category:12} {method:19} n={n:2} off_venue@10={m['off_venue@10']:.3f} "
                 f"via_ccxt@10={m['via_ccxt@10']:.3f} top1={m['off_venue@1']:.2f}")
   for r in off_venue_paired_rows(src):
       holm = "" if r.holm_p is None else f" p_Holm={r.holm_p:.3f}"
       print(f"{r.a} - {r.b:7} {r.metric:13} {r.scope:12} n={r.n:2} "
             f"{r.diff:+.3f} [{r.low:+.3f}, {r.high:+.3f}] better/worse/ties "
             f"{r.better}/{r.worse}/{r.ties}{holm}")
   PY
   ```

2. Read the output: lower is better for both shares, so `better` counts the questions where hybrid_rerank has the smaller share. The adversarial rows, 13 questions below the 50 that methodology §7 requires for a p-value, are descriptive.

## Time one cross-encoder call against one per venue

1. Run the bench: `uv run crossvenue bench-rerank -q evals/datasets/v0.2.1-gold.jsonl`, which writes `results/<timestamp>_bench/`.
2. Read the findings table from `summary.csv` (`calls_per_query`, `pairs_per_call_mean`, `total_p50_ms`, `total_p95_ms`, `total_mean_ms`) and the load during timing and the discarded pairs from `machine.json`. Before each pair the bench waits for a quiet machine, and it stops with an error that names `--force` when it cannot get one.
3. To publish another bench run, point `BENCH_RUN` in `src/crossvenue_rag/reporting/run_ids.py` at it; `make reproduce` then names every sentence to update.

## Simulate the power of the recall comparison

1. Run the simulation: `uv run crossvenue power --run results/20260925T132613Z`, which writes `results/<timestamp>_power/`.
2. Read `delta_80_holm_first_step`, `delta_80_unadjusted` and the counts in the run's `power.json`; the findings quote those of `results/20260926T015648Z_power`. The simulation is the Holm-adjusted paired randomization test on per-question recall@10 over the comparison questions, with the discordance and difference sizes of the run's comparison family (methodology §7).

## Measure agreement between annotation rounds

1. Run `uv run crossvenue kappa`.
2. Read the registered reading (κ with round 1 as corrected, over all sampled questions), its bootstrap interval and the three sensitivity readings of methodology §5. Each sensitivity reading changes one or both of two choices: round 1 as first judged instead of as corrected, and only the questions not listed in `evals/selection/round2_exposed.txt` instead of all.

## Count answer refusals

1. Count the prompt v1 refusals from `answers.jsonl`, with no model call, under the outcome rule of methodology §9a. An answer flagged partial with no claim is a refusal; a partial answer is one flagged partial that has a claim:

   ```bash
   F=results/20260925T175128Z_answers/answers.jsonl
   REFUSED='.generation.answer | .refused or (.partial == true and (.claims | length) == 0)'
   PARTIAL='.generation.answer | .partial == true and (.claims | length) > 0'
   jq -r "select($REFUSED) | [.config, .category] | @tsv" $F | sort | uniq -c
   ```

2. Sort the refusals into groups a, b and c by comparing, per named venue, `generation.chunk_ids` with the question's `gold_chunk_ids` in `evals/datasets/v0.2.1-gold.jsonl`, each chunk's venue taken from `data/processed/chunks.jsonl`. Group a has a gold chunk of every named venue in the context, b of some, c of none.
3. Count prompt v2's refusals and partial answers (`results/20260925T202827Z_answers`):

   ```bash
   F=results/20260925T202827Z_answers/answers.jsonl
   jq -r "select($REFUSED) | [.config, .category] | @tsv" $F | sort | uniq -c
   jq -r "select($PARTIAL) | [.config, .category] | @tsv" $F | sort | uniq -c
   ```

4. Check the counts against the runs' `summary.csv` (`refused`, `partial`) and `per_answer.csv` (`outcome`), which apply the same rule. The cost and call duration per configuration are in each run's `summary.csv` (`cost_usd`, `cost_usd_new`, `latency_p50_ms`, `latency_p95_ms`, `output_tokens`, `invalid_responses`).

## Judge the answers

1. Run the judging commands, which make paid calls through OpenRouter. Every response that passes validation is cached, so a repeat with the same answers and the same judge pays only for the calls that never got a valid response.

   ```bash
   uv run crossvenue judge-sample --judge sonnet        # calibration sample, both candidates
   uv run crossvenue judge-sample --judge opus
   uv run crossvenue rate --rater <name>                # the human ratings
   uv run crossvenue judge-sample --judge opus --replicate 1
   uv run crossvenue select-judge --run results/20260925T175128Z_answers
   uv run crossvenue judge-run --run results/20260925T175128Z_answers --judge opus \
     --set generation.generator_prompt=prompts/generator.v1.md --set generation.answer_schema_version=1
   uv run crossvenue judge-run --run results/20260925T202827Z_answers --judge opus
   uv run crossvenue compare-prompts --v1 results/<ts>_judged --v2 results/<ts>_judged
   ```

2. Read the outputs, which `make reproduce` also reads: `calibration.json` and `calibration.md` from `select-judge`, the primary and secondary comparisons of methodology §9 in the judged runs' `paired.csv`, and H-v2 of §9a in the `_prompts` run's `comparison.csv`. In the committed judged runs the faithfulness columns are uncalibrated and enter no result.

## Compare PancakeSwap with Uniswap

1. Fit TF-IDF over both venues' chunks and compare each PancakeSwap chunk with its best Uniswap match; `scikit-learn` is a direct dependency in `pyproject.toml`.

   ```bash
   uv run python - <<'PY'
   import json, numpy as np
   from sklearn.feature_extraction.text import TfidfVectorizer
   rows=[json.loads(l) for l in open('data/processed/chunks.jsonl')]
   uni=[r['text'] for r in rows if r['venue']=='uniswap']
   pan=[r['text'] for r in rows if r['venue']=='pancakeswap']
   X=TfidfVectorizer(min_df=2, max_features=60000, stop_words='english').fit_transform(uni+pan)
   U, P = X[:len(uni)], X[len(uni):]
   best=np.concatenate([(P[i:i+200] @ U.T).toarray().max(axis=1) for i in range(0,P.shape[0],200)])
   for t in (0.9,0.7,0.6,0.5,0.4):
       print(f">={t}: {(best>=t).sum()} ({(best>=t).mean()*100:.2f}%)")
   print("median:", round(float(np.median(best)),3))
   PY
   ```

## Count duplicate text

1. Count the duplicate groups from `chunks.jsonl`, where every occurrence of a merged text is in the chunk's `sources`; `make stats` prints the same totals under "exact deduplication".

   ```bash
   uv run python - <<'PY'
   import json, collections
   rows=[json.loads(l) for l in open('data/processed/chunks.jsonl')]
   occ=sum(len(r['sources']) for r in rows)
   groups=[r for r in rows if len(r['sources'])>1]
   dup=sum(len(r['sources']) for r in groups)
   print('occurrences', occ, '| in groups', dup, f'({dup/occ:.1%})', '| groups', len(groups))
   print(sorted(collections.Counter(r['venue'] for r in groups).items(), key=lambda kv: -kv[1]))
   venues=collections.defaultdict(set)
   for r in rows: venues[r['text']].add(r['venue'])
   print('texts on more than one venue', sum(len(v)>1 for v in venues.values()))
   PY
   ```

2. List the page pairs behind the Binance groups:

   ```bash
   uv run python - <<'PY'
   import json, collections
   rows=[json.loads(l) for l in open('data/processed/chunks.jsonl')]
   g=[r for r in rows if len(r['sources'])>1 and r['venue']=='binance']
   c=collections.Counter(tuple(sorted({s['doc_path'] for s in r['sources']})) for r in g)
   print(c.most_common(3))
   PY
   ```
