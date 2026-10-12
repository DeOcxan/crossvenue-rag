# Label a question set

How to select, label, correct, augment, validate and freeze a question set. What the current set contains is in the [datasheet](../../evals/datasheet.md), and why each step works the way it does is in the [methodology](../../evals/methodology.md). Where each file is, in [`evals/README.md`](../../evals/README.md); what each command shows and checks, in the [CLI reference](../reference/cli.md#question-set).

## Select a set from candidates

1. Validate every dataset, the newest candidate file among them: `make validate-evals`.
2. Walk the candidates with `make select` (or `crossvenue select`; `--category comparison` for one category, `--include-skipped` to revisit skipped candidates). Categories come in order of risk of running out.
3. Reject with a reason: `not grounded`, `paraphrased heading`, `duplicate information need`, `unrealistic phrasing`, `wrong category`, `other`.
4. Write the working set: `make select-finalize`. It refuses a set that misses a category target ([methodology decision 2](../../evals/methodology.md#decisions-at-a-glance)) or has fewer than 50 answerable comparison questions.
5. Keep or drop a question here, never by rewriting a candidate file; every decision is recorded in `evals/selection/selection.jsonl`.

## Label (round 1)

1. Start the session with `make label` (or `crossvenue label --limit 25`; `--all` revisits labeled questions, `-q evals/other.jsonl` labels another file). Each unlabeled question shows 25 distinct candidates, pooled from every pool method's top 20 with RRF.
2. For each question, enter the relevant numbers (`1 4 7`, `2-5`), `n` if nothing in the pool answers it, or Enter to skip. Progress is saved after every question; Ctrl-C is safe.
3. Judge against the written criterion in [methodology §5a](../../evals/methodology.md#5a-annotation-instructions), and add a decision the criterion does not cover there, dated.
4. Answer the checks the tool runs before it saves. For a comparison, say which venue the marks miss (`b` back, `a` absent from that venue's docs, `m` no suitable candidate in the pool). For a `version_sensitive` or `adversarial` question, confirm the `environment` and `api_version` of the marks (Enter saves, `b` goes back).

## Correct round 1

1. Re-open labeled questions on the pool round 1 recorded: `crossvenue label --review cv-0043 cv-0046`, or `crossvenue label --review-file evals/selection/review_round1.txt`. A change is appended to `corrections` in the journal with the previous set.
2. Fill the `venue_coverage` values that follow from recorded data, leaving the questions under review to the annotator: `crossvenue derive-coverage --exclude-file evals/selection/review_round1.txt`.

## Judge a method that is not in the pool

Before comparing the recall of a method that is not in the pool the questions are labeled from:

1. Retrieve its top 20 without metrics: `crossvenue pool-run -m venue_split -m venue_split_oracle --k 20 --category comparison`, which writes `results/<ts>_pool/`.
2. List the chunks round 1 never judged:

   ```bash
   crossvenue augment-candidates --run results/<ts>_pool -m venue_split -m venue_split_oracle \
       --k 20 --category comparison -o evals/selection/augment_round1.jsonl
   ```

3. Judge them: `crossvenue label --augment --question-file evals/selection/augment_round1.jsonl`. Relevant chunks are appended to gold.

## Measure agreement (round 2)

1. Draw the stratified list once: `make round2-sample` writes `evals/selection/round2_sample.txt`. Leave the file unchanged, so the questions judged are fixed in advance.
2. No earlier than the date in [methodology decision 7](../../evals/methodology.md#decisions-at-a-glance), and without looking at round 1, run `make relabel`. Its screen shows no earlier verdict, and it writes only to the journal.
3. Compute Cohen's kappa over the candidates judged in both rounds: `make kappa`.
4. Write the result to `evals/agreement.md`.

## Validate

1. Run the full check locally: `make validate-evals` (every `evals/datasets/*.jsonl`), or `crossvenue validate-evals -d evals/datasets/v0.2.1-gold.jsonl` for one file.
2. Read every error and warning; the [CLI reference](../reference/cli.md#validate-evals) lists what each is.

## Freeze a version

1. Run the full `make validate-evals` locally and read its output: `ok`, and every warning resolved or recorded. CI runs `validate-evals --skip-near-duplicates`, since the near-duplicate check needs the embedding model, so a green CI says nothing about near-duplicate questions.
2. Copy `evals/questions.jsonl` to `evals/datasets/<version>.jsonl` and confirm with `cmp`.
3. Copy the journal `evals/annotations/questions.rounds.json` to `evals/datasets/annotations/<version>.rounds.json`.
4. Write `<version>.meta.json`: dataset and schema version, corpus snapshot (`crossvenue_rag.dataset.manifest_fingerprint("data/manifest.csv")`, `index_version`), source licenses, generator, counts recomputed from the file, and the sha256 of the two files as `dataset_sha256` and `journal_sha256` (`shasum -a 256`).
5. Run `make validate-evals` again; it recomputes the counts, compares the manifest hash and `index_version` with the corpus, compares every gold list with what its journal entries resolve to, and checks both hashes.
6. Add a `datasets/CHANGELOG.md` entry: what changed, why, how it is verified.
7. Leave a frozen version under `evals/datasets/` as it is, and make a change as another version with its own `.meta.json` and CHANGELOG entry. A metric is interpretable only against the exact set it was computed on.

## Cite a question

1. Cite a question outside its own file (in reports, commit messages, `notes` fields and CHANGELOG entries) as its id plus the first six words of its text: `cv-0045 "do binance and bybit ban you"`. Ids are positional and sets get renumbered, so an id alone can point at another question while reading as specific.
2. Grep the quote from the question file before writing it.

## Show that the corpus lacks an answer

1. Count the term under every spelling the vendor uses, by the convention of the [corpus datasheet](../reference/corpus.md#counting-a-term), before recording that a question has no answer in the corpus or that a venue lacks a concept.
