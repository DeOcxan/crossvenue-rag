# Datasheet: v0.2.1-gold

A datasheet for the gold evaluation set `evals/datasets/v0.2.1-gold.jsonl`, following *Datasheets for Datasets* (Gebru et al., 2021, [doi:10.1145/3458723](https://doi.org/10.1145/3458723); full citation in the [methodology references](methodology.md#references)). The protocol it is built under is [methodology](methodology.md), and where the set departs from it is listed in [Deviations from protocol](methodology.md#deviations-from-protocol). The version record is [`datasets/CHANGELOG.md`](datasets/CHANGELOG.md); the corpus it is labeled against is described in [`docs/reference/corpus.md`](../docs/reference/corpus.md).

| - | - |
|---|---|
| File | `evals/datasets/v0.2.1-gold.jsonl` (sha256 `2105ce83…`, in full in `results/20260925T012314Z/index.json`), byte-identical to `evals/questions.jsonl` |
| Metadata | `v0.2.1-gold.meta.json` |
| Judgment journal | `datasets/annotations/v0.2.1-gold.rounds.json` |
| Corpus | manifest `c419b4be…` (in full in the `corpus_snapshot` of `v0.2.1-gold.meta.json`; 1,665 documents), `index_version` `2825a7160c4d09bf`, 11,934 chunks |
| Supersedes | `v0.2-gold`, the same 114 questions without the augmentation's 17 gold chunks |

## Motivation

Purpose. To measure whether retrieval returns the venue a question asks about when near-identical documentation from other venues competes, and whether a system declines when the corpus holds no answer. The headline measurement is the 50-question comparison subgroup, sized to detect a recall@10 gap of 0.20 with 80% power ([methodology §7](methodology.md#7-set-size-and-statistical-power)).

Creator. The repository's author, as sole annotator; no external funding.

## Composition

114 questions; 94 answerable, 20 unanswerable; 90 carry gold chunks, 277 gold chunks in all.

| category | questions | answerable | with gold | gold chunks |
|---|---:|---:|---:|---:|
| comparison | 50 | 50 | 50 | 164 |
| unanswerable | 20 | 0 | 0 | 0 |
| adversarial | 13 | 13 | 11 | 28 |
| fact | 8 | 8 | 8 | 23 |
| ambiguous | 6 | 6 | 6 | 22 |
| version_sensitive | 6 | 6 | 5 | 16 |
| noisy | 6 | 6 | 6 | 14 |
| multihop | 5 | 5 | 4 | 10 |

Instances. One JSON record per question: the text, the venues it concerns, `qtype` and the finer `category`, `difficulty`, `is_answerable`, `expected_behavior` (answer 88, refuse 20, ask_clarification 6), `adversarial_type` for adversarial items, `notes`, `source_chunk_ids` (provenance), `gold_chunk_ids` (the judgment), `venue_coverage` for comparisons, annotation stamps, and `generator` / `prompt_hash` ([record format](#record-format)).

Venues. Questions per venue named: binance 42, bybit 34, hyperliquid 27, bingx 24, uniswap 18, pancakeswap 18, dydx 14, ccxt 6. 55 questions name one venue, 53 two, 6 three or more; 57 name at least one DEX.

Question text. Median length 11 words. Difficulty: easy 6, medium 61, hard 47.

Word overlap with gold. `validate-evals` measures the longest run of consecutive words a question shares with any of its gold chunks, spans in backticks or double quotes excluded (methodology §6, check 2). That run is at most 1 word for 41 of the 90 questions with gold, 2 for 40, 3 for 7 and 4 for 2. No question reaches the 6-word threshold. The two 4-word runs are "the same as on" in `cv-0128 "is the ip ban on binance"` and "403 access too frequent" in `cv-0004 "bybit sends 403 access too frequent,"`, the second an error message a developer types.

Four answerable questions have no gold chunk, each judged "nothing in the pool answers this". For `cv-0067 "bybit caps limit at 1000, how"`, `cv-0088 "binance postOnly timeInForce isnt working, what"` and `cv-0090 "my hyperliquid api wallet address returns"`, the source chunks never reached the pool. For `cv-0132 "is switch-isolated still how i change"`, the review removed its only marked chunk, the deprecated `docs/v5/abandon/` page. They count in no recall average.

Sensitive content. None.

### Record format

The five fields the record schema requires:

```json
{"id": "...", "question": "...", "tier": "gold|silver|bronze",
 "venues": ["binance", "bybit"], "qtype": "fact|comparison|multihop"}
```

Versioned files carry the full schema on the same record:

| Field | Meaning |
|---|---|
| `tier` | `gold` · `silver` · `bronze`; required by the record schema and shown in the selection screen, and read by no metric, report or selection rule. `v0.2.1-gold` holds 108 `silver` and 6 `bronze` |
| `category` | `fact` · `comparison` · `multihop` · `unanswerable` · `ambiguous` · `adversarial` · `version_sensitive` · `noisy`; finer than `qtype`, and what breakdowns and the refusal measurement key off |
| `difficulty` | `easy` · `medium` · `hard` |
| `is_answerable` | whether the corpus contains the answer at all |
| `expected_behavior` | `answer` · `refuse` · `ask_clarification` |
| `adversarial_type` | names the trap; required when `category` is `adversarial` |
| `source_chunk_ids` | chunks the question is written from: provenance and a check on pool completeness, never a gold label |
| `gold_chunk_ids` | the judgment |
| `venue_coverage` | per named venue of a labeled answerable comparison: `covered` · `absent_in_docs` · `missing_from_pool` |
| `annotator` · `annotated_at` · `annotation_round` | stamped by the labeling tool |
| `created_at` · `generator` · `prompt_hash` | how the question is produced |

The fields are cross-checked (`QuestionRecord` in `src/crossvenue_rag/dataset/records.py`, checked by `make validate-evals`). An unanswerable question expects refusal and carries no gold chunks, an ambiguous one expects a clarifying question or a refusal, an adversarial one names its trap, and a comparison names at least two venues. `venue_coverage` is `covered` exactly where a gold chunk is from that venue.

## Collection process

Candidates. 204 candidate questions across `v0.1-candidates` … `v0.1.6-candidates` are the drafts of an LLM, Claude Opus 5 (`claude-opus-5`). The `generator` field of every record names the model and the drafting pass, and `prompt_hash` holds a hash of the brief. Each draft starts from a recurring developer problem mined from SDK issue trackers ([domain research](domain-research.md)) and is grounded in corpus chunks recorded as `source_chunk_ids`. Every unanswerable premise and every "absent on one venue" comparison is checked by searching the concept under every vendor spelling with `crossvenue count-term`, counts in `notes`. For the unanswerable questions listed under [Known limitations](#known-limitations) the search did not find the answer ([deviation D23](methodology.md#deviations-from-protocol)). Candidate files are immutable; each supplement is another version (comparison supplement in `v0.1.2`, headroom for the thin categories in `v0.1.3`, id migration after deduplication in `v0.1.5`).

Selection. `crossvenue select`, category by category, thin categories first. Of the 204 candidates, 117 were shown: 114 accepted, 1 rejected (`cv-0139 "bingx websoket close always, need send"`, unrealistic phrasing), 2 skipped. The other 87 were never shown, because the category targets were met first. With 1 rejection among the 117 shown, the set followed the order of acceptance, and question text is the LLM draft unchanged. Every decision is in `evals/selection/selection.jsonl`.

## Labeling

Round 1 (2026-09-24, one annotator). For each question the tool pooled the top 20 of bm25, dense, hybrid and hybrid_rerank with RRF and showed the top 25 distinct chunks. The annotator marked the relevant ones against the written criterion ([methodology §5a](methodology.md#5a-annotation-instructions)). Every judgment is stamped with the `index_version`, and the journal records the pool shown as well as the chunks marked. Time between consecutive saves, from the round-1 `annotated_at` timestamps in the journal: median 17 s per question, 0.7 s per candidate at 25 candidates. The longer session saved 113 questions over 3,325 s, a mean of 29.7 s between consecutive saves and 1.2 s per candidate. Round 2 reproduced these labels at κ 0.470 [0.317, 0.645] (95% percentile bootstrap over questions; [agreement](agreement.md)), below the registered target, so the pace and the agreement are this version's main caveats on its labels.

Review. 26 questions re-opened on their recorded round-1 pool, chosen by quality checks (comparisons whose gold missed a named venue with candidates shown; adversarial and version-sensitive items with trap-side chunks marked relevant). The labels of 11 changed: 3 chunks added, 9 removed. Corrections are journaled with the previous set.

Venue coverage. Every labeled answerable comparison records, per venue it names, whether the gold covers it (`covered`), the venue's documentation does not answer it (`absent_in_docs`), or no suitable candidate of the venue is in the pool (`missing_from_pool`). 101 values on 51 questions, the 50 comparison questions (100 values) and one version_sensitive question that also records a value: covered 86, missing_from_pool 14, absent_in_docs 1. Values that follow from recorded data are derived mechanically (`crossvenue derive-coverage`); only the annotator sets `absent_in_docs`.

Augmentation. `venue_split` and `venue_split_oracle` joined evaluation after round 1, and judging their unjudged chunks was decided after their results were seen ([deviation D4](methodology.md#deviations-from-protocol)). On the 50 comparison questions, the chunks in their top 20 that round 1 never showed (534, 211 of them in the top 10) were judged (`crossvenue label --augment`, run `results/20260925T003721Z_pool`). 17 were relevant, 13 of them at ranks 11–20; they are this version's only change from `v0.2-gold`, appended to 15 questions' gold, nothing removed. Nine venue values became `covered`: 8 from `missing_from_pool`, and 1 from `absent_in_docs`, bingx on `cv-0153 "bybit cancels my orders when i"`, whose answer is at rank 20 of venue_split. The journal keeps that correction as a wrong assessment.

Agreement. Round 2 re-judged 25 questions blind on 2026-10-10. The sample is stratified by category and was fixed before round 2 in `evals/selection/round2_sample.txt` (comparison 11, unanswerable 5, adversarial 3, fact 2, one each of ambiguous, multihop, noisy and version_sensitive; methodology §5). Cohen's κ between round 1 as corrected and round 2 is 0.470 [0.317, 0.645] over 750 candidate decisions, below the registered target of 0.61. The reproducibility of these labels is therefore a limitation of this version ([agreement](agreement.md)).

Relabel. Round 3, registered in [methodology §5b](methodology.md#5b-annotation-round-3-pre-registered), is scheduled for 2026-10-31 and will give `v0.3-gold`, with every label under one criterion.

## Version v0.2.2-gold

`evals/datasets/v0.2.2-gold.jsonl` supersedes this version. Every other section of this datasheet describes `v0.2.1-gold`, the version the registered results are scored on. `v0.2.2-gold` holds the same 114 questions in the same order, with the round-2 disagreements adjudicated and the questions of deviation D23 relabeled, under [methodology §5a](methodology.md#amendment-of-2026-10-10) as amended on 2026-10-10 ([CHANGELOG](datasets/CHANGELOG.md#v022-gold-2026-10-10)). Its counts beside this version's:

| Count | `v0.2.1-gold` | `v0.2.2-gold` |
|---|---|---|
| questions with gold | 90 | 92 |
| gold chunks | 277 | 282 |
| fact questions | 8 | 10 |
| unanswerable questions | 20 | 18 |

The adjudication decided the 63 pairs on which round 1 and round 2 disagree, on 17 questions, and changed the gold of 6 of them. The D23 relabel made 2 questions fact and kept 1 unanswerable. In all, the gold of 8 questions differs from this version. The 63 disagreeing pairs of the 25 sampled questions and the D23 questions follow the amended criterion. Every other label follows the round-1 criterion, among them every label of the other 87 questions ([deviation D25](methodology.md#deviations-from-protocol)), so recall on `v0.2.1-gold` and on `v0.2.2-gold` is not comparable. The journal, `datasets/annotations/v0.2.2-gold.rounds.json`, holds round 1, round 2, the adjudication and the relabel; `make validate-evals` checks each record's gold against what its entries resolve to.

## Known limitations

- Pooling bias. Gold is recall within the pool: a relevant chunk no pooled method found is never judged, and further judging could move absolute recall either way ([methodology §1](methodology.md#1-test-collection-construction-and-pooling)). How much scoring the same retrievals against `v0.2-gold` and `v0.2.1-gold` changes every method's comparison recall@10 is a result in [findings](../docs/results/findings.md#adding-17-gold-chunks-changes-every-methods-comparison-recall10-by-0034-to-0092). The judged depth is set by the display cap, not the pool depth: round 1 judged the fused top 25 of the four round-1 pool methods. The augmentation was decided after results were seen ([deviation D4](methodology.md#deviations-from-protocol)). In `results/20260925T012314Z` on `v0.2.1-gold`, after the augmentation, each round-1 pool method's own top 10 on the 50 comparison questions has an unjudged@10 of 0.032–0.088 (bm25 0.088, dense 0.078, hybrid and hybrid_rerank 0.032). venue_split and the oracle, augmented to depth 20, are fully judged there. A recall comparison between a round-1 pool method and an augmented one is therefore judged more completely on one side. A method in neither group has every relevant chunk only it finds scored as a miss and must be augmented first ([methodology §1](methodology.md#1-test-collection-construction-and-pooling)).
- Augmentation covers comparison questions only. On the six ambiguous questions, venue_split_oracle's unjudged@10 is 0.467 (`results/20260925T012314Z`), a result in [findings](../docs/results/findings.md#three-limits-on-the-retrieval-results).
- One-sided gold. 15 of the 50 comparisons have a named venue recorded `missing_from_pool` (14) or `absent_in_docs` (1); recall on them is measured against the side the pool found.
- Small slices. Outside comparison the labeled categories hold 4–11 questions; their per-category figures are descriptive, never significance claims.
- One annotator and fast judgments, reproduced by the same annotator at κ 0.470 [0.317, 0.645] (95% percentile bootstrap over questions; [agreement](agreement.md)). The answer-rating screen showed the gold chunks of 9 of the 25 round-2 questions during the quiet period (`evals/selection/round2_exposed.txt`); κ is reported with and without them ([methodology §5](methodology.md#decision-for-this-project-4)).
- Source-chunk marker. The round-1 screen marked the chunk each question was drafted from; shown source chunks were marked relevant 88 times out of 89, other pool chunks 172 times out of 2,759. Round 2 hid the marker.
- Two questions labeled unanswerable have answers in the corpus: `cv-0115 "does bybit have a mass cancel"` (Bybit documents `POST /v5/order/cancel-all`) and `cv-0121 "which bingx endpoint returns the usdt-m"` (BingX's `GET /openApi/swap/v2/quote/fundingRate` takes a time range). Both are fact in [`v0.2.2-gold`](#version-v022-gold) ([deviation D23](methodology.md#deviations-from-protocol)). `cv-0106 "which binance endpoint sets leverage on"`, which D23 also listed, stays unanswerable: the corpus has no Binance futures documentation, and CCXT's `setLeverage` chunk names Binance's Change Initial Leverage page by URL and gives no endpoint. An answer built from it would attribute a CCXT page to Binance, the wrapper kind of attribution in deviation D14.
- LLM-drafted questions carry the generator-bias risk of [methodology §2](methodology.md#2-synthetic-question-generation-used-as-a-drafting-aid-only). The lexical shortcut is measured: no question shares 6 or more consecutive words with any of its gold chunks, spans in backticks or double quotes excluded (Word overlap with gold, above).
- ccxt appears in 6 questions, below the protocol's floor of 8 per venue.

## Uses

Intended. Comparing retrieval methods on this corpus at `index_version` `2825a7160c4d09bf`: recall, nDCG, nDCG′ and MRR at k ≤ 10 within one gold version, and `venue_coverage@k` on the comparison subgroup. Refusal behavior on the 20 unanswerable questions is measured for four generator configurations under prompt v1 ([findings, prompt v1 refusals](../docs/results/findings.md#under-prompt-v1-geminivenue_split-refuses-25-of-the-50-comparison-questions)) and prompt v2 ([findings, prompt v2 refusals](../docs/results/findings.md#prompt-v2-lowers-wrong-refusals-in-every-configuration-exploratory)).

Not intended. Quoting absolute recall as a property of a method; comparing recall across gold versions; a p-value for a category or slice of fewer than 50 questions ([methodology §7](methodology.md#the-power-ceiling-as-a-number)); evaluating against another `index_version` (the harness refuses); training.

## Distribution

In the repository, alongside its `.meta.json` and journal. The questions and labels are the project's own; the corpus text they point at is not redistributed ([corpus datasheet](../docs/reference/corpus.md#distribution)).

## Maintenance

The file is immutable. A changed label, question or index is another version with its own `.meta.json` and a [`CHANGELOG.md`](datasets/CHANGELOG.md) entry; an index change that only renames chunk ids is carried across with `crossvenue id-map` / `remap-ids`. Round 2 writes only to the journal, never to this file.
