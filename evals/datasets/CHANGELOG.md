# Dataset changelog

Every dataset file in this directory is immutable. A change to the question set is a new version, never an edit in place, because a metric is only interpretable against the exact set it was computed on. `results/<timestamp>/` records the config that produced a run; this file records the changes to the data, its later dated notes included.

Versioning:

- `vX.Y-candidates`: generated, unfiltered, unlabeled. Input to human selection.
- `vX.Y-gold`: selected and labeled. What the harness is pointed at.

Each version ships two files: `<version>.jsonl` (the questions) and `<version>.meta.json` (dataset version, schema version, corpus snapshot hash, source licenses, generator, and the distribution across venues, types and categories). A gold version also ships its judgment journal, `annotations/<version>.rounds.json`. `make validate-evals` checks each. A version's freeze date is the UTC date its files were written, `corpus_snapshot.recorded_at` in its `.meta.json`, which the tool records to the second and which matches `created_at` from v0.1.4 on. The hand-entered `created_at` of v0.1.2 (21:40Z) and v0.1.3 (18:30Z) put v0.1.3 first, against the ids, the record counts, v0.1.3's brief (which names v0.1.2) and `recorded_at` (20:37Z and 21:36Z). The date, 2026-09-20, is the same either way.

Paths in the entries below and in the `.meta.json` notes are as they were when each version was made. The renamed files: `evals/METHODOLOGY.md` is [`evals/methodology.md`](../methodology.md), `evals/DATASHEET.md` is [`evals/datasheet.md`](../datasheet.md), `docs/DESIGN.md` is [`docs/explanation/design.md`](../../docs/explanation/design.md), `docs/CORPUS.md` is [`docs/reference/corpus.md`](../../docs/reference/corpus.md), `docs/FINDINGS.md` is [`docs/results/findings.md`](../../docs/results/findings.md), `evals/README.md` is [`docs/how-to/label.md`](../../docs/how-to/label.md). `evals/selection.jsonl`, `evals/review_round1.txt` and `evals/augment_round1.jsonl` are in `evals/selection/`. The function `dataset.questions.resolved_relevant` is `dataset.journal.QuestionJournal.resolved_relevant`.

## v0.2.2-gold (2026-10-10)

The round-2 disagreements adjudicated and the questions of deviation D23 relabeled, both under [methodology §5a](../methodology.md#amendment-of-2026-10-10) as amended on 2026-10-10 after round 2. The gold of 8 of the 114 questions changed (277 to 282 gold chunks: 13 added, 8 removed). 2 questions were recategorized from unanswerable to fact (unanswerable 20 to 18, fact 8 to 10, questions with gold 90 to 92), and the notes of the 3 D23 questions were rewritten. Same 114 questions in the same order. Supersedes `v0.2.1-gold`, which stays on disk unchanged with its journal.

Dataset sha256 `6fda6f6a7133555407e5aaa04884f169e849c64c815565ba58539ecf38e48286` and journal sha256 `0052f19d476c53a6b222217446c9ed35a815ac52c9a4f27636ec97ed3c2901ed`, recorded as `dataset_sha256` and `journal_sha256` in `v0.2.2-gold.meta.json`, which `make validate-evals` checks. The journal, `annotations/v0.2.2-gold.rounds.json`, is a byte copy of `evals/annotations/questions.rounds.json`, holding round 1, round 2 and the two passes below; its round-1 entries are identical to those of `v0.2.1-gold`. `index_version` `2825a7160c4d09bf`, manifest `c419b4be…`: unchanged. `evals/questions.jsonl`, the working set, is unchanged and still equals `v0.2.1-gold`.

- Why. Round 2 measured κ 0.470 [0.317, 0.645] (95% percentile bootstrap over questions) against the target 0.61 ([agreement](../agreement.md)). Methodology §5 resolves the disagreements by a third adjudication pass and refines the criterion where they show it ambiguous; the resolved labels ship as another version, and κ stays computed on the unresolved rounds. Deviation D23 scheduled the review of three unanswerable labels for this version.
- Adjudication. `crossvenue label --adjudicate` (`make adjudicate`) showed each of the 63 pairs on which round 1 as corrected and round 2 disagree, on 17 questions, with both earlier verdicts. Each decision is journaled under `adjudicated` with the criterion date; rounds 1 and 2 are untouched. 14 pairs were decided relevant and 49 not relevant; 46 decisions agree with round 1 and 17 with round 2. The gold of 6 questions changed (9 chunks added, 8 removed): `cv-0006 "how does hyperliquid work out my"`, `cv-0060 "does bingx sign the same string"`, `cv-0127 "whats the default page size on"`, `cv-0140 "my short term order dissapear after"`, `cv-0145 "binance gives me 6000 weight a"` and `cv-0159 "do subaccounts get their own rate"`. On the other 11 every decision kept round 1. No `venue_coverage` value changed.
- D23 relabel. `crossvenue label --relabel` (`make relabel-d23`) showed each question's round-1 candidates followed by the answer chunks its pool missed (`evals/selection/d23_answer_chunks.jsonl`), on the round-1 screen; each verdict is journaled under `relabeled`. `cv-0115 "does bybit have a mass cancel"` is fact with 3 gold chunks and `cv-0121 "which bingx endpoint returns the usdt-m"` fact with 1. `cv-0106 "which binance endpoint sets leverage on"` stays unanswerable with no gold. The reason for each is in its `notes` field.
- Gold. Each record's `gold_chunk_ids` is what its journal entries resolve to (`dataset.questions.resolved_relevant`): a `relabeled` judgment replaces round 1, and otherwise each `adjudicated` decision is applied to round 1's corrected set. `validate-evals` checks every gold version this way.
- Criterion. The 63 disagreeing pairs of the 25 sampled questions and the 3 D23 questions follow §5a as amended. Every other label follows the round-1 criterion until a full relabel, among them every label of the other 87 questions (deviation D25). Recall on this version is not comparable with recall on `v0.2.1-gold`.

## v0.2.1-gold (2026-09-25)

17 gold chunks added to 15 of the 114 questions (260 to 277) and 9 `venue_coverage` values changed to `covered`, from judging venue_split's and venue_split_oracle's never-judged top 20 on the 50 comparison questions. Nothing else changed: the same 114 questions in the same order, 90 with gold. Supersedes `v0.2-gold`, which stays on disk unchanged with its journal.

Byte copy of `evals/questions.jsonl` (sha256 `2105ce83…`, in full as `questions_sha256` in `results/20260925T012314Z/index.json`, checked with `cmp`); judgment history in `annotations/v0.2.1-gold.rounds.json` (sha256 `c932f34f553478df148fccd45d6984027e7a5d8f13c98da86db8601b657a01a4`). `index_version` `2825a7160c4d09bf`, manifest `c419b4be…` (in full as `corpus_snapshot.manifest_sha256` in `v0.2.1-gold.meta.json`): unchanged.

- Why. `venue_split` and `venue_split_oracle` were added after round 1, so their retrievals had never been judged: `venue_split`'s unjudged@10 on the 50 comparison questions was 0.402 (deviation D4), and every unjudged chunk scores as a miss. [Methodology §1](../methodology.md#1-test-collection-construction-and-pooling) rule 3 requires judging a new method's unjudged top 20 before its recall is compared.
- How. A retrieval-only pool-construction run (`crossvenue pool-run`, `results/20260925T003721Z_pool`, both methods, k = 20, the 50 comparison questions; no metrics) gave the candidates. `crossvenue augment-candidates` kept the 534 never judged (211 in the top 10, 323 at ranks 11–20). `crossvenue label --augment` showed only those, with the current gold as context, on 2026-09-25. All 534 were shown, one record per question, journaled under `augmentations` in round 1 with the chunks shown and those marked.
- Result. 13 of the 17 added chunks came from ranks 11–20, which is what the top-20 depth was for. No existing label was removed or reordered: every v0.2-gold gold list is a prefix of its v0.2.1 list, checked per record.
- `venue_coverage`: of the 9 values changed to `covered`, 8 were `missing_from_pool` and 1 `absent_in_docs`, recorded as a wrong earlier assessment: bingx on `cv-0153 "bybit cancels my orders when i"`, where the answer turned up at rank 20. Totals: covered 86, missing_from_pool 14, absent_in_docs 1 (101 values on 51 questions).
- Round 2 ran on 2026-10-10 on the whole 25-question subsample, 15.7 to 15.9 days after round 1 counting this augmentation, which met the 14-day minimum ([methodology decision 7](../methodology.md#decisions-at-a-glance)). Its result is in the [`v0.2.2-gold` entry](#v022-gold-2026-10-10) and in [agreement](../agreement.md).

## v0.2-gold (2026-09-24)

The first labeled set: 114 questions selected from `v0.1.6-candidates`, 90 of them with gold chunks, 260 gold chunks in all, from round 1 and its review. `v0.2-gold.jsonl` is a byte-for-byte copy of `evals/questions.jsonl` at freeze (sha256 `df27874b…`, in full as `questions_sha256` in `results/20260924T234623Z/index.json`, checked with `cmp`); it loads cleanly under the dataset schema. All judgments were made against `index_version` `2825a7160c4d09bf` (manifest `c419b4be…`, in full in `v0.2-gold.meta.json`, 11,934 chunks).

Composition. 114 questions: comparison 50, unanswerable 20, adversarial 13, fact 8, ambiguous 6, noisy 6, version_sensitive 6, multihop 5. 94 answerable, 20 unanswerable. The 4 answerable questions without gold were judged "nothing in the pool answers this". Round 1 judged 3 of them so, because their source chunks never reached the pool: `cv-0067 "bybit caps limit at 1000, how"`, `cv-0088 "binance postOnly timeInForce isnt working, what"`, `cv-0090 "my hyperliquid api wallet address returns"`. For `cv-0132 "is switch-isolated still how i change"`, review removed its only chunk, the deprecated `docs/v5/abandon/` page.

Selection (`evals/selection.jsonl`, one decision per id). 117 of the 204 candidates were reviewed: 114 accepted, 1 rejected (`cv-0139 "bingx websoket close always, need send"`, `unrealistic phrasing`) and 2 skipped without a reason, `cv-0180 "who is supposed to send the"` and `cv-0181 "which id do i pass to"`. The other 87 were never shown, because the category targets were met first (fact 34, comparison 12, adversarial 11, multihop 9, version_sensitive 7, noisy 6, ambiguous 5, unanswerable 3). With 1 rejection among 117 candidates shown, the set is close to the order of acceptance, and the selection is no evidence of a strict filter.

Round-1 quality checks, before the review (114 judged, 89 with gold, 266 gold chunks):

- 5 answerable questions got "none relevant"; in all 5 no source chunk reached the pool: all 5 are pool misses, none a misclassification.
- 23 labeled comparisons had gold missing one of the named venues: 3 with no candidate from that venue in the pool, 20 with candidates shown and none marked. The 20 went to review.
- Unanswerable questions with gold: 0.
- Adversarial / version_sensitive: no marked chunk from the wrong venue. By reading, 3 marked a chunk from the environment or version the notes call the trap or covered only one side (`cv-0093 "binance demo trading api, is that"`, `cv-0129 "can i reuse my spot api"`, `cv-0130 "the error strings changed between uniswap"`), plus `cv-0132 "is switch-isolated still how i change"`. All four went to review.
- Time per question (gap between consecutive saves, one session break excluded): median 17 s, minimum 11 s, none under 5 s; fact median 87 s. The median gap of 17 s is 0.7 s per candidate at 25 candidates per question. The second session saved 113 questions over 3,325 s, a mean of 29.7 s between consecutive saves and 1.2 s per candidate. Both are derived from the round-1 `annotated_at` timestamps in the journal, and the speed is the main caveat on these labels. Round 2 measured their agreement at κ 0.470 [0.317, 0.645] (95% percentile bootstrap over questions; [agreement](../agreement.md)).
- Source chunk outside the pool: 25 of the 94 questions that have source chunks (26.6%); comparison 18 of 50 (36.0%).

Review (`crossvenue label --review`, corrections to round 1, journaled under `corrections` with the previous set). 26 questions reviewed: the 24 of `evals/review_round1.txt` plus `cv-0144 "can i move the price of"` and `cv-0166 "hyperliquid never sends me a 429,"`. 11 got changed labels: 3 chunks added, 9 removed (266 to 260); the other 15 changed only `venue_coverage`.

`venue_coverage`. Every labeled answerable comparison records one value per venue it names. The value is `covered` when the gold set covers the venue, `absent_in_docs` when the venue's documentation does not answer it, and `missing_from_pool` when no suitable candidate of the venue was in the pool. 101 values on 51 questions: covered 77, missing_from_pool 22, absent_in_docs 2.

- Derived mechanically: 57 values on 29 questions (`crossvenue derive-coverage --exclude-file evals/review_round1.txt`, 2026-09-24), by one rule over data round 1 had already recorded. The rule sets `covered` where a gold chunk is from the venue (53 values). It sets `missing_from_pool` where round 1's recorded pool holds no candidate from the venue at all (4 values: `cv-0049 "is permit2 deployed at the same"` pancakeswap, `cv-0058 "bybit hands me a cursor, does"` bingx, `cv-0142 "binance stp wants EXPIRE_MAKER, whats the"` bybit, `cv-0166 "hyperliquid never sends me a 429,"` binance). 25 of the 29 were covered on every venue. `absent_in_docs` is never derived. Each derived value is journaled in round 1's `corrections` with `derived: true`, the rule and the previous (empty) value, and no annotator.
- The annotator set 44 values on 23 questions during the review. They cover the 21 comparison-typed questions of `evals/review_round1.txt`, both venues of `cv-0144 "can i move the price of"`, and hyperliquid on `cv-0166 "hyperliquid never sends me a 429,"`. The values are 24 `covered`, 18 `missing_from_pool`, 2 `absent_in_docs` (`cv-0153 "bybit cancels my orders when i"` bingx, `cv-0162 "can i get an aggregated book"` dydx).
- Consequence for metrics: 22 of the 50 comparison questions are one-sided because of the pool, each with one venue recorded `missing_from_pool`; a gap in the documentation (`absent_in_docs`) accounts for 2 values. On those, recall is measured against a gold set known to be incomplete, and "every named venue in the top 10" cannot be credited by the gold labels alone.

Validated with the encoder-backed near-duplicate check: `ok`; one warning, venue coverage lopsided (binance 42 vs ccxt 6), known since selection.

Judgment history (added 2026-09-25, before any augmentation). `annotations/v0.2-gold.rounds.json` (sha256 `9bd7ba7b713ed4bd8f59e36cced986305514318065e70056eb2e70a37387f6a1`) is a byte copy of `evals/annotations/questions.rounds.json` at this version (every question's `relevant` and `venue_coverage` checked equal to the dataset). It records which chunks round 1 judged, so `unjudged@k` can be measured against this version. The dataset file itself is unchanged.

## v0.1.6-candidates (2026-09-24)

Two of the 204 records differ, and only in their `notes` field: `cv-0091 "on binance spot testnet how long"` and `cv-0128 "is the ip ban on binance"`. Same questions, ids and order as `v0.1.5-candidates`; all earlier versions stay on disk unchanged.

- Why. The two were written for a trap exact deduplication removed: two byte-identical chunks, `rest-api.md` and `testnet/rest-api.md`, told apart only by `doc_path`. Their source is now one chunk (`9372e7af283d5470`) whose `sources` are the mainnet page (primary) and the testnet page. The answer text, "from 2 minutes to 3 days", occurs in exactly 2 Binance chunks (`crossvenue count-term`, chunks, case-insensitive), `rest-api.md` and `web-socket-api.md`. Both carry a mainnet and a testnet occurrence, so no testnet-only twin is left anywhere.
- Not changed: category (`adversarial` / `version_sensitive`), `adversarial_type`, question text, grounding. Whether the two still belong in their categories is a selection decision.
- Verification. Line-by-line against v0.1.5: exactly the two lines differ, and in them only `notes`; the same two notes were synchronized into `evals/questions.jsonl`, checked the same way.
- `corpus_snapshot` and `index_version` `2825a7160c4d09bf` unchanged.

## v0.1.5-candidates (2026-09-24)

Only `source_chunk_ids` changed, mechanically, after exact deduplication: 181 of the 204 records rewritten, 354 source ids became 350. Same questions, ids and order as `v0.1.4-candidates`; all earlier versions stay on disk unchanged.

- Why. The corpus is now exactly deduplicated ([design, exact deduplication](../../docs/explanation/design.md#exact-deduplication)): byte-identical chunks within a venue are stored once. A chunk id is a hash of venue + text instead of venue + document + position + text. Every chunk id in the corpus changed, so every pointer into it had to be carried across.
- How. `crossvenue id-map` mapped each of the 12,783 pre-deduplication chunk ids onto the canonical chunk with the same venue and text (11,934 canonical ids, 0 old ids without a counterpart). It wrote the map to `evals/migrations/2026-09-24-exact-dedup.idmap.json`, which records the sha256 of the old `chunks.jsonl`, `c4dab899…`, and the target `index_version` `2825a7160c4d09bf`. `crossvenue remap-ids` rewrote the id lists through it.
- Verification. Per record, the rewritten line with the old `source_chunk_ids` / `gold_chunk_ids` put back is byte-identical to the v0.1.4 line; the tool refuses to write otherwise, and an independent script re-checked all 204. The 23 records not rewritten have no source chunks.
- Four records lost a source id because both of their source ids pointed at byte-identical copies, which are now one chunk:
  - `cv-0004 "bybit sends 403 access too frequent,"` and `cv-0094 "bybit p2p says 600 per 5"`: the IP-limit paragraph shared verbatim by `docs/v5/rate-limit/rate-limit.mdx` and `docs/p2p/rate-limit.mdx`. The canonical chunk cites the p2p page (primary-source rule: lexicographic doc path), and keeps the v5 page in `sources`.
  - `cv-0091 "on binance spot testnet how long"` and `cv-0128 "is the ip ban on binance"`: `rest-api.md` and `testnet/rest-api.md`. The canonical chunk cites `rest-api.md` and keeps the testnet page in `sources`.
- The `notes` field of `cv-0091 "on binance spot testnet how long"` and `cv-0128 "is the ip ban on binance"` describes a corpus that no longer exists. After deduplication there is one chunk whose `sources` name both environments, so a retriever can no longer return the chunk of the other environment. The trap these two questions set is gone, and the environment distinction moves to the citation. The notes are not edited here (this version changes ids only); whether the two stay in the set is a selection decision, to be recorded in `evals/selection.jsonl`.
- `corpus_snapshot` `c419b4be…` (in full in `v0.1.5-candidates.meta.json`), 1,665 documents, unchanged; `index_version` `2825a7160c4d09bf` added to it.

## v0.1.4-candidates (2026-09-20)

Four of the 204 records differ, and only in their `notes` field: 4 of the 75 checkable term counters quoted as evidence were wrong and are recounted. Same questions, ids and order as `v0.1.3-candidates`; all earlier versions stay on disk unchanged.

- Why. The counters had been produced by two different counting conventions. Recounted under the project's single convention (chunks containing the term, case-insensitive, over `data/processed/chunks.jsonl`, now `crossvenue count-term`):

  | record | term | scope | was | is |
  |---|---|---|---:|---:|
  | `cv-0112 "which venues here accept GTX as"` | `GTX` | corpus-wide | 0 | 2 |
  | `cv-0202 "which dydx flag stops my own"` | `selfTradePrevention` | binance | 157 | 176 |
  | `cv-0204 "where is the FIX session config"` | `FIX API` | binance | 252 | 150 |
  | `cv-0205 "what does a uniswap swapping api"` | `pricing` | uniswap | 48 | 40 |

  Two were case sensitivity (157 to 176, 0 to 2), one was occurrences counted instead of chunks (252 to 150), and one is unexplained (48 to 40).
- Note (added 2026-09-29). The "is" column was counted on the corpus before exact deduplication (`v0.1.5`), where a chunk merged later still counts once per copy. On the deduplicated corpus `crossvenue count-term` gives `selfTradePrevention` 128 and `FIX API` 109 for binance; `GTX` (2) and `pricing` (40) are unchanged. The records keep the counts of this version.
- `cv-0112 "which venues here accept GTX as"` is still unanswerable, and the recount is what proves it. Both `GTX` hits are substring accidents inside the `clientOrderId` `CUhLgTXnX5n2c0gWiLpV4d` in two Binance websocket examples; case-sensitively the count is 0. The recount, its convention and the reason the hits do not count are in its `notes` field.
- `cv-0202 "which dydx flag stops my own"` and `cv-0204 "where is the FIX session config"` also had a hand-classified `STP` / `FIX` count, recounted the same way in their `notes` field: 5 and 19 case-insensitive hits, all substring accidents.
- Nothing else changed. No question text, category, venue, grounding, answerability or id. Gold labels collected against `v0.1.3` remain valid; a `diff` against it touches exactly four lines.
- `corpus_snapshot` `c419b4be…` (in full in `v0.1.4-candidates.meta.json`), 1,665 documents, unchanged.
- Validated with the encoder-backed near-duplicate check enabled: `ok`, no warnings.

## v0.1.3-candidates (2026-09-20)

26 records added at `cv-0180` … `cv-0205`: 7 ambiguous, 7 noisy, 6 version_sensitive and 6 unanswerable. All 178 v0.1.2 records are carried through byte-for-byte, in the same order, with the same ids and no re-serialization; nothing was dropped or edited. Totals go 178 to 204. `v0.1.2-candidates`, `v0.1.1-candidates` and `v0.1-candidates` stay on disk unchanged.

- Why. Against the [methodology §7](../methodology.md#7-set-size-and-statistical-power) targets of 5 · 5 · 5 · 15, the pool held ambiguous 6, noisy 6, version_sensitive 7 and unanswerable 17. Selection would therefore have had to accept 5 of 6 ambiguous, 5 of 6 noisy, 5 of 7 version_sensitive and 15 of 17 unanswerable candidates. The 26 raise them to 13, 13, 13 and 23.
- Venue spread of the 26, counting a record once per venue in its `venues` list: bingx 6, bybit 6, dydx 6, hyperliquid 6, pancakeswap 6, uniswap 6, binance 4, ccxt 2; no venue above 6 of 26 (23.1%).
- The reason for each record is in its `notes` field (26 of 26 carry one). For the 7 ambiguous records it gives the venues and the different answer each gives, and for the 7 noisy records the kind of noise (one kind per record). For the 6 version_sensitive records it gives the two versions or environments, both grounded. For the 6 unanswerable records it gives the concept, searched under every spelling the vendor uses, restricted to the target venue and corpus-wide. It also gives the counts and the near miss that a retriever returns instead.
- Five drafts were dropped after the search showed the corpus answers them: a PancakeSwap Permit2 address, a BingX testnet environment, a Hyperliquid minimum order value, a Bybit index-price-constituents item and a Bybit websocket `args` cap.
- Grounding. All 19 answerable added records carry `source_chunk_ids`; the 7 ambiguous ones cite at least one chunk per venue. All 63 pointers resolve against `data/processed/chunks.jsonl` at the recorded snapshot. They remain provenance; gold labels come only from judging. The 6 unanswerable records carry empty `source_chunk_ids` and `gold_chunk_ids`; `reference_answer`, the annotation stamps and `adversarial_type` are empty or null on all 26, and no added record cross-references another question id.
- Note (added 2026-09-30). The grounding line above counts 19 answerable added records; 26 added records less the 6 unanswerable ones is 20.
- `corpus_snapshot` in the meta was recomputed with `crossvenue_rag.dataset.manifest_fingerprint("data/manifest.csv")` and still matches v0.1/v0.1.1/v0.1.2 (`c419b4be…`, in full in `v0.1.3-candidates.meta.json`, 1,665 documents); `counts` were recomputed from the finished file.
- `prompt_hash` on all 26 is `df6de014544b2c1c`, `dataset.prompt_hash()` over the v0.1.3 headroom brief quoted in the `.meta.json` `notes`.
- Validation. `crossvenue validate-evals -d evals/datasets/v0.1.3-candidates.jsonl`, with the encoder-backed near-duplicate check at cosine 0.92 over all 204 records: `ok`, no warnings and no near-duplicate pair. An earlier draft raised the warning `venue coverage is lopsided: binance=56 vs ccxt=13`; `ccxt` was added to the `venues` of `cv-0181 "which id do i pass to"` and `cv-0185 "where do i read my own"`, where it belongs: the ccxt manual documents `clientOrderId` support as exchange-dependent and has a unified fee section.

## v0.1.2-candidates (2026-09-20)

38 comparison candidates added at `cv-0142` … `cv-0179`. All 140 v0.1.1 records are carried through byte-for-byte, in the same order, with the same ids and no re-serialization; nothing was dropped or edited. Totals go 140 to 178, comparison 24 to 62. `v0.1.1-candidates` and `v0.1-candidates` stay on disk unchanged.

- Why. [Methodology §7](../methodology.md#7-set-size-and-statistical-power) was retargeted from 35 comparison questions to 50, because the headline claim (cross-venue confusion) is measured in that subgroup. At ψ = 0.25 a subgroup of 35 detects a recall@10 gap of 0.24 at 80% power, while 50 detects 0.20. v0.1.1 held 24 comparison candidates; 62 leaves a 50-question selection 12 spare.
- Pair coverage of the 38. By venue type: CEX-CEX 12, DEX-DEX 10, CEX-DEX 16, of which two pair a DEX with `ccxt` (14 with `ccxt` not counted as a venue; `validate-evals` counts it). By pair: binance×bybit 5, uniswap×pancakeswap 5, binance×bingx 4, binance×hyperliquid 4, dydx×hyperliquid 4, bybit×bingx 3, bybit×hyperliquid 3, bybit×dydx 2, bingx×hyperliquid 2, binance×dydx 1, bingx×dydx 1, uniswap×hyperliquid 1, bybit×pancakeswap 1, ccxt×hyperliquid 1, ccxt×dydx 1. All eight venues appear; ccxt, a client library, in 2 of 38.
- Information needs, split 10 / 17 / 11: numeric divergence under a shared term, the same name for a different mechanism, and a capability present on one venue and absent on the other. Each record's `notes` (38 of 38 carry one) names its need and its confusable venue, and for the 11 capability records the absence counts under every spelling the vendor uses.
- Grounding. Every added record has `source_chunk_ids` filled with at least one chunk per venue, because a comparison needs evidence on both sides; all 76 pointers resolve against `data/processed/chunks.jsonl` at the recorded snapshot. They remain provenance; gold labels come only from judging. `gold_chunk_ids`, `reference_answer` and the annotation stamps are empty by design; `adversarial_type` is null on all 38. No added record cross-references another question id.
- `corpus_snapshot` in the meta was recomputed with `crossvenue_rag.dataset.manifest_fingerprint("data/manifest.csv")` and still matches v0.1/v0.1.1 (`c419b4be…`, in full in `v0.1.2-candidates.meta.json`, 1,665 documents); `counts` were recomputed from the file.
- `prompt_hash` on all 38 is `e1e2519d848abd35`, `dataset.prompt_hash()` over the v0.1.2 comparison-supplement brief quoted in the `.meta.json` `notes`.
- Validation. `crossvenue validate-evals -d evals/datasets/v0.1.2-candidates.jsonl`, with the encoder-backed near-duplicate check at cosine 0.92: `ok`, no warnings and no near-duplicate pair. An earlier draft raised the warning `venue coverage is lopsided: binance=53 vs ccxt=13`; its binance×uniswap item was rewritten as `cv-0177 "bybit takes a slippageTolerance on market"`, which drops binance to 52 and gives pancakeswap its only CEX-DEX question.

## v0.1.1-candidates (2026-09-20)

One of the 140 records replaced: `cv-0110 "what tick size should i round"` dropped, `cv-0141 "im deriving pancakeswap v3 pool addresses"` added; still 140 questions, 17 of them unanswerable. `v0.1-candidates` stays on disk unchanged.

- Dropped `cv-0110 "what tick size should i round"`, for two reasons. It was mislabeled: Hyperliquid price rounding is documented. `tickSize` has zero occurrences in Hyperliquid chunks, but `szDecimals` / `nSigFigs` / "significant figures" appear in 29 of them, so the question is answerable and the answer is a vocabulary correction. The 29 is the union of the three spellings, counted by a script; `crossvenue count-term`, which counts one term, gives `szDecimals` 26, `nSigFigs` 3 and "significant figures" 3 in Hyperliquid chunks. And its `question` duplicated that of `cv-0101 "what tick size does hyperliquid use"` at 0.926 cosine, above the project's 0.92 near-duplicate threshold; `cv-0101` already tests that vocabulary correction as an adversarial item with `expected_behavior: answer`.
- Added `cv-0141 "im deriving pancakeswap v3 pool addresses"`, whose record carries its `category`, `venues` and `difficulty`: a different venue and a different theme from `cv-0110`, chosen so that no more unanswerable questions name Binance.
- Gap evidence (searched by concept, under every spelling): across `pancakeswap` chunks, `CREATE2`, `keccak`, `salt`, `bytecode`, `encodePacked`, `deployer` and `getPool` all return 0, and every spelling of the constant (`init code hash`, `INIT_CODE_HASH`, `initCodeHash`, `init_code`, `POOL_INIT`, `creation code`, `bytecode hash`) returns 0 anywhere outside `uniswap`. PancakeSwap publishes the `PancakeV3Factory` address and one code comment naming `computePoolAddress()` (an on-chain lookup, no hash). Uniswap publishes a full CREATE2 recipe with a literal `keccak256(init_code)`, for Uniswap v2, the wrong protocol version and the wrong fork.
- The other 139 records are carried over byte-for-byte: same ids, same field values, no renumbering. `cv-0141` takes the next free id and leaves `cv-0110`'s unused.
- `corpus_snapshot` in the meta was recomputed with `crossvenue_rag.dataset.manifest_fingerprint("data/manifest.csv")` and matches v0.1 (`c419b4be…`, in full in `v0.1.1-candidates.meta.json`, 1,665 documents); `counts` were recomputed from the file.
- `cv-0141.prompt_hash` (`62ea8a5c438d1ba6`) is `dataset.prompt_hash()` over the replacement brief, whose exact text is `v0.1.1 replacement brief: one unanswerable candidate for the cv-0110 slot; different venue and theme from Hyperliquid tick size; corpus gap proven against data/processed/chunks.jsonl across every plausible spelling before the question was written.`

## v0.1-candidates (2026-09-20)

The generated candidate pool, before human selection: 140 questions (fact 42, adversarial 24, comparison 24, unanswerable 17, multihop 14, version_sensitive 7, ambiguous 6, noisy 6). Not an evaluation set: it has no gold labels and has not been filtered for quality.

- Built against the corpus snapshot recorded in `v0.1-candidates.meta.json`.
- Questions are grounded in real corpus chunks and in recurring themes found in exchange SDK issue trackers ([domain research](../domain-research.md), [methodology](../methodology.md)).
- `source_chunk_ids` are provenance pointers to the chunks a question was written from. They are not gold labels and must not be used as such. They record where a question came from, and let the annotator notice when the candidate pool did not contain the chunk it was written from.

Known caveats at this version:

- unlabeled, so no retrieval metric can be computed from it;
- `chunk_id` values move whenever the chunking config changes, so `source_chunk_ids` are valid only against the recorded corpus snapshot;
- generated questions are a starting point for the real user queries discussed in [methodology](../methodology.md), and no substitute for them.
