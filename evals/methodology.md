# Evaluation methodology

How the labeled question set for `crossvenue-rag` is built, why it is built that way, and what it is and is not allowed to prove. What the resulting set contains is in [datasheet](datasheet.md); how to run each step is in [labeling a question set](../docs/how-to/label.md).

The corpus is 11,934 chunks ([`docs/reference/corpus.md`](../docs/reference/corpus.md)) of API documentation from seven trading venues and the CCXT library. Their pages describe the same kinds of object (orders, rate limits, funding, fee tiers), where a source documents them, with different numbers underneath. That makes the evaluation set as much a part of the experiment as the retriever: a question set whose wording follows the chunks measures vocabulary overlap and reports it as retrieval quality.

Every decision below is stated with its reason and, where a number is involved, with the number. Sections 1–7 cover general practice; the domain research the questions are drawn from is in [domain research](domain-research.md) (§8). Where the set as built departs from this protocol, the departure is recorded in [Deviations from protocol](#deviations-from-protocol), with its reason.

## Decisions at a glance

| # | Decision | Value | Why |
|---|---|---|---|
| 1 | Labeled set size | 105 questions, hard floor 85, sized from the subgroups: comparison ≥ 50 (all answerable), unanswerable ≥ 15, nothing below 5 | The headline measurement is cross-venue confusion, which is measured in the comparison subgroup, so the subgroup sets the size and the total is whatever the subgroups add up to. At n = 50 the smallest detectable δ is 0.20; at 35 it is 0.24 (§7) |
| 2 | Category split | comparison 50 · unanswerable 15 · adversarial 12 · fact 8 · multihop 5 · ambiguous 5 · version_sensitive 5 · noisy 5 = 105 | The three constraints (comparison 50, unanswerable 15, nothing below 5) do not fit in 100; 105 is what they sum to. A total of 100 would cut a category below the floor at which `validate-evals` stops calling a breakdown row noise. Comparison is 47.6% of the set because it carries the headline claim and is the only subgroup sized for a significance test. Everything else is reported descriptively |
| 3 | Venue coverage | every venue is the answer venue for ≥ 8 questions; ≥ 45% DEX | Prevents a CEX-shaped set that never exercises the Uniswap/PancakeSwap trap |
| 4 | Question source | Grounded in real developer questions ([domain research](domain-research.md)), hand-written or heavily rewritten, never a raw single-chunk LLM paraphrase | Single-chunk generated questions restate the chunk and inflate retrievability |
| 5 | Pooling | Depth 20 from each of the five pool methods (bm25, dense, hybrid, hybrid_rerank, venue_split; not `venue_split_oracle`, which reads the labeled `venues`), fused with RRF, top 25 shown to the annotator; `eval.pool_depth_per_method` / `eval.pool_display` (§1) | Single-retriever labeling makes that retriever unbeatable by construction. Depth must exceed `max(k_values)`=10, which a config validator enforces ([ADR 0005](../docs/adr/0005-labeling-pool.md)). The display cap, not the depth, sets how much of each pool method's top 10 is judged. Round 1 judged the fused top 25; after the augmentation, each round-1 pool method's unjudged@10 on the 50 comparison questions is 0.032–0.088. Augmented methods are judged to depth 20. The unjudged@10 range is from `results/20260925T012314Z`, on `v0.2.1-gold`, whose comparison gold was augmented after results were seen ([D4](#deviations-from-protocol)) |
| 6 | Pooling-bias disclosure | Every `results/*/report.md` states each method's `unjudged@k` (share of its top k never judged). Recall is reported as recall-within-pool. A method not in the pool has its unjudged top 20 judged (augmentation) before its recall is compared (§1) | The bias cannot be removed at this labeling budget, so the unjudged share is stated with every result |
| 7 | Intra-rater reliability | 25 questions re-annotated, stratified by category and fixed in advance in `evals/selection/round2_sample.txt`, after a ≥ 14-day gap. Round 2 was blind, on exactly the candidates round 1 judged (recorded pool plus augmentation), shuffled with a recorded seed, with no marks and no "found by". Cohen's κ per candidate-level decision, target κ ≥ 0.61. Round 1 and its augmentation are dated 2026-09-24 local time (UTC−3), part of it 2026-09-25 in the journal's UTC timestamps. Round 2 ran on 2026-10-10 for the whole subsample, at least 15 days after the last round-1 judgment by either date; κ is 0.470 [0.317, 0.645] (95% percentile bootstrap over questions; [agreement](agreement.md)). A 7-day fallback is defined in §5 as a declared deviation ([ADR 0006](../docs/adr/0006-single-annotator.md)) | Only one annotator exists, so test-retest is the only agreement number available |
| 8 | Question dedup threshold | cosine ≥ 0.92 on `bge-base-en-v1.5` question embeddings flags a near-duplicate pair for manual merge; `eval.duplicate_threshold: 0.92` in `configs/default.yaml` | Chroma found naive generation producing identical and near-identical query pairs above 0.9 cosine. A threshold below that would delete legitimate venue-parallel questions. The expectation (§6) is that 0.92 catches restatements while keeping "…on Bybit?" / "…on Binance?" pairs |
| 9 | Significance testing | Paired bootstrap over questions (10,000 resamples, seed 20260924) for intervals; paired randomization test (sign flip, 10,000 permutations, same seed) for p-values; exact McNemar on discordant questions for binary per-question outcomes; one primary comparison, venue_split − hybrid_rerank on venue_coverage@10 over the comparison questions; every other comparison secondary, Holm-adjusted within its metric and scope; effect size and CI reported, never a bare p-value | Randomization/bootstrap/t agree in IR; sign and Wilcoxon are discouraged |
| 10 | Metric under incompleteness | recall@k/nDCG@k on the pooled qrels, reported with `unjudged@k` per method and condensed-list nDCG′@k | Unjudged-as-irrelevant is the standard bias; disclosing coverage is the cheap mitigation |
| 11 | Answer evaluation | Four generator configurations. Primary comparison: gemini@venue_split − gemini@hybrid_rerank on venue_attribution_error and coverage over comparison questions; secondary pairs Holm-adjusted. A judge dimension is used only if its agreement with human ratings reaches κ ≥ 0.61 (Spearman ≥ 0.70 for coverage) | Registered before any answer was generated (§9) |
| 12 | Full relabel | Annotation round 3 over all 114 questions, blind, on the recorded candidates, under the criterion as amended; `v0.3-gold` takes its labels from it alone. No round 4 ([§5b](#5b-annotation-round-3-pre-registered)) | Every label of a set follows one criterion. Registered before the first judgment of the round |

## 1. Test collection construction and pooling

### Pool formation

The Cranfield/TREC model does not judge the whole collection. Participants submit ranked runs, NIST takes the union of the top *k* documents from each judged run (the pool), and only the pool is assessed. Everything outside it is treated as non-relevant by the scoring measures. TREC's own description of relevance judgments says only that "judging is done using a pooling technique" and points at the track overviews for the depth ([TREC: English relevance judgments](https://trec.nist.gov/data/reljudge_eng.html)). Depth 100 became the ad-hoc default. The [TREC 2017 Common Core track overview](https://trec.nist.gov/pubs/trec26/papers/Overview-CC.pdf) is explicit that it was departing from depth-*k* pooling. It judged depth-10 pools, then chose further documents to depth 100 with a bandit method, and judged a 20% sample of ranks 11–75 for inferred measures.

### Pooling bias and its handling in the literature

Two results define the boundaries.

[Zobel (1998), "How reliable are the results of large-scale information retrieval experiments?"](https://doi.org/10.1145/290941.291014) scored each run with the official qrels and again with the relevant documents only that run contributed removed. The check was later generalized to leaving out a whole participating group. His conclusion is the one the field still quotes: relative system comparisons hold under pooling, but recall is overestimated, because relevant documents that no pooled system retrieved were never found. He also found collection quality degraded noticeably when pools were downsampled to depth 10 but not at greater depths.

[Buckley, Dimmick, Soboroff & Voorhees (2007), "Bias and the limits of pooling for large collections"](https://doi.org/10.1007/s10791-007-9032-x) (Information Retrieval 10(6):491–508) showed the limit case. A pool too small for a very large collection is biased toward documents containing the topic title's words, and leaves enough unjudged relevant material that runs outside the pool are scored unfairly. [Büttcher, Clarke, Yeung & Soboroff (2007)](https://doi.org/10.1145/1277741.1277755) made the same point empirically on TREC Terabyte and proposed rebuilding an unbiased judgment set from the biased one.

The counterweight is [Voorhees, Soboroff & Lin (2022), "Can Old TREC Collections Reliably Evaluate Modern Neural Retrieval Models?"](https://arxiv.org/abs/2201.11086). They re-judged TREC-8 against modern neural runs and found that those runs retrieved documents outside the original pool. The re-judging pools, 9 to 359 documents per topic over 50 topics, held 158 relevant documents in total, no run retrieved more than 23 of them, and system rankings were unchanged (Kendall's τ > 0.99). Their collection had a deep pool over a diverse set of runs; the pool here is shallow and narrow, so their result does not carry over.

Measure-level mitigations exist. [Buckley & Voorhees (2004), "Retrieval evaluation with incomplete information"](https://doi.org/10.1145/1008992.1009000) ([NIST copy](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469)) introduced bpref, which scores only judged documents and is therefore insensitive to how unjudged ones are ranked. [Sakai (2007), "Alternatives to Bpref"](https://doi.org/10.1145/1277741.1277756) argued that a simpler fix is better: compute AP, Q-measure or nDCG over a condensed list with unjudged documents removed (AP′, Q′, nDCG′). Sakai also argued that graded relevance further improves robustness to incompleteness.

### Decision for this project

The pool is depth 20 across five methods (bm25, dense, hybrid, hybrid_rerank and venue_split, `retrieval.methods.POOL_METHODS`), fused with RRF, because a ground truth labeled from one retriever's output holds only the relevant chunks that retriever finds. `venue_split_oracle` is not a pool method: it reads the question's labeled `venues`, and a pool built from the labels it is judged against would be circular.

Five systems is still a thin pool, and three of them (hybrid, hybrid_rerank, venue_split) are built from the other two, so their top 20 is expected to overlap heavily with those of bm25 and dense. No saved run holds the five methods' top 20, so the overlap is not measured. Three rules follow:

1. Recall is reported and named as recall-within-pool. It bounds true recall in neither direction. A relevant chunk no pooled method found is missing from the denominator and raises it; an unjudged relevant chunk in the top k counts as a miss and lowers it. How much the gold chunks of `v0.2.1-gold` change every method's comparison recall@10 is a result in [findings](../docs/results/findings.md#adding-17-gold-chunks-changes-every-methods-comparison-recall10-by-0034-to-0092).
2. Every run's `report.md` records `unjudged@k` per method: the share of each method's top *k* no annotator ever judged. Each such chunk counts as a miss, so a method with a high value may be scored pessimistically; the share does not say by how much.
3. A retrieval method that is not in the pool a question is labeled from has its never-judged top 20 on that question judged before its recall is compared. This is the leave-one-group-out discipline applied forward instead of backward. Depth is the pool depth, not the top 10 the metrics score, so the judged set extends past the ranks being scored for that method too.

Rule 3 is carried out as an augmentation of the round the question is labeled in. A retrieval-only pool-construction run (`crossvenue pool-run`, k = 20, no metrics, never quoted as an evaluation) produces the candidates; `crossvenue augment-candidates` keeps the never-judged ones; `crossvenue label --augment` shows only those, with the current gold as context. Every chunk shown is recorded, relevant ones are appended to gold, nothing is removed. What the augmentation of `v0.2.1-gold` judged and found is in [datasheet](datasheet.md#labeling).

bpref is not the headline metric. With binary relevance over 277 gold chunks on 90 labeled questions, bpref is expected to be noisy. No run computes bpref; its bootstrap interval beside that of nDCG′@10 on the same run would test the expectation. Condensed-list nDCG′@10 is reported alongside nDCG@10. Their gap is the change when never-judged chunks are dropped from the ranking instead of counted as irrelevant. A relevant chunk that no pooled method retrieved is never judged, so neither metric counts it as relevant. It is `ndcg_prime@k` in every `report.md` (`evaluation.metrics.ndcg_condensed_at_k`). One difference from Sakai (2007): the harness retrieves max(k) = 10 deep, so the condensed list is the top 10 with never-judged chunks removed; no rank below 10 is retrieved, so nothing refills it.

## 2. Synthetic question generation, used as a drafting aid only

### The standard recipe

The recipe takes a chunk, asks an LLM to write a question the chunk answers, and keeps the chunk as the gold label. [Ragas](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/) implements this pattern as a pipeline, in which documents become a knowledge graph, transformations enrich it (headline and keyphrase extraction), and synthesizers emit single-hop and multi-hop queries against a configurable distribution. [ARES](https://arxiv.org/abs/2311.09476) (Saad-Falcon et al., NAACL 2024) uses the same generation step to generate training triples for lightweight LLM judges, correcting them with a few hundred human annotations via prediction-powered inference. [BEIR](https://arxiv.org/abs/2104.08663) took the opposite route: rather than generating, it assembled 18 existing human-built retrieval datasets, so that the queries came from somewhere other than the corpus.

### Weaknesses of single-chunk generated questions

Four separate failure modes, each with evidence:

Lexical shortcut. A question written from a chunk inherits the chunk's vocabulary, so retrieving it is a term-matching exercise rather than a retrieval one. Chroma's [Generative Benchmarking](https://www.trychroma.com/research/generative-benchmarking) report (Hong, Troynikov, Huber & McGuire, April 2025) measured the extreme form. Prompted naively, LLMs reproduced existing benchmark queries verbatim or with minimal rewording, with 11.91% of query pairs above 0.9 cosine similarity on their Wikipedia set. Their fix (feeding ground-truth queries as negative examples) lowered the mean cosine between a generated query and the ground-truth query it was matched with from 0.716 to 0.628, while preserving model rankings.

Distribution mismatch. [Kucia & Gawlik (2026), "Beyond Benchmark Scores"](https://arxiv.org/abs/2609.14579) compared 1,851 synthetic questions against 322 authentic queries collected from students for the same university information system. Authentic queries averaged 6.8 words vs 15.7 for synthetic ones and drew on 53 unique sources vs 165. Real traffic concentrates; generated traffic spreads evenly over the corpus. For choosing a retriever they report: synthetic benchmarks inverted retrieval-system selection, favoring a hybrid retriever whose sparse component scored well on long synthetic questions and did not help on short real ones.

Generator/stylistic bias. [van Elburg, van der Putten & Marx (2025), "Can we Evaluate RAGs with Synthetic Data?"](https://arxiv.org/abs/2508.11758) found synthetic benchmarks ranked retriever configurations reliably but failed to rank generator architectures consistently. They attributed this to task mismatch plus stylistic bias toward the generating model. [Rahmani, Ramineni, Yilmaz, Craswell & Mitra (2025), "Towards Understanding Bias in Synthetic Data for Evaluation"](https://arxiv.org/abs/2506.10301) (CIKM) corroborates the length effect and finds real queries significantly more likely to begin with "what".

Diversity collapse. [DataMorgana](https://arxiv.org/abs/2501.12789) (Filice et al., 2025) exists because naive generation produces homogeneous sets. Its authors also warn that the usual diversity metrics are themselves misleading, since a set of simulated expert questions scored more diverse on every metric they tried than a set of non-expert ones.

### Decision for this project

Generation is used as a drafting aid, never as the source of the question. Concretely:

- Question topics come from [domain research](domain-research.md): real developer questions mined from GitHub issues and from the corpus's own FAQ/troubleshooting pages. The information need is real before a word is written.
- A question is written from the need and then checked against the corpus; no question is written from a chunk. Where an LLM drafts a phrasing, the draft is rewritten to the vocabulary of the issue reporter, which differs from the vocabulary of the docs.
- Hard constraint: no question may share a distinctive multi-word phrase with its gold chunk unless that phrase is an API identifier a developer would type (`recvWindow`, `LIMIT_MAKER`, `X-MBX-USED-WEIGHT`). Identifiers are legitimate query terms; prose lifted from a heading is not.
- The target median question length is in the real-query band (8–14 words for fact questions), below the 15–20-word synthetic band. Kucia & Gawlik found that the sparse half of a hybrid retriever helped long synthetic questions and did not help on short authentic ones, and the BM25-versus-dense comparison is one this repository makes.
- Gold labels are assigned by pooled retrieval + human judgment; the chunk a question was generated from is gold only when that judgment marks it. A question generated from chunk X can have better answers at chunks Y and Z.

## 3. Unanswerable and negative examples

### The SQuAD 2.0 tradition

[Rajpurkar, Jia & Liang (2018), "Know What You Don't Know: Unanswerable Questions for SQuAD"](https://arxiv.org/abs/1806.03822) added over 50,000 unanswerable questions, which crowdworkers wrote adversarially to look answerable. The writers referenced entities in the paragraph and ensured a plausible but wrong answer is present. A system at 86% F1 on SQuAD 1.1 dropped to 66% F1 on 2.0, against a human ceiling of 89.5 F1.

The proportions, from the paper's dataset table:

| Split | Total | Unanswerable | Share |
|---|---:|---:|---:|
| Train | 130,319 | 43,498 | 33.4% |
| Dev | 11,873 | 5,945 | 50.1% |
| Test | 8,862 | 4,332 | 48.9% |

The evaluation splits are deliberately balanced so that a model that always abstains and a model that never abstains both score around chance.

[Natural Questions](https://doi.org/10.1162/tacl_a_00276) (Kwiatkowski et al., TACL 2019) also has unanswerable questions, collected the opposite way. Its questions are real anonymized Google queries, and annotators mark null when the Wikipedia page contains no answer, so a large unanswerable fraction arises naturally rather than by design.

In the RAG setting specifically, [Chen, Lin, Han & Sun (2024), RGB](https://arxiv.org/abs/2309.01431) (AAAI) makes negative rejection (whether an LLM declines when no retrieved context is useful) one of its four measured abilities. It measures that ability as the rate at which a model declines when every retrieved document is noise. For this corpus the failure is an answer stated confidently from another venue's chunk when the asked venue's documentation has no answer.

### Decision for this project

At least 15 questions are unanswerable (15 of the 105-question target, 14.3%), below SQuAD 2.0's 50% and above zero, for three reasons:

1. The primary deliverable is a retrieval ablation, and unanswerable items contribute no recall@k signal: they are excluded from recall averages and scored on refusal behavior in the answer evaluation (§9). Half the labeling budget would then go to items that give no recall@k signal and cannot discriminate bm25 from hybrid_rerank.
2. 15 items is enough to make "never refuses" visibly costly, and few enough to keep the labeling budget on answerable items.
3. SQuAD 2.0's 50% is a training-and-testing balance for a classifier that must learn abstention; this project trains no model.

They come in three kinds, all adversarial in SQuAD 2.0's sense: they must look answerable from this corpus:

- Documented gaps. Facts that are not in the corpus because of how it is built. They are anything about Binance futures funding or leverage (the corpus is `binance-spot-api-docs` only, and contains zero funding chunks for Binance), dYdX v3 (excluded on purpose) and Bybit release-note history (the changelog is excluded).
- Wrong-venue premises. Questions that ask for a real concept on a venue that does not have it. Such questions ask for a `recvWindow` value for dYdX, a `tickSize` for Hyperliquid (it uses `szDecimals` and `nSigFigs`), a `GTX` time-in-force anywhere (no venue in this corpus uses that spelling).

- Plausible-but-absent specifics. A named endpoint, error code or parameter that does not exist, phrased in the house style of the venue it names.

Every gap is proved against the corpus before the question is kept. The concept is searched under every spelling the vendor uses, restricted to the target venue and corpus-wide, with `crossvenue count-term` (chunks containing the term, case-insensitive), and the counts go into the record's `notes`.

Each carries `gold_chunk_ids: []` and a `notes` field stating why it is unanswerable, so a future reader can tell a deliberate negative from an unlabeled question. The harness distinguishes these: unlabeled questions run for latency but are excluded from metric averages rather than scored as zero.

## 4. Adversarial cases for retrieval

### Adversarial retrieval cases in the literature

Three strands are relevant, and this corpus combines all three.

Distractor and noise passages. RGB's noise robustness testbed injects documents that are topically relevant to the question but contain no answer, at a controlled noise ratio ([Chen et al. 2024](https://arxiv.org/abs/2309.01431)). Its counterfactual robustness testbed goes further and injects contradictory information.

Ambiguity. [Min, Michael, Hajishirzi & Zettlemoyer (2020), AmbigQA](https://aclanthology.org/2020.emnlp-main.466/) (EMNLP) found that over half of open-domain Natural Questions are ambiguous. They defined the task as producing every plausible answer plus a disambiguated rewrite of the question for each. The dataset, AmbigNQ, covers 14,042 questions. In a cross-venue corpus, "how often is funding paid?" has a different correct answer per venue, and the correct system behavior is to say so.

Conflicting evidence. [Wang, Prasad, Stengel-Eskin & Bansal (2025), "Retrieval-Augmented Generation with Conflicting Evidence"](https://arxiv.org/abs/2504.13079) builds RAMDocs, which combines ambiguity, misinformation and noise in one setting. It argues the three need different responses: ambiguity should produce multiple disambiguated answers, misinformation and noise should be filtered. In this corpus cross-venue disagreement is ambiguity, which needs enumeration rather than filtering. Bybit and Binance are both right about their own rate limits.

Near-duplicate sources. None of the three sources above builds its test from near-identical documents of different sources. RGB's distractors are relevant passages that lack the answer, AmbigQA's ambiguity is in the question, and RAMDocs combines ambiguity, misinformation and noise. The closest framing in this document is the deduplication work in §6: near-identical documents are a case where lexical or embedding proximity does not imply interchangeability.

### Decision for this project

At least 12 questions are adversarial near-duplicate questions, because cross-venue confusion is the thesis; `v0.2.1-gold` has 13. Each was built from a pair of sources that describe the same concept in overlapping words with a different value. Both sources are venues, or a venue's mainnet and testnet documentation trees (the intra-venue traps below). The question asks about one source, and the other's chunk is the intended distractor. The run measures off-venue retrieval at k (run `results/20260925T012314Z`, exploratory, [findings](../docs/results/findings.md#unnamed-venues-fill-0036-of-hybrid_reranks-comparison-top-10-and-0214-of-bm25s-exploratory)). Its top-1 counts on the adversarial questions are in that section.

Three sub-shapes, from the trap catalog in [domain research](domain-research.md#the-sharpest-near-duplicate-traps-in-this-corpus):

- Same word, different number. Uniswap's 0.30% v3 fee tier vs PancakeSwap's 0.25%; Binance's `recvWindow` max of 60000 vs BingX's max of 5000. Binance's 418 ban scales from 2 minutes to 3 days, and Bybit's 403 lasts at least 10 minutes.
- Same name, different mechanism. "API wallet" on Hyperliquid (an EIP-712 agent wallet) vs on dYdX (a Cosmos permissioned key); `PostOnly` as a time-in-force on Bybit/BingX vs `LIMIT_MAKER` as an order type on Binance vs `"Alo"` title-case on Hyperliquid.
- Intra-venue traps. The corpus's largest duplication is inside Binance, not across DEXes: its mainnet and testnet doc trees are near-clones. Of the 167 chunks of mainnet `rest-api.md`, 101 have a byte-identical testnet occurrence and 105 an identical one once the word "testnet" is removed. Binance additionally has a third environment, Demo Mode, that is not the testnet. Uniswap v2/v3/v4 use three unrelated error vocabularies.

Every adversarial question carries a `notes` field naming the confusable venue, and the confusable venue is recorded so that a failure can be traced to it. The harness reports `venue_coverage@k` per question and per method, the venue of every retrieved chunk (`retrieved_venues` in `per_query.csv`) and per-venue metric tables (`by_venue.csv`). It reports the share of each method's top 10 that unnamed venues take (§9b), but not which unnamed venue outranked the asked one. `eval --venue-filter` restricts every query to the question's labeled venues, but no committed run uses it, so the gap between filtered and unfiltered retrieval is not measured. Both measurements are listed in [directions](../docs/explanation/directions.md#venue-filter-as-a-control).

Ambiguous questions that name no venue are kept to a small labeled sub-case: `"venues": [...]` with more than one entry and a `notes` field saying the correct behavior is enumeration. Generated answers are scored on whether they enumerate rather than pick.

## 5. Annotator agreement with one annotator

### Cohen's κ and its scales

Agreement between two raters on categorical labels is conventionally reported as Cohen's κ, interpreted against the bands from [Landis & Koch (1977), "The Measurement of Observer Agreement for Categorical Data"](https://doi.org/10.2307/2529310) (Biometrics 33(1):159–174). The bands are < 0.00 poor, 0.00–0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect. These bands are conventions: Landis and Koch offered no evidence for them (the survey of the critique on [Wikipedia: Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) collects the primary objections). They are used here because they are the shared vocabulary, with that caveat attached.

### Agreement with a single annotator

Inter-rater reliability is undefined with one rater. The substitute is intra-rater (test-retest) reliability: the same person re-annotates a subsample after a gap long enough that they do not remember their first decisions, and κ is computed between the two passes. This measures whether the criterion is stable, which is the thing at risk when one person labels 114 questions over several sessions.

Two caveats from the reliability literature apply (not checked against a source; listed under [Not verified](#not-verified--left-open)). First, Cohen's κ computes chance agreement from each rater's own marginal rates, so a systematic difference between the first and the second pass (rater bias) does not lower it. That is a defect for test-retest specifically; the intraclass kappa is the more correct statistic when the two ratings come from the same rater. Second, κ is unstable on small, prevalence-imbalanced tables, which is the shape of chunk-level relevance judgments (on `v0.2-gold`, 260 of the 2,848 candidates shown were marked relevant). Reporting raw percent agreement alongside κ, and reporting κ on the binary relevant/not decision at the candidate level rather than at the question level, mitigates both.

### Decision for this project

- Subsample: 25 of the questions round 1 judged (gold chunks or an explicit "nothing in the pool answers this"), stratified by category so that comparison, adversarial and unanswerable items are all represented. Each category gets a share proportional to its size, at least one question, with largest-remainder rounding, and questions within a category are drawn with seed 20260924 (`dataset.sampling`). On `v0.2.1-gold` that is comparison 11, unanswerable 5, adversarial 3, fact 2, and one each of ambiguous, multihop, noisy and version_sensitive. The list is drawn before round 2 (`make round2-sample`) and written to `evals/selection/round2_sample.txt`, so the questions judged on the day are fixed in advance.
- Gap: ≥ 14 days between the first and second pass, and the second pass is run blind (`make relabel`, which labels the listed questions). `make relabel` shows no previous verdict and writes only to the judgment journal, so round 1's gold cannot be overwritten by the reliability check itself.
- Same material: round 2 showed, for each question, exactly the candidates round 1 judged: the pool recorded in the journal plus, for comparison questions, the chunks the augmentation showed (`labeling.pool.retest_candidates`). Nothing was re-retrieved, so a changed pool could not enter the comparison. The order was shuffled per question with seed 20260924, recorded as `shuffle_seed` in each round-2 journal entry, so list position was no cue to the first verdict. The screen showed no marks, no "found by" column and no source-chunk markers.
- Unit of agreement: the candidate-level binary decision, over the pairs both rounds judged. On the fixed sample that is 750 decisions (25 to 41 per question; 11 questions carry augmentation chunks). Its κ is reported with a bootstrap interval, whose width shows how stable it is, and a candidate-level decision needs no per-question comparison of each round's relevant set. Only pairs judged in both rounds are compared; a candidate that one round never displayed is unjudged in that round and enters no comparison.
- Round-1 reference, registered on 2026-09-25 before round 2: κ is computed against round 1 as it stands in `v0.2.1-gold`, after review and augmentation. Against round 1 as first judged (each entry's `previous_relevant` before its earliest correction, plus augmentation) κ is reported as a sensitivity check. 11 of the 25 sampled questions carry a correction (8 from review, 3 derived), and on 3 of them the relevant set differs between the two references.
- Exposure during the quiet period: the answer-rating screen (`crossvenue rate`) shows each question's gold chunks, because grading an answer's coverage needs them. The judge calibration on 2026-09-25 showed in this way the gold chunks of 9 of the 25 sampled questions, listed in `evals/selection/round2_exposed.txt`. The annotator reports rating the answers and not using those chunks as labels. The registered estimate stays κ over all 25 questions; κ over the 16 questions whose chunks were not shown is reported beside it as a sensitivity check. `crossvenue kappa` prints all four readings (two references, two question sets). Deviation D12.
- Source-chunk marker: the round-1 screen marked the chunk a question was drafted from. Shown source chunks were marked relevant 88 times out of 89, other pool chunks 172 times out of 2,759 (`v0.2-gold` journal). Round 2 hid the marker, so part of any difference between the rounds may reflect the missing cue rather than an unstable criterion.
- For later calibrations: the answers rated by hand for a judge calibration come from questions outside the sample of a scheduled re-annotation round, so the rating screen never shows that sample's gold before the round.
- Report: Cohen's κ, raw percent agreement, and the per-category breakdown, in `evals/agreement.md`. Target κ ≥ 0.61 (Landis & Koch "substantial"). Below that, κ is reported as a limitation of this version of the set, the categories with the most disagreement are named, and the criterion in §5a is refined where they show it ambiguous. Re-labeling the set against the refined criterion is the next step, outside this version ([ADR 0006](../docs/adr/0006-single-annotator.md)).
- Disagreements between the two passes are resolved by a third adjudication pass, and the resolved labels ship as another dataset version; the κ is computed on the unresolved pair, since resolving first would inflate it.

### Schedule, and the fallback

| Milestone | Date |
|---|---|
| Round 1: all 114 questions judged, reviewed, frozen as `v0.2-gold` | 2026-09-24 |
| Round-1 augmentation of the 50 comparison questions | 2026-09-24 local; journal timestamps 2026-09-25 UTC |
| Quiet period, no contact with the labels apart from deviation D12 | 2026-09-25 to 2026-10-09 |
| Round 2 re-annotation of the 25-question subsample | 2026-10-10, the whole subsample on one date |
| κ computed and written to `evals/agreement.md`: 0.470 [0.317, 0.645] (95% percentile bootstrap over questions) | 2026-10-10 |
| Round 3, the full relabel, registered in §5b | scheduled for 2026-10-31 |

Dates are UTC, as the judgment journal records them, except where marked local; the work was done in UTC−3, so an evening session local was the next day in the journal. Round 2 met the 14-day minimum under either reading: each question's round-1 and round-2 judgments are 15.7 to 15.9 days apart, from its last round-1 record, augmentation included.

The 14-day gap is what makes the number mean anything. Test-retest reliability measures whether the guidelines are reproducible, and a second pass taken while the first is still in memory measures recall of the first pass instead.

Fallback: a 7-day gap, for when 14 days cannot be kept. The schedule above kept 14, so the fallback did not apply to `v0.2.1-gold`. It is a declared deviation from decision 7, not an equivalent protocol, and it carries a specific, directional cost:

- The reliability literature reports that episodic memory of a specific judgment decays over days rather than hours (not checked against a source; listed under [Not verified](#not-verified--left-open)). At 7 days the probability of recalling rather than re-deriving a verdict is therefore materially higher, particularly for the memorable items, the adversarial and unanswerable ones.
- Recall inflates agreement, so κ is biased upward. The direction is known; the magnitude is not estimable from a single subsample.
- Therefore a κ obtained at 7 days is reported as "κ = X (7-day gap; upper bound, see methodology §5)" every time it appears, never as a bare number. It may not be used to argue that the guidelines are sound. It can only be used the other way: a κ that is low despite the upward bias is strong evidence the guidelines need rewriting.
- The deviation, its date and its reason are recorded in `evals/agreement.md` and in the dataset's `CHANGELOG.md` entry.

Below 7 days the number stops being informative in either direction, so there is no shorter fallback: if even 7 days does not fit, κ is reported as not measured.

## 5a. Annotation instructions

This section is the criterion the labeling follows. It is written down because round 2 ran at least fourteen days after round 1 (15.7 to 15.9 days, [agreement](agreement.md)). Cohen's kappa measures whether the criterion is reproducible, which requires that the annotator's memory is not the criterion. If the rule is only in the annotator's head on day 1, a low kappa on day 15 cannot be told apart from ordinary forgetting. A high one might only mean the questions were memorable. So the instruction has to be complete enough that a stranger (or the same person two weeks later, with no recollection of the first pass) reaches the same verdict from it alone. Anything decided during labeling that is not already written here gets written here, and the change is dated, because a criterion that drifts silently between rounds makes the kappa meaningless in the other direction.

### The criterion

> A chunk is relevant if it contains a fact without which a correct answer to the question cannot be assembled. A chunk that mentions the topic but adds no necessary fact is not relevant.

"Cannot be assembled" is the operative test, and it is deliberately harsher than "is about the right thing". This corpus is full of topical similarity, and topical similarity is where retrieval fails here, so a criterion of "looks related" would mark chunks relevant for their topic alone and measure nothing.

### Relevance by category

| Category | What counts as relevant |
|---|---|
| fact | One or two chunks, or more when the answer is spread across a section: a parameter table plus the note that constrains it. |
| comparison | At least one chunk from every venue the question names, and all of them are relevant. A set covering only one side is incomplete: the question cannot be answered from it. |
| multihop | Every link in the chain, including the intermediate ones. A chunk that is useless on its own but is the only thing connecting two others is relevant. |
| ambiguous | Whatever answers the question under any reasonable reading. The question is ambiguous because several readings are legitimate, so the gold set spans them rather than picking one. |
| unanswerable | Expected to be empty. If the pool contains a chunk that answers the question, the question is misclassified, and it is flagged rather than labeled. Marking it relevant would quietly convert a refusal test into a retrieval test. |
| version_sensitive | Only the chunk for the version, environment or deployment asked about. A chunk from the other version is not relevant even when the wording is nearly identical. Byte-identical text within a venue is one chunk; it is relevant when the version asked about is among its occurrences (`v<n>` lists them), because the same words on both pages are equally true of both. |
| adversarial | Only the correct venue or version. The trap chunk is non-relevant by definition; it is the thing being measured, and marking it relevant would erase the measurement. |

### Boundary cases

Three cases that would otherwise be decided differently on different days.

1. A general rule and a concrete number in separate chunks: both relevant. "Rate limits are counted per UID per second" and "order creation: 10/s" are each incomplete. Without the rule the number gets applied to the wrong scope, and the answer is confidently wrong.
2. Two chunks stating the same fact in different words: both are relevant. Recall is measured against every chunk that would let the answer be built; a minimal cover would keep one of the two. Picking a favorite would penalize a retriever for finding the other one, which is a property of the annotator.
3. A code example with no prose: relevant if the answer is visible in the code. A request body showing `"timeInForce": "PostOnly"` answers "what values does timeInForce take" as well as a sentence would. What counts is what the chunk lets a reader conclude, in whatever form it is written.

### Amendment of 2026-10-10

Written on 2026-10-10, after round 2, from the pairs on which round 1 and round 2 disagree ([agreement](agreement.md)). Neither round applied it. It governs the adjudication of those disagreements and the relabeling of the questions of deviation D23, both recorded in the judgment journal apart from round 1 and round 2 ([deviation D25](#deviations-from-protocol)), and round 3 ([§5b](#5b-annotation-round-3-pre-registered)).

- A fact repeated on many pages. When the question asks about a venue's rule or policy, the chunk that states the rule is relevant. A page of one endpoint that shows only that endpoint's value is relevant only when the question is about that endpoint, or when the candidates contain no chunk stating the rule.
- The premise venue of a comparison. When a comparison question states a fact about one venue as its premise, the chunk that confirms the premise is relevant; that venue's chunks without the premise fact are not.
- An outcome in a response example. An API response example that shows an outcome (for example "banned until epoch …") is not relevant to a question about a duration or rule unless the duration or rule follows from it.

### Recording

Relevance is marked with the numbers shown; `n` records "nothing in this pool answers the question", which is a judgment and is stored as one. It is not the same as skipping, and on an `unanswerable` item it is the expected outcome rather than a failure. Where a question turns out to be misclassified, or the pool missed a chunk that the `source_chunk_ids` marker shows should have been there, that is noted. Those two observations are how the set and the pool get corrected, and neither is visible anywhere else.

## 5b. Annotation round 3 (pre-registered)

Registered on 2026-10-30, before the first judgment of the round ([ADR 0014](../docs/adr/0014-full-relabel-round-3.md)). Round 3 is the full relabel that [deviation D25](#deviations-from-protocol) and §5 name. It is scheduled for 2026-10-31. Dates are UTC, as the journal records them.

### Scope

All 114 questions, one annotator, no subsample. Every field but the labels comes from `v0.2.2-gold`: categories, `expected_behavior`, `venues` and notes. `make round3` reads them from that file and not from `evals/questions.jsonl`, which still equals `v0.2.1-gold`.

### Candidates

Exactly the set round 2's rule gives each question: the pool round 1 recorded plus the chunks its augmentation showed. The three questions of deviation D23 add the chunks of `evals/selection/d23_answer_chunks.jsonl`, which come to 3 chunks on 2 questions; the list of `cv-0106 "which binance endpoint sets leverage on"` is empty. Nothing is retrieved again. The index is `2825a7160c4d09bf`, the one every earlier judgment was made against, and `label` refuses another. The session holds these decisions:

| Category | Questions | Candidate decisions |
|---|---:|---:|
| comparison | 50 | 1,784 |
| unanswerable | 18 | 450 |
| adversarial | 13 | 325 |
| fact | 10 | 251 |
| ambiguous | 6 | 150 |
| noisy | 6 | 150 |
| version_sensitive | 6 | 150 |
| multihop | 5 | 125 |
| total | 114 | 3,385 |

### Screen

The blindness of round 2. Candidates are shuffled per question with seed 20261031, recorded as `shuffle_seed` in each entry; round 2's seed would repeat its order. The questions come in file order. The screen shows no marks, no "found by" column and no source-chunk markers. It shows the criterion of §5a with the amendment of 2026-10-10 on every question, as `--adjudicate` does.

The save check on adversarial and version_sensitive questions prints the question's notes after the marks are entered, as it did in round 2, because the adversarial criterion needs the trap named. That is 19 questions, and on those questions the notes are visible before the verdict is saved.

The session writes only round-3 entries of the journal, one per question when it is answered, and resumes at the first question without an entry. An `unanswerable` question refuses numbers: `n` records that nothing answers it, and `x` records the same with `flagged: true`, which is how §5a's "flagged rather than labeled" is recorded. A flag changes neither the category nor the gold of `v0.3-gold`; the CHANGELOG lists the flags, and a recategorization belongs to a later version. `u` re-opens only the previous question, within the current session, without its marks. Round 3 has no review pass, and any later correction is a recorded deviation.

### Exposure

The annotator has seen part of the labels. The adjudication showed both verdicts on 63 pairs of 17 questions, and the D23 relabel showed 3 questions with their added chunks. The answer-rating screen showed the gold chunks of the calibration questions. The gap since the last judgment of 2026-10-10 is therefore no reliability condition.

### Gold

`v0.3-gold` is `v0.2.2-gold` with `gold_chunk_ids`, `venue_coverage`, `annotator`, `annotated_at` and `annotation_round` taken from each question's round-3 entry. `make round3-gold` builds it. It refuses an unfinished round, and it refuses a journal in which any entry besides round 3 differs from the journal frozen with `v0.2.2-gold`. `relevant` is stored in candidate order, so a gold set that does not change gives the judge an identical prompt. Earlier rounds, the adjudication and the relabel stay in the journal as history. Frozen datasets and committed runs are not modified.

### Reliability

There is no round 4. κ 0.470 stays the only test-retest figure, and it belongs to the round-1 criterion; `make kappa` refuses round 3. Agreement of round 3 with round 1 as corrected is reported as descriptive agreement across a criterion revision, over the pairs both rounds judged: the verdict counts, raw agreement and the direction of disagreement by category, with no κ, no threshold and no reliability claim. Reliability under the amended criterion was not measured.

### What is computed afterwards

- Retrieval: the six methods on `v0.3-gold` with `embedding.device=cpu`, to pair with the two CPU runs on `v0.2.1-gold` and `v0.2.2-gold`, and the registered comparisons of `report.md`. `venue_coverage@k` and `off_venue@k` need no gold, so the retrieval primary comparison cannot change, and `unjudged@k` keeps its values because the judged set does not change.
- Answers: none is generated again. The judge reads gold, so both judged runs are judged again on `v0.3-gold`, reusing cached calls, and `compare-prompts` runs on the results. The comparisons, Holm families and δ rules are those of §7, §9 and §9a, and no hypothesis is added.
- The judge calibration is not repeated for `v0.3-gold`. Its agreement was measured on `v0.2.1-gold`, and the documents say so wherever a re-judged figure appears.

### Reporting

Registered results stay on `v0.2.1-gold`. Results on `v0.3-gold` are reported beside them as another version. A conclusion that differs on `v0.3-gold` is reported as a difference and does not replace the registered one. A difference in recall between gold versions is an effect of the labels and not of retrieval.

## 6. Decontamination and deduplication of questions

### Effect of near-duplicate and leaked questions

Two near-identical questions in an evaluation set are not two observations. They are one observation counted twice, and they break the independence assumption of every paired significance test in section 7. If the duplicates are unevenly distributed across categories, which happens when two questions are generated from adjacent chunks, the duplicated region of the corpus is silently up-weighted.

The contamination literature supplies both the detection machinery and the cautionary numbers. [Lee, Ippolito, Nystrom et al. (2022), "Deduplicating Training Data Makes Language Models Better"](https://arxiv.org/abs/2107.06499) (ACL) built NearDup (MinHash over document n-gram sets, in the tradition of Broder's resemblance) and ExactSubstr (suffix-array exact substring matching). They reported that train-test overlap affects over 4% of the validation sets of standard datasets. On average ExactSubstr removed more content than NearDup (on C4, 7.18% of tokens against 3.04% of examples). [Abbas, Tirumala, Simig, Ganguli & Morcos (2023), SemDeDup](https://arxiv.org/abs/2303.09540) moved the same idea into embedding space. It clusters with k-means, computes pairwise cosine within clusters only (reducing O(N²) to roughly O(N²/k)), and calls pairs above a threshold parameterized as 1 − ε semantic duplicates. They explicitly decline to prescribe a universal ε and note that as ε grows the matches range from perceptual duplicates to merely semantically redundant pairs.

And a leak inflates results. A retrieval eval trains no model, so the mechanism differs, and the effect is the same. A question whose phrasing already appeared in the set, or whose phrasing is lifted from the gold chunk, is answered by memorization of the corpus surface rather than by retrieval.

### Decision for this project

Two checks, both run by `crossvenue validate-evals` and both cheap:

1. Question-question near-duplicate check. All questions are encoded with `bge-base-en-v1.5`, and any pair with cosine ≥ 0.92 is flagged for manual merge or rewrite. It is the same model the dense retriever uses, so the threshold means the same thing as the retrieval scores.

   The number is chosen for this corpus. Chroma found 11.91% of naively generated query pairs above 0.9 cosine, identical or nearly so. This corpus is deliberately full of parallel questions: "what is the maximum `recvWindow` on Binance?" and "…on BingX?" differ in one token. They will score high on any sentence encoder, yet they are the most valuable pairs in the set. A threshold at 0.85 or 0.90 would delete the venue-parallel items the experiment needs. 0.92 is expected to lie above the cosine of a one-token venue swap and below that of the same question reworded. No saved output measures the cosine of venue-parallel pairs, so the threshold is set on this expectation. Flagged pairs are reviewed by hand, never dropped automatically; pairs that differ only by venue are kept and annotated as an intentional parallel.

   Config key: `eval.duplicate_threshold: 0.92` in `configs/default.yaml`, so changing it is an ablation and not a code edit.

2. Question-chunk leakage check. For each question, the longest word n-gram it shares with its gold chunks is computed. Any shared n-gram of ≥ 6 tokens that is not an API identifier or a quoted error string fails review and the question is rephrased. This is the n-gram overlap method the contamination literature uses ([GPT-3](https://arxiv.org/abs/2005.14165) used 13-gram overlap, [GPT-4](https://arxiv.org/abs/2303.08774) 50-character substrings), scaled down because questions are short.

   `validate-evals` reports a question whose longest run shared with a gold chunk reaches 6 words, after cutting out spans in backticks or double quotes (`dataset.validate.LEAKAGE_NGRAM`). On `v0.2.1-gold` no question does: the longest shared run is 4 words ([datasheet](datasheet.md#composition)).

Neither check changes the corpus itself. Corpus-internal duplication is handled at ingestion, and only its exact form: the 536 groups of byte-identical chunk text (433 of them Binance mainnet/testnet pairs) are each stored once with every occurrence kept as provenance. The cross-venue phenomenon under test is in near-identical text, which stays; identical text is the same string on mainnet and testnet ([ADR 0003](../docs/adr/0003-exact-deduplication.md), [`docs/results/findings.md`](../docs/results/findings.md#108-of-chunk-occurrences-are-byte-identical)).

## 7. Set size and statistical power

### Topic counts in the IR literature

[Voorhees & Buckley (2002), "The effect of topic set size on retrieval experiment error"](https://doi.org/10.1145/564376.564432) ([NIST record](https://www.nist.gov/publications/effect-topic-set-size-retrieval-experiment-error)) is the reference. They empirically derived, from TREC runs, the rate at which a different topic set of the same size would have reversed the conclusion, as a function of topic-set size and observed score difference. Their error rates were computed directly for sets up to size 25 and extrapolated above that. By their error rates, even at TREC's standard 50 topics a small difference in mean effectiveness does not support a confident claim of superiority.

[Sakai (2016), "Topic set size design"](https://doi.org/10.1007/s10791-015-9273-z) (Information Retrieval Journal 19(3):256–283) studies the same question from the power-analysis side. It notes that pooled test collections typically have n = 50–100 topics without a stated reason for the number. It supplies sample-size methods based on the paired t-test, one-way ANOVA and confidence intervals, driven by within-system variance estimated from past topic-by-run score matrices.

### Significance testing

[Smucker, Allan & Carterette (2007), "A comparison of statistical significance tests for information retrieval evaluation"](https://doi.org/10.1145/1321440.1321528) (CIKM) compared the paired t-test, Wilcoxon signed-rank, sign test, bootstrap and Fisher's randomization test across TREC 3 and 5–8. They found little practical difference between randomization, bootstrap and t tests, took the randomization test as the reference standard, and recommend against the Wilcoxon signed-rank and sign tests for differences between means. Their [2009 follow-up](https://doi.org/10.1145/1571941.1572050) examined agreement as topic counts shrank toward 10 and found the tests increasingly disagree (randomization producing smaller p-values than t below p = 0.1, bootstrap systematically below t). That is the concrete reason not to run a 15-question evaluation and report a p-value from it.

### The arithmetic for this project

For a paired comparison, the normal approximation for the sample size needed to detect a standardized effect *d* (mean paired difference ÷ SD of the differences) at α = 0.05 two-sided and 80% power is

```
n ≈ (z_0.975 + z_0.80)² / d² = (1.96 + 0.84)² / d² = 7.84 / d²
```

Cohen's conventional effect sizes are not checked against a source (listed under [Not verified](#not-verified--left-open)). Three cases: Cohen's conventional large and medium effects, and this project's target, 0.30, between Cohen's small (0.2) and medium (0.5):

| Effect | d | n needed |
|---|---:|---:|
| Large | 0.8 | 13 |
| Medium | 0.5 | 32 |
| Project target | 0.30 | 88 |

Inverted, the smallest *d* each candidate size can detect:

| Set | scorable n | smallest detectable *d* |
|---|---:|---:|
| 105 questions (15 unanswerable excluded from recall) | 90 | 0.30 |
| 85 floor (15 excluded) | 70 | 0.33 |
| comparison subgroup, 35 | 35 | 0.47 |
| comparison subgroup, 50 | 50 | 0.40 |

### Scorable n against set size

`recall@k` is undefined without a gold set, so the 15 unanswerable questions are excluded from every recall average: they measure refusal, which is a different question asked of a different stage. A set of 105 therefore has a scorable n of 90, and the size that matters for power is always the scorable one.

The same applies inside the subgroup, and it is a binding constraint on selection: all 50 comparison questions must be answerable, or the subgroup that carries the headline claim is smaller than it looks. An unanswerable comparison question ("which of these two venues documents X", where neither does) is a legitimate item, and it belongs in the `unanswerable` budget; the comparison budget holds answerable questions only. If ten of the fifty turned out unanswerable, the subgroup would be n = 40 and the detectable δ would rise from 0.20 to 0.22.

`crossvenue select --finalize` enforces it: it refuses to write a set whose comparison subgroup has fewer than 50 answerable questions.

### The binary McNemar model

For a question with one gold chunk the per-question outcome is binary: the gold chunk is in the top 10 or it is not. For a question with more than one it is fractional (below); the model treats every outcome as binary. For a binary outcome the sharper tool is McNemar's test on the discordant pairs, the questions where exactly one of the two methods hits. In the model, each question is discordant with probability ψ, and a discordant question favors the better method with probability ½ + δ/2ψ, where δ is the recall@10 gap. Power is then the probability that the exact two-sided binomial test on the discordant split reaches α = 0.05.

Computed exactly at ψ = 0.25 (the two methods disagreeing on a quarter of questions), the smallest δ reaching 80% power. The table is this calculation: a binary per-question hit and the exact McNemar test without adjustment. It is not the power of the comparisons as they are run (below):

| n | smallest detectable δ (recall@10) | power at δ = 0.20 |
|---:|---:|---:|
| 35 | 0.24 | 0.59 |
| 40 (if 10 of the 50 were unanswerable) | 0.22 | 0.68 |
| 50 (the comparison subgroup) | 0.20 | 0.81 |
| 60 | 0.19 | 0.89 |
| 90 (the whole scorable set) | 0.16 | 0.98 |

The mean realization is not the power. At n = 120, ψ = 0.25 and δ = 0.10 the expected discordant split is 21/9, which is significant (p = 0.043). The probability of landing on a significant split is 0.52, a coin flip. Detecting δ = 0.10 at ψ = 0.25 with 80% power needs n ≥ 207.

The comparisons are run differently. Per-question recall@10 is fractional for a question with more than one gold chunk (156 of the 300 values of the six methods on the 50 comparison questions). The test is the paired randomization test, and each comparison is Holm-adjusted within a family of 6. `crossvenue power` simulates that test with the family's observed discordance (127 of 300 question pairs differ, ψ = 0.42) and difference sizes (mean 0.51), varying only which method a discordant question favors (`results/20260926T015648Z_power`, on `v0.2.1-gold`). Its comparison gold was augmented after results were seen ([D4](#deviations-from-protocol)). It reaches 80% power from δ = 0.17 when the comparison is Holm's first step (α/6) and from δ = 0.15 unadjusted. At δ = 0.20 the power is 0.97 and 1.00. With the discordance fixed at the observed level, a δ above 0.22 cannot occur in this model.

### Decision for this project

105 questions, hard floor 85, sized from the subgroups rather than the total. Comparison ≥ 50 and all of them answerable, unanswerable ≥ 15, no category below 5, and the rest as decision 2 sets out. 105 is what those constraints add up to.

The reasoning starts from what the project claims. The headline measurement is cross-venue confusion: whether retrieval returns the venue the question asked about when a near-identical page from another venue is competing. That claim is measured in the comparison subgroup only. Sizing the total and letting the subgroup fall out of it sizes the wrong thing.

- 105 total gives 90 scorable items once the 15 unanswerable ones are excluded from recall averages. That detects *d* = 0.30, this project's target, between Cohen's small (0.2) and medium (0.5) effects, and is inside Sakai's observed 50–100 band for pooled collections. Cohen's values are not checked against a source (listed under [Not verified](#not-verified--left-open)). The floor of 85 gives 70 scorable and detects *d* = 0.33.
- As the topic count falls toward 10 the paired tests increasingly disagree with each other (Smucker et al. 2009), so a p-value from a much smaller set is not worth reporting.
- Above ~120 the binding constraint stops being statistics and becomes annotator consistency: at 25 pooled candidates per question, 105 questions is already 2,625 relevance decisions by one person. Section 5's κ is the check on whether that held.
- Comparison at 50 is 47.6% of the 105-question design. That is deliberate, and the other categories are smaller for it: it is the only subgroup large enough for a significance claim, and everything else is reported descriptively with a confidence interval. At 35 the smallest detectable δ would be 0.24, against 0.20 at 50.

### The power ceiling, as a number

δ = 0.10 in recall@10 is not detectable at any set size one person can label. At ψ = 0.25 it needs n ≥ 207 comparison questions (168 at ψ = 0.20, 249 at ψ = 0.30), against the 2,625 relevance decisions that 105 questions already require. It is a computed limit of this study design, and it is reported as one:

> Comparison subgroup n = 50, all answerable. Smallest recall@10 difference detectable at 80% power, α = 0.05, in the binary McNemar model: δ = 0.20. Whole scorable set n = 90, δ = 0.16. A δ of 0.10 would require n ≥ 207 in the subgroup and is outside this design. For the Holm-adjusted randomization test on fractional recall, with the observed discordance: δ = 0.17 at Holm's first step, 0.15 unadjusted. The discordance is observed against comparison gold augmented after results were seen (deviation D4).

Consequences that follow from it, and are binding:

- A comparison result smaller than 0.20 is reported as inconclusive, with the effect size and its confidence interval, never as "no significant difference", which readers hear as "no difference" ([style rule 13](../docs/style.md)).
- This design assumes that the cross-venue effect is large. If it is modest, the output of this project is "not resolved at this sample size", and that is an acceptable result: it is the design's stated limit, fixed before the data.
- A scope defined by a category or a slice of fewer than 50 questions carries no p-value at all. The rule counts the questions of the scope, whether or not they carry a value. A registered family over a scope of 50 or more questions keeps its p-values when fewer answers carry the metric, as venue_attribution_error does over the answers with claims. A category or slice below 50 supports a descriptive claim and nothing more. The claim is a count per method with an interval: "method A finds a gold chunk in the top 10 on k of the n adversarial questions, method B on m".

Deviation [D16](#deviations-from-protocol) records the metric and scope the first of these rules is applied to, and the rule each smaller published result follows.

Testing protocol. Paired bootstrap over questions with 10,000 resamples (seed 20260924) gives 95% percentile intervals on the difference, following Smucker et al. (2007). A paired randomization test (sign flip, 10,000 permutations, seed 20260924) gives the p-value for every metric, since Smucker et al. (2007) use it as the reference test (`evaluation.paired`). For outcomes that are binary per question (a hit in the top 10, every named venue in the top 10), the exact two-sided McNemar test on the discordant questions gives a second p-value. Effect size and CI are reported first and the p-value second; a difference is not described as an improvement unless the CI excludes zero.

One primary comparison. The study is designed around one test: venue_split − hybrid_rerank on venue_coverage@10 over the 50 comparison questions (`eval.primary_comparison`). It is reported with its own p-values, unadjusted. It was fixed after the `v0.2-gold` run had measured the coverage failure on the same 50 questions (deviation D13). It therefore measures the detector-plus-split mechanism on the questions it was designed on, and no held-out set tests it. Every other configured pair, on every metric and scope, is secondary. Within each family of one metric on one scope, the randomization p-values are Holm-adjusted (`holm_p` in `paired.csv`, "Holm p" in `report.md`). Only a secondary result whose adjusted p is below 0.05 is called significant. Holm controls the family-wise error rate without assuming the tests independent, which they are not: the same questions enter every pair, and every pair shares a method with another pair.

## 8. Domain research: what developers ask

The developer questions and the near-duplicate traps (T1–T13) the questions are drawn from are in [domain research](domain-research.md).

## 9. Answer generation and judging (pre-registered)

Registered on 2026-09-25, before any answer was generated for evaluation. The five-question trial run that checks the infrastructure is not an analysis and no conclusion is drawn from it.

### Configurations

Every model is called through OpenRouter. One rule holds for all generators: reasoning effort minimal, at most 4,000 output tokens, temperature 0.0 where the endpoint accepts it (Gemini's does not), JSON-schema output, and at most two retries of an invalid response. Four configurations, each a generator over a retriever's top 8 chunks (`generation.matrix`):

| configuration | generator | retriever |
|---|---|---|
| gemini@venue_split | `google/gemini-3.8-flash` (OpenRouter's Vertex endpoint, no fallback) | venue_split |
| qwen@venue_split | `qwen/qwen3.8-27b` (DeepInfra, bf16, no fallback) | venue_split |
| mercury@venue_split | `inception/mercury-2.5` | venue_split |
| gemini@hybrid_rerank | `google/gemini-3.8-flash` | hybrid_rerank |

[ADR 0009](../docs/adr/0009-generators.md) gives the reasons for these generators.

### Metrics

A judge grades each claim of an answer and the answer as a whole. Two candidates, `anthropic/claude-sonnet-5` and `anthropic/claude-opus-5.5`, get the same prompt (`prompts/judge.md` v2), schema and effort (medium); the one used on the full matrix is chosen by the calibration below ([ADR 0010](../docs/adr/0010-judge.md)). Per answer:

| metric | definition | applies to |
|---|---|---|
| faithfulness | share of claims its cited chunks state | answers with claims |
| venue_attribution_error | share of claims attributed to a venue other than the one whose chunk states them | answers with claims |
| coverage | per venue with gold chunks, facts the answer states over facts a correct answer needs, averaged over those venues | questions with gold chunks |
| `behaviour_correct` | answered, refused or asked for clarification as `expected_behavior` requires | every question |

A metric that does not apply to an answer is missing, never 0, and averages are over the answers it applies to.

### Primary comparison

gemini@venue_split − gemini@hybrid_rerank on venue_attribution_error and on coverage, over the 50 comparison questions. The generator is held fixed, so the comparison isolates retrieval: whether venue_split's gain in venue coverage ([findings](../docs/results/findings.md#splitting-retrieval-by-venue-raises-comparison-coverage-from-070-to-098)) changes the answers. The expected direction is lower venue_attribution_error and higher coverage for venue_split.

Each of the two is tested on the questions both configurations answered and the judge graded: 95% paired bootstrap interval (10,000 resamples, seed 20260924) and a two-sided paired randomization test (10,000 permutations, same seed). The two form the primary family, Holm-adjusted within it; a primary result is significant when its adjusted p is below 0.05.

### Secondary comparisons

Every other pair of configurations, on every metric, over the comparison questions and over all questions. Holm-adjusted within each family of one metric on one scope, as for retrieval (§7). A secondary result is significant only when its adjusted p is below 0.05; secondary results are reported, and the primary conclusion is drawn from the primary comparison alone.

### Conditions for using a judge dimension

A judge dimension enters a conclusion only if the judge agrees with the human calibration ratings at or above the thresholds below. The calibration sample is 40 answers of the full generation run, stratified by configuration and category, seeded, and fixed in advance in `evals/selection/judge_calibration_sample.txt`. Answers to the five trial questions (`cv-0003 "how long does binance keep my"`, `cv-0045 "do binance and bybit ban you"`, `cv-0081 "bingx rejects my timestamp, can i"`, `cv-0105 "how often does binance pay funding"`, `cv-0122 "how often is funding paid on"`) are excluded from it: the judge prompt was revised on them, and agreement measured on them would favor the revised prompt. An answer identical to another configuration's (gemini@hybrid_rerank on a question where its chunks match venue_split's, which is one cached call) is drawn at most once, under the first configuration. The sample drawn from `results/20260925T175128Z_answers` holds gemini@venue_split 12, mercury@venue_split 12, qwen@venue_split 11 and gemini@hybrid_rerank 5 answers; comparison 18, unanswerable 4 and 3 of each other category. Each judge candidate grades the same 40 answers; the candidate that meets the thresholds on more dimensions is used, with ties broken by agreement on `venue_correct` and coverage, then by cost.

| dimension | measure | threshold |
|---|---|---|
| `supported` (faithfulness) | Cohen's κ over claims | ≥ 0.61 |
| `venue_correct` (venue_attribution_error) | Cohen's κ over claims | ≥ 0.61 |
| `behaviour_correct` | Cohen's κ over answers | ≥ 0.61 |
| coverage | Spearman correlation over answers | ≥ 0.70 |

The binary threshold is the one the gold labels are held to (decision 7): a judge is not trusted further than the human is trusted to agree with themselves. The coverage threshold, Spearman 0.70, is a judgment fixed before any rating; no published standard sets it. A judge meets it when it orders answers by coverage close to the order the human gives, since the comparisons use coverage only to order configurations. The judge's agreement with itself on a repeated run of 40 answers is reported beside each threshold; a dimension on which the judge disagrees with itself more than with the human is reported as noise.

If a primary metric's dimension fails its threshold, the primary result is reported with the measured agreement and is not interpreted. Thresholds are not changed after the ratings are seen.

### Calibration result

Recorded 2026-09-25, after the 40 answers were rated and before either matrix was judged. One answer is left out of `behaviour_correct` by a decision made after the agreement with it had been computed ([D20](#deviations-from-protocol)). Agreement with the human ratings, 95% bootstrap intervals over answers (10,000 resamples, seed 20260924), from `results/20260925T175128Z_answers/calibration.json` (`crossvenue select-judge`), on the calibration sample drawn from the questions of `v0.2.1-gold`:

| dimension | threshold | sonnet | opus |
|---|---|---:|---:|
| `supported` (91 claims) | κ ≥ 0.61 | 0.165 [0.000, 0.490] | 0.238 [−0.051, 0.558] |
| `venue_correct` (91 claims) | κ ≥ 0.61 | 0.492 [0.000, 0.662] | 0.662 [0.647, 0.663] |
| `behaviour_correct` (39 answers) | κ ≥ 0.61 | 0.780 [0.559, 0.947] | 0.722 [0.486, 0.935] |
| coverage (34 answers) | Spearman ≥ 0.70 | 0.843 [0.670, 0.949] | 0.899 [0.782, 0.964] |

Opus meets three thresholds and Sonnet two, so Opus is the judge ([ADR 0010](../docs/adr/0010-judge.md)). What this does to the analyses:

- `supported` fails for both candidates, so faithfulness enters no conclusion: the H-v2 guard "without lowering faithfulness" (§9a) cannot be tested, and neither can any claim about unsupported statements. Faithfulness is reported as the judge's grade only, marked uncalibrated, except in the `report.md` and `paired.csv` of the two committed judged runs, which predate the mark ([findings](../docs/results/findings.md#answer-generation)).
- venue_attribution_error and coverage, the two primary metrics, are graded on dimensions that pass. coverage meets its threshold on the point estimate and on the interval; `venue_correct` on the point estimate only. The 2 of 91 claims either grader marks wrong are both in one answer (`qwen@venue_split` on `cv-0121 "which bingx endpoint returns the usdt-m"`). Both cite a CCXT chunk and name BingX, and the graders disagree on one of them. Its κ lies in [0.647, 0.663] or is undefined (undefined in 3,634 of 10,000 resamples), so its interval is near-degenerate and cannot establish the threshold, and the sample holds no cross-exchange misattribution. That kind was added after the judged answers were read ([D14](#deviations-from-protocol)). Therefore, venue_attribution_error is not resolved by this calibration. The findings and the README report it as unresolved; the `report.md` and `paired.csv` of the two committed judged runs show it with no mark, as their `results/runs.json` notes say.
- `behaviour_correct` passes on the point estimate only, and no calibration answer is a partial answer, so the wrong-refusal share of H-v2 is read from the recorded flags.

Only Opus graded the calibration answers a second time, by an instruction written after both candidates' first gradings ([D15](#deviations-from-protocol)). Opus's agreement with its own repeat on the 40 answers (`judgements_opus_r1.jsonl`): κ 0.852 [0.000, 1.000] on `supported`, 1.000 [1.000, 1.000] on `venue_correct` and `behaviour_correct`, Spearman 0.964 [0.910, 0.990] on coverage. It agrees with itself more than with the human on every dimension, so no dimension is reported as noise. On `supported` its self-agreement is inconclusive at this size, since its interval runs from chance agreement to perfect agreement. That it follows a different criterion from the human's is an expectation a larger repeat would test.

One rated answer was shown to the human with a different flag than the judges saw. `qwen@venue_split` on `cv-0138 "swap fail STF error mean what"` set both `refused` and `asked_clarification`, and the rating screen, built after the outcome rule was added, showed `asked_clarification` as false. The judges saw it as true. The rating screen shows the model's flag. The answer is left out of `behaviour_correct` against the human (`generation.calibration_exclude_behaviour`), after the agreement with it had been computed ([D20](#deviations-from-protocol)). That dimension is measured on 39 answers; with it, κ was 0.789 for Sonnet and 0.734 for Opus, and both pass either way. It stays in the other three dimensions, whose verdicts do not depend on the flag, and in the agreement between the two judges, who saw the same message.

### Attribution errors by kind (reporting rule)

Added on 2026-09-25 local time, after the judged answers were read (deviation D14); the judge prompt and its verdicts are unchanged and nothing is judged again. A claim the judge marks misattributed is reported in one of two kinds, from the venue of the chunks it cites and the venue it names (`generation.attribution`):

- cross-exchange: the cited chunks belong to one exchange, or are CCXT's documentation of one exchange, and the claim names another;
- through a wrapper: the cited chunks are CCXT's documentation of an exchange, and the claim names that exchange.

A misattributed claim that fits neither is counted as other. venue_attribution_error is reported in total and per kind. The calibration sample is 40 answers of `results/20260925T175128Z_answers`, on the questions of `v0.2.1-gold`, with 91 claims. Every claim of it that either grader marks misattributed is of the wrapper kind, so the calibration has no negative case for cross-exchange attribution, the error the corpus was built to provoke.

### Cost and latency

Every call records its requested and actual model, provider, tokens, cost and latency (`results/<ts>_answers/`). Cost and latency are reported per configuration and are not tested.

### Observation on the v1 run

Recorded after the matrix `results/20260925T175128Z_answers` (generator prompt v1, `prompts/generator.v1.md`, the questions of `v0.2.1-gold`) was generated and before any answer was judged. Gemini refused 25 of the 50 comparison questions with venue_split and 29 with hybrid_rerank; Qwen refused 17, Mercury 7. Of Gemini's 25 venue_split refusals, 3 had a gold chunk of every named venue in the context, 12 of some named venues and 10 of none. In 24 of the 25 the context held chunks of every named venue. The groups are by comparison gold augmented after retrieval results were seen ([D4](#deviations-from-protocol)). Rule 5 of prompt v1 allows a full answer or a refusal: "If the chunks do not contain the answer, refuse … and give no claims". The expectation, which prompt v2 tests (§9a), is that a comparison with one side missing is refused whole. Tables: [findings](../docs/results/findings.md#under-prompt-v1-geminivenue_split-refuses-25-of-the-50-comparison-questions).

The primary comparison above stays on the v1 run and is not changed by this observation.

## 9a. Generator prompt v2 (exploratory)

Registered on 2026-09-25, before any v2 answer was generated.

### Status

Exploratory. Prompt v2 was written after reading the v1 answers, so nothing tested here is pre-registered in the sense of §9, and no result of this section changes §9's conclusions (deviation D10).

### The change

One edit to `prompts/generator.md`, made once, with no iteration on v2's results. Rule 5 allows three outcomes: a full answer, a partial answer (claims for the venues the chunks cover, and a `missing` entry for each venue they do not) and a refusal. Rule 6 of v2 lets an absence be a claim: "the provided <venue> documentation does not list <thing> among the <options> it enumerates". The claim cites the chunk that enumerates them and never says that the venue lacks the thing. Rule 3 (a fact of one venue is a fact of that venue only) is unchanged. The answer schema adds `partial` and `missing`. Everything else is held fixed: the four configurations, models, providers, parameters, retrievers and top 8. The v2 answers are cached separately (the request, and so the cache key, contains the prompt).

### Outcome rule

An answer flagged `partial` that carries no claim counts as a refusal in every count and metric (`generation.schemas.outcome`). The rule was fixed on 2026-09-25, before any answer of either matrix was judged, and applies to v1 and v2 and to every generator alike. It states no fact, so it is a refusal whatever flag the model set. The judge receives it as refused and the human rating screen shows it as refused; the answer record keeps the flags the model set. v1 has no partial answers, so the rule changes no v1 number; under v2 it applies to 4 answers, all mercury@venue_split. The rule was fixed after the v2 answers were read (deviation D11). The v1 and v2 answers are `results/20260925T175128Z_answers` and `results/20260925T202827Z_answers`, on the questions of `v0.2.1-gold`.

### Hypothesis

H-v2: against v1, v2 lowers the share of wrong refusals (`refused` on a question whose `expected_behavior` is answer or ask_clarification, 94 questions) without raising venue_attribution_error and without lowering faithfulness.

### Tests

v2 − v1 per configuration, paired on the same questions. The wrong-refusal share is tested by exact McNemar; venue_attribution_error and faithfulness by the §9 procedure (95% paired bootstrap, two-sided randomization test, seed 20260924) on the answers both versions have claims for. Holm-adjusted within each metric across the four configurations. H-v2 holds for a configuration when, at adjusted p < 0.05, its wrong-refusal share falls and no guard metric moves the wrong way. A guard whose adjusted p is 0.05 or above is inconclusive at this size, so its interval is reported beside it.

### Secondary comparison

gemini@venue_split − gemini@hybrid_rerank under v2, on venue_attribution_error and coverage over the 50 comparison questions, by the §9 procedure, Holm-adjusted within the two.

Judge metrics here use the judge chosen by §9's calibration, with judge prompt v2 unchanged; an absence claim is graded like any other claim, against the chunk it cites. The 40 calibration answers of `results/20260925T175128Z_answers`, on the questions of `v0.2.1-gold`, come from prompt v1. None is a partial answer and none of their 91 claims has the absence form, so the judge's grading of both is not calibrated.

## 9b. Off-venue retrieval (exploratory)

Registered on 2026-09-26, before any off-venue number was computed.

### Status

Exploratory. The measure was defined after the retrieval results of `results/20260925T012314Z` were seen, so nothing here is pre-registered in the sense of §7 or §9, and no result of this section changes their conclusions.

### Definitions

- Off-venue share@10: the share of a question's top 10 whose chunk venue is not among the question's labeled `venues`.
- Top-1 off-venue: 1 if the first chunk's venue is not among the question's labeled `venues` and is not ccxt on a question that does not name ccxt, else 0.
- Top-1 via-ccxt: 1 if the first chunk is a ccxt chunk on a question that does not name ccxt, else 0.
- Via ccxt: a chunk from ccxt on a question that does not name ccxt is counted separately, as via-ccxt share@10, reported beside the off-venue numbers and excluded from them. A ccxt chunk on a question that names ccxt is on-venue.

Every number is computed from the columns `retrieved_ids` and `retrieved_venues` that the run saved in `per_query.csv`, without running retrieval again, and from the questions' `venues` and `category`; no gold label enters it.

### Scope

The comparison (50) and adversarial (13) questions of `v0.2.1-gold`, all six methods, run `results/20260925T012314Z`.

### Comparisons

hybrid_rerank − bm25, hybrid_rerank − dense and hybrid_rerank − hybrid on off-venue share@10 and on top-1 off-venue, each by a 95% paired bootstrap interval (10,000 resamples, seed 20260924, as in §7). On the 50 comparison questions each comparison also gets a two-sided sign-flip randomization test with the same resamples and seed, Holm-adjusted within each metric. The 13 adversarial questions are a category of fewer than 50 questions, so under §7 their comparisons are descriptive: intervals only, with no p-value and no Holm adjustment. venue_split and venue_split_oracle are reported descriptively: they retrieve per named venue by construction, so a low off-venue share follows from their design.

### Reading

An off-venue chunk measures competition for a slot in the top 10, not an error in an answer: a question can be answered correctly with off-venue chunks beside the right ones.

## Deviations from protocol

Where the data or the analysis depart from the protocol above, with when the departure was made relative to the data, its reason and its effect on the conclusions. A draft of the protocol revised before the data it governs existed is not listed ([style rule 43](../docs/style.md)), and the numbers of removed entries are not reused. The Record column names where each entry is recorded besides this table: the dataset [`CHANGELOG.md`](datasets/CHANGELOG.md), an [ADR](../docs/adr/), or a section of this document.

| # | Protocol | As built | When | Why | Effect on conclusions | Record |
|---|---|---|---|---|---|---|
| D1 | Decision 4, §2: questions hand-written or heavily rewritten from an LLM draft | An LLM, Claude Opus 5 (`claude-opus-5`), drafted all 114 questions, which were accepted with the draft text unchanged: 114 of 114 identical to `v0.1.6-candidates`. The model is in the `generator` field of every record and the brief's hash in `prompt_hash`. The human steps are selection and labeling. | after collection, before analysis | Candidate files are immutable, and selection accepts or rejects rather than edits. The drafts were grounded in the themes of [domain research](domain-research.md) and in corpus chunks, and their median length is 11 words, inside the real-query band §2 targets. | Retrieval results hold for LLM-drafted questions of this style; §2's generator-bias risks apply to them in full. | CHANGELOG `v0.1-candidates` … `v0.2-gold`; [ADR 0006](../docs/adr/0006-single-annotator.md) |
| D2 | Decisions 1–2: 105 questions | 114. Targets are floors in `select --finalize`; selection accepted 9 above target: unanswerable 20 (+5), adversarial 13 (+1), ambiguous, noisy and version_sensitive 6 (+1 each). Comparison is exactly 50, all answerable. | after collection, before analysis | `select` does not close a category at its target, and `evals/selection/selection.jsonl` records no reason for the extra 9. The 5 extra unanswerable items enter no recall average; the other 4 add one labeled question each to categories reported descriptively. | The comparison subgroup and the primary test are unchanged; the 5 extra unanswerable questions enter no recall average. | CHANGELOG `v0.2-gold` |
| D3 | Decision 3: every venue the answer venue for ≥ 8 questions | ccxt names 6 questions. Every other venue names 14 or more; 57 of 114 questions (50.0%) name a DEX, meeting the ≥ 45% DEX floor. | after collection, before analysis | ccxt is a client library, held deliberately small in the comparison supplement (2 of 38). `validate-evals` reports the imbalance as a known warning. | Results that concern ccxt have 6 questions and are descriptive only. | CHANGELOG `v0.1.2-candidates`, `v0.2-gold` |
| D4 | Decision 5, §1: pool from every pool method | Round 1 pooled four methods (bm25, dense, hybrid, hybrid_rerank). `venue_split` became a pool method after round 1; on the 50 comparison questions the never-judged top 20 of `venue_split` and `venue_split_oracle` was judged by augmentation (rule 3). The 64 other questions are not augmented. | after results were seen | Round 1's pool left 22 named venues `missing_from_pool` and an unjudged@10 of 0.402 for `venue_split` on the 50 comparison questions. Both are of `v0.2-gold`, the unjudged@10 from the retrievals of `results/20260925T002151Z`. The oracle is augmented but never pooled: augmentation only adds judgments. | Recall is comparable only within one gold version. The 64 questions that were not augmented keep round-1 gold, and venue_split_oracle's unjudged@10 on the six ambiguous questions is 0.467, so more judging could move its recall there either way. That share is from `results/20260925T012314Z`, on `v0.2.1-gold`. | [ADR 0005](../docs/adr/0005-labeling-pool.md), [ADR 0007](../docs/adr/0007-venue-split.md); CHANGELOG `v0.2.1-gold` |
| D9 | §9 as registered: every generator at reasoning effort low and temperature 0; judge `claude-sonnet-5` called on Anthropic's API directly | Reasoning effort minimal for every generator; Gemini on OpenRouter's Vertex endpoint without temperature; two judge candidates, both through OpenRouter, chosen by calibration. Changed before any evaluation answer was generated; the primary and secondary comparisons are unchanged. | before collection | The trial run: Qwen at effort low took 57 s per answer on average and once over 300 s; Gemini's only schema-enforcing endpoint rejects temperature; the project uses OpenRouter as its only model provider. | None on the registered comparisons; generator results apply at reasoning effort minimal. | [ADR 0009](../docs/adr/0009-generators.md), [ADR 0010](../docs/adr/0010-judge.md) |
| D10 | §9: one generator prompt (v1) | A second prompt, v2, generated over the same four configurations and analyzed in §9a as exploratory; §9's primary and secondary comparisons stay on v1. | after results were seen | Written after reading the v1 answers: under v1's rule 5 the generators refused comparisons with one side missing. | §9a is exploratory and changes no §9 conclusion. | §9a; [ADR 0013](../docs/adr/0013-generator-prompt-v2.md) |
| D11 | §9, §9a: an answer's outcome is what the model flags | An answer flagged `partial` with no claim counts as a refusal in every count and metric, for every generator ([§9a outcome rule](#outcome-rule)). | after collection, before analysis | Fixed after the v2 answers were read and before either matrix was judged: in `results/20260925T202827Z_answers`, on the questions of `v0.2.1-gold`, 4 Mercury v2 answers flag `partial` and state no fact. v1 has no partial answers, so no v1 number and nothing in §9 changes. | No v1 number changes; under v2, 4 Mercury answers count as refusals. | §9a |
| D12 | §5: a quiet period with no contact with the labels, 2026-09-25 to 2026-10-09 | The answer-rating screen showed the gold chunks of 9 of the 25 round-2 questions on 2026-09-25 (`evals/selection/round2_exposed.txt`) | before collection | Grading an answer's coverage needs the gold; the annotator reports rating the answers and not using the chunks as labels | κ over all 25 questions stays the registered estimate; κ over the 16 unexposed questions is reported beside it (§5). | §5; `evals/selection/round2_exposed.txt` |
| D13 | §7: the retrieval primary comparison declared in advance | venue_split and the primary comparison (venue_split − hybrid_rerank on venue_coverage@10, comparison questions) were fixed after the `v0.2-gold` run measured the coverage failure on the same 50 questions (`results/20260925T002151Z` already holds the result) | after results were seen | The method was designed in response to that failure | The test measures the detector-plus-split mechanism on the questions it was designed on; no held-out set tests it. | [findings](../docs/results/findings.md#splitting-retrieval-by-venue-raises-comparison-coverage-from-070-to-098), [ADR 0007](../docs/adr/0007-venue-split.md) |
| D14 | §9: venue_attribution_error as one metric | Reported in total and in two kinds, cross-exchange and through the CCXT wrapper, by the venues of the cited chunks and of the claim | after results were seen | Added after the judged answers were read: every misattributed claim cites CCXT. These are 2 of the 91 claims of the calibration sample of `results/20260925T175128Z_answers` that either grader marked, and all 34 claims Opus marks in `results/20260926T000001Z_judged` and `results/20260926T004836Z_judged`, on the questions of `v0.2.1-gold` | None on the judge's verdicts, which are unchanged; the calibration has no case of cross-exchange attribution, so results on that kind are unresolved. | §9 "Attribution errors by kind" |
| D15 | [ADR 0010](../docs/adr/0010-judge.md), §9: each judge candidate grades the 40 calibration answers a second time, for self-consistency | Both candidates graded the 40 calibration answers once and Opus was selected by the §9 rule (`calibration.md`); only Opus graded them a second time (`judgements_opus_r1.jsonl`). | after results were seen. The run instruction for the judging step was written after the human ratings and both candidates' first gradings (2026-09-25 UTC). It placed self-consistency after the judge selection (`calibration.md`) and for the selected judge only (the Opus repeat, 2026-09-26 UTC). | The instruction gave no reason for dropping the Sonnet repeat and did not record the departure from ADR 0010. | Sonnet's self-agreement is not measured. No result uses Sonnet's grades, and the noise rule reads only the selected judge's repeat, so no result changes. | §9; [ADR 0010](../docs/adr/0010-judge.md) |
| D16 | §7: a comparison result below the detectable δ is reported as inconclusive | The rule names no metric or scope. It follows from the power of the binary McNemar model for recall@10 on the comparison subgroup, and is applied there. §9 and §9a register their own decision rules for answers. A primary or secondary result is significant at a Holm-adjusted p below the significance level. H-v2 holds for a configuration at an adjusted p below it. §9b tests off-venue shares by §7's Holm rule. The table after this one lists every published result on `v0.2.1-gold` reported as an effect whose size is below δ, with the rule it was judged by. | after results were seen | The rule was written beside the recall@10 power calculation; the answer and off-venue results were judged by the rules registered for them. | One result changes: hybrid_rerank − bm25 on recall@10 over the comparison questions, first reported as a gain, is reported as inconclusive at this size. No other conclusion changes. | §7; [findings](../docs/results/findings.md#against-hybrid-a-recall10-loss-above-0033-from-the-reranker-is-excluded-at-95) |
| D17 | §9: each primary metric is tested on the questions both configurations answered and the judge graded | The coverage and `behaviour_correct` tests pair every comparison question on which both configurations have a judged answer record, refusals included. Coverage is graded on the answer text, a refusal's too, and `behaviour_correct` scores a refusal against the expected behavior. The venue_attribution_error test pairs the questions where both answers carry claims: 17 for the primary pair. | after results were seen | The analysis code paired the coverage and `behaviour_correct` tests on every judged record, and the venue_attribution_error test on the answers that carry claims, before the results existed. The protocol's word "answered" was read against it only when the findings used the word for an answer that does not refuse. | None: the coverage result is the test on every comparison question, as run, and venue_attribution_error is tested on the questions where both answers carry claims. Read as "not refused", the protocol would name, for coverage, a smaller, configuration-dependent set that no run tests. | §9; [findings](../docs/results/findings.md#under-prompt-v1-splitting-retrieval-by-venue-raises-answer-coverage) |
| D18 | §7: the exact two-sided McNemar test for outcomes binary per question, a top-10 hit and every named venue in the top-10 list | McNemar runs on venue_coverage@k and, in the exploratory comparisons of §9b, on the top-1 off-venue rate. No hit-in-the-top-10 metric is computed; recall@k, fractional for a question with more than one gold chunk, is tested by the paired randomization test. | after results were seen | The comparisons were built on recall@k and venue_coverage@k; a hit metric was never added, and the protocol sentence was compared with the comparison code after the results. | None: no published result is a McNemar test on a top-10 hit, and the recall results use the randomization test §7 registers for fractional outcomes. | §7; `evaluation/comparisons.py` |
| D19 | Decision 6 and §1: every `report.md` states `unjudged@k` per method and the condensed-list `ndcg_prime@k` | The `report.md` of `results/20260924T200525Z`, `results/20260924T234623Z` and `results/20260925T002151Z` has neither column, and that of `results/20260925T012314Z` has no `ndcg_prime@k`. Every later retrieval report has both. | after results were seen | These runs were made before the columns existed, and a committed run is not rewritten; their `results/runs.json` notes say so. | None on the published numbers: the findings read `unjudged@10` and nDCG′@10 from runs whose tables hold them. | §1; `results/runs.json` |
| D20 | §9: a judge dimension is measured on the 40 calibration answers | `behaviour_correct` against the human is measured on 39 of the 40: one answer is left out (`generation.calibration_exclude_behaviour`), and it stays in the other dimensions and in the agreement between the judges ([Calibration result](#calibration-result)). | after results were seen. The agreement with the answer included had been computed (κ 0.789 for Sonnet, 0.734 for Opus) on the calibration sample of `results/20260925T175128Z_answers`, drawn from the questions of `v0.2.1-gold`. | The rating screen showed the human the answer's `asked_clarification` flag as false where the judges saw it as true, so the human and the judges did not rate the same answer. | None on the judge selection or on which dimensions pass: both candidates pass `behaviour_correct` with the answer and without it. | §9 "Calibration result"; `configs/default.yaml` |
| D21 | §4: a `tickSize` for Hyperliquid is an example of a wrong-venue premise, a concept the venue does not have | Hyperliquid documents the tick size under its own fields: `crossvenue count-term` counts "tick size" in 4 Hyperliquid chunks and `szDecimals` in 26. The candidate built on that premise, `cv-0110 "what tick size should i round"` of `v0.1-candidates`, was mislabeled and was dropped in `v0.1.1-candidates`. | after collection, before analysis | The example names a concept the venue has under other names. | None on the results: the candidate was dropped before selection, and in that version `cv-0101 "what tick size does hyperliquid use"`, an adversarial question, tests the vocabulary correction. | [`CHANGELOG.md`](datasets/CHANGELOG.md#v011-candidates-2026-09-20) |
| D22 | §9a: venue_attribution_error and faithfulness are tested v2 − v1 by the §9 procedure | Faithfulness is not compared between v1 and v2: `compare-prompts` tests the wrong-refusal share and venue_attribution_error only. | after collection and before analysis | Faithfulness is graded on the judge's `supported` dimension, which failed calibration (§9): the calibration was recorded before either matrix was judged. | H-v2's guard "without lowering faithfulness" is not tested, so no conclusion of §9a uses faithfulness; H-v2 is decided on the wrong-refusal share and venue_attribution_error. | §9 "Calibration result"; `generation/prompt_compare.py` |
| D23 | §3: every gap is proved against the corpus before the question is kept, the concept searched under every spelling the vendor uses with `crossvenue count-term` | The questions [datasheet, known limitations](datasheet.md#known-limitations) lists are labeled unanswerable. `cv-0115 "does bybit have a mass cancel"` and `cv-0121 "which bingx endpoint returns the usdt-m"` have answers in the corpus; for the first, Bybit documents `POST /v5/order/cancel-all`. `cv-0106 "which binance endpoint sets leverage on"` was listed too and was found to have none. | after collection, before analysis: a question's unanswerable label is set when it is drafted, and selection kept these | Neither the datasheet nor the dataset CHANGELOG records why the searches did not find these answers. | They were reviewed after round 2: in `v0.2.2-gold` `cv-0115 "does bybit have a mass cancel"` and `cv-0121 "which bingx endpoint returns the usdt-m"` are fact, and `cv-0106 "which binance endpoint sets leverage on"` stays unanswerable, since CCXT's `setLeverage` chunk names Binance's page and gives no endpoint ([deviation D25](#deviations-from-protocol)). Every count on unanswerable questions in `v0.2.1-gold` includes them as labeled. No recall average on `v0.2.1-gold` includes them, because unanswerable questions enter none. | [datasheet](datasheet.md#known-limitations) |
| D24 | §9 "Conditions for using a judge dimension": coverage is held to a Spearman threshold because "the comparisons use coverage only to order configurations" | The primary and secondary comparisons on coverage test the size of the difference: a paired bootstrap interval of the mean difference and a paired randomization test on the differences (§9 "Primary comparison"). The findings report that size. | before collection: the reason and the test were registered together, before any answer was generated; the mismatch was found after results were seen | The reason was written with the threshold and was not compared with the registered test. | The Spearman threshold checks that the judge orders answers by coverage as the human does; it leaves the scale of the judge's coverage values unchecked. The sizes of the coverage differences depend on the judge's scale. | §9 "Conditions for using a judge dimension" |
| D25 | §5a: one written criterion for every label; §5: a third adjudication pass resolves the disagreements of round 2 | The adjudication of the 63 disagreeing pairs of round 2 and the relabeling of the D23 questions apply §5a as amended on 2026-10-10 ([Amendment of 2026-10-10](#amendment-of-2026-10-10)). Of the 89 questions outside the round-2 sample, 87 keep labels made under the round-1 criterion until the full relabel; the other 2 are D23 questions. | after results were seen: the amendment was written after κ was computed and the disagreements were read | The disagreements show the criterion ambiguous in the three cases the amendment decides, and §5 refines the criterion where the categories with the most disagreement show it ambiguous. | In `v0.2.2-gold` the 63 disagreeing pairs of the 25 sampled questions and the D23 questions follow the amended criterion; every other label follows the round-1 criterion, among them every label of 87 questions. κ is computed on round 1 and round 2 under the round-1 criterion, and the adjudication does not change it. | §5a; [§5b](#5b-annotation-round-3-pre-registered); [agreement](agreement.md) |
| D26 | Decision 2: the three constraints (comparison 50, unanswerable 15, nothing below 5) "do not fit in 100; 105 is what they sum to" | Over the 8 categories the three constraints sum to 95 (50 + 15 + 6 × 5), so a total of 100 holds them with no category below 5. The 105 of decisions 1 and 2 holds adversarial at its target of 12 and fact at 8, the targets selection used (`dataset/selection.py`, `TARGETS`). | before collection: the sum was registered with the category targets, before selection; the mismatch was found after results were seen | The sum was written beside the category split of decision 2 and was not recomputed from the three constraints alone. | None on the results: selection used the targets of decision 2 (D2), so no count, test or power figure changes. The total of 105 follows from the adversarial and fact targets as well as from the three constraints. | [Decisions at a glance](#decisions-at-a-glance) |

D16 covers these results on `v0.2.1-gold`, each from the run the table names, over the n questions its comparison pairs. Each is reported as an effect (an interval that excludes 0, or a p_Holm below the significance level) with a size below δ, and each difference has its 95% paired bootstrap interval:

| result | run | n | difference | p_Holm | reported in | decision rule |
|---|---|---:|---:|---:|---|---|
| hybrid_rerank − bm25, `recall@10`, comparison | `20260925T132613Z` | 50 | +0.155 [+0.021, +0.287] | 0.145 | [findings](../docs/results/findings.md#against-hybrid-a-recall10-loss-above-0033-from-the-reranker-is-excluded-at-95) | §7: below δ, inconclusive at this size |
| hybrid_rerank − bm25, `recall@10`, labeled | `20260925T132613Z` | 90 | +0.182 [+0.087, +0.277] | 0.004 | [findings](../docs/results/findings.md#against-hybrid-a-recall10-loss-above-0033-from-the-reranker-is-excluded-at-95) | §7: significant |
| hybrid_rerank − dense, `venue_coverage@10`, comparison | `20260925T132613Z` | 50 | +0.180 [+0.020, +0.340] | 0.255 | [findings](../docs/results/findings.md#against-hybrid-a-recall10-loss-above-0033-from-the-reranker-is-excluded-at-95) | §7: inconclusive at this size |
| venue_split_oracle − venue_split, `venue_coverage@10`, multi-venue | `20260925T132613Z` | 59 | +0.102 [+0.034, +0.186] | 0.119 | [findings](../docs/results/findings.md#venue-detection-finds-every-named-venue-on-47-of-50-comparison-questions) | §7: inconclusive at this size |
| gemini@venue_split − gemini@hybrid_rerank, `coverage`, comparison | `20260926T000001Z_judged` | 50 | +0.158 [+0.040, +0.279] | 0.024 | [findings](../docs/results/findings.md#under-prompt-v1-splitting-retrieval-by-venue-raises-answer-coverage), [README](../README.md) | §9 primary: significant |
| gemini@venue_split − qwen@venue_split, `coverage`, comparison | `20260926T000001Z_judged` | 50 | −0.180 [−0.270, −0.096] | < 0.001 | [findings](../docs/results/findings.md#qwen-has-the-highest-coverage-0621-and-mercury-refuses-7-of-50) | §9 secondary: significant |
| gemini@venue_split − qwen@venue_split, `behaviour_correct`, comparison | `20260926T000001Z_judged` | 50 | −0.180 [−0.300, −0.060] | 0.033 | [findings](../docs/results/findings.md#qwen-has-the-highest-coverage-0621-and-mercury-refuses-7-of-50) | §9 secondary: significant |
| qwen@venue_split − mercury@venue_split, `coverage`, comparison | `20260926T000001Z_judged` | 50 | +0.149 [+0.077, +0.221] | < 0.001 | [findings](../docs/results/findings.md#qwen-has-the-highest-coverage-0621-and-mercury-refuses-7-of-50) | §9 secondary: significant |
| qwen@venue_split − mercury@venue_split, `behaviour_correct`, comparison | `20260926T000001Z_judged` | 50 | −0.120 [−0.220, −0.020] | 0.144 | [findings](../docs/results/findings.md#qwen-has-the-highest-coverage-0621-and-mercury-refuses-7-of-50) | §9 secondary: inconclusive at this size |
| qwen@venue_split − mercury@venue_split, `venue_attribution_error`, all | `20260926T000001Z_judged` | 84 | +0.038 [+0.006, +0.079] | 0.753 | [findings](../docs/results/findings.md#qwen-has-the-highest-coverage-0621-and-mercury-refuses-7-of-50) | §9 secondary: inconclusive at this size |
| mercury@venue_split − gemini@hybrid_rerank, `coverage`, comparison | `20260926T000001Z_judged` | 50 | +0.189 [+0.078, +0.305] | 0.004 | [findings](../docs/results/findings.md#qwen-has-the-highest-coverage-0621-and-mercury-refuses-7-of-50) | §9 secondary: significant |
| gemini@venue_split v2 − v1, `wrong_refusal` | `20260926T004851Z_prompts` | 94 | −0.191 [−0.277, −0.117] | < 0.001 | [findings](../docs/results/findings.md#prompt-v2-lowers-wrong-refusals-in-every-configuration-exploratory) | §9a H-v2: significant |
| qwen@venue_split v2 − v1, `wrong_refusal` | `20260926T004851Z_prompts` | 94 | −0.191 [−0.277, −0.117] | < 0.001 | [findings](../docs/results/findings.md#prompt-v2-lowers-wrong-refusals-in-every-configuration-exploratory) | §9a H-v2: significant |
| mercury@venue_split v2 − v1, `wrong_refusal` | `20260926T004851Z_prompts` | 94 | −0.085 [−0.149, −0.032] | 0.008 | [findings](../docs/results/findings.md#prompt-v2-lowers-wrong-refusals-in-every-configuration-exploratory) | §9a H-v2: significant |
| hybrid_rerank − bm25, `off_venue@10`, comparison | `20260925T012314Z` | 50 | −0.178 [−0.226, −0.134] | < 0.001 | [findings](../docs/results/findings.md#unnamed-venues-fill-0036-of-hybrid_reranks-comparison-top-10-and-0214-of-bm25s-exploratory) | §9b: significant |
| hybrid_rerank − bm25, `off_venue@1`, comparison | `20260925T012314Z` | 50 | −0.160 [−0.260, −0.060] | 0.026 | [findings](../docs/results/findings.md#unnamed-venues-fill-0036-of-hybrid_reranks-comparison-top-10-and-0214-of-bm25s-exploratory) | §9b: significant |
| hybrid_rerank − hybrid, `recall@10`, labeled | `20261010T222553Z` | 90 | +0.088 [+0.002, +0.174] | 0.202 | [findings](../docs/results/findings.md#on-v022-gold-no-retrieval-comparison-changes-sign-significance-or-whether-its-interval-excludes-0) | §7: inconclusive at this size |

## References

### Test collections, pooling and incompleteness
- Zobel, J. (1998). *How reliable are the results of large-scale information retrieval experiments?* SIGIR '98, 307–314. [doi:10.1145/290941.291014](https://doi.org/10.1145/290941.291014)
- Buckley, C., Dimmick, D., Soboroff, I. & Voorhees, E. (2007). *Bias and the limits of pooling for large collections.* Information Retrieval 10(6):491–508. [doi:10.1007/s10791-007-9032-x](https://doi.org/10.1007/s10791-007-9032-x)
- Büttcher, S., Clarke, C. L. A., Yeung, P. C. K. & Soboroff, I. (2007). *Reliable information retrieval evaluation with incomplete and biased judgments.* SIGIR '07, 63–70. [doi:10.1145/1277741.1277755](https://doi.org/10.1145/1277741.1277755)
- Buckley, C. & Voorhees, E. M. (2004). *Retrieval evaluation with incomplete information.* SIGIR '04, 25–32. [doi:10.1145/1008992.1009000](https://doi.org/10.1145/1008992.1009000) · [NIST PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469)
- Sakai, T. (2007). *Alternatives to Bpref.* SIGIR '07, 71–78. [doi:10.1145/1277741.1277756](https://doi.org/10.1145/1277741.1277756)
- Voorhees, E. M., Soboroff, I. & Lin, J. (2022). *Can Old TREC Collections Reliably Evaluate Modern Neural Retrieval Models?* [arXiv:2201.11086](https://arxiv.org/abs/2201.11086)
- Allan, J., Harman, D., Kanoulas, E., Li, D., Van Gysel, C. & Voorhees, E. (2017). *TREC 2017 Common Core Track Overview.* TREC 2017. [PDF](https://trec.nist.gov/pubs/trec26/papers/Overview-CC.pdf)
- [TREC: English relevance judgments](https://trec.nist.gov/data/reljudge_eng.html)

### Synthetic question generation and its limits
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. & Gurevych, I. (2021). *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
- Saad-Falcon, J., Khattab, O., Potts, C. & Zaharia, M. (2024). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* NAACL. [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) · [code](https://github.com/stanford-futuredata/ARES)
- Ragas. [Testset Generation for RAG](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/) · [source](https://github.com/explodinggradients/ragas)
- Hong, K., Troynikov, A., Huber, J. & McGuire, M. (2025). *Generative Benchmarking.* Chroma Research. [report](https://www.trychroma.com/research/generative-benchmarking)
- Kucia, F. J. & Gawlik, B. M. (2026). *Beyond Benchmark Scores: How Synthetic and Authentic Query Distributions Diverge in RAG Evaluation.* [arXiv:2609.14579](https://arxiv.org/abs/2609.14579)
- van Elburg, J., van der Putten, P. & Marx, M. (2025). *Can we Evaluate RAGs with Synthetic Data?* SynDAiTE workshop, ECML-PKDD. [arXiv:2508.11758](https://arxiv.org/abs/2508.11758)
- Rahmani, H. A., Ramineni, V., Yilmaz, E., Craswell, N. & Mitra, B. (2025). *Towards Understanding Bias in Synthetic Data for Evaluation.* CIKM. [arXiv:2506.10301](https://arxiv.org/abs/2506.10301)
- Filice, S., Horowitz, G., Carmel, D. et al. (2025). *Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana.* [arXiv:2501.12789](https://arxiv.org/abs/2501.12789)

### Unanswerable, adversarial and conflicting evidence
- Rajpurkar, P., Jia, R. & Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL. [arXiv:1806.03822](https://arxiv.org/abs/1806.03822) · [leaderboard](https://rajpurkar.github.io/SQuAD-explorer/)
- Kwiatkowski, T. et al. (2019). *Natural Questions: A Benchmark for Question Answering Research.* TACL 7:453–466. [doi:10.1162/tacl_a_00276](https://doi.org/10.1162/tacl_a_00276) · [ACL Anthology](https://aclanthology.org/Q19-1026/)
- Chen, J., Lin, H., Han, X. & Sun, L. (2024). *Benchmarking Large Language Models in Retrieval-Augmented Generation.* AAAI. [arXiv:2309.01431](https://arxiv.org/abs/2309.01431)
- Min, S., Michael, J., Hajishirzi, H. & Zettlemoyer, L. (2020). *AmbigQA: Answering Ambiguous Open-domain Questions.* EMNLP, 5783–5797. [doi:10.18653/v1/2020.emnlp-main.466](https://doi.org/10.18653/v1/2020.emnlp-main.466) · [ACL Anthology](https://aclanthology.org/2020.emnlp-main.466/)
- Wang, H., Prasad, A., Stengel-Eskin, E. & Bansal, M. (2025). *Retrieval-Augmented Generation with Conflicting Evidence.* COLM. [arXiv:2504.13079](https://arxiv.org/abs/2504.13079)

### Agreement
- Landis, J. R. & Koch, G. G. (1977). *The Measurement of Observer Agreement for Categorical Data.* Biometrics 33(1):159–174. [doi:10.2307/2529310](https://doi.org/10.2307/2529310)
- [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa): collects the standard critiques of the Landis & Koch bands.

### Deduplication and contamination
- Lee, K., Ippolito, D., Nystrom, A. et al. (2022). *Deduplicating Training Data Makes Language Models Better.* ACL. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- Abbas, A., Tirumala, K., Simig, D., Ganguli, S. & Morcos, A. S. (2023). *SemDeDup: Data-efficient learning at web-scale through semantic deduplication.* [arXiv:2303.09540](https://arxiv.org/abs/2303.09540)
- Broder, A. Z. (1997). *On the resemblance and containment of documents.* Compression and Complexity of Sequences. [doi:10.1109/SEQUEN.1997.666900](https://doi.org/10.1109/SEQUEN.1997.666900)
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners* (13-gram contamination analysis, appendix C). NeurIPS 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- OpenAI (2023). *GPT-4 Technical Report* (50-character substring contamination check). [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)

### Sample size and significance
- Voorhees, E. M. & Buckley, C. (2002). *The effect of topic set size on retrieval experiment error.* SIGIR '02, 316–323. [doi:10.1145/564376.564432](https://doi.org/10.1145/564376.564432) · [NIST record](https://www.nist.gov/publications/effect-topic-set-size-retrieval-experiment-error)
- Sakai, T. (2016). *Topic set size design.* Information Retrieval Journal 19(3):256–283. [doi:10.1007/s10791-015-9273-z](https://doi.org/10.1007/s10791-015-9273-z)
- Smucker, M. D., Allan, J. & Carterette, B. (2007). *A comparison of statistical significance tests for information retrieval evaluation.* CIKM '07, 623–632. [doi:10.1145/1321440.1321528](https://doi.org/10.1145/1321440.1321528)
- Smucker, M. D., Allan, J. & Carterette, B. (2009). *Agreement among statistical significance tests for information retrieval evaluation at varying sample sizes.* SIGIR '09 (poster), 630–631. [doi:10.1145/1571941.1572050](https://doi.org/10.1145/1571941.1572050)

### Retrieval and documentation formats
- Cormack, G. V., Clarke, C. L. A. & Büttcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR '09, 758–759. [doi:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)
- Gebru, T. et al. (2021). *Datasheets for Datasets.* Communications of the ACM 64(12):86–92. [doi:10.1145/3458723](https://doi.org/10.1145/3458723)

### Not verified / left open
- The Smucker, Allan & Carterette 2009 poster is quoted from its abstract: disagreement grows as topics fall toward 10, with no threshold, so none is quoted.
- The exact null-answer percentages for Natural Questions were not confirmed against the paper, so no NQ proportion is quoted.
- The two caveats on κ in §5, and the memory-decay statements behind the §5 gap, are from the reliability literature in general, and none was checked against a specific source. The first caveat is that Cohen's κ computes chance agreement from each rater's own marginal rates, so rater bias does not lower it, and the intraclass kappa is the statistic for test-retest. The second is that κ is unstable on small, prevalence-imbalanced tables.
- Cohen's conventional small, medium and large effect sizes for *d* in §7 were not checked against a work of Cohen's, so no reference entry is given for them.

