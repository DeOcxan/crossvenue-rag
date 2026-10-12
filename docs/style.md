# Writing style

Rules for every Markdown file in the repository; the rules for the Python code are in [`code-style.md`](code-style.md). The base is the [Google developer documentation style guide](https://developers.google.com/style); where this file differs from it, this file applies. Every rule is checked by reading; the tools in [Checks](#checks) are aids.

The governing rule is that every sentence carries a fact, a number, or a decision with its reason. A sentence that carries none of these is deleted.

Examples of what to avoid are in code font, so the linter skips them.

## Document types

Each document is one type, except that a decision record is also history. The table says what a type contains and what it leaves to another document; the layout is in [`docs/README.md`](README.md).

| Type | Files | Contains | Leaves out |
|---|---|---|---|
| overview | `README.md` | what the project is, how to run it, headline results with their runs | full tables, rationale, history |
| explanation | `docs/explanation/` | how the pipeline is built and why, with the alternatives | procedures, result tables (links to findings instead) |
| reference | `docs/reference/`, `evals/datasheet.md`, `evals/methodology.md`, `evals/domain-research.md` | facts to look up: datasheets, commands, config keys, the evaluation protocol and its deviations, and the measurements the protocol's own rules act on (the judge calibration, the counts that set the question set) | steps to follow, other results |
| results | `docs/results/`, and `evals/agreement.md` at the location methodology §5 registers for it | measured results, each with its run | design rationale (links to explanation instead) |
| how-to | `docs/how-to/` | numbered steps to reach one goal | explanation beyond one sentence per step |
| decision records | `docs/adr/` | one decision, its context, the alternatives and the consequences | results measured later (links to findings) |
| history | `evals/datasets/CHANGELOG.md`, `docs/adr/` | the order in which versions and decisions were made | anything a reader needs to use the current version |
| table of contents | `docs/README.md`, `evals/README.md`, `docs/adr/README.md` | each document of its directory, with what it holds | the content itself |
| rules | `docs/style.md`, `docs/code-style.md` | the rules for documents and code, and the tools that check them | how any document or module came to follow them |

Results are not one of Diátaxis's four types; `docs/results/` is a fifth section of this project, written with reference discipline: neutral, complete, one layout per kind of table.

## Time and history

1. Documents describe the repository as it is. How it came to be so is recorded in the ADRs, the dataset CHANGELOG and git. The exceptions are rules 4, 5 and 43: a version comparison, an order-of-analysis disclosure and a deviation stay with the result they qualify.
   - Bad: `Since 2026-09-25 venue_split is one of POOL_METHODS.`
   - Good: `venue_split` is one of `POOL_METHODS`.

2. No time-relative words: `now`, `currently`, `new`, `latest`, `soon`, `recently`, `previously`, `no longer`, `at present`, `as of this writing`, `after the fix`, `first version`, `week 1`, `since <date>`. `new` is replaced with the version name or with `another`. The ADRs and `evals/datasets/CHANGELOG.md` are history documents and are exempt from this rule, as `.vale.ini` configures.

3. Present tense for what the system does and what a result is. Past tense for a procedure that was carried out once: labeling rounds, selection, calibration, registration.
   - Good: Round 1 pooled four methods. The pool depth is 20.

4. A version comparison is a result and belongs in the results: v1 against v2, with both numbers and the difference.
   - Good: Under prompt v2, gemini@venue_split refuses 17 of the 94 questions that expect an answer or a clarifying question, against 35 under v1.

5. An order-of-analysis disclosure stays wherever the result is reported: what was fixed before or after which data were seen. It changes what the result can prove.
   - Good: Prompt v2 was written after reading the v1 answers, so this section is exploratory.

6. A defect that changed no reported number is not documented. A defect that changed a reported number, or data, is a deviation in `evals/methodology.md` or a CHANGELOG entry.
   - Bad: `The labelling tool at one point took 10 per method and showed 20.`

7. A refuted assumption is a result, stated with its number; the story of finding it is left out.
   - Bad: `The expectation that this makes it the strongest method is refuted.`
   - Good: On the comparison questions BM25's recall@10 is 0.465, against 0.620 for hybrid_rerank.

8. Dates appear only as provenance: a pre-registration, the retrieval date of an external count (GitHub issue counts, stars), the freeze date of a dataset version, a scheduled event. A scheduled event is rewritten as a record once it has happened. A date is also provenance when a rule depends on it (the retest gap between labeling rounds). So is a date that discloses the order of analysis (a rule fixed before answers were judged, a section added after results were read).

## Reporting numbers

9. Every result names its run (`results/<timestamp>`), its dataset version, and n with its unit (questions, claims, answers). They are stated at the start of each results section and in the caption of every table, the sentence directly above it; prose inside the section inherits them. The first interval in a section states its level and method; a table caption does the same for the table.

10. A difference between methods or configurations is written A − B with its sign and interval: +0.280 [+0.160, +0.400]. A change caused by another gold version or a recount is written with its sign and no interval: −0.092. The direction is stated once per table.

11. Precision:

    | Quantity | Format | Example |
    |---|---|---|
    | means and differences of metrics in [0, 1], κ, Spearman | three decimals, interval at the same precision | 0.620, κ 0.238 [−0.051, 0.558] |
    | a share of a small count | the share at two decimals and the count; a heading may omit the count | 0.98 (49/50) |
    | power-analysis parameters: detectable δ, power, probabilities | two decimals: they are design values that no run measures | δ = 0.20 at power 0.80 |
    | p-values | three decimals; below 0.001 as `< 0.001` | p_Holm = 0.024, p < 0.001 |
    | descriptive shares of the corpus or the set | percent, one decimal | 10.8% |
    | latency | ms, integer from 10 ms, one decimal below | 2,531 ms, 0.1 ms |
    | model-call duration | seconds, integer; it is not retrieval latency | 243 s |
    | human annotation time | seconds, as measured; it is not latency | 17 s per question |
    | cost | USD, three decimals | $0.657 |
    | counts | numerals, thousands separator | 11,934 chunks |
    | a ratio or mean of counts | the decimals its constant in `formatting.py` sets, one constant per kind | 2.47 tokens per word, 7 to 138 times |

    The conventions "80% power" and "α = 0.05" keep their literature form.

    Seeds, config values, identifiers and quoted documentation keep their own form: `20260924`, `60000`.

12. Units follow the number after a space: 2,531 ms, 400 MB. Percent takes no space: 10.8%. Dates are ISO 8601: 2026-09-25.

## Wording a claim

13. The wording follows the evidence:

    | What the data show | Wording |
    |---|---|
    | primary comparison, interval excludes 0 | "venue_split raises venue_coverage@10 by +0.280 [+0.160, +0.400]" with its p-value |
    | secondary comparison, p_Holm < 0.05 | the same, with p_Holm and the word secondary where the family is not obvious |
    | interval includes 0 | "inconclusive at this size", the interval, and the smallest difference the test detects at 80% power when it is known |
    | the interval bounds a loss | "a loss larger than 0.033 is excluded at 95%" |
    | equivalence | only against a margin fixed before the data; otherwise not claimed |
    | analysis chosen after the data were seen | "exploratory", in the heading or the first sentence |

    Never `no difference`, `no effect`, or `not significant` on its own: readers hear "no difference".

14. A claim is scoped to what was measured: this corpus, this judge, this prompt, this n.
    - Bad: `An LLM judge does not measure faithfulness.`
    - Good: Opus, as judge, agrees with the human on 2 of the 10 claims the human rejects (κ 0.238).

15. A mechanism that no run measures is either stated as an expectation with the measurement that would test it, or removed.
    - Bad: `That makes the pair a sharp test of dense retrieval and a weak test of BM25.`

16. A results heading states the result and its direction, without evaluative verbs.
    - Bad: `Splitting retrieval by venue fixes comparison coverage`
    - Good: Splitting retrieval by venue raises comparison coverage from 0.70 to 0.98

17. Quantifiers carry their count: `almost always`, `most`, `rarely` become "15 of 16".

## Sentences and words

18. Active voice. The subject stands next to its verb; known information comes first and the number being reported last.
    - Good: On the 48 questions it splits, venue_split's p50 is 5,408 ms.

19. No `we` and no `our`. Procedures in how-to documents address the reader in the imperative; everything else is impersonal.

20. US English in prose: labeling, normalization, randomization, behavior, judgment, artifact. An identifier keeps its own spelling in code font: `behaviour_correct`, `judgements_opus_r1.jsonl`.

21. Words that carry no fact are deleted: `important`, `crucial`, `key` (as an adjective), `notably`, `essentially`, `robust`, `seamless`, `comprehensive`, `leverage` (as a verb), `delve`, `underscore`, `showcase`, `pivotal`, `it is worth noting`, `in summary`, `in other words`, `furthermore`, `moreover`, `simply`, `just`, `easy`, `obviously`, `clearly`.

22. No figurative verbs for things: `lives in`, `sits`, `owns`, `surfaces`, `buys`, `does the heavy lifting`, `load-bearing`, `serves as`, `stands as`. They are replaced with `is`, `has` or `is in`.
    - Bad: `50 buys a detectable δ of 0.20.`
    - Good: At n = 50 the smallest detectable δ is 0.20.

23. One hedge per claim, and only a hedge the data justify: no `could potentially`, `may possibly`, `arguably`.

24. `for example` and `that is`, never `e.g.` and `i.e.`

25. No em dash. An en dash marks a range only (0.03–0.09). A table cell with no value holds a single dash.

26. No `not X, but Y` framing and no trailing `, not X.` A sentence states what is.
    - Bad: `The overlap is conceptual, not lexical.`
    - Good: The two share concepts and differ in wording: median best-match cosine 0.143.

27. No rhetorical lead-in with a colon, no rhetorical question.
    - Bad: `The answer: split by venue.`

28. No rule of three by habit: list as many items as there are facts. No closing sentence that repeats the section or states its significance.

29. Sentence length varies. A sentence over 35 words is split unless it is a list of values.

## Formatting

30. A paragraph is one line: no hard wrapping. A list item is one line. Diffs are reviewed with `git diff --word-diff`.

31. One H1 per file, no skipped heading levels, sentence case. How-to headings are imperative (Reproduce the benchmark); explanation and reference headings are noun phrases; results headings are the result (rule 16). In a results document an H2 that groups results may be a noun phrase. Each H3 under it states its result, except a subsection listing the limits of the results above it, which may be a noun phrase.

32. Lists use `-`. A list holds like items. No bold or italic label at the start of every item: items with several attributes each are a table.

33. No bold in running text. Italic is rare and marks a term at its definition, a cited title or a math variable (*d*, *k*, *n*).

34. No horizontal rules between sections. No emoji and no decorative arrows or check marks.

35. Code font for file names, paths, commands, fields, config keys, metric identifiers in formulas, run ids and hashes.

36. Links inside the repository are relative and resolve, anchors included. Link text names the target.
    - Bad: `see [here](../evals/methodology.md#7-set-size-and-statistical-power)`, `[README.md](../evals/methodology.md#7-set-size-and-statistical-power)`
    - Good: [methodology §7](../evals/methodology.md#7-set-size-and-statistical-power)

37. A hash is shortened to eight hex characters with `…` only when the full value is in a file the sentence names.

## Citing

38. A question is cited as its id and the first six words of its text, checked against the question file: `cv-0045 "do binance and bybit ban you"`. An id alone is not enough: ids are positional, and a renumbered set would make it point at another question.

39. A paper is cited with authors, year, title, venue and a DOI or arXiv link. A preprint's venue is its arXiv link; a technical report with neither DOI nor arXiv is linked where its publisher hosts it. A statement attributed to a source is checked against the source; one that is not checked is listed under "Not verified" in that document.

40. A count from a live external source (GitHub search, a leaderboard) carries the date it was retrieved.

## Specific documents

41. ADRs follow Nygard (Title, Status, Context, Decision, Consequences) with a Date line; an Alternatives section is allowed. An accepted ADR is not edited, except its status, links and typos. An ADR is frozen from the first published commit; before it, ADRs follow this file like any other document. A changed decision is another ADR, and the old one's status becomes "Superseded by NNNN".

42. `evals/datasets/CHANGELOG.md` has one entry per version, newest first, headed `## <version> (<freeze date>)`. The first paragraph says what changed, with counts; details follow. Text that the records carry in their own fields is referenced by field name.

43. A deviation in `evals/methodology.md` is a departure of the data or the analysis from the registered protocol. A draft of the protocol revised before the data it governs existed is not a deviation. Each deviation records the protocol, what was done, when relative to the data (before collection, after collection and before analysis, after results were seen), why, and the effect on the conclusions.

44. Prompt files under `prompts/` are versioned artifacts: their sha256 is pinned in the config and written into every answer and judgment. They are never edited; a change is another version.

## Files outside these rules

Files under `results/` are written by a run and their sha256 is in `results/runs.json`; files under `prompts/` are pinned by sha256 (rule 44). Neither is ever edited. The rules apply to the templates in `src/` that write run reports, for future runs. The Markdown under `tests/fixtures/` is vendor-format input the tests parse as it is.

## Checks

Every rule is checked by reading. The tools are aids: each runs for the rules its row names, and a clean run does not show that a document meets them. `make lint-docs` runs Vale over `README.md`, `docs/` and `evals/` with the styles in `.vale/styles/crossvenue/`; `.vale.ini` sets `MinAlertLevel = warning`, and any alert fails the target. Code spans, code blocks and italic text are not linted (`IgnoredScopes`). A verbatim quotation, such as a GitHub issue title, keeps its own words and punctuation against rules 2, 20 and 25; Vale is switched off around it with `<!-- vale … = NO -->` comments.

| Tool | Rules it runs for |
|---|---|
| Vale, Timeless (off for the history documents, as rule 2 says) | 2 |
| Vale, Claims | 13, 16 |
| Vale, Hedges | 17, 23 |
| Vale, FirstPerson | 19 |
| Vale, USSpelling | 20 |
| Vale, Filler and Minimizers | 21 |
| Vale, Figurative | 22 |
| Vale, Latin | 24 |
| Vale, EmDash and EnDash | 25 |
| Vale, NotButY and NotComma | 26 |
| Vale, ColonLeadIn | 27 |
| Vale, Inflation | 21, 28 |
| Vale, SentenceLength, at suggestion level, which `make lint-docs` does not show; `vale --minAlertLevel=suggestion README.md docs evals` lists it | 29 |
| Vale, Headings | 31 |
| Vale, Bold | 33 |
| Vale, Emoji | 34 |
| `tests/test_repo_hygiene.py`, over every published Markdown file, the committed run reports included | 30, 31, 32, 34 |
| lychee, offline, over `README.md`, `docs/` and `evals/`: every relative link and anchor resolves | 36 |
| `make reproduce`, over the documents `reporting/reproduce.py` lists | 9, 10, 11 |

Every number written in prose is re-derived from its source, never copied from another document, and every copy of a number is updated in the same change.
