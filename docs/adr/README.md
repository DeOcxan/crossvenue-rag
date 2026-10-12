# Architecture decision records

One file per decision, in [Nygard's format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) (Title, Status, Context, Decision, Consequences), with a Date line this repository adds. A date is the local calendar day (UTC−3) the decision was taken. A record is frozen from the first published commit ([style rule 41](../style.md)); after it, a changed decision is another record, and the old one's status becomes "Superseded by NNNN".

| # | Decision | Status |
|---|---|---|
| [0001](0001-no-framework.md) | Plain Python pipeline, no RAG framework | Accepted |
| [0002](0002-structural-chunking.md) | Heading-aware chunking with a budget in the embedder's tokens | Accepted |
| [0003](0003-exact-deduplication.md) | Exact deduplication with provenance, never across venues | Accepted |
| [0004](0004-index-version.md) | `index_version` and the gates on labels and results | Accepted |
| [0005](0005-labeling-pool.md) | The labeling pool: methods, depth 20, 25 shown, augmentation | Accepted |
| [0006](0006-single-annotator.md) | Human labeling by one annotator, reliability by test-retest κ | Accepted |
| [0007](0007-venue-split.md) | `venue_split` retrieval and dictionary venue detection | Accepted |
| [0008](0008-package-structure.md) | Package by domain, layering enforced by import-linter | Accepted |
| [0009](0009-generators.md) | Three generators from three model families, through OpenRouter | Accepted |
| [0010](0010-judge.md) | Two judge candidates, chosen by agreement with a human | Accepted |
| [0011](0011-preregistration.md) | Answer comparisons registered before the run | Accepted |
| [0012](0012-deferred-corpus-changes.md) | Corpus corrections deferred to the next corpus version | Accepted |
| [0013](0013-generator-prompt-v2.md) | A second generator prompt, analyzed as exploratory | Accepted |
