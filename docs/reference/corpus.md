# Corpus datasheet

A datasheet for the document corpus, following *Datasheets for Datasets* (Gebru et al., 2021, [doi:10.1145/3458723](https://doi.org/10.1145/3458723); full citation in the [methodology references](../../evals/methodology.md#references)). Every count is from the corpus fetched on 2026-09-20 and ingested at `index_version` `2825a7160c4d09bf`. `make stats` prints the corpus tables, and `make reproduce` checks the other counts against the corpus or lists them as given. `data/manifest.csv` holds per-document provenance.

## Motivation

Purpose. To measure where retrieval over API documentation confuses one venue with another. The documentation of the seven venues and the CCXT library describes the same kinds of object (orders, rate limits, funding, fees), where a source documents them, with different numbers underneath. The corpus exists to count how often a retriever returns a paragraph of a venue the question does not name.

Creator and funding. The repository's author assembled the corpus for this project; no external funding.

## Composition

Instances. A document is one markdown page of a venue's API documentation. A chunk is a heading-bounded, token-budgeted passage of a document; retrieval, labeling and scoring work on chunks.

| Venue | Type | Acquisition | License | Docs | Chunks | Tokens | p50 | p95 |
|---|---|---|---|---:|---:|---:|---:|---:|
| [CCXT](https://github.com/ccxt/ccxt) | cex (library) | git, sparse `wiki/` | MIT | 8 | 1,436 | 316,995 | 212 | 434 |
| [Bybit](https://github.com/bybit-exchange/docs) | cex | git, sparse `docs/` | no license found | 570 | 3,146 | 661,895 | 166 | 446 |
| [Binance](https://github.com/binance/binance-spot-api-docs) | cex | git | no license found | 39 | 1,189 | 326,696 | 281 | 449 |
| [BingX](https://github.com/BingX-API/api-ai-skills) | cex | git | MIT (`package.json`) | 34 | 507 | 101,002 | 178 | 429 |
| [Uniswap](https://github.com/Uniswap/docs) | dex | git, sparse `content/` | MIT | 198 | 1,782 | 382,684 | 172 | 441 |
| [dYdX v4](https://github.com/dydxprotocol/v4-documentation) | dex | git, sparse `docs/pages/` | no license found | 398 | 1,044 | 189,496 | 146 | 424 |
| [Hyperliquid](https://hyperliquid.gitbook.io/hyperliquid-docs) | dex | GitBook `llms.txt` | no license found | 148 | 809 | 147,371 | 126 | 437 |
| [PancakeSwap](https://docs.pancakeswap.finance/) | dex | GitBook `llms.txt`, English only | no license found | 270 | 2,021 | 358,738 | 138 | 422 |
| total | - | - | - | 1,665 | 11,934 | 2,484,877 | 173 | 442 |

Token counts use the embedding model's tokenizer (`BAAI/bge-base-en-v1.5`); p50 and p95 are chunk lengths in tokens. CCXT is a client library rather than a venue. It maps one unified vocabulary (`createOrder`, `fetchFundingRate`) onto venues whose semantics differ, so one term in its pages covers the differing mechanisms of every venue it maps. PancakeSwap v2 and v3 are forks of Uniswap v2 and v3, documented in PancakeSwap's own text ([findings](../results/findings.md#pancakeswaps-median-best-match-cosine-to-uniswap-is-0143)).

Chunk lengths in tokens, as `make stats` bins them (lower bound included, upper bound excluded): 0–49: 659; 50–99: 2,596; 100–199: 3,394; 200–299: 2,201; 300–399: 1,464; 400–499: 1,463; 500–749: 100; 750–999: 43; 1,000 and over: 14. Over the 450-token budget are 18 prose chunks and 191 fenced code blocks, kept whole on purpose.

Structural metadata. Each chunk carries `environment`, `api_version`, `product` and `doc_type`, read off paths and headings by rules in `data/sources.yaml`. Share of chunks with a value: `doc_type` 88.5%, `api_version` 44.8%, `product` 38.4%, `environment` 10.9%. A null means no rule gave a value: none matched the page, or a stop rule (`value: null`) matched it; [design](../explanation/design.md#structural-metadata) lists the null environments per venue.

Duplicate text. 12,783 chunk occurrences are stored as 11,934 chunks: 536 groups of byte-identical text within a venue, 433 of them Binance mainnet pages against their `testnet/` copies. Every occurrence is kept in the chunk's `sources`. No text occurs on two venues ([findings](../results/findings.md#108-of-chunk-occurrences-are-byte-identical)).

Included on purpose.

- `binance/testnet/**`, a near-verbatim copy of the mainnet pages. Where a testnet chunk is byte-identical to a mainnet one, the two are one chunk citing the mainnet page with the testnet page in `sources` (433 chunks). 349 Binance chunks occur only on testnet pages: 227 from the mirrored reference pages, 1 from the testnet terms page and 121 from `testnet/CHANGELOG.md`. The mirrored reference pages differ from mainnet in titles, hosts and limits and are the version-confusion trap.
- Binance's changelogs and terms pages, which the include globs match because fnmatch's `*` crosses `/`: `testnet/CHANGELOG.md` (121 chunks), `demo-mode/CHANGELOG.md` and `demo-mode/general-info.md` (6 each), and the three terms-of-use pages (1 each), one of them under `demo-mode/`. Unlike Bybit's changelog below, gold labels point into them (`cv-0148 "do i still need a listenKey"`, `cv-0093 "binance demo trading api, is that"`).
- `bybit/docs/v3`, superseded by v5 and seven documents wide: kept so that a question can name a superseded version.

Excluded.

| Excluded | Why |
|---|---|
| `bybit/docs/changelog/**` | A single file naming every endpoint and every limit Bybit has changed. |
| `bybit/docs/api-explorer/**` | OpenAPI JSON in MDX frontmatter; normalization keeps only the title, and every page duplicates a reference page. |
| `binance/**/*_CN.md` | Chinese translations of documents already in the corpus. |
| `dydx/old-docs/**` | Describes v3, a different system with the same nouns. |
| PancakeSwap non-English | `llms.txt` lists fifteen translations; the same text in fifteen languages measures the wrong thing. |

Known gaps.

- Binance covers spot only. `developers.binance.com` is a compiled SPA with no documentation repository and no `llms.txt`. A question about Binance perpetual funding is therefore unanswerable from this corpus. The other perpetual venues state different intervals. Hyperliquid's is 1 hour ("funding interval is 1 hour": 1 chunk; "every hour": 4, one of them, `42436e891f0bc10a`, about the interest on borrowing and not funding), and dYdX charges funding every hour ("every hour": 2). Bybit sets it per symbol, in minutes, in the instruments endpoint's `fundingInterval`. BingX's 20 chunks containing "funding" give `nextFundingTime` and no interval. Counts are chunks containing the term, case-insensitive, per venue (`crossvenue count-term TERM --venue X`).
- BingX is its AI-skills repository. `bingx-api.github.io` compiles its markdown into JS bundles and `BingX-swap-api-doc` is a one-line redirect; the corpus uses `BingX-API/api-ai-skills` from the same organization: per-domain endpoint references plus shared pages for rate limits, error codes and authentication. It is the smallest venue, at 507 chunks.

Known limitations, measured on the index and kept until another corpus version ([ADR 0012](../adr/0012-deferred-corpus-changes.md) records each fix and what it changes):

- 4,522 of the 8,820 chunks that start at a heading miss at least one of their parent headings: the breadcrumb is written only into chunks that do not start with a heading. The two Binance "Timing security" chunks, one from the REST API and one from the WebSocket API, name neither. The fix rewrites the text, and so the id, of those 4,522 chunks.
- The Binance include globs let fnmatch's `*` cross `/`, so they match the changelogs and terms pages listed under "Included on purpose". Narrowing the globs removes them, and with them the gold labels that point into them.
- PancakeSwap is served on its own domain, so the path mapping drops the first real URL segment: `welcome-to-pancakeswap` (102 pages), `trade` (67), `earn` (33), `archive` (31) and smaller sections. The 206 chunks of the 31 archived pages lose the `archive` marker in `doc_path`, and the metadata rules type them from the rest of the path. The fix changes every PancakeSwap `local_path` in the manifest and the metadata of those 206 chunks; the chunk text does not change.
- Normalization keeps GitBook page furniture: 270 PancakeSwap chunks (61,955 tokens) are the "Querying This Documentation" section GitBook adds for AI agents, and every PancakeSwap and Hyperliquid page opens with an `llms.txt` banner. Stripping them removes the 270 chunks.
- 153 chunks exceed 510 tokens of text, the reading window of the bi-encoder (512 with its two special tokens) (`bge-base-en-v1.5`); the dense index does not embed their tail. They are code blocks kept whole. The cross-encoder (`bge-reranker-base`) reads the query and the chunk in one window of 512 tokens with four special tokens, so less of each chunk fits in its window than in the bi-encoder's.
- 18 dYdX pages normalize to headings alone or to nothing and yield no chunk, so 1,647 of the 1,665 documents are in the index.
- Normalization collapses runs of blank lines inside 19 fenced code blocks; no other change reaches code. Keeping them changes the text and id of the chunks holding those blocks.

Sensitive content. None: public technical documentation, no personal data.

## Collection process

Two acquisition modes, both reading markdown the vendor publishes:

- `git`: a clone at depth 1 (`--depth 1`) with no blobs until checkout (`--filter=blob:none`), sparse where the source sets `sparse_paths`, of the repository `data/sources.yaml` names. `make fetch` clones with `--no-checkout`, fetches the commit the manifest records by its sha at depth 1, checks it out, and compares the sha256 of every file with the manifest. It therefore rebuilds this corpus after the upstream branch has moved. `crossvenue fetch --follow-branch` clones the branch head instead and rewrites the manifest, which starts another corpus version. Blobless matters for CCXT, whose repository, generated SDKs included, is 8,537,212 KB by the GitHub API's `size` field (retrieved 2026-09-26); with a sparse checkout of `wiki/`, only the wiki blobs are downloaded.
- `gitbook`: Hyperliquid and PancakeSwap have no documentation repository. Both are GitBook spaces that publish `/llms.txt`, an index of per-page `.md` URLs; fetching those gives the markdown the site renders, with no HTML cleanup. Both `robots.txt` files, checked on 2026-09-26, give `User-agent: *` with `Allow: /` and no `Disallow`, so fetching the pages is permitted; their `Content-Signal` line is a separate statement about use. Requests are serialized with `fetch.request_delay_seconds`. `make fetch` compares each page with the sha256 the manifest records, counts per venue the pages that changed, disappeared or appeared, and never replaces a recorded copy already on disk. A page that fails to download fails the fetch unless `--allow-missing` is given.

There is no headless-browser scraper: it would serve one source (BingX) and make the corpus unpinnable and unverifiable.

Per-source include/exclude globs, rationale and caveats are in [`data/sources.yaml`](../../data/sources.yaml), the machine-readable form of the tables above. `fetched_at` is 2026-09-20 for all 1,665 documents.

## Preprocessing

Each stage is explained in [design](../explanation/design.md); the raw pages are kept in `data/raw/`.

- Normalization strips per-stack markup: Docusaurus MDX imports, `<Tabs>` and admonitions (Bybit), JSX components (Uniswap, dYdX), GitBook `{% hint %}` blocks and HTML entities (Hyperliquid, PancakeSwap), docsify directives (CCXT). Fenced code blocks and inline code spans are kept, apart from blank-line runs collapsed in 19 blocks ([design](../explanation/design.md#normalization)).
- Chunking breaks at headings once a chunk reaches 40 tokens and keeps fenced code atomic up to 1,200 tokens. It carries the heading breadcrumb into the text and holds the rendered chunk to 450 tokens of the embedding model's tokenizer. A chunk closed because the next block would pass that limit carries its trailing blocks after the last code block, whole and up to 60 tokens, into the next chunk ([design](../explanation/design.md#chunking), [ADR 0002](../adr/0002-structural-chunking.md)).
- Structural metadata is assigned by path and heading rules ([design](../explanation/design.md#structural-metadata)).
- Exact deduplication stores byte-identical text within a venue once, with every occurrence in `sources` and a primary occurrence chosen deterministically (non-testnet first, then `doc_path`, then position) ([design](../explanation/design.md#exact-deduplication), [ADR 0003](../adr/0003-exact-deduplication.md)).

`chunk_id` is `sha256(venue + "\x1f" + text)[:16]`, stable across re-ingestion of an unchanged corpus.

## Counting a term

A count of a term is the number of chunks in `data/processed/chunks.jsonl` that contain it, case-insensitive, as `crossvenue count-term TERM [--venue X]` prints it. Every count of one term quoted in a document or a `notes` field is this command's output, so two people counting the same thing agree. Counting occurrences instead of chunks, or matching case, gives other numbers for the same corpus. A count that could be read another way states this convention beside it, and names its suspicious hits: `GTX` = 2 chunks, both a substring of the same `clientOrderId` value in Binance examples.

A concept is searched under every spelling the vendor uses (`tickSize`, `szDecimals`, "significant figures") before it is called present or absent, and the counts are reported. One spelling with no hit does not show that the corpus lacks the concept.

## Uses

Intended. Evaluating retrieval methods on cross-venue confusion, version confusion and unanswerable questions, with the gold set described in [`evals/datasheet.md`](../../evals/datasheet.md).

Not intended. As a source of truth about any venue's current API. The documentation is a snapshot; limits, fees and endpoints change, and the corpus does not track them. Binance derivatives are absent, so the corpus cannot stand for "all of Binance".

## Distribution

Nothing from the sources is redistributed. `data/raw/` and `data/processed/` are gitignored; of `data/`, only `data/manifest.csv` (URLs, commit SHAs, content hashes) and `data/sources.yaml` are committed. `make fetch ingest` rebuilds this corpus from the six git sources at their recorded commits, and `ingest` refuses any raw file whose sha256 differs from the manifest. The two GitBook sources (Hyperliquid, PancakeSwap: 418 documents) cannot be fetched at a pinned version. `make fetch` stops when a page's content differs from the recorded hash, with the count of such pages per venue, and the corpus is rebuilt only from copies that match.

Licenses are recorded as found. Five of the eight sources record no license; that is recorded verbatim as `no license found`, with no license inferred. Binance ships `PROD-TERMS-OF-USE.md` instead of a `LICENSE` file; BingX's license is in `package.json`.

## Maintenance

`data/manifest.csv` records per document `venue`, `venue_type`, `source_url`, `local_path`, `fetched_at`, `commit_sha` and `sha256`. Git sources record the commit they were fetched at; GitBook sources have no commit, so the content hash is the anchor.

`make fetch` never writes the manifest; it exits with an error that counts, per venue, the files that differ from it. Drift in the sources is measured by following the branches ([reproduce](../how-to/reproduce.md#measure-drift-in-the-sources)). That rewrites the manifest and keeps `fetched_at` for every file whose hash and commit are unchanged. A git source whose branch moved therefore changes every row of its venue in a diff of the manifest. Each of those rows records the checked-out commit in `commit_sha` and `source_url`, and the time of that fetch in `fetched_at`. A GitBook source changes only the rows of pages whose content changed.

The models are pinned to Hugging Face commits in `configs/default.yaml` (`embedding.revision`, `reranker.revision`) and recorded under `model_revisions` in `data/index/index_meta.json`: `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` (weights and the tokenizer that measures the chunk budget) and `BAAI/bge-reranker-base` at `2cfc18c9415c912f9d8155881c133215df768a70`. A tokenizer that cannot be loaded stops `ingest`; there is no fallback counter.

A change to the manifest, the chunking configuration or the embedding model changes the `index_version` ([ADR 0004](../adr/0004-index-version.md)). The model revisions and the code of normalization and chunking are not part of it ([ADR 0004](../adr/0004-index-version.md#consequences)); `eval` refuses a gold id that the current chunks do not contain. Gold labels made against another `index_version` are refused until they are carried across with `crossvenue id-map` / `remap-ids` or re-judged. `crossvenue_rag.dataset.manifest_fingerprint()` hashes the whole manifest, and every versioned eval dataset stores it.
