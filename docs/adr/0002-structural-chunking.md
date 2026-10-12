# 0002. Heading-aware chunking with a budget in the embedder's tokens

- Date: 2026-09-20
- Status: Accepted

## Context

API documentation is a tree of headings whose leaves are parameter tables and code samples. A fixed-window splitter cuts a parameter table away from the endpoint it belongs to and cuts signatures in half. Either makes a chunk that cannot be retrieved for its real content or cited as an answer.

A token budget is only meaningful in the tokens of the model that encodes the chunk. A word count undercounts the tokens of the model's WordPiece tokenizer on API text: on the deduplicated corpus, the 11,934 chunks of `data/processed/chunks.jsonl` hold 2,484,877 tokens (`token_count`, counted with that tokenizer) and 1,005,525 whitespace-separated words, 2.47 tokens per word. A "450-word" chunk therefore overruns a 512-token encoder window.

A first implementation that counted the budget block by block had three defects, each found by tests and each silently breaking the budget. The breadcrumb was in the text but not in the budget. Tokenizers are not additive, so summing separately counted blocks undershot the joined text by the separators. Carried-over overlap could leave no room for the next block, which was then appended anyway. With them, 124 prose chunks of the 12,783 the corpus had before deduplication (ADR 0003) exceeded the budget. Measuring on the rendered text and dropping overlap before exceeding the budget reduced that to 18.

## Decision

- Chunks break at headings, once the running chunk has at least `min_tokens` (40) of body, so `## Parameters` merges into its body instead of becoming a chunk.
- A fenced code block is atomic: it may exceed `max_tokens` (450) up to `max_code_block_tokens` (1,200) rather than be split; above that it splits on line boundaries.
- The heading breadcrumb is part of the chunk text, so a continuation chunk still names its endpoint.
- Overlap (60 tokens) is taken in whole trailing blocks, never from a code block.
- Tokens are counted with the embedding model's own tokenizer (`BAAI/bge-base-en-v1.5`), and the budget is measured on the rendered chunk text, breadcrumb and separators included.

## Consequences

- Some chunks exceed `max_tokens`, code blocks on purpose; their counts on the deduplicated corpus are in [design](../explanation/design.md#chunking).
- Chunk boundaries depend on the tokenizer, so changing the embedding model re-chunks the corpus and changes `index_version` (ADR 0004).
- Tests inject a word-count token counter and never load the tokenizer.
