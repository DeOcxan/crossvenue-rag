"""What more than one document gives: numbers no run measures, and the reasons they carry."""

from __future__ import annotations

# The level of an interval: a convention that no run measures.
INTERVAL_LEVEL = "interval level"
# A provider or uptime figure read from OpenRouter's model listing, which no saved run holds.
OPENROUTER = "OpenRouter's model listing on 2026-09-25, an external source"
REGISTERED = "a target or parameter the protocol registers"
POWER_MODEL = "an input of the power model, chosen in section 7"
TRIAL = "the five-question answer trial run, which is not committed"
DOCS_QUOTE = "a value quoted from the venue documentation in the corpus"
LIST_NUMBERING = "list numbering"
LIST_FOLLOWS = "the list that follows"
LIST_IN_SENTENCE = "the list that follows in the sentence"
STATUS_CODE = "the status code `service.app` returns"
GITHUB_STATUS = "the status code the GitHub API returned"
CITED = "a citation"
ARXIV_CITATION = "a citation: authors, year, arXiv id"
TEST_NAME = "the test's name"
ROOT_DEFINITION = "the definition of the root"
SERVE_ADDRESS = "the address `make serve` binds"
# The fenced code blocks normalization changes, and how the count was made.
COLLAPSED_FENCES = "19"
COLLAPSED_FENCES_REASON = (
    "counted by comparing each raw fenced block in data/raw with its normalized form; no "
    "committed function recomputes it"
)
# The reading window of the embedding model's tokenizer, and its special tokens.
BGE_TOKENIZER = (
    "the model_max_length of the bge-base-en-v1.5 tokenizer and the two special tokens it "
    "adds, read from the Hugging Face cache; no saved file holds them"
)
# The bi-encoder's reading window and the special tokens its tokenizer adds (`BGE_TOKENIZER`).
BGE_WINDOW = 512
BGE_SPECIAL_TOKENS = 2
