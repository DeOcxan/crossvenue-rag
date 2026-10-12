"""Token counting with the embedding model's own tokenizer.

Chunking measures its budget in the tokens of the model that will encode the
chunks, so "450 tokens" means the same thing at chunk time and at encode time.
Only the tokenizer is needed, so this module is in `corpus` beside the chunker
rather than in `index.embeddings`, which loads the model; transformers is
imported when a tokenizer is first loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from .. import formatting
from ..errors import UserError

# The tokenizers kept loaded, one per (model, revision, max_tokens): a process counts with
# one model, and a test or a config override may add a few more.
CACHED_TOKENIZERS = 4


@lru_cache(maxsize=CACHED_TOKENIZERS)
def token_counter_for(
    model_name: str, revision: str | None, max_tokens: int
) -> Callable[[str], int]:
    """Token counter backed by the embedding model's own tokenizer, at `revision`.

    Cached per model, revision and `max_tokens`, because loading a tokenizer takes about
    a second. A tokenizer that cannot be loaded is an error: a
    different counter would chunk the corpus differently under the same
    `index_version`.

    Raises:
        UserError: The tokenizer cannot be loaded for a reason the user can fix,
            or `max_tokens` exceeds the tokens the encoder reads. The causes are a
            model or revision in neither the cache nor the Hub (`OSError`, which
            every Hugging Face Hub error of a missing file subclasses), a malformed
            name or revision (`ValueError`) and a missing transformers
            (`ImportError`). Any other exception keeps its traceback.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    except (OSError, ValueError, ImportError) as exc:
        raise UserError(
            f"cannot load the tokenizer of {model_name}{formatting.revision_text(revision)}: {exc}",
            hint="the chunk budget is counted in this tokenizer's tokens; download the model "
            "once with network access (it is then read from the Hugging Face cache)",
        ) from exc

    # A tokenizer that states no window is not checked. The count below leaves out the
    # special tokens the encoder adds, so the budget is the window less those.
    window = getattr(tokenizer, "model_max_length", None)
    if isinstance(window, int):
        text_window = window - tokenizer.num_special_tokens_to_add()
        if max_tokens > text_window:
            raise UserError(
                f"chunking.max_tokens ({formatting.own_form(max_tokens)}) exceeds the "
                f"{formatting.count(text_window)} tokens of text {model_name} encodes "
                f"({formatting.count(window)} with its special tokens); the encoder would cut "
                "every longer chunk",
                hint=f"set chunking.max_tokens to at most {formatting.own_form(text_window)}",
            )

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
