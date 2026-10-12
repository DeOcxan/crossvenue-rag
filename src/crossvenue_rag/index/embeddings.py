"""Dense encoder wrapper holding three facts that must not drift apart.

The facts are the model name, the asymmetric query and document prefixes bge-*
was trained with, and whether vectors are L2-normalized (they must be, because
the store uses cosine distance). The model is loaded on first use, and torch and
sentence_transformers with it, so importing this module loads neither.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..config_pipeline import EmbeddingConfig
from ..errors import UserError
from ..formatting import revision_text

if TYPE_CHECKING:  # pragma: no cover - imported for the type checker only, never at runtime
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


# What a model that cannot be loaded is told to do.
MODEL_HINT = (
    "check the model name and revision in the config; the model is read from the Hugging "
    "Face cache, so download it once with network access"
)


def resolve_device(requested: str | None) -> str:
    """The torch device to use: explicit request, else mps > cuda > cpu.

    Raises:
        UserError: The requested device's backend is not available on this machine.
    """
    import torch

    if requested:
        kind = requested.split(":")[0]
        available = {
            "cpu": True,
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
        }
        if not available.get(kind, False):
            raise UserError(
                f"device {requested} is not available on this machine",
                hint="set embedding.device to cpu, or to null to pick the fastest available",
            )
        return requested

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    """The configured dense encoder, loaded on first use, with its prefixes applied."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        """An encoder for `cfg`; the model is not loaded until first used."""
        self.cfg = cfg
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """The sentence-transformers model, loaded at the configured revision on first access.

        Raises:
            UserError: The model or revision is not in the cache or on the Hub (`OSError`), is
                malformed (`ValueError`), or sentence_transformers is missing.
        """
        if self._model is None:
            device = resolve_device(self.cfg.device)
            log.info("loading %s on %s", self.cfg.model_name, device)
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.cfg.model_name, device=device, revision=self.cfg.revision
                )
            except (OSError, ValueError, ImportError) as exc:
                raise UserError(
                    f"cannot load embedding.model_name {self.cfg.model_name}"
                    f"{revision_text(self.cfg.revision)}: {exc}",
                    hint=MODEL_HINT,
                ) from exc
        return self._model

    def encode_documents(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        """One float32 vector per text, row for row, with the document prefix applied."""
        prefixed = [self.cfg.document_prefix + t for t in texts]
        return self._encode(prefixed, show_progress=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """The float32 vector of `query`, with the query prefix applied."""
        vector: np.ndarray = self._encode([self.cfg.query_prefix + query], show_progress=False)[0]
        return vector

    def _encode(self, texts: list[str], *, show_progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def model_fingerprint(self) -> str:
        """The model name and dimension, `<model_name>@dim<dimension>`, stored with every vector.

        The other settings that change a stored vector, `document_prefix` and `normalize`,
        are not in it; they are components of the `index_version` (ADR 0004).
        """
        return f"{self.cfg.model_name}@dim{self.cfg.dimension}"
