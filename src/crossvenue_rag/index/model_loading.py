"""Loading a sentence-transformers model: the torch device and the hint for a failed load.

`index.embeddings` and `retrieval.rerank` take the device and the hint from it. torch is
imported where a device is resolved, so importing this module loads no heavy library.
"""

from __future__ import annotations

from ..errors import UserError, install_hint_or

# What a model that cannot be loaded is told to do.
MODEL_HINT = (
    "check the model name and revision in the config; the model is read from the Hugging "
    "Face cache, so download it once with network access"
)


def load_hint(exc: Exception) -> str:
    """The hint for a model load that raised `exc`: `install_hint_or` with `MODEL_HINT`."""
    return install_hint_or(exc, MODEL_HINT)


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
