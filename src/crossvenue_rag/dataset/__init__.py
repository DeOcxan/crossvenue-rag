"""Question files, versioned datasets, validation, selection and agreement.

`manifest_fingerprint` is re-exported here because `v0.1.2-candidates.meta.json`
and several CHANGELOG entries name it as
`crossvenue_rag.dataset.manifest_fingerprint`.
"""

from .records import manifest_fingerprint

__all__ = ["manifest_fingerprint"]
