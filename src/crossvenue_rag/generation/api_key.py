"""The OpenRouter API key, read from the environment or the repository's dotenv file."""

from __future__ import annotations

import os
import re
from pathlib import Path

from ..errors import UserError

KEY_NAME = "OPENROUTER_API_KEY"


# src/crossvenue_rag/generation/api_key.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
# The dotenv file the key is read from when the environment lacks it; tests point it elsewhere.
DOTENV_FILE = REPO_ROOT / ".env"


def _dotenv_value(raw: str) -> str:
    """A dotenv value: quoted as written, or unquoted up to a ` #` comment."""
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        return raw[1:end] if end > 0 else raw[1:]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def load_api_key() -> str:
    """OPENROUTER_API_KEY from the environment, else from `DOTENV_FILE`. Never printed.

    `DOTENV_FILE` is the dotenv file at the repository root, whatever the working
    directory. A value of only whitespace counts as not set. The key goes only into
    the request header; it never enters a cache key, a record, a log line or an
    error message.

    Raises:
        UserError: The key is set neither in the environment nor in `DOTENV_FILE`, or that file is
            not UTF-8 text.
    """
    value = os.environ.get(KEY_NAME, "").strip()
    path = DOTENV_FILE
    if not value and path.is_file():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            # The file's content is never echoed: it may hold the key.
            raise UserError(
                f"{path} is not UTF-8 text", hint=f"save {path} with UTF-8 encoding"
            ) from exc
        for line in text.splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Z_]+)\s*=\s*(.*)$", line)
            if match and match.group(1) == KEY_NAME:
                value = _dotenv_value(match.group(2)).strip()
    if not value:
        raise UserError(
            f"{KEY_NAME} is not set", hint="copy .env.example to .env and fill in the key"
        )
    return value
