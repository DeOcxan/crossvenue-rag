"""The scripts import each other by module name, as `python scripts/figures.py` runs them."""

from __future__ import annotations

import sys

from tests.published import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
