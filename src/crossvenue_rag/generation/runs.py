"""The gates a saved answer run passes before use: its index, pinned prompts, calibrated metrics.

The index and prompt gates refuse a run; the calibration gate marks the metrics whose judge
dimension failed calibration, and the reports state that those support no conclusion.
"""

from __future__ import annotations

import hashlib
import re
from functools import cache
from pathlib import Path

from .. import formatting
from ..config_generation import GenerationConfig
from ..errors import UserError, require_file
from ..run_files import RUNS_REGISTRY, read_index, read_registry
from .calibration_file import CALIBRATION_FILE, read_calibration
from .records import ANSWERS_FILE

# Judge dimension (calibration) -> the answer metric computed from it.
DIMENSION_METRIC = {
    "supported": "faithfulness",
    "venue_correct": "venue_attribution_error",
    "behaviour_correct": "behaviour_correct",
    "coverage": "coverage",
}
_VERSION_LINE = re.compile(r"generator prompt v(\d+)\b")


def answer_run_index_version(run_dir: Path, registry: Path) -> str | None:
    """The run's own `index.json`, else its entry in the registry of saved runs.

    The registry (`results/runs.json`) lists the runs without an `index.json`,
    with the version derived from the retrieval runs.
    """
    own = read_index(Path(run_dir))
    if own is not None:
        return own.index_version
    if registry.exists():
        entry = read_registry(registry).runs.get(Path(run_dir).name)
        return None if entry is None else entry.index_version
    return None


def require_answer_run(run_dir: Path) -> Path:
    """`run_dir`, after checking it is an answer or judged run: a directory with answers.jsonl.

    Raises:
        UserError: `run_dir` is missing, not a directory, or holds no answers.jsonl (a
            retrieval, pool, bench or power run).
    """
    if not (Path(run_dir) / ANSWERS_FILE).is_file():
        raise UserError(
            f"{run_dir} is not an answer run: it has no {ANSWERS_FILE}",
            hint="pass an answer run, results/<ts>_answers, or a judged run, results/<ts>_judged",
        )
    return Path(run_dir)


def require_run_index(run_dir: Path, current: str) -> str:
    """The index version `run_dir` records, which must be `current`.

    Answers carry chunk ids, and judging or rating them against another index
    resolves the ids to other text.

    Raises:
        UserError: `run_dir` is not an answer or judged run (`require_answer_run`), or
            records no index version, or another one than `current`.
    """
    require_answer_run(run_dir)
    recorded = answer_run_index_version(run_dir, Path(RUNS_REGISTRY))
    if recorded is None:
        raise UserError(
            f"{run_dir} records no index_version",
            hint=f"its chunk ids cannot be checked against the current index {current}; "
            f"add the run to {RUNS_REGISTRY} with the version it was made against",
        )
    if recorded != current:
        raise UserError(
            f"{run_dir} was made against index {recorded}, the current index is {current}",
            hint="its chunk ids may point at different text in this index; rebuild that index "
            "(`make fetch ingest index` from the manifest it used) or answer again",
        )
    return recorded


def load_prompt(path: str | Path, expected_sha256: str | None) -> tuple[str, str]:
    """The prompt text and the sha256 of the file it came from.

    With `expected_sha256`, a file whose hash differs is refused: the run's
    config would otherwise name a prompt that is not the one that ran. A pinned
    prompt is read once per process (prompt files are never edited, docs/style.md
    rule 44), so a run does not reread it for every answer it generates or judges.
    """
    if expected_sha256 is not None:
        return _pinned_prompt(str(Path(path).resolve()), expected_sha256)
    return _read_prompt(path, None)


@cache
def _pinned_prompt(path: str, expected_sha256: str) -> tuple[str, str]:
    return _read_prompt(path, expected_sha256)


def _read_prompt(path: str | Path, expected_sha256: str | None) -> tuple[str, str]:
    hint = "set generation.generator_prompt and judge_prompt to prompt files under prompts/"
    raw = require_file(path, hint=hint).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise UserError(
            f"{path} has sha256 {digest}; the config pins {expected_sha256}",
            hint="a changed prompt is another version: save it under another name and pin "
            "its hash, or restore the pinned file",
        )
    try:
        return raw.decode("utf-8"), digest
    except UnicodeDecodeError as exc:
        raise UserError(f"{path} is not UTF-8 text: {exc.reason}", hint=hint) from exc


def generator_prompt(cfg: GenerationConfig) -> tuple[str, str]:
    """The generator prompt, checked against its pinned hash and the schema version.

    Raises:
        UserError: The prompt's first line names no version, or the version is not
            `answer_schema_version`.
    """
    text, digest = load_prompt(cfg.generator_prompt, cfg.generator_prompt_sha256)
    match = _VERSION_LINE.search(text.split("\n", 1)[0])
    if match is None:
        raise UserError(
            f"{cfg.generator_prompt} has no version line",
            hint="its first line names the version, as `<!-- generator prompt vN (...)`",
        )
    if int(match.group(1)) != cfg.answer_schema_version:
        schema = formatting.own_form(cfg.answer_schema_version)
        raise UserError(
            f"{cfg.generator_prompt} is prompt v{match.group(1)}, generation."
            f"answer_schema_version is {schema}",
            hint=f"set generation.answer_schema_version to {match.group(1)}, or set "
            f"generation.generator_prompt to a prompt v{schema} file and "
            "generator_prompt_sha256 to its sha256",
        )
    return text, digest


def judge_prompt(cfg: GenerationConfig) -> tuple[str, str]:
    """The judge prompt and its sha256, checked against its pinned hash."""
    return load_prompt(cfg.judge_prompt, cfg.judge_prompt_sha256)


def uncalibrated_metrics(cfg: GenerationConfig, judge: str | None) -> list[str]:
    """Metrics whose judge dimension failed calibration for `judge`.

    A dimension fails when it misses its pre-registered threshold (methodology
    section 9). Read from `generation.calibration_run`/calibration.json. Every
    metric counts as uncalibrated when no calibration names the judge.

    Raises:
        UserError: `generation.calibration_run` is set and holds no calibration.json.
    """
    if judge is None:
        return []
    run = cfg.calibration_run
    if run is None:
        return list(DIMENSION_METRIC.values())
    if not (run / CALIBRATION_FILE).is_file():
        raise UserError(
            f"generation.calibration_run {run} has no {CALIBRATION_FILE}",
            hint=f"run `crossvenue select-judge --run {run}`, or set "
            "generation.calibration_run to null",
        )
    candidate = read_calibration(run).candidates.get(judge)
    if candidate is None:
        return list(DIMENSION_METRIC.values())
    return [DIMENSION_METRIC[d] for d in candidate.fails if d in DIMENSION_METRIC]
