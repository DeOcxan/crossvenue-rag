"""The calibration sample: stratified answers of an answer run, saved as the raters' file."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from .. import formatting
from ..config import Config
from ..dataset.sampling import stratified_quotas
from ..errors import UserError, require_file
from ..index_version import require_current_index
from .records import ANSWERS_FILE, AnswerRecord, read_answers
from .runs import require_run_index

# The header line that holds the run directory and nothing else.
_RUN_LINE = "# Run: "
# The one-line header of the files written before `_RUN_LINE`, which every committed
# sample has: `... 40 answers from results/<run>, stratified by ...`.
_FROM_RUN = re.compile(r" answers? from ")
_AFTER_RUN = ", stratified"


def stratum(record: AnswerRecord) -> str:
    """The sampling stratum of `record`: its configuration and category, `config|category`."""
    return f"{record.config}|{record.category}"


def calibration_sample(
    records: Sequence[AnswerRecord],
    n: int,
    seed: int,
    exclude_questions: Sequence[str],
) -> list[AnswerRecord]:
    """`n` distinct generated answers, stratified by configuration and category.

    None comes from the questions in `exclude_questions`.

    Strata get proportional quotas, at least one each
    (`dataset.sampling.stratified_quotas`), drawn with `seed`. With more strata
    than `n`, only the `n` largest strata get one answer each, a tie in size going
    to the stratum name that sorts first.

    Two configurations with the same generator give the same answer wherever
    their retrievers return the same chunks (one cached call). Only the first
    such record, in matrix order, is eligible: grading one text twice adds
    nothing and would count one agreement twice.

    Raises:
        ValueError: `n` is below 1 (`generation.calibration_size` is at least 1).
        UserError: `exclude_questions` names a question no record answers, as a
            mistyped id in `generation.calibration_exclude` would, or `n` exceeds the
            eligible answers.
    """
    if n < 1:
        raise ValueError(f"a calibration sample needs n >= 1, got {formatting.count(n)}")
    excluded = set(exclude_questions)
    unknown = sorted(excluded - {r.question_id for r in records})
    if unknown:
        raise UserError(
            f"generation.calibration_exclude names questions the run did not answer: "
            f"{', '.join(unknown)}",
            hint="list ids of questions the run answered, or remove them",
        )
    seen_answers: set[str] = set()
    eligible = []
    for r in records:
        if r.generation is None or r.question_id in excluded:
            continue
        answer_key = r.generation.call.cache_key
        if answer_key in seen_answers:
            continue
        seen_answers.add(answer_key)
        eligible.append(r)
    by_stratum: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in eligible:
        by_stratum[stratum(record)].append(record)
    if len(by_stratum) > n:
        largest = sorted(by_stratum, key=lambda s: (-len(by_stratum[s]), s))[:n]
        quotas = {s: 1 for s in largest}
    else:
        quotas = stratified_quotas({s: len(rs) for s, rs in by_stratum.items()}, n, "answer")
    rng = random.Random(seed)
    chosen: set[str] = set()
    for name in sorted(quotas):
        ids = sorted(r.answer_id for r in by_stratum[name])
        chosen.update(rng.sample(ids, quotas[name]))
    return [r for r in eligible if r.answer_id in chosen]


def write_calibration_sample(
    path: str | Path,
    run_dir: str | Path,
    sample: Sequence[AnswerRecord],
    seed: int,
    excluded: Sequence[str],
) -> None:
    """Writes the sample list to `path`, its provenance in comment lines.

    The list is written before any rating, so the answers graded are fixed in
    advance. The run directory has a line of its own, so a path holding a comma
    reads back whole.
    """
    lines = [
        f"# Judge calibration sample: {formatting.plural(len(sample), 'answer')}, "
        f"stratified by configuration and category, seed {formatting.own_form(seed)}.",
        f"{_RUN_LINE}{run_dir}",
        f"# Excluded questions (judge prompt revised on them): {' '.join(excluded) or 'none'}.",
        "# Written by `crossvenue rate-sample`; read by `crossvenue rate`.",
    ]
    lines += [f"{r.answer_id}  {stratum(r)}" for r in sample]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_sample(path: str | Path) -> tuple[str, list[str]]:
    """(run directory, answer ids) from a calibration sample file.

    The run is read from one of two header formats:

    - `# Run: <run directory>`, a line of its own, as `write_calibration_sample` writes it;
    - `# Judge calibration sample: 40 answers from <run directory>, stratified ...`, the
      one-line form of every committed sample file, read up to its last `, stratified`.

    Every other line starting with `#` is a comment; each remaining line starts with an
    answer id.

    Raises:
        UserError: Nothing is at `path`, or a directory is. The file is not UTF-8 text, or it
            names no run or no answer, so it was not written by `crossvenue rate-sample`.
    """
    hint = "pass --sample-file a sample written by `crossvenue rate-sample`"
    try:
        text = require_file(path, hint=hint).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UserError(
            f"{path} is not UTF-8 text: {exc.reason}",
            hint="write the sample again with `crossvenue rate-sample`, which saves UTF-8",
        ) from exc
    run_line, one_line_header, ids = "", "", []
    for line in text.splitlines():
        if line.startswith(_RUN_LINE):
            run_line = line.removeprefix(_RUN_LINE).strip()
        elif line.startswith("#"):
            header = _FROM_RUN.split(line, maxsplit=1)
            if len(header) > 1:
                one_line_header = header[1].rsplit(_AFTER_RUN, 1)[0].strip()
        elif line.strip():
            ids.append(line.split()[0])
    run_dir = run_line or one_line_header
    if not run_dir:
        raise UserError(f"{path} names no run", hint=hint)
    if not ids:
        raise UserError(f"{path} names no answer", hint=hint)
    return run_dir, ids


def draw_calibration_sample(cfg: Config, run: Path, out: Path) -> dict[str, int]:
    """Draws the calibration sample of `run` and writes it to `out`: its answers per stratum.

    Raises:
        UserError: `run` is not an answer or judged run, or records no index version or
            another one than the current index. Or `generation.calibration_exclude` names a
            question the run did not answer, or `generation.calibration_size` exceeds the
            run's eligible answers.
    """
    require_run_index(run, require_current_index(cfg))
    gen = cfg.generation
    sample = calibration_sample(
        read_answers(run / ANSWERS_FILE),
        gen.calibration_size,
        gen.calibration_seed,
        gen.calibration_exclude,
    )
    write_calibration_sample(out, run, sample, gen.calibration_seed, gen.calibration_exclude)
    return dict(Counter(stratum(r) for r in sample))
