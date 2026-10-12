"""Power of the paired recall comparison as it is run, by simulation.

The comparisons use a two-sided sign-flip randomization test on per-question recall@10
(fractional when a question has several gold chunks), Holm-adjusted within a family.
Methodology section 7's power table models a binary hit and an unadjusted McNemar test.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from .. import formatting
from ..config import Config
from ..errors import UserError
from ..models import Question
from ..run_files import (
    INDEX_FILE,
    PAIRED_FILE,
    RUN_FILE_HINT,
    RunIndex,
    create_run_dir,
    file_sha256,
    read_index,
    read_json_model,
    run_path,
    write_index,
)
from ..stats import mean
from .comparisons import LABELLED_CATEGORY_SCOPES, read_paired
from .metrics import recall_at_k
from .paired import sign_flip_p
from .results import PER_QUERY_FILE, read_per_query

# How a comparison's power is read: at Holm's first step (the lowest), or unadjusted (the highest).
Adjustment = Literal["holm_first_step", "unadjusted"]


def _checked(adjustment: str) -> Adjustment:
    if adjustment == "holm_first_step":
        return "holm_first_step"
    if adjustment == "unadjusted":
        return "unadjusted"
    raise ValueError(f"unknown adjustment {adjustment!r}; expected holm_first_step or unadjusted")


class PowerPoint(BaseModel):
    """The simulated power at one true difference, with and without the Holm first step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: float
    reachable: bool
    power_holm_first_step: float | None  # p <= alpha / family
    power_unadjusted: float | None  # p <= alpha

    def power_under(self, adjustment: str) -> float | None:
        """The power under `adjustment`.

        Raises:
            ValueError: `adjustment` is not an `Adjustment`.
        """
        if _checked(adjustment) == "holm_first_step":
            return self.power_holm_first_step
        return self.power_unadjusted


@dataclass(frozen=True)
class PowerDesign:
    """The settings of one power simulation.

    They are the observed discordance `psi` and nonzero difference `magnitudes`,
    the deltas to try, `n` questions, the level `alpha` and the Holm `family`
    size, `studies` per delta and `permutations` per test.

    Raises:
        ValueError: `magnitudes` is empty, `psi` is outside (0, 1], `alpha` outside
            (0, 1), or `n`, `family`, `studies` or `permutations` is below 1: a study
            of zero simulations has no power, and one of zero sign flips rejects nothing.
    """

    magnitudes: Sequence[float]
    psi: float
    deltas: Sequence[float]
    n: int
    alpha: float
    family: int
    studies: int
    permutations: int
    seed: int

    @property
    def ceiling(self) -> float:
        """Psi E[m]: the largest expected difference the model reaches."""
        return self.psi * sum(self.magnitudes) / len(self.magnitudes)

    def __post_init__(self) -> None:
        """Refuses a design outside the simulation's domain.

        Raises:
            ValueError: No magnitude, `psi` outside (0, 1], `alpha` outside (0, 1), or a
                count below 1.
        """
        if not self.magnitudes:
            raise ValueError("a power design needs at least one difference magnitude")
        if not 0 < self.psi <= 1:
            raise ValueError(
                f"psi is a probability above 0, got {formatting.design_parameter(self.psi)}"
            )
        if not 0 < self.alpha < 1:
            raise ValueError(
                "alpha must lie strictly between 0 and 1, got "
                f"{formatting.design_parameter(self.alpha)}"
            )
        for name in ("n", "family", "studies", "permutations"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {formatting.count(getattr(self, name))}"
                )


def simulate_power(design: PowerDesign) -> list[PowerPoint]:
    """Power of the Holm-adjusted randomization test at each delta.

    The model is the methodology's, with fractional differences and observed
    parameters. A question is discordant (the two methods' recall differs) with
    probability psi. A discordant question's difference has a magnitude drawn
    from the observed nonzero magnitudes, and favors the better method with
    probability 1/2 + delta / (2 psi E[m]), so the expected difference is delta.
    A delta above psi E[m] cannot be reached in this model and is returned as
    unreachable.

    Under Holm the level a comparison is tested at depends on the other p-values of its
    family. It is alpha / family when it has the smallest p and the others are
    null (the lowest power, `power_holm_first_step`). It is alpha when every other
    comparison is rejected first (the highest, `power_unadjusted`).
    """
    rng = np.random.default_rng(design.seed)
    m = np.asarray(design.magnitudes, dtype=float)
    ceiling = design.ceiling
    out = []
    for delta in design.deltas:
        if delta > ceiling:
            out.append(
                PowerPoint(
                    delta=float(delta),
                    reachable=False,
                    power_holm_first_step=None,
                    power_unadjusted=None,
                )
            )
            continue
        favor = 0.5 + delta / (2 * ceiling)
        shape = (design.studies, design.n)
        discordant = rng.random(shape) < design.psi
        size = m[rng.integers(0, m.size, size=shape)]
        sign = np.where(rng.random(shape) < favor, 1.0, -1.0)
        p = sign_flip_p(discordant * size * sign, design.permutations, rng)
        out.append(
            PowerPoint(
                delta=float(delta),
                reachable=True,
                power_holm_first_step=float((p <= design.alpha / design.family).mean()),
                power_unadjusted=float((p <= design.alpha).mean()),
            )
        )
    return out


def smallest_simulated_delta(
    points: Sequence[PowerPoint], target: float, *, holm: bool
) -> float | None:
    """The smallest simulated delta whose power reaches `target`."""
    for point in sorted(points, key=lambda p: p.delta):
        power = point.power_under("holm_first_step" if holm else "unadjusted")
        if power is not None and power >= target:
            return point.delta
    return None


@dataclass(frozen=True)
class PowerRequest:
    """The `power` options: the retrieval run, the category and depth, the simulation size."""

    run: Path
    category: str
    k: int
    studies: int
    permutations: int


# The significance level of every test the study reports, and of every comparison the
# documents call significant (methodology sections 7 and 9).
SIGNIFICANCE_LEVEL = 0.05
# The power the simulated deltas of power.json are named after (`delta_80_...`).
# The power a detectable difference and a sample size are planned for, in percent.
TARGET_POWER_PERCENT = 80
# The same power as a share.
TARGET_POWER = TARGET_POWER_PERCENT / formatting.PER_CENT
# The file a power run writes.
POWER_FILE = "power.json"
# The default simulation size of `crossvenue power`: studies per delta, and sign flips per
# randomization test.
POWER_STUDIES = 2000
POWER_PERMUTATIONS = 2000
# The deltas simulated: 0.05 to 0.25 in steps of 0.01.
DELTAS = tuple(round(0.05 + 0.01 * i, 2) for i in range(21))


# What a run that is not a retrieval run is told to do.
RETRIEVAL_RUN_HINT = "pass --run a retrieval run, results/<ts>, as `crossvenue eval` writes it"


def run_index(run: Path) -> RunIndex:
    """A retrieval run's index.json; every retrieval run from 20260924T234623Z on has one.

    Raises:
        UserError: `run` has no index.json.
    """
    index = read_index(run)
    if index is None:
        raise UserError(f"{run} has no {INDEX_FILE}", hint=RETRIEVAL_RUN_HINT)
    return index


def run_questions_path(run: Path) -> Path:
    """The question file a retrieval run read, from its index.json.

    Raises:
        UserError: `run` has no index.json, or is not a retrieval run (no paired.csv or
            per_query.csv: a pool, bench, answer or power run), the run records no question
            file, or the file is gone. Or its sha256 differs from the one the run recorded, so
            its gold is not the gold the run scored.
    """
    index = run_index(run)
    missing = [f for f in (PAIRED_FILE, PER_QUERY_FILE) if not (run / f).is_file()]
    if missing:
        raise UserError(f"{run} has no {' or '.join(missing)}", hint=RETRIEVAL_RUN_HINT)
    if index.questions is None:
        raise UserError(f"{run} records no question file", hint="pass a retrieval run")
    path = Path(index.questions)
    if not path.exists():
        raise UserError(
            f"{path}, the question file {run} scored, is missing",
            hint="restore it, or run `crossvenue eval` again and pass that run",
        )
    if index.questions_sha256 is not None and file_sha256(path) != index.questions_sha256:
        raise UserError(
            f"{path} differs from the question file {run} scored (sha256 "
            f"{formatting.short_hash(index.questions_sha256)} in its {INDEX_FILE})",
            hint="restore that version of the file, or run `crossvenue eval` again and pass "
            "that run",
        )
    return path


class PowerStudy(BaseModel):
    """power.json: the recall family of a retrieval run, its discordance and simulated power.

    Every committed power.json has these fields in this order, indented by two with
    no final newline: `results/20260926T015648Z_power` is the one committed run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: str
    metric: str
    category: str
    questions: int
    family: list[tuple[str, str]]
    fractional_values: int
    values: int
    discordant_pairs: int
    question_pairs: int
    psi: float
    mean_magnitude: float
    ceiling: float
    alpha: float
    studies: int
    permutations: int
    seed: int
    points: list[PowerPoint]
    delta_80_holm_first_step: float | None
    delta_80_unadjusted: float | None

    def delta_at_target(self, adjustment: str) -> float | None:
        """`delta_80_<adjustment>`.

        Raises:
            ValueError: `adjustment` is not an `Adjustment`.
        """
        if _checked(adjustment) == "holm_first_step":
            return self.delta_80_holm_first_step
        return self.delta_80_unadjusted


def read_power_study(run_dir: Path) -> PowerStudy:
    """The power.json of a power run (`PowerStudy` lists its one format)."""
    return read_json_model(run_dir / POWER_FILE, PowerStudy)


def power_json(study: PowerStudy) -> str:
    """power.json as written: the fields in order, indented by two, no final newline."""
    return json.dumps(study.model_dump(mode="json"), indent=2)


def power_study(request: PowerRequest, questions: Sequence[Question], seed: int) -> PowerStudy:
    """The power.json content of a retrieval run.

    It holds the family of recall@k comparisons on the category, the discordance
    and differences it shows, and the simulated power.

    psi and the magnitudes are pooled over the family's comparisons on the same
    questions.

    Raises:
        UserError: The run compares no recall@k on the category. The run has no labeled
            question of the category, or the family shows no discordant pair, so the
            difference is 0. Or per_query.csv has no row of a method of the family on a
            question another one has (the message names it).
        ValueError: per_query.csv has a question `questions` lacks, or `request.studies`
            or `request.permutations` is below 1 (`PowerDesign`).
    """
    pairs = _family(request)
    recall = _recall(request, questions, {m for pair in pairs for m in pair})
    ids = sorted({qid for _, qid in recall})
    if not ids:
        raise UserError(
            f"no labeled {request.category} question in {request.run}",
            hint="pass --category with a category the run's question file labels",
        )
    if lacking := [(m, q) for pair in pairs for m in pair for q in ids if (m, q) not in recall]:
        raise UserError(
            f"{request.run / PER_QUERY_FILE}: {lacking[0][0]} has no row on {lacking[0][1]}",
            hint=RUN_FILE_HINT,
        )
    differences = [recall[(a, qid)] - recall[(b, qid)] for a, b in pairs for qid in ids]
    magnitudes = [abs(d) for d in differences if d != 0]
    if not magnitudes:
        raise UserError(
            f"the recall@{formatting.own_form(request.k)} family on {request.category} shows no "
            "discordant pair: every pair of methods has the same recall on every question",
            hint="pass --k or --category for a family whose methods differ on a question, "
            f"as the run's {PAIRED_FILE} shows",
        )
    psi = len(magnitudes) / len(differences)
    design = PowerDesign(
        magnitudes,
        psi,
        DELTAS,
        len(ids),
        SIGNIFICANCE_LEVEL,
        len(pairs),
        request.studies,
        request.permutations,
        seed,
    )
    points = simulate_power(design)
    return PowerStudy(
        run=str(request.run),
        metric=f"recall@{request.k}",
        category=request.category,
        questions=len(ids),
        family=pairs,
        fractional_values=sum(1 for v in recall.values() if 0 < v < 1),
        values=len(recall),
        discordant_pairs=len(magnitudes),
        question_pairs=len(differences),
        psi=psi,
        mean_magnitude=mean(magnitudes),
        ceiling=design.ceiling,
        alpha=SIGNIFICANCE_LEVEL,
        studies=request.studies,
        permutations=request.permutations,
        seed=seed,
        points=points,
        delta_80_holm_first_step=smallest_simulated_delta(points, TARGET_POWER, holm=True),
        delta_80_unadjusted=smallest_simulated_delta(points, TARGET_POWER, holm=False),
    )


def _family(request: PowerRequest) -> list[tuple[str, str]]:
    """The run's secondary recall@k comparisons over the labeled category.

    Raises:
        UserError: The run compares no recall@k on that category; the hint lists the
            recall cutoffs and categories it does compare.
    """
    rows = read_paired(request.run / PAIRED_FILE).rows
    pairs = [
        (r.a, r.b)
        for r in rows
        if r.metric == f"recall@{request.k}"
        and r.scope == LABELLED_CATEGORY_SCOPES.get(request.category)
        and r.role == "secondary"
    ]
    if not pairs:
        category_of = {scope: c for c, scope in LABELLED_CATEGORY_SCOPES.items()}
        compared = sorted(
            {
                f"--k {r.metric.removeprefix('recall@')} --category {category_of[r.scope]}"
                for r in rows
                if r.metric.startswith("recall@")
                and r.scope in category_of
                and r.role == "secondary"
            }
        )
        raise UserError(
            f"no recall@{formatting.own_form(request.k)} comparisons on {request.category} in "
            f"{request.run / PAIRED_FILE}",
            hint=f"pass the options of a comparison the run makes: "
            f"{'; '.join(compared) or 'it compares no recall on a category'}",
        )
    return pairs


def _recall(
    request: PowerRequest, questions: Sequence[Question], methods: set[str]
) -> dict[tuple[str, str], float]:
    """recall@k of each method on each labeled question of the category, from per_query.csv.

    The category is `Question.effective_category`, the one the paired family scoped on.
    """
    by_id = {q.id: q for q in questions}
    recall: dict[tuple[str, str], float] = {}
    for r in read_per_query(request.run).rows:
        if r.question_id not in by_id:
            raise ValueError(f"{request.run} has {r.question_id}, which `questions` lacks")
        q = by_id[r.question_id]
        if q.effective_category == request.category and q.is_labelled and r.method in methods:
            recall[(r.method, q.id)] = recall_at_k(r.ids, q.gold_chunk_ids, request.k)
    return recall


def write_power_study(results_dir: Path, study: PowerStudy, source: Path) -> Path:
    """Writes `results/<ts>_power/` and returns that run directory.

    The directory holds power.json, and index.json with the source retrieval
    run's `index_version` and question file.

    Raises:
        UserError: `source` has no index.json.
    """
    run_dir = create_run_dir(run_path(results_dir, "_power"))
    (run_dir / POWER_FILE).write_text(power_json(study), encoding="utf-8")
    # The source's `index_version` and question file, each only if the source recorded it.
    kept = {"index_version", "questions", "questions_sha256"}
    copied = run_index(source).model_dump(exclude_unset=True, include=kept)
    write_index(run_dir, RunIndex(**copied), final_newline=True)
    return run_dir


@dataclass(frozen=True)
class PowerRun:
    """A power run as written: its study and its directory."""

    study: PowerStudy
    run_dir: Path


def power_run(cfg: Config, request: PowerRequest, questions: Sequence[Question]) -> PowerRun:
    """The power study of `request.run`, written to `results/<ts>_power/` under `cfg`.

    `questions` are the questions of the run's question file (`run_questions_path`); the
    study is seeded with `eval.bootstrap_seed`.

    Raises:
        UserError: `request.run` has no index.json, or compares no recall@k on the category,
            or has no labeled question of the category, or no discordant pair there.
        ValueError: per_query.csv has a question `questions` lacks, or `request.studies`
            or `request.permutations` is below 1 (`PowerDesign`).
    """
    study = power_study(request, questions, cfg.eval.bootstrap_seed)
    return PowerRun(study, write_power_study(cfg.paths.results, study, request.run))
