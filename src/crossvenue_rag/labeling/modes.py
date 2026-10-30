"""The `label` command's modes, chosen from its options.

`crossvenue label` labels a round (`label_session`), re-opens round-1
questions for correction (`--review`, `review_session`) or judges the chunks a
candidate list names (`--augment`, `augment_session`). It also decides the pairs rounds 1
and 2 disagree on (`--adjudicate`, `adjudicate_session`), judges named questions
again under the amended criterion (`--relabel`, `relabel_session`) or relabels the
whole set (`--round 3`, `round3_session`). `run_label` refuses option combinations
that mix modes before any index is opened, then runs the one mode the options
select. The last three read chunks from the corpus file and need no database.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import formatting
from ..config import Config
from ..corpus.artifacts import load_chunks
from ..dataset.added_chunks import read_added_chunks
from ..dataset.journal import FULL_RELABEL_ROUND
from ..dataset.questions import load_questions, require_label_versions
from ..dataset.sampling import read_sample_file
from ..errors import UserError
from ..index_version import require_current_index
from ..retrieval.retriever import Retriever
from .adjudicate import AdjudicationTarget, adjudicate_session
from .augment import augment_session
from .pool import pool_depth_of
from .relabel import RelabelTarget, relabel_session
from .review import review_session
from .round3 import Round3Target, round3_session
from .session import LabelPlan, LabelTarget, QueueSpec, default_annotator, label_session


@dataclass(frozen=True)
class LabelRequest:
    """The `label` command's options, one field per option."""

    questions_path: str
    ids: Sequence[str] = ()
    review: bool = False
    review_file: Path | None = None
    augment: bool = False
    question_file: Path | None = None
    revisit: bool = False
    limit: int | None = None
    pool_depth: int | None = None
    annotation_round: int = 1
    annotator: str = ""
    sample: int | None = None
    seed: int = 0
    sample_file: Path | None = None
    adjudicate: bool = False
    relabel: bool = False
    add_chunks: Path | None = None
    metadata_from: Path | None = None

    @property
    def augments(self) -> bool:
        """Whether any `--augment` option was given."""
        return self.augment or self.question_file is not None

    @property
    def other_modes(self) -> list[str]:
        """The options of the round, review and augment modes this request sets."""
        given = {
            "--review": self.review or self.review_file is not None,
            "--augment": self.augments,
            "--round": self.annotation_round != 1,
            "--sample": self.sample is not None,
            "--sample-file": self.sample_file is not None,
            "--all": self.revisit,
        }
        return [option for option, on in given.items() if on]


def check_adjudicate(request: LabelRequest) -> None:
    """Refuses `--adjudicate` combined with question ids or another mode's options.

    Raises:
        UserError: The request names question ids or another mode's options.
    """
    others = request.other_modes + (["--relabel"] if request.relabel else [])
    others += ["--add-chunks"] if request.add_chunks is not None else []
    if request.ids or others:
        raise UserError(
            "--adjudicate walks every disagreeing pair and takes no ids or other mode: "
            + " ".join(list(request.ids) + others),
            hint="crossvenue label --adjudicate",
        )


def check_relabel(request: LabelRequest) -> list[str]:
    """The ids `--relabel` judges; refuses it without ids or with another mode's options.

    Raises:
        UserError: `--add-chunks` comes without `--relabel`, no id is given, or another
            mode's options are set.
    """
    if request.add_chunks is not None and not request.relabel:
        raise UserError("--add-chunks goes with --relabel")
    if not request.ids:
        raise UserError(
            "--relabel needs question ids", hint="crossvenue label --relabel ID [ID ...]"
        )
    if request.other_modes:
        raise UserError(f"--relabel cannot be combined with {' '.join(request.other_modes)}")
    return list(request.ids)


def check_round3(request: LabelRequest) -> tuple[Path, Path]:
    """The added-chunks file and metadata file of `--round 3`; refuses anything else.

    Raises:
        UserError: The round is later than round 3, `--metadata-from` comes without
            `--round 3`, a required option is missing, or another mode's option is set.
    """
    own = formatting.own_form(FULL_RELABEL_ROUND)
    if request.annotation_round > FULL_RELABEL_ROUND:
        raise UserError(
            f"there is no round {formatting.own_form(request.annotation_round)}",
            hint=f"methodology section 5b ends the labeling at round {own}",
        )
    if request.annotation_round != FULL_RELABEL_ROUND:
        raise UserError(f"--metadata-from goes with --round {own}")
    if request.add_chunks is None or request.metadata_from is None:
        raise UserError(
            f"--round {own} needs --add-chunks and --metadata-from",
            hint="make round3",
        )
    given = {
        "--review": request.review or request.review_file is not None,
        "--augment": request.augments,
        "--adjudicate": request.adjudicate,
        "--relabel": request.relabel,
        "--all": request.revisit,
        "--sample": request.sample is not None,
        "--sample-file": request.sample_file is not None,
        "--limit": request.limit is not None,
        "--pool-depth": request.pool_depth is not None,
    }
    others = [option for option, on in given.items() if on]
    if request.ids or others:
        raise UserError(
            f"--round {own} walks every question and takes no ids or other mode: "
            + " ".join(list(request.ids) + others),
            hint="make round3",
        )
    return request.add_chunks, request.metadata_from


def check_augment(request: LabelRequest) -> Path:
    """The candidate list `--augment` judges; refuses it combined with another mode.

    Raises:
        UserError: `--augment` and `--question-file` do not come together, or another
            mode's options are set.
    """
    if not request.augment or request.question_file is None:
        raise UserError(
            "--augment and --question-file go together",
            hint=(
                "crossvenue label --augment --question-file evals/selection/augment_round1.jsonl"
            ),
        )
    if request.ids or request.review or request.review_file or request.annotation_round != 1:
        raise UserError("--augment extends round 1 and cannot be combined with other modes")
    return request.question_file


def review_ids(request: LabelRequest) -> list[str]:
    """The ids `--review` re-opens, empty when labeling; refuses a mixed request.

    Raises:
        UserError: Ids come without `--review`, `--review` comes without ids, `--sample-file`
            comes outside a later round or with `--sample`, or ids come with `--round`.
    """
    ids = list(request.ids)
    if request.review_file is not None:
        ids += read_sample_file(request.review_file)
    reviewing = request.review or request.review_file is not None
    if ids and not reviewing:
        raise UserError(
            f"question ids given without --review: {' '.join(ids)}",
            hint="use `crossvenue label --review ID [ID ...]`",
        )
    if reviewing and not ids:
        raise UserError("--review needs question ids", hint="or pass --review-file PATH")
    _check_sample_file(request)
    if ids and request.annotation_round != 1:
        raise UserError("--review corrects round 1; it cannot be combined with --round")
    return ids


def _check_sample_file(request: LabelRequest) -> None:
    """Refuses `--sample-file` outside a later round or together with `--sample`."""
    if request.sample_file is not None and (
        request.annotation_round == 1 or request.sample is not None
    ):
        raise UserError(
            "--sample-file selects a re-annotation subsample",
            hint="use it with --round 2 and without --sample",
        )


def label_plan(request: LabelRequest, index_version: str, per_method: int) -> LabelPlan:
    """The labeling session the options describe, against `index_version`.

    `per_method` is the pool depth `pool.pool_depth_of` returned for `--pool-depth`.
    """
    sample_ids = read_sample_file(request.sample_file) if request.sample_file else None
    return LabelPlan(
        target=LabelTarget(request.questions_path, index_version, request.annotator),
        queue=QueueSpec(
            annotation_round=request.annotation_round,
            only_unlabelled=not request.revisit,
            sample=request.sample,
            seed=request.seed,
            sample_ids=sample_ids,
        ),
        per_method=per_method,
        limit=request.limit,
    )


def run_offline(cfg: Config, request: LabelRequest) -> None:
    """Runs `--adjudicate` or `--relabel` on the chunks file, checked against the built index."""
    if request.adjudicate:
        check_adjudicate(request)
        ids: list[str] = []
    else:
        ids = check_relabel(request)
    added = read_added_chunks(request.add_chunks) if request.add_chunks else {}
    version = require_current_index(cfg)
    require_label_versions(request.questions_path, version, all_rounds=True)
    chunks = {c.chunk_id: c for c in load_chunks(cfg)}
    annotator = request.annotator or default_annotator()
    if request.adjudicate:
        adjudicate_session(AdjudicationTarget(request.questions_path, chunks, version, annotator))
        return
    relabel_session(RelabelTarget(request.questions_path, ids, added, chunks, version, annotator))


def run_round3(cfg: Config, request: LabelRequest) -> None:
    """Runs `--round 3` on the chunks file, checked against the built index."""
    added_file, metadata = check_round3(request)
    added = read_added_chunks(added_file)
    version = require_current_index(cfg)
    require_label_versions(request.questions_path, version, all_rounds=True)
    require_label_versions(metadata, version)
    target = Round3Target(
        request.questions_path,
        load_questions(metadata),
        added,
        {c.chunk_id: c for c in load_chunks(cfg)},
        version,
        request.annotator or default_annotator(),
    )
    round3_session(target)


def run_label(cfg: Config, request: LabelRequest) -> None:
    """Checks the options, opens the index and runs the mode they select."""
    if request.annotation_round >= FULL_RELABEL_ROUND or request.metadata_from is not None:
        run_round3(cfg, request)
        return
    if request.adjudicate or request.relabel or request.add_chunks is not None:
        run_offline(cfg, request)
        return
    if request.augments:
        candidates = check_augment(request)
        with Retriever(cfg) as retriever:
            target = LabelTarget(
                request.questions_path, retriever.require_current_index(), request.annotator
            )
            augment_session(retriever, target, candidates)
        return
    ids = review_ids(request)
    per_method = pool_depth_of(cfg, request.pool_depth, "--pool-depth")
    with Retriever(cfg) as retriever:
        version = retriever.require_current_index()
        retriever.warmup()
        if ids:
            target = LabelTarget(request.questions_path, version, request.annotator)
            review_session(cfg, retriever, target, ids)
            return
        label_session(cfg, retriever, label_plan(request, version, per_method))
