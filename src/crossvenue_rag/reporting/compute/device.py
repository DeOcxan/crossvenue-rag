"""The compute device's effect on retrieval: the CPU run on `v0.2.1-gold` against the MPS run.

`run_ids.SENSITIVITY_BASE_RUN` was made with `embedding.device=cpu` and `run_ids.RETRIEVAL_RUN`
on MPS, against the same index and question file, so a difference between their lists comes
from the device.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..run_ids import RETRIEVAL_RUN, SENSITIVITY_BASE_RUN
from ..sources import Sources
from .memo import per_sources

CPU_RUN = SENSITIVITY_BASE_RUN
MPS_RUN = RETRIEVAL_RUN
# The metrics a reordering of a list leaves unchanged: they change only when its chunks change.
MEMBERSHIP_METRICS = ("recall", "precision")
# A difference "in the fourth decimal" is below this.
FOURTH_DECIMAL = 0.001


@dataclass(frozen=True)
class ListChange:
    """How many (question, method) lists the two runs share, and which of them differ."""

    shared: int
    differing: int
    methods: frozenset[str]  # the methods with at least one differing list


@dataclass(frozen=True)
class CellChange:
    """One by_category.csv cell on the two devices."""

    category: str
    method: str
    metric: str
    cpu: float
    mps: float


@per_sources
def list_change(src: Sources) -> ListChange:
    """The retrieved lists of the CPU run against those of the MPS run."""
    cpu = {(r.question_id, r.method): r.retrieved_ids for r in src.run(CPU_RUN).per_query}
    mps = {(r.question_id, r.method): r.retrieved_ids for r in src.run(MPS_RUN).per_query}
    shared = set(cpu) & set(mps)
    differing = [key for key in shared if cpu[key] != mps[key]]
    return ListChange(len(shared), len(differing), frozenset(m for _, m in differing))


def changed_cells(src: Sources) -> list[CellChange]:
    """Every metric cell of by_category.csv that differs between the devices.

    Largest difference first; latency is left out.
    """
    cpu, mps = src.run(CPU_RUN).by_category, src.run(MPS_RUN).by_category
    changes = []
    for key in sorted(set(cpu) & set(mps)):
        for column in mps[key].model_extra or {}:
            # A metric is `mrr` or a metric at a cutoff; the slice column and latency are not.
            if not ("@" in column or column == "mrr") or column not in (cpu[key].model_extra or {}):
                continue
            a, b = cpu[key].number(column), mps[key].number(column)
            if a is not None and b is not None and a != b:
                changes.append(CellChange(*key, column, a, b))
    return sorted(changes, key=lambda c: -abs(c.cpu - c.mps))


def membership_changes(src: Sources) -> list[CellChange]:
    """The recall and precision cells among `changed_cells`, largest difference first."""
    return [c for c in changed_cells(src) if c.metric.split("@")[0] in MEMBERSHIP_METRICS]


def ranking_only_changes(src: Sources, category: str) -> bool:
    """Whether on `category` only nDCG and MRR change, below 0.001, besides dense's unjudged@10."""
    for c in changed_cells(src):
        if c.category != category:
            continue
        if (c.method, c.metric) == ("dense", "unjudged@10"):
            continue
        if c.metric.split("@")[0] not in ("ndcg", "mrr") or abs(c.cpu - c.mps) >= FOURTH_DECIMAL:
            return False
    return True
