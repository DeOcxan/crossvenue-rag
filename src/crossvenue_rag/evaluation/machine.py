"""The machine a timing runs on: its load, the processes competing with the timing, machine.json.

`bench-rerank` checks the machine before and after each timed pair and saves the
state around the whole run; command lines of other processes are never saved.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .. import formatting
from ..clock import utc_now_with_offset
from ..run_files import read_json_model

# The columns of the `ps` line read: pid, ppid, pcpu and the command.
PS_FIELDS = 4


# Processes that would compete with the timing for CPU or the accelerator.
COMPETITORS = ("crossvenue", "pytest", "uvicorn", "sentence_transformers")
# A process of any name at or above this CPU share is recorded as busy; it does not block timing.
BUSY_CPU_PERCENT = 50.0
# A process named in COMPETITORS competes from this CPU share; below it, it is idle.
COMPETING_CPU_PERCENT = 5.0
# The 1-minute load average per CPU above which the machine is too busy to time on.
BUSY_LOAD_PER_CPU = 0.5
# The characters of a process's command line a record keeps.
COMMAND_EXCERPT = 120

MACHINE_FILE = "machine.json"
# The decimals of a load average in machine.json and raw.csv.
LOAD_DECIMALS = 2


class Process(BaseModel):
    """A process running while the bench checks the machine."""

    model_config = ConfigDict(frozen=True)

    pid: int
    cpu_percent: float
    command: str


class MachineState(BaseModel):
    """The load average and the processes that compete for the machine or keep it busy."""

    model_config = ConfigDict(frozen=True)

    measured_at: str
    cpu_count: int | None
    load_average: tuple[float, float, float]
    competing_processes: tuple[Process, ...]
    busy_processes: tuple[Process, ...]


class RecordedState(BaseModel):
    """A machine state as machine.json saves it: the processes as counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    measured_at: str
    cpu_count: int | None
    load_average: tuple[float, float, float]
    competing_processes: int
    busy_processes: int


class LoadDuring(BaseModel):
    """The 1-minute load average while timing; no field when no row recorded it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    median: float | None = None
    max: float | None = None


class MachineRecord(BaseModel):
    """machine.json: the machine before and after timing and the load while timing.

    One format, the bench run 20260925T225850Z_bench's: two-space indent, no
    final newline, `load1_during` an empty object when no row recorded a load.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    before: RecordedState
    after: RecordedState
    discarded_pairs: int
    load1_during: LoadDuring


def machine_json(record: MachineRecord) -> str:
    """machine.json's text for `record`."""
    return json.dumps(record.model_dump(mode="json", exclude_unset=True), indent=2)


def read_machine(run_dir: Path) -> MachineRecord:
    """machine.json of a bench run."""
    return read_json_model(run_dir / MACHINE_FILE, MachineRecord)


@dataclass(frozen=True)
class ProcessRow:
    """One row of the process table."""

    pid: int
    ppid: int
    cpu_percent: float
    command: str


def machine_state() -> MachineState:
    """The machine's load average and competing processes at the moment of the call.

    The bench takes it before timing, at every quiet check and after timing.
    """
    load1, load5, load15 = os.getloadavg()
    ps = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pcpu=,command="], capture_output=True, text=True, check=False
    ).stdout
    competitors, hot = classify_processes(parse_process_table(ps), os.getpid())
    return MachineState(
        measured_at=utc_now_with_offset(),
        cpu_count=os.cpu_count(),
        load_average=(
            round(load1, LOAD_DECIMALS),
            round(load5, LOAD_DECIMALS),
            round(load15, LOAD_DECIMALS),
        ),
        competing_processes=tuple(competitors),
        busy_processes=tuple(hot),
    )


def parse_process_table(ps: str) -> list[ProcessRow]:
    """The rows of `ps -axo pid=,ppid=,pcpu=,command=` output; an unparsable line is skipped.

    A line parses when it has four fields and the first three are a pid, a parent pid and a
    CPU share.
    """
    table = []
    for line in ps.splitlines():
        parts = line.split(None, PS_FIELDS - 1)
        if len(parts) != PS_FIELDS:
            continue
        try:
            table.append(ProcessRow(int(parts[0]), int(parts[1]), float(parts[2]), parts[3]))
        # A header, or a field ps could not fill: the line is not a process row.
        except ValueError:
            continue
    return table


def classify_processes(
    table: Sequence[ProcessRow], own_pid: int
) -> tuple[list[Process], list[Process]]:
    """The competing and the busy processes of `table`, leaving out `own_pid` and its parents."""
    # This process and the shells and `uv run` that started it are not competitors.
    parent_of = {row.pid: row.ppid for row in table}
    mine = {own_pid}
    pid = own_pid
    while pid in parent_of and parent_of[pid] not in mine and parent_of[pid] > 1:
        pid = parent_of[pid]
        mine.add(pid)
    competitors: list[Process] = []
    hot: list[Process] = []
    for row in table:
        pid, cpu, command = row.pid, row.cpu_percent, row.command
        # The shell wrapping this command names it; it is not a competitor either.
        if pid in mine or "bench-rerank" in command:
            continue
        # An idle interactive session (`crossvenue rate` waiting for a key) is
        # not a run; a process competes when it matches and uses CPU.
        if any(name in command for name in COMPETITORS) and cpu >= COMPETING_CPU_PERCENT:
            competitors.append(Process(pid=pid, cpu_percent=cpu, command=command[:COMMAND_EXCERPT]))
        elif cpu >= BUSY_CPU_PERCENT:
            hot.append(Process(pid=pid, cpu_percent=cpu, command=command[:COMMAND_EXCERPT]))
    return competitors, hot


def recordable(state: MachineState) -> RecordedState:
    """The machine state as saved: load and counts, no process command lines.

    Command lines are left out because they name other people's and other
    projects' work.
    """
    return RecordedState(
        measured_at=state.measured_at,
        cpu_count=state.cpu_count,
        load_average=state.load_average,
        competing_processes=len(state.competing_processes),
        busy_processes=len(state.busy_processes),
    )


def busy_reasons(state: MachineState) -> list[str]:
    """Why the machine is too busy to time on, empty when it is quiet.

    A reason is a 1-minute load average above `BUSY_LOAD_PER_CPU` per CPU, or another
    run, named by its pid, an identifier printed without a thousands separator.
    Processes merely using CPU (a browser) are recorded in `busy_processes` and
    do not block.
    """
    reasons = []
    cpus = state.cpu_count or 1
    one_minute = state.load_average[0]
    if one_minute > cpus * BUSY_LOAD_PER_CPU:
        shown = formatting.ratio(one_minute, formatting.LOAD_AVERAGE_DECIMALS)
        reasons.append(f"1-minute load average {shown} on {formatting.cpus(state.cpu_count)}")
    reasons += [f"pid {formatting.own_form(p.pid)}: {p.command}" for p in state.competing_processes]
    return reasons
