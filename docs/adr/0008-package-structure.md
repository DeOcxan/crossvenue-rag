# 0008. Package by domain, layering enforced by import-linter

- Date: 2026-09-24
- Status: Accepted

## Context

The package was 35 flat modules. Two held several responsibilities each (the labeling tool at 873 lines, the CLI at 844), and the import direction between corpus, index, retrieval and evaluation code was a convention nobody checked.

## Decision

- Ten subpackages (`corpus`, `index`, `retrieval`, `evaluation`, `dataset`, `generation`, `labeling`, `reporting`, `cli`, `service`) and a root holding only what at least two layers share (`config`, `config_section`, `config_pipeline`, `config_eval`, `config_generation`, `config_loading`, `models`, `errors`, `stats`, `index_version`, `clock`, `jsonl`, `metric_direction`, `formatting`, `listing`, `hashes`, `run_files`, `retries`, `yaml_files`, `confined_paths`, `rich_text`).
- The import graph runs as the layers contract in `pyproject.toml` defines it, from the bottom: corpus, index, retrieval, then evaluation or dataset, then generation, then labeling or reporting, then cli or service. Four `lint-imports` contracts in `pyproject.toml` enforce it in `make lint` and CI: the layers, a root that imports no subpackage, no cycles among the modules of a subpackage (`reporting.claims` and `reporting.compute` included), and `reporting.compute` never importing `reporting.claims`.
- No module above about 400 lines of code ([code-style rule 2](../code-style.md)); classes only where there is state or several implementations of one contract.
- `tests/` mirrors the package, one test file per module.

## Consequences

- The restructuring is meant to change no behavior; the evaluation that checks it is in [findings](../results/findings.md#moving-the-code-into-subpackages-changes-no-metric).
- `crossvenue_rag.dataset.manifest_fingerprint` is kept as a re-export, because immutable dataset metadata and changelog entries name it.
- A new module has a layer; importing upward fails CI.
