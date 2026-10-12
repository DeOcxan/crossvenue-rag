# Code style

Rules for the Python in `src/`, `tests/` and `scripts/`. The base is [PEP 8](https://peps.python.org/pep-0008/) as `ruff format` applies it and the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings; where this file differs from them, this file applies. Every rule is checked by reading; the tools in [Checks](#checks) are aids.

The governing rule is that a module has one responsibility that its name states, and a function does one thing a caller can name.

## Structure

1. The package is split by domain, one subpackage per domain, and the import direction between them is fixed ([ADR 0008](adr/0008-package-structure.md)). `lint-imports` enforces the layers.

2. A module has one responsibility, stated in its docstring in one sentence. A module whose docstring needs "and" to say what it does is two modules. Above about 400 lines of code, a module is split along its responsibilities.

3. Code that two modules need is in the lowest layer both can import, in a module named by its subject: `jsonl.py`, `clock.py`, `stats.py`, `formatting.py`. No module is named `utils`, `helpers`, `common` or `misc`: a name that says nothing about the contents collects unrelated code.

4. One implementation per concept. A function is written only after a search for one that does the same, and an existing one is extended rather than a second one written. Two public functions with the same name and different behavior are a defect, even in different modules; a private function's name (`_name`) is local to its module. The one exception is a `cli/` command function named after its command that calls a library function of the same name for the same action: `ingest` in `cli/corpus.py` calls `corpus.ingest.ingest`. A script's entry point is `main`, as the CLI's is.
   - Bad: one `mean` returns 0.0 on empty input and another `mean` returns None.

5. Input and output stay at the edges. `cli/` and `service/` parse arguments, call one library function and render its result; they hold no computation. A function that reads or writes files is separate from the function that computes on the data, so the computation is testable without files.

6. Classes only where there is state or several implementations of one contract; everything else is functions. A record passed between functions is a dataclass or a pydantic model with named fields, never a `dict[str, Any]` whose keys the reader has to guess. One exception, with its reason. Where JSON leaves the program, a function that serializes a record for its writer returns `dict[str, Any]` (`AnswerRecord.to_json`, the `to_dict` of `models`, `CsvRow.cells`, a search hit's metadata), because the writer takes any mapping. The judgment journal is a model, `dataset.journal`, read and written as its records.

7. Dependencies are passed in: models, clients, token counters, clocks. Tests pass fakes, so no test loads weights, calls a network or needs a database.

## Functions

8. A function does one thing its name states. A part is extracted into its own function when it has its own name and can be tested alone. It stays inline when a reader would need both parts open to understand either one.

9. Cyclomatic complexity is at most 10 (ruff `C901`). Above about 40 lines of body, the function is reconsidered for a split along rule 8.

10. At most 6 parameters (ruff `PLR0913`). More become a parameter object: a frozen dataclass named for what it configures. CLI command functions are exempt, since each parameter is a command-line option.

11. Boolean flags are keyword-only (ruff `FBT`): `load(path, strict=True)`, never `load(path, True)`. CLI command functions are exempt, since each boolean is a command-line flag.

12. No unnamed numbers in logic. A value with a meaning is a named constant or a config key in `configs/default.yaml`; 0, 1 and index offsets are exempt.

13. A function that computes returns its result. It does not print, log or write files; the caller decides what to do with the result.

## Errors

14. A problem the user can fix raises `errors.UserError` with a message that names the fix. Any other exception propagates with its traceback. User input is validated where it enters: a command option in the command's first lines, a config key in the config model (a `Field` bound or a validator), a file in the loader that reads it. A library function no user calls directly states its preconditions in its docstring and raises `ValueError` when one fails.

15. No bare `except`, no `except Exception`, and no handler that discards the error. A handler catches the exact exception it expects and either recovers in a stated way or raises `UserError`. The one exception is the CLI error boundary, `friendly_errors` in `cli/errors.py`. It catches `Exception`, prints an expected failure (a `UserError`, or an expected external failure: the database refusing a connection, a model call that fails) as its message and hint, and re-raises everything else with its traceback.

16. `assert` appears only in tests. An invariant in `src/` raises an explicit exception, because `python -O` removes assertions.

17. No special case for a particular question id, run, venue or file. Code that reads an earlier format of a saved artifact is in one reader per artifact kind, whose docstring lists every format it accepts.

## Types

18. Every function in `src/` and `scripts/` is fully annotated, and `mypy --strict` passes on them. `Any` appears only where JSON enters the program, and the value is converted to a model in the same function; rule 6's exception applies here too.

19. Test functions need no annotations; fixtures and fakes shared across files are annotated.

## Docstrings and comments

20. Every module starts with a docstring of one to three sentences saying what the module is for. Files carry no author, date, version history or changelog: git records them.

21. Every public function and class (a name that does not start with `_`) has a Google-style docstring. The first line is a one-line summary, a descriptive phrase of what the function returns or does (`The time of a record: ...`). `Args`, `Returns` and `Raises` sections appear only for what the name and the type hints do not already say. The docstring is complete when a caller can call the function correctly without reading its body. A private function whose name and types say everything has no docstring.

22. A comment says why, or states what the code cannot: a decision, a constraint, a trap, a source. It never restates the code.
    - Bad: `# loop over the chunks`
    - Good: `# Byte-identical text on two venues must never merge, so the venue is part of the hash.`

23. No commented-out code, no `TODO` or `FIXME`, no divider comments (`# ------`), no numbered step comments, and no definition that only its own module's tests use. Unfinished work is recorded where it can be tracked, such as a direction or an ADR. No definition, parameter, option or abstraction that nothing in `src/` or `scripts/` uses.

24. Comments and docstrings are prose and follow [`style.md`](style.md).

## Output text

25. Text that reaches a person is anything printed and every Markdown file written into `results/`: CLI output, error messages and their hints, `--help` text, run reports and the error details the service returns. It follows [`style.md`](style.md), and formats numbers through one module (`formatting`) that applies style rule 11 and writes the minus sign as U+2212. It has no `(s)` plurals, no `--` as a dash and no raw representation of a list or dict. CSV and JSON files are data for programs: each writer keeps the precision the committed runs of its kind have (`formatting.py`). A value in printed text is printed with markup and highlighting off, through `cli.app.say`; styled static text is a rich `Text` or a style argument, never markup in a string that also holds a value.

26. Names in a saved data schema (CSV columns, JSON fields) keep their spelling once a committed run uses them. Renaming one is a schema change, and its reader accepts both names.

## Tests

27. `tests/` mirrors `src/`: every module has one test file of its name (`tests/labeling/test_judge.py` tests `labeling/judge.py`), and every test file is named after a module. A test calls the public function and checks behavior, never a private helper's internals. The exceptions are exactly these, and `tests/test_repo_hygiene.py` checks both directions against the same list:
    - suite-level tests, which exercise no single module: `test_mutations.py`, `test_pipeline_smoke.py`, `test_repo_hygiene.py`, `test_code_hygiene.py`, `test_claims_adr.py`, `test_input_domains.py`;
    - modules under `reporting/claims/`, which `make reproduce` and the mutation test check against the documents;
    - modules with no behavior, only constants or Protocol classes: `reporting/run_ids.py`, `retrieval/protocols.py`.

28. A test name states the behavior it checks: `test_a_file_that_differs_from_its_recorded_hash_fails_the_fetch`.

29. A fixed defect comes with a test that fails without the fix.

## Imports

30. Imports are at the top of the module. An import inside a function is kept only when it defers a heavy dependency, so that commands which do not need it start fast. The heavy dependencies are psycopg, pgvector, bm25s, Stemmer, torch, sentence_transformers, transformers, scipy and scikit-learn, and a module that loads them, such as the claims. The module's docstring says so once, and `pyproject.toml` allows it for that module alone. Tests are exempt, and `pyproject.toml` allows it for `tests/` as a whole: a test may import inside the test function, since the start-up time this rule protects is that of the commands.

## Checks

Every rule is checked by reading. The tools are aids: each runs for the rules its row names, and a clean run does not show that the code meets them. `make lint` runs the first six; `make reproduce` runs the last.

| Tool | Rules it runs for |
|---|---|
| `ruff format --check` | PEP 8 layout |
| `ruff check`, with the rules `pyproject.toml` selects | 9, 10, 11, 12, 13, 15, 16, 20, 21, 23, 30 |
| `mypy`, strict for `src/` and `scripts/`, relaxed for `tests/` | 18 |
| `lint-imports`, with the contracts in `pyproject.toml` | 1 |
| `tests/test_repo_hygiene.py` | 25, 27 |
| `tests/test_code_hygiene.py` | 2 and 6 over `src/` and `scripts/`; 4, 15, 23, 24, 25 |
| `make reproduce` | 2 |

`pyproject.toml` holds two kinds of exemption. Policy, each entry naming its rule and reason: `tests/*` (rules 12, 16 and 30, and rules 20 and 21 because rule 28 makes a test's name state its behavior), the command modules of `cli/` (rules 10 and 11), the modules that defer a heavy import (rule 30), and mypy's relaxed checks for `tests/` under `[[tool.mypy.overrides]]` (rule 19). Debt: a rule a file does not meet yet, listed under `per-file-ignores` and removed when the file is fixed, so that list only shrinks; it holds no entry.
