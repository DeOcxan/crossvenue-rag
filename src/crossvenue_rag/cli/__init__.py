"""The `crossvenue` command line.

Commands are plain functions in one module per group (corpus, index, evaluate,
dataset, labeling, generation, reporting). They are registered here, in the
order `--help` lists them, so moving a function between modules never changes
the CLI.
"""

from __future__ import annotations

from . import corpus as _corpus
from . import dataset as _dataset
from . import evaluate as _evaluate
from . import generation as _generation
from . import index as _index
from . import labeling as _labeling
from . import reporting as _reporting
from .app import app, console
from .errors import friendly_errors

app.command()(_corpus.fetch)
app.command()(_corpus.ingest)
app.command()(_corpus.stats)
app.command()(_index.index)
app.command()(_evaluate.retrieve)
app.command("eval")(_evaluate.eval_command)
app.command()(_labeling.label)
app.command("pool-run")(_evaluate.pool_run_command)
app.command("bench-rerank")(_evaluate.bench_rerank_command)
app.command("power")(_evaluate.power_command)
app.command("augment-candidates")(_labeling.augment_candidates_command)
app.command("derive-coverage")(_labeling.derive_coverage_command)
app.command("validate-evals")(_dataset.validate_evals)
app.command("round-sample")(_dataset.round_sample)
app.command()(_dataset.kappa)
app.command()(_dataset.select)
app.command("count-term")(_corpus.count_term)
app.command("id-map")(_index.id_map)
app.command("remap-ids")(_index.remap_ids)
app.command("ask")(_generation.ask_command)
app.command("answer-eval")(_generation.answer_eval_command)
app.command("rate-sample")(_generation.rate_sample_command)
app.command("rate")(_generation.rate_command)
app.command("judge-sample")(_generation.judge_sample_command)
app.command("judge-agreement")(_generation.judge_agreement_command)
app.command("select-judge")(_generation.select_judge_command)
app.command("judge-run")(_generation.judge_run_command)
app.command("compare-prompts")(_generation.compare_prompts_command)
app.command("reproduce")(_reporting.reproduce_command)
app.command("runs-manifest")(_reporting.runs_manifest_command)
app.command()(_corpus.check)


def main() -> None:
    """Console-script entry point, wrapped in the friendly-error boundary."""
    with friendly_errors(console):
        app()


__all__ = ["app", "main"]
