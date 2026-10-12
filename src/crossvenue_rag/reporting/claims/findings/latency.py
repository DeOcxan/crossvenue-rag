"""docs/results/findings.md, the time a query takes: the runs and the reranker bench."""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import bench, datasets, retrieval, retrieval_tables
from ...run_ids import RETRIEVAL_RUN
from ...sources import Sources
from ..comparisons import level, min_tested, question_count_text, signed_ms_parts

# How the bench table names each mode.
MODE_LABELS: dict[str, str] = {
    bench.PER_VENUE: "one call per venue",
    bench.ONE_CALL: "one call for all venues",
}


def latency(doc: Doc, src: Sources) -> None:
    """Latency per retrieval step."""
    doc.caption(
        "run `results/{run}`, the {} questions of `v0.2.1-gold`, methods looped",
        (RETRIEVAL_RUN,),
        question_count_text(src),
    )
    doc.table(
        "| method | encode |",
        {m: [formatting.ms(v) for v in values] for m, values in retrieval_tables.latency_rows(src)},
    )


def reranker_bench(doc: Doc, src: Sources) -> None:
    """The bench of one cross-encoder call per venue against one for all venues."""
    count = formatting.count(bench.timed_question_count(src))
    doc.claim(
        "times both on the {} questions of `v0.2.1-gold` where it splits",
        count,
        holds=bench.times_split_questions(src),
        what="the bench timed exactly the questions venue_split splits",
    )
    doc.table(
        "| venue_split | calls per query |",
        {
            MODE_LABELS[mode]: [
                formatting.ratio(calls, formatting.BENCH_SUMMARY_DECIMALS),
                formatting.ratio(pairs, formatting.COUNT_MEAN_DECIMALS),
                formatting.ms(p50),
                formatting.ms(p95),
                formatting.ms(mean),
            ]
            for mode, (calls, pairs, p50, p95, mean) in bench.mode_rows(src)
        },
    )


def bench_saving(doc: Doc, src: Sources) -> None:
    """What one call saves, paired by question, and what both modes retrieve."""
    count = formatting.count(bench.timed_question_count(src))
    rerank = bench.paired_difference(src, "rerank_ms")
    total = bench.paired_difference(src, "total_ms")
    first, second = bench.outside_rerank(src)
    doc.claim(
        "Paired by question, per-venue calls \u2212 one call is {} ms [{}, {}] ({}% paired "
        "bootstrap interval) in the rerank step; one call is faster on {} of {} "
        "questions. The {} questions are a slice of fewer than {}, so under methodology \u00a77 "
        "this comparison is descriptive. On the steps sum per-venue calls \u2212 one call is {} "
        "ms [{}, {}], "
        "inconclusive at this size. The mode timed first in each pair spends more outside the "
        "rerank step (mean {} ms against {} ms)",
        *signed_ms_parts(rerank),
        level(),
        formatting.count(bench.one_call_faster(src)),
        count,
        count,
        min_tested(),
        *signed_ms_parts(total),
        formatting.ms(first),
        formatting.ms(second),
        holds=bench.descriptive_with_saving(src),
        what=f"the bench slice is below {min_tested()} questions and the rerank saving's "
        "interval excludes 0",
    )
    doc.claim(
        "score the same {} pairs per question and return the same retrieval on all {} questions",
        formatting.ratio(
            bench.mode_value(src, bench.ONE_CALL, "pairs_per_query_mean"),
            formatting.COUNT_MEAN_DECIMALS,
        ),
        count,
        holds=bench.modes_agree(src),
        what="both modes score the same pairs and retrieve the same ids on every question",
    )


def bench_machine(doc: Doc, src: Sources) -> None:
    """The machine the bench ran on, and the bench against the evaluation run."""
    median, highest = bench.load_average(src)
    doc.check(
        "the 1-minute load average during timing had a median of {} and a maximum of {} on {} CPUs",
        formatting.ratio(median, formatting.LOAD_AVERAGE_DECIMALS),
        formatting.ratio(highest, formatting.LOAD_AVERAGE_DECIMALS),
        formatting.count(bench.cpu_count(src)),
    )
    doc.check(
        "and {} pairs that failed the second check", formatting.count(bench.discarded_pairs(src))
    )
    doc.check(
        "On the same {} questions the evaluation run measured venue_split's p50 at {} ms",
        formatting.count(len(datasets.split_ids(src))),
        formatting.ms(retrieval.split_p50(src, "venue_split")),
    )
    doc.check(
        "{}% below the per-venue mode here ({} ms)",
        formatting.percent_figure(*bench.eval_below_bench(src)),
        formatting.ms(bench.mode_value(src, bench.PER_VENUE, bench.TOTAL_P50)),
    )
    doc.check(
        "Its {} ms over all {} questions includes the {} questions venue_split answers as plain "
        "hybrid_rerank.",
        formatting.ms(retrieval.summary_metric(src, "venue_split", "p50_ms")),
        question_count_text(src),
        formatting.count(datasets.unsplit_count(src)),
    )
