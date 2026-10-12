"""Tests for `cli.render`: a table shows what it is given, and nothing more."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from crossvenue_rag.cli import render
from crossvenue_rag.corpus.artifacts import TermCount
from crossvenue_rag.dataset.agreement import AgreementReport
from crossvenue_rag.models import Result, Timings
from tests.conftest import power_study_of


def text(renderable) -> str:
    console = Console(record=True, width=160)
    console.print(renderable)
    return console.export_text()


def test_term_counts_list_venues_by_count_with_the_total():
    out = text(render.term_counts("recvWindow", TermCount(10, {"bybit": 3, "binance": 7}), None))
    assert '"recvWindow"' in out and "chunks containing" in out
    assert out.index("binance") < out.index("bybit") and "10" in out


def test_the_agreement_table_names_kappa_and_its_reading():
    report = AgreementReport(10, 2, 7, 1, 0.9, 0.6, 0.75, 3)
    out = text(render.rounds_agreement_table(report, 1, 2))
    assert "rounds 1 vs 2" in out and "0.750" in out and "substantial" in out
    assert "comparison" in text(render.kappa_by_category_table([("comparison", report)]))


def test_retrieval_results_show_the_steps_sum():
    result = Result("c1", 0.5, "text", "bybit", "cex", "a.md", ["H"], "u", sources=[])
    out = text(render.retrieval_results("bm25", 1, [result], Timings(search_ms=1.5)))
    assert "bybit" in out and "total 1.5 ms" in out


def test_the_retrieval_title_writes_its_depth_in_the_form_of_the_k_option():
    result = Result("c1", 0.5, "text", "bybit", "cex", "a.md", ["H"], "u", sources=[])
    out = text(render.retrieval_results("bm25", 1000, [result], Timings()))
    assert "bm25 | k=1000 |" in out


def test_rendered_tables_have_no_output_defects():
    from crossvenue_rag.cli.render import dedup_summary, metadata_coverage
    from crossvenue_rag.corpus.artifacts import DedupSummary, MetadataCoverage
    from tests.conftest import output_defects

    stats = {
        "chunks_before": 12783,
        "chunks_after": 11934,
        "removed": 849,
        "groups": 536,
        "occurrences_in_groups": 1385,
        "groups_by_venue": {"binance": 433},
        "testnet_mirror_groups": 433,
        "cross_venue_groups": 0,
        "cross_venue_examples": [],
    }
    out = text(dedup_summary(DedupSummary.model_validate(stats)))
    assert "12,783" in out and "1,385 occurrences" in out
    coverage = {
        "chunks": 11934,
        "fields": {
            "environment": {
                "assigned": 1298,
                "share": 0.1088,
                "values": {"mainnet": 893},
                "unassigned_by_venue": {"bybit": 3145},
            }
        },
    }
    out += text(metadata_coverage(MetadataCoverage.model_validate(coverage)))
    assert "10.9%" in out and "bybit=3,145" in out
    result = Result("c1", -0.51234, "text", "bybit", "cex", "a.md", ["H"], "u", sources=[])
    out += text(render.retrieval_results("bm25", 1, [result], Timings(rerank_ms=2505.3)))
    assert "2,505 ms" in out and "\u22120.512" in out
    assert output_defects(out) == []


def test_fetch_summary_groups_rows_by_venue_and_dashes_a_missing_commit():
    from crossvenue_rag.corpus.manifest import ManifestRow

    def row(venue: str, sha: str) -> ManifestRow:
        return ManifestRow(venue, "cex", "https://example.invalid", "a.md", "t", sha, "0" * 64)

    out = text(render.fetch_summary([row("okx", ""), row("bybit", "a" * 40), row("bybit", "")]))
    assert "aaaaaaaa\u2026" in out and "a" * 9 not in out
    assert out.index("bybit") < out.index("okx") and " - " in out and "total" in out


def test_corpus_tables_show_venues_totals_and_the_histogram():
    from crossvenue_rag.corpus.artifacts import CorpusStats

    venue = {
        "venue": "bybit",
        "venue_type": "cex",
        "documents": 2,
        "chunks": 1200,
        "tokens_total": 5000,
        "tokens_mean": 4.2,
        "tokens_p50": 4,
        "tokens_p95": 9,
        "tokens_max": 12,
    }
    totals = {"documents": 2, "chunks": 1200, "tokens": 5000, "tokens_p50": 4, "tokens_p95": 9}

    def stats(histogram: dict[str, int]) -> CorpusStats:
        return CorpusStats.model_validate(
            {"by_venue": [venue], "totals": totals, "chunk_length_histogram": histogram}
        )

    out = text(render.corpus_stats_table(stats({})))
    assert "bybit" in out and "1,200" in out and "total" in out
    saved = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", "500-750"]
    histogram = {**dict.fromkeys(saved, 0), "0-50": 4, "750-1000": 0, "1000+": 2}
    hist = text(render.chunk_length_histogram(stats(histogram)))
    assert "0\u201349" in hist and "\u2588" * 40 in hist
    assert "750\u2013999" in hist and "1000" not in hist
    assert "\u2588" * 21 not in hist.split("1,000 and over")[1]


def test_validation_output_prints_messages_as_written_and_sorts_counts():
    from crossvenue_rag.dataset.validate import ValidationReport, ValidationStats

    stats = ValidationStats(total=3, with_gold=2, answerable=2, unanswerable=1)
    stats.by_venue = {"okx": 1, "bybit": 2}
    out = text(render.validation_stats(stats))
    assert "bybit=2  okx=1" in out and "2 / 1" in out and "with gold chunks" in out
    report = ValidationReport(errors=["bad [x]"], warnings=["odd"])
    messages = render.validation_messages(report)
    assert "warn" in messages[0] and "error" in messages[1]
    assert "bad [x]" in text(Text("\n").join(messages))


def test_agreement_variants_mark_the_sensitivity_readings():
    from crossvenue_rag.dataset.agreement import AgreementVariant

    report = AgreementReport(10, 2, 7, 1, 0.9, 0.6, 0.75, 3)
    variants = [
        AgreementVariant("corrected", "all", report),
        AgreementVariant("first pass", "unexposed", report),
    ]
    out = text(render.agreement_variants_table(variants, 1, 2))
    assert "kappa by reading" in out and "first pass (sensitivity)" in out
    assert "corrected (sensitivity)" not in out


def test_kappa_intervals_show_bounds_and_undefined_resamples():
    from crossvenue_rag.dataset.agreement import AgreementVariant
    from crossvenue_rag.stats import Interval

    report = AgreementReport(10, 2, 7, 1, 0.9, 0.6, 0.75, 3)
    variants = [
        AgreementVariant("corrected", "all", report, Interval(0.31, 0.64, 2, 1000)),
        AgreementVariant("first pass", "all", report, Interval(None, None, 1000, 1000)),
        AgreementVariant("corrected", "unexposed", report),
    ]
    out = text(render.kappa_intervals_table(variants, 1, 2))
    assert "[0.310, 0.640]" in out and "2 of 1,000" in out and "1,000 of 1,000" in out
    assert "unexposed" not in out


def test_direction_and_source_tables_name_the_round_that_said_relevant():
    from crossvenue_rag.dataset.agreement import SourceChunks, VerdictTable

    rows = [("comparison", VerdictTable(1, 4, 37, 0)), ("fact", VerdictTable(0, 3, 2, 0))]
    out = text(render.disagreement_direction_table(VerdictTable(1, 7, 39, 0), rows, 1, 2))
    assert "relevant in round 1 only" in out and "relevant in round 2 only" in out
    assert out.index("comparison") < out.index("fact") < out.index("total")
    assert "46" in out and "39" in out
    sources = text(render.source_chunks_table(SourceChunks(VerdictTable(15, 3, 0, 0), 17), 1, 2))
    assert "source chunks" in sources and "not judged in both rounds" in sources
    assert "17" in sources


def test_composition_marks_categories_short_of_target():
    from collections import Counter

    from crossvenue_rag.dataset.selection import TARGETS, CompositionReport

    accepted = Counter(TARGETS)
    accepted["noisy"] = 1
    out = text(render.composition(CompositionReport(accepted, {}, 3, [])))
    assert "short" in out and "ok" in out and "answerable comparison" in out
    full = text(render.composition(CompositionReport(Counter(TARGETS), {}, 50, [])))
    assert "short" not in full
    reasons = text(render.rejection_reasons(Counter({"duplicate": 2, "vague": 5})))
    assert reasons.index("vague") < reasons.index("duplicate") and "7" in reasons


def test_power_lines_list_reachable_deltas_in_steps_of_five():
    point = {"reachable": True, "power_holm_first_step": 0.5, "power_unadjusted": 0.7}
    study = {
        "family": [["a", "b"], ["a", "c"]],
        "metric": "recall@10",
        "questions": 40,
        "category": "comparison",
        "fractional_values": 3,
        "values": 80,
        "discordant_pairs": 5,
        "question_pairs": 40,
        "psi": 0.125,
        "mean_magnitude": 0.2,
        "ceiling": 0.3,
        "points": [
            {**point, "delta": 0.05},
            {**point, "delta": 0.07},
            {**point, "delta": 0.10, "reachable": False},
        ],
        "delta_80_holm_first_step": 0.2,
        "delta_80_unadjusted": 0.15,
    }
    lines = render.power_lines(power_study_of(**study))
    assert len(lines) == 3
    assert "family of 2" in lines[0] and "psi 0.12" in lines[0]
    assert lines[1].startswith("  delta 0.05: power 0.50 at alpha/2")
    assert lines[2] == "80% power from delta 0.20 (alpha/2), 0.15 (alpha)"


def test_power_lines_format_every_number_by_style_rule_11():
    from tests.conftest import output_defects

    study = {
        "family": [["a", "b"]] * 3,
        "metric": "recall@10",
        "questions": 1200,
        "category": "comparison",
        "fractional_values": 2500,
        "values": 3600,
        "discordant_pairs": 1100,
        "question_pairs": 3600,
        "psi": 0.30555,
        "mean_magnitude": 0.41666,
        "ceiling": 0.12731,
        "points": [
            {"delta": 0.1, "reachable": True, "power_holm_first_step": 0.6, "power_unadjusted": 1}
        ],
        "delta_80_holm_first_step": None,
        "delta_80_unadjusted": 0.1,
    }
    lines = render.power_lines(power_study_of(**study))
    assert lines[0] == (
        "recall@10 on 1,200 comparison questions, family of 3: 2,500 of 3,600 per-question "
        "values fractional; 1,100 of 3,600 question pairs differ (psi 0.31), mean |difference| "
        "0.417, largest reachable delta 0.13"
    )
    assert lines[1] == "  delta 0.10: power 0.60 at alpha/3, 1.00 at alpha"
    assert lines[2] == "80% power from delta not reached (alpha/3), 0.10 (alpha)"
    assert output_defects("\n".join(lines)) == []


def test_power_lines_show_every_fifth_reachable_delta_and_the_80_percent_deltas():
    points = [
        {"delta": d, "reachable": d < 0.2, "power_holm_first_step": 0.5 if d < 0.2 else None,
         "power_unadjusted": 0.9 if d < 0.2 else None}
        for d in (0.05, 0.06, 0.1, 0.2)
    ]  # fmt: skip
    study = {"metric": "recall@10", "questions": 30, "category": "comparison",
             "family": [["a", "b"], ["a", "c"]], "fractional_values": 4, "values": 90,
             "discordant_pairs": 12, "question_pairs": 60, "psi": 0.2, "mean_magnitude": 0.75,
             "ceiling": 0.15, "points": points, "delta_80_holm_first_step": None,
             "delta_80_unadjusted": 0.05}  # fmt: skip
    lines = render.power_lines(power_study_of(**study))
    assert lines[0] == (
        "recall@10 on 30 comparison questions, family of 2: 4 of 90 per-question values "
        "fractional; 12 of 60 question pairs differ (psi 0.20), mean |difference| 0.750, "
        "largest reachable delta 0.15"
    )
    assert lines[1:-1] == [
        "  delta 0.05: power 0.50 at alpha/2, 0.90 at alpha",
        "  delta 0.10: power 0.50 at alpha/2, 0.90 at alpha",
    ]
    assert lines[-1] == "80% power from delta not reached (alpha/2), 0.05 (alpha)"


BRACKETED = ("[/x]", "[bold]")


def test_term_counts_print_a_bracketed_term_and_venue_as_typed():
    for typed in BRACKETED:
        out = text(render.term_counts(typed, TermCount(1, {typed: 1}), typed))
        assert f'"{typed}" in {typed}' in out and out.count(typed) == 3


def test_the_histogram_of_an_empty_corpus_shows_every_bucket_at_zero():
    from crossvenue_rag.corpus.ingest import LENGTH_BINS, corpus_stats

    stats = corpus_stats([], {})
    out = text(render.chunk_length_histogram(stats))
    assert out.count(" 0 ") == len(LENGTH_BINS) and "\u2588" not in out


def test_table_cells_and_titles_print_brackets_as_they_are():
    from crossvenue_rag.dataset.validate import ValidationStats

    counts = TermCount(total=1, by_venue={"[/v]": 1})
    out = text(render.term_counts("[/x]", counts, "[bold]b"))
    assert 'chunks containing "[/x]" in [bold]b' in " ".join(out.split()) and "[/v]" in out
    stats = ValidationStats(total=1, by_category={"[bold]c": 1}, by_venue={"[/x]": 1})
    table = text(render.validation_stats(stats))
    assert "[bold]c=1" in table and "[/x]=1" in table
