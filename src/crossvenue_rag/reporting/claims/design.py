"""What docs/explanation/design.md claims.

Each function checks one section of the document or a part of one, in document order except
`index_version`.
"""

from __future__ import annotations

from ... import formatting
from ...models import MULTI_VENUE, RETRIEVAL_STEPS
from ..check import Doc
from ..compute import answers, corpus, corpus_sheet, device, repository, retrieval
from ..run_ids import (
    PRE_DEDUP_RUN,
    RETRIEVAL_RUN,
)
from ..sources import Sources
from . import comparisons
from .reasons import (
    CITED,
    COLLAPSED_FENCES,
    COLLAPSED_FENCES_REASON,
    ROOT_DEFINITION,
    SERVE_ADDRESS,
    STATUS_CODE,
)

RECORD_NUMBER = "a record number"
# The sample sizes of the mutation test, set in tests/reporting/test_mutations.py.
MUTATION_SAMPLES = (
    "a sample size of tests/reporting/test_mutations.py, which no document check reads"
)


def design(doc: Doc, src: Sources) -> None:
    """States the claims of docs/explanation/design.md."""
    _no_framework(doc, src)
    _module_size(doc, src)
    _module_layout(doc, src)
    _tests(doc, src)
    _normalization(doc)
    _chunking(doc, src)
    _deduplication(doc, src)
    _provenance(doc, src)
    _structural_metadata(doc, src)
    _retrieval(doc, src)
    _latency(doc)
    _generation(doc, src)
    _interface(doc)
    _errors(doc)
    # The `index_version` section comes before Retrieval in the document and is checked
    # after Error handling: the call order sets the order of the problems and listed numbers.
    _index_version(doc, src)
    _checking(doc)


def _no_framework(doc: Doc, src: Sources) -> None:
    counted = repository.source_lines(src)
    stated = repository.stated_source_lines(doc.collapsed)
    doc.claim(
        "about {} lines under `src/`",
        formatting.count(stated),
        holds=repository.source_lines_agree(stated, counted),
        what="the stated line count is the "
        f"{formatting.count(counted)} lines counted, to the nearest "
        f"{formatting.count(repository.SOURCE_LINES_ROUNDING)}",
    )


def _module_size(doc: Doc, src: Sources) -> None:
    path, lines = repository.largest_module(src)
    doc.claim(
        "bounds at about {} lines of code",
        formatting.count(repository.MODULE_CODE_LINES_BOUND),
        holds=lines <= repository.MODULE_CODE_LINES_BOUND,
        what=f"no module above the bound; {path} has {formatting.count(lines)} lines of code",
    )


def _module_layout(doc: Doc, src: Sources) -> None:
    doc.check("{} subpackages, one per domain", formatting.count(repository.subpackage_count(src)))
    doc.check(
        "in `make lint` and CI with {} contracts",
        formatting.count(repository.import_contract_count(src)),
    )
    doc.definition("at least two layers need", reason=ROOT_DEFINITION)


def _tests(doc: Doc, src: Sources) -> None:
    doc.check(
        "{} files test no single module",
        formatting.count(len(repository.suite_level_test_files(src))),
    )
    doc.check(
        "one end-to-end run over a {}-document fixture corpus",
        formatting.count(repository.fixture_pages(src)),
    )


def _normalization(doc: Doc) -> None:
    doc.definition("from four publishing stacks", reason="the four stacks the sentence lists")
    doc.given(
        "collapses runs of blank lines inside {} of them",
        COLLAPSED_FENCES,
        reason=COLLAPSED_FENCES_REASON,
    )


def _chunking(doc: Doc, src: Sources) -> None:
    doc.definition("becomes a two-word chunk", reason="`## Parameters` is two words")
    budget = formatting.count(corpus.chunk_token_budget(src))
    doc.check('so a "{}-token" chunk is {} tokens', budget, budget)
    prose, code = corpus.over_budget_chunks(src)
    doc.check(
        "Over the {}-token budget are {} prose chunks of {}, plus {} code blocks kept whole",
        budget,
        formatting.count(prose),
        formatting.count(corpus.chunk_count(src)),
        formatting.count(code),
    )
    doc.check(
        "identical text on {} venues can never share", formatting.count(corpus.CROSS_VENUE_VENUES)
    )


def _deduplication(doc: Doc, src: Sources) -> None:
    doc.check(
        "sections are near-identical (difflib ratio {}) and stay {} chunks",
        formatting.metric(corpus.timing_security_ratio(src)),
        formatting.count(corpus.timing_security_chunks(src)),
    )
    dup = corpus.duplicates(src)
    doc.check("- Identical text on {} venues", formatting.count(corpus.CROSS_VENUE_VENUES))
    doc.check("The corpus has none ({} groups)", formatting.count(dup.cross_venue_texts))
    doc.check(
        "The {} chunk occurrences are stored as {} chunks: {} groups merged, {} of them Binance",
        formatting.count(dup.occurrences),
        formatting.count(corpus.chunk_count(src)),
        formatting.count(dup.groups),
        formatting.count(dup.by_venue["binance"]),
    )


def _provenance(doc: Doc, src: Sources) -> None:
    doc.check(
        "makes the {} Binance mirrors cite",
        formatting.count(corpus_sheet.testnet_mirror_groups(src)),
    )


def _structural_metadata(doc: Doc, src: Sources) -> None:
    doc.definition("rather than either of the two", reason="mainnet and testnet")
    chunks = corpus.chunk_count(src)
    doc.check(
        "`doc_type` {}%, `api_version` {}%, `product` {}%, `environment` {}%",
        formatting.percent_figure(corpus.filled_chunks(src, "doc_type"), chunks),
        formatting.percent_figure(corpus.filled_chunks(src, "api_version"), chunks),
        formatting.percent_figure(corpus.filled_chunks(src, "product"), chunks),
        formatting.percent_figure(corpus.filled_chunks(src, "environment"), chunks),
    )
    null = corpus.null_chunks_by_venue(src, "environment")
    doc.check(
        "Null environments per venue: bybit {}, pancakeswap {}, uniswap {}, ccxt {}, dydx {}, "
        "hyperliquid {}, bingx {} and binance {}.",
        *[
            formatting.count(null[v])
            for v in (
                "bybit",
                "pancakeswap",
                "uniswap",
                "ccxt",
                "dydx",
                "hyperliquid",
                "bingx",
                "binance",
            )
        ],
    )
    doc.check("Binance's {} are `demo-mode/`", formatting.count(null["binance"]))


def _retrieval(doc: Doc, src: Sources) -> None:
    methods = formatting.count(retrieval.method_count(src))
    doc.check("{} methods behind one signature", methods)
    doc.check("One signature rather than {} code paths", methods)
    doc.ref("0.620 for hybrid_rerank (`results/{}`", RETRIEVAL_RUN)
    doc.check(
        "On the {} comparison questions of `v0.2.1-gold` its recall@10 is {}, "
        "against {} for dense and {} for hybrid_rerank",
        comparisons.comparison_questions(src),
        *[
            formatting.metric(v)
            for v in retrieval.category_metrics(
                src,
                "comparison",
                ("bm25", "dense", "hybrid_rerank"),
                retrieval.at_top_k(src, "recall"),
            )
        ],
    )
    doc.check("with fewer than {} venues named", formatting.count(MULTI_VENUE))
    doc.definition("cosine similarities are in [\u22121, 1]", reason="the range of a cosine")
    doc.check("The constant `k` ({}, from", formatting.count(src.retrieval.config.retrieval.rrf_k))
    doc.definition("Cormack et al. 2009, [doi:10.1145/1571941.1572114]", reason=CITED)
    doc.definition("cannot leave half-migrated rows", reason="a description of the migration")


def _latency(doc: Doc) -> None:
    doc.check("the sum of the {} measured steps", formatting.count(len(RETRIEVAL_STEPS)))
    doc.definition("Two things the harness does", reason="the two items of the list that follows")


def _generation(doc: Doc, src: Sources) -> None:
    doc.definitions(("ADRs [0009]", "[0010]", "[0011]"), reason=RECORD_NUMBER)
    holes, statuses = comparisons.retried_status_parts()
    doc.check("retries the HTTP statuses " + holes + ", and", *statuses)
    doc.check("at most {}, for a response", formatting.count(answers.max_invalid_retries(src)))
    doc.definition("It allows four outcomes", reason="the four outcomes listed after it")


def _interface(doc: Doc) -> None:
    doc.definition("Two thin entry points", reason="the two entry points listed after it")
    doc.definition("on 127.0.0.1:8000", reason=SERVE_ADDRESS)
    doc.definitions(("is a 422", "is a 502", "is a 503"), reason=STATUS_CODE)


def _errors(doc: Doc) -> None:
    doc.definition("a hint and exit 1", reason="the exit status `cli.errors` sets")


def _index_version(doc: Doc, src: Sources) -> None:
    doc.ref("`index_version` `{}` (`data/index/index_meta.json`)", corpus.index_version(src))
    doc.ref("except `results/{}`, which was made before exact deduplication", PRE_DEDUP_RUN)
    doc.unsaved(
        "changed {} of the",
        reason="one index rebuild compared with the committed retrieval run; no saved run holds it",
    )
    doc.unsaved(
        "from the venue btree index for each of the {} venues at k of {} and {}",
        reason="EXPLAIN of the single-venue dense query on the built index; no saved run holds it",
    )
    doc.check(
        "of the {} top-10 lists of the dense-dependent methods",
        formatting.count(retrieval.dense_dependent_lists(src)),
    )
    _device(doc, src)


def _device(doc: Doc, src: Sources) -> None:
    """The compute device's effect, the CPU run against the MPS run."""
    doc.ref("the CPU run `results/{}` and the MPS run `results/{}`", device.CPU_RUN, device.MPS_RUN)
    lists = device.list_change(src)
    doc.claim(
        "differ in {} of {} retrieval lists, none of them BM25's.",
        formatting.count(lists.differing),
        formatting.count(lists.shared),
        holds="bm25" not in lists.methods,
        what="no BM25 list differs between the devices",
    )
    cpu, mps = src.run(device.CPU_RUN).by_category, src.run(device.MPS_RUN).by_category
    unjudged = "unjudged@10"
    doc.claim(
        "On the {} comparison questions only nDCG and MRR differ, in the fourth decimal, and "
        "dense's unjudged@10, {} against {}.",
        comparisons.comparison_questions(src),
        formatting.metric(cpu[("comparison", "dense")].value(unjudged)),
        formatting.metric(mps[("comparison", "dense")].value(unjudged)),
        holds=device.ranking_only_changes(src, "comparison"),
        what="on the comparison questions only nDCG and MRR change, below 0.001, besides "
        "dense's unjudged@10",
    )
    largest = device.membership_changes(src)[0]
    doc.claim(
        f"most in {largest.category} {largest.method} {largest.metric}, {{}} against {{}}",
        formatting.metric(largest.cpu),
        formatting.metric(largest.mps),
        holds=all(c.category != "comparison" for c in device.membership_changes(src)),
        what="no recall or precision cell of the comparison questions differs",
    )


def _checking(doc: Doc) -> None:
    """Checking the documents against their sources: what `make reproduce` does."""
    doc.ref("(`results/{}` read the working file)", PRE_DEDUP_RUN)
    doc.definition(
        "Numbers written as words (two to twelve, half, twice, tenfold),",
        reason="the number words the checker reads",
    )
    doc.given(
        "it changes {} sampled numbers",
        "500",
        reason=MUTATION_SAMPLES,
    )
    doc.given(
        "`make test` runs a fixed sample of {} of these mutations", "53", reason=MUTATION_SAMPLES
    )
