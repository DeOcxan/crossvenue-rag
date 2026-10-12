"""What evals/domain-research.md claims."""

from __future__ import annotations

from ... import formatting
from ..check import Doc
from ..compute import corpus, corpus_pancakeswap, corpus_sheet
from ..sources import Sources
from . import comparisons
from .reasons import DOCS_QUOTE, GITHUB_STATUS, LIST_NUMBERING

TITLE = "a verbatim GitHub issue title"
THEMES = "a theme or trap number of this document"
LISTED = "the items the cell or sentence lists"
UNFITTED = "a TF-IDF cosine whose vectorizer's fitting set is not recorded; no saved file holds it"
# The sources in the order theme 5 lists their "rate limit" chunk counts.
RATE_LIMIT_ORDER = (
    "bingx",
    "bybit",
    "binance",
    "ccxt",
    "dydx",
    "hyperliquid",
    "uniswap",
    "pancakeswap",
)
# The HMAC-SHA256 spellings of theme 2, and the venues whose chunks use each, among the three
# venues the sentence compares.
HMAC_SPELLINGS = ("HMAC-SHA-256", "HMAC-SHA256", "HMAC_SHA256", "HMAC SHA256")
HMAC_VENUES = ("binance", "bingx", "bybit")
HMAC_USERS = {
    "HMAC-SHA-256": ["binance"],
    "HMAC-SHA256": ["bingx", "bybit"],
    "HMAC_SHA256": ["bingx", "bybit"],
    "HMAC SHA256": ["bingx"],
}
# The mainnet pages of trap T8, in the document's order.
T8_PAGES = ("rest-api.md", "web-socket-api.md", "errors.md")

# The numbers inside the issue titles the document quotes.
TITLE_NUMBERS = (
    "was 1000ms ahead",
    "return 400 Bad Request",
    "every 5 seconds triggers",
    "(ErrCode: 10001)",
    "returns only 3 values",
    "amongst 3 Liquidity Pools",
    "affects 62 of 226 markets (27%)",
    "in two different APIs",
    "expected 11000, got 10999",
    '"Which of the three fee-tier pools is the real price?"',
)
# The values the traps quote from the vendors' documentation.
QUOTES = (
    '"0.01% \u2026 1%"',
    "Uniswap v3: 0.01 / 0.05 / 0.30 / 1.00%",
    "0.01 / 0.05 / 0.25 / 1%",
    "PancakeSwap on Solana: 18 tiers",
    "V2 base fee 0.30% vs 0.25%",
    "PancakeSwap on Solana has 18 tiers",
    "set slippage \u226512% for tax tokens and hold 30% more tokens",
    '"recvWindow \u2026 request validity window \u2026 default 5000"',
    'Binance: max 60000. Bybit FIX: default 5000, max 60000. BingX: "Default is 5000; '
    'maximum is 5000"',
    "Binance: 429, then 418 auto-ban scaling from 2 minutes to 3 days, per-IP",
    'Bybit: 403 "access too frequent", a wait of at least 10 minutes, 600 req / 5 s per IP '
    "plus 10 rps "
    "read / 5 rps write per UID",
    "BingX: 429, then 418,",
    "Hyperliquid: 1200 weight/min per IP and an address-based limit of 1 request per 1 USDC traded",
    "means three different things across three venues",
    "Binance: client messages capped at 5/s (ping frames count), connection dies at 24 h, 1024 "
    "streams",
    "Hyperliquid: server closes if it hasn't sent anything in 60 s",
    "dYdX: protocol-level control frames every 30 s, pong within 10 s",
    "Binance klines: default 500, max 1000",
    "limit [1,1000]",
    "hard 20 levels per side",
    '"8 hour funding rate"',
    'Funding interval "8 hours"',
    "funding is charged hourly at 1/8, capped 4%/hour",
    "Hitting 429 /",
)
# The counts of things a cell or sentence lists after them.
LISTS = (
    "(three key types:",
    "Four venues, four incompatible data models",
    "with five mutually incompatible keepalive protocols",
    "has five different correct answers",
    "Three incompatible pagination models",
    "binance has two (",
    "the two order lifetimes",
    "all five API venues",
    "are two chunks, told apart",
    "has two mutually exclusive correct answers",
    "naming one of the two would test it",
    "explicitly two-venue questions",
    "They cover three information needs",
    "questions needing two chunks",
    "across the three sub-shapes",
)
# Theme and trap numbers the text refers to.
THEME_NUMBERS = (
    "Same reason as 17",
    "themes 1\u201316 and none of 17\u201318",
    "fact: themes 1, 2, 3, 5, 8, 9, 10, 14, 15, 16,",
    "(themes 7 and 14 on Binance",
)


def domain_research(doc: Doc, src: Sources) -> None:
    """States the claims of evals/domain-research.md."""
    _github(doc, src)
    _themes(doc, src)
    _traps(doc, src)
    _rate_limits(doc, src)
    doc.definitions(THEME_NUMBERS, reason=THEMES)


def _github(doc: Doc, src: Sources) -> None:
    doc.definitions(TITLE_NUMBERS, reason=TITLE)
    doc.definition("(repo GET returns 404, search returns 422)", reason=GITHUB_STATUS)
    doc.definition("1006/1008 disconnects", reason="the WebSocket close codes")
    doc.claim(
        "and has no chunk with `Transaction too old`",
        holds=corpus.sources_with_term(src, "Transaction too old") == []
        and corpus.sources_with_term(src, "PancakeRouter: EXPIRED") == ["pancakeswap"]
        and corpus.sources_with_term(src, "PancakeRouter: INSUFFICIENT\\_OUTPUT\\_AMOUNT")
        == ["pancakeswap"],
        what="no chunk holds `Transaction too old`, and PancakeSwap's chunks hold the two "
        "PancakeRouter strings (the second with its underscores escaped, as the page has it)",
    )


def _themes(doc: Doc, src: Sources) -> None:
    doc.claim(
        "Binance spells it `HMAC-SHA-256`, Bybit and BingX `HMAC-SHA256` and `HMAC_SHA256`, "
        "and BingX alone `HMAC SHA256`",
        holds=corpus.venues_with_spellings(src, HMAC_SPELLINGS, HMAC_VENUES) == HMAC_USERS,
        what="each HMAC spelling occurs in the chunks of exactly the venues named",
    )
    doc.definitions_matching(r"(?m)^### \d+\. ", reason=LIST_NUMBERING)
    doc.definitions(LISTS, reason=LISTED)
    doc.check("| the {} venues and ccxt (`sandbox`)", formatting.count(corpus.venue_count(src)))
    fees = comparisons.fee_tier_texts()
    doc.claim(
        "Uniswap's medium tier is {}%, PancakeSwap's is {}%",
        fees["uniswap"],
        fees["pancakeswap"],
        holds=corpus_pancakeswap.fee_tiers_stated(src),
        what=f"Uniswap's documentation states a {fees['uniswap']}% tier and PancakeSwap's a "
        f"{fees['pancakeswap']}% tier",
    )


def _traps(doc: Doc, src: Sources) -> None:
    doc.definitions(QUOTES, reason=DOCS_QUOTE)
    copies = [corpus.testnet_copies(src, page) for page in T8_PAGES]
    doc.check(
        "of the mainnet chunks of `rest-api.md` ({}), `web-socket-api.md` ({}) and "
        "`errors.md` ({}), {}, {} and {} have a testnet occurrence",
        *[formatting.count(c.chunks) for c in copies],
        *[formatting.count(c.normalized) for c in copies],
    )
    doc.check(
        "Byte-identical pairs ({}) are one chunk each",
        formatting.count(corpus_sheet.testnet_mirror_groups(src)),
    )
    doc.unsaved("TF-IDF cosine {}, with the vectorizer parameters", reason=UNFITTED)
    doc.unsaved("| {} of the", reason=UNFITTED)
    doc.check(
        "of the {} ccxt chunks have a raw-venue chunk",
        formatting.count(len(corpus.venue_chunks(src, "ccxt"))),
    )
    doc.definition("at TF-IDF cosine \u2265 0.3 (vectorizer", reason="the threshold of the count")


def _rate_limits(doc: Doc, src: Sources) -> None:
    doc.check(
        'Chunks containing "rate limit" (`crossvenue count-term`): bingx {}, bybit {}, binance {}, '
        "ccxt {}, dydx {}, hyperliquid {}, uniswap {}, pancakeswap {}",
        *[formatting.count(corpus.term_chunks(src, "rate limit", v)) for v in RATE_LIMIT_ORDER],
    )
