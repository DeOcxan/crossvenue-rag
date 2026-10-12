"""Tests for `crossvenue_rag.corpus.normalize`."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.corpus.normalize import normalize_markdown, strip_frontmatter

FIXTURES = Path(__file__).parent.parent / "fixtures"


DOCUSAURUS = (FIXTURES / "sample_docusaurus.md").read_text(encoding="utf-8")


GITBOOK = (FIXTURES / "sample_gitbook.md").read_text(encoding="utf-8")


MDX = (FIXTURES / "sample_mdx.mdx").read_text(encoding="utf-8")


def test_a_frontmatter_block_is_stripped_and_a_page_without_one_is_kept():
    body = strip_frontmatter(DOCUSAURUS)
    assert not body.startswith("---")
    assert "sidebar_label" not in body
    assert strip_frontmatter("# Funding\n\ntext") == "# Funding\n\ntext"


def test_mdx_markup_is_removed_but_children_survive():
    """Wrappers like <Tabs>/<details> go; the text inside them is the content."""
    out = normalize_markdown(MDX)
    assert "import" not in out and "export const meta" not in out
    assert "<FeeTable" not in out
    assert "Uniswap v3 supports four fee tiers" in out
    assert "Stable pairs need a tier" in out  # <details> children kept

    docusaurus = normalize_markdown(DOCUSAURUS)
    assert "<Tabs>" not in docusaurus and "<TabItem" not in docusaurus
    assert "POST /v5/order/create HTTP/1.1" in docusaurus


def test_docusaurus_admonitions_keep_their_body_and_tables_survive():
    out = normalize_markdown(DOCUSAURUS)
    assert ":::" not in out
    assert "**Info:**" in out
    assert "Only unified accounts can use" in out
    assert "Rate limit: 10 requests per second per UID." in out
    # A regex that removes table rules would silently empty every parameter table.
    assert "| category | true | string |" in out


def test_gitbook_blocks_are_removed_but_prose_survives():
    out = normalize_markdown(GITBOOK)
    assert "{%" not in out and "%}" not in out
    assert "Funding is paid every hour, not every eight hours." in out
    # A content-ref card is a link with no prose; it should not become a chunk.
    assert "[perpetuals.md]" not in out
    assert "&#x20;" not in out and "clamped to <= 4%" in out


def test_code_blocks_are_never_modified():
    """The one invariant everything else is built around."""
    out = normalize_markdown(DOCUSAURUS)
    assert '"timeInForce": "PostOnly"' in out
    assert "X-BAPI-SIGN: XXXXX" in out
    assert "function createPool(address tokenA, address tokenB, uint24 fee)" in normalize_markdown(
        MDX
    )


def test_inline_code_placeholders_survive_tag_stripping():
    """`<SYMBOL>` looks exactly like an HTML tag to a regex."""
    assert "`POST /v5/order/create?symbol=<SYMBOL>`" in normalize_markdown(DOCUSAURUS)


def test_nav_lines_dropped_but_prose_starting_with_next_is_kept():
    out = normalize_markdown(GITBOOK)
    assert "\nPrevious" not in out and not out.rstrip().endswith("Next")
    assert normalize_markdown("Next, call the order endpoint.\n").strip() == (
        "Next, call the order endpoint."
    )


def test_docsify_heading_directives_are_removed():
    """`### fetchFundingRate{docsify-ignore}`: the CCXT wiki attaches the marker to
    the one token that identifies the method."""
    out = normalize_markdown("### fetchFundingRate{docsify-ignore}\n\nfetch the rate\n")
    assert "docsify" not in out
    assert "### fetchFundingRate" in out


def test_a_code_fence_is_a_fence_line_of_backticks_or_tildes():
    from crossvenue_rag.corpus.normalize import has_code_fence

    assert has_code_fence("Example:\n```python\nx = 1\n```\n")
    assert has_code_fence("Example:\n  ~~~\nx = 1\n  ~~~\n")
    assert not has_code_fence("Inline ``` backticks in prose are no fence.")
    assert not has_code_fence("No code here.")
