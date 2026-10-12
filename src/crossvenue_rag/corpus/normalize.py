"""Normalizes vendor documentation to plain markdown; in fenced code only blank-line runs collapse.

Each publishing stack adds non-content syntax (MDX imports, JSX components,
admonitions, GitBook `{% hint %}` blocks, HTML entities) that would become tokens
matching across every venue at once, so it is stripped before indexing. Every
transformation but the last runs only on the prose between fences, because an endpoint
signature with a regex-mangled parameter is worse than no chunk.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>```+|~~~+)(?P<info>[^\n]*)$", re.MULTILINE)


def has_code_fence(text: str) -> bool:
    """Whether `text` has a fence line as `split_code_blocks` reads one.

    A fence line is ``` or ~~~ at a line start, after any spaces or tabs.
    """
    return _FENCE_RE.search(text) is not None


@dataclass(slots=True)
class Segment:
    """A run of the document that is either all prose or all fenced code."""

    text: str
    is_code: bool


def split_code_blocks(text: str) -> list[Segment]:
    """Markdown split into alternating prose / fenced-code segments.

    A closing fence must be at least as long as its opener and carry no info
    string, which is how nested fences inside a ```` ```` block stay intact.
    An unterminated fence runs to the end of the document: treating the tail as
    code is the conservative choice, since the alternative mangles it.
    """
    segments: list[Segment] = []
    cursor = 0
    position = 0
    while True:
        opener = _FENCE_RE.search(text, position)
        if not opener:
            break
        fence = opener.group("fence")
        close_re = re.compile(
            rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$", re.MULTILINE
        )
        closer = close_re.search(text, opener.end())
        end = closer.end() if closer else len(text)

        if opener.start() > cursor:
            segments.append(Segment(text[cursor : opener.start()], is_code=False))
        segments.append(Segment(text[opener.start() : end], is_code=True))
        cursor = position = end
        if not closer:
            break

    if cursor < len(text):
        segments.append(Segment(text[cursor:], is_code=False))
    return segments


_FRONTMATTER_RE = re.compile(r"\A---\r?\n(?P<body>.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)


def strip_frontmatter(text: str) -> str:
    """`text` without its YAML frontmatter block, if it opens with one."""
    match = _FRONTMATTER_RE.match(text)
    return text[match.end() :] if match else text


_MDX_IMPORT_RE = re.compile(r"^\s*(?:import|export)\s+[^\n]*?(?:;|\n|$)", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_JSX_COMMENT_RE = re.compile(r"\{\s*/\*.*?\*/\s*\}", re.DOTALL)

# Docusaurus admonitions: ::: note / :::tip Title ... :::
_ADMONITION_OPEN_RE = re.compile(
    r"^:::+\s*(?P<kind>note|tip|info|caution|warning|danger|important)?[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)
_ADMONITION_CLOSE_RE = re.compile(r"^:::+\s*$", re.MULTILINE)

# Docsify heading directives, such as `### fetchFundingRate{docsify-ignore}` all over
# the CCXT wiki. Left in, they attach to the one token that identifies the method.
_DOCSIFY_DIRECTIVE_RE = re.compile(r"\{docsify-[a-z-]+\}")

# GitBook block syntax: {% hint style="info" %} ... {% endhint %}, {% tabs %}, etc.
_GITBOOK_TAG_RE = re.compile(r"\{%\s*(?P<tag>[a-zA-Z-]+)[^%]*%\}")
# GitBook cross-reference cards carry no prose, only a link target.
_GITBOOK_REF_BLOCK_RE = re.compile(
    r"\{%\s*(?:content-ref|embed|file|swagger)[^%]*%\}.*?\{%\s*end[\w-]+\s*%\}", re.DOTALL
)

# JSX/HTML tags. Container tags are dropped but their children survive; this keeps
# the text inside <Tabs>/<TabItem>/<details> while discarding the wrapper.
_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9.:-]*(?:\s[^<>]*?)?/?>", re.DOTALL)
# A self-closing JSX component with props and no children carries no content.
_SELF_CLOSING_COMPONENT_RE = re.compile(r"<[A-Z][A-Za-z0-9.]*(?:\s[^<>]*?)?/>", re.DOTALL)

_BLANK_RUN_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)

# Inline code spans are in prose segments but must survive tag stripping intact:
# `GET /v5/order/create?symbol=<SYMBOL>` would otherwise lose its placeholder.
_INLINE_CODE_RE = re.compile(r"(?P<ticks>`+)(?P<body>.+?)(?P=ticks)", re.DOTALL)
_MASK = "\x00CVRAG{}\x00"


def _mask_inline_code(text: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def take(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return _MASK.format(len(spans) - 1)

    return _INLINE_CODE_RE.sub(take, text), spans


def _unmask_inline_code(text: str, spans: list[str]) -> str:
    for index, span in enumerate(spans):
        text = text.replace(_MASK.format(index), span)
    return text


def _clean_prose(text: str) -> str:
    text, inline_code = _mask_inline_code(text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _JSX_COMMENT_RE.sub("", text)
    text = _MDX_IMPORT_RE.sub("", text)

    text = _GITBOOK_REF_BLOCK_RE.sub("", text)
    text = _GITBOOK_TAG_RE.sub("", text)
    text = _DOCSIFY_DIRECTIVE_RE.sub("", text)

    # An admonition fence loses its delimiters; its body stays as a paragraph.
    text = _ADMONITION_CLOSE_RE.sub("", text)
    text = _ADMONITION_OPEN_RE.sub(
        lambda m: f"**{m.group('kind').capitalize()}:**" if m.group("kind") else "",
        text,
    )

    text = _SELF_CLOSING_COMPONENT_RE.sub("", text)
    text = _TAG_RE.sub("", text)

    # GitBook emits `&#x20;` for significant spaces and escapes `&`, `<`, `>`.
    text = html.unescape(text)

    text = _unmask_inline_code(text, inline_code)
    text = _TRAILING_WS_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text


_NAV_LINE_PATTERNS = [
    # Page furniture that survives as literal text. Anchored to whole lines so
    # prose that merely starts with "Next," is not mistaken for a nav control.
    re.compile(r"^\s*\[?(previous|next)\]?\s*(\(.*\))?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(edit this page|last updated)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*(?:\*|-)\s*$"),  # empty list bullets left by stripped components
    # Markdown table rules (`|---|:---:|---|`). They carry no information once
    # the table is plain text, and the wide ones break the chunk budget. One rule
    # in Binance's fix-api.md is a single 951-token run of dashes with no spaces,
    # which no word-boundary splitter can break up.
    re.compile(r"^\s*\|?[\s|:-]*\|[\s|:-]*$"),
]


def _drop_nav_lines(text: str) -> str:
    kept = [
        line
        for line in text.split("\n")
        if not any(pattern.match(line) for pattern in _NAV_LINE_PATTERNS)
    ]
    return "\n".join(kept)


def normalize_markdown(text: str) -> str:
    """A page without its publishing-stack markup; in code blocks, only blank-line runs collapse.

    The final pass collapses runs of blank lines inside fenced code too, a defect
    deferred in ADR 0012.
    """
    body = strip_frontmatter(text)
    out: list[str] = []
    for segment in split_code_blocks(body):
        if segment.is_code:
            out.append(segment.text)
        else:
            out.append(_drop_nav_lines(_clean_prose(segment.text)))
    return _BLANK_RUN_RE.sub("\n\n", "".join(out)).strip() + "\n"
