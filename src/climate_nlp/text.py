"""
Text preparation for SEC fund filings.

A fund shareholder report (form N-CSR) is HTML that mixes narrative prose, mostly
the Management Discussion of Fund Performance, with long financial tables (the
schedule of holdings, the statements, returns by share class). Only the prose
carries what the climate measure needs.

This strips the HTML to plain text, then keeps sentences that read like prose and
drops table rows. Table rows are short and dense with numbers, tickers and
punctuation, so three density filters remove them.
"""
from __future__ import annotations

import re

_SCRIPT_STYLE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
_TAG = re.compile(r"(?s)<[^>]+>")
_MULTISPACE = re.compile(r"[ \t]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUMERIC_TOKEN = re.compile(r"[\d.,()$%\-]+")

# A sentence shorter than this many words is almost never real prose.
_MIN_WORDS = 8
# Above these thresholds the line looks like a table row rather than a sentence.
_MAX_DIGIT_SHARE = 0.10
_MAX_NUMERIC_TOKEN_SHARE = 0.20


def html_to_text(html: str) -> str:
    """Convert raw filing HTML into plain text.

    Scripts and styles are removed first, then every remaining tag, then the
    most common HTML entities are decoded and runs of whitespace collapsed.
    """
    text = _SCRIPT_STYLE.sub(" ", html)
    text = _TAG.sub(" ", text)
    text = (
        text.replace("&#160;", " ")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    return _MULTISPACE.sub(" ", text)


def extract_prose(text: str) -> list[str]:
    """Split text into sentences and keep only the narrative ones.

    Returns the list of sentences that pass all three table filters. The input
    can be either raw filing text or the output of :func:`html_to_text`.
    """
    flat = re.sub(r"\s+", " ", text)
    sentences = _SENTENCE_SPLIT.split(flat)

    kept: list[str] = []
    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) < _MIN_WORDS:
            continue
        digit_share = sum(c.isdigit() for c in sentence) / max(len(sentence), 1)
        if digit_share > _MAX_DIGIT_SHARE:
            continue
        numeric_tokens = sum(bool(_NUMERIC_TOKEN.fullmatch(t)) for t in tokens)
        if numeric_tokens / len(tokens) > _MAX_NUMERIC_TOKEN_SHARE:
            continue
        kept.append(sentence)
    return kept
