"""
climate_nlp
===========

A text based measure of how much a US mutual fund discusses climate change in
its shareholder reports. The method follows Sautner, van Lent, Vilkov and Zhang
(2023) and is adapted here from earnings call transcripts to SEC fund filings.

Typical use::

    from climate_nlp import load_lexicon, score_text

    lexicon = load_lexicon()
    score = score_text(open("filing.html").read(), lexicon)
    print(score.exposure, score.risk, score.sentiment)
"""
from __future__ import annotations

from .lexicon import Lexicon, load_lexicon, stem
from .measure import ClimateScore, score_sentences
from .text import extract_prose, html_to_text
from .funds import score_by_fund, name_pattern

__all__ = [
    "Lexicon",
    "load_lexicon",
    "stem",
    "ClimateScore",
    "score_sentences",
    "score_text",
    "extract_prose",
    "html_to_text",
    "score_by_fund",
    "name_pattern",
]


def score_text(html: str, lexicon: Lexicon, scale: float = 1000.0) -> ClimateScore:
    """Score a full document given as raw text or HTML.

    Convenience wrapper that runs HTML stripping, prose extraction and the
    measure in one call.
    """
    return score_sentences(extract_prose(html_to_text(html)), lexicon, scale)
