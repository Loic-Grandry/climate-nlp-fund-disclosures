"""
Splitting a trust filing into individual funds.

A single N-CSR is often filed by a trust that holds many funds, called series
in SEC language. The narrative for each fund appears one after another in the
same document, each introduced by the fund name. To score funds separately we
locate every fund name in the text and cut the document at those anchor points,
so each slice holds the prose that belongs to one fund.

Fund names in the body rarely match the registered name character for character.
"U.S." shows up as "US", ampersands turn into "and", and stray spaces or
punctuation slip in. The matcher below is tolerant of those differences while
still requiring the words to appear in order.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .lexicon import Lexicon
from .measure import ClimateScore, score_sentences
from .text import extract_prose, html_to_text


def name_pattern(name: str) -> re.Pattern | None:
    """Build a tolerant regular expression for a registered fund name."""
    normalized = re.sub(r"\bU\.?\s?S\.?\b", "US", name, flags=re.I).replace("&", "and")
    tokens = re.findall(r"[A-Za-z0-9]+", normalized)
    if not tokens:
        return None
    parts = [
        r"u\.?\s?s\.?" if t.lower() == "us" else re.escape(t) for t in tokens
    ]
    # Allow one to four non alphanumeric characters between consecutive words.
    joined = r"[^A-Za-z0-9]{1,4}".join(parts)
    return re.compile(rf"\b{joined}\b", re.I)


def score_by_fund(
    html: str,
    series: dict[str, str],
    lexicon: Lexicon,
    scale: float = 1000.0,
) -> dict[str, object]:
    """Score every fund covered by a trust filing.

    Parameters
    ----------
    html:
        Raw filing text or HTML.
    series:
        Mapping from a series identifier to the fund's registered name.
    lexicon:
        Loaded climate and tone dictionaries.

    Returns
    -------
    A dictionary keyed by series identifier. Each value is a tuple of the
    fund's :class:`ClimateScore` (or ``None`` when no prose could be attributed
    to it) and the number of times its name was found. The special key
    ``"_trust"`` holds the score of the whole document, which is the right
    fallback when the filing covers a single fund.
    """
    raw = re.sub(r"\s+", " ", html_to_text(html))
    trust_score = score_sentences(extract_prose(raw), lexicon, scale)

    anchors: list[tuple[int, str]] = []
    for series_id, name in series.items():
        pattern = name_pattern(name)
        if pattern is None:
            continue
        for match in pattern.finditer(raw):
            anchors.append((match.start(), series_id))
    anchors.sort()

    results: dict[str, object] = {"_trust": trust_score}

    if not anchors:
        # No name found, so the document is treated as a single fund.
        for series_id in series:
            results[series_id] = (trust_score, 0)
        return results

    slices: dict[str, str] = defaultdict(str)
    for i, (position, series_id) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(raw)
        slices[series_id] += " " + raw[position:end]

    for series_id in series:
        text = slices.get(series_id, "")
        n_anchors = sum(1 for _, s in anchors if s == series_id)
        if text.strip():
            score = score_sentences(extract_prose(text), lexicon, scale)
        else:
            score = None
        results[series_id] = (score, n_anchors)
    return results
