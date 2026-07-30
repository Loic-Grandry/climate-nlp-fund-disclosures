"""
The climate exposure measure.

The measure follows Sautner, van Lent, Vilkov and Zhang (2023). It reads a
document as a bag of bigrams and measures what share of them are about climate. A
bigram counts as climate related when it belongs to the climate lexicon. Dividing
that count by the total number of bigrams gives a frequency that is comparable
across documents of very different lengths.

Four quantities come out of a single pass over the text.

Exposure
    How much the document talks about climate at all. This is the count of
    climate bigrams divided by the total bigram count. Three topical variants
    split it into opportunity, regulatory and physical climate exposure.

Risk
    How much of the climate discussion is framed in terms of risk or
    uncertainty. A climate bigram contributes only when its sentence also
    contains a risk word.

Sentiment
    Whether the climate discussion carries a positive or negative tone. A
    climate bigram contributes with a positive sign when its sentence contains a
    positive tone word and a negative sign when it contains a negative one.

Every quantity is a frequency, so it is multiplied by a fixed scale (1000 by
default) to give readable numbers rather than tiny decimals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .lexicon import Lexicon, stem

_WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class ClimateScore:
    """Result of scoring one document.

    All exposure, risk and sentiment fields are frequencies per ``scale``
    bigrams. ``n_bigrams`` and ``n_climate`` are raw counts kept for auditing.
    """

    n_bigrams: int
    n_climate: int
    exposure: float
    exposure_opportunity: float
    exposure_regulatory: float
    exposure_physical: float
    risk: float
    sentiment_positive: float
    sentiment_negative: float
    sentiment: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _tokenize(sentence: str) -> list[str]:
    """Lowercase, keep alphabetic tokens only, then Porter stem each one."""
    return [stem(w) for w in _WORD.findall(sentence.lower())]


def score_sentences(
    sentences: list[str],
    lexicon: Lexicon,
    scale: float = 1000.0,
) -> ClimateScore:
    """Compute the climate score from a list of narrative sentences.

    Bigrams are formed inside each sentence and never across a sentence
    boundary, which keeps unrelated ideas from being glued together. Risk and
    tone are evaluated at the sentence level, so a climate bigram inherits the
    framing of the sentence it sits in.
    """
    total_bigrams = 0
    climate_hits = 0
    opportunity_hits = 0
    regulatory_hits = 0
    physical_hits = 0
    risk_hits = 0
    positive_hits = 0
    negative_hits = 0

    for sentence in sentences:
        tokens = _tokenize(sentence)
        if len(tokens) < 2:
            continue

        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        total_bigrams += len(bigrams)

        words = set(tokens)
        has_risk = not words.isdisjoint(lexicon.risk)
        has_positive = not words.isdisjoint(lexicon.tone_positive)
        has_negative = not words.isdisjoint(lexicon.tone_negative)

        for bigram in bigrams:
            if bigram in lexicon.climate:
                climate_hits += 1
                if has_risk:
                    risk_hits += 1
                if has_positive:
                    positive_hits += 1
                if has_negative:
                    negative_hits += 1
            if bigram in lexicon.opportunity:
                opportunity_hits += 1
            if bigram in lexicon.regulatory:
                regulatory_hits += 1
            if bigram in lexicon.physical:
                physical_hits += 1

    denominator = max(total_bigrams, 1)
    factor = scale / denominator

    return ClimateScore(
        n_bigrams=total_bigrams,
        n_climate=climate_hits,
        exposure=factor * climate_hits,
        exposure_opportunity=factor * opportunity_hits,
        exposure_regulatory=factor * regulatory_hits,
        exposure_physical=factor * physical_hits,
        risk=factor * risk_hits,
        sentiment_positive=factor * positive_hits,
        sentiment_negative=factor * negative_hits,
        sentiment=factor * (positive_hits - negative_hits),
    )
