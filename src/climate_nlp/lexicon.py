"""
Climate lexicon and tone dictionaries.

The measure works on bigrams (pairs of adjacent words). A bigram counts as a
climate term when it appears in the climate lexicon C. The lexicon is stored as
plain English pairs such as "carbon tax" or "renewable energy" and is stemmed
here so that "renewable energies" and "renewable energy" match the same entry.

Three topical sub-lexicons refine the overall measure into opportunity,
regulatory and physical climate exposure. Following the original paper, a bigram
that would fall into more than one topic is dropped from all three, so the
topical measures stay mutually exclusive.

Tone is measured with the Loughran and McDonald (2011) finance word lists, the
standard positive and negative dictionaries for financial text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()

# Package ships its dictionaries next to the source so the library is
# self contained. A user can point DATA_DIR elsewhere if they maintain a
# private copy of the full author lexicon.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """Porter stem of a single lowercase word, cached for speed."""
    return _STEMMER.stem(word)


def _stem_bigrams(pairs: Iterable[str]) -> set[str]:
    """Turn a list of raw two word phrases into a set of stemmed bigrams."""
    out: set[str] = set()
    for phrase in pairs:
        words = phrase.split()
        if len(words) == 2:
            out.add(f"{stem(words[0])} {stem(words[1])}")
    return out


@dataclass(frozen=True)
class Lexicon:
    """Stemmed climate and tone dictionaries ready for matching."""

    climate: frozenset[str]
    opportunity: frozenset[str]
    regulatory: frozenset[str]
    physical: frozenset[str]
    tone_positive: frozenset[str]
    tone_negative: frozenset[str]
    # Risk words used to build the risk conditioned measure. These follow the
    # uncertainty vocabulary used in the firm level literature.
    risk: frozenset[str]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "climate": len(self.climate),
            "opportunity": len(self.opportunity),
            "regulatory": len(self.regulatory),
            "physical": len(self.physical),
            "tone_positive": len(self.tone_positive),
            "tone_negative": len(self.tone_negative),
            "risk": len(self.risk),
        }


# Vocabulary flagging a sentence as expressing risk or uncertainty. The measure
# uses it to separate climate risk from neutral climate discussion.
_RISK_WORDS = (
    "risk risky uncertain uncertainty unknown fluctuate fluctuation possible probable "
    "probability variable variability volatile volatility threat threaten danger dangerous "
    "hazard hazardous doubt unpredictable unforeseen unstable instability sudden chance "
    "likelihood contingency speculative tentative cautious warn warning worry concern fear "
    "exposure"
).split()


def load_lexicon(data_dir: Path | None = None) -> Lexicon:
    """Load and stem the climate lexicon and the tone dictionaries.

    Parameters
    ----------
    data_dir:
        Folder holding ``climate_lexicon.json`` and ``lm_tone.json``. Defaults
        to the ``data`` directory shipped with the package.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    climate_raw = json.loads((root / "climate_lexicon.json").read_text())
    tone_raw = json.loads((root / "lm_tone.json").read_text())

    climate = _stem_bigrams(climate_raw["C"])
    opportunity = _stem_bigrams(climate_raw["opp"])
    regulatory = _stem_bigrams(climate_raw["reg"])
    physical = _stem_bigrams(climate_raw["phy"])

    # Keep the three topics mutually exclusive, as in the original method.
    shared = (
        (opportunity & regulatory)
        | (opportunity & physical)
        | (regulatory & physical)
    )
    opportunity -= shared
    regulatory -= shared
    physical -= shared

    # Tone lists on disk are already stored stemmed and lowercase, but we stem
    # again so the module stays correct whatever the source file looks like.
    tone_positive = {stem(w.lower()) for w in tone_raw["pos"]}
    tone_negative = {stem(w.lower()) for w in tone_raw["neg"]}
    risk = {stem(w) for w in _RISK_WORDS}

    return Lexicon(
        climate=frozenset(climate),
        opportunity=frozenset(opportunity),
        regulatory=frozenset(regulatory),
        physical=frozenset(physical),
        tone_positive=frozenset(tone_positive),
        tone_negative=frozenset(tone_negative),
        risk=frozenset(risk),
    )
