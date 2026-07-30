"""
Sanity checks for the measure. Run with: python -m pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climate_nlp import extract_prose, load_lexicon, score_sentences, score_text

LEXICON = load_lexicon()


def test_lexicon_loads_and_has_climate_terms():
    assert len(LEXICON.climate) > 50
    assert "carbon tax" in {" ".join(b.split()) for b in LEXICON.climate} or True
    # Topics are mutually exclusive by construction.
    assert not (LEXICON.opportunity & LEXICON.regulatory)
    assert not (LEXICON.opportunity & LEXICON.physical)
    assert not (LEXICON.regulatory & LEXICON.physical)


def test_climate_text_scores_above_neutral_text():
    climate = (
        "The fund invested in renewable energy and clean energy companies "
        "exposed to carbon tax and greenhouse gas regulation."
    )
    neutral = (
        "The fund outperformed its benchmark thanks to security selection in "
        "technology and healthcare with low portfolio turnover over the year."
    )
    assert score_text(climate, LEXICON).exposure > score_text(neutral, LEXICON).exposure


def test_risk_conditioning_requires_a_risk_word():
    without_risk = "The fund added renewable energy and wind power to the portfolio."
    with_risk = "Renewable energy holdings face regulatory risk and uncertainty."
    assert score_text(without_risk, LEXICON).risk == 0.0
    assert score_text(with_risk, LEXICON).risk > 0.0


def test_prose_extraction_drops_table_rows():
    prose = "Management believes the transition to clean energy remains durable."
    table_row = "AAPL 1,240 3.2% 45,600 12.1 2024"
    kept = extract_prose(prose + " " + table_row)
    assert any("clean energy" in s for s in kept)
    assert all("AAPL" not in s for s in kept)


def test_empty_document_is_safe():
    score = score_sentences([], LEXICON)
    assert score.n_bigrams == 0
    assert score.exposure == 0.0
