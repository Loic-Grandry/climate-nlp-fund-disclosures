"""
Worked example on real filings downloaded from EDGAR.

This shows the measure at full scale, not on toy text. It does two things.

First it scores three real fund shareholder reports from 2017, so you can see the
document sizes and the climate numbers on genuine narrative.

Then it takes one trust filing that covers four Calvert funds and splits it by
fund, to show how per fund segmentation recovers a climate signal that is diluted
at the trust level.

It needs network access and a contact string for the SEC. Set your own email:

    export EDGAR_USER_AGENT="Your Name your.email@example.com"
    python examples/worked_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climate_nlp import load_lexicon, score_by_fund, score_text
from climate_nlp.edgar import download, user_agent
from climate_nlp.text import extract_prose, html_to_text

# Three real N-CSR filings from 2017, chosen across the climate spectrum.
FILINGS = [
    (
        "Guinness Atkinson Alternative Energy Fund",
        "https://www.sec.gov/Archives/edgar/data/919160/000110465917015754/a17-1912_2ncsr.htm",
    ),
    (
        "Pax Global Environmental Markets Fund",
        "https://www.sec.gov/Archives/edgar/data/76721/000139834417003017/fp0024398_ncsr.htm",
    ),
    (
        "Calvert Impact Funds trust (covers four funds)",
        "https://www.sec.gov/Archives/edgar/data/1121624/000162828017011880/calvertimpactfundncsr930doc.htm",
    ),
]

# Public SEC series names for the funds in the Calvert trust filing above.
CALVERT_SERIES = {
    "S000017171": "Calvert Global Energy Solutions Fund",
    "S000017172": "Calvert Global Water Fund",
    "S000030675": "Calvert Green Bond Fund",
    "S000008724": "Calvert Small-Cap Fund",
}


def scale_line(name: str, html: str, lexicon) -> None:
    prose = extract_prose(html_to_text(html))
    words = sum(len(s.split()) for s in prose)
    score = score_text(html, lexicon)
    print(f"\n{name}")
    print(
        f"   HTML {len(html):,} chars, prose {words:,} words in {len(prose):,} "
        f"sentences, {score.n_bigrams:,} bigrams"
    )
    print(
        f"   exposure {score.exposure:.2f}  opportunity {score.exposure_opportunity:.2f}"
        f"  regulatory {score.exposure_regulatory:.2f}  risk {score.risk:.2f}"
        f"  sentiment {score.sentiment:.2f}"
    )


def main() -> None:
    print(f"User-Agent sent to EDGAR: {user_agent()}")
    lexicon = load_lexicon()

    print("\n" + "=" * 70)
    print("Part 1. Three real filings, scored whole")
    print("=" * 70)
    calvert_html = None
    for name, url in FILINGS:
        html = download(url)
        if name.startswith("Calvert"):
            calvert_html = html
        scale_line(name, html, lexicon)

    print("\n" + "=" * 70)
    print("Part 2. Splitting the Calvert trust filing by fund")
    print("=" * 70)
    results = score_by_fund(calvert_html, CALVERT_SERIES, lexicon)
    print(
        f"\nWhole trust exposure {results['_trust'].exposure:.2f}, "
        f"diluted across {len(CALVERT_SERIES)} funds"
    )
    print("\nAfter isolating each fund's own narrative:")
    scored = []
    for series_id, value in results.items():
        if series_id == "_trust":
            continue
        score, n_anchors = value
        if score is not None and n_anchors > 0:
            scored.append((CALVERT_SERIES[series_id], score, n_anchors))
    scored.sort(key=lambda row: -row[1].exposure)
    for name, score, n_anchors in scored:
        print(
            f"   exposure {score.exposure:6.2f}   bigrams {score.n_bigrams:6,}   "
            f"name matches {n_anchors:2d}   {name}"
        )


if __name__ == "__main__":
    main()
