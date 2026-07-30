"""
Quickstart that runs with no network access.

It scores two short passages so you can see the measure react. The first talks
about climate at length, the second is ordinary fund commentary. Run it with:

    python examples/quickstart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climate_nlp import load_lexicon, score_text

CLIMATE_PASSAGE = """
The fund increased its allocation to renewable energy and clean energy producers
over the period. Management believes the transition to solar energy and wind
power creates a durable investment opportunity. At the same time, carbon tax
proposals and tighter greenhouse gas regulation remain a meaningful risk for
carbon intensive holdings, and extreme weather events add physical uncertainty
to insurance and utility names.
"""

NEUTRAL_PASSAGE = """
The fund outperformed its benchmark during the period, driven by strong security
selection in the technology and healthcare sectors. Management reduced exposure
to consumer staples and added to financials as interest rates rose. Turnover
remained low and the expense ratio was unchanged from the prior year.
"""


def show(title: str, text: str, lexicon) -> None:
    score = score_text(text, lexicon)
    print(f"\n{title}")
    print(f"  bigrams counted     {score.n_bigrams}")
    print(f"  climate bigrams     {score.n_climate}")
    print(f"  exposure            {score.exposure:.2f}")
    print(f"    opportunity       {score.exposure_opportunity:.2f}")
    print(f"    regulatory        {score.exposure_regulatory:.2f}")
    print(f"    physical          {score.exposure_physical:.2f}")
    print(f"  risk                {score.risk:.2f}")
    print(f"  sentiment           {score.sentiment:.2f}")


def main() -> None:
    lexicon = load_lexicon()
    print("Lexicon sizes:", lexicon.sizes)
    show("Climate heavy passage", CLIMATE_PASSAGE, lexicon)
    show("Ordinary fund commentary", NEUTRAL_PASSAGE, lexicon)


if __name__ == "__main__":
    main()
