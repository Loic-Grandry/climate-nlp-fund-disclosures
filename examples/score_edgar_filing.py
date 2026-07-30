"""
Download a real shareholder report from EDGAR and score it.

This example needs network access and a contact string for the SEC. Set it once
in your shell, using your own email:

    export EDGAR_USER_AGENT="Your Name your.email@example.com"

Then run, optionally passing a filer CIK:

    python examples/score_edgar_filing.py            # uses the default CIK
    python examples/score_edgar_filing.py 884546     # any fund filer CIK

The script lists the filer's recent N-CSR filings, downloads the most recent
one and prints its climate score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climate_nlp import load_lexicon, score_text
from climate_nlp.edgar import download, list_ncsr_filings, user_agent

# A large fund complex used only as a default target. Any fund filer CIK works.
DEFAULT_CIK = 884546


def main() -> None:
    cik = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CIK
    print(f"User-Agent sent to EDGAR: {user_agent()}")

    filings = list_ncsr_filings(cik, limit=5)
    if not filings:
        print(f"No N-CSR filing found for CIK {cik}.")
        return

    print(f"\nRecent N-CSR filings for CIK {cik}:")
    for f in filings:
        print(f"  {f['date']}  {f['form']:8s}  {f['url']}")

    target = filings[0]
    print(f"\nDownloading {target['date']} filing ...")
    html = download(target["url"])

    lexicon = load_lexicon()
    score = score_text(html, lexicon)

    print("\nClimate score for the whole filing:")
    for key, value in score.as_dict().items():
        print(f"  {key:22s} {value:.3f}")


if __name__ == "__main__":
    main()
