"""
A small EDGAR client for fund shareholder reports.

Every SEC filing is public and free to download from EDGAR. The SEC asks each
program to identify itself in the HTTP ``User-Agent`` header with a contact
address and to stay under ten requests per second. This client reads the
contact string from the ``EDGAR_USER_AGENT`` environment variable so that no
personal address is ever written into the source code. Set it once in your
shell before running anything that touches the network:

    export EDGAR_USER_AGENT="Your Name your.email@example.com"

The functions here cover only what the demos need: list a filer's N-CSR
filings and download a filing document. They are deliberately thin. For heavy
collection work a dedicated library such as ``sec-edgar-downloader`` is a better
fit.
"""
from __future__ import annotations

import os
import time
from urllib.request import Request, urlopen

BASE = "https://www.sec.gov"
DATA = "https://data.sec.gov"

# Neutral fallback used only if the environment variable is not set. It carries
# no personal information. Replace it by exporting EDGAR_USER_AGENT.
_FALLBACK_USER_AGENT = (
    "climate-nlp-fund-disclosures (set the EDGAR_USER_AGENT variable "
    "with your contact email)"
)

# Be a good citizen and stay well under the published rate limit.
_MIN_INTERVAL_SECONDS = 0.2
_last_request = 0.0


def user_agent() -> str:
    """Return the User-Agent string, preferring the environment variable."""
    return os.environ.get("EDGAR_USER_AGENT", _FALLBACK_USER_AGENT)


def _get(url: str) -> bytes:
    """Fetch a URL with the required header and a gentle rate limit."""
    global _last_request
    wait = _MIN_INTERVAL_SECONDS - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    request = Request(url, headers={"User-Agent": user_agent()})
    with urlopen(request) as response:
        data = response.read()
    _last_request = time.time()
    return data


def list_ncsr_filings(cik: str | int, limit: int = 10) -> list[dict]:
    """List the most recent N-CSR filings for a filer.

    Parameters
    ----------
    cik:
        Central Index Key of the filer, with or without leading zeros.
    limit:
        Maximum number of filings to return, most recent first.

    Returns
    -------
    A list of dictionaries with the accession number, filing date and the URL
    of the primary document.
    """
    cik10 = str(int(cik)).zfill(10)
    import json

    submissions = json.loads(_get(f"{DATA}/submissions/CIK{cik10}.json"))
    recent = submissions["filings"]["recent"]

    out: list[dict] = []
    for form, accession, date, primary in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["filingDate"],
        recent["primaryDocument"],
    ):
        if not form.startswith("N-CSR"):
            continue
        acc_nodash = accession.replace("-", "")
        url = f"{BASE}/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary}"
        out.append(
            {
                "form": form,
                "accession": accession,
                "date": date,
                "url": url,
            }
        )
        if len(out) >= limit:
            break
    return out


def download(url: str) -> str:
    """Download a filing document and return it as text."""
    return _get(url).decode("utf-8", errors="replace")
