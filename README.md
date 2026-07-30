# Measuring climate exposure in fund disclosures

This project measures how much a US mutual fund talks about climate change in
its shareholder reports, and how it frames that discussion. It reads the
narrative text of filings that funds send to the Securities and Exchange
Commission, and it turns that text into a small set of numbers: how much of the
document is about climate, whether the climate content is about opportunities or
about risks, and whether the tone is positive or negative.

The method comes from a finance paper. The contribution here is to take a
measure built for company earnings calls and rebuild it for fund filings, using
only data that anyone can download for free from the SEC.

Everything runs on public data and open source tools. No proprietary database is
needed.

## What this method is, and what it is not

This is a dictionary based textual analysis. It counts word pairs that belong to
a curated climate vocabulary and normalises the count by document length. It is
not a neural network and not a large language model, and that is on purpose. It
reproduces the academic method described below, which is dictionary based for
good reasons: every number can be traced back to the exact words that produced
it, the same text always gives the same score, and the whole thing scales to tens
of thousands of filings with no GPU and no training data. The real work is not a
heavy model. It is the pipeline that turns very messy filings into a clean,
comparable signal, and the adaptation of a firm level measure to funds. The full
algorithm is written out in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Where the idea comes from

The starting point is Sautner, van Lent, Vilkov and Zhang, *Firm-Level Climate
Change Exposure*, Journal of Finance, 2023. Their idea is simple and powerful.
When a company is exposed to climate change, whether as a threat or as a business
opportunity, that exposure tends to surface in how managers and analysts talk.
So instead of relying on a rating agency, you can measure exposure directly from
the language used in a firm's quarterly earnings calls.

The paper builds on an earlier method by Hassan, Hollander, van Lent and Tahoun
(*Firm-Level Political Risk*, Quarterly Journal of Economics, 2019), which counts
how often a text uses word pairs tied to a topic. A word pair, or bigram, is two
words that sit next to each other, such as "renewable energy" or "carbon tax".
Working with bigrams rather than single words matters, because "carbon" alone is
ambiguous while "carbon tax" is clearly about climate policy.

The authors do not fix the list of climate bigrams by hand. They learn it from
data with a machine learning approach that starts from a small set of seed terms
and expands to the bigrams that keep the same company each time they appear. The
result is a climate dictionary, plus three narrower dictionaries for the three
channels through which climate matters to a firm:

- **Opportunity**, the upside, for example "solar energy" or "electric vehicle".
- **Regulatory**, the policy risk, for example "carbon tax" or "greenhouse gas".
- **Physical**, the direct physical risk, for example "sea level" or "extreme weather".

## The four measures

Every measure is a frequency. You count how many bigrams in a document belong to
the climate dictionary, and you divide by the total number of bigrams in that
document. Dividing by the length is what makes a short letter and a long annual
report comparable. The frequency is then multiplied by a fixed number, 1000
here, so the output reads as 12.5 rather than 0.0125.

Write `B` for the number of bigrams in a document and `C` for the climate
dictionary. The overall exposure is

```
Exposure = (1 / B) * count of bigrams that are in C
```

The three topical measures are the same formula with the opportunity, regulatory
or physical dictionary in place of `C`.

The risk measure looks only at the climate bigrams whose sentence also contains
a word about risk or uncertainty, such as "risk", "threat" or "volatile". It
captures how much of the climate talk is framed as something to worry about.

```
Risk = (1 / B) * count of climate bigrams whose sentence also mentions risk
```

The sentiment measure does the same with tone. A climate bigram counts positive
when its sentence contains a positive tone word and negative when it contains a
negative one. Sentiment is the positive count minus the negative count. Tone
words come from the Loughran and McDonald finance dictionaries, which are the
standard word lists for financial text.

```
Sentiment = (1 / B) * (positive climate bigrams - negative climate bigrams)
```

Reading tone and risk at the sentence level, rather than across the whole
document, keeps the framing attached to the climate content it belongs to.

## What changes to move from firms to funds

The original measure reads earnings call transcripts of individual companies.
This project reads a different kind of document and has to solve problems that do
not exist in the firm setting.

**The source is fund filings, not transcripts.** US mutual funds file a report
to shareholders twice a year on form N-CSR, and its half-year variant N-CSRS.
These reports are public and free on the SEC EDGAR system. The part that matters
is the Management Discussion of Fund Performance, where the manager explains in
plain language what drove returns and what the fund is exposed to. That is the
fund equivalent of a manager speaking on an earnings call.

**Filings are messy HTML full of tables.** A shareholder report is not clean
prose. It is a web page that mixes narrative paragraphs with very large financial
tables: the full list of holdings, the statements of assets and operations,
returns broken down by share class. Those tables are noise for this measure and
would badly distort the bigram counts. The pipeline strips the HTML to plain
text and then keeps only sentences that read like prose. Table rows are easy to
spot because they are short and dense with numbers, tickers and punctuation, so
three simple density filters remove them. This step is described in
[`text.py`](src/climate_nlp/text.py).

**One filing can cover many funds.** A fund trust often files a single N-CSR that
contains the reports of dozens of funds one after another, each introduced by its
name. To score funds separately, the pipeline finds every fund name in the text
and cuts the document at those points, so each slice holds the prose of one fund.
Fund names in the body rarely match the registered name exactly, so the matcher
is tolerant of small differences such as "U.S." written as "US" or an ampersand
written as "and". This lives in [`funds.py`](src/climate_nlp/funds.py).

**The full author dictionary is not public, so it is reconstructed.** The
complete learned climate dictionary ships only inside the authors' replication
package. This project rebuilds a working dictionary from the seed tables printed
in the paper's internet appendix, expanded with the highest ranked terms the
authors report. The dictionary in [`data/climate_lexicon.json`](data/climate_lexicon.json)
holds 116 climate bigrams, split into 94 opportunity, 94 regulatory and 51
physical bigrams after removing pairs that fall into more than one channel. This
is a faithful but partial version of the full list. If you have access to the
authors' complete dictionary, you can drop it into the same JSON file and every
number updates with no code change. This is an honest limitation and it is
flagged again below.

**Every dictionary is stemmed.** Words are reduced to their root with the Porter
stemmer, so "renewable energies" and "renewable energy" match the same entry. The
climate bigrams and the tone words are stemmed the same way, which keeps the
matching consistent.

## Repository layout

```
climate-nlp-fund-disclosures/
├── src/climate_nlp/
│   ├── lexicon.py     load and stem the climate and tone dictionaries
│   ├── text.py        turn filing HTML into clean narrative sentences
│   ├── measure.py     the exposure, risk and sentiment measure
│   ├── funds.py       split a trust filing into individual funds
│   └── edgar.py       a small client to download filings from EDGAR
├── data/
│   ├── climate_lexicon.json   climate, opportunity, regulatory, physical bigrams
│   └── lm_tone.json           Loughran-McDonald positive and negative words
├── docs/
│   └── METHODOLOGY.md         the full algorithm, stage by stage
├── examples/
│   ├── quickstart.py          runs offline on two sample passages
│   ├── score_edgar_filing.py  downloads a real N-CSR and scores it
│   └── worked_example.py      three real filings plus a per fund split
└── tests/
    └── test_measure.py
```

## Installation

```bash
pip install -r requirements.txt
```

The only real dependency is `nltk`, for the Porter stemmer.

## Quickstart, no network needed

```bash
python examples/quickstart.py
```

It scores one climate heavy passage and one ordinary passage of fund commentary,
so you can see the numbers move. The climate passage returns a high exposure with
a clear split across opportunity and regulatory channels, while the ordinary
commentary returns zero.

Using the library directly:

```python
from climate_nlp import load_lexicon, score_text

lexicon = load_lexicon()
score = score_text(open("filing.html").read(), lexicon)

print(score.exposure)              # overall climate exposure
print(score.exposure_regulatory)   # regulatory channel
print(score.risk)                  # climate risk framing
print(score.sentiment)             # positive minus negative
```

## Scoring a real filing from EDGAR

The SEC asks every program that downloads from EDGAR to identify itself with a
contact email in the request header, and to stay under ten requests per second.
This project reads that contact string from an environment variable, so no
personal address is ever written into the code. Set it once with your own email:

```bash
export EDGAR_USER_AGENT="Your Name your.email@example.com"
python examples/score_edgar_filing.py
```

The script lists a filer's recent N-CSR filings, downloads the most recent one
and prints its full climate score. Pass any fund filer CIK as an argument to
point it at a different filer.

## Worked example on real filings

`examples/worked_example.py` downloads three real 2017 shareholder reports and
scores them at full scale, then splits one trust filing into its individual
funds. The output looks like this.

```
Guinness Atkinson Alternative Energy Fund
   HTML 3,367,042 chars, prose 28,919 words in 1,189 sentences, 27,450 bigrams
   exposure 1.20  opportunity 0.87  regulatory 0.00  risk 0.22  sentiment 0.29

Pax Global Environmental Markets Fund
   HTML 8,876,161 chars, prose 28,599 words in 1,028 sentences, 27,005 bigrams
   exposure 0.96  opportunity 0.26  regulatory 0.19  risk 0.11  sentiment 0.11

Calvert Impact Funds trust (covers four funds)
   HTML 3,726,337 chars, prose 31,212 words in 1,247 sentences, 29,807 bigrams
   exposure 0.30
```

These are real documents. Each one is several megabytes of HTML that the prose
filter reduces to around 30,000 words of narrative and roughly 27,000 bigrams.

The Calvert filing is a good illustration of why the per fund split matters. The
trust as a whole scores a modest 0.30, because most of its funds are not about
climate. Splitting it by fund recovers the signal.

```
Whole trust exposure 0.30, diluted across 4 funds

After isolating each fund's own narrative:
   exposure   3.16   Calvert Global Energy Solutions Fund
   exposure   0.40   Calvert Global Water Fund
   exposure   0.10   Calvert Green Bond Fund
   exposure   0.00   Calvert Small-Cap Fund
```

The Global Energy Solutions Fund, whose whole mandate is the energy transition,
scores ten times the trust average once its own text is isolated. The measure
picks out the fund a reader would expect, from language alone.

## Limitations

- The climate dictionary is a reconstruction from the published seed tables, not
  the authors' full learned list. It captures the core vocabulary well but will
  miss rarer bigrams. Swapping in the complete list is a one file change.
- The measure counts vocabulary. It does not understand meaning, so it cannot
  tell a genuine climate strategy from boilerplate risk language. It is best read
  as a relative signal across funds and over time, not as an absolute judgement
  of one fund.
- Prose extraction is tuned for the table heavy style of N-CSR filings. Very
  differently formatted documents may need the density thresholds in `text.py`
  adjusted.

## References

- Sautner, Z., van Lent, L., Vilkov, G., and Zhang, R. (2023). Firm-Level Climate
  Change Exposure. *Journal of Finance*, 78(3), 1449-1498.
- Hassan, T. A., Hollander, S., van Lent, L., and Tahoun, A. (2019). Firm-Level
  Political Risk: Measurement and Effects. *Quarterly Journal of Economics*,
  134(4), 2135-2202.
- Loughran, T., and McDonald, B. (2011). When Is a Liability Not a Liability?
  Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35-65.
