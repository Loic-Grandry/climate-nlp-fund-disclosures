# Methodology

This document explains, step by step, how a fund filing becomes a set of climate
numbers. It covers what kind of method this is, the full processing pipeline, the
exact rules at each stage, and the design choices behind them.

## What kind of method this is

This is a dictionary based textual analysis, sometimes called a keyword or
pattern based approach. It is not a neural network and not a large language
model. It counts how often a document uses word pairs that belong to a curated
climate vocabulary, and it normalises that count by document length.

This is a deliberate choice, not a shortcut. The method reproduces Sautner, van
Lent, Vilkov and Zhang (2023), which itself follows the firm level political risk
method of Hassan, Hollander, van Lent and Tahoun (2019). Both are dictionary
based, for three good reasons that matter in finance research.

- **It is transparent.** Every number can be traced back to the exact bigrams
  that produced it, which matters when a result has to survive a referee or a
  risk committee.
- **It is stable over time.** The same text always gives the same score, with no
  dependence on a model version or a random seed.
- **It is cheap and scales.** Scoring tens of thousands of filings needs no GPU
  and no training data.

The sophistication is not in a heavy model. It is in the pipeline that turns very
messy filings into a clean, comparable signal, and in the adaptation of a firm
level measure to the fund setting.

## The pipeline at a glance

```
raw filing (HTML, up to ~15 MB)
        |
        v
[1] strip HTML to plain text
        |
        v
[2] split into sentences, keep prose, drop table rows
        |
        v
[3] tokenize, lowercase, keep letters, Porter stem
        |
        v
[4] build in-sentence bigrams, match against the climate lexicon
        |
        v
[5] condition on sentence level risk and tone words
        |
        v
climate exposure, risk and sentiment, per document or per fund
```

## Stage 1, HTML to text

A shareholder report is delivered as an HTML file that can reach fifteen megabytes
because it embeds full holdings tables and XBRL data. The first stage removes
script and style blocks, deletes every remaining tag, decodes the common HTML
entities and collapses runs of whitespace. See `html_to_text` in
[`src/climate_nlp/text.py`](../src/climate_nlp/text.py).

## Stage 2, keeping prose and dropping tables

This is the stage that makes the fund adaptation work. A firm earnings call
transcript is almost all prose. A fund filing is mostly tables: the schedule of
investments, the statement of assets and liabilities, returns by share class.
Those tables carry ticker symbols, numbers and fragments of company names that
would pollute the bigram counts.

The text is split into sentences on sentence ending punctuation. A sentence is
kept only if it passes all three filters below. Anything that fails looks like a
table row rather than narrative.

| Filter | Rule | Why |
| --- | --- | --- |
| Length | at least 8 words | table cells and headers are short |
| Digit density | fewer than 10 percent of characters are digits | table rows are full of figures |
| Numeric tokens | fewer than 20 percent of tokens are pure numbers or symbols | catches rows of prices and weights |

On a real Calvert filing this stage cuts a 14 megabyte document down to roughly
30,000 words of genuine narrative across about 1,200 sentences, which is around 40
percent of the stripped text. The rest was tables.

## Stage 3, tokenization and stemming

Each kept sentence is lowercased. Only alphabetic tokens are retained, so numbers
and punctuation drop out. Every token is then reduced to its root with the Porter
stemmer, so "renewable energies" and "renewable energy" collapse to the same
stem. The climate lexicon and the tone word lists are stemmed with the same
stemmer, which is what makes the matching consistent. Stemming is cached, so each
distinct word is stemmed only once even across tens of thousands of filings.

## Stage 4, bigrams and matching

Inside each sentence, every pair of adjacent tokens forms a bigram. Bigrams never
cross a sentence boundary, so the last word of one sentence is never glued to the
first word of the next. The total number of bigrams in the document is `B`, the
denominator of every measure.

Each bigram is looked up in the climate lexicon `C` and in the three topical
lexicons. Lookups are set membership tests, so scoring a document is linear in
its length. The overall exposure is the count of climate bigrams divided by `B`,
multiplied by 1000.

```
for each sentence:
    tokens  = stem(letters_only(lowercase(sentence)))
    bigrams = adjacent pairs of tokens
    B += len(bigrams)
    for bg in bigrams:
        if bg in C:            climate_hits += 1
        if bg in C_opportunity: opportunity_hits += 1
        if bg in C_regulatory:  regulatory_hits += 1
        if bg in C_physical:    physical_hits += 1

exposure = 1000 * climate_hits / B
```

## Stage 5, risk and tone conditioning

Risk and sentiment reuse the same climate bigram hits but ask an extra question
about the sentence they sit in.

For each climate bigram, the code checks the set of words in its sentence. If that
set contains a risk word, the bigram also counts toward the risk measure. If it
contains a positive tone word, it counts positive, and if it contains a negative
tone word, it counts negative. Sentiment is the positive count minus the negative
count, again divided by `B`.

Doing this at the sentence level, rather than across the whole document, keeps the
framing attached to the climate content it belongs to. A sentence about a carbon
tax threat contributes to risk, a sentence about a solar growth opportunity does
not, even though both are in the same filing.

## Per fund segmentation

A single N-CSR is often filed by a trust that holds many funds. Their narratives
sit one after another in the same file, each introduced by the fund name. To score
funds separately the pipeline locates every fund name in the text, cuts the
document at those points, and scores each slice on its own.

Fund names in the body rarely match the registered name character for character,
so the matcher tolerates small differences such as "U.S." written as "US", an
ampersand written as "and", and extra spaces or punctuation between words, while
still requiring the words in order. See `name_pattern` and `score_by_fund` in
[`src/climate_nlp/funds.py`](../src/climate_nlp/funds.py).

This step matters because the trust level score is diluted. In the worked example,
the Calvert trust as a whole scores 0.30, but once the text of the Calvert Global
Energy Solutions Fund is isolated from its siblings, that fund scores 3.16. The
climate signal was there all along, spread thin across a document that also covers
a water fund, a green bond fund and a small cap fund.

## The lexicon

The full climate dictionary that Sautner et al. learn from data is distributed
only inside their replication package. This project rebuilds a working dictionary
from the seed tables printed in the paper's internet appendix, expanded with the
top ranked terms the authors report. After stemming and after removing bigrams
that fall into more than one topic, the dictionary holds:

| Lexicon | Size |
| --- | --- |
| Climate, overall | 116 bigrams |
| Opportunity | 94 bigrams |
| Regulatory | 94 bigrams |
| Physical | 51 bigrams |
| Positive tone (Loughran-McDonald) | 140 stems |
| Negative tone (Loughran-McDonald) | 876 stems |
| Risk words | 31 stems |

This is a faithful but partial version of the authors' full list. Swapping in the
complete dictionary is a one file change in
[`data/climate_lexicon.json`](../data/climate_lexicon.json) and needs no code
change.

## Scale and cost

The method is linear in document length and needs no training. A single large
filing, around 14 megabytes of HTML that reduces to roughly 50,000 bigrams, is
downloaded and scored in a couple of seconds on a laptop. Running the measure
across a full universe of tens of thousands of filings is an overnight job on one
machine, with no GPU.

## Honest limits

- The dictionary is a reconstruction, so it misses rarer bigrams that the full
  learned list would catch.
- The method counts vocabulary, it does not read for meaning. It cannot tell a
  real climate strategy from boilerplate risk language, so it is best read as a
  relative signal across funds and over time.
- Modern tailored shareholder reports, introduced by the SEC in 2024, are much
  shorter than the older N-CSR format and carry little narrative, so they produce
  low scores by design. The measure is most informative on the richer pre 2024
  filings.
