# AEJ Homework — Energy Desk Information Stream

Take-home assignment: extract structured quote records, classify market sentiment, and investigate whether sentiment predicts price action — all from ~29,000 raw OTC trading chat messages.

---

## Structure

```
.
├── candidate_sample.csv      # raw input: ~29k chat messages
├── price_series.csv          # daily prices: brent, brent_prompt_spread, gasoil_crack, brent_wti_arb
├── cleaning.py               # Part 1 — quote extraction pipeline
├── classify.py               # Part 2 — sentiment classification + daily index
├── predictions_quotes.csv    # output: one row per resolved quote record
├── predictions_sentiment.csv # output: one row per message with sentiment label
├── requirements.txt
└── BRIEF.md                  # original task brief
```
---

## Setup

```bash
pip install -r requirements.txt
```
---

## Running

**Part 1 — Quote extraction:**
```bash
python cleaning.py
```
Reads `candidate_sample.csv`, outputs `predictions_quotes.csv`.

**Part 2 — Sentiment classification:**
```bash
python classify.py
```
Reads `predictions_quotes.csv`, outputs `predictions_sentiment.csv`.

---

## Part 1 — Quote Extraction (`cleaning.py`)

### Approach

Hybrid regex + rule-based pipeline. The choice over an LLM-only approach was speed and determinism: with ~29k messages and a well-defined domain vocabulary, regex patterns cover the vast majority of cases without per-message API calls.

**Pipeline stages:**

**Ingestion** — the raw CSV has commas inside message text that break standard parsing. Excess fields are joined with `|` as a safe delimiter, then re-parsed cleanly.

**Timestamp normalisation** — timestamps arrive in two formats (ISO strings with mixed timezones, and Unix epoch integers). A numeric mask splits them into two paths, both normalised to UTC.

**Deduplication** — `FW:`, `>` (quote-reply), `[repost]`, and `RT` prefixes are detected and filtered before extraction. These are copies of existing messages and must not produce duplicate records.

**Instrument extraction** — ten regex patterns cover the full OTC instrument taxonomy: flat price (WTI/Brent), time spreads, crack spreads (gasoil, jet, naphtha, marine, VLSFO, HSFO, fuel oil), ratio cracks (3-2-1, 0.5%, 3.5%), CFDs, dated/DFL, DTD, the arb, BWCS, and WTI/Brent cross-spread. A shared `PERIOD` token handles all date/quarter/calendar-year conventions.Social and noise messages (`drinks after work?`, `invoice 7781 paid`) were filtered out as irrelevant.

**State resolution** — messages are processed in chronological order per author. A `last_quote` register tracks the most recent live quote per author:
- `scrap that`, `pull that`, `kill it`, `cancel that`, `withdraw my bid` → cancel the last live quote
- `scrap my <instrument>`, `pull the <period>` → fuzzy-match against last quote and cancel if overlap found
- `amend to`, `make that`, `update`, `correction`, `scrap that, now` → mark last quote as amended, inherit instrument, apply new price

**Bid/offer splitting** — a priority-ordered regex chain handles all price formats seen in the data: `X at Y`, `X / Y`, `X-Y`, `bid X off Y`, `Xb` (bid-only), `Xo` (offer-only), `sell X`, `pay X`, `X bid`, `X offered`. Size tokens (`x50`, `x25kb`, `2x2`, `10kb`) are stripped before number extraction to avoid mismatches.

**Multi-instrument messages** — messages quoting two instruments separated by `|` (the injected delimiter from ingestion) are split per segment, each producing a separate output row.

### Output schema

`predictions_quotes.csv`:

| Column | Description |
|---|---|
| `message_id` | original message ID |
| `instrument` | extracted instrument string |
| `time_reference` | normalised UTC timestamp |
| `bid` | bid price (float or blank) |
| `offer` | offer price (float or blank) |
| `source` | author |
| `confidence` | extraction confidence (1.0 for regex matches) |
| `text` | original message text |
| `instrument_type` | instrument category |
| `status` | `live`, `amended`, `cancelled` |
| `ref_message_id` | message ID of the quote this amends/cancels |

### Known limitations

- Amendments with no instrument hint inherit the last author quote, which can mis-link if the author switched instruments between messages.
- Vague levels ("around 0.50", "low 80s") are not filtered — confidence is set uniformly to 1.0 rather than graded.

---

## Part 2 — Sentiment Classification (`classify.py`)

### Approach

[FinBERT](https://huggingface.co/ProsusAI/finbert) (ProsusAI) — a BERT model fine-tuned on financial news. Chosen over general-purpose sentiment models because it understands domain-specific bullish/bearish language (`stocks drew hard, products flying`, `crude getting hit, risk off`, `cracks rolling over`) without requiring further fine-tuning.

Labels are mapped: `positive -> bullish`, `negative -> bearish`, `neutral -> neutral`. 

Each message is scored as `+confidence` (bullish), `-confidence` (bearish), or `0` (neutral) to produce a scalar in `[-1, +1]` for time-series aggregation.

### Output schema
`predictions_sentiment.csv`: `message_id, sentiment, confidence, score`
