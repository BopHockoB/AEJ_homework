# Take-home brief — information stream: records, sentiment, signals

You're given ~29,000 short, free-text messages captured from a real-time
information stream over roughly one trading quarter (an energy / commodities
desk), plus a small daily price file covering the same period. The messages are
unstructured: price quotes embedded in casual language, market commentary,
social chatter, and noise. This is a sample of a feed that runs at far higher
volume in production. There are three parts.

> The Part 1 core task is given verbatim; the points under it are part of the
> requirement.

---

## Part 1 — Build the table of records

Build ua pipeline that extracts structred quote records from this corpus. A
quote trecord should contain: instrument idenifier, time reference, bid, offer,
source/author, timestamp, and an extraction confidence score.

- Decide your own approach. Regex, classical ML, fine-tuned small model,
  LLM-assisted, hybrid — your call. Justify it.
- You may use any open-source library and any LLM API. Tell us what you used and why.

The feed is real, so it is messy. Your output should be the **current,
de-duplicated set of records**, not a transcript of every line:

- **Retransmissions / forwards.** The same quote is re-sent, forwarded (`FW:`),
  quote-replied (`>`), reposted, echoed in different case. These are the same
  quote and must not produce duplicate records.
- **Amendments.** A quote is often updated later ("scrap that, now 0.50/0.60").
  The record should reflect the **latest** values, not the superseded ones.
  Amendment messages may not repeat the instrument or tenor — link them.
- **Cancellations.** A pulled quote ("pull my 0.45 bid", "cancel that") should
  **not** appear in your records.
- **Timestamps** arrive in mixed formats and time zones — normalize them.
- A few rows are **malformed or multi-line** — ingest robustly.

---

## Part 2 — Sentiment score / index over time

Classify each message's market sentiment (`bullish` / `bearish` / `neutral`,
per message, independent of whether it contains a quote; social and noise are
neutral) and aggregate it into a **sentiment index over time** — a daily index
showing how market mood moved across the quarter, including trends and any
regime shifts. Show the index and describe how you built it. Quote levels in the
chat are independent of sentiment, so the index must come from the language.

---

## Part 3 — Signals: sentiment vs price action

`price_series.csv` contains daily levels for four market series over the same
period: `brent`, `brent_prompt_spread`, `gasoil_crack`, `brent_wti_arb`.

Using the sentiment index you built in Part 2, investigate whether sentiment has
any **predictive** relationship to these series, and present your findings.
We're interested in how you separate a genuine lead from a coincident move or a
spurious correlation, and how honest you are about significance given the sample.
Be specific about which series (if any) sentiment leads, at what horizon, and how
confident you are. A chart or two is welcome. This part is open-ended and is the
basis for a short presentation.

---

## Presentation

You will **present your approach to us in person.** Come prepared to walk a
panel through, in roughly 15 minutes: how you built the extraction pipeline and
why (including how you handled the duplicates, amendments, cancels and dirty
data); how you built the sentiment index; and your Part 3 findings on whether
sentiment predicts price. Expect questions throughout. We care as much about how
you reason and justify your choices as about the final numbers, so be ready to
defend trade-offs and to be honest about what didn't work and where the limits
are. A few slides or a notebook to share your screen is fine; polish is not the
point.

---

## Deliverables

- Code runnable from a single command. README required.
- A short write-up (max 2–3 pages): your approach for each part; honest accuracy
  estimates for extraction and sentiment; your Part 3 findings; what you'd do
  differently with more time; **how this scales to production volume (millions
  of messages per day) and what it would cost**, being specific about where the
  expensive steps are and how you'd keep cost and latency down; and what you'd
  need from us to productionize.
- Output CSVs (so we can score on a held-out set):

`predictions_quotes.csv` — one row per **resolved** record:

`message_id, instrument, time_reference, bid, offer, source, timestamp (normalized, ISO 8601 UTC preferred), confidence`.

`predictions_sentiment.csv` — one row per message: `message_id, sentiment`.

`predictions_index.csv` *(optional)* — your daily index: `date, index`.

Notes: a message may carry more than one quote (one row each). One-sided quotes
("0.45 bid", "offered 0.55") are valid, missing side blank. Vague levels
("around 0.50", "low 80s") are not firm quotes — use confidence to say so.
