# %%
import io
import pandas as pd
import re

def fix_text(line):
    parts = line.split(",")
    if len(parts) > 4:
        return ",".join(parts[:4]) + "|" + "|".join(parts[4:])

    return line

# File contain 2 formats: str and unix so unify format in 2 separate actions
# 1. Create unix mask and turn value into date
# 2. Apply save function for the rest str values
def parse_datetime(df):
    ts = df["timestamp"]

    # True where the value is a unix timestamp (numeric)
    unix_mask = pd.to_numeric(ts, errors="coerce").notna()

    # Parse unix timestamps
    parsed_unix = pd.to_datetime(
        pd.to_numeric(ts[unix_mask], errors="coerce"), unit="s", errors="coerce",             utc=True
    )

    # Parse string timestamps
    parsed_str = pd.to_datetime(ts[~unix_mask], format="mixed" ,errors="coerce",             utc=True)

    # Combine into a single column
    df["parsed_timestamp"] = pd.NaT
    df.loc[unix_mask, "parsed_timestamp"] = parsed_unix.values
    df.loc[~unix_mask, "parsed_timestamp"] = parsed_str.values


    failed = df[df["parsed_timestamp"].isna()]

    if not failed.empty:
        print(f"{len(failed)} rows failed to parse:")
        print(failed[["message_id", "timestamp"]])
    else:
        print("All timestamps parsed successfully")


    df["timestamp"] = df["parsed_timestamp"]
    df = df.drop(columns=["parsed_timestamp"])


    return df

# Text contains different tags (e.g. FW, repost),
# so it also requires preprocessing so copies are not affecting final prediction
# if any of those are unique then need to keep it
# otherwise delete

def clean_text(df):
    mask = (
    ~df["text"].str.contains(r"\bFW:", case=False, na=False) &
    ~df["text"].str.contains(r">", na=False) &
    ~df["text"].str.contains(r"\[reposted\]|\[repost\]", case=False, na=False)
)
    df_clean = df[mask]

    df_clean["text"] = df_clean["text"].str.lower()
    df_clean["text"] = df_clean["text"].str.strip()

    return df_clean

#pattens for period extraction
PERIOD = (
    r'(?:'
    r'q[1-4][\']?\d{0,4}' #Quarters
    r'|bal\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*'  # Bal Jul
    r'|cal\s*\d{2,4}'  #calendar year
    r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\-\s]?\d{0,4}'  # Jan-25, Feb25
    r')'
)

# instrument and method extraction
PATTERNS = {

    "flat_price": re.compile(
        rf'(?i)\b(wti|brt|brent)\s+(?:flat\s+)?(?:bal\s+)?{PERIOD}\b'
    ),

    "time_spread": re.compile(
        r'(?i)\b(wti|brt|brent)\s+(?:flat\s+)?(?:bal\s+)?'
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        r'[\-/]?'
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        r'\d{0,4}\b'
    ),

    "crack_spread": re.compile(
        rf'(?i)\b(gaso(?:il)?|gas(?:oline)?|jet|nap(?:htha)?|marine|vlsfo|hsfo|fuel|mogas|go)'
        rf'\s+(?:cr[ka]k?|diff)\s*{PERIOD}\b'
    ),

    "ratio_crack": re.compile(
        rf'(?i)\b(3[\-\.]?2[\-\.]?1|321|3[\.\-]5%|0[\.\-]5%)\s+crack\s+{PERIOD}\b'
    ),

    "cfd": re.compile(
        r'(?i)\b(?:brent\s+)?cfd\s+'
        r'(?:wk\s*)?'
        r'(?:w?\d[\-/]w?\d'           # W0-W3, 0-3, 3-6
        r'|' + PERIOD + r')\b'
    ),

    "dated_dfl": re.compile(
        rf'(?i)\b(d(?:ate)?[td][\-\s]?(?:fl|frontline)|dfl)\s+{PERIOD}\b'
    ),

    "dtd": re.compile(
        r'(?i)\b(dtd\s+brent|dated\s+dtd|dtd[\-\s]?dtd|dtd\s+prompt|dated\s+prompt|dated\s+front(?:line)?)\b'
    ),

    "arb": re.compile(
        rf'(?i)\b(?:the\s+)?arb\s+{PERIOD}\b'
    ),

    "bwcs": re.compile(
        rf'(?i)\bBWCS\s+{PERIOD}\b'
    ),

    "wti_brent_spread": re.compile(
        rf'(?i)\b(wti|brent)[/\-](brent|wti)\s+{PERIOD}\b'
    ),
}

CANCEL_FULL = re.compile(
    r'(?i)^(?:scrap\s+that|pull\s+that|cancel\s+that'
    r'|kill\s+it|withdraw\s+my\s+(?:bid|offer)'
    r'|that\s+ones?\s+gone|done)\s*$'
)

CANCEL_INSTRUMENT = re.compile(
    r'(?i)^(?:scrap\s+my|pull\s+the)\s+(.+)'
)

CANCEL = re.compile(
    r'(?i)^(?:scrap\s+(?:that|my\s+(.+))|pull\s+(?:that|the\s+(.+))'
    r'|cancel\s+that|kill\s+it|withdraw\s+my\s+(?:bid|offer)'
    r'|that\s+ones?\s+gone|done)\s*$'
)

AMEND = re.compile(
    r'(?i)^(?:amend\s+to|make\s+that|update\s+|correction\s*|scrap\s+that,?\s+now)\s*(.+)'
)
# prefixes and qualifiers are limited to those
STRIP_PREFIXES = re.compile(
    r'^(?:morning\s*[-–]|fwiw|mkt|rt\s+|ok\s+|so\s+|right,\s*'
    r'|hearing\s+|showing\s+|anyone\s+|where\'s\s+|got\s+|i make\s+)\s*',
    re.I
)

STRIP_QUALIFIERS = re.compile(
    r'\b(?:in decent size|bid up|tight up here|feels well bid|looks heavy'
    r'|rolling over|sellers lined up|any interest|let me know|thoughts'
    r'|fyi|choice|in size|x\d+kb|x\d+|grinding higher|feels toppy'
    r'|offered down|buyers around)\b',
    re.I
)

# %%

def _find_instruments(text: str) -> list[dict]:
    """Run all instrument patterns against a cleaned text string."""
    text = STRIP_PREFIXES.sub('', text.strip())
    text = STRIP_QUALIFIERS.sub('', text)

    results = []
    for instrument_type, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            results.append({
                "instrument_type": instrument_type,
                "instrument":      match.group().strip(),
                "span":            match.span(),
            })

    # deduplicate overlapping matches, keep longest
    results.sort(key=lambda x: (x["span"][0], -(x["span"][1] - x["span"][0])))
    deduped, last_end = [], -1
    for r in results:
        if r["span"][0] >= last_end:
            deduped.append(r)
            last_end = r["span"][1]

    return deduped

def extract_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process messages in chronological order, resolving
    cancellations and amendments against the last live
    quote per author.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # last live quote per author  {author: dict}
    last_quote: dict[str, dict] = {}
    rows = []

    for _, row in df.iterrows():
        text   = str(row["text"]).strip()
        author = row["author"]
        msg_id = row["message_id"]

        base = {
            "message_id": msg_id,
            "author":     author,
            "timestamp":  row["timestamp"],
            "text":       text,
        }

        # cancellation
        # full cancellation
        if CANCEL_FULL.match(text):
            ref = last_quote.get(author)
            rows.append({
                **base,
                "instrument_type": ref["instrument_type"] if ref else None,
                "instrument":      ref["instrument"]      if ref else None,
                "status":          "cancelled",
                "ref_message_id":  ref["message_id"]      if ref else None,
            })
            # mark reference as dead
            if ref:
                ref["status"] = "cancelled"
            continue

        # instrument-specific cancellation
        inst_cancel = CANCEL_INSTRUMENT.match(text)
        if inst_cancel:
            hint        = inst_cancel.group(1).strip().lower()
            hint_tokens = set(hint.split())
            ref         = last_quote.get(author)

            # check hint overlaps with last quoted instrument
            if ref and hint_tokens & set(ref["instrument"].lower().split()):
                rows.append({
                    **base,
                    "instrument_type": ref["instrument_type"],
                    "instrument":      ref["instrument"],
                    "status":          "cancelled",
                    "ref_message_id":  ref["message_id"],
                })
                ref["status"] = "cancelled"
            else:
                # no match found, log as unresolved
                rows.append({
                    **base,
                    "instrument_type": None,
                    "instrument":      None,
                    "status":          "cancel_unresolved",
                    "ref_message_id":  None,
                })
            continue

        # amendment
        amend_match = AMEND.match(text)
        if amend_match:
            new_value = amend_match.group(1).strip()
            ref       = last_quote.get(author)

            if ref:
                # mark old quote as amended
                ref["status"] = "amended"
                # inherit instrument from reference, new price from text
                new_instruments = _find_instruments(new_value)
                instrument_type = (
                    new_instruments[0]["instrument_type"]
                    if new_instruments else ref["instrument_type"]
                )
                instrument = (
                    new_instruments[0]["instrument"]
                    if new_instruments else ref["instrument"]
                )
                new_quote = {
                    **base,
                    "instrument_type": instrument_type,
                    "instrument":      instrument,
                    "status":          "live",
                    "ref_message_id":  ref["message_id"],
                }
                last_quote[author] = new_quote
                rows.append(new_quote)
            else:
                # amendment with no prior quote to reference
                rows.append({
                    **base,
                    "instrument_type": None,
                    "instrument":      None,
                    "status":          "amend_unresolved",
                    "ref_message_id":  None,
                })
            continue

        # new quote
        instruments = _find_instruments(text)
        if instruments:
            for inst in instruments:
                new_quote = {
                    **base,
                    "instrument_type": inst["instrument_type"],
                    "instrument":      inst["instrument"],
                    "status":          "live",
                    "ref_message_id":  None,
                }
                # each new quote replaces the last for this author
                last_quote[author] = new_quote
                rows.append(new_quote)
        # else:
        #     rows.append({
        #         **base,
        #         "instrument_type": None,
        #         "instrument":      None,
        #         "status":          "no_instrument",
        #         "ref_message_id":  None,
        #     })

    return pd.DataFrame(rows)

def audit(result_df: pd.DataFrame):
    trading = result_df[result_df["instrument_type"].notna()]
    print(f"Total messages:    {result_df['message_id'].nunique()}")
    print(f"Trading messages:  {trading['message_id'].nunique()}")
    print(f"\nBreakdown by type:")
    print(trading["instrument_type"].value_counts().to_string())
    print(f"\nSample matches:")
    print(trading[["text", "instrument_type", "instrument"]].head(20).to_string())


def split_bid_offer(text: str) -> tuple:
        if not isinstance(text, str) or not text.strip():
            return None, None

        NUM = r'-?[\d]+\.?[\d]*'
        t = re.sub(r'\b\d+x\d+\b|\bx\d+kb\b|\bx\d+\b|\b\d+kb\b|\bin\s+\d+\b', ' ', text, flags=re.I)

        for pat, fn in [
            # two-sided: explicit keywords
            (rf'({NUM})\s+(?:bid|off)\s+({NUM})', lambda m: (float(m.group(1)), float(m.group(2)))),
            # two-sided: separators
            (rf'({NUM})\s+at\s+({NUM})', lambda m: (float(m.group(1)), float(m.group(2)))),
            (rf'({NUM})\s*/\s*({NUM})', lambda m: (float(m.group(1)), float(m.group(2)))),
            (rf'({NUM})\s*[-–—]{{1,2}}\s*(?={NUM})({NUM})', lambda m: (float(m.group(1)), float(m.group(2)))),
            # single-sided: suffixes
            (rf'({NUM})\s*b\b', lambda m: (float(m.group(1)), None)),
            (rf'({NUM})\s*o\b', lambda m: (None, float(m.group(1)))),
            # single-sided: keywords after price
            (rf'({NUM})\s+bid\b', lambda m: (float(m.group(1)), None)),
            (rf'({NUM})\s+offered\b', lambda m: (None, float(m.group(1)))),
            # single-sided: keywords before price
            (rf'\bbid\s+({NUM})', lambda m: (float(m.group(1)), None)),
            (rf'\boff\s+({NUM})', lambda m: (None, float(m.group(1)))),
            (rf'\bsell\s+({NUM})', lambda m: (None, float(m.group(1)))),
            (rf'\bpay\s+({NUM})', lambda m: (float(m.group(1)), None)),
            (rf'\boffered\s+({NUM})', lambda m: (None, float(m.group(1)))),
            # whole integer pairs: 4-12, 25/29, 64/75
            (r'(\d{1,3})\s*[-/]\s*(\d{1,3})\b', lambda m: (float(m.group(1)), float(m.group(2)))),
        ]:
            m = re.search(pat, t, re.I)
            if m:
                return fn(m)

        return None, None

def run():

    # Fix the rows that have commas in the text (replace them with "|" since does not affect the meaning)

    with open( "candidate_sample.csv", "r+") as f:
        lines = f.readlines()

    fixed_file = "".join(fix_text(line) for line in lines)


    messages_df = pd.read_csv(
        io.StringIO(fixed_file),
        names=["message_id", "timestamp", "author", "text"],
        engine="python",
        header=0,
    )

    messages_df.dropna(inplace=True)


    parsed_messages_df = parse_datetime(messages_df)
    cleaned_messages_df = clean_text(parsed_messages_df)
    result = extract_instruments(cleaned_messages_df)
    audit(result)

    result.to_csv("predictions_quotes.csv")

    bid_offer = result["text"].apply(lambda t: pd.Series(split_bid_offer(t)))
    result["bid"] = bid_offer[0]
    result["offer"] = bid_offer[1]

    result = result.rename(columns={
        "author": "source",
        "timestamp": "time_reference",
    })

    result["confidence"] = 1

    final_cols = [
        "message_id", "instrument", "time_reference",
        "bid", "offer", "source", "confidence",
        # remaining columns kept
        "text", "instrument_type", "status", "ref_message_id",
    ]

    result = result[[c for c in final_cols if c in result.columns]]
    result = result[result["instrument_type"].notna()]

    result.to_csv("predictions_quotes.csv", index=False)

if __name__ == "__main__":
    run()