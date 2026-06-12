#%%

# I used google colab to run BERT
#!pip install transformers torch pandas numpy matplotlib scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from scipy.signal import savgol_filter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# Use FinBERT to classify sentiments

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CSV_PATH = "predictions_quotes.csv"      # ← your input file
OUTPUT_CSV = "predictions_sentiment.csv"

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_TOKEN_LEN = 512                     # FinBERT hard limit


def load_finbert():
    print(f"Loading {FINBERT_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=MAX_TOKEN_LEN,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    return clf


# label mapping
LABEL_MAP = {
    "positive": "bullish",
    "negative": "bearish",
    "neutral": "neutral",
}


def classify_batch(clf, texts: list[str]) -> list[dict]:
    results = clf(texts)
    return [
        {
            "sentiment": LABEL_MAP.get(r["label"].lower(), "neutral"),
            "confidence": round(r["score"], 4),
        }
        for r in results
    ]


def classify_dataframe(df: pd.DataFrame, clf) -> pd.DataFrame:
    df = df.copy()

    df["sentiment"]  = "neutral"
    df["confidence"] = 1.0

    texts = df["text"].tolist()
    results = classify_batch(clf, texts)

    df["sentiment"] = [r["sentiment"]  for r in results]
    df["confidence"] = [r["confidence"] for r in results]

    return df


def sentiment_score(row) -> float:
    """
    Convert label + confidence to a scalar in [-1, +1].
    bullish  → +confidence
    bearish  → -confidence
    neutral  →  0
    """
    if row["sentiment"] == "bullish":
        return row["confidence"]
    if row["sentiment"] == "bearish":
        return -row["confidence"]
    return 0.0


def score_messages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df.apply(sentiment_score, axis=1)
    return df



def regime_shifts(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a table of dates where the regime changed."""
    shifts = daily[daily["regime"] != daily["regime"].shift()].copy()
    shifts = shifts[["date", "regime", "mean_score", "msg_count"]]
    shifts.columns = ["date", "new_regime", "score_on_day", "messages"]
    return shifts.reset_index(drop=True)



def run(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path, parse_dates=["time_reference"])
    print(f"Loaded {len(df)} rows from {csv_path}")

    clf = load_finbert()
    classified = classify_dataframe(df, clf)

    classified = score_messages(classified)

    classified.to_csv(OUTPUT_CSV, index=False)
    print(f"Per-message sentiment saved: {OUTPUT_CSV}")
    print(classified[["text", "sentiment", "confidence", "score"]].head(20).to_string())

    return classified


if __name__ == "__main__":
    classified = run()
