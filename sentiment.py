"""
sentiment.py
Dual-engine sentiment analysis: VADER (fast) + FinBERT (accurate).
Falls back gracefully when dependencies aren't available.
"""

import logging
import pandas as pd
from typing import Literal

logger = logging.getLogger(__name__)

SentimentModel = Literal["vader", "finbert"]


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------

def _get_vader():
    try:
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("NLTK not installed; VADER unavailable.")
        return None


def score_vader(text: str) -> dict:
    """Return VADER compound score and label."""
    sia = _get_vader()
    if sia is None:
        return {"label": "Neutral", "score": 0.0, "model": "vader"}
    compound = sia.polarity_scores(text)["compound"]
    label = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")
    return {"label": label, "score": round(compound, 4), "model": "vader"}


# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline
        logger.info("Loading FinBERT model (first load may take ~30s)…")
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        return _finbert_pipeline
    except Exception as exc:
        logger.warning("FinBERT unavailable (%s); falling back to VADER.", exc)
        return None


def score_finbert(text: str) -> dict:
    """Return FinBERT label and score. Falls back to VADER on error."""
    pipe = _get_finbert()
    if pipe is None:
        return score_vader(text)
    try:
        result = pipe(text[:512])[0]
        raw_label = result["label"].lower()
        score = result["score"]
        label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
        label = label_map.get(raw_label, "Neutral")
        # Convert to –1..+1 scale for consistency with VADER
        signed_score = score if label == "Positive" else (-score if label == "Negative" else 0.0)
        return {"label": label, "score": round(signed_score, 4), "model": "finbert"}
    except Exception as exc:
        logger.warning("FinBERT inference failed: %s", exc)
        return score_vader(text)


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def run_sentiment(
    df: pd.DataFrame,
    model: SentimentModel = "vader",
    text_col: str = "content",
) -> pd.DataFrame:
    """
    Adds sentiment_label, sentiment_score, sentiment_model columns to df.
    Persists results to SQLite if 'id' column is present.
    """
    from modules.news_fetcher import update_sentiment_in_db  # lazy import

    scorer = score_finbert if model == "finbert" else score_vader

    labels, scores, models = [], [], []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        result = scorer(text)
        labels.append(result["label"])
        scores.append(result["score"])
        models.append(result["model"])

        # Persist
        if "id" in row and pd.notna(row["id"]):
            try:
                update_sentiment_in_db(
                    row["id"], result["label"], result["score"], result["model"]
                )
            except Exception:
                pass

    df = df.copy()
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    df["sentiment_model"] = models
    return df


# ---------------------------------------------------------------------------
# Comparison helper (for interview talking point)
# ---------------------------------------------------------------------------

def compare_models(texts: list[str]) -> pd.DataFrame:
    """Run both VADER and FinBERT on a list of texts and return a comparison DF."""
    rows = []
    for t in texts[:20]:  # cap at 20
        v = score_vader(t)
        f = score_finbert(t)
        rows.append({
            "text_preview": t[:80],
            "vader_label": v["label"],
            "vader_score": v["score"],
            "finbert_label": f["label"],
            "finbert_score": f["score"],
            "agree": v["label"] == f["label"],
        })
    return pd.DataFrame(rows)
