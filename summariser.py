"""
summariser.py
AI-powered article summarisation.
Primary:  Anthropic Claude (claude-sonnet-4-6)
Fallback: extractive summarisation via sumy (works offline, no API key needed)
"""

import logging
import os
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extractive fallback (sumy)
# ---------------------------------------------------------------------------

def _extractive_summary(texts: list[str], sentence_count: int = 3) -> str:
    """Use sumy LSA summariser as an offline fallback."""
    try:
        import nltk
        for resource in ("punkt", "punkt_tab", "stopwords"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer

        combined = " ".join(texts)[:8000]
        parser = PlaintextParser.from_string(combined, Tokenizer("english"))
        summarizer = LsaSummarizer()
        sentences = summarizer(parser.document, sentence_count)
        return " ".join(str(s) for s in sentences)
    except ImportError:
        # Last resort: first sentences
        words = " ".join(texts)[:600]
        return words + "…"
    except Exception as exc:
        logger.warning("sumy failed: %s", exc)
        return " ".join(texts[0:1])[:400] + "…"


# ---------------------------------------------------------------------------
# Claude summariser
# ---------------------------------------------------------------------------

def _claude_summary(ticker: str, texts: list[str], api_key: str) -> Optional[str]:
    """Call Anthropic Claude to produce a financial market summary."""
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed; falling back to extractive.")
        return None

    combined = "\n\n---\n\n".join(t[:600] for t in texts[:10])
    n = len(texts)

    prompt = (
        f"You are a senior financial analyst writing for Bloomberg. "
        f"Summarise these {n} news articles about {ticker} in exactly 3 concise sentences. "
        f"Sentence 1: what happened. Sentence 2: market implications / risks. "
        f"Sentence 3: short-term outlook. Be specific and data-driven.\n\n"
        f"Articles:\n{combined}"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as exc:
        logger.error("Claude API error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def summarise(
    df: pd.DataFrame,
    ticker: str,
    anthropic_api_key: Optional[str] = None,
    force_regenerate: bool = False,
    text_col: str = "content",
) -> dict:
    """
    Generate a market summary for a ticker.

    Returns:
        {
            "summary": str,
            "model": "claude" | "extractive",
            "cached": bool,
        }
    """
    from modules.news_fetcher import get_cached_summary, save_summary

    # 1. Try cache
    if not force_regenerate:
        cached = get_cached_summary(ticker)
        if cached:
            return {
                "summary": cached["summary"],
                "model": cached["model"],
                "cached": True,
            }

    texts = df[text_col].dropna().tolist()
    if not texts:
        return {
            "summary": "No articles available to summarise.",
            "model": "none",
            "cached": False,
        }

    # 2. Try Claude
    api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
    summary = None
    model_used = "extractive"

    if api_key:
        summary = _claude_summary(ticker, texts, api_key)
        if summary:
            model_used = "claude-sonnet-4-6"

    # 3. Extractive fallback
    if not summary:
        summary = _extractive_summary(texts)
        model_used = "extractive (sumy)"

    # 4. Cache result
    article_ids = df["id"].tolist() if "id" in df.columns else []
    save_summary(ticker, summary, model_used, [str(i) for i in article_ids])

    return {"summary": summary, "model": model_used, "cached": False}
