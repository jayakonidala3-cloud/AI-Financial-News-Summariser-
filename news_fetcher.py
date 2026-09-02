"""
news_fetcher.py
Fetches financial news from NewsAPI and stores results in SQLite.
"""

import os
import sqlite3
import hashlib
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "finews.db")
NEWSAPI_BASE = "https://newsapi.org/v2/everything"

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id          TEXT PRIMARY KEY,
            ticker      TEXT NOT NULL,
            title       TEXT,
            description TEXT,
            content     TEXT,
            source      TEXT,
            url         TEXT,
            published   TEXT,
            fetched_at  TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            sentiment_model TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            summary     TEXT,
            model_used  TEXT,
            created_at  TEXT,
            article_ids TEXT
        )
    """)
    conn.commit()
    conn.close()


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_news(
    ticker: str,
    api_key: str,
    days_back: int = 7,
    page_size: int = 30,
) -> pd.DataFrame:
    """
    Fetch news articles for a stock ticker from NewsAPI.
    Returns a DataFrame and persists new articles to SQLite.
    """
    init_db()

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q": ticker,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("NewsAPI request failed: %s", exc)
        return _load_from_db(ticker)

    articles = data.get("articles", [])
    if not articles:
        logger.warning("No articles returned for %s", ticker)
        return _load_from_db(ticker)

    rows = []
    now_str = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for art in articles:
        url = art.get("url", "")
        if not url:
            continue
        art_id = _article_id(url)
        title = art.get("title") or ""
        description = art.get("description") or ""
        content = art.get("content") or description
        full_text = f"{title}. {description}"

        row = {
            "id": art_id,
            "ticker": ticker.upper(),
            "title": title,
            "description": description,
            "content": full_text,
            "source": art.get("source", {}).get("name", "Unknown"),
            "url": url,
            "published": art.get("publishedAt", ""),
            "fetched_at": now_str,
            "sentiment_label": None,
            "sentiment_score": None,
            "sentiment_model": None,
        }
        rows.append(row)

        cur.execute("""
            INSERT OR IGNORE INTO articles
            (id,ticker,title,description,content,source,url,published,fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            art_id, row["ticker"], row["title"], row["description"],
            row["content"], row["source"], row["url"],
            row["published"], now_str,
        ))

    conn.commit()
    conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return _load_from_db(ticker)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    return df.sort_values("published", ascending=False).reset_index(drop=True)


def _load_from_db(ticker: str) -> pd.DataFrame:
    """Load existing articles for ticker from SQLite (fallback)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM articles WHERE ticker=? ORDER BY published DESC",
        conn, params=(ticker.upper(),)
    )
    conn.close()
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    return df


def update_sentiment_in_db(
    article_id: str,
    label: str,
    score: float,
    model: str,
) -> None:
    """Persist sentiment results back to the articles table."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE articles
        SET sentiment_label=?, sentiment_score=?, sentiment_model=?
        WHERE id=?
    """, (label, score, model, article_id))
    conn.commit()
    conn.close()


def save_summary(ticker: str, summary: str, model_used: str, article_ids: list) -> None:
    """Cache a generated summary in SQLite."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO summaries (ticker, summary, model_used, created_at, article_ids)
        VALUES (?,?,?,?,?)
    """, (ticker.upper(), summary, model_used,
          datetime.utcnow().isoformat(), ",".join(article_ids)))
    conn.commit()
    conn.close()


def get_cached_summary(ticker: str) -> Optional[dict]:
    """Return the most recent cached summary for a ticker (if < 1 hr old)."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT summary, model_used, created_at FROM summaries
        WHERE ticker=?
        ORDER BY created_at DESC LIMIT 1
    """, (ticker.upper(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    created = datetime.fromisoformat(row[2])
    if (datetime.utcnow() - created).total_seconds() < 3600:
        return {"summary": row[0], "model": row[1], "created_at": row[2]}
    return None
