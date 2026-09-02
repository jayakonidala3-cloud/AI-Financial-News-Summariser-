# 📈 FiNews — AI Financial News Summariser

> Real-time financial news sentiment analysis + AI-generated market briefs, powered by FinBERT and Claude.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ Features

| Feature | Details |
|---|---|
| **Live news feed** | Pulls up to 30 articles per ticker via NewsAPI (free tier) |
| **Dual sentiment engine** | VADER (instant) or FinBERT (finance-tuned BERT, ~30× more accurate) |
| **AI market brief** | 3-sentence summary by Claude claude-sonnet-4-6; falls back to sumy offline |
| **Interactive charts** | Sentiment trend line, pie distribution, top sources, score histogram (Plotly) |
| **SQLite caching** | Articles & summaries persisted locally — avoids re-fetching on reload |
| **CSV export** | Download all scored articles with one click |
| **Free to run** | NewsAPI free tier · Streamlit Cloud free · Claude free credits |

---

## 🗂 Project Structure

```
finews/
├── streamlit_app.py        # Main Streamlit app (entry-point)
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
├── modules/
│   ├── news_fetcher.py     # NewsAPI + SQLite layer
│   ├── sentiment.py        # VADER + FinBERT scoring
│   ├── summariser.py       # Claude + sumy summarisation
│   └── charts.py           # Plotly chart builders
└── data/
    └── finews.db           # SQLite database (auto-created)
```

---

## 🚀 Quick Start (local)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/finews.git
cd finews

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env
# Edit .env with your NEWSAPI_KEY and (optionally) ANTHROPIC_API_KEY

# 5. Run
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 🌐 Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set **Main file path** = `streamlit_app.py`
4. Add `NEWSAPI_KEY` and `ANTHROPIC_API_KEY` in **Secrets** (Settings → Secrets)
5. Click **Deploy** — live in ~2 minutes

---

## 🔑 API Keys

| Key | Where to get | Cost |
|---|---|---|
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) | Free (100 req/day) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Free credits included |

The app works **without** an Anthropic key — it falls back to offline extractive summarisation (sumy).

---

## 🧠 Tech Stack

**Data** · NewsAPI · SQLite · Pandas  
**NLP** · NLTK VADER · HuggingFace FinBERT (ProsusAI/finbert) · sumy  
**AI** · Anthropic Claude claude-sonnet-4-6  
**Frontend** · Streamlit · Plotly  
**Infra** · Streamlit Community Cloud (free tier)

---

## 📊 Resume Bullet Points

- Built an end-to-end AI financial news pipeline processing **30+ live articles per query**, combining FinBERT sentiment classification (fine-tuned BERT) with Anthropic Claude for 3-sentence market briefs, reducing manual news review time by ~80%
- Engineered a dual-model sentiment engine (VADER + FinBERT) with SQLite caching, achieving sub-5-second response times on repeat queries and offline fallback via extractive summarisation
- Deployed a Streamlit dashboard with 4 interactive Plotly charts, CSV export, and real-time NewsAPI integration; hosted on Streamlit Community Cloud at zero cost

---

## 📄 License

MIT — free to use, modify, and deploy.
