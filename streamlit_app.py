"""
streamlit_app.py  ← root entry-point for Streamlit Community Cloud
AI Financial News Summariser — full dashboard
"""

import os
import sys
import pandas as pd
import streamlit as st

# Make sure `modules/` is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from modules.news_fetcher import fetch_news, init_db
from modules.sentiment import run_sentiment
from modules.summariser import summarise
from modules import charts

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FiNews — AI Financial News Summariser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Card style for article items */
.article-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.article-card h4 { margin: 0 0 6px 0; font-size: 14px; color: #1e293b; }
.article-card .meta { font-size: 12px; color: #64748b; margin-bottom: 6px; }
.badge-pos  { background:#dcfce7; color:#15803d; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; }
.badge-neg  { background:#fee2e2; color:#b91c1c; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; }
.badge-neu  { background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; }
.summary-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border-left: 4px solid #6366f1;
    padding: 16px 20px;
    border-radius: 0 10px 10px 0;
    font-size: 15px;
    line-height: 1.7;
    color: #1e293b;
    margin: 8px 0 16px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — inputs & API keys
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("FiNews")
    st.caption("AI-powered financial news intelligence")

    st.divider()

    ticker = st.text_input("Stock / Company ticker", value="AAPL",
                           help="e.g. AAPL, TSLA, NVDA, MSFT, RELIANCE").upper().strip()
    days_back = st.slider("Look-back period (days)", 1, 30, 7)
    sentiment_model = st.selectbox("Sentiment model", ["VADER (fast)", "FinBERT (accurate)"],
                                   help="FinBERT is slower but far more accurate for financial text.")

    st.divider()
    st.markdown("**🔑 API Keys**")
    newsapi_key = st.text_input("NewsAPI key", type="password",
                                value=os.getenv("NEWSAPI_KEY", ""),
                                help="Get a free key at newsapi.org")
    anthropic_key = st.text_input("Anthropic API key (optional)", type="password",
                                  value=os.getenv("ANTHROPIC_API_KEY", ""),
                                  help="Leave blank to use offline extractive summarisation")

    st.divider()
    analyse_btn = st.button("🚀  Fetch & Analyse", use_container_width=True, type="primary")
    regenerate = st.checkbox("Force regenerate summary", value=False)

    st.divider()
    st.markdown(
        "<small>Built with NewsAPI · NLTK VADER · FinBERT · Claude · Streamlit</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key in ("df", "summary_result", "analysed_ticker"):
    if key not in st.session_state:
        st.session_state[key] = None


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📈 AI Financial News Summariser")
st.markdown(
    "Enter a stock ticker in the sidebar, hit **Fetch & Analyse**, "
    "and get sentiment-scored news + an AI-written market brief in seconds."
)

if analyse_btn:
    if not newsapi_key:
        st.error("⚠️ Please enter your NewsAPI key in the sidebar (free at newsapi.org).")
        st.stop()
    if not ticker:
        st.error("⚠️ Please enter a valid ticker symbol.")
        st.stop()

    model_flag = "finbert" if sentiment_model.startswith("FinBERT") else "vader"

    with st.spinner(f"Fetching news for **{ticker}**…"):
        df_raw = fetch_news(ticker, newsapi_key, days_back=days_back)

    if df_raw.empty:
        st.warning("No articles found. Try a different ticker or extend the look-back period.")
        st.stop()

    with st.spinner(f"Running {sentiment_model} sentiment analysis…"):
        df = run_sentiment(df_raw, model=model_flag)

    with st.spinner("Generating AI summary…"):
        summary_result = summarise(
            df, ticker,
            anthropic_api_key=anthropic_key or None,
            force_regenerate=regenerate,
        )

    st.session_state["df"] = df
    st.session_state["summary_result"] = summary_result
    st.session_state["analysed_ticker"] = ticker


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
df: pd.DataFrame = st.session_state["df"]
summary_result: dict = st.session_state["summary_result"]
aticker: str = st.session_state["analysed_ticker"]

if df is not None and not df.empty:

    # ── AI Summary ──────────────────────────────────────────────────────────
    st.subheader(f"🤖 AI Market Brief — {aticker}")
    model_label = summary_result.get("model", "")
    cached_tag = " _(cached)_" if summary_result.get("cached") else ""
    st.caption(f"Generated by **{model_label}**{cached_tag}")
    st.markdown(
        f'<div class="summary-box">{summary_result.get("summary", "")}</div>',
        unsafe_allow_html=True,
    )

    # ── Metric cards ────────────────────────────────────────────────────────
    total = len(df)
    pos = (df["sentiment_label"] == "Positive").sum()
    neg = (df["sentiment_label"] == "Negative").sum()
    neu = (df["sentiment_label"] == "Neutral").sum()
    avg_score = df["sentiment_score"].mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Articles", total)
    m2.metric("Positive 🟢", pos)
    m3.metric("Neutral ⚪", neu)
    m4.metric("Negative 🔴", neg)
    m5.metric("Avg Score", f"{avg_score:+.3f}")

    st.divider()

    # ── Charts ──────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(charts.sentiment_trend(df), use_container_width=True)
    with col_r:
        st.plotly_chart(charts.sentiment_pie(df), use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.plotly_chart(charts.top_sources_bar(df), use_container_width=True)
    with col_r2:
        st.plotly_chart(charts.sentiment_score_histogram(df), use_container_width=True)

    st.divider()

    # ── Article cards ───────────────────────────────────────────────────────
    st.subheader("📰 Article Feed")

    label_filter = st.multiselect(
        "Filter by sentiment",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )
    filtered_df = df[df["sentiment_label"].isin(label_filter)]

    for _, row in filtered_df.iterrows():
        label = row.get("sentiment_label", "Neutral")
        badge_cls = {"Positive": "badge-pos", "Negative": "badge-neg"}.get(label, "badge-neu")
        score = row.get("sentiment_score", 0.0)
        pub = str(row.get("published", ""))[:16].replace("T", " ")
        source = row.get("source", "Unknown")
        title = row.get("title", "(no title)")
        url = row.get("url", "#")
        description = row.get("description", "")

        st.markdown(f"""
        <div class="article-card">
            <h4><a href="{url}" target="_blank" style="text-decoration:none;color:inherit;">{title}</a></h4>
            <div class="meta">{source} &nbsp;·&nbsp; {pub} UTC</div>
            <span class="{badge_cls}">{label}</span>
            <span style="font-size:11px;color:#94a3b8;margin-left:8px;">score {score:+.3f}</span>
            <p style="font-size:13px;color:#475569;margin-top:8px;">{description[:200]}{'…' if len(description)>200 else ''}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Download ─────────────────────────────────────────────────────────────
    csv = filtered_df.drop(columns=["id"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download results as CSV",
        data=csv,
        file_name=f"finews_{aticker}_{pd.Timestamp.now().date()}.csv",
        mime="text/csv",
    )

else:
    # Welcome state
    st.info("👈 Enter a ticker and click **Fetch & Analyse** to get started.")

    st.markdown("""
    ### What this app does
    1. **Fetches** up to 30 recent news articles from NewsAPI for any stock ticker
    2. **Scores** each article as Positive / Neutral / Negative using VADER or FinBERT
    3. **Summarises** the news landscape with a 3-sentence AI market brief (Claude or offline extractive)
    4. **Visualises** sentiment trends, distribution, and top sources with interactive Plotly charts
    5. **Exports** results to CSV for further analysis

    ### Quick start
    - Get a **free NewsAPI key** at [newsapi.org](https://newsapi.org)
    - Optionally add an **Anthropic API key** for Claude-powered summaries
    - Try tickers: `AAPL`, `TSLA`, `NVDA`, `MSFT`, `AMZN`, `RELIANCE`, `INFY`
    """)
