"""
charts.py
Plotly chart builders for the Streamlit dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "Positive": "#22c55e",
    "Neutral":  "#94a3b8",
    "Negative": "#ef4444",
}


def sentiment_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart: average daily sentiment score over time."""
    if df.empty or "published" not in df.columns:
        return _empty_fig("No data available")

    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["published"]).dt.date
    daily = (
        tmp.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "avg_score"})
    )
    daily = daily.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["avg_score"],
        mode="lines+markers",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=6),
        name="Avg Sentiment",
        hovertemplate="%{x}<br>Score: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(
        title="Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Avg Compound Score (–1 to +1)",
        yaxis=dict(range=[-1.05, 1.05]),
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def sentiment_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart: distribution of Positive / Neutral / Negative."""
    if df.empty or "sentiment_label" not in df.columns:
        return _empty_fig("No data available")

    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["label", "count"]

    fig = go.Figure(go.Pie(
        labels=counts["label"],
        values=counts["count"],
        hole=0.45,
        marker_colors=[COLORS.get(l, "#94a3b8") for l in counts["label"]],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} articles<extra></extra>",
    ))
    fig.update_layout(
        title="Sentiment Distribution",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def top_sources_bar(df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Horizontal bar: top news sources by article count."""
    if df.empty or "source" not in df.columns:
        return _empty_fig("No data available")

    top = df["source"].value_counts().head(top_n).reset_index()
    top.columns = ["source", "count"]
    top = top.sort_values("count")

    fig = go.Figure(go.Bar(
        x=top["count"],
        y=top["source"],
        orientation="h",
        marker_color="#6366f1",
        hovertemplate="%{y}: %{x} articles<extra></extra>",
    ))
    fig.update_layout(
        title="Top News Sources",
        xaxis_title="Article count",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=40),
    )
    return fig


def sentiment_score_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of raw sentiment scores."""
    if df.empty or "sentiment_score" not in df.columns:
        return _empty_fig("No data available")

    fig = px.histogram(
        df,
        x="sentiment_score",
        nbins=20,
        color_discrete_sequence=["#6366f1"],
        title="Sentiment Score Distribution",
        labels={"sentiment_score": "Compound Score (–1 to +1)"},
    )
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#94a3b8"))
    fig.update_layout(template="plotly_white", height=280)
    return fig
