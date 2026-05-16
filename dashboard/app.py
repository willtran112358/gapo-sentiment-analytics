"""Streamlit real-time monitor — trending topics & sentiment mix."""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics.trending import build_trending_frame
from src.crawler.feed_crawler import GapoFeedCrawler
from src.ml.sentiment_model import load_or_train, predict_sentiment

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment_v1.joblib"

st.set_page_config(page_title="Gapo Sentiment Monitor", layout="wide")
st.title("Gapo Social — Sentiment & Trending Monitor")

refresh = st.sidebar.slider("Auto-refresh (sec)", 5, 60, 15)
if st.sidebar.button("Refresh now") or True:
    crawler = GapoFeedCrawler(base_url="https://api.demo.gapo.local", api_key="demo")
    posts = list(crawler.fetch_recent(limit=20))
    texts = [p.text for p in posts]
    meta = pd.DataFrame(
        [
            {
                "post_id": p.post_id,
                "like_count": p.like_count,
                "comment_count": p.comment_count,
            }
            for p in posts
        ]
    )
    pipe = load_or_train(MODEL_PATH)
    results = predict_sentiment(pipe, texts)
    trending = build_trending_frame(results, meta)

    col1, col2, col3 = st.columns(3)
    labels = [r.label for r in results]
    col1.metric("Posts scored", len(results))
    col2.metric("Positive %", f"{100 * labels.count('positive') / max(len(labels), 1):.0f}%")
    col3.metric("Negative %", f"{100 * labels.count('negative') / max(len(labels), 1):.0f}%")

    st.subheader("Sentiment distribution")
    st.bar_chart(pd.Series(labels).value_counts())

    st.subheader("Trending keywords")
    if not trending.empty:
        st.dataframe(trending.head(15), use_container_width=True)
    else:
        st.info("No trending data yet.")

    st.subheader("Latest posts")
    st.dataframe(
        pd.DataFrame(
            [{"text": r.text, "sentiment": r.label, "confidence": round(r.confidence, 3)} for r in results]
        ),
        use_container_width=True,
    )

time.sleep(refresh)
