"""Aggregate trending topics and sentiment mix from scored posts."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from src.ml.sentiment_model import SentimentResult

STOPWORDS = {"và", "là", "của", "có", "được", "một", "cho", "trên", "với", "quá"}


def extract_keywords(text: str, top_n: int = 3) -> list[str]:
    tokens = re.findall(r"[\wÀ-ỹ]+", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return [w for w, _ in Counter(filtered).most_common(top_n)]


def build_trending_frame(results: list[SentimentResult], posts_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r, meta in zip(results, posts_meta.to_dict("records")):
        for kw in extract_keywords(r.text):
            rows.append(
                {
                    "keyword": kw,
                    "sentiment": r.label,
                    "confidence": r.confidence,
                    "engagement": meta.get("like_count", 0) + meta.get("comment_count", 0),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (
        df.groupby(["keyword", "sentiment"])
        .agg(mentions=("confidence", "count"), avg_confidence=("confidence", "mean"), engagement=("engagement", "sum"))
        .reset_index()
    )
    return agg.sort_values(["mentions", "engagement"], ascending=False)
