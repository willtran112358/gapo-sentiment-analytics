"""End-to-end: crawl → score → trending aggregate."""

from pathlib import Path

import pandas as pd

from src.analytics.trending import build_trending_frame
from src.crawler.feed_crawler import GapoFeedCrawler
from src.ml.sentiment_model import load_or_train, predict_sentiment

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "sentiment_v1.joblib"


def main() -> None:
    posts = list(GapoFeedCrawler("https://api.demo.gapo.local", "demo").fetch_recent(12))
    texts = [p.text for p in posts]
    meta = pd.DataFrame([{"like_count": p.like_count, "comment_count": p.comment_count} for p in posts])
    pipe = load_or_train(MODEL)
    results = predict_sentiment(pipe, texts)
    trending = build_trending_frame(results, meta)
    print("--- Scored posts ---")
    for r in results:
        print(f"[{r.label:8}] {r.confidence:.2f} | {r.text[:60]}")
    print("\n--- Trending ---")
    print(trending.head(10).to_string(index=False) if not trending.empty else "(empty)")


if __name__ == "__main__":
    main()
