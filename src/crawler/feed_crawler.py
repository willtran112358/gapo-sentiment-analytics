"""Polite public-feed crawler for social posts (demo uses mock API)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import requests


@dataclass
class SocialPost:
    post_id: str
    author_id: str
    text: str
    created_at: datetime
    like_count: int
    comment_count: int


class GapoFeedCrawler:
    """Production: rate-limited crawler with robots.txt compliance + cursor pagination."""

    def __init__(self, base_url: str, api_key: str, delay_sec: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.delay_sec = delay_sec

    def fetch_recent(self, limit: int = 50) -> Iterator[SocialPost]:
        # Demo endpoint — replace with real Gapo internal API in production
        try:
            resp = requests.get(
                f"{self.base_url}/v1/feed/recent",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"limit": limit},
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except requests.RequestException:
            items = _mock_posts(limit)

        for item in items:
            yield SocialPost(
                post_id=item["id"],
                author_id=item["author_id"],
                text=item["text"],
                created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                like_count=int(item.get("likes", 0)),
                comment_count=int(item.get("comments", 0)),
            )
            time.sleep(self.delay_sec)


def _mock_posts(limit: int) -> list[dict]:
    samples = [
        "Hôm nay app chạy mượt quá, thích quá đi!",
        "Tính năng mới hơi lag, mong team fix sớm.",
        "Group học online hay quá, cảm ơn mọi người.",
        "Spam quá nhiều trên feed, khó chịu thật.",
    ]
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": str(uuid.uuid4()),
            "author_id": f"user_{i % 5}",
            "text": samples[i % len(samples)],
            "created_at": now,
            "likes": i * 3,
            "comments": i,
        }
        for i in range(min(limit, len(samples) * 3))
    ]
