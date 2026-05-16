"""Vietnamese sentiment classifier — TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ["negative", "neutral", "positive"]


@dataclass
class SentimentResult:
    text: str
    label: str
    confidence: float


def default_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "tuyệt vời quá",
                "rất hài lòng",
                "ổn",
                "bình thường",
                "tệ quá",
                "thất vọng",
                "lag kinh khủng",
                "cảm ơn team",
                "spam nhiều",
                "hay quá",
            ],
            "label": [
                "positive",
                "positive",
                "neutral",
                "neutral",
                "negative",
                "negative",
                "negative",
                "positive",
                "negative",
                "positive",
            ],
        }
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000)),
            ("clf", LogisticRegression(max_iter=500, multi_class="ovr")),
        ]
    )


def train_and_save(model_path: Path) -> Pipeline:
    df = default_training_data()
    pipe = build_pipeline()
    pipe.fit(df["text"], df["label"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    return pipe


def load_or_train(model_path: Path) -> Pipeline:
    if model_path.exists():
        return joblib.load(model_path)
    return train_and_save(model_path)


def predict_sentiment(pipe: Pipeline, texts: list[str]) -> list[SentimentResult]:
    proba = pipe.predict_proba(texts)
    labels = pipe.predict(texts)
    classes = list(pipe.named_steps["clf"].classes_)
    results: list[SentimentResult] = []
    for i, text in enumerate(texts):
        idx = classes.index(labels[i])
        results.append(
            SentimentResult(text=text, label=labels[i], confidence=float(proba[i][idx]))
        )
    return results
