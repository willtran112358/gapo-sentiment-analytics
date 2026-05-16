"""FastAPI scoring service for real-time sentiment pipeline."""

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.ml.sentiment_model import load_or_train, predict_sentiment

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "sentiment_v1.joblib"
app = FastAPI(title="Gapo Sentiment API", version="1.0.0")
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = load_or_train(MODEL_PATH)
    return _pipe


class ScoreRequest(BaseModel):
    texts: list[str]


class ScoreResponse(BaseModel):
    results: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/sentiment/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    results = predict_sentiment(get_pipe(), req.texts)
    return ScoreResponse(
        results=[{"text": r.text, "label": r.label, "confidence": r.confidence} for r in results]
    )
