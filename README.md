# Gapo Social — Sentiment Analysis & Real-Time Monitor

Open-source **Python ML** platform for Vietnamese social sentiment: auto crawler, TF-IDF classifier, **FastAPI** scoring service, and **Streamlit** dashboard for trending topics and user tone on Gapo Social (2020).

**Role:** ML Engineer · **Year:** 2021

## Tech Stack

| Component | Library / Tool |
|-----------|----------------|
| Crawler | `requests`, rate-limited feed API client |
| NLP | `underthesea` (Vietnamese tokenization ready), scikit-learn |
| Model | TF-IDF + Logistic Regression (`sentiment_v1.joblib`) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit (auto-refresh) |
| Queue (prod) | Redis stream between crawler & scorer |

## Architecture

```mermaid
flowchart TB
    subgraph INGEST["🕷️ Ingestion"]
        CRAWL["Feed Crawler<br/>Gapo API / public feed"]
        Q["Redis Stream<br/>raw posts"]
    end

    subgraph ML["🧠 ML Scoring"]
        API["FastAPI<br/>/v1/sentiment/score"]
        MODEL["Sentiment Model<br/>TF-IDF + LR"]
    end

    subgraph ANALYTICS["📈 Analytics"]
        TREND["Trending Engine<br/>keywords • engagement"]
        STORE["Postgres / Parquet<br/>scored history"]
    end

    subgraph UI["🖥️ Real-Time UI"]
        DASH["Streamlit Dashboard<br/>tone • trending • feed"]
        OPS["Community Ops Team"]
    end

    CRAWL --> Q --> API
    API --> MODEL
    MODEL --> TREND --> STORE
    TREND --> DASH --> OPS
    STORE --> DASH

    style INGEST fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style ML fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style ANALYTICS fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style UI fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

## Quick Start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py

# API
uvicorn src.api.main:app --reload --port 8000

# Dashboard
streamlit run dashboard/app.py
```

## API Example

```bash
curl -X POST http://localhost:8000/v1/sentiment/score \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"app chạy mượt quá\", \"lag kinh khủng\"]}"
```

## Repository Layout

```
gapo-sentiment-analytics/
├── src/crawler/       # Feed ingestion
├── src/ml/            # Train + inference
├── src/analytics/     # Trending aggregation
├── src/api/           # FastAPI service
├── dashboard/         # Streamlit monitor
└── scripts/           # Batch pipeline demo
```

---

*Portfolio reconstruction. No Gapo production credentials or user data.*
