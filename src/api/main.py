# main.py

import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

import joblib

# ==============================
# Configuration & Constants
# ==============================

# Resolve project root: src/api/main.py → go up 2 levels
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.joblib")


# ==============================
# Pydantic Models (Validation)
# ==============================

class CommentBatch(BaseModel):
    comments: List[str] = Field(..., min_items=1, description="Liste non vide de commentaires")

    @validator("comments")
    def strip_and_validate_comments(cls, v):
        stripped = [text.strip() for text in v]
        if not any(stripped):
            raise ValueError("Au moins un commentaire doit être non vide.")
        return stripped


class SentimentPrediction(BaseModel):
    text: str
    sentiment: str
    sentiment_score: int
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: List[SentimentPrediction]
    statistics: Dict[str, Any]
    timestamp: str


# ==============================
# FastAPI App Setup
# ==============================

app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API pour analyser le sentiment des commentaires YouTube",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & vectorizer
model = None
vectorizer = None


# ==============================
# Utility Functions
# ==============================

def label_to_sentiment(label: int) -> str:
    return {-1: "negative", 0: "neutral", 1: "positive"}.get(label, "unknown")


# ==============================
# Startup Event: Load Model
# ==============================

@app.on_event("startup")
async def load_model():
    global model, vectorizer
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"Vectoriseur introuvable : {VECTORIZER_PATH}")

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print(" Modèle et vectoriseur chargés avec succès.")
    except Exception as e:
        error_msg = f" Échec du chargement du modèle : {e}"
        print(error_msg)
        raise RuntimeError(error_msg)


# ==============================
# Endpoints
# ==============================

@app.get("/")
async def root():
    return {
        "message": "YouTube Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict_batch"
    }


@app.get("/health")
async def health_check():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: CommentBatch):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        valid_comments = [c for c in batch.comments if c.strip()]
        if not valid_comments:
            raise HTTPException(status_code=400, detail="Aucun commentaire valide.")

        X_tfidf = vectorizer.transform(valid_comments)
        predictions = model.predict(X_tfidf)
        probabilities = model.predict_proba(X_tfidf)

        results = []
        for text, pred, proba in zip(valid_comments, predictions, probabilities):
            results.append(
                SentimentPrediction(
                    text=text[:200],
                    sentiment=label_to_sentiment(int(pred)),
                    sentiment_score=int(pred),
                    confidence=round(float(np.max(proba)), 4)
                )
            )

        total = len(predictions)
        pos = int(np.sum(predictions == 1))
        neu = int(np.sum(predictions == 0))
        neg = int(np.sum(predictions == -1))

        stats = {
            "total_comments": total,
            "sentiment_counts": {"positive": pos, "neutral": neu, "negative": neg},
            "sentiment_percentages": {
                "positive": round(pos / total * 100, 2),
                "neutral": round(neu / total * 100, 2),
                "negative": round(neg / total * 100, 2),
            },
            "average_confidence": round(float(np.mean([r.confidence for r in results])), 4)
        }

        return BatchPredictionResponse(
            predictions=results,
            statistics=stats,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur interne dans /predict_batch : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}")


# ==============================
# Local Development Entry Point
# ==============================

if __name__ == "__main__":
    import uvicorn
    # Use "src.api.main:app" if running from project root, but since this is __main__,
    # "main:app" works only if you run from src/api/
    # Safer: run via CLI from project root → `uvicorn src.api.main:app --reload`
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)