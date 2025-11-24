from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import numpy as np
from datetime import datetime

# Initialisation
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API pour analyser le sentiment des commentaires YouTube",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle
model = joblib.load("models/sentiment_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

# Modèles Pydantic
class CommentBatch(BaseModel):
    comments: List[str] = Field(..., min_items=1)

class SentimentPrediction(BaseModel):
    text: str
    sentiment: str
    sentiment_score: int
    confidence: float

class BatchPredictionResponse(BaseModel):
    predictions: List[SentimentPrediction]
    statistics: Dict
    timestamp: str

def label_to_sentiment(label: int) -> str:
    mapping = {-1: "negative", 0: "neutral", 1: "positive"}
    return mapping.get(label, "unknown")

@app.get("/")
async def root():
    return {
        "message": "YouTube Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: CommentBatch):
    try:
        valid_comments = [c.strip() for c in batch.comments if c.strip()]
        
        if not valid_comments:
            raise HTTPException(status_code=400, detail="Aucun commentaire valide")
        
        X = vectorizer.transform(valid_comments)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        results = []
        for text, pred, proba in zip(valid_comments, predictions, probabilities):
            results.append(SentimentPrediction(
                text=text[:200],
                sentiment=label_to_sentiment(int(pred)),
                sentiment_score=int(pred),
                confidence=round(float(np.max(proba)), 4)
            ))
        
        sentiment_counts = {
            "positive": int(np.sum(predictions == 1)),
            "neutral": int(np.sum(predictions == 0)),
            "negative": int(np.sum(predictions == -1))
        }
        
        total = len(predictions)
        sentiment_percentages = {
            "positive": round(sentiment_counts["positive"] / total * 100, 2),
            "neutral": round(sentiment_counts["neutral"] / total * 100, 2),
            "negative": round(sentiment_counts["negative"] / total * 100, 2)
        }
        
        statistics = {
            "total_comments": total,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "average_confidence": round(float(np.mean([r.confidence for r in results])), 4)
        }
        
        return BatchPredictionResponse(
            predictions=results,
            statistics=statistics,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")