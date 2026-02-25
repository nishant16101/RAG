import sys
import time
import uuid
sys.path.insert(0, ".")

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.config import settings
from app.dependencies import get_generator
from monitoring.latency_logger import track_latency, get_stats
from evaluation.faithfulness_check import check_faithfulness
from evaluation.hallucination_score import hallucination_score

app = FastAPI(title="Financial RAG API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str
    top_k: int = settings.top_k

class QueryResponse(BaseModel):
    request_id: str
    answer: str
    confidence: float
    sources: list
    latency_ms: float
    evaluation: dict  # live metrics on every request

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.ollama_model}

@app.get("/metrics")
def metrics():
    return get_stats()

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, generator=Depends(get_generator)):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    # Generate answer
    result = generator(req.question, top_k=req.top_k)

    # Build context from sources for evaluation
    context = " ".join([s["text"] for s in result["sources"]])

    # Run evaluation on the fly
    faithfulness = check_faithfulness(req.question, result["answer"], context)
    hallucination = hallucination_score(result["answer"], context)

    latency_ms = (time.perf_counter() - start) * 1000
    track_latency(latency_ms)

    return QueryResponse(
        request_id=request_id,
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["sources"],
        latency_ms=round(latency_ms, 2),
        evaluation={
            "faithfulness_score": faithfulness,       # 1.0 = fully grounded
            "hallucination_score": hallucination,     # 0.0 = no hallucination
            "confidence": result["confidence"],       # model self-reported
        }
    )