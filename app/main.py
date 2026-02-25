import sys
import time 
import uuid
sys.path.insert(0, ".")

from fastapi import FastAPI,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.config import settings
from app.dependencies import get_retriever,get_generator  

from monitoring.latency_logger import track_latency,get_stats

app = FastAPI(title="Financial RAG",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

class QueryRequest(BaseModel):
    question:str
    top_k:int = settings.top_k

class QueryResponse(BaseModel):
    request_id:str
    answer:str
    confidence:float
    sources:list
    latency_ms:float


@app.get("/health")
def health():
    return {"status":"ok","model":settings.ollama_model}

@app.get("/metrics")
def metrics():
    return get_stats()

@app.post("/query",response_model=QueryResponse)
def query(req:QueryRequest,generator=Depends(get_generator)):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    result = generator(req.question,top_k=req.top_k)
    latency_ms = (time.perf_counter() - start) * 1000
    track_latency(latency_ms)

    return QueryResponse(
        request_id = request_id,
        answer = result["answer"],
        confidence = result["confidence"],
        sources = result["sources"],
        latency_ms = round(latency_ms,2)
    )


