import numpy as np
from collections import deque

_query_embeddings = deque(maxlen=500)

def add_query_embedding(embedding:np.ndarray):
    _query_embeddings.append(embedding)

def detect_drift()->dict:
    if len(_query_embeddings) < 50:
        return {"status":"not enough data"}
    embeddings = np.array(list(_query_embeddings))
    half = len(embeddings)//2
    early = embeddings[:half]
    recent = embeddings[half:]

    early_mean = np.mean(early,axis=0)
    recent_mean= np.mean(recent,axis=0)
    drift_score = float(np.linalg.norm(recent_mean - early_mean))

    return {
        "drift_score": round(drift_score, 4),
        "status": "drift detected" if drift_score > 0.3 else "stable",
        "samples": len(embeddings)
    }

if __name__ == "__main__":
    # Quick test with random embeddings
    for _ in range(100):
        add_query_embedding(np.random.rand(384))
    print(detect_drift())