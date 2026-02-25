from collections import deque
import statistics
_latencies = deque(maxlen=1000)

def track_latency(latency_ms:float):
    _latencies.append(latency_ms)

def get_stats()->dict:
    if not _latencies:
        return {"message": "No requests yet"}
    lat = list(_latencies)
    return {
        "total_requests": len(lat),
        "avg_ms": round(statistics.mean(lat), 2),
        "p50_ms": round(statistics.median(lat), 2),
        "p95_ms": round(sorted(lat)[int(len(lat) * 0.95)], 2),
        "min_ms": round(min(lat), 2),
        "max_ms": round(max(lat), 2),
    }