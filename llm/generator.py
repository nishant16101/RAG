import sys
import re
import requests
sys.path.insert(0, ".")

from llm.prompt_templates import SYSTEM_PROMPT, build_prompt
from retrieval.reranker import rerank

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

def answer(question: str, top_k: int = 3) -> dict:
    chunks = rerank(question, top_k=top_k)
    user_prompt = build_prompt(question, chunks)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False
    })
    raw = response.json()["response"]

    confidence = 0.5
    match = re.search(r"CONFIDENCE:\s*([0-9.]+)", raw)
    if match:
        confidence = float(match.group(1))
        raw = raw[:match.start()].strip()

    return {
        "answer": raw,
        "confidence": confidence,
        "sources": [
            {"text": r["chunk"]["text"][:200], "meta": r["chunk"]["metadata"]}
            for r in chunks
        ],
    }

if __name__ == "__main__":
    result = answer("What are the major risk factors for Apple?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nConfidence: {result['confidence']}")
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['meta']['company']} {s['meta']['year']}: {s['text'][:100]}...")