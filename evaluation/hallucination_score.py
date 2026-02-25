import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

def hallucination_score(answer: str, context: str) -> float:
    """Returns 0.0 (no hallucination) to 1.0 (fully hallucinated)."""
    prompt = f"""Does the answer contain any information NOT present in the context?
Reply with a single float: 0.0 means no hallucination, 1.0 means completely hallucinated.

Context: {context[:500]}
Answer: {answer}

Score:"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    try:
        return float(response.json()["response"].strip())
    except:
        return 0.5

if __name__ == "__main__":
    score = hallucination_score(
        answer="Apple has 200,000 employees and was founded in 1990.",
        context="Apple was founded in 1976 and designs consumer electronics."
    )
    print(f"Hallucination score: {score}")