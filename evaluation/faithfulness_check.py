import requests
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

def check_faithfulness(question:str,answer:str,context:str)->float:
    prompt = f"""Given the context and answer below, rate how faithful the answer is
to ONLY the information in the context. Reply with a single float between 0.0 and 1.0 only.

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
    score = check_faithfulness(
        question="What are Apple's risk factors?",
        answer="Apple faces risks from competition and supply chain issues.",
        context="Apple's business can be affected by competition, supply chain disruptions and regulatory changes."
    )
    print(f"Faithfulness score: {score}")
