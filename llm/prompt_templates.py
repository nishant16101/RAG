SYSTEM_PROMPT = """You are a financial analyst assistant.
Answer strictly using the provided context only.
If answer not found, say: "Information not found in provided documents."
Always cite the source company and year.
Do NOT give investment advice."""

def build_prompt(question: str, chunks: list) -> str:
    context = ""
    for i, r in enumerate(chunks):
        meta = r["chunk"]["metadata"]
        context += f"[Source {i+1}] Company: {meta['company']} | Year: {meta['year']}\n"
        context += r["chunk"]["text"][:300] + "\n\n"  # was sending full chunk, now limit to 300 chars

    return f"""Context:
{context}
Question: {question}

Answer with source citations e.g. [Source 1].
At the end write: CONFIDENCE: 0.x"""