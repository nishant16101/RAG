import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROCESSED_DIR = "data/processed"
EMBEDDINGS_DIR = "data/embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"


def embed_all():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []

    # Load all processed chunk files
    for filename in os.listdir(PROCESSED_DIR):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(PROCESSED_DIR, filename), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {filename}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings...")

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True   # IMPORTANT for cosine similarity
    )

    embeddings = np.array(embeddings).astype("float32")

    # Save embeddings
    np.save(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"), embeddings)

    # Save metadata
    with open(os.path.join(EMBEDDINGS_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nSaved {len(all_chunks)} embeddings successfully.")


if __name__ == "__main__":
    embed_all()