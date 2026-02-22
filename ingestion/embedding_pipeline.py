import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROCESSED_DIR = "data/processed"
EMBEDDINGS_DIR = "data/embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

def embed_all():
    os.makedirs(EMBEDDINGS_DIR,exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    all_chunks =[]
    for filename in os.listdir(PROCESSED_DIR):
        if not filename.endswith('.json'):
            continue

        with open(f"{PROCESSED_DIR}/{filename}") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        print(f"loaded {len(chunks)} chunks from {filename}")


    print(f"total_chunks:{len(all_chunks)}")
    print("Embedding")
    text = [c["text"] for c in all_chunks]
    embeddings = model.encode(text,batch_size=64)

    np.save(f"{EMBEDDINGS_DIR}/embeddings.npy",embeddings)
    with open(f"{EMBEDDINGS_DIR}/chunks.pkl","wb") as f:
        pickle.dump(all_chunks,f)

    print(f"Done Saved {len(all_chunks)}")


if __name__== "__main__":
    embed_all()   