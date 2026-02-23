import pickle
from retrieval.dense_retriever import retrieve as dense_retrieve
from retrieval.sparse_retriever import retrieve as sparse_retrieve


def retrieve(query:str,top_k:int=5,alpha:float=0.6)->list:
    dense_results = dense_retrieve(query,top_k=20)
    sparse_results = sparse_retrieve(query,top_k=20)

    scores = {}
    chunks_map ={}
    for rank,r in enumerate(dense_results):
        cid = id(r["chunk"]["text"])
        scores[cid] = scores.get(cid,0) + alpha*(1/(rank+1))
        chunks_map[cid] = r["chunk"]

    for rank,r in enumerate(sparse_results):
        cid = id(r["chunk"]["text"])
        scores[cid] = scores.get(cid,0) + (1-alpha) * (1/(rank+1))
        chunks_map[cid] = r["chunk"]

    sorted_ids = sorted(scores,key=scores.get,reverse=True)[:top_k]
    return [{"chunk": chunks_map[cid], "score": scores[cid]} for cid in sorted_ids]


if __name__ == "__main__":
    query = "What are the major risk factors for Apple?"
    results = retrieve(query, top_k=5)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Company: {r['chunk']['metadata']['company']} | Year: {r['chunk']['metadata']['year']}")
        print(r['chunk']['text'][:300])