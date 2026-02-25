from functools import lru_cache
from retrieval import reranker
from llm.generator import answer

@lru_cache(maxsize= 1)
def get_retriever():
    from retrieval.hybrid_retriever import retrieve
    return retrieve

@lru_cache(maxsize=1)
def get_generator():
    return answer