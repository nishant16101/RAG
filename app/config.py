from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2"
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "data/embeddings/faiss.index"
    chunks_path: str = "data/embeddings/chunks.pkl"
    top_k: int = 3
    chunk_size: int = 150
    chunk_overlap: int = 30

    class Config:
        env_file = ".env"

settings = Settings()