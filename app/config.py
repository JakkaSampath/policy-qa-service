from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Absolute path: .env is found no matter which folder you run commands from.
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Ollama (runs on the developer/tester machine, default port)
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "embeddinggemma-2"
    llm_model: str = "gemma3"

    # Data
    docs_dir: Path = ROOT / "mock-policies"
    index_dir: Path = ROOT / "data" / "qdrant"
    collection: str = "policies"

    # Retrieval
    top_k: int = 4
    min_similarity: float = 0.5   # placeholder; re-measure with embeddinggemma, tune in step 6

    # Abuse protection
    max_question_chars: int = 500
    llm_timeout_s: float = 60.0      # laptop CPU: ~11-18 s per answer
    llm_max_tokens: int = 300
    max_concurrent_llm: int = 1      # answers at once; extra requests get 503 busy
    rate_limit_per_min: int = 10
    cache_size: int = 256


settings = Settings()
