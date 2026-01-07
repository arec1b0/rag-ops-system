import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Ops System"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    
    OPENAI_API_KEY: str

    # Milvus
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_COLLECTION_NAME: str = "rag_knowledge_base"
    EMBEDDING_DIMENSION: int = 1536 

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "agentic_rag_v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        case_sensitive=True
    )

settings = Settings()