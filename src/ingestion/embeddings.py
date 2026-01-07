from typing import List
from langchain_openai import OpenAIEmbeddings
from src.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.model.embed_query(text)