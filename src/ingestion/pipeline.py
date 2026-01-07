from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.milvus_client import MilvusHandler
from src.ingestion.embeddings import EmbeddingService

class IngestionPipeline:
    def __init__(self):
        self.milvus = MilvusHandler()
        self.embedder = EmbeddingService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def run(self, documents: List[Dict[str, str]]):
        """
        Ingests a list of raw documents into Milvus.
        documents format: [{"text": "...", "source": "filename"}]
        """
        print(f"Starting ingestion for {len(documents)} documents...")
        
        # 1. Chunking
        texts_to_embed = []
        sources = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])
            for chunk in chunks:
                texts_to_embed.append(chunk)
                sources.append(doc["source"])
                
        print(f"Created {len(texts_to_embed)} chunks.")

        # 2. Embedding
        # In production, batch this if > 1000 chunks
        print("Generating embeddings...")
        vectors = self.embedder.embed_documents(texts_to_embed)

        # 3. Insert into Milvus
        collection = self.milvus.get_collection()
        
        # Data must be list of columns: [[text_1, ...], [vector_1, ...], [source_1, ...]]
        # Note: 'id' is auto-generated
        data = [
            texts_to_embed,
            vectors,
            sources
        ]
        
        print("Inserting into Milvus...")
        collection.insert(data)
        collection.flush() # Ensure data is written to disk
        print(f"Successfully inserted {len(texts_to_embed)} vectors.")

# Helper to run manually
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    
    # UPDATED: Added a 3rd document with the Tech Stack info
    sample_docs = [
        {
            "text": "Dani has Type 1 Bipolar Disorder and ADHD. He prefers direct communication and practical solutions. MLOps priorities include drift detection and automated dashboards.",
            "source": "user_profile_2025.txt"
        },
        {
            "text": "Production RAG requires robust monitoring. Latency is a key metric. Agentic RAG adds complexity but improves reasoning.",
            "source": "rag_architecture_guide.pdf"
        },
        {
            "text": "The Tech Stack for the backend includes the Python ecosystem as the main language, Docker and Kubernetes for containerization, and FastAPI for backends. MLflow is used for experiment tracking, and Google Colab for prototyping. Production requires CI/CD, monitoring, and logging.",
            "source": "tech_stack_requirements.txt"
        }
    ]
    
    pipeline.run(sample_docs)