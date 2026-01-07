from langchain_core.tools import tool
from src.ingestion.milvus_client import MilvusHandler
from src.ingestion.embeddings import EmbeddingService

# Initialize singletons outside the function to avoid reconnecting on every call
milvus_handler = MilvusHandler()
embedding_service = EmbeddingService()

@tool
def retriever_tool(query: str) -> str:
    """
    Search the knowledge base for documents relevant to the query.
    Returns the top 3 most relevant text chunks.
    """
    try:
        # 1. Embed Query
        query_vector = embedding_service.embed_query(query)
        
        # 2. Search Milvus
        collection = milvus_handler.get_collection()
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}, 
        }
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=3,
            output_fields=["text", "source"]
        )
        
        # 3. Format Results
        # Milvus returns a 2D list (one list of hits per query)
        hits = results[0]
        
        if not hits:
            return "No relevant documents found in the database."
            
        formatted_docs = []
        for hit in hits:
            # hit.entity.get("text") retrieves the scalar field
            formatted_docs.append(f"Content: {hit.entity.get('text')}\nSource: {hit.entity.get('source')}")
            
        return "\n\n".join(formatted_docs)
        
    except Exception as e:
        return f"Error connecting to vector database: {str(e)}"