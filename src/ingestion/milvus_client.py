from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from src.core.config import settings

class MilvusHandler:
    def __init__(self):
        self.uri = settings.MILVUS_URI
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dim = settings.EMBEDDING_DIMENSION
        self._connect()

    def _connect(self):
        """Establish connection to Milvus."""
        print(f"Connecting to Milvus at {self.uri}...")
        connections.connect(alias="default", uri=self.uri)
        print("Connected to Milvus.")

    def create_collection_if_not_exists(self):
        """Creates the collection schema if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            return Collection(self.collection_name)

        print(f"Creating collection '{self.collection_name}'...")
        
        # 1. Define Fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            # Metadata fields for filtering
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        ]

        # 2. Define Schema
        schema = CollectionSchema(fields, "Knowledge base for RAG Agent")

        # 3. Create Collection
        collection = Collection(self.collection_name, schema)

        # 4. Create Index (IVF_FLAT is good for general purpose, or HNSW for performance)
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="vector", index_params=index_params)
        
        # 5. Load collection into memory
        collection.load()
        print(f"Collection '{self.collection_name}' created and loaded.")
        return collection

    def get_collection(self) -> Collection:
        """Returns the collection object, loading it if necessary."""
        self.create_collection_if_not_exists()
        collection = Collection(self.collection_name)
        collection.load()
        return collection