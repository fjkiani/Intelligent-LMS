import chromadb
from typing import Dict, List, Any

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("content_embeddings")
    
    async def store(self, 
                   content_id: str, 
                   embeddings: List[float], 
                   metadata: Dict[str, Any]):
        self.collection.add(
            embeddings=[embeddings],
            documents=[content_id],
            metadatas=[metadata]
        )
    
    async def search(self, query_embedding: List[float], n_results: int = 5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        ) 