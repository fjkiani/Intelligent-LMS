from typing import Dict, Any
from .processors.base import ContentProcessor
from .storage.vector_store import VectorStore

class UnstructuredFramework:
    def __init__(self):
        self.processors: Dict[str, ContentProcessor] = {}
        self.vector_store = VectorStore()
        
    def register_processor(self, content_type: str, processor: ContentProcessor):
        """Register a content processor for a specific content type"""
        self.processors[content_type] = processor
        
    async def process_content(self, content: Any, content_type: str):
        """Process content through the pipeline"""
        processor = self.processors.get(content_type)
        if not processor:
            raise ValueError(f"No processor registered for {content_type}")
            
        # Extract content
        extracted_content = await processor.extract_content(content)
        
        # Generate metadata
        metadata = await processor.generate_metadata(extracted_content)
        
        # Create embeddings
        embeddings = await processor.create_embeddings(extracted_content)
        
        # Store in vector database
        await self.vector_store.store(
            content_id=str(hash(extracted_content)),
            embeddings=embeddings,
            metadata=metadata
        )
        
        return {
            "content": extracted_content,
            "metadata": metadata
        } 