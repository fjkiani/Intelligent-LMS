from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ContentProcessor(ABC):
    """Base class for all content processors"""
    
    @abstractmethod
    async def extract_content(self, content: Any) -> str:
        """Extract text content from the source"""
        pass
    
    @abstractmethod
    async def generate_metadata(self, content: str) -> Dict[str, Any]:
        """Generate metadata from content"""
        pass
    
    @abstractmethod
    async def create_embeddings(self, content: str) -> List[float]:
        """Create embeddings from content"""
        pass 