from typing import Dict, Any
from .framework import UnstructuredFramework

class LLMInterface:
    def __init__(self, framework: UnstructuredFramework):
        self.framework = framework
        
    async def query(self, question: str) -> Dict[str, Any]:
        # Get relevant content from vector store
        relevant_content = await self.get_relevant_content(question)
        
        # Generate response using LLM
        response = await self.generate_response(question, relevant_content)
        
        return {
            "answer": response,
            "sources": relevant_content
        } 