from .base import ContentProcessor
import PyPDF2
from sentence_transformers import SentenceTransformer

class PDFProcessor(ContentProcessor):
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def extract_content(self, pdf_file):
        # Enhanced version of your current extract_text_from_pdf
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    async def generate_metadata(self, content: str):
        # Enhanced version of your current metadata extraction
        return {
            "key_concepts": self.extract_key_concepts(content),
            "sections": self.identify_sections(content),
            "estimated_reading_time": len(content.split()) / 200  # words per minute
        }
    
    async def create_embeddings(self, content: str):
        return self.embedding_model.encode(content) 