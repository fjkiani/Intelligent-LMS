import pytest
from ..framework import UnstructuredFramework
from ..processors.pdf_processor import PDFProcessor

@pytest.mark.asyncio
async def test_pdf_processing():
    framework = UnstructuredFramework()
    framework.register_processor("pdf", PDFProcessor())
    
    # Test with a sample PDF
    with open("tests/sample.pdf", "rb") as pdf_file:
        result = await framework.process_content(pdf_file, "pdf")
        
    assert "content" in result
    assert "metadata" in result 