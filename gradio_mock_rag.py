import gradio as gr
import random
import time

# Use the same mock_documents and functions from the Streamlit app
# ... (copy the mock_documents, retrieve_documents, and generate_rag_response functions)

def respond(message, history):
    # Simulate document retrieval
    retrieved_docs = retrieve_documents(message)
    
    # Generate response
    response = generate_rag_response(message, retrieved_docs)
    
    return response

demo = gr.ChatInterface(
    respond,
    title="Mock RAG System Demo",
    description="Ask questions about Post-OCR, Knowledge Graphs, or DevOps",
    theme="default"
)

if __name__ == "__main__":
    demo.launch(share=False) 