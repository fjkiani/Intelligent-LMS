from rag_bot import retrieval_chain

def test_rag():
    # Test with a simple question
    result = retrieval_chain.invoke(
        {
            "input": "What is post-OCR correction?",
            "chat_history": [],
        }
    )
    
    print("\n== Test Results ==\n")
    print(result["answer"])

if __name__ == "__main__":
    test_rag() 