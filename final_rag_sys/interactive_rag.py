from rag_bot import retrieval_chain

def interactive_rag():
    print("RAG System Interactive Mode")
    print("Type 'exit' to quit")
    
    chat_history = []
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        result = retrieval_chain.invoke(
            {
                "input": user_input,
                "chat_history": chat_history,
            }
        )
        
        print("\nAnswer:")
        print(result["answer"])
        
        # Update chat history for context
        chat_history.append((user_input, result["answer"]))

if __name__ == "__main__":
    interactive_rag() 