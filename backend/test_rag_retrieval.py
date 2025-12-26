#!/usr/bin/env python3
"""
Test script to verify RAG retrieval functionality
"""
import os
import sys
from dotenv import load_dotenv
from src.services.rag import RAGService
from src.models.chat import ChatRequest

# Load environment variables
load_dotenv('.env')

def test_rag_retrieval():
    """Test RAG retrieval functionality"""
    print("Testing RAG retrieval functionality...")

    try:
        # Initialize RAG service
        rag_service = RAGService()
        print("[OK] RAG service initialized successfully")

        # Test search functionality
        test_query = "What is artificial intelligence?"
        print(f"\nTesting search for query: '{test_query}'")

        # Search for documents
        results = rag_service.search_documents(test_query, limit=3)
        print(f"[OK] Found {len(results)} results")

        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    ID: {result.id}")
            print(f"    Score: {result.score}")
            print(f"    Content preview: {result.content[:100]}...")
            print(f"    Metadata: {result.metadata}")

        # Test response generation
        print(f"\nTesting response generation...")
        response = rag_service.generate_response(test_query, results)
        print(f"[OK] Generated response: {response[:200]}...")

        # Test full chat processing
        print(f"\nTesting full chat processing...")
        chat_request = ChatRequest(
            message=test_query,
            user_id=None,
            context_mode="book_wide",
            selected_text=None
        )
        chat_response = rag_service.process_chat_request(chat_request)
        print(f"[OK] Chat response: {chat_response.response[:200]}...")
        print(f"  Sources: {chat_response.sources}")
        print(f"  Context used: {chat_response.context_used}")

        print("\n[OK] All RAG functionality tests passed!")

    except Exception as e:
        print(f"[ERROR] RAG functionality test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_retrieval()