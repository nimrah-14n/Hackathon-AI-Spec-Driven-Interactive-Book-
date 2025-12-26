#!/usr/bin/env python3
"""
Simple test script to verify API connectivity
"""
import os
from dotenv import load_dotenv

# Load environment variables from backend .env file
load_dotenv('.env')

print("Environment variables check:")
print(f"OPENROUTER_API_KEY: {'SET' if os.getenv('OPENROUTER_API_KEY') else 'NOT SET'}")
print(f"QDRANT_URL: {os.getenv('QDRANT_URL', 'NOT SET')}")
print(f"QDRANT_API_KEY: {'SET' if os.getenv('QDRANT_API_KEY') else 'NOT SET'}")

# Test OpenRouter API
openrouter_key = os.getenv('OPENROUTER_API_KEY')
if openrouter_key:
    print("\nTesting OpenRouter API connection...")
    try:
        from src.services.openrouter_client import OpenRouterClient
        client = OpenRouterClient(api_key=openrouter_key)

        # Test embedding API
        print("Testing embedding API...")
        response = client.Embedding(
            input="test",
            model="text-embedding-ada-002"
        )
        print("[OK] Embedding API test successful")
        print(f"  Embedding vector length: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"[ERROR] Embedding API test failed: {e}")
else:
    print("[ERROR] OpenRouter API key not available")

# Test Qdrant connection
qdrant_url = os.getenv('QDRANT_URL')
qdrant_key = os.getenv('QDRANT_API_KEY')
if qdrant_url and qdrant_key:
    print("\nTesting Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_key,
        )

        # Test connection
        collections = qdrant_client.get_collections()
        print("[OK] Qdrant connection successful")
        print(f"  Available collections: {[col.name for col in collections.collections]}")
    except Exception as e:
        print(f"[ERROR] Qdrant connection failed: {e}")
else:
    print("[ERROR] Qdrant credentials not available")