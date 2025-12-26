#!/usr/bin/env python3
"""
Check Qdrant client available methods
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv('.env')

# Connect to Qdrant
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

print("Qdrant Client methods:")
methods = [method for method in dir(client) if not method.startswith('_')]
for method in sorted(methods):
    print(f"  {method}")

print("\nChecking for search methods:")
search_methods = [method for method in methods if 'search' in method.lower()]
for method in search_methods:
    print(f"  {method}")