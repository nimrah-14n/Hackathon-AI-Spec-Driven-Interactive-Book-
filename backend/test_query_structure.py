#!/usr/bin/env python3
"""
Test the structure of query_points response
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.services.openrouter_client import OpenRouterClient

# Load environment variables
load_dotenv('.env')

# Connect to Qdrant
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "ai-spec-driven-interactive"

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# Create a test embedding
client = OpenRouterClient(api_key=os.getenv("OPENROUTER_API_KEY"))
test_embedding = client.Embedding(
    input="test query",
    model="text-embedding-ada-002"
)
query_embedding = test_embedding.data[0].embedding

# Test the query_points method
try:
    result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=3,
        with_payload=True
    )

    print(f"Result type: {type(result)}")
    print(f"Result attributes: {dir(result)}")

    if hasattr(result, 'points'):
        print(f"Has points attribute")
        print(f"Points type: {type(result.points)}")
        print(f"Number of points: {len(result.points)}")

        if result.points:
            first_point = result.points[0]
            print(f"First point type: {type(first_point)}")
            print(f"First point attributes: {dir(first_point)}")
            print(f"First point: {first_point}")
    else:
        print("No points attribute found")
        print(f"Result: {result}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()