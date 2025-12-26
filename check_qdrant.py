#!/usr/bin/env python3
"""
Script to check Qdrant collection status and verify if documents exist
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Use backend .env file for consistency
backend_env_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
if os.path.exists(backend_env_path):
    load_dotenv(backend_env_path, override=True)

# Get Qdrant configuration
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME", "ai-spec-driven-interactive")

print(f"Qdrant URL: {qdrant_url}")
print(f"Collection Name: {collection_name}")

if not qdrant_url or not qdrant_api_key:
    print("ERROR: Qdrant URL or API key not found in environment variables")
    exit(1)

try:
    # Connect to Qdrant
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"[OK] Collection '{collection_name}' exists")
        print(f"Points count: {collection_info.points_count}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance: {collection_info.config.params.vectors.distance}")

        # If there are points, get a few samples
        if collection_info.points_count > 0:
            print(f"\nSample points (first 3):")
            points = client.scroll(
                collection_name=collection_name,
                limit=3,
                with_payload=True,
                with_vector=False
            )
            for i, (point, _) in enumerate(points[0]):
                print(f"  Point {i+1}:")
                print(f"    ID: {point.id}")
                if 'text' in point.payload:
                    text_preview = point.payload['text'][:100] + "..." if len(point.payload['text']) > 100 else point.payload['text']
                    print(f"    Text preview: {text_preview}")
                print(f"    Payload keys: {list(point.payload.keys())}")
        else:
            print("No points found in the collection")

    except Exception as e:
        print(f"[ERROR] Collection '{collection_name}' does not exist or error occurred: {e}")

except Exception as e:
    print(f"[ERROR] Error connecting to Qdrant: {e}")