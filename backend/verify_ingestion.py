import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Connect to Qdrant
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "ai-spec-driven-interactive"

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

try:
    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' exists")
    print(f"Points count: {collection_info.points_count}")
    print(f"Vector size: {collection_info.config.params.vectors.size}")

    # Try to search for a sample query to verify content is searchable
    if collection_info.points_count > 0:
        # Create a simple embedding for test search
        from src.services.openrouter_client import OpenRouterClient
        openrouter_client = OpenRouterClient(api_key=os.getenv("OPENROUTER_API_KEY"))

        # Create a test embedding
        test_embedding = openrouter_client.Embedding(
            input="AI and Robotics",
            model="text-embedding-ada-002"
        )
        test_vector = test_embedding.data[0].embedding

        # Search for similar content
        search_results = client.search(
            collection_name=collection_name,
            query_vector=test_vector,
            limit=3
        )

        print(f"\nSearch results count: {len(search_results)}")
        for i, result in enumerate(search_results):
            print(f"Result {i+1}:")
            print(f"  Score: {result.score}")
            print(f"  Source: {result.payload.get('source_path', 'Unknown')}")
            print(f"  Content preview: {result.payload.get('text', '')[:100]}...")
            print()
    else:
        print("Collection is empty")

except Exception as e:
    print(f"Error: {e}")