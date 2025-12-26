#!/usr/bin/env python3
"""
Simple test script to verify ingestion works
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.services.openrouter_client import OpenRouterClient
import logging

# Load environment variables from backend .env file
load_dotenv('.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------
# CONFIG
# -------------------------------------
COLLECTION_NAME = "ai-spec-driven-interactive"
CHUNK_SIZE = 1200  # Maximum characters per chunk

# Initialize OpenRouter client
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenRouterClient(api_key=OPENROUTER_API_KEY)

# Connect to Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def extract_text_from_markdown(content):
    """Extract text content from markdown, removing headers and metadata"""
    # Remove frontmatter (YAML between ---)
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

    # Remove markdown headers, but keep the text
    # Remove # headers
    content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
    # Remove other markdown elements but keep text
    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold
    content = re.sub(r'\*(.*?)\*', r'\1', content)      # Italic
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Links

    return content.strip()

def chunk_text(text, max_chars=CHUNK_SIZE):
    """Split text into chunks of maximum max_chars"""
    chunks = []
    while len(text) > max_chars:
        # Find the last sentence ending before max_chars
        split_pos = text[:max_chars].rfind(". ")
        if split_pos == -1:
            # If no sentence ending found, split at max_chars
            split_pos = max_chars
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    if text:  # Add the remaining text if any
        chunks.append(text)
    return chunks

def embed(text):
    """Generate embedding for text using OpenRouter API"""
    try:
        response = client.Embedding(
            input=text,
            model="text-embedding-ada-002"  # OpenRouter compatible embedding model
        )
        return response.data[0].embedding  # Return the first embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

def save_chunk_to_qdrant(chunk, chunk_id, source_path):
    """Save a text chunk to Qdrant vector database"""
    vector = embed(chunk)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "source_path": source_path,
                    "text": chunk,
                    "chunk_id": chunk_id,
                    "type": "markdown_content"
                }
            )
        ]
    )
    logger.info(f"Saved chunk {chunk_id} to Qdrant")

def test_ingestion():
    """Test ingestion with a small test text"""
    logger.info("Starting test ingestion...")

    # Create a small test text instead of reading from file
    test_text = """
    # Introduction to AI and Robotics

    This is a test document for the AI and Robotics Learning Platform.
    The goal of this platform is to provide comprehensive education on artificial intelligence and robotics.

    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
    The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal.

    Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others.
    Robotics deals with the design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing.

    The combination of AI and robotics creates intelligent robots that can perform complex tasks in dynamic environments.
    These robots can perceive their environment, make decisions, and execute actions based on AI algorithms.

    Some key applications of AI and robotics include:
    - Industrial automation
    - Healthcare robotics
    - Autonomous vehicles
    - Service robots
    - Educational robots

    The future of AI and robotics promises even more advanced capabilities with developments in machine learning, deep learning, and neural networks.
    """

    logger.info(f"Using test text of {len(test_text)} characters")

    # Create chunks
    chunks = chunk_text(test_text)
    logger.info(f"Created {len(chunks)} chunks")

    # Create collection if it doesn't exist
    try:
        qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"Collection {COLLECTION_NAME} already exists")
    except:
        logger.info(f"Creating Qdrant collection: {COLLECTION_NAME}")
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1536,        # OpenAI text-embedding-ada-002 dimension
                distance=Distance.COSINE
            )
        )
        logger.info(f"Collection {COLLECTION_NAME} created successfully")

    # Save chunks to Qdrant
    chunk_id = 1
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Only save non-empty chunks
            logger.info(f"Processing chunk {i+1}/{len(chunks)} (size: {len(chunk)} chars)")
            save_chunk_to_qdrant(chunk, chunk_id, "test_document.md")
            chunk_id += 1

    logger.info(f"\n✔️ Test ingestion completed!")
    logger.info(f"Total chunks stored: {chunk_id - 1}")

    # Verify the collection has data
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"Collection points count: {collection_info.points_count}")

        # Try a search to verify retrieval works
        if collection_info.points_count > 0:
            # Create a test query embedding
            query_embedding = embed("What is artificial intelligence?")

            # Search for similar content
            search_results = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=3
            )

            logger.info(f"\nSearch test results: {len(search_results)} matches found")
            for i, result in enumerate(search_results):
                logger.info(f"  Result {i+1}: Score={result.score:.3f}, Text preview='{result.payload.get('text', '')[:50]}...'")
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")

if __name__ == "__main__":
    test_ingestion()