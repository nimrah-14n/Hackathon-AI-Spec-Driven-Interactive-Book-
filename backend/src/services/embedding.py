import os
from typing import List
import numpy as np
from ..config.settings import settings
from .openrouter_client import OpenRouterClient
import logging

logger = logging.getLogger(__name__)


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using OpenRouter API
    """
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    # Initialize OpenRouter client
    client = OpenRouterClient(api_key=settings.openrouter_api_key)

    try:
        response = client.Embedding(
            input=text,
            model="text-embedding-ada-002"  # This model is available through OpenRouter
        )

        # Extract the embedding from the response (v1.x API format)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using OpenRouter API
    """
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    # Initialize OpenRouter client
    client = OpenRouterClient(api_key=settings.openrouter_api_key)

    try:
        response = client.Embedding(
            input=texts,
            model="text-embedding-ada-002"
        )

        # Extract all embeddings from the response (v1.x API format)
        embeddings = [item.embedding for item in response.data]
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


class EmbeddingService:
    def __init__(self):
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        # Initialize OpenRouter client
        self.client = OpenRouterClient(api_key=settings.openrouter_api_key)

    def create_document_embedding(self, text: str) -> List[float]:
        """Create embedding for a document chunk"""
        return generate_embedding(text)

    def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for a search query"""
        return generate_embedding(query)

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Convert to numpy arrays for calculation
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))