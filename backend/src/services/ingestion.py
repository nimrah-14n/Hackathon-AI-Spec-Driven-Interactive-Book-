import requests
import xml.etree.ElementTree as ET
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from ..config.settings import settings
import trafilatura
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = settings.qdrant_collection_name

    def get_all_urls_from_sitemap(self, sitemap_url: str) -> List[str]:
        """Extract URLs from sitemap XML"""
        try:
            xml = requests.get(sitemap_url).text
            root = ET.fromstring(xml)

            urls = []
            for child in root:
                loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc_tag is not None:
                    urls.append(loc_tag.text)

            logger.info(f"Found {len(urls)} URLs from sitemap")
            return urls
        except Exception as e:
            logger.error(f"Error extracting URLs from sitemap: {e}")
            raise

    def extract_text_from_url(self, url: str) -> Optional[str]:
        """Extract text content from a URL"""
        try:
            html = requests.get(url).text
            text = trafilatura.extract(html)

            if not text:
                logger.warning(f"No text extracted from: {url}")
                return None

            return text
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {e}")
            return None

    def chunk_text(self, text: str, max_chars: int = 1200) -> List[str]:
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

    def create_collection(self):
        """Create Qdrant collection for document storage"""
        try:
            # Check if collection exists
            try:
                self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} already exists")
                return
            except:
                pass  # Collection doesn't exist, create it

            # Create new collection with OpenAI embedding dimensions (1536)
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI text-embedding-ada-002 dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            raise

    def save_chunk_to_qdrant(self, chunk: str, chunk_id: int, url: str, metadata: dict = None):
        """Save a text chunk to Qdrant vector database"""
        if metadata is None:
            metadata = {}

        metadata.update({
            "url": url,
            "chunk_id": chunk_id,
            "source": "docusaurus_book",
            "created_at": str(datetime.now())
        })

        try:
            # Generate embedding using the embedding service
            from .embedding import generate_embedding
            vector = generate_embedding(chunk)

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload=metadata
                    )
                ]
            )
            logger.info(f"Saved chunk {chunk_id} to Qdrant")
        except Exception as e:
            logger.error(f"Error saving chunk {chunk_id} to Qdrant: {e}")
            raise

    def ingest_book_content(self, sitemap_url: str):
        """Main ingestion pipeline to process all book content"""
        logger.info("Starting book content ingestion...")

        # Create collection if it doesn't exist
        self.create_collection()

        # Get all URLs from sitemap
        urls = self.get_all_urls_from_sitemap(sitemap_url)

        chunk_id = 1
        for url in urls:
            logger.info(f"Processing: {url}")
            text = self.extract_text_from_url(url)

            if not text:
                continue

            chunks = self.chunk_text(text)

            for chunk in chunks:
                self.save_chunk_to_qdrant(chunk, chunk_id, url)
                logger.info(f"Saved chunk {chunk_id}")
                chunk_id += 1

        logger.info(f"Ingestion completed! Total chunks stored: {chunk_id - 1}")
        return {"total_chunks": chunk_id - 1}


# Import datetime here to avoid circular imports
from datetime import datetime