import os
import re
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
from src.services.openrouter_client import OpenRouterClient
import logging

# Load environment variables from .env file
load_dotenv()

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
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

client = OpenRouterClient(api_key=OPENROUTER_API_KEY)

# Connect to Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# -------------------------------------
# Step 1 — Read all markdown files from docs directory
# -------------------------------------
def read_markdown_files(docs_dir="../docs"):
    """Read all markdown files from the docs directory"""
    files = []
    docs_path = Path(docs_dir)

    # Find all .md and .mdx files recursively
    for file_path in docs_path.rglob("*"):
        if file_path.suffix in ['.md', '.mdx']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        files.append({
                            'path': str(file_path),
                            'content': content,
                            'filename': file_path.name
                        })
                        logger.info(f"Read file: {file_path}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Total files read: {len(files)}")
    return files

# -------------------------------------
# Step 2 — Extract text content from markdown
# -------------------------------------
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

# -------------------------------------
# Step 3 — Chunk the text
# -------------------------------------
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

# -------------------------------------
# Step 4 — Create embedding
# -------------------------------------
def embed(text):
    """Generate embedding for text using OpenRouter API"""
    try:
        response = client.Embedding(
            input=text,
            model="text-embedding-ada-002"  # OpenRouter compatible embedding model
        )
        return response['data'][0]['embedding']  # Return the first embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

# -------------------------------------
# Step 5 — Store in Qdrant
# -------------------------------------
def create_collection():
    """Create Qdrant collection for document storage"""
    logger.info(f"Creating Qdrant collection: {COLLECTION_NAME}")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,        # OpenAI text-embedding-ada-002 dimension
            distance=Distance.COSINE
        )
    )
    logger.info(f"Collection {COLLECTION_NAME} created successfully")

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
    logger.info(f"Saved chunk {chunk_id} from {source_path}")

# -------------------------------------
# MAIN INGESTION PIPELINE
# -------------------------------------
def ingest_local_docs():
    """Main function to ingest all local documentation files"""
    logger.info("Starting local documentation ingestion...")

    # Read all markdown files
    files = read_markdown_files()

    if not files:
        logger.warning("No markdown files found in docs directory")
        return

    # Create collection
    create_collection()

    chunk_id = 1

    for file_info in files:
        logger.info(f"Processing: {file_info['path']}")

        # Extract clean text from markdown
        text = extract_text_from_markdown(file_info['content'])

        if not text.strip():
            logger.warning(f"No text extracted from: {file_info['path']}")
            continue

        # Create chunks
        chunks = chunk_text(text)

        logger.info(f"Created {len(chunks)} chunks from {file_info['path']}")

        for chunk in chunks:
            if chunk.strip():  # Only save non-empty chunks
                save_chunk_to_qdrant(chunk, chunk_id, file_info['path'])
                chunk_id += 1

    logger.info(f"\n✔️ Ingestion completed!")
    logger.info(f"Total chunks stored: {chunk_id - 1}")

    # Verify the collection has data
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"Collection points count: {collection_info.points_count}")
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")

if __name__ == "__main__":
    ingest_local_docs()