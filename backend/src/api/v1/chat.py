from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from ...models.chat import ChatRequest, ChatResponse, IngestionRequest
from ...services.hybrid_rag_service import HybridRAGService
from ...services.ingestion import DocumentIngestionService
from ...services.database import DatabaseService
from ...services.feature_flags import FeatureFlagsService
from ...config.settings import settings
from ...config.database import get_db, init_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])

# Initialize services
# Note: These are initialized at module level, but error handling is important
# for cases where API keys are not available during initialization
try:
    rag_service = HybridRAGService()
except Exception as e:
    # Create a placeholder that will handle errors gracefully
    class PlaceholderRAGService:
        def process_chat_request(self, chat_request, language_preference="en"):
            from ...models.chat import ChatResponse
            return ChatResponse(
                response="Service temporarily unavailable: API key configuration issue.",
                sources=[],
                context_used=chat_request.context_mode
            )
    rag_service = PlaceholderRAGService()
    import logging
    logging.getLogger(__name__).warning(f"RAG Service initialization failed: {e}")

ingestion_service = DocumentIngestionService()
feature_flags = FeatureFlagsService()


def get_database_service(db: Session = Depends(get_db)):
    """Dependency to get database service"""
    return DatabaseService(db)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest, db_service: DatabaseService = Depends(get_database_service)):
    """
    Main chat endpoint that processes user queries using Hybrid RAG
    """
    try:
        # Validate context mode
        if chat_request.context_mode not in ["book_wide", "selected_text"]:
            raise HTTPException(status_code=400, detail="Invalid context mode. Use 'book_wide' or 'selected_text'")

        # Check if selected_text mode is enabled when requested
        if chat_request.context_mode == "selected_text" and not feature_flags.is_enabled("selected_text_mode"):
            raise HTTPException(status_code=403, detail="Selected text mode is not enabled")

        # Create or get chat session
        # For anonymous users, user_id will be None, which is allowed
        chat_session = db_service.create_chat_session(user_id=chat_request.user_id)

        # Save the user's message to the database
        db_service.create_chat_message(
            session_id=chat_session.id,
            role="user",
            content=chat_request.message,
            context_used=chat_request.context_mode
        )

        # Process the chat request using Hybrid RAG service with language preference
        response = rag_service.process_chat_request(chat_request, language_preference=chat_request.language_preference)

        # Save the assistant's response to the database
        db_service.create_chat_message(
            session_id=chat_session.id,
            role="assistant",
            content=response.response,
            context_used=chat_request.context_mode
        )

        return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing chat request")


@router.get("/feature-flags")
async def get_feature_flags():
    """
    Get current feature flag status
    """
    try:
        return feature_flags.get_enabled_features()
    except Exception as e:
        logger.error(f"Error getting feature flags: {e}")
        raise HTTPException(status_code=500, detail="Internal server error getting feature flags")


@router.post("/ingest")
async def ingest_content(ingestion_request: IngestionRequest):
    """
    Endpoint to trigger content ingestion from URLs
    """
    try:
        # This is a simplified version - in a real implementation, you might want to
        # process specific URLs rather than the sitemap
        # For now, we'll use the sitemap URL from settings
        sitemap_url = "https://hackathon-ai-spec-driven-interactiv-drab.vercel.app/sitemap.xml"  # This should come from settings or request
        result = ingestion_service.ingest_book_content(sitemap_url)
        return result

    except Exception as e:
        logger.error(f"Error in ingestion endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during content ingestion")


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "RAG Chatbot API"}


@router.get("/config")
async def get_config():
    """
    Get current configuration (without sensitive data)
    """
    return {
        "app_title": settings.app_title,
        "app_version": settings.app_version,
        "debug": settings.debug,
        "selected_text_mode_enabled": settings.selected_text_mode_enabled,
        "auth_enabled": settings.auth_enabled,
        "personalization_enabled": settings.personalization_enabled,
        "urdu_translation_enabled": settings.urdu_translation_enabled
    }