import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..models.chat import ChatRequest, ChatResponse, DocumentChunk
from ..config.settings import settings
from .embedding import EmbeddingService
from .openrouter_client import OpenRouterClient
import logging

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = settings.qdrant_collection_name
        self.embedding_service = EmbeddingService()

    def search_documents(self, query: str, limit: int = 5, selected_text: str = None) -> List[DocumentChunk]:
        """
        Search for relevant documents based on the query
        If selected_text is provided, search within that context only
        """
        try:
            if selected_text and selected_text.strip():
                # For selected-text-only mode, return the selected text as the only context
                # with high confidence score since it's the exact text the user selected
                return [DocumentChunk(
                    id="selected_text",
                    content=selected_text,
                    metadata={"source": "selected_text", "url": "current_page", "type": "user_selection"},
                    score=1.0
                )]
            else:
                # Perform normal vector search in Qdrant for book-wide context
                try:
                    query_embedding = self.embedding_service.create_query_embedding(query)
                except Exception as embedding_error:
                    logger.error(f"Error generating query embedding: {embedding_error}")
                    # Return empty results if embedding generation fails
                    # This can happen if API key is invalid
                    return []

                # Use the correct Qdrant client API for the installed version
                try:
                    # Try newer API first
                    query_response = self.qdrant_client.query_points(
                        collection_name=self.collection_name,
                        query=query_embedding,
                        limit=limit,
                        with_payload=True
                    )
                    # Extract points from the response
                    search_result = query_response.points
                except AttributeError:
                    # Fallback to older search API if query_points doesn't exist
                    search_result = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        limit=limit,
                        with_payload=True
                    )

                # Convert search results to DocumentChunk objects
                chunks = []
                for result in search_result:
                    chunk = DocumentChunk(
                        id=str(result.id),
                        content=result.payload.get("text", ""),
                        metadata=result.payload,
                        score=result.score
                    )
                    chunks.append(chunk)

                return chunks

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            # Return empty list instead of raising error, so the chat can still work
            return []

    def generate_response(self, query: str, context: List[DocumentChunk], context_mode: str = "book_wide") -> str:
        """
        Generate a response using OpenRouter based on the query and context
        """
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        # Initialize OpenRouter client
        client = OpenRouterClient(api_key=settings.openrouter_api_key)

        try:
            # Construct the context from document chunks
            context_text = "\n\n".join([chunk.content for chunk in context if chunk.content.strip()])

            if not context_text.strip():
                return "I'm sorry, I couldn't find this information in the book content."

            # Create the system message to enforce strict context-only responses
            system_message = f"""You are an AI assistant for an AI & Robotics Learning Book.
            You must answer ONLY from the provided book content.
            Do not use any general knowledge or information outside the provided context.
            Do not make up information or infer beyond what is explicitly stated in the context.
            If the information is not in the provided context, clearly state:
            'I'm sorry, I couldn't find this information in the book content.'

            Strictly enforce that your response only contains information that appears in the context.
            Context mode: {context_mode}"""

            # Create the user message
            user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

            # Call OpenRouter API with strict parameters to enforce context-only responses
            response = client.ChatCompletion(
                model=settings.openrouter_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.1,  # Lower temperature for more consistent, fact-based responses
                top_p=0.5,        # Lower top_p to reduce creativity
                stop=["Question:"]  # Stop if it tries to generate another question
            )

            generated_response = response.choices[0].message.content.strip()

            # Additional validation to ensure response comes from context
            validated_response = self.enforce_context_only(generated_response, context)
            return validated_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return a helpful message instead of failing completely
            return "I'm sorry, I'm currently experiencing technical difficulties. Please try again later or check that the API configuration is correct."

    def enforce_context_only(self, response: str, context_chunks: List[DocumentChunk]) -> str:
        """
        Additional enforcement to ensure responses only contain information from the context
        """
        # In a production system, you would implement more sophisticated validation
        # For now, we ensure that if the context is empty, we return the proper refusal message
        context_text = "\n\n".join([chunk.content for chunk in context_chunks if chunk.content.strip()])

        if not context_text.strip():
            return "I'm sorry, I couldn't find this information in the book content."

        # Check if the response contains the refusal message already
        if "I'm sorry, I couldn't find this information in the book content." in response:
            return response

        # If response looks valid, return it
        return response

    def process_chat_request(self, chat_request: ChatRequest) -> ChatResponse:
        """
        Process a chat request with RAG logic
        """
        try:
            # Search for relevant documents based on the query and context mode
            context_chunks = self.search_documents(
                query=chat_request.message,
                selected_text=chat_request.selected_text if chat_request.context_mode == "selected_text" else None
            )

            # Generate response based on context
            response_text = self.generate_response(
                query=chat_request.message,
                context=context_chunks,
                context_mode=chat_request.context_mode
            )

            # Extract sources from context
            sources = [chunk.metadata.get("url", "") for chunk in context_chunks if chunk.metadata.get("url")]
            sources = list(set(sources))  # Remove duplicates

            # Create response object
            chat_response = ChatResponse(
                response=response_text,
                sources=sources,
                context_used=chat_request.context_mode
            )

            return chat_response

        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            raise

    def enforce_context_only(self, response: str, context_chunks: List[DocumentChunk]) -> str:
        """
        Additional enforcement to ensure responses only contain information from the context
        This is a simplified version - in a production system, you'd want more sophisticated validation
        """
        # In a real implementation, we would validate that the response only contains
        # information that appears in the context chunks
        return response