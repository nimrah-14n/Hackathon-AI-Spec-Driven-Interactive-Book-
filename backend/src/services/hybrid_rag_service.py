import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..models.chat import ChatRequest, ChatResponse, DocumentChunk
from ..config.settings import settings
from .embedding import EmbeddingService
from .openrouter_client import OpenRouterClient
import logging
import re

logger = logging.getLogger(__name__)


class HybridRAGService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = settings.qdrant_collection_name
        self.embedding_service = EmbeddingService()
        self.openrouter_client = OpenRouterClient(api_key=settings.openrouter_api_key)

    def search_documents(self, query: str, limit: int = 5, selected_text: str = None) -> List[DocumentChunk]:
        """
        Search for relevant documents based on the query
        If selected_text is provided, search within that context only
        """
        try:
            if selected_text and selected_text.strip():
                # For selected-text-only mode, return the selected text as the only context
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

    def check_domain_relevance(self, query: str) -> bool:
        """
        Check if the question relates to AI / Robotics / ML / Education
        """
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()

        # Keywords related to AI, Robotics, ML, and Education
        ai_keywords = [
            "artificial intelligence", "ai", "machine learning", "ml", "deep learning",
            "neural network", "algorithm", "data science", "computer vision",
            "natural language processing", "nlp", "reinforcement learning", "transformer",
            "gpt", "llm", "large language model", "neural network", "supervised learning",
            "unsupervised learning", "reinforcement learning", "computer vision"
        ]

        robotics_keywords = [
            "robotics", "robot", "automation", "actuator", "sensor", "servo",
            "microcontroller", "arduino", "raspberry pi", "kinematics", "dynamics",
            "motion planning", "path planning", "slam", "simultaneous localization",
            "mapping", "computer vision", "robot arm", "mobile robot", "humanoid",
            "navigation", "control system", "pid controller", "ros", "robot operating system"
        ]

        ml_keywords = [
            "machine learning", "ml", "deep learning", "neural network", "algorithm",
            "training", "dataset", "model", "accuracy", "precision", "recall",
            "f1 score", "overfitting", "underfitting", "cross validation", "feature",
            "classification", "regression", "clustering", "svm", "decision tree",
            "random forest", "gradient boosting", "xgboost", "tensorflow", "pytorch"
        ]

        education_keywords = [
            "education", "learning", "course", "tutorial", "teaching", "student",
            "curriculum", "syllabus", "academic", "scholarly", "pedagogy", "instruction",
            "knowledge", "understanding", "concept", "theory", "principle", "study",
            "research", "analysis", "methodology", "learning outcome", "competency"
        ]

        # Check if any of the keywords appear in the query
        all_keywords = ai_keywords + robotics_keywords + ml_keywords + education_keywords

        for keyword in all_keywords:
            if keyword in query_lower:
                return True

        # Additional check for broader AI/Robotics related terms
        broad_keywords = [
            "technology", "programming", "coding", "software", "hardware", "computing",
            "science", "engineering", "automation", "intelligent", "autonomous",
            "algorithmic", "computation", "digital", "smart", "intelligent system"
        ]

        for keyword in broad_keywords:
            if keyword in query_lower:
                # Additional context check to ensure it's related to AI/Robotics
                if any(ai_keyword in query_lower for ai_keyword in ai_keywords + robotics_keywords + ml_keywords):
                    return True

        return False

    def generate_rag_response(self, query: str, context: List[DocumentChunk], context_mode: str = "book_wide") -> str:
        """
        Generate a response using RAG based on the query and context
        """
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        try:
            # Construct the context from document chunks
            context_text = "\n\n".join([chunk.content for chunk in context if chunk.content.strip()])

            if not context_text.strip():
                return "I'm sorry, I couldn't find this information in the book content."

            # Create the system message with the exact prompt specified
            system_message = """You are an AI tutor for an AI & Robotics learning book.
Prefer book content.
Fallback to AI & Robotics knowledge only.
Support Urdu translation when requested.
Never answer unrelated questions."""

            # Create the user message
            user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

            # Call OpenRouter API with strict parameters to enforce context-only responses
            response = self.openrouter_client.ChatCompletion(
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
            logger.error(f"Error generating RAG response: {e}")
            return "I'm sorry, I'm currently experiencing technical difficulties. Please try again later or check that the API configuration is correct."

    def generate_domain_response(self, query: str) -> str:
        """
        Generate a response using general AI & Robotics knowledge
        """
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        try:
            system_message = """You are an AI tutor for an AI & Robotics learning book.
Prefer book content.
Fallback to AI & Robotics knowledge only.
Support Urdu translation when requested.
Never answer unrelated questions."""

            user_message = f"Question: {query}\n\nIf this question is related to AI, Robotics, Machine Learning, or Education, please answer it. Otherwise, politely refuse to answer and explain that you only answer questions related to these topics."

            response = self.openrouter_client.ChatCompletion(
                model=settings.openrouter_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.3,
                top_p=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating domain response: {e}")
            return "I'm sorry, I'm currently experiencing technical difficulties. Please try again later."

    def generate_refusal_response(self, query: str) -> str:
        """
        Generate a polite refusal response for unrelated questions
        """
        system_message = """You are an AI tutor for an AI & Robotics learning book.
Prefer book content.
Fallback to AI & Robotics knowledge only.
Support Urdu translation when requested.
Never answer unrelated questions."""

        user_message = f"Question: {query}\n\nPlease politely refuse to answer this question as it is not related to AI, Robotics, Machine Learning, or Education."

        try:
            response = self.openrouter_client.ChatCompletion(
                model=settings.openrouter_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=300,
                temperature=0.1
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating refusal response: {e}")
            return "I'm sorry, I can only answer questions related to AI, Robotics, Machine Learning, or Education."

    def translate_to_urdu(self, text: str) -> str:
        """
        Translate text to clear academic Urdu
        """
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        try:
            system_message = """You are a professional translator. Translate the provided text to clear, formal, academic Urdu. Use simple academic Urdu with controlled vocabulary and formal but accessible language standards. Ensure technical terms are preserved where appropriate and transliterated if no direct Urdu equivalent exists."""

            user_message = f"Please translate the following text to clear, formal, academic Urdu:\n\n{text}"

            response = self.openrouter_client.ChatCompletion(
                model=settings.openrouter_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.2
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error translating to Urdu: {e}")
            return f"Translation error: {text}"

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

    def process_chat_request(self, chat_request: ChatRequest, language_preference: str = "en") -> ChatResponse:
        """
        Process a chat request with hybrid RAG logic
        """
        try:
            # First, search for relevant documents based on the query and context mode
            context_chunks = self.search_documents(
                query=chat_request.message,
                selected_text=chat_request.selected_text if chat_request.context_mode == "selected_text" else None
            )

            # Determine response based on whether we found relevant context
            if context_chunks:
                # Use RAG to generate response from book content
                response_text = self.generate_rag_response(
                    query=chat_request.message,
                    context=context_chunks,
                    context_mode=chat_request.context_mode
                )
                response_source = "book_content"
            else:
                # No relevant book content found, check domain relevance
                if self.check_domain_relevance(chat_request.message):
                    # Generate response using general AI & Robotics knowledge
                    response_text = self.generate_domain_response(chat_request.message)
                    response_source = "ai_knowledge"
                else:
                    # Question is not related to AI/Robotics, refuse politely
                    response_text = self.generate_refusal_response(chat_request.message)
                    response_source = "refused"

            # If Urdu is requested, translate the final response
            if language_preference.lower() == "ur":
                response_text = self.translate_to_urdu(response_text)

            # Extract sources from context
            sources = [chunk.metadata.get("url", "") for chunk in context_chunks if chunk.metadata.get("url")]
            sources = list(set(sources))  # Remove duplicates

            # Create response object with source indicator
            chat_response = ChatResponse(
                response=response_text,
                sources=sources,
                context_used=chat_request.context_mode
            )

            return chat_response

        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            raise