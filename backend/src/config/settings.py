from pydantic_settings import BaseSettings as Settings
from typing import Optional


class Settings(Settings):
    # Qdrant Configuration
    qdrant_url: str = "https://51a7be68-cba5-46d3-8735-040adcf7cd2a.us-east4-0.gcp.cloud.qdrant.io:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "ai-spec-driven-interactive"

    # OpenRouter Configuration
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "openai/gpt-3.5-turbo"  # Default model (OpenRouter format)

    # Neon Postgres Configuration
    database_url: Optional[str] = None

    # Better Auth Configuration
    better_auth_secret: Optional[str] = None
    better_auth_url: Optional[str] = None
    frontend_url: Optional[str] = "http://localhost:3000"
    backend_url: Optional[str] = "http://localhost:8000"

    # Application Configuration
    app_title: str = "RAG Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Feature Flags
    selected_text_mode_enabled: bool = True
    auth_enabled: bool = True
    personalization_enabled: bool = True
    urdu_translation_enabled: bool = True  # Enable Urdu translation feature

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,  # Changed to False to handle case variations
        "extra": "allow"
    }


# Create settings instance
settings = Settings()