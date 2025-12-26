from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .settings import settings
import logging

logger = logging.getLogger(__name__)

# Create database engine
if settings.database_url:
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=300,    # Recycle connections every 5 minutes
    )
    logger.info("Connected to Neon Postgres database")
else:
    # For development, we might use a local SQLite database
    engine = create_engine("sqlite:///./rag_chatbot.db",
                          connect_args={"check_same_thread": False})
    logger.warning("Using SQLite for development. Set DATABASE_URL for production.")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize the database by creating all tables
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")