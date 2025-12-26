from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1.chat import router as chat_router
from .config.settings import settings
from .config.database import init_db
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing API routers
from .api import auth, chapters

# Create FastAPI app
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables on startup
@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Database initialized successfully")

# Include API routers
app.include_router(chat_router)  # RAG chatbot API
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(chapters.router, prefix="/chapters", tags=["chapters"])

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API", "version": settings.app_version}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RAG Chatbot API"}