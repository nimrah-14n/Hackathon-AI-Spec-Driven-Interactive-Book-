from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API routers
from backend.src.api import auth, chapters

app = FastAPI(
    title="AI/Spec-Driven Interactive Book API",
    description="Backend API for the ROS 2 Educational Module and other modules",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(chapters.router, prefix="/chapters", tags=["chapters"])
# Additional routers will be added as they're implemented

@app.get("/")
def read_root():
    return {"message": "AI/Spec-Driven Interactive Book Backend API"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "backend-api"}