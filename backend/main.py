import uvicorn
from src.config.settings import settings

# This file serves as the entry point for the FastAPI application
# The actual application is defined in src.main

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )