# backend/main.py
"""
StructuraAI - Enterprise RAG Platform for Structured Data Querying
Main FastAPI application entry point
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
import time
from pathlib import Path

from backend.database import engine, Base
from backend.api.routes import (
    projects, sources, dictionary, embeddings, 
    search, chat, admin
)
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting StructuraAI...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base._metadata.create_all)
    
    # Create directories
    for directory in ['uploads', 'indexes', 'logs']:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("✅ StructuraAI started successfully!")
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down StructuraAI...")
    await engine.dispose()
    logger.info("✅ StructuraAI shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="StructuraAI",
    description="Enterprise RAG Platform for Structured Data Querying",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"🔍 {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"✅ {request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    
    return response

# API Routes
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(sources.router, prefix="/api/sources", tags=["sources"])
app.include_router(dictionary.router, prefix="/api/dictionary", tags=["dictionary"])
app.include_router(embeddings.router, prefix="/api/embeddings", tags=["embeddings"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "StructuraAI",
        "version": "1.0.0",
        "timestamp": time.time()
    }

# Serve static files and SPA
if Path("frontend/dist").exists():
    app.mount("/static", StaticFiles(directory="frontend/dist/static"), name="static")
    
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Serve React SPA"""
        file_path = Path("frontend/dist") / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse("frontend/dist/index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "StructuraAI API is running! Frontend not built yet."}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"🔥 Global exception: {exc}")
    return HTTPException(
        status_code=500,
        detail="Internal server error"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )