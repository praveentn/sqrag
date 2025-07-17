# run.py
"""
StructuraAI Application Runner
Main entry point for running the StructuraAI backend server
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
from backend.database import init_db, check_db_health
from config import Config, get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment and validate configuration"""
    
    # Create necessary directories
    directories = ['uploads', 'indexes', 'logs', 'indexes/faiss', 'indexes/tfidf']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    # Validate configuration
    try:
        Config.validate_config()
        logger.info("✅ Configuration validation passed")
    except ValueError as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

async def check_dependencies():
    """Check if all required dependencies are available"""
    
    try:
        # Check database connectivity
        db_healthy = await check_db_health()
        if db_healthy:
            logger.info("✅ Database connection healthy")
        else:
            logger.warning("⚠️ Database connection failed - using SQLite fallback")
        
        # Check LLM configuration
        if Config.LLM_CONFIG['azure']['api_key']:
            logger.info("✅ Azure OpenAI configuration found")
        else:
            logger.warning("⚠️ Azure OpenAI API key not configured")
        
        # Initialize database
        await init_db()
        logger.info("✅ Database initialized")
        
    except Exception as e:
        logger.error(f"❌ Dependency check failed: {e}")
        sys.exit(1)

def print_startup_info():
    """Print startup information"""
    
    print("\n" + "="*60)
    print("🚀 StructuraAI - Enterprise RAG Platform")
    print("="*60)
    print(f"📍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"🗄️  Database: {Config.SQLALCHEMY_DATABASE_URI}")
    print(f"🔗 API URL: http://localhost:8000")
    print(f"📱 Frontend: http://localhost:3000")
    print(f"📚 API Docs: http://localhost:8000/api/docs")
    print("="*60)
    print("📝 Available endpoints:")
    print("   • /api/projects - Project management")
    print("   • /api/sources - Data source management")
    print("   • /api/dictionary - Business dictionary")
    print("   • /api/embeddings - Vector embeddings")
    print("   • /api/search - Search functionality")
    print("   • /api/chat - Natural language queries")
    print("   • /api/admin - Admin panel")
    print("="*60)
    print("🎯 Ready to serve requests!")
    print("   Press Ctrl+C to stop the server")
    print("="*60 + "\n")

async def main():
    """Main application entry point"""
    
    logger.info("🚀 Starting StructuraAI...")
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    await check_dependencies()
    
    # Print startup info
    print_startup_info()
    
    # Get configuration
    env = os.getenv('ENVIRONMENT', 'development')
    config_class = get_config(env)
    
    # Run the server
    uvicorn_config = {
        "app": "backend.main:app",
        "host": "0.0.0.0",
        "port": int(os.getenv('PORT', 8000)),
        "reload": config_class.DEBUG,
        "log_level": "info",
        "access_log": True,
        "loop": "asyncio"
    }
    
    # Add SSL config for production
    if env == 'production':
        ssl_keyfile = os.getenv('SSL_KEYFILE')
        ssl_certfile = os.getenv('SSL_CERTFILE')
        if ssl_keyfile and ssl_certfile:
            uvicorn_config.update({
                "ssl_keyfile": ssl_keyfile,
                "ssl_certfile": ssl_certfile
            })
    
    await uvicorn.run(**uvicorn_config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutting down StructuraAI...")
        print("\n" + "="*60)
        print("👋 StructuraAI has been stopped")
        print("   Thank you for using StructuraAI!")
        print("="*60)
    except Exception as e:
        logger.error(f"❌ Failed to start StructuraAI: {e}")
        sys.exit(1)