# run.py
"""
RAG Data Platform Startup Script

This script initializes and runs the RAG Data Platform application.
It handles database initialization, creates necessary directories,
and starts the Flask development server.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after path setup
from app import app, db
from models import *  # Import all models
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the application"""
    directories = [
        'uploads',
        'logs',
        'indexes',
        'indexes/faiss',
        'backups'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def init_database():
    """Initialize the database with tables"""
    try:
        with app.app_context():
            # Create all tables
            db.create_all()
            logger.info("Database tables created successfully")
            
            # Check if we have any data sources
            from models import DataSource
            source_count = DataSource.query.count()
            logger.info(f"Found {source_count} existing data sources")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'flask',
        'sqlalchemy',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("All core dependencies are available")

def check_optional_dependencies():
    """Check optional dependencies and warn if missing"""
    optional_packages = {
        'sentence_transformers': 'Embedding functionality will be limited',
        'faiss': 'FAISS similarity search will not be available',
        'openai': 'OpenAI/Azure OpenAI integration will not work',
        'anthropic': 'Anthropic Claude integration will not work'
    }
    
    for package, warning in optional_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ Optional package '{package}' is available")
        except ImportError:
            logger.warning(f"⚠ Optional package '{package}' not found: {warning}")

def cleanup_sample_data():
    """Clean up all sample data from the database"""
    try:
        with app.app_context():
            from models import DataSource, DictionaryEntry, EmbeddingIndex, ChatSession, QueryExecution
            
            # Count existing data
            sources_count = DataSource.query.count()
            dict_count = DictionaryEntry.query.count()
            embed_count = EmbeddingIndex.query.count()
            chat_count = ChatSession.query.count()
            query_count = QueryExecution.query.count()
            
            if sources_count == 0 and dict_count == 0 and embed_count == 0:
                logger.info("No data to clean up")
                return
            
            # Delete all data
            QueryExecution.query.delete()
            ChatSession.query.delete()  # This will cascade to ChatMessage
            EmbeddingIndex.query.delete()
            DictionaryEntry.query.delete()
            
            # Delete data sources (this will cascade to tables and columns)
            for source in DataSource.query.all():
                # Delete associated files
                if source.file_path and os.path.exists(source.file_path):
                    try:
                        os.remove(source.file_path)
                        logger.info(f"Deleted file: {source.file_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete file {source.file_path}: {str(e)}")
            
            DataSource.query.delete()
            
            db.session.commit()
            
            logger.info(f"Cleaned up: {sources_count} sources, {dict_count} dictionary entries, "
                       f"{embed_count} embeddings, {chat_count} chats, {query_count} queries")
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error cleaning up sample data: {str(e)}")
        raise

def print_startup_info():
    """Print useful startup information"""
    config = Config()
    
    print("\n" + "="*60)
    print("🚀 RAG Data Platform Starting Up")
    print("="*60)
    print(f"📊 Database: {config.SQLALCHEMY_DATABASE_URI}")
    print(f"🤖 LLM Provider: {config.LLM_CONFIG['provider']}")
    print(f"🔍 Default Embedding Model: {config.EMBEDDING_CONFIG['default_model']}")
    print(f"📁 Upload Directory: {project_root / 'uploads'}")
    print(f"💾 Index Directory: {project_root / 'indexes'}")
    print("="*60)
    print("\n📝 Getting Started:")
    print("1. Open your browser to http://localhost:5555")
    print("2. Add your first data source in the Data Sources section")
    print("3. Generate a dictionary from your data")
    print("4. Build embedding indexes for semantic search")
    print("5. Start chatting with your data!")
    print("\n💡 Tips:")
    print("- For production, set environment variables for API keys")
    print("- Use PostgreSQL for better performance with large datasets")
    print("- Enable Azure OpenAI or OpenAI for full LLM functionality")
    print("="*60)

def main():
    """Main startup function"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='RAG Data Platform')
        parser.add_argument('--cleanup', action='store_true', 
                          help='Clean up all existing data before starting')
        parser.add_argument('--sample-data', action='store_true',
                          help='Create sample data for demonstration')
        parser.add_argument('--port', type=int, default=5555,
                          help='Port to run the server on (default: 5555)')
        parser.add_argument('--host', default='0.0.0.0',
                          help='Host to bind to (default: 0.0.0.0)')
        parser.add_argument('--debug', action='store_true',
                          help='Run in debug mode')
        
        args = parser.parse_args()
        
        logger.info("Starting RAG Data Platform...")
        
        # Check dependencies
        check_dependencies()
        check_optional_dependencies()
        
        # Create necessary directories
        create_directories()
        
        # Initialize database
        init_database()
        
        # Print startup information
        print_startup_info()
        
        # Start the Flask application
        logger.info("Starting Flask development server...")
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False  # Disable reloader to avoid running this script twice
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
