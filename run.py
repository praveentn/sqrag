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

def setup_sample_data():
    """Setup sample data for demonstration (optional)"""
    try:
        with app.app_context():
            from models import DictionaryEntry, DataSource, Table, Column
            import pandas as pd
            import os
            
            # Check if we already have sample data
            if DataSource.query.count() > 0:
                logger.info("Sample data sources already exist, skipping setup")
                return
            
            # Create sample CSV data
            sample_customers = pd.DataFrame({
                'customer_id': [1, 2, 3, 4, 5],
                'customer_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
                'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com'],
                'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
                'signup_date': ['2024-01-15', '2024-01-20', '2024-02-01', '2024-02-10', '2024-02-15']
            })
            
            sample_orders = pd.DataFrame({
                'order_id': [101, 102, 103, 104, 105, 106],
                'customer_id': [1, 2, 1, 3, 2, 4],
                'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Webcam'],
                'order_amount': [999.99, 29.99, 79.99, 299.99, 199.99, 89.99],
                'order_date': ['2024-03-01', '2024-03-02', '2024-03-05', '2024-03-07', '2024-03-10', '2024-03-12'],
                'status': ['Completed', 'Completed', 'Pending', 'Completed', 'Shipped', 'Completed']
            })
            
            # Ensure uploads directory exists
            uploads_dir = os.path.join(project_root, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Save sample CSV files
            customers_file = os.path.join(uploads_dir, 'sample_customers.csv')
            orders_file = os.path.join(uploads_dir, 'sample_orders.csv')
            
            sample_customers.to_csv(customers_file, index=False)
            sample_orders.to_csv(orders_file, index=False)
            
            # Create data sources
            from services.data_source_manager import DataSourceManager
            dsm = DataSourceManager()
            
            # Create customers data source
            customers_source = dsm.create_source(
                name='Sample Customers',
                type='csv',
                file_path=customers_file,
                _metadata={'description': 'Sample customer data for demonstration'}
            )
            
            # Create orders data source
            orders_source = dsm.create_source(
                name='Sample Orders',
                type='csv',
                file_path=orders_file,
                _metadata={'description': 'Sample order data for demonstration'}
            )
            
            logger.info("Created sample data sources with CSV files")
            
            # Create some sample dictionary entries if they don't exist
            if DictionaryEntry.query.count() == 0:
                sample_entries = [
                    {
                        'term': 'customer',
                        'definition': 'An individual or organization that purchases goods or services',
                        'category': 'business_term',
                        'approved': True
                    },
                    {
                        'term': 'revenue',
                        'definition': 'The total amount of income generated by a business',
                        'category': 'financial',
                        'approved': True
                    },
                    {
                        'term': 'order',
                        'definition': 'A request to purchase goods or services',
                        'category': 'business_term',
                        'approved': True
                    }
                ]
                
                for entry_data in sample_entries:
                    entry = DictionaryEntry(**entry_data)
                    db.session.add(entry)
                
                db.session.commit()
                logger.info(f"Created {len(sample_entries)} sample dictionary entries")
            
    except Exception as e:
        logger.error(f"Error setting up sample data: {str(e)}")

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
        logger.info("Starting RAG Data Platform...")
        
        # Check dependencies
        # check_dependencies()
        # check_optional_dependencies()
        
        # Create necessary directories
        # create_directories()
        
        # Initialize database
        init_database()
        
        # Setup sample data (optional)
        # setup_sample_data()
        
        # Print startup information
        print_startup_info()
        
        # Start the Flask application
        logger.info("Starting Flask development server...")
        app.run(
            host='0.0.0.0',
            port=5555,
            debug=True,
            use_reloader=False  # Disable reloader to avoid running this script twice
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
