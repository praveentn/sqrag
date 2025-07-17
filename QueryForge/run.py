# run.py
#!/usr/bin/env python3
"""
QueryForge Pro - Startup Script
Provides easy startup and management commands for the application
"""

import os
import sys
import argparse
import subprocess
import time
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Configure logging for the startup script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/startup.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are met"""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.warning(".env file not found. Using .env.template as reference")
        if os.path.exists('.env.template'):
            logger.info("Copy .env.template to .env and configure your settings")
        return False
    
    # Check required directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('indexes', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Check if requirements are installed
    try:
        import flask
        import sqlalchemy
        import pandas
        logger.info("✓ Python requirements check passed")
    except ImportError as e:
        logger.error(f"Missing Python requirements: {e}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    return True

def init_database():
    """Initialize the database with tables"""
    logger = logging.getLogger(__name__)
    
    try:
        from app import create_app
        from models import db
        
        logger.info("Initializing database...")
        app = create_app()
        
        with app.app_context():
            # Create all tables
            db.create_all()
            logger.info("✓ Database tables created successfully")
            
            # Check if we can connect
            db.session.execute('SELECT 1')
            logger.info("✓ Database connection verified")
            
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def build_frontend():
    """Build the React frontend"""
    logger = logging.getLogger(__name__)
    
    frontend_dir = Path('static')
    if not frontend_dir.exists():
        logger.error("Frontend directory 'static' not found")
        return False
    
    # Check if Node.js is installed
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        node_version = result.stdout.strip()
        logger.info(f"Node.js version: {node_version}")
    except FileNotFoundError:
        logger.error("Node.js not found. Please install Node.js 16+")
        return False
    
    # Check if npm dependencies are installed
    if not (frontend_dir / 'node_modules').exists():
        logger.info("Installing npm dependencies...")
        try:
            subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
            logger.info("✓ npm dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"npm install failed: {e}")
            return False
    
    # Build frontend
    logger.info("Building frontend...")
    try:
        subprocess.run(['npm', 'run', 'build'], cwd=frontend_dir, check=True)
        logger.info("✓ Frontend built successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Frontend build failed: {e}")
        return False

def start_development():
    """Start the application in development mode"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting QueryForge Pro in development mode...")
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = 'True'
    
    try:
        from app import create_app
        app = create_app('development')
        
        logger.info("=" * 60)
        logger.info("🚀 QueryForge Pro - Enterprise RAG Platform")
        logger.info("=" * 60)
        logger.info("Application URL: http://localhost:5000")
        logger.info("Environment: Development")
        logger.info("Debug Mode: Enabled")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        return False

def start_production():
    """Start the application in production mode"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting QueryForge Pro in production mode...")
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'production'
    os.environ['FLASK_DEBUG'] = 'False'
    
    try:
        # Check if gunicorn is available
        import gunicorn
        
        # Start with gunicorn
        logger.info("Starting with Gunicorn...")
        
        cmd = [
            'gunicorn',
            '--bind', '0.0.0.0:5000',
            '--workers', '4',
            '--worker-class', 'sync',
            '--timeout', '300',
            '--keep-alive', '2',
            '--max-requests', '1000',
            '--max-requests-jitter', '100',
            '--log-level', 'info',
            '--access-logfile', 'logs/access.log',
            '--error-logfile', 'logs/error.log',
            'app:create_app()'
        ]
        
        subprocess.run(cmd, check=True)
        
    except ImportError:
        logger.warning("Gunicorn not found, falling back to Flask development server")
        logger.warning("For production, install gunicorn: pip install gunicorn")
        
        from app import create_app
        app = create_app('production')
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start production server: {e}")
        return False

def run_tests():
    """Run the application tests"""
    logger = logging.getLogger(__name__)
    
    logger.info("Running QueryForge Pro tests...")
    
    try:
        # Check if pytest is available
        import pytest
        
        # Run tests
        exit_code = pytest.main([
            'tests/',
            '-v',
            '--tb=short',
            '--cov=src',
            '--cov-report=term-missing'
        ])
        
        if exit_code == 0:
            logger.info("✓ All tests passed")
            return True
        else:
            logger.error("Some tests failed")
            return False
            
    except ImportError:
        logger.error("pytest not found. Install with: pip install pytest pytest-cov")
        return False
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False

def health_check():
    """Perform a health check on the application"""
    logger = logging.getLogger(__name__)
    
    logger.info("Performing health check...")
    
    try:
        import requests
        import time
        
        # Start the app in background for testing
        # This is a simplified health check
        from app import create_app
        from models import db
        
        app = create_app()
        
        with app.app_context():
            # Test database connection
            db.session.execute('SELECT 1')
            logger.info("✓ Database connection: OK")
            
            # Test configuration
            from config import Config
            Config.validate_config()
            logger.info("✓ Configuration: OK")
            
            # Test file system
            test_dirs = ['uploads', 'indexes', 'logs']
            for dir_name in test_dirs:
                if os.path.exists(dir_name) and os.access(dir_name, os.W_OK):
                    logger.info(f"✓ Directory {dir_name}: OK")
                else:
                    logger.warning(f"Directory {dir_name}: Not accessible")
            
            logger.info("✓ Health check completed successfully")
            return True
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating sample data...")
    
    try:
        from app import create_app
        from models import db, Project, DictionaryEntry
        
        app = create_app()
        
        with app.app_context():
            # Create sample project
            sample_project = Project(
                name="Sample E-commerce Project",
                description="Demo project with sample e-commerce data",
                owner="demo_user"
            )
            db.session.add(sample_project)
            db.session.flush()
            
            # Create sample dictionary entries
            sample_terms = [
                {
                    "term": "Customer",
                    "definition": "An individual or organization that purchases products or services",
                    "category": "business_term",
                    "domain": "sales"
                },
                {
                    "term": "Revenue",
                    "definition": "Total income generated from sales of goods or services",
                    "category": "business_term", 
                    "domain": "finance"
                },
                {
                    "term": "Order",
                    "definition": "A request to purchase products or services",
                    "category": "business_term",
                    "domain": "sales"
                }
            ]
            
            for term_data in sample_terms:
                entry = DictionaryEntry(
                    project_id=sample_project.id,
                    **term_data
                )
                db.session.add(entry)
            
            db.session.commit()
            logger.info(f"✓ Sample project created: {sample_project.name}")
            logger.info(f"✓ Created {len(sample_terms)} sample dictionary entries")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        return False

def main():
    """Main entry point for the startup script"""
    parser = argparse.ArgumentParser(
        description='QueryForge Pro - Enterprise RAG Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                     # Start in development mode
  python run.py --prod             # Start in production mode  
  python run.py --init             # Initialize database
  python run.py --build            # Build frontend
  python run.py --test             # Run tests
  python run.py --health           # Health check
  python run.py --sample-data      # Create sample data
        """
    )
    
    parser.add_argument('--prod', action='store_true', 
                       help='Start in production mode')
    parser.add_argument('--init', action='store_true', 
                       help='Initialize database')
    parser.add_argument('--build', action='store_true', 
                       help='Build frontend')
    parser.add_argument('--test', action='store_true', 
                       help='Run tests')
    parser.add_argument('--health', action='store_true', 
                       help='Perform health check')
    parser.add_argument('--sample-data', action='store_true', 
                       help='Create sample data')
    parser.add_argument('--skip-checks', action='store_true', 
                       help='Skip requirement checks')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("QueryForge Pro - Enterprise RAG Platform")
    logger.info("=" * 50)
    
    # Check requirements unless skipped
    if not args.skip_checks:
        if not check_requirements():
            logger.error("Requirement checks failed. Fix issues above or use --skip-checks")
            sys.exit(1)
    
    # Execute requested action
    try:
        if args.init:
            success = init_database()
        elif args.build:
            success = build_frontend()
        elif args.test:
            success = run_tests()
        elif args.health:
            success = health_check()
        elif args.sample_data:
            success = create_sample_data()
        elif args.prod:
            success = start_production()
        else:
            # Default: development mode
            success = start_development()
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()