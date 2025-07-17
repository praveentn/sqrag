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
from sqlalchemy import text

# Force UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    # Set environment variables for UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Configure logging for the startup script with Windows UTF-8 support"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging with UTF-8 encoding for Windows
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create handlers with UTF-8 encoding
    handlers = []
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler with UTF-8 encoding
    try:
        file_handler = logging.FileHandler('logs/startup.log', mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file log handler: {e}")
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
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

def create_sample_data():
    """Create sample data for testing"""
    logger = logging.getLogger(__name__)
    
    try:
        from app import create_app
        from models import db, Project
        
        logger.info("Creating sample data...")
        app = create_app()
        
        with app.app_context():
            # Create sample project if none exists
            if Project.query.count() == 0:
                sample_project = Project(
                    name="Sample Project",
                    description="A sample project for testing QueryForge Pro",
                    created_by="system"
                )
                db.session.add(sample_project)
                db.session.commit()
                logger.info("✓ Sample project created")
            else:
                logger.info("✓ Sample data already exists")
                
        return True
        
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")
        return False

def health_check():
    """Perform application health check"""
    logger = logging.getLogger(__name__)
    
    try:
        import requests
        import time
        
        logger.info("Performing health check...")
        
        # Start the app in a separate process for testing
        # This is a basic implementation - in production you'd use proper testing
        
        # Check if port 5000 is available
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5000))
        sock.close()
        
        if result == 0:
            logger.info("✓ Application is running on port 5000")
            
            # Try to make a health check request
            try:
                response = requests.get('http://localhost:5000/api/health', timeout=5)
                if response.status_code == 200:
                    logger.info("✓ Health endpoint responding correctly")
                    return True
                else:
                    logger.error(f"Health endpoint returned status {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Could not reach health endpoint: {e}")
                return False
        else:
            logger.error("Application is not running on port 5000")
            return False
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

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
            db.session.execute(text('SELECT 1'))
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
    # Set UTF-8 encoding for Windows
    if sys.platform.startswith('win'):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
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
    if sys.platform.startswith('win'):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
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