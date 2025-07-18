# run_dev.py
"""
Quick development setup script for QueryForge
This script handles environment setup and starts the application
"""
import os
import sys
import logging
from datetime import datetime

def setup_environment():
    """Set up development environment variables"""
    print("🔧 Setting up development environment...")
    
    # Set Flask environment
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = 'False'  # Set to False to avoid reloader issues
    
    # Set default values for missing environment variables
    if 'SECRET_KEY' not in os.environ:
        os.environ['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    
    if 'DATABASE_URL' not in os.environ:
        os.environ['DATABASE_URL'] = 'sqlite:///queryforge.db'
    
    # Optional Azure OpenAI (will use defaults if not set)
    if 'AZURE_OPENAI_API_KEY' not in os.environ:
        os.environ['AZURE_OPENAI_API_KEY'] = 'your-azure-openai-api-key'
    
    if 'AZURE_OPENAI_ENDPOINT' not in os.environ:
        os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://your-resource.openai.azure.com/'
    
    print("✅ Environment variables set")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'flask',
        'flask_sqlalchemy', 
        'flask_cors',
        'pandas',
        'numpy',
        'sentence_transformers',
        'scikit-learn',
        'faiss-cpu',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            # Handle alternative import names
            if package == 'flask_sqlalchemy':
                try:
                    __import__('flask_sqlalchemy')
                except ImportError:
                    missing_packages.append(package)
            elif package == 'flask_cors':
                try:
                    __import__('flask_cors')
                except ImportError:
                    missing_packages.append(package)
            elif package == 'sentence_transformers':
                try:
                    __import__('sentence_transformers')
                except ImportError:
                    missing_packages.append(package)
            elif package == 'scikit-learn':
                try:
                    __import__('sklearn')
                except ImportError:
                    missing_packages.append('scikit-learn')
            elif package == 'faiss-cpu':
                try:
                    __import__('faiss')
                except ImportError:
                    missing_packages.append('faiss-cpu')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("\n📥 Installing missing packages...")
        
        import subprocess
        for package in missing_packages:
            try:
                print(f"   Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"   ✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}")
                print(f"   Please run: pip install {package}")
                return False
    
    print("✅ All dependencies available")
    return True

def initialize_database():
    """Initialize the database"""
    print("🗄️  Initializing database...")
    
    try:
        # Import and create app
        from app import create_app
        from models import db
        
        app = create_app('development')
        
        with app.app_context():
            # Create all tables
            db.create_all()
            
            # Check if we have any projects
            from models import Project
            project_count = Project.query.count()
            
            if project_count == 0:
                # Create a default project
                default_project = Project(
                    name="Default Project",
                    description="Default project created during setup",
                    owner="system"
                )
                db.session.add(default_project)
                db.session.commit()
                print("✅ Created default project")
            else:
                print(f"✅ Database already has {project_count} project(s)")
            
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        return False

def create_upload_directory():
    """Create upload directory if it doesn't exist"""
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        print(f"✅ Created upload directory: {upload_dir}")
    else:
        print(f"✅ Upload directory exists: {upload_dir}")

def start_application():
    """Start the Flask application"""
    print("🚀 Starting QueryForge application...")
    
    try:
        from app import app
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     🔥 QueryForge Pro v1.0.0                     ║
║                    Enterprise RAG Platform                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🌐 Application URL: http://localhost:5000                       ║
║  📊 Health Check:    http://localhost:5000/api/health            ║
║  🔧 Admin Panel:     http://localhost:5000 (Admin Tab)           ║
║                                                                  ║
║  📁 Upload files, create dictionaries, search data!             ║
║  💬 Chat with your data using natural language                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

🎯 Quick Start:
1. Open http://localhost:5000 in your browser
2. Create a new project or use the default one
3. Upload a CSV/Excel file in the Data Sources tab
4. Generate dictionary terms in the Dictionary tab
5. Create embeddings in the Embeddings tab
6. Chat with your data!

⚠️  Note: Press Ctrl+C to stop the server

""")
        
        # Start the Flask app
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False,  # Set to False to avoid reloader issues
            use_reloader=False  # Disable reloader for clean startup
        )
        
    except KeyboardInterrupt:
        print("\n👋 Stopping QueryForge application...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the development setup"""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              🔥 QueryForge Pro - Development Setup                ║
║                     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                     ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        sys.exit(1)
    
    # Step 3: Create directories
    create_upload_directory()
    
    # Step 4: Initialize database
    if not initialize_database():
        print("\n❌ Database initialization failed")
        sys.exit(1)
    
    # Step 5: Start application
    start_application()

if __name__ == '__main__':
    main()