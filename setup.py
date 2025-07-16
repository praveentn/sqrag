# setup.py
"""
RAG Data Platform Setup Script

This script helps set up the RAG Data Platform with all necessary configurations,
dependencies, and initial data.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request
import zipfile
import platform

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🚀 RAG Data Platform Setup                          ║
    ║                                                              ║
    ║         Transform your data into conversational insights     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")

def check_git():
    """Check if Git is available"""
    print("📦 Checking Git availability...")
    
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("✅ Git is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Git not found (optional for development)")
        return False

def create_virtual_environment():
    """Create Python virtual environment"""
    print("🏠 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        sys.exit(1)

def get_python_executable():
    """Get the path to Python executable in virtual environment"""
    if platform.system() == "Windows":
        return Path("venv/Scripts/python.exe")
    else:
        return Path("venv/bin/python")

def get_pip_executable():
    """Get the path to pip executable in virtual environment"""
    if platform.system() == "Windows":
        return Path("venv/Scripts/pip.exe")
    else:
        return Path("venv/bin/pip")

def install_dependencies():
    """Install Python dependencies"""
    print("📚 Installing Python dependencies...")
    
    pip_path = get_pip_executable()
    
    try:
        # Upgrade pip first
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try running manually: pip install -r requirements.txt")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    print("📁 Creating application directories...")
    
    directories = [
        "uploads",
        "logs", 
        "indexes",
        "indexes/faiss",
        "backups",
        "static",
        "migrations/versions"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully")

def setup_environment_file():
    """Set up environment configuration"""
    print("⚙️  Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("🔧 Please edit .env file with your configuration")
    else:
        # Create basic .env file
        basic_env = """# Basic RAG Platform Configuration
SECRET_KEY=dev-secret-key-change-in-production
FLASK_DEBUG=True
DATABASE_URL=sqlite:///rag_platform.db
LLM_PROVIDER=mock
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
"""
        with open(env_file, 'w') as f:
            f.write(basic_env)
        print("✅ Created basic .env file")
        print("🔧 Consider configuring LLM provider for full functionality")

def initialize_database():
    """Initialize database"""
    print("🗄️  Initializing database...")
    
    python_path = get_python_executable()
    
    try:
        # Run database initialization
        subprocess.run([str(python_path), "migrate.py", "init"], check=True)
        print("✅ Database initialized successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize database: {e}")
        print("💡 You can initialize manually later with: python migrate.py init")

def download_sample_data():
    """Download sample data files"""
    print("📊 Setting up sample data...")
    
    # Create sample CSV data
    sample_csv = Path("uploads/sample_customers.csv")
    if not sample_csv.exists():
        sample_data = """customer_id,customer_name,email,city,country,signup_date,total_orders,total_spent
1,John Doe,john.doe@email.com,New York,USA,2023-01-15,12,1250.00
2,Jane Smith,jane.smith@email.com,London,UK,2023-02-20,8,890.50
3,Bob Johnson,bob.johnson@email.com,Toronto,Canada,2023-03-10,15,2100.75
4,Alice Brown,alice.brown@email.com,Sydney,Australia,2023-01-05,6,450.25
5,Charlie Wilson,charlie.wilson@email.com,Berlin,Germany,2023-04-12,20,3200.00
"""
        sample_csv.parent.mkdir(exist_ok=True)
        with open(sample_csv, 'w') as f:
            f.write(sample_data)
        print("✅ Sample CSV data created")
    
    # Create sample Excel data
    try:
        import pandas as pd
        
        sample_excel = Path("uploads/sample_products.xlsx")
        if not sample_excel.exists():
            products_data = {
                'product_id': [1, 2, 3, 4, 5],
                'product_name': ['Laptop Pro', 'Wireless Mouse', 'USB Keyboard', 'Monitor 24"', 'Webcam HD'],
                'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
                'price': [1299.99, 49.99, 79.99, 299.99, 89.99],
                'stock_quantity': [50, 200, 150, 75, 120],
                'supplier': ['TechCorp', 'AccessoryInc', 'AccessoryInc', 'DisplayTech', 'CameraCorp']
            }
            
            df = pd.DataFrame(products_data)
            df.to_excel(sample_excel, index=False)
            print("✅ Sample Excel data created")
            
    except ImportError:
        print("⚠️  Pandas not available, skipping Excel sample data")

def run_tests():
    """Run basic tests to verify setup"""
    print("🧪 Running setup verification...")
    
    python_path = get_python_executable()
    
    try:
        # Test imports
        test_script = """
import sys
sys.path.insert(0, '.')

try:
    from app import app
    from models import db
    from config import Config
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

try:
    config = Config()
    print("✅ Configuration loaded")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)

print("✅ Basic setup verification passed")
"""
        
        result = subprocess.run(
            [str(python_path), "-c", test_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Setup verification failed:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run verification: {e}")

def print_next_steps():
    """Print next steps for the user"""
    python_cmd = "python" if platform.system() == "Windows" else "python"
    activate_cmd = "venv\\Scripts\\activate" if platform.system() == "Windows" else "source venv/bin/activate"
    
    next_steps = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     🎉 Setup Complete!                      ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📋 Next Steps:
    
    1. 🔧 Configure your LLM provider (optional but recommended):
       • Edit .env file with your API keys
       • For Azure OpenAI: Set AZURE_OPENAI_* variables
       • For OpenAI: Set OPENAI_API_KEY
       • Or use LLM_PROVIDER=mock for development
    
    2. 🚀 Start the application:
       • {activate_cmd}
       • {python_cmd} run.py
       • Open http://localhost:5000 in your browser
    
    3. 📊 Add your first data source:
       • Use the sample files in uploads/ folder
       • Or connect to your own database
       • Go to Data Sources → Add Data Source
    
    4. 🧠 Build your knowledge base:
       • Generate dictionary: Dictionary → Auto-Generate
       • Create embeddings: Embeddings → Create Index
       • Start chatting: Chat Workspace
    
    📚 Helpful Commands:
    • {python_cmd} migrate.py --help     # Database migrations
    • {python_cmd} run.py --help         # Application options
    
    🆘 Need Help?
    • Check README.md for detailed instructions
    • Visit the admin panel for system monitoring
    • Check logs/ directory for troubleshooting
    
    Happy data querying! 🚀
    """
    
    print(next_steps)

def main():
    """Main setup function"""
    print_banner()
    
    try:
        # Setup steps
        check_python_version()
        check_git()
        create_virtual_environment()
        install_dependencies()
        create_directories()
        setup_environment_file()
        initialize_database()
        download_sample_data()
        run_tests()
        
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during setup: {e}")
        print("Please check the error and try again, or set up manually using README.md")
        sys.exit(1)

if __name__ == "__main__":
    main()

