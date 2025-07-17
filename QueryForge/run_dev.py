# run_dev.py
"""
Quick startup script for development environment
"""
import os
import sys

def setup_development_environment():
    """Set up development environment and run the app"""
    
    print("🚀 QueryForge Pro - Development Setup")
    print("=" * 50)
    
    # Set environment variables for development
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Set default values if not already set
    if not os.environ.get('SECRET_KEY'):
        os.environ['SECRET_KEY'] = 'dev-secret-key-for-testing'
        print("✅ Using development SECRET_KEY")
    
    # Check if we need to create database
    db_file = 'queryforge.db'
    need_db_init = not os.path.exists(db_file)
    
    if need_db_init:
        print("📊 Initializing database...")
        try:
            from app import create_app
            from models import db
            
            app = create_app('development')
            with app.app_context():
                db.create_all()
                print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
            return False
    else:
        print("✅ Database already exists")
        # Check if existing database has owner field issues
        try:
            from app import create_app
            from models import db, Project
            
            app = create_app('development')
            with app.app_context():
                # Try to update any projects with NULL owner
                result = db.session.execute(
                    "UPDATE projects SET owner = 'default_user' WHERE owner IS NULL OR owner = ''"
                )
                if result.rowcount > 0:
                    print(f"🔧 Fixed {result.rowcount} projects with missing owner")
                    db.session.commit()
        except Exception as e:
            print(f"⚠️  Note: {e}")
            print("💡 If you have issues, try: python fix_database.py")
    
    # Create required directories
    required_dirs = ['uploads', 'logs', 'indexes', 'backups']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    print("\n🎯 Starting QueryForge Pro...")
    print("📍 URL: http://localhost:5000")
    print("🔧 Environment: Development")
    print("💡 To stop: Press Ctrl+C")
    print("=" * 50)
    
    return True

if __name__ == '__main__':
    if setup_development_environment():
        try:
            from app import app
            app.run(host='0.0.0.0', port=5000, debug=True)
        except KeyboardInterrupt:
            print("\n\n👋 QueryForge Pro stopped. Thanks for using!")
        except Exception as e:
            print(f"\n❌ Error starting application: {e}")
            sys.exit(1)
    else:
        print("❌ Failed to set up development environment")
        sys.exit(1)