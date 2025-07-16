# migrate.py
"""Database migration utility script"""

import os
import sys
from pathlib import Path
from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine, text

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config as AppConfig

def get_alembic_config():
    """Get Alembic configuration"""
    alembic_cfg = Config("migrations/alembic.ini")
    alembic_cfg.set_main_option("script_location", "migrations")
    
    # Set database URL from app config
    app_config = AppConfig()
    alembic_cfg.set_main_option("sqlalchemy.url", app_config.SQLALCHEMY_DATABASE_URI)
    
    return alembic_cfg

def init_db():
    """Initialize database with latest migrations"""
    try:
        print("🚀 Initializing database...")
        
        # Create migrations directory if it doesn't exist
        migrations_dir = project_root / "migrations" / "versions"
        migrations_dir.mkdir(parents=True, exist_ok=True)
        
        alembic_cfg = get_alembic_config()
        
        # Check if database exists and has alembic version table
        app_config = AppConfig()
        engine = create_engine(app_config.SQLALCHEMY_DATABASE_URI)
        
        with engine.connect() as conn:
            # Check if alembic_version table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'alembic_version'
                );
            """))
            
            has_alembic = result.fetchone()[0] if app_config.SQLALCHEMY_DATABASE_URI.startswith('postgresql') else True
            
            if not has_alembic:
                print("📝 Stamping database with initial revision...")
                command.stamp(alembic_cfg, "head")
            else:
                print("⬆️ Running database migrations...")
                command.upgrade(alembic_cfg, "head")
                
        print("✅ Database initialization completed!")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)

def create_migration(message):
    """Create a new migration"""
    try:
        print(f"📝 Creating migration: {message}")
        
        alembic_cfg = get_alembic_config()
        command.revision(alembic_cfg, message=message, autogenerate=True)
        
        print("✅ Migration created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating migration: {e}")
        sys.exit(1)

def upgrade_db():
    """Upgrade database to latest version"""
    try:
        print("⬆️ Upgrading database...")
        
        alembic_cfg = get_alembic_config()
        command.upgrade(alembic_cfg, "head")
        
        print("✅ Database upgraded successfully!")
        
    except Exception as e:
        print(f"❌ Error upgrading database: {e}")
        sys.exit(1)

def downgrade_db(revision="base"):
    """Downgrade database to specified revision"""
    try:
        print(f"⬇️ Downgrading database to {revision}...")
        
        alembic_cfg = get_alembic_config()
        command.downgrade(alembic_cfg, revision)
        
        print("✅ Database downgraded successfully!")
        
    except Exception as e:
        print(f"❌ Error downgrading database: {e}")
        sys.exit(1)

def show_history():
    """Show migration history"""
    try:
        alembic_cfg = get_alembic_config()
        command.history(alembic_cfg)
        
    except Exception as e:
        print(f"❌ Error showing history: {e}")
        sys.exit(1)

def show_current():
    """Show current database revision"""
    try:
        alembic_cfg = get_alembic_config()
        command.current(alembic_cfg)
        
    except Exception as e:
        print(f"❌ Error showing current revision: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration utility")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    subparsers.add_parser("init", help="Initialize database with migrations")
    
    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("message", help="Migration message")
    
    # Upgrade command
    subparsers.add_parser("upgrade", help="Upgrade database to latest version")
    
    # Downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade database")
    downgrade_parser.add_argument("--revision", default="base", help="Target revision")
    
    # History command
    subparsers.add_parser("history", help="Show migration history")
    
    # Current command
    subparsers.add_parser("current", help="Show current database revision")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_db()
    elif args.command == "create":
        create_migration(args.message)
    elif args.command == "upgrade":
        upgrade_db()
    elif args.command == "downgrade":
        downgrade_db(args.revision)
    elif args.command == "history":
        show_history()
    elif args.command == "current":
        show_current()
    else:
        parser.print_help()
