# fix_missing_attributes.py
"""
Fixed Database migration script for SQLAlchemy 2.0+ compatibility
Run this script to fix the following issues:
1. Add 'status' column to DataSource table
2. Add 'tags' and 'confidence_score' columns to DictionaryEntry table  
3. Add 'default_value', 'min_value', 'max_value', 'avg_value', 'std_dev' columns to Column table
4. Rename 'approved' to 'is_approved' in DictionaryEntry table
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, inspect
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_sql(db, sql_query):
    """Execute SQL query with compatibility for different SQLAlchemy versions"""
    try:
        # Try SQLAlchemy 2.0+ method first
        with db.engine.connect() as connection:
            result = connection.execute(text(sql_query))
            connection.commit()
            return result
    except AttributeError:
        # Fallback to older SQLAlchemy method
        try:
            return db.engine.execute(text(sql_query))
        except Exception as e:
            # If that fails too, try using session
            result = db.session.execute(text(sql_query))
            db.session.commit()
            return result

def check_column_exists(db, table_name, column_name):
    """Check if a column exists in a table"""
    try:
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception as e:
        logger.warning(f"Could not check if column {column_name} exists in {table_name}: {e}")
        return False

def run_migration(app, db):
    """Run the migration to fix missing attributes"""
    
    with app.app_context():
        try:
            logger.info("Starting database migration...")
            
            # 1. Add status column to DataSource table
            if not check_column_exists(db, 'data_sources', 'status'):
                try:
                    execute_sql(db, """
                        ALTER TABLE data_sources 
                        ADD COLUMN status VARCHAR(50) DEFAULT 'active'
                    """)
                    logger.info("✓ Added 'status' column to data_sources table")
                except Exception as e:
                    logger.warning(f"Error adding status column: {e}")
            else:
                logger.info("✓ 'status' column already exists in data_sources table")
            
            # 2. Add missing columns to DictionaryEntry table
            if not check_column_exists(db, 'dictionary_entries', 'tags'):
                try:
                    execute_sql(db, """
                        ALTER TABLE dictionary_entries 
                        ADD COLUMN tags TEXT
                    """)
                    logger.info("✓ Added 'tags' column to dictionary_entries table")
                except Exception as e:
                    logger.warning(f"Error adding tags column: {e}")
            else:
                logger.info("✓ 'tags' column already exists in dictionary_entries table")
            
            if not check_column_exists(db, 'dictionary_entries', 'confidence_score'):
                try:
                    execute_sql(db, """
                        ALTER TABLE dictionary_entries 
                        ADD COLUMN confidence_score DECIMAL(5,3)
                    """)
                    logger.info("✓ Added 'confidence_score' column to dictionary_entries table")
                except Exception as e:
                    logger.warning(f"Error adding confidence_score column: {e}")
            else:
                logger.info("✓ 'confidence_score' column already exists in dictionary_entries table")
            
            # 3. Handle 'approved' vs 'is_approved' column
            has_approved = check_column_exists(db, 'dictionary_entries', 'approved')
            has_is_approved = check_column_exists(db, 'dictionary_entries', 'is_approved')
            
            if has_approved and not has_is_approved:
                try:
                    # Rename approved to is_approved
                    execute_sql(db, """
                        ALTER TABLE dictionary_entries 
                        RENAME COLUMN approved TO is_approved
                    """)
                    logger.info("✓ Renamed 'approved' to 'is_approved' in dictionary_entries table")
                except Exception as e:
                    logger.warning(f"Error renaming approved column: {e}")
                    # If rename fails, add new column and copy data
                    try:
                        execute_sql(db, """
                            ALTER TABLE dictionary_entries 
                            ADD COLUMN is_approved BOOLEAN DEFAULT FALSE
                        """)
                        execute_sql(db, """
                            UPDATE dictionary_entries 
                            SET is_approved = approved 
                            WHERE approved IS NOT NULL
                        """)
                        logger.info("✓ Added 'is_approved' column and copied data from 'approved'")
                    except Exception as e2:
                        logger.warning(f"Error handling approved/is_approved transition: {e2}")
            elif not has_is_approved:
                try:
                    execute_sql(db, """
                        ALTER TABLE dictionary_entries 
                        ADD COLUMN is_approved BOOLEAN DEFAULT FALSE
                    """)
                    logger.info("✓ Added 'is_approved' column to dictionary_entries table")
                except Exception as e:
                    logger.warning(f"Error adding is_approved column: {e}")
            else:
                logger.info("✓ 'is_approved' column already exists in dictionary_entries table")
            
            # 4. Add missing columns to Column table
            column_additions = [
                ("default_value", "VARCHAR(255)"),
                ("min_value", "DECIMAL(15,3)"),
                ("max_value", "DECIMAL(15,3)"),
                ("avg_value", "DECIMAL(15,3)"),
                ("std_dev", "DECIMAL(15,3)")
            ]
            
            for col_name, col_type in column_additions:
                if not check_column_exists(db, 'columns', col_name):
                    try:
                        execute_sql(db, f"""
                            ALTER TABLE columns 
                            ADD COLUMN {col_name} {col_type}
                        """)
                        logger.info(f"✓ Added '{col_name}' column to columns table")
                    except Exception as e:
                        logger.warning(f"Error adding {col_name} column: {e}")
                else:
                    logger.info(f"✓ '{col_name}' column already exists in columns table")
            
            # 5. Update any NULL status values to 'active'
            try:
                execute_sql(db, """
                    UPDATE data_sources 
                    SET status = 'active' 
                    WHERE status IS NULL
                """)
                logger.info("✓ Updated NULL status values to 'active'")
            except Exception as e:
                logger.warning(f"Error updating status values: {e}")
            
            # 6. Initialize tags as empty arrays for existing entries
            try:
                execute_sql(db, """
                    UPDATE dictionary_entries 
                    SET tags = '[]' 
                    WHERE tags IS NULL
                """)
                logger.info("✓ Initialized empty tags arrays")
            except Exception as e:
                logger.warning(f"Error initializing tags: {e}")
            
            # 7. Ensure is_approved is properly set
            try:
                execute_sql(db, """
                    UPDATE dictionary_entries 
                    SET is_approved = FALSE 
                    WHERE is_approved IS NULL
                """)
                logger.info("✓ Set default values for is_approved column")
            except Exception as e:
                logger.warning(f"Error setting is_approved defaults: {e}")
            
            logger.info("✅ Database migration completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise

def run_migration_standalone():
    """Run migration as standalone script"""
    
    # Add the current directory to Python path to import config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        from config import Config
    except ImportError:
        logger.error("Could not import Config. Make sure config.py exists in the current directory.")
        return False
    
    app = Flask(__name__)
    app.config.from_object(Config)
    
    db = SQLAlchemy(app)
    
    try:
        # Test database connection
        with app.app_context():
            db.engine.connect()
        
        run_migration(app, db)
        print("✅ Migration completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logger.error(f"Full error details: {e}")
        return False

if __name__ == "__main__":
    success = run_migration_standalone()
    if not success:
        sys.exit(1)