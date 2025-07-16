# config.py
import os
import yaml
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///rag_platform.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # File upload settings
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    
    # LLM Configuration
    LLM_CONFIG = {
        'provider': os.environ.get('LLM_PROVIDER', 'azure_openai'),
        'azure': {
            'endpoint': os.environ.get('AZURE_OPENAI_ENDPOINT', 'https://your-endpoint.openai.azure.com/'),
            'api_key': os.environ.get('AZURE_OPENAI_API_KEY', ''),
            'api_version': os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'),
            'deployment_name': os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-4'),
            'model_name': os.environ.get('AZURE_OPENAI_MODEL', 'gpt-4'),
        },
        'openai': {
            'api_key': os.environ.get('OPENAI_API_KEY', ''),
            'model': os.environ.get('OPENAI_MODEL', 'gpt-4'),
        },
        'anthropic': {
            'api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
            'model': os.environ.get('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
        }
    }
    
    # Embedding Configuration
    EMBEDDING_CONFIG = {
        'default_model': os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        'batch_size': int(os.environ.get('EMBEDDING_BATCH_SIZE', '32')),
        'max_length': int(os.environ.get('EMBEDDING_MAX_LENGTH', '512')),
        'backends': {
            'faiss': {
                'index_type': 'IndexFlatIP',  # Inner product for cosine similarity
                'index_path': os.environ.get('FAISS_INDEX_PATH', 'indexes/faiss'),
            },
            'pgvector': {
                'connection_string': os.environ.get('PGVECTOR_URL', 'postgresql://user:pass@localhost/vectordb'),
                'table_prefix': 'embedding_',
            },
            'tfidf': {
                'max_features': int(os.environ.get('TFIDF_MAX_FEATURES', '10000')),
                'ngram_range': (1, 2),
            }
        }
    }
    
    # SQL Execution Configuration
    SQL_CONFIG = {
        'max_execution_time': int(os.environ.get('SQL_MAX_EXECUTION_TIME', '30')),  # seconds
        'max_rows': int(os.environ.get('SQL_MAX_ROWS', '10000')),
        'allowed_statements': ['SELECT', 'WITH', 'EXPLAIN'],
        'blocked_keywords': ['DELETE', 'UPDATE', 'INSERT', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE'],
        'preview_rows': int(os.environ.get('SQL_PREVIEW_ROWS', '100')),
    }
    
    # Entity Extraction Configuration
    ENTITY_CONFIG = {
        'similarity_threshold': float(os.environ.get('ENTITY_SIMILARITY_THRESHOLD', '0.7')),
        'max_entities': int(os.environ.get('MAX_ENTITIES', '20')),
        'fuzzy_threshold': float(os.environ.get('FUZZY_THRESHOLD', '0.8')),
        'ranking_weights': {
            'exact_match': float(os.environ.get('EXACT_MATCH_WEIGHT', '1.0')),
            'fuzzy_match': float(os.environ.get('FUZZY_MATCH_WEIGHT', '0.8')),
            'embedding_similarity': float(os.environ.get('EMBEDDING_SIMILARITY_WEIGHT', '0.6')),
            'table_importance': float(os.environ.get('TABLE_IMPORTANCE_WEIGHT', '0.3')),
        }
    }
    
    # RAG Pipeline Configuration
    RAG_CONFIG = {
        'max_context_length': int(os.environ.get('RAG_MAX_CONTEXT_LENGTH', '4000')),
        'max_candidate_tables': int(os.environ.get('RAG_MAX_CANDIDATE_TABLES', '5')),
        'confidence_threshold': float(os.environ.get('RAG_CONFIDENCE_THRESHOLD', '0.5')),
        'retry_attempts': int(os.environ.get('RAG_RETRY_ATTEMPTS', '3')),
    }
    
    # System Prompts
    PROMPTS = {
        'entity_extraction': '''
You are an expert data analyst. Given a natural language query, extract entities that might correspond to table names, column names, or business terms.

Query: {query}

Available context:
- Tables: {tables}
- Columns: {columns}
- Dictionary terms: {dictionary_terms}

Extract relevant entities and return them as a JSON list with confidence scores:
[{{"entity": "entity_name", "type": "table|column|term", "confidence": 0.95}}]
''',
        
        'sql_generation': '''
You are an expert SQL analyst. Generate an executable SQL query based on the user's natural language request.

User Query: {query}

Available Tables and Schemas:
{schemas}

Entity Context:
{entities}

Guidelines:
1. Use only the provided tables and columns
2. Generate valid SQL for the detected database type
3. Include appropriate JOINs when needed
4. Add meaningful column aliases
5. Use LIMIT clause for large result sets
6. Return only SELECT statements

Respond with JSON:
{{
    "sql": "SELECT ...",
    "rationale": "Explanation of the query logic",
    "confidence": 0.95,
    "tables_used": ["table1", "table2"],
    "assumptions": ["Any assumptions made"]
}}
''',
        
        'sql_refinement': '''
You are an expert SQL analyst. The previous query didn't meet user expectations. Refine the SQL based on their feedback.

Original Query: {original_query}
Generated SQL: {generated_sql}
Execution Result: {result}
User Feedback: {feedback}

Available Tables and Schemas:
{schemas}

Refine the SQL to better match user expectations. Respond with JSON:
{{
    "sql": "SELECT ...",
    "rationale": "Explanation of changes made",
    "confidence": 0.95,
    "changes": ["List of changes made"]
}}
'''
    }
    
    # Database connector settings
    DATABASE_CONNECTORS = {
        'postgresql': {
            'driver': 'postgresql+psycopg2',
            'default_port': 5432,
        },
        'mysql': {
            'driver': 'mysql+pymysql',
            'default_port': 3306,
        },
        'sqlite': {
            'driver': 'sqlite',
            'default_port': None,
        },
        'mssql': {
            'driver': 'mssql+pyodbc',
            'default_port': 1433,
        },
        'oracle': {
            'driver': 'oracle+cx_oracle',
            'default_port': 1521,
        }
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'json': {
                'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            }
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'INFO',
                'formatter': 'json',
                'class': 'logging.FileHandler',
                'filename': 'logs/rag_platform.log',
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    
    # Cache configuration
    CACHE_CONFIG = {
        'type': os.environ.get('CACHE_TYPE', 'simple'),
        'redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
        'default_timeout': int(os.environ.get('CACHE_TIMEOUT', '300')),
    }
    
    # Security settings
    SECURITY_CONFIG = {
        'max_query_length': int(os.environ.get('MAX_QUERY_LENGTH', '1000')),
        'rate_limit': os.environ.get('RATE_LIMIT', '100 per hour'),
        'allowed_file_extensions': ['.csv', '.xlsx', '.xls', '.json'],
        'max_file_size': 100 * 1024 * 1024,  # 100MB
    }
    
    @classmethod
    def load_from_file(cls, config_path='config.yaml'):
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Update class attributes with file config
            for key, value in file_config.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
        
        return cls
    
    @classmethod
    def get_database_uri(cls, db_type='default'):
        """Get database URI for specific database type"""
        if db_type == 'default':
            return cls.SQLALCHEMY_DATABASE_URI
        
        # For external database connections
        connectors = cls.DATABASE_CONNECTORS
        if db_type in connectors:
            return f"{connectors[db_type]['driver']}://..."
        
        return None
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration settings"""
        required_settings = [
            'SECRET_KEY',
            'SQLALCHEMY_DATABASE_URI',
        ]
        
        missing = []
        for setting in required_settings:
            if not getattr(cls, setting, None):
                missing.append(setting)
        
        if missing:
            raise ValueError(f"Missing required configuration: {missing}")
        
        return True

# Development configuration
class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///rag_platform_dev.db'

# Testing configuration
class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Production configuration
class ProductionConfig(Config):
    DEBUG = False
    # Use environment variables for production settings

# Configuration factory
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(env_name='default'):
    """Get configuration based on environment"""
    return config.get(env_name, DevelopmentConfig)
