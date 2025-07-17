# config.py
import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Basic Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///queryforge.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_timeout': 20,
        'pool_recycle': -1,
        'pool_pre_ping': True
    }
    
    # File upload configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
    
    # Azure OpenAI Configuration
    LLM_CONFIG = {
        'azure': {
            'api_key': os.environ.get('AZURE_OPENAI_API_KEY') or 'your-azure-openai-api-key',
            'endpoint': os.environ.get('AZURE_OPENAI_ENDPOINT') or 'https://your-resource.openai.azure.com/',
            'api_version': os.environ.get('AZURE_OPENAI_API_VERSION') or '2024-02-01',
            'deployment_name': os.environ.get('AZURE_OPENAI_DEPLOYMENT') or 'gpt-4',
            'model_name': os.environ.get('AZURE_OPENAI_MODEL') or 'gpt-4',
            'max_tokens': 2000,
            'temperature': 0.1
        }
    }
    
    # Embedding Configuration
    EMBEDDING_CONFIG = {
        'default_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'batch_size': 32,
        'max_sequence_length': 512,
        'available_models': [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/distilbert-base-nli-mean-tokens',
            'sentence-transformers/paraphrase-MiniLM-L6-v2'
        ]
    }
    
    # Entity Extraction Configuration
    ENTITY_CONFIG = {
        'similarity_threshold': 0.5,
        'max_entities': 20,
        'confidence_threshold': 0.3,
        'max_mappings_per_entity': 3
    }
    
    # Search Configuration
    SEARCH_CONFIG = {
        'default_top_k': 10,
        'max_top_k': 100,
        'min_similarity_score': 0.1,
        'result_timeout_seconds': 30
    }
    
    # AI Prompts
    PROMPTS = {
        'entity_extraction': """
        Extract entities from the following natural language query that could map to database elements.
        
        Query: "{query}"
        
        Available database schema:
        Tables: {tables}
        Columns: {columns}
        Dictionary Terms: {dictionary_terms}
        
        Please identify:
        1. Table names or concepts
        2. Column names or attributes  
        3. Business terms or metrics
        4. Values or filters
        
        Return a JSON array of entities with this format:
        [
            {{
                "text": "entity text",
                "type": "table|column|business_term|value|filter",
                "confidence": 0.8,
                "context": "additional context"
            }}
        ]
        
        Focus on entities that are likely to exist in the database schema.
        """,
        
        'sql_generation': """
        Generate a SQL query based on the natural language query and entity mappings.
        
        Original Query: "{query}"
        
        Extracted Entities: {entities}
        
        Entity to Schema Mappings: {mappings}
        
        Database Schema: {schema}
        
        Relationships: {relationships}
        
        Generate a valid SQL SELECT query that answers the original question.
        Return a JSON object with this format:
        {{
            "sql": "SELECT statement here",
            "explanation": "Brief explanation of the query logic",
            "confidence": 0.8,
            "assumptions": ["any assumptions made"]
        }}
        
        Guidelines:
        - Only use SELECT statements
        - Use proper table and column names from the mappings
        - Include appropriate WHERE clauses for filters
        - Add ORDER BY if results should be sorted
        - Use LIMIT if a specific number of results is requested
        - Join tables when necessary based on relationships
        """,
        
        'answer_generation': """
        Generate a clear, natural language answer based on the SQL query results.
        
        Original Question: "{original_query}"
        SQL Query: "{sql_query}"
        Results: {results}
        Total Rows: {row_count}
        
        Provide a concise, business-friendly answer that:
        1. Directly answers the original question
        2. Highlights key insights from the data
        3. Mentions important numbers or trends
        4. Is easy to understand for non-technical users
        
        Keep the response under 200 words and focus on actionable insights.
        """,
        
        'dictionary_enhancement': """
        Improve this data dictionary definition to be more clear and comprehensive.
        
        Term: "{term}"
        Current Definition: "{definition}"
        Context: This term appears in a {context_type} database with tables: {tables}
        
        Provide an enhanced definition that:
        1. Is clear and concise (under 100 words)
        2. Explains what the term represents in business context
        3. Mentions how it relates to the data
        4. Would be useful for data analysts and business users
        
        Return only the improved definition text.
        """
    }
    
    # Data Processing Configuration
    DATA_PROCESSING = {
        'max_sample_rows': 1000,
        'max_sample_values_per_column': 10,
        'data_type_inference': {
            'numeric_threshold': 0.8,  # 80% of values must be numeric
            'datetime_threshold': 0.8,
            'boolean_threshold': 0.9
        },
        'pii_detection': {
            'email_pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_pattern': r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            'ssn_pattern': r'\b\d{3}-?\d{2}-?\d{4}\b'
        }
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            }
        },
        'handlers': {
            'wsgi': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://flask.logging.wsgi_errors_stream',
                'formatter': 'default'
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'logs/app.log',
                'formatter': 'default'
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi', 'file']
        }
    }
    
    # Security Configuration
    SECURITY_CONFIG = {
        'allowed_sql_keywords': ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT'],
        'forbidden_sql_keywords': ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC'],
        'max_query_length': 5000,
        'rate_limit': {
            'requests_per_minute': 60,
            'requests_per_hour': 1000
        }
    }
    
    # Feature Flags
    FEATURE_FLAGS = {
        'enable_ai_features': True,
        'enable_auto_dictionary': True,
        'enable_advanced_search': True,
        'enable_chat_interface': True,
        'enable_admin_panel': True,
        'enable_file_upload': True,
        'enable_database_connections': True
    }
    
    # Performance Configuration
    PERFORMANCE_CONFIG = {
        'max_concurrent_jobs': 5,
        'job_timeout_seconds': 300,
        'cache_timeout_seconds': 3600,
        'max_search_results': 1000,
        'pagination_size': 20
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DEVELOPMENT = True
    
    # Use environment variables if available, otherwise use defaults
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'sqlite:///queryforge_dev.db'
    
    # More verbose logging in development
    LOGGING_CONFIG = Config.LOGGING_CONFIG.copy()
    LOGGING_CONFIG['root']['level'] = 'DEBUG'
    
    # Relaxed security for development
    SECURITY_CONFIG = Config.SECURITY_CONFIG.copy()
    SECURITY_CONFIG['rate_limit']['requests_per_minute'] = 1000

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    DEVELOPMENT = False
    
    # Set defaults, validation happens in validate_production_config()
    SECRET_KEY = os.environ.get('SECRET_KEY') or Config.SECRET_KEY
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or Config.SQLALCHEMY_DATABASE_URI
    
    # Enhanced security in production
    SECURITY_CONFIG = Config.SECURITY_CONFIG.copy()
    SECURITY_CONFIG['rate_limit']['requests_per_minute'] = 30
    
    # Production logging
    LOGGING_CONFIG = Config.LOGGING_CONFIG.copy()
    LOGGING_CONFIG['root']['level'] = 'WARNING'
    
    @classmethod
    def validate_production_config(cls):
        """Validate production configuration when actually using production"""
        errors = []
        
        if not os.environ.get('SECRET_KEY'):
            errors.append("SECRET_KEY environment variable should be set in production")
        
        if not os.environ.get('DATABASE_URL'):
            errors.append("DATABASE_URL environment variable should be set in production")
        
        if errors:
            import warnings
            for error in errors:
                warnings.warn(f"Production Warning: {error}")
        
        return len(errors) == 0

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable AI features for faster testing
    FEATURE_FLAGS = Config.FEATURE_FLAGS.copy()
    FEATURE_FLAGS['enable_ai_features'] = False
    
    # Reduced timeouts for testing
    PERFORMANCE_CONFIG = Config.PERFORMANCE_CONFIG.copy()
    PERFORMANCE_CONFIG['job_timeout_seconds'] = 30

# Configuration mapping
config_mapping = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name='default'):
    """Get configuration class by name"""
    # Determine config from environment if not specified
    if config_name == 'default':
        config_name = os.environ.get('FLASK_ENV', 'development').lower()
    
    config_class = config_mapping.get(config_name, DevelopmentConfig)
    
    # Validate production config if being used
    if config_class == ProductionConfig:
        config_class.validate_production_config()
    
    return config_class

# Environment-specific configurations
def init_app_config(app):
    """Initialize app-specific configuration"""
    
    # Create required directories
    required_dirs = [
        app.config['UPLOAD_FOLDER'],
        'logs',
        'indexes',
        'backups'
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Set up logging
    if not app.debug:
        import logging
        import logging.config
        logging.config.dictConfig(app.config['LOGGING_CONFIG'])
    
    # Validate critical configurations
    if app.config.get('FEATURE_FLAGS', {}).get('enable_ai_features', False):
        llm_config = app.config.get('LLM_CONFIG', {}).get('azure', {})
        if not llm_config.get('api_key') or llm_config.get('api_key') == 'your-azure-openai-api-key':
            app.logger.warning("Azure OpenAI API key not configured. AI features will be disabled.")
            app.config['FEATURE_FLAGS']['enable_ai_features'] = False

# Database connection validation
def validate_database_config(config):
    """Validate database configuration"""
    try:
        from sqlalchemy import create_engine
        engine = create_engine(config.SQLALCHEMY_DATABASE_URI)
        connection = engine.connect()
        connection.close()
        return True
    except Exception as e:
        print(f"Database validation failed: {str(e)}")
        return False

# Default configuration for immediate use
Config.SQLALCHEMY_DATABASE_URI = 'sqlite:///queryforge.db'
Config.DEBUG = True