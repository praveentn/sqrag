# models.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON, ARRAY
from sqlalchemy import Text, LargeBinary, Numeric, Index as SQLIndex
import json
import pickle

db = SQLAlchemy()

class Project(db.Model):
    """Main project entity - all other entities are tied to projects"""
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(Text)
    owner = db.Column(db.String(100), nullable=False, default='default_user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = db.Column(db.String(50), default='active')  # active, archived, deleted
    
    # Relationships - Fixed naming
    sources = db.relationship('DataSource', backref='project', lazy=True, cascade='all, delete-orphan')
    dictionary_entries = db.relationship('DictionaryEntry', backref='project', lazy=True, cascade='all, delete-orphan')
    embeddings = db.relationship('Embedding', backref='project', lazy=True, cascade='all, delete-orphan')
    indexes = db.relationship('Index', backref='project', lazy=True, cascade='all, delete-orphan')
    search_logs = db.relationship('SearchLog', backref='project', lazy=True, cascade='all, delete-orphan')
    nlq_feedback = db.relationship('NLQFeedback', backref='project', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'owner': self.owner,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'status': self.status,
            'sources_count': len(self.sources),
            'dictionary_entries_count': len(self.dictionary_entries)
        }

class DataSource(db.Model):
    """Data sources: files, databases, APIs"""
    __tablename__ = 'sources'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # file, database, api
    subtype = db.Column(db.String(50))  # csv, excel, postgres, mysql, etc.
    
    # Connection/file info
    connection_string = db.Column(Text)  # DB connection or file path
    file_size_bytes = db.Column(db.BigInteger)
    
    # Processing status
    ingest_status = db.Column(db.String(50), default='pending')  # pending, processing, completed, failed
    ingest_error = db.Column(Text)
    
    # Metadata
    metadata_json = db.Column(JSON)  # Extra metadata specific to source type
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tables = db.relationship('Table', backref='source', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'name': self.name,
            'type': self.type,
            'subtype': self.subtype,
            'file_size_bytes': self.file_size_bytes,
            'ingest_status': self.ingest_status,
            'ingest_error': self.ingest_error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'tables_count': len(self.tables)
        }

class Table(db.Model):
    """Tables within data sources"""
    __tablename__ = 'tables'
    
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('sources.id'), nullable=False)
    
    # Basic info
    name = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(255))
    description = db.Column(Text)
    
    # Schema and stats
    row_count = db.Column(db.BigInteger, default=0)
    column_count = db.Column(db.Integer, default=0)
    schema_json = db.Column(JSON)  # Full schema definition
    
    # Data quality metrics
    completeness_score = db.Column(Numeric(5, 2))  # % of non-null values
    uniqueness_score = db.Column(Numeric(5, 2))    # % of unique values
    validity_score = db.Column(Numeric(5, 2))      # % of valid values
    
    # Business metadata
    business_domain = db.Column(db.String(100))
    data_classification = db.Column(db.String(50))  # public, internal, confidential, restricted
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    columns = db.relationship('Column', backref='table', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'source_id': self.source_id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'schema_json': self.schema_json,
            'completeness_score': round(float(self.completeness_score), 2) if self.completeness_score else None,
            'uniqueness_score': round(float(self.uniqueness_score), 2) if self.uniqueness_score else None,
            'validity_score': round(float(self.validity_score), 2) if self.validity_score else None,
            'business_domain': self.business_domain,
            'data_classification': self.data_classification,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Column(db.Model):
    """Columns within tables"""
    __tablename__ = 'columns'
    
    id = db.Column(db.Integer, primary_key=True)
    table_id = db.Column(db.Integer, db.ForeignKey('tables.id'), nullable=False)
    
    # Basic info
    name = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(255))
    description = db.Column(Text)
    data_type = db.Column(db.String(100))
    
    # Schema constraints
    is_nullable = db.Column(db.Boolean, default=True)
    is_primary_key = db.Column(db.Boolean, default=False)
    is_foreign_key = db.Column(db.Boolean, default=False)
    foreign_key_ref = db.Column(db.String(255))  # table.column reference
    
    # Statistical properties
    distinct_count = db.Column(db.BigInteger)
    null_count = db.Column(db.BigInteger)
    min_value = db.Column(Numeric(precision=15, scale=3))
    max_value = db.Column(Numeric(precision=15, scale=3))
    avg_value = db.Column(Numeric(precision=15, scale=3))
    
    # Sample data (JSON array)
    sample_values = db.Column(JSON)
    
    # Data quality and governance - FIXED NAMING
    pii_flag = db.Column(db.Boolean, default=False)  # FIXED: was is_pii
    sensitivity_level = db.Column(db.String(50))  # low, medium, high, critical
    business_category = db.Column(db.String(100))  # dimension, measure, identifier, etc.
    
    # Pattern and format info
    common_patterns = db.Column(JSON)  # Array of common data patterns found
    data_format = db.Column(db.String(100))  # date format, number format, etc.
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'table_id': self.table_id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'data_type': self.data_type,
            'is_nullable': self.is_nullable,
            'is_primary_key': self.is_primary_key,
            'is_foreign_key': self.is_foreign_key,
            'foreign_key_ref': self.foreign_key_ref,
            'distinct_count': self.distinct_count,
            'null_count': self.null_count,
            'min_value': round(float(self.min_value), 3) if self.min_value else None,
            'max_value': round(float(self.max_value), 3) if self.max_value else None,
            'avg_value': round(float(self.avg_value), 3) if self.avg_value else None,
            'sample_values': self.sample_values,
            'pii_flag': self.pii_flag,
            'sensitivity_level': self.sensitivity_level,
            'business_category': self.business_category,
            'common_patterns': self.common_patterns,
            'data_format': self.data_format,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class DictionaryEntry(db.Model):
    """Business data dictionary entries"""
    __tablename__ = 'dictionary'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    # Core definition
    term = db.Column(db.String(255), nullable=False)
    definition = db.Column(Text, nullable=False)
    category = db.Column(db.String(100))  # business_term, technical_term, metric, dimension
    domain = db.Column(db.String(100))  # finance, sales, hr, operations, etc.
    
    # Alternative representations
    synonyms = db.Column(JSON)  # Array of synonyms
    abbreviations = db.Column(JSON)  # Array of abbreviations
    examples = db.Column(JSON)  # Array of example values/usage
    
    # Governance
    status = db.Column(db.String(50), default='draft')  # draft, approved, deprecated, archived
    approval_date = db.Column(db.DateTime)
    approved_by = db.Column(db.String(100))
    
    # Auto-generation metadata
    is_auto_generated = db.Column(db.Boolean, default=False)
    confidence_score = db.Column(Numeric(5, 3))  # For auto-generated terms
    source_tables = db.Column(JSON)  # Tables/columns this term was derived from
    
    # Lineage and usage
    created_by = db.Column(db.String(100), default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'term': self.term,
            'definition': self.definition,
            'category': self.category,
            'domain': self.domain,
            'synonyms': self.synonyms or [],
            'abbreviations': self.abbreviations or [],
            'examples': self.examples or [],
            'status': self.status,
            'approval_date': self.approval_date.isoformat() if self.approval_date else None,
            'approved_by': self.approved_by,
            'is_auto_generated': self.is_auto_generated,
            'confidence_score': round(float(self.confidence_score), 3) if self.confidence_score else None,
            'source_tables': self.source_tables or [],
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Embedding(db.Model):
    """Embeddings for various objects"""
    __tablename__ = 'embeddings'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    # Object being embedded
    object_type = db.Column(db.String(50), nullable=False)  # table, column, dictionary_entry
    object_id = db.Column(db.Integer, nullable=False)
    object_text = db.Column(Text)  # Text used for embedding
    
    # Embedding metadata
    model_name = db.Column(db.String(255), nullable=False)
    vector_dimension = db.Column(db.Integer)
    vector = db.Column(LargeBinary)  # Pickled numpy array
    vector_norm = db.Column(Numeric(precision=10, scale=3))
    
    # Additional metadata
    emb__metadata = db.Column(JSON)  # Fixed: using emb__metadata instead of metadata
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Index for faster lookups
    __table_args__ = (
        SQLIndex('idx_embedding_project_type', 'project_id', 'object_type'),
        SQLIndex('idx_embedding_object', 'object_type', 'object_id'),
    )
    
    def get_vector(self):
        """Get the vector as numpy array"""
        if self.vector:
            return pickle.loads(self.vector)
        return None
    
    def set_vector(self, vector_array):
        """Set the vector from numpy array"""
        self.vector = pickle.dumps(vector_array)
        if hasattr(vector_array, 'shape'):
            self.vector_dimension = vector_array.shape[0]
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'object_type': self.object_type,
            'object_id': self.object_id,
            'object_text': self.object_text[:200] + '...' if self.object_text and len(self.object_text) > 200 else self.object_text,
            'model_name': self.model_name,
            'vector_dimension': self.vector_dimension,
            'vector_norm': round(float(self.vector_norm), 3) if self.vector_norm else None,
            'metadata': self.emb__metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Index(db.Model):
    """Search indexes built from embeddings"""
    __tablename__ = 'indexes'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    # Basic info
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(Text)
    index_type = db.Column(db.String(50), nullable=False)  # faiss, tfidf, bm25, etc.
    
    # Configuration - FIXED: Using correct field names
    embedding_model = db.Column(db.String(255))  # Filter embeddings by model
    object_types = db.Column(JSON)  # Array of object types to include - FIXED: was object_scope
    config_json = db.Column(JSON)  # Index-specific configuration - FIXED: was build_params
    
    # Status and metrics
    status = db.Column(db.String(50), default='building')  # building, ready, error
    build_progress = db.Column(Numeric(5, 2), default=0.0)
    total_vectors = db.Column(db.Integer, default=0)
    
    # File paths
    index_file_path = db.Column(db.String(500))
    metadata_file_path = db.Column(db.String(500))
    
    # Performance metrics
    index_size_mb = db.Column(Numeric(10, 2))
    build_time_seconds = db.Column(Numeric(10, 2))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'name': self.name,
            'description': self.description,
            'index_type': self.index_type,
            'embedding_model': self.embedding_model,
            'object_types': self.object_types or [],
            'config_json': self.config_json,
            'status': self.status,
            'build_progress': round(float(self.build_progress), 2) if self.build_progress else 0.0,
            'total_vectors': self.total_vectors,
            'index_size_mb': round(float(self.index_size_mb), 2) if self.index_size_mb else None,
            'build_time_seconds': round(float(self.build_time_seconds), 2) if self.build_time_seconds else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class SearchLog(db.Model):
    """Search query logs for analytics"""
    __tablename__ = 'search_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    # Query details
    query = db.Column(Text, nullable=False)
    query_type = db.Column(db.String(50))  # semantic, keyword, hybrid
    
    # Results
    results_count = db.Column(db.Integer, default=0)
    search_time_ms = db.Column(Numeric(10, 2))
    
    # Context
    user_id = db.Column(db.String(100))
    session_id = db.Column(db.String(255))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'query': self.query,
            'query_type': self.query_type,
            'results_count': self.results_count,
            'search_time_ms': round(float(self.search_time_ms), 2) if self.search_time_ms else None,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class NLQFeedback(db.Model):
    """Natural Language Query feedback for improving NL to SQL"""
    __tablename__ = 'nlq_feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    # Query details
    natural_query = db.Column(Text, nullable=False)
    generated_sql = db.Column(Text)
    corrected_sql = db.Column(Text)
    
    # Feedback
    rating = db.Column(db.Integer)  # 1-5 rating
    feedback_type = db.Column(db.String(50))  # correct, incorrect, partial
    feedback_text = db.Column(Text)
    
    # Metadata
    execution_time_ms = db.Column(Numeric(10, 2))
    error_message = db.Column(Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'natural_query': self.natural_query,
            'generated_sql': self.generated_sql,
            'corrected_sql': self.corrected_sql,
            'rating': self.rating,
            'feedback_type': self.feedback_type,
            'feedback_text': self.feedback_text,
            'execution_time_ms': round(float(self.execution_time_ms), 2) if self.execution_time_ms else None,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TableRelationship(db.Model):
    """Track relationships between tables"""
    __tablename__ = 'table_relationships'
    
    id = db.Column(db.Integer, primary_key=True)
    parent_table_id = db.Column(db.Integer, db.ForeignKey('tables.id'), nullable=False)
    child_table_id = db.Column(db.Integer, db.ForeignKey('tables.id'), nullable=False)
    relationship_type = db.Column(db.String(50))  # foreign_key, logical, derived
    join_condition = db.Column(Text)
    confidence_score = db.Column(Numeric(5, 3))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EntityMapping(db.Model):
    """Store entity to table/column mappings for faster lookups"""
    __tablename__ = 'entity_mappings'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    entity_text = db.Column(db.String(255), nullable=False)
    entity_type = db.Column(db.String(50))  # table, column, term
    
    # Mapped objects
    table_id = db.Column(db.Integer, db.ForeignKey('tables.id'))
    column_id = db.Column(db.Integer, db.ForeignKey('columns.id'))
    dictionary_id = db.Column(db.Integer, db.ForeignKey('dictionary.id'))
    
    # Mapping metadata
    mapping_confidence = db.Column(Numeric(5, 3))
    mapping_method = db.Column(db.String(50))  # exact, fuzzy, semantic
    
    # Usage tracking
    usage_count = db.Column(db.Integer, default=0)
    last_used_at = db.Column(db.DateTime)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Database indexes for performance
# These will be created automatically when the database is initialized