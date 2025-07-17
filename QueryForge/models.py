# models.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON, ARRAY
from sqlalchemy import Text, LargeBinary
import json

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
    
    # Connection details (stored as JSON for flexibility)
    connection_config = db.Column(JSON)
    
    # File specific
    file_path = db.Column(db.String(500))
    file_size = db.Column(db.BigInteger)
    
    # Status tracking - Keep original column names to match existing DB
    ingest_status = db.Column(db.String(50), default='pending')  # pending, processing, completed, failed
    ingest_progress = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    error_message = db.Column(Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tables = db.relationship('Table', backref='source', lazy=True, cascade='all, delete-orphan')
    
    # Property to provide consistent interface
    @property
    def status(self):
        return self.ingest_status
    
    @status.setter
    def status(self, value):
        self.ingest_status = value
    
    @property
    def progress(self):
        return self.ingest_progress
    
    @progress.setter
    def progress(self, value):
        self.ingest_progress = value
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'name': self.name,
            'type': self.type,
            'subtype': self.subtype,
            'connection_config': self.connection_config,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'status': self.ingest_status,  # Use the actual column name
            'progress': round(self.ingest_progress, 3) if self.ingest_progress else 0.0,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'tables_count': len(self.tables)
        }

class Table(db.Model):
    """Tables from data sources"""
    __tablename__ = 'tables'
    
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('sources.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(255))
    description = db.Column(Text)
    
    # Schema and metadata
    schema_json = db.Column(JSON)
    row_count = db.Column(db.BigInteger, default=0)
    column_count = db.Column(db.Integer, default=0)
    
    # Data quality metrics
    completeness_score = db.Column(db.Float)  # 0.0 to 1.0
    quality_score = db.Column(db.Float)  # 0.0 to 1.0
    
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
            'schema_json': self.schema_json,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'completeness_score': round(self.completeness_score, 3) if self.completeness_score else None,
            'quality_score': round(self.quality_score, 3) if self.quality_score else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'columns_count': len(self.columns)
        }

class Column(db.Model):
    """Columns from tables"""
    __tablename__ = 'columns'
    
    id = db.Column(db.Integer, primary_key=True)
    table_id = db.Column(db.Integer, db.ForeignKey('tables.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(255))
    description = db.Column(Text)
    
    # Data type and constraints
    data_type = db.Column(db.String(100))
    is_nullable = db.Column(db.Boolean, default=True)
    is_primary_key = db.Column(db.Boolean, default=False)
    is_foreign_key = db.Column(db.Boolean, default=False)
    foreign_key_ref = db.Column(db.String(255))  # table.column reference
    
    # Data statistics
    distinct_count = db.Column(db.BigInteger)
    null_count = db.Column(db.BigInteger)
    min_value = db.Column(db.String(255))
    max_value = db.Column(db.String(255))
    avg_value = db.Column(db.Float)
    sample_values = db.Column(JSON)  # Array of sample values
    
    # Data classification
    pii_flag = db.Column(db.Boolean, default=False)
    sensitivity_level = db.Column(db.String(50))  # public, internal, confidential, restricted
    business_category = db.Column(db.String(100))  # finance, hr, sales, etc.
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
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
            'min_value': self.min_value,
            'max_value': self.max_value,
            'avg_value': round(self.avg_value, 3) if self.avg_value else None,
            'sample_values': self.sample_values,
            'pii_flag': self.pii_flag,
            'sensitivity_level': self.sensitivity_level,
            'business_category': self.business_category
        }

class DictionaryEntry(db.Model):
    """Data dictionary entries: terms, definitions, synonyms"""
    __tablename__ = 'dictionary'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    # Core content
    term = db.Column(db.String(255), nullable=False)
    definition = db.Column(Text, nullable=False)
    category = db.Column(db.String(100))  # business_term, technical_term, abbreviation, etc.
    
    # Extended metadata
    synonyms = db.Column(JSON)  # Array of synonyms
    abbreviations = db.Column(JSON)  # Array of abbreviations
    related_terms = db.Column(JSON)  # Array of related term IDs
    domain = db.Column(db.String(100))  # finance, hr, sales, etc.
    
    # Versioning and approval
    version = db.Column(db.Integer, default=1)
    status = db.Column(db.String(50), default='draft')  # draft, approved, archived
    approved_by = db.Column(db.String(100))
    approved_at = db.Column(db.DateTime)
    
    # Auto-generation metadata
    is_auto_generated = db.Column(db.Boolean, default=False)
    confidence_score = db.Column(db.Float)  # For auto-generated terms
    source_tables = db.Column(JSON)  # Tables this term was derived from
    
    created_by = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'term': self.term,
            'definition': self.definition,
            'category': self.category,
            'synonyms': self.synonyms or [],
            'abbreviations': self.abbreviations or [],
            'related_terms': self.related_terms or [],
            'domain': self.domain,
            'version': self.version,
            'status': self.status,
            'approved_by': self.approved_by,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'is_auto_generated': self.is_auto_generated,
            'confidence_score': round(self.confidence_score, 3) if self.confidence_score else None,
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
    object_text = db.Column(Text)  # The text that was embedded
    
    # Embedding metadata
    model_name = db.Column(db.String(255), nullable=False)
    model_version = db.Column(db.String(100))
    vector_dimension = db.Column(db.Integer)
    
    # The actual embedding vector
    vector = db.Column(LargeBinary)  # Stored as binary for efficiency
    vector_norm = db.Column(db.Float)  # L2 norm for optimization
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'object_type': self.object_type,
            'object_id': self.object_id,
            'object_text': self.object_text,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'vector_dimension': self.vector_dimension,
            'vector_norm': round(self.vector_norm, 3) if self.vector_norm else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Index(db.Model):
    """Search indexes built on embeddings"""
    __tablename__ = 'indexes'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(Text)
    
    # Index configuration
    index_type = db.Column(db.String(50), nullable=False)  # faiss, pgvector, tfidf, bm25
    metric = db.Column(db.String(50))  # cosine, euclidean, dot_product
    
    # Object scope
    object_scope = db.Column(JSON)  # Which objects are included
    embedding_model = db.Column(db.String(255))
    
    # Build parameters
    build_params = db.Column(JSON)
    
    # Status and metrics
    status = db.Column(db.String(50), default='building')  # building, ready, error
    build_progress = db.Column(db.Float, default=0.0)
    total_vectors = db.Column(db.BigInteger, default=0)
    index_size_mb = db.Column(db.Float)
    
    # Performance metrics
    avg_query_time_ms = db.Column(db.Float)
    last_used_at = db.Column(db.DateTime)
    
    # File paths for stored indexes
    index_file_path = db.Column(db.String(500))
    metadata_file_path = db.Column(db.String(500))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'name': self.name,
            'description': self.description,
            'index_type': self.index_type,
            'metric': self.metric,
            'object_scope': self.object_scope,
            'embedding_model': self.embedding_model,
            'build_params': self.build_params,
            'status': self.status,
            'build_progress': round(self.build_progress, 3) if self.build_progress else 0.0,
            'total_vectors': self.total_vectors,
            'index_size_mb': round(self.index_size_mb, 2) if self.index_size_mb else None,
            'avg_query_time_ms': round(self.avg_query_time_ms, 2) if self.avg_query_time_ms else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class SearchLog(db.Model):
    """Search activity logs"""
    __tablename__ = 'search_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    query = db.Column(Text, nullable=False)
    search_type = db.Column(db.String(50))  # semantic, keyword, hybrid
    result_count = db.Column(db.Integer)
    
    # Performance metrics
    search_time_seconds = db.Column(db.Float)
    
    # Context
    user_id = db.Column(db.String(100))
    session_id = db.Column(db.String(100))
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'query': self.query,
            'search_type': self.search_type,
            'result_count': self.result_count,
            'search_time_seconds': round(self.search_time_seconds, 3) if self.search_time_seconds else None,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class NLQFeedback(db.Model):
    """Natural Language Query feedback"""
    __tablename__ = 'nlq_feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    
    # Original query and context
    user_query = db.Column(Text, nullable=False)
    extracted_entities = db.Column(JSON)
    generated_sql = db.Column(Text)
    
    # User feedback
    feedback_type = db.Column(db.String(50))  # thumbs_up, thumbs_down, correction
    feedback_text = db.Column(Text)
    correct_sql = db.Column(Text)  # If user provides correction
    
    # Performance metrics
    entity_extraction_time_ms = db.Column(db.Float)
    sql_generation_time_ms = db.Column(db.Float)
    sql_execution_time_ms = db.Column(db.Float)
    total_time_ms = db.Column(db.Float)
    
    # Context
    user_id = db.Column(db.String(100))
    session_id = db.Column(db.String(100))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'user_query': self.user_query,
            'extracted_entities': self.extracted_entities,
            'generated_sql': self.generated_sql,
            'feedback_type': self.feedback_type,
            'feedback_text': self.feedback_text,
            'correct_sql': self.correct_sql,
            'entity_extraction_time_ms': round(self.entity_extraction_time_ms, 2) if self.entity_extraction_time_ms else None,
            'sql_generation_time_ms': round(self.sql_generation_time_ms, 2) if self.sql_generation_time_ms else None,
            'sql_execution_time_ms': round(self.sql_execution_time_ms, 2) if self.sql_execution_time_ms else None,
            'total_time_ms': round(self.total_time_ms, 2) if self.total_time_ms else None,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Additional helper tables for many-to-many relationships
class TableRelationship(db.Model):
    """Track relationships between tables"""
    __tablename__ = 'table_relationships'
    
    id = db.Column(db.Integer, primary_key=True)
    parent_table_id = db.Column(db.Integer, db.ForeignKey('tables.id'), nullable=False)
    child_table_id = db.Column(db.Integer, db.ForeignKey('tables.id'), nullable=False)
    relationship_type = db.Column(db.String(50))  # foreign_key, logical, derived
    join_condition = db.Column(Text)
    confidence_score = db.Column(db.Float)
    
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
    mapping_confidence = db.Column(db.Float)
    mapping_method = db.Column(db.String(50))  # exact, fuzzy, semantic
    
    # Usage tracking
    usage_count = db.Column(db.Integer, default=0)
    last_used_at = db.Column(db.DateTime)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)