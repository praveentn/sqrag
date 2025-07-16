# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class DataSource(db.Model):
    """Data source _metadata table"""
    __tablename__ = 'data_sources'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True)
    type = db.Column(db.String(50), nullable=False)  # csv, excel, database, view
    connection_string = db.Column(db.Text)
    file_path = db.Column(db.String(500))
    _metadata = db.Column(db.JSON)
    row_count = db.Column(db.Integer, default=0)
    last_refresh = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tables = db.relationship('Table', backref='data_source', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<DataSource {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'connection_string': self.connection_string,
            'file_path': self.file_path,
            '_metadata': self._metadata,
            'row_count': self.row_count,
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Table(db.Model):
    """Table _metadata"""
    __tablename__ = 'tables'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(255))
    description = db.Column(db.Text)
    schema_name = db.Column(db.String(255))
    source_id = db.Column(db.Integer, db.ForeignKey('data_sources.id'), nullable=False)
    row_count = db.Column(db.Integer, default=0)
    sampling_stats = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    columns = db.relationship('Column', backref='table', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Table {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'schema_name': self.schema_name,
            'source_id': self.source_id,
            'row_count': self.row_count,
            'sampling_stats': self.sampling_stats,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'columns': [col.to_dict() for col in self.columns]
        }

class Column(db.Model):
    """Column _metadata"""
    __tablename__ = 'columns'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(255))
    description = db.Column(db.Text)
    data_type = db.Column(db.String(50))
    is_nullable = db.Column(db.Boolean, default=True)
    is_primary_key = db.Column(db.Boolean, default=False)
    is_foreign_key = db.Column(db.Boolean, default=False)
    table_id = db.Column(db.Integer, db.ForeignKey('tables.id'), nullable=False)
    sample_values = db.Column(db.JSON)  # Array of sample values
    unique_count = db.Column(db.Integer)
    null_count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Column {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'data_type': self.data_type,
            'is_nullable': self.is_nullable,
            'is_primary_key': self.is_primary_key,
            'is_foreign_key': self.is_foreign_key,
            'table_id': self.table_id,
            'sample_values': self.sample_values,
            'unique_count': self.unique_count,
            'null_count': self.null_count,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class DictionaryEntry(db.Model):
    """Data dictionary entries"""
    __tablename__ = 'dictionary_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    term = db.Column(db.String(255), nullable=False, unique=True)
    definition = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), default='general')
    synonyms = db.Column(db.JSON)  # Array of synonyms
    abbreviations = db.Column(db.JSON)  # Array of abbreviations
    source_table = db.Column(db.String(255))  # Source table if auto-generated
    source_column = db.Column(db.String(255))  # Source column if auto-generated
    approved = db.Column(db.Boolean, default=False)
    version = db.Column(db.Integer, default=1)
    created_by = db.Column(db.String(255))
    approved_by = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<DictionaryEntry {self.term}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'term': self.term,
            'definition': self.definition,
            'category': self.category,
            'synonyms': self.synonyms or [],
            'abbreviations': self.abbreviations or [],
            'source_table': self.source_table,
            'source_column': self.source_column,
            'approved': self.approved,
            'version': self.version,
            'created_by': self.created_by,
            'approved_by': self.approved_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class EmbeddingIndex(db.Model):
    """Embedding indexes"""
    __tablename__ = 'embedding_indexes'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True)
    scope = db.Column(db.String(50), nullable=False)  # table, column, dictionary
    backend = db.Column(db.String(50), nullable=False)  # faiss, pgvector, tfidf
    model_name = db.Column(db.String(255))
    dimensions = db.Column(db.Integer)
    index_path = db.Column(db.String(500))
    status = db.Column(db.String(50), default='pending')  # pending, building, ready, error
    progress = db.Column(db.Float, default=0.0)
    error_message = db.Column(db.Text)
    item_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<EmbeddingIndex {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'scope': self.scope,
            'backend': self.backend,
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'index_path': self.index_path,
            'status': self.status,
            'progress': round(self.progress, 2),
            'error_message': self.error_message,
            'item_count': self.item_count,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ChatSession(db.Model):
    """Chat sessions"""
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.String(255))  # For future auth implementation
    _metadata = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<ChatSession {self.title}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'user_id': self.user_id,
            '_metadata': self._metadata,
            'message_count': len(self.messages),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ChatMessage(db.Model):
    """Chat messages"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # user, assistant, system
    content = db.Column(db.Text, nullable=False)
    _metadata = db.Column(db.JSON)  # Store SQL, entities, confidence, etc.
    parent_message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ChatMessage {self.role}: {self.content[:50]}...>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'role': self.role,
            'content': self.content,
            '_metadata': self._metadata,
            'parent_message_id': self.parent_message_id,
            'created_at': self.created_at.isoformat()
        }

class QueryExecution(db.Model):
    """Query execution history"""
    __tablename__ = 'query_executions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'))
    message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'))
    sql_query = db.Column(db.Text, nullable=False)
    source_id = db.Column(db.Integer, db.ForeignKey('data_sources.id'))
    status = db.Column(db.String(20))  # success, error, timeout
    execution_time = db.Column(db.Float)  # in seconds
    row_count = db.Column(db.Integer)
    error_message = db.Column(db.Text)
    result_preview = db.Column(db.JSON)  # First few rows for preview
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<QueryExecution {self.status}: {self.sql_query[:50]}...>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'message_id': self.message_id,
            'sql_query': self.sql_query,
            'source_id': self.source_id,
            'status': self.status,
            'execution_time': round(self.execution_time, 3) if self.execution_time else None,
            'row_count': self.row_count,
            'error_message': self.error_message,
            'result_preview': self.result_preview,
            'created_at': self.created_at.isoformat()
        }

class SystemLog(db.Model):
    """System logs for audit trail"""
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20), nullable=False)  # info, warning, error
    category = db.Column(db.String(50), nullable=False)  # auth, query, admin, etc.
    message = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.String(255))
    session_id = db.Column(db.String(255))
    correlation_id = db.Column(db.String(255))
    _metadata = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SystemLog {self.level}: {self.message[:50]}...>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'level': self.level,
            'category': self.category,
            'message': self.message,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            '_metadata': self._metadata,
            'created_at': self.created_at.isoformat()
        }
    