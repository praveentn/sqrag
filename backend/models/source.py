# backend/models/source.py
"""
Source model for data sources and tables
"""

from sqlalchemy import Column, String, Text, JSON, Enum, Integer, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from typing import Dict, Any, List, Optional

from backend.models.base import BaseModel, AuditModel, ProjectScoped

class SourceType(PyEnum):
    """Source type enumeration"""
    FILE_UPLOAD = "file_upload"
    DATABASE = "database"
    API = "api"
    CLOUD_STORAGE = "cloud_storage"

class SourceStatus(PyEnum):
    """Source status enumeration"""
    PENDING = "pending"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    INGESTING = "ingesting"
    READY = "ready"
    ERROR = "error"
    DISABLED = "disabled"

class Source(ProjectScoped, AuditModel):
    """Data source model"""
    __tablename__ = "sources"
    
    name = Column(String(200), nullable=False)
    type = Column(Enum(SourceType), nullable=False)
    status = Column(Enum(SourceStatus), default=SourceStatus.PENDING)
    
    # Connection details (encrypted in production)
    connection_uri = Column(Text, nullable=True)
    connection_params = Column(JSON, nullable=True)
    
    # File details (for file uploads)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    file_type = Column(String(50), nullable=True)
    
    # Schema metadata
    schema_metadata = Column(JSON, nullable=True)
    
    # Ingestion details
    ingest_status = Column(String(50), default="not_started")
    ingest_progress = Column(Float, default=0.0)
    ingest_log = Column(Text, nullable=True)
    
    # Statistics
    total_tables = Column(Integer, default=0)
    total_columns = Column(Integer, default=0)
    total_rows = Column(Integer, default=0)
    
    # Relationships
    project = relationship("Project", back_populates="sources")
    tables = relationship("Table", back_populates="source", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Source(id={self.id}, name='{self.name}', type='{self.type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "tables_count": len(self.tables) if self.tables else 0,
            "connection_params": self.connection_params or {},
            "schema_metadata": self.schema_metadata or {},
        })
        return data
    
    def update_ingest_progress(self, progress: float, log_message: str = None):
        """Update ingestion progress"""
        self.ingest_progress = min(100.0, max(0.0, progress))
        if log_message:
            current_log = self.ingest_log or ""
            self.ingest_log = f"{current_log}\n{log_message}" if current_log else log_message
    
    def set_error(self, error_message: str):
        """Set source to error state"""
        self.status = SourceStatus.ERROR
        self.ingest_status = "error"
        self.update_ingest_progress(0.0, f"ERROR: {error_message}")
    
    def set_ready(self):
        """Set source to ready state"""
        self.status = SourceStatus.READY
        self.ingest_status = "completed"
        self.update_ingest_progress(100.0, "Ingestion completed successfully")
    
    def get_connection_string(self) -> str:
        """Get formatted connection string"""
        if self.type == SourceType.FILE_UPLOAD:
            return f"file://{self.file_path}"
        elif self.type == SourceType.DATABASE:
            return self.connection_uri or ""
        return ""

class Table(BaseModel):
    """Table model for database tables"""
    __tablename__ = "tables"
    
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    name = Column(String(200), nullable=False)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    
    # Table metadata
    schema_name = Column(String(100), nullable=True)
    table_type = Column(String(50), default="table")  # table, view, materialized_view
    
    # Statistics
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    data_size = Column(Integer, default=0)  # in bytes
    
    # Schema as JSON
    schema_json = Column(JSON, nullable=True)
    
    # Indexing flags
    is_indexed = Column(Boolean, default=False)
    last_indexed_at = Column(String, nullable=True)
    
    # Relationships
    source = relationship("Source", back_populates="tables")
    columns = relationship("Column", back_populates="table", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Table(id={self.id}, name='{self.name}', source_id={self.source_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "columns_count": len(self.columns) if self.columns else 0,
            "full_name": f"{self.schema_name}.{self.name}" if self.schema_name else self.name,
            "schema_json": self.schema_json or {},
        })
        return data
    
    def get_qualified_name(self) -> str:
        """Get fully qualified table name"""
        if self.schema_name:
            return f"{self.schema_name}.{self.name}"
        return self.name
    
    def update_statistics(self, row_count: int = None, data_size: int = None):
        """Update table statistics"""
        if row_count is not None:
            self.row_count = row_count
        if data_size is not None:
            self.data_size = data_size
        self.column_count = len(self.columns) if self.columns else 0

class Column(BaseModel):
    """Column model for table columns"""
    __tablename__ = "columns"
    
    table_id = Column(Integer, ForeignKey("tables.id"), nullable=False)
    name = Column(String(200), nullable=False)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    
    # Column metadata
    data_type = Column(String(100), nullable=False)
    max_length = Column(Integer, nullable=True)
    precision = Column(Integer, nullable=True)
    scale = Column(Integer, nullable=True)
    
    # Constraints
    is_nullable = Column(Boolean, default=True)
    is_primary_key = Column(Boolean, default=False)
    is_foreign_key = Column(Boolean, default=False)
    is_unique = Column(Boolean, default=False)
    
    # Data profiling
    unique_count = Column(Integer, nullable=True)
    null_count = Column(Integer, nullable=True)
    min_value = Column(String(500), nullable=True)
    max_value = Column(String(500), nullable=True)
    sample_values = Column(JSON, nullable=True)
    
    # Privacy & compliance
    pii_flag = Column(Boolean, default=False)
    pii_type = Column(String(50), nullable=True)  # email, phone, ssn, etc.
    
    # Relationships
    table = relationship("Table", back_populates="columns")
    
    def __repr__(self):
        return f"<Column(id={self.id}, name='{self.name}', table_id={self.table_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "full_name": f"{self.table.get_qualified_name()}.{self.name}" if self.table else self.name,
            "sample_values": self.sample_values or [],
        })
        return data
    
    def get_qualified_name(self) -> str:
        """Get fully qualified column name"""
        if self.table:
            return f"{self.table.get_qualified_name()}.{self.name}"
        return self.name
    
    def update_profile(self, unique_count: int = None, null_count: int = None, 
                      min_value: str = None, max_value: str = None, 
                      sample_values: List[str] = None):
        """Update column data profiling"""
        if unique_count is not None:
            self.unique_count = unique_count
        if null_count is not None:
            self.null_count = null_count
        if min_value is not None:
            self.min_value = str(min_value)
        if max_value is not None:
            self.max_value = str(max_value)
        if sample_values is not None:
            self.sample_values = sample_values[:10]  # Keep only first 10 samples
    
    def detect_pii(self, sample_data: List[str] = None) -> bool:
        """Detect PII in column data"""
        # Simple PII detection patterns
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
        
        # Check column name for PII indicators
        name_lower = self.name.lower()
        pii_keywords = ['email', 'phone', 'ssn', 'social', 'credit', 'card', 'password', 'secret']
        
        for keyword in pii_keywords:
            if keyword in name_lower:
                self.pii_flag = True
                self.pii_type = keyword
                return True
        
        # Check sample data if provided
        if sample_data:
            import re
            for pii_type, pattern in pii_patterns.items():
                for sample in sample_data:
                    if sample and re.search(pattern, str(sample)):
                        self.pii_flag = True
                        self.pii_type = pii_type
                        return True
        
        return False
