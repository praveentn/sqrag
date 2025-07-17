# backend/schemas/source.py
"""
Source schemas for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

from backend.models import SourceType, SourceStatus

class SourceBase(BaseModel):
    """Base source schema"""
    name: str = Field(..., min_length=1, max_length=200, description="Source name")
    type: SourceType = Field(..., description="Source type")
    connection_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Connection parameters")

class SourceCreate(SourceBase):
    """Schema for creating a source"""
    project_id: int = Field(..., description="Project ID")

class DatabaseSourceCreate(BaseModel):
    """Schema for creating a database source"""
    project_id: int = Field(..., description="Project ID")
    name: str = Field(..., min_length=1, max_length=200, description="Source name")
    db_type: str = Field(..., description="Database type (postgresql, mysql, etc.)")
    host: str = Field(..., description="Database host")
    port: int = Field(..., ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    schema_name: Optional[str] = Field(None, description="Schema name")
    ssl_mode: Optional[str] = Field("prefer", description="SSL mode")

class FileUploadResponse(BaseModel):
    """Schema for file upload response"""
    filename: str
    file_size: int
    file_type: str
    upload_path: str
    tables_detected: List[str] = []

class SourceUpdate(BaseModel):
    """Schema for updating a source"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[SourceStatus] = None
    connection_params: Optional[Dict[str, Any]] = None

class TableResponse(BaseModel):
    """Schema for table response"""
    id: int
    source_id: int
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    schema_name: Optional[str] = None
    table_type: str = "table"
    row_count: int = 0
    column_count: int = 0
    data_size: int = 0
    is_indexed: bool = False
    last_indexed_at: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ColumnResponse(BaseModel):
    """Schema for column response"""
    id: int
    table_id: int
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    data_type: str
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False
    unique_count: Optional[int] = None
    null_count: Optional[int] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    sample_values: List[str] = []
    pii_flag: bool = False
    pii_type: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ColumnUpdate(BaseModel):
    """Schema for updating column metadata"""
    display_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    pii_flag: Optional[bool] = None
    pii_type: Optional[str] = None

class TableUpdate(BaseModel):
    """Schema for updating table metadata"""
    display_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)

class SourceResponse(SourceBase):
    """Schema for source response"""
    id: int
    project_id: int
    status: SourceStatus
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    schema_metadata: Dict[str, Any] = {}
    ingest_status: str = "not_started"
    ingest_progress: float = 0.0
    ingest_log: Optional[str] = None
    total_tables: int = 0
    total_columns: int = 0
    total_rows: int = 0
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    
    # Related data
    tables_count: int = 0
    
    class Config:
        from_attributes = True

class SourceListResponse(BaseModel):
    """Schema for source list response"""
    sources: List[SourceResponse]
    total: int
    skip: int
    limit: int

class TableListResponse(BaseModel):
    """Schema for table list response"""
    tables: List[TableResponse]
    total: int
    skip: int
    limit: int

class ColumnListResponse(BaseModel):
    """Schema for column list response"""
    columns: List[ColumnResponse]
    total: int
    skip: int
    limit: int

class SchemaIntrospectionResponse(BaseModel):
    """Schema for database schema introspection"""
    tables: List[Dict[str, Any]]
    views: List[Dict[str, Any]]
    total_tables: int
    total_columns: int
    schema_metadata: Dict[str, Any]

class ConnectionTestResponse(BaseModel):
    """Schema for database connection test"""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    database_info: Optional[Dict[str, Any]] = None

class DataPreviewResponse(BaseModel):
    """Schema for data preview"""
    table_name: str
    columns: List[str]
    data: List[Dict[str, Any]]
    total_rows: int
    sample_size: int

class IngestStatusResponse(BaseModel):
    """Schema for ingestion status"""
    source_id: int
    status: str
    progress: float
    log_messages: List[str]
    current_step: Optional[str] = None
    estimated_time_remaining: Optional[float] = None
    tables_processed: int = 0
    tables_total: int = 0