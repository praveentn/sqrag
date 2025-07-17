# backend/schemas/embedding.py
"""
Embedding and indexing schemas for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

from backend.models import ObjectType, IndexType, IndexStatus, EmbeddingModel

class EmbeddingCreateRequest(BaseModel):
    """Schema for creating a single embedding"""
    project_id: int = Field(..., description="Project ID")
    object_type: ObjectType = Field(..., description="Type of object to embed")
    object_id: int = Field(..., description="ID of object to embed")
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )

class EmbeddingBatchRequest(BaseModel):
    """Schema for batch embedding creation"""
    project_id: int = Field(..., description="Project ID")
    object_types: List[ObjectType] = Field(..., description="Types of objects to embed")
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    overwrite_existing: bool = Field(False, description="Overwrite existing embeddings")

class EmbeddingResponse(BaseModel):
    """Schema for embedding response"""
    id: int
    project_id: int
    object_type: ObjectType
    object_id: int
    model_name: str
    model_version: Optional[str] = None
    dimensions: int
    text_content: str
    metadata: Dict[str, Any] = {}
    embedding_time: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class EmbeddingListResponse(BaseModel):
    """Schema for embedding list response"""
    embeddings: List[EmbeddingResponse]
    total: int
    skip: int
    limit: int

class IndexCreateRequest(BaseModel):
    """Schema for creating a search index"""
    project_id: int = Field(..., description="Project ID")
    name: str = Field(..., min_length=1, max_length=200, description="Index name")
    type: IndexType = Field(..., description="Index type")
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name for vector indexes"
    )
    object_types: List[ObjectType] = Field(..., description="Object types to index")
    dimensions: Optional[int] = Field(None, description="Vector dimensions")
    build_params: Dict[str, Any] = Field(default_factory=dict, description="Index build parameters")

    @validator('build_params')
    def validate_build_params(cls, v, values):
        """Validate build parameters based on index type"""
        index_type = values.get('type')
        
        if index_type == IndexType.FAISS:
            allowed_params = ['metric', 'index_type', 'nprobe']
            if 'metric' not in v:
                v['metric'] = 'cosine'
        elif index_type == IndexType.TFIDF:
            allowed_params = ['max_features', 'ngram_range', 'stop_words']
            if 'max_features' not in v:
                v['max_features'] = 10000
        elif index_type == IndexType.PGVECTOR:
            allowed_params = ['metric', 'ef_construction', 'm']
            if 'metric' not in v:
                v['metric'] = 'cosine'
        
        return v

class IndexResponse(BaseModel):
    """Schema for index response"""
    id: int
    project_id: int
    name: str
    type: IndexType
    status: IndexStatus
    model_name: str
    object_scope: Dict[str, Any] = {}
    build_params: Dict[str, Any] = {}
    total_objects: int = 0
    indexed_objects: int = 0
    dimensions: Optional[int] = None
    build_time: Optional[float] = None
    index_size: Optional[int] = None
    index_path: Optional[str] = None
    metadata_path: Optional[str] = None
    build_progress: float = 0.0
    build_log: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    is_deleted: bool = False

    class Config:
        from_attributes = True

class IndexListResponse(BaseModel):
    """Schema for index list response"""
    indexes: List[IndexResponse]
    total: int
    skip: int
    limit: int

class IndexStatsResponse(BaseModel):
    """Schema for index statistics"""
    index_id: int
    name: str
    type: IndexType
    status: IndexStatus
    total_objects: int = 0
    indexed_objects: int = 0
    dimensions: Optional[int] = None
    build_time: Optional[float] = None
    index_size: Optional[int] = None
    build_progress: float = 0.0
    created_at: datetime
    updated_at: datetime
    
    # Performance metrics
    avg_search_time: Optional[float] = None
    total_searches: int = 0
    last_search: Optional[datetime] = None

class IndexBuildResponse(BaseModel):
    """Schema for index build response"""
    message: str
    index_id: int
    status: IndexStatus
    estimated_time: Optional[float] = None

class EmbeddingStatsResponse(BaseModel):
    """Schema for embedding statistics"""
    project_id: int
    total_embeddings: int = 0
    embeddings_by_type: Dict[str, int] = {}
    embeddings_by_model: Dict[str, int] = {}
    total_dimensions: Dict[str, int] = {}
    avg_embedding_time: Optional[float] = None
    storage_size: Optional[int] = None

class SearchRequest(BaseModel):
    """Schema for search request"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    index_ids: Optional[List[int]] = Field(None, description="Specific indexes to search")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    search_type: str = Field("hybrid", description="Search type: semantic, keyword, or hybrid")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

class SearchResult(BaseModel):
    """Schema for search result"""
    object_type: ObjectType
    object_id: int
    object_name: str
    similarity_score: float
    rank: int
    matched_text: Optional[str] = None
    context_metadata: Dict[str, Any] = {}
    index_id: int
    search_method: str

class SearchResponse(BaseModel):
    """Schema for search response"""
    results: List[SearchResult]
    total_found: int
    query: str
    search_time: float
    indexes_used: List[int]
    search_type: str

class ModelListResponse(BaseModel):
    """Schema for available models response"""
    models: List[Dict[str, Any]]
    default_model: str
    supported_types: List[str]