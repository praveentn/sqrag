# backend/models/embedding.py
"""
Embedding and indexing models
"""

from sqlalchemy import Column, String, Text, JSON, Enum, Integer, Boolean, ForeignKey, Float, LargeBinary
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from typing import Dict, Any, List, Optional
import numpy as np
import json

from backend.models.base import BaseModel, AuditModel, ProjectScoped

class EmbeddingModel(PyEnum):
    """Embedding model types"""
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI_TEXT_EMBEDDING = "openai-text-embedding"
    AZURE_OPENAI_EMBEDDING = "azure-openai-embedding"
    HUGGINGFACE_BERT = "huggingface-bert"
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"

class ObjectType(PyEnum):
    """Object types for embeddings"""
    TABLE = "table"
    COLUMN = "column"
    DICTIONARY_ENTRY = "dictionary_entry"
    QUERY = "query"
    DOCUMENT = "document"
    CELL_VALUE = "cell_value"

class IndexType(PyEnum):
    """Index types"""
    FAISS = "faiss"
    PGVECTOR = "pgvector"
    TFIDF = "tfidf"
    BM25 = "bm25"
    ELASTICSEARCH = "elasticsearch"
    ANNOY = "annoy"

class IndexStatus(PyEnum):
    """Index status"""
    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"
    DISABLED = "disabled"

class Embedding(ProjectScoped, BaseModel):
    """Embedding model for vector representations"""
    __tablename__ = "embeddings"
    
    # Object reference
    object_type = Column(Enum(ObjectType), nullable=False)
    object_id = Column(Integer, nullable=False)
    
    # Embedding details
    model_name = Column(String(200), nullable=False)
    model_version = Column(String(50), nullable=True)
    
    # Vector data (stored as JSON for portability)
    vector_data = Column(JSON, nullable=False)
    dimensions = Column(Integer, nullable=False)
    
    # Metadata
    text_content = Column(Text, nullable=False)  # Original text that was embedded
    _metadata = Column(JSON, nullable=True)
    
    # Performance metrics
    embedding_time = Column(Float, nullable=True)  # Time taken to generate embedding
    
    # Relationships
    project = relationship("Project", back_populates="embeddings")
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, object_type='{self.object_type}', model='{self.model_name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "vector_data": self.vector_data,
            "metadata": self._metadata or {},
        })
        return data
    
    def get_vector(self) -> np.ndarray:
        """Get vector as numpy array"""
        if isinstance(self.vector_data, list):
            return np.array(self.vector_data)
        elif isinstance(self.vector_data, str):
            return np.array(json.loads(self.vector_data))
        return np.array(self.vector_data)
    
    def set_vector(self, vector: np.ndarray):
        """Set vector from numpy array"""
        if isinstance(vector, np.ndarray):
            self.vector_data = vector.tolist()
        else:
            self.vector_data = list(vector)
        self.dimensions = len(self.vector_data)
    
    def calculate_similarity(self, other_vector: np.ndarray, metric: str = "cosine") -> float:
        """Calculate similarity with another vector"""
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.spatial.distance import euclidean
        
        vector1 = self.get_vector().reshape(1, -1)
        vector2 = other_vector.reshape(1, -1)
        
        if metric == "cosine":
            return cosine_similarity(vector1, vector2)[0][0]
        elif metric == "euclidean":
            return 1.0 / (1.0 + euclidean(vector1[0], vector2[0]))
        elif metric == "dot":
            return np.dot(vector1[0], vector2[0])
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    @classmethod
    def create_from_text(cls, project_id: int, object_type: ObjectType, 
                        object_id: int, text_content: str, 
                        model_name: str, vector: np.ndarray, 
                        metadata: Dict[str, Any] = None) -> 'Embedding':
        """Create embedding from text content"""
        embedding = cls(
            project_id=project_id,
            object_type=object_type,
            object_id=object_id,
            model_name=model_name,
            text_content=text_content,
            metadata=metadata or {}
        )
        embedding.set_vector(vector)
        return embedding

class Index(ProjectScoped, AuditModel):
    """Index model for search indexes"""
    __tablename__ = "indexes"
    
    name = Column(String(200), nullable=False)
    type = Column(Enum(IndexType), nullable=False)
    status = Column(Enum(IndexStatus), default=IndexStatus.PENDING)
    
    # Configuration
    model_name = Column(String(200), nullable=False)
    object_scope = Column(JSON, nullable=False)  # Which objects to index
    build_params = Column(JSON, nullable=True)  # Index-specific parameters
    
    # Statistics
    total_objects = Column(Integer, default=0)
    indexed_objects = Column(Integer, default=0)
    dimensions = Column(Integer, nullable=True)
    
    # Performance
    build_time = Column(Float, nullable=True)
    index_size = Column(Integer, nullable=True)  # Size in bytes
    
    # File paths (for file-based indexes)
    index_path = Column(String(500), nullable=True)
    metadata_path = Column(String(500), nullable=True)
    
    # Progress tracking
    build_progress = Column(Float, default=0.0)
    build_log = Column(Text, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="indexes")
    
    def __repr__(self):
        return f"<Index(id={self.id}, name='{self.name}', type='{self.type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "object_scope": self.object_scope or {},
            "build_params": self.build_params or {},
        })
        return data
    
    def update_build_progress(self, progress: float, log_message: str = None):
        """Update build progress"""
        self.build_progress = min(100.0, max(0.0, progress))
        if log_message:
            current_log = self.build_log or ""
            self.build_log = f"{current_log}\n{log_message}" if current_log else log_message
    
    def set_ready(self, total_objects: int, build_time: float = None):
        """Set index to ready state"""
        self.status = IndexStatus.READY
        self.total_objects = total_objects
        self.indexed_objects = total_objects
        self.build_progress = 100.0
        if build_time:
            self.build_time = build_time
        self.update_build_progress(100.0, "Index build completed successfully")
    
    def set_error(self, error_message: str):
        """Set index to error state"""
        self.status = IndexStatus.ERROR
        self.update_build_progress(0.0, f"ERROR: {error_message}")
    
    def is_ready(self) -> bool:
        """Check if index is ready for use"""
        return self.status == IndexStatus.READY
    
    def get_scope_filter(self) -> Dict[str, Any]:
        """Get object scope filter"""
        return self.object_scope or {}
    
    def get_build_params(self) -> Dict[str, Any]:
        """Get build parameters"""
        return self.build_params or {}
    
    def supports_object_type(self, object_type: ObjectType) -> bool:
        """Check if index supports given object type"""
        scope = self.get_scope_filter()
        return object_type.value in scope.get("object_types", [])
    
    @classmethod
    def create_faiss_index(cls, project_id: int, name: str, model_name: str, 
                          object_types: List[ObjectType], dimensions: int,
                          metric: str = "cosine") -> 'Index':
        """Create FAISS index"""
        return cls(
            project_id=project_id,
            name=name,
            type=IndexType.FAISS,
            model_name=model_name,
            dimensions=dimensions,
            object_scope={
                "object_types": [obj_type.value for obj_type in object_types]
            },
            build_params={
                "metric": metric,
                "index_type": "IndexFlatIP" if metric == "cosine" else "IndexFlatL2",
                "nprobe": 10
            }
        )
    
    @classmethod
    def create_tfidf_index(cls, project_id: int, name: str, 
                          object_types: List[ObjectType],
                          max_features: int = 10000) -> 'Index':
        """Create TF-IDF index"""
        return cls(
            project_id=project_id,
            name=name,
            type=IndexType.TFIDF,
            model_name="tfidf",
            object_scope={
                "object_types": [obj_type.value for obj_type in object_types]
            },
            build_params={
                "max_features": max_features,
                "ngram_range": [1, 2],
                "stop_words": "english"
            }
        )
    
    @classmethod
    def create_pgvector_index(cls, project_id: int, name: str, model_name: str,
                             object_types: List[ObjectType], dimensions: int) -> 'Index':
        """Create pgvector index"""
        return cls(
            project_id=project_id,
            name=name,
            type=IndexType.PGVECTOR,
            model_name=model_name,
            dimensions=dimensions,
            object_scope={
                "object_types": [obj_type.value for obj_type in object_types]
            },
            build_params={
                "metric": "cosine",
                "ef_construction": 200,
                "m": 16
            }
        )

class SearchResult(BaseModel):
    """Search result model"""
    __tablename__ = "search_results"
    
    # Query information
    query_text = Column(Text, nullable=False)
    query_vector = Column(JSON, nullable=True)
    
    # Result information
    object_type = Column(Enum(ObjectType), nullable=False)
    object_id = Column(Integer, nullable=False)
    
    # Scoring
    similarity_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    
    # Context
    matched_text = Column(Text, nullable=True)
    context_metadata = Column(JSON, nullable=True)
    
    # Index information
    index_id = Column(Integer, ForeignKey("indexes.id"), nullable=False)
    search_method = Column(String(50), nullable=False)  # faiss, tfidf, bm25, etc.
    
    # Performance
    search_time = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<SearchResult(id={self.id}, object_type='{self.object_type}', score={self.similarity_score})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "query_vector": self.query_vector,
            "context_metadata": self.context_metadata or {},
        })
        return data
    