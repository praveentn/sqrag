# backend/models/__init__.py
"""
Database models for StructuraAI
"""

from .base import BaseModel, UUIDModel, AuditModel, ProjectScoped
from .project import Project, ProjectStatus
from .source import Source, Table, Column, SourceType, SourceStatus
from .dictionary import DictionaryEntry, DictionaryStatus, DictionaryCategory
from .embedding import (
    Embedding, Index, SearchResult, 
    EmbeddingModel, ObjectType, IndexType, IndexStatus
)
from .query import (
    NLQSession, EntityExtraction, EntityMapping, NLQFeedback, QueryTemplate,
    SessionStatus, EntityType, MappingConfidence, FeedbackType
)

__all__ = [
    # Base models
    "BaseModel",
    "UUIDModel", 
    "AuditModel",
    "ProjectScoped",
    
    # Project models
    "Project",
    "ProjectStatus",
    
    # Source models
    "Source",
    "Table", 
    "Column",
    "SourceType",
    "SourceStatus",
    
    # Dictionary models
    "DictionaryEntry",
    "DictionaryStatus",
    "DictionaryCategory",
    
    # Embedding models
    "Embedding",
    "Index",
    "SearchResult",
    "EmbeddingModel",
    "ObjectType",
    "IndexType", 
    "IndexStatus",
    
    # Query models
    "NLQSession",
    "EntityExtraction",
    "EntityMapping", 
    "NLQFeedback",
    "QueryTemplate",
    "SessionStatus",
    "EntityType",
    "MappingConfidence",
    "FeedbackType",
]