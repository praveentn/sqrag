# backend/schemas/chat.py
"""
Chat and NLQ schemas for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

from backend.models import FeedbackType

class ChatRequest(BaseModel):
    """Schema for chat/NLQ request"""
    project_id: int = Field(..., description="Project ID")
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    auto_execute: bool = Field(False, description="Auto-execute generated SQL")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query content"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class EntityExtractionResponse(BaseModel):
    """Schema for entity extraction response"""
    session_id: str
    query: str
    entities: List[Dict[str, Any]] = []
    confidence: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""

class EntityMappingResponse(BaseModel):
    """Schema for entity mapping response"""
    session_id: str
    mappings: List[Dict[str, Any]] = []
    confidence_scores: Dict[str, float] = {}
    unmapped_entities: List[Dict[str, Any]] = []
    processing_time: float = 0.0

class SQLGenerationResponse(BaseModel):
    """Schema for SQL generation response"""
    session_id: str
    sql: str = ""
    rationale: str = ""
    confidence: float = 0.0
    tables_used: List[str] = []
    assumptions: List[str] = []
    is_valid: bool = False
    processing_time: float = 0.0

class ChatResponse(BaseModel):
    """Schema for complete chat response"""
    session_id: str
    query: str
    status: str  # started, entity_extraction, entity_mapping, sql_generation, sql_ready, completed, error
    entities: Optional[Dict[str, Any]] = None
    mappings: Optional[Dict[str, Any]] = None
    sql_generation: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    step_times: Optional[Dict[str, float]] = None
    user_feedback: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class FeedbackRequest(BaseModel):
    """Schema for feedback submission"""
    session_id: str = Field(..., description="Session ID")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    step: str = Field(..., description="Step the feedback is for")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5")
    comment: Optional[str] = Field(None, max_length=1000, description="Feedback comment")
    suggested_entities: Optional[List[Dict[str, Any]]] = Field(None, description="Suggested entities")
    suggested_mappings: Optional[Dict[str, Any]] = Field(None, description="Suggested mappings")
    suggested_sql: Optional[str] = Field(None, description="Suggested SQL")

class ChatSessionResponse(BaseModel):
    """Schema for chat session summary"""
    session_id: str
    query: str
    status: str
    created_at: datetime
    total_time: Optional[float] = None
    has_result: bool = False
    has_feedback: bool = False

class SessionListResponse(BaseModel):
    """Schema for session list response"""
    sessions: List[ChatSessionResponse]
    total: int
    skip: int
    limit: int

class QueryRefinementRequest(BaseModel):
    """Schema for query refinement request"""
    session_id: str = Field(..., description="Session ID")
    feedback: str = Field(..., min_length=1, max_length=500, description="User feedback")
    original_sql: Optional[str] = Field(None, description="Original SQL query")

class QueryTemplateResponse(BaseModel):
    """Schema for query template response"""
    id: int
    name: str
    description: Optional[str] = None
    pattern: str
    sql_template: str
    parameters: Dict[str, Any] = {}
    required_tables: List[str] = []
    usage_count: int = 0
    success_rate: float = 0.0
    category: Optional[str] = None
    tags: List[str] = []

class ChatStatsResponse(BaseModel):
    """Schema for chat statistics"""
    project_id: int
    total_sessions: int = 0
    completed_sessions: int = 0
    success_rate: float = 0.0
    avg_processing_time: float = 0.0
    most_common_entities: List[Dict[str, Any]] = []
    most_used_tables: List[Dict[str, Any]] = []
    feedback_summary: Dict[str, Any] = {}
    error_patterns: List[Dict[str, Any]] = []