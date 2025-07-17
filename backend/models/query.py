# backend/models/query.py
"""
Query and NLQ session models
"""

from sqlalchemy import Column, String, Text, JSON, Enum, Integer, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.models.base import BaseModel, AuditModel, ProjectScoped

class SessionStatus(PyEnum):
    """NLQ session status"""
    STARTED = "started"
    ENTITY_EXTRACTION = "entity_extraction"
    ENTITY_MAPPING = "entity_mapping"
    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class EntityType(PyEnum):
    """Entity types for extraction"""
    TABLE = "table"
    COLUMN = "column"
    VALUE = "value"
    FUNCTION = "function"
    OPERATOR = "operator"
    KEYWORD = "keyword"
    BUSINESS_TERM = "business_term"

class MappingConfidence(PyEnum):
    """Mapping confidence levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

class FeedbackType(PyEnum):
    """Feedback types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    SUGGESTION = "suggestion"
    CORRECTION = "correction"

class NLQSession(ProjectScoped, BaseModel):
    """Natural Language Query session"""
    __tablename__ = "nlq_sessions"
    
    # Session details
    session_id = Column(String(50), nullable=False, unique=True)
    status = Column(Enum(SessionStatus), default=SessionStatus.STARTED)
    
    # Query information
    query_text = Column(Text, nullable=False)
    query_intent = Column(String(100), nullable=True)
    
    # Processing results
    extracted_entities = Column(JSON, nullable=True)
    entity_mappings = Column(JSON, nullable=True)
    selected_tables = Column(JSON, nullable=True)
    generated_sql = Column(Text, nullable=True)
    
    # Execution results
    sql_result = Column(JSON, nullable=True)
    result_summary = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)
    
    # User interactions
    user_confirmations = Column(JSON, nullable=True)
    user_feedback = Column(JSON, nullable=True)
    
    # Performance metrics
    total_time = Column(Float, nullable=True)
    step_times = Column(JSON, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_step = Column(String(50), nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="nlq_sessions")
    feedback_entries = relationship("NLQFeedback", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<NLQSession(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "extracted_entities": self.extracted_entities or [],
            "entity_mappings": self.entity_mappings or {},
            "selected_tables": self.selected_tables or [],
            "sql_result": self.sql_result or {},
            "user_confirmations": self.user_confirmations or {},
            "user_feedback": self.user_feedback or {},
            "step_times": self.step_times or {},
        })
        return data
    
    def set_status(self, status: SessionStatus, error_message: str = None):
        """Set session status"""
        self.status = status
        if error_message:
            self.error_message = error_message
            self.error_step = status.value
    
    def add_step_time(self, step: str, duration: float):
        """Add step timing"""
        if not self.step_times:
            self.step_times = {}
        self.step_times[step] = duration
    
    def set_entities(self, entities: List[Dict[str, Any]]):
        """Set extracted entities"""
        self.extracted_entities = entities
        self.set_status(SessionStatus.ENTITY_EXTRACTION)
    
    def set_mappings(self, mappings: Dict[str, Any]):
        """Set entity mappings"""
        self.entity_mappings = mappings
        self.set_status(SessionStatus.ENTITY_MAPPING)
    
    def set_sql(self, sql: str, tables: List[str] = None):
        """Set generated SQL"""
        self.generated_sql = sql
        if tables:
            self.selected_tables = tables
        self.set_status(SessionStatus.SQL_GENERATION)
    
    def set_result(self, result: Dict[str, Any], execution_time: float = None):
        """Set SQL execution result"""
        self.sql_result = result
        if execution_time:
            self.execution_time = execution_time
        self.set_status(SessionStatus.SQL_EXECUTION)
    
    def add_user_confirmation(self, step: str, confirmed: bool, feedback: str = None):
        """Add user confirmation"""
        if not self.user_confirmations:
            self.user_confirmations = {}
        self.user_confirmations[step] = {
            "confirmed": confirmed,
            "feedback": feedback,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def complete_session(self, summary: str = None):
        """Complete the session"""
        self.set_status(SessionStatus.COMPLETED)
        if summary:
            self.result_summary = summary
        
        # Calculate total time
        if self.step_times:
            self.total_time = sum(self.step_times.values())
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Dict[str, Any]]:
        """Get entities by type"""
        if not self.extracted_entities:
            return []
        return [e for e in self.extracted_entities if e.get("type") == entity_type.value]
    
    def get_mapped_tables(self) -> List[Dict[str, Any]]:
        """Get mapped tables from entity mappings"""
        if not self.entity_mappings:
            return []
        
        tables = []
        for mapping in self.entity_mappings.get("mappings", []):
            if mapping.get("target_type") == "table":
                tables.append(mapping)
        return tables
    
    def get_mapped_columns(self) -> List[Dict[str, Any]]:
        """Get mapped columns from entity mappings"""
        if not self.entity_mappings:
            return []
        
        columns = []
        for mapping in self.entity_mappings.get("mappings", []):
            if mapping.get("target_type") == "column":
                columns.append(mapping)
        return columns

class EntityExtraction(BaseModel):
    """Entity extraction results"""
    __tablename__ = "entity_extractions"
    
    session_id = Column(String(50), ForeignKey("nlq_sessions.session_id"), nullable=False)
    project_id = Column(Integer, nullable=False)
    
    # Entity details
    entity_text = Column(String(500), nullable=False)
    entity_type = Column(Enum(EntityType), nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Context
    context_start = Column(Integer, nullable=False)
    context_end = Column(Integer, nullable=False)
    surrounding_text = Column(Text, nullable=True)
    
    # Processing metadata
    extraction_method = Column(String(50), nullable=False)  # llm, regex, dictionary
    model_used = Column(String(100), nullable=True)
    
    def __repr__(self):
        return f"<EntityExtraction(id={self.id}, text='{self.entity_text}', type='{self.entity_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return super().to_dict()

class EntityMapping(BaseModel):
    """Entity mapping results"""
    __tablename__ = "entity_mappings"
    
    session_id = Column(String(50), ForeignKey("nlq_sessions.session_id"), nullable=False)
    project_id = Column(Integer, nullable=False)
    
    # Source entity
    entity_text = Column(String(500), nullable=False)
    entity_type = Column(Enum(EntityType), nullable=False)
    
    # Target mapping
    target_type = Column(String(50), nullable=False)  # table, column, dictionary_entry
    target_id = Column(Integer, nullable=False)
    target_name = Column(String(500), nullable=False)
    
    # Mapping quality
    similarity_score = Column(Float, nullable=False)
    confidence_level = Column(Enum(MappingConfidence), nullable=False)
    mapping_method = Column(String(50), nullable=False)  # exact, fuzzy, semantic, dictionary
    
    # User validation
    is_confirmed = Column(Boolean, nullable=True)
    user_feedback = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<EntityMapping(id={self.id}, entity='{self.entity_text}', target='{self.target_name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return super().to_dict()
    
    def confirm_mapping(self, feedback: str = None):
        """Confirm the mapping"""
        self.is_confirmed = True
        if feedback:
            self.user_feedback = feedback
    
    def reject_mapping(self, feedback: str = None):
        """Reject the mapping"""
        self.is_confirmed = False
        if feedback:
            self.user_feedback = feedback

class NLQFeedback(BaseModel):
    """User feedback on NLQ results"""
    __tablename__ = "nlq_feedback"
    
    session_id = Column(String(50), ForeignKey("nlq_sessions.session_id"), nullable=False)
    project_id = Column(Integer, nullable=False)
    
    # Feedback details
    feedback_type = Column(Enum(FeedbackType), nullable=False)
    rating = Column(Integer, nullable=True)  # 1-5 scale
    comment = Column(Text, nullable=True)
    
    # Context
    feedback_step = Column(String(50), nullable=False)  # entities, mappings, sql, results
    nlq_text = Column(Text, nullable=False)
    sql_text = Column(Text, nullable=True)
    
    # Suggestions
    suggested_entities = Column(JSON, nullable=True)
    suggested_mappings = Column(JSON, nullable=True)
    suggested_sql = Column(Text, nullable=True)
    
    # Processing flags
    is_processed = Column(Boolean, default=False)
    is_incorporated = Column(Boolean, default=False)
    
    # Relationships
    session = relationship("NLQSession", back_populates="feedback_entries")
    
    def __repr__(self):
        return f"<NLQFeedback(id={self.id}, type='{self.feedback_type}', rating={self.rating})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "suggested_entities": self.suggested_entities or [],
            "suggested_mappings": self.suggested_mappings or {},
        })
        return data
    
    def mark_processed(self):
        """Mark feedback as processed"""
        self.is_processed = True
    
    def mark_incorporated(self):
        """Mark feedback as incorporated"""
        self.is_incorporated = True
        self.is_processed = True

class QueryTemplate(ProjectScoped, BaseModel):
    """Query templates for common patterns"""
    __tablename__ = "query_templates"
    
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Template pattern
    pattern = Column(Text, nullable=False)
    sql_template = Column(Text, nullable=False)
    
    # Parameters
    parameters = Column(JSON, nullable=True)
    required_tables = Column(JSON, nullable=True)
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Categorization
    category = Column(String(100), nullable=True)
    tags = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<QueryTemplate(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "parameters": self.parameters or {},
            "required_tables": self.required_tables or [],
            "tags": self.tags or [],
        })
        return data
    
    def matches_pattern(self, query: str) -> bool:
        """Check if query matches template pattern"""
        import re
        pattern = self.pattern.replace("{entity}", r"[\w\s]+").replace("{value}", r"[\w\s\d]+")
        return bool(re.search(pattern, query, re.IGNORECASE))
    
    def increment_usage(self, success: bool = True):
        """Increment usage statistics"""
        self.usage_count += 1
        if success:
            self.success_rate = (self.success_rate * (self.usage_count - 1) + 1) / self.usage_count
        else:
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count
