# backend/schemas/dictionary.py
"""
Dictionary schemas for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

from backend.models import DictionaryStatus, DictionaryCategory

class DictionaryEntryBase(BaseModel):
    """Base dictionary entry schema"""
    term: str = Field(..., min_length=1, max_length=200, description="Dictionary term")
    definition: str = Field(..., min_length=1, max_length=2000, description="Term definition")
    category: DictionaryCategory = Field(default=DictionaryCategory.BUSINESS_TERM, description="Entry category")
    synonyms: Optional[List[str]] = Field(default=[], description="List of synonyms")
    abbreviations: Optional[List[str]] = Field(default=[], description="List of abbreviations")
    domain_tags: Optional[List[str]] = Field(default=[], description="Domain tags")
    context: Optional[str] = Field(None, max_length=1000, description="Context or usage notes")
    examples: Optional[List[str]] = Field(default=[], description="Usage examples")

class DictionaryEntryCreate(DictionaryEntryBase):
    """Schema for creating a dictionary entry"""
    project_id: int = Field(..., description="Project ID")

class DictionaryEntryUpdate(BaseModel):
    """Schema for updating a dictionary entry"""
    term: Optional[str] = Field(None, min_length=1, max_length=200)
    definition: Optional[str] = Field(None, min_length=1, max_length=2000)
    category: Optional[DictionaryCategory] = None
    synonyms: Optional[List[str]] = None
    abbreviations: Optional[List[str]] = None
    domain_tags: Optional[List[str]] = None
    context: Optional[str] = Field(None, max_length=1000)
    examples: Optional[List[str]] = None
    status: Optional[DictionaryStatus] = None

class DictionaryEntryResponse(DictionaryEntryBase):
    """Schema for dictionary entry response"""
    id: int
    project_id: int
    status: DictionaryStatus
    version: int = 1
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    usage_count: int = 0
    last_used_at: Optional[str] = None
    is_auto_generated: bool = False
    confidence_score: Optional[float] = None
    source_type: Optional[str] = None
    related_terms: List[int] = []
    table_mappings: List[int] = []
    column_mappings: List[int] = []
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    is_deleted: bool = False

    class Config:
        from_attributes = True

class DictionaryListResponse(BaseModel):
    """Schema for dictionary list response"""
    entries: List[DictionaryEntryResponse]
    total: int
    skip: int
    limit: int

class DictionaryBulkEntryCreate(DictionaryEntryBase):
    """Schema for bulk dictionary entry creation"""
    pass

class DictionaryBulkCreate(BaseModel):
    """Schema for bulk dictionary creation"""
    project_id: int = Field(..., description="Project ID")
    entries: List[DictionaryBulkEntryCreate] = Field(..., description="List of entries to create")
    is_auto_generated: bool = Field(False, description="Whether entries are auto-generated")

class DictionarySearchResult(DictionaryEntryResponse):
    """Schema for dictionary search result"""
    similarity_score: float = Field(..., description="Similarity score")
    match_type: str = Field(..., description="Type of match (exact, fuzzy)")

class DictionarySearchResponse(BaseModel):
    """Schema for dictionary search response"""
    results: List[Dict[str, Any]]
    total_found: int
    query: str
    exact_match: bool

class DictionaryApprovalRequest(BaseModel):
    """Schema for dictionary approval request"""
    approved: bool = Field(..., description="Whether to approve or reject")
    reason: Optional[str] = Field(None, description="Reason for approval/rejection")

class DictionaryStatsResponse(BaseModel):
    """Schema for dictionary statistics"""
    project_id: int
    total_entries: int = 0
    status_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    auto_generated_count: int = 0
    manual_count: int = 0
    top_domain_tags: List[Dict[str, Any]] = []
    usage_stats: Dict[str, Any] = {}

class DictionaryImportRequest(BaseModel):
    """Schema for dictionary import request"""
    project_id: int = Field(..., description="Project ID")
    file_format: str = Field(..., description="File format (csv, xlsx, json)")
    mapping: Dict[str, str] = Field(..., description="Column mapping")
    overwrite_existing: bool = Field(False, description="Overwrite existing terms")

class DictionaryExportRequest(BaseModel):
    """Schema for dictionary export request"""
    project_id: int = Field(..., description="Project ID")
    file_format: str = Field("csv", description="Export format (csv, xlsx, json)")
    status_filter: Optional[List[DictionaryStatus]] = Field(None, description="Filter by status")
    category_filter: Optional[List[DictionaryCategory]] = Field(None, description="Filter by category")
    include_metadata: bool = Field(True, description="Include metadata fields")