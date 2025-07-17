# backend/schemas/project.py
"""
Project schemas for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from backend.models import ProjectStatus

class ProjectBase(BaseModel):
    """Base project schema"""
    name: str = Field(..., min_length=1, max_length=200, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Project settings")

class ProjectCreate(ProjectBase):
    """Schema for creating a project"""
    pass

class ProjectUpdate(BaseModel):
    """Schema for updating a project"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[ProjectStatus] = None
    settings: Optional[Dict[str, Any]] = None

class ProjectCloneRequest(BaseModel):
    """Schema for cloning a project"""
    name: str = Field(..., min_length=1, max_length=200, description="New project name")
    description: Optional[str] = Field(None, max_length=1000, description="New project description")
    clone_data: bool = Field(True, description="Whether to clone data sources")
    clone_embeddings: bool = Field(False, description="Whether to clone embeddings")
    clone_indexes: bool = Field(False, description="Whether to clone indexes")

class ProjectResponse(ProjectBase):
    """Schema for project response"""
    id: int
    owner: str
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    is_deleted: bool = False
    
    # Computed fields
    sources_count: int = 0
    dictionary_entries_count: int = 0
    embeddings_count: int = 0
    indexes_count: int = 0

    class Config:
        from_attributes = True

class ProjectListResponse(BaseModel):
    """Schema for project list response"""
    projects: List[ProjectResponse]
    total: int
    skip: int
    limit: int

class ProjectStatsResponse(BaseModel):
    """Schema for project statistics"""
    project_id: int
    sources_count: int = 0
    tables_count: int = 0
    columns_count: int = 0
    dictionary_entries_count: int = 0
    indexes_count: int = 0
    embeddings_count: int = 0
    nlq_sessions_count: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    avg_query_time: float = 0.0
    last_activity: Optional[datetime] = None