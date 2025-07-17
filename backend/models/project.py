# backend/models/project.py
"""
Project model for organizing all resources
"""

from sqlalchemy import Column, String, Text, JSON, Enum, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from typing import Dict, Any, Optional

from backend.models.base import BaseModel, AuditModel

class ProjectStatus(PyEnum):
    """Project status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"

class Project(AuditModel):
    """Project model for organizing all resources"""
    __tablename__ = "projects"
    
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    owner = Column(String(100), nullable=False)
    status = Column(Enum(ProjectStatus), default=ProjectStatus.ACTIVE)
    
    # Project settings (JSON configuration)
    settings = Column(JSON, default=lambda: {
        "llm_provider": "azure_openai",
        "default_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_query_length": 1000,
        "similarity_threshold": 0.7,
        "max_results": 50,
        "enable_pii_detection": True,
        "auto_generate_dictionary": True,
        "feedback_collection": True
    })
    
    # Relationships
    sources = relationship("Source", back_populates="project", cascade="all, delete-orphan")
    dictionary_entries = relationship("DictionaryEntry", back_populates="project", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="project", cascade="all, delete-orphan")
    indexes = relationship("Index", back_populates="project", cascade="all, delete-orphan")
    nlq_sessions = relationship("NLQSession", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', owner='{self.owner}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with additional fields"""
        data = super().to_dict()
        data.update({
            "sources_count": len(self.sources) if self.sources else 0,
            "dictionary_entries_count": len(self.dictionary_entries) if self.dictionary_entries else 0,
            "embeddings_count": len(self.embeddings) if self.embeddings else 0,
            "indexes_count": len(self.indexes) if self.indexes else 0,
        })
        return data
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get project setting"""
        return self.settings.get(key, default) if self.settings else default
    
    def update_setting(self, key: str, value: Any):
        """Update project setting"""
        if not self.settings:
            self.settings = {}
        self.settings[key] = value
    
    def is_active(self) -> bool:
        """Check if project is active"""
        return self.status == ProjectStatus.ACTIVE and not self.is_deleted
    
    def archive(self, user_id: str = None):
        """Archive the project"""
        self.status = ProjectStatus.ARCHIVED
        self.updated_by = user_id
    
    def activate(self, user_id: str = None):
        """Activate the project"""
        self.status = ProjectStatus.ACTIVE
        self.updated_by = user_id
    
    def clone(self, new_name: str, new_owner: str, user_id: str = None) -> 'Project':
        """Clone project with new name and owner"""
        clone_settings = self.settings.copy() if self.settings else {}
        
        cloned_project = Project(
            name=new_name,
            description=f"Cloned from: {self.name}",
            owner=new_owner,
            settings=clone_settings,
            created_by=user_id
        )
        
        return cloned_project
    