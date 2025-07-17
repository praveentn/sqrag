# backend/models/base.py
"""
Base model with common fields and utilities
"""

from sqlalchemy import Column, Integer, DateTime, String, Text, Boolean, func
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
from typing import Any, Dict
import uuid

from backend.database import Base

class BaseModel(Base):
    """Base model with common fields"""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now())
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model from dictionary"""
        return cls(**{
            key: value for key, value in data.items()
            if hasattr(cls, key)
        })

class UUIDModel(BaseModel):
    """Base model with UUID primary key"""
    __abstract__ = True
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

class AuditModel(BaseModel):
    """Base model with audit fields"""
    __abstract__ = True
    
    created_by = Column(String(100), nullable=True)
    updated_by = Column(String(100), nullable=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(String(100), nullable=True)
    
    def soft_delete(self, user_id: str = None):
        """Soft delete the record"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        self.deleted_by = user_id
    
    def restore(self, user_id: str = None):
        """Restore soft deleted record"""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None
        self.updated_by = user_id

class ProjectScoped(BaseModel):
    """Base model for project-scoped entities"""
    __abstract__ = True
    
    @declared_attr
    def project_id(cls):
        return Column(Integer, nullable=False, index=True)
    
    @declared_attr
    def __table_args__(cls):
        return (
            {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'},
        )