# backend/models/dictionary.py
"""
Dictionary model for managing business terms and definitions
"""

from sqlalchemy import Column, String, Text, JSON, Enum, Integer, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.models.base import BaseModel, AuditModel, ProjectScoped

class DictionaryStatus(PyEnum):
    """Dictionary entry status"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"

class DictionaryCategory(PyEnum):
    """Dictionary entry category"""
    BUSINESS_TERM = "business_term"
    TECHNICAL_TERM = "technical_term"
    ABBREVIATION = "abbreviation"
    ACRONYM = "acronym"
    METRIC = "metric"
    DIMENSION = "dimension"
    ENTITY = "entity"
    PROCESS = "process"
    SYSTEM = "system"
    DOMAIN = "domain"

class DictionaryEntry(ProjectScoped, AuditModel):
    """Dictionary entry model"""
    __tablename__ = "dictionary"
    
    term = Column(String(200), nullable=False, index=True)
    definition = Column(Text, nullable=False)
    category = Column(Enum(DictionaryCategory), default=DictionaryCategory.BUSINESS_TERM)
    status = Column(Enum(DictionaryStatus), default=DictionaryStatus.DRAFT)
    
    # Additional metadata
    synonyms = Column(JSON, default=list)  # List of alternative terms
    abbreviations = Column(JSON, default=list)  # List of abbreviations
    domain_tags = Column(JSON, default=list)  # List of domain tags
    
    # Context and examples
    context = Column(Text, nullable=True)
    examples = Column(JSON, default=list)  # List of usage examples
    
    # Relationships and mappings
    related_terms = Column(JSON, default=list)  # List of related term IDs
    table_mappings = Column(JSON, default=list)  # List of mapped table IDs
    column_mappings = Column(JSON, default=list)  # List of mapped column IDs
    
    # Approval workflow
    version = Column(Integer, default=1)
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(String, nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    last_used_at = Column(String, nullable=True)
    
    # Auto-generation metadata
    is_auto_generated = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True)
    source_type = Column(String(50), nullable=True)  # column_name, table_name, data_sample
    
    # Relationships
    project = relationship("Project", back_populates="dictionary_entries")
    
    def __repr__(self):
        return f"<DictionaryEntry(id={self.id}, term='{self.term}', category='{self.category}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "synonyms": self.synonyms or [],
            "abbreviations": self.abbreviations or [],
            "domain_tags": self.domain_tags or [],
            "examples": self.examples or [],
            "related_terms": self.related_terms or [],
            "table_mappings": self.table_mappings or [],
            "column_mappings": self.column_mappings or [],
        })
        return data
    
    def add_synonym(self, synonym: str):
        """Add a synonym"""
        if not self.synonyms:
            self.synonyms = []
        if synonym not in self.synonyms:
            self.synonyms.append(synonym)
    
    def add_abbreviation(self, abbreviation: str):
        """Add an abbreviation"""
        if not self.abbreviations:
            self.abbreviations = []
        if abbreviation not in self.abbreviations:
            self.abbreviations.append(abbreviation)
    
    def add_domain_tag(self, tag: str):
        """Add a domain tag"""
        if not self.domain_tags:
            self.domain_tags = []
        if tag not in self.domain_tags:
            self.domain_tags.append(tag)
    
    def add_example(self, example: str):
        """Add a usage example"""
        if not self.examples:
            self.examples = []
        if example not in self.examples:
            self.examples.append(example)
    
    def add_related_term(self, term_id: int):
        """Add a related term"""
        if not self.related_terms:
            self.related_terms = []
        if term_id not in self.related_terms:
            self.related_terms.append(term_id)
    
    def map_to_table(self, table_id: int):
        """Map term to a table"""
        if not self.table_mappings:
            self.table_mappings = []
        if table_id not in self.table_mappings:
            self.table_mappings.append(table_id)
    
    def map_to_column(self, column_id: int):
        """Map term to a column"""
        if not self.column_mappings:
            self.column_mappings = []
        if column_id not in self.column_mappings:
            self.column_mappings.append(column_id)
    
    def approve(self, approved_by: str):
        """Approve the dictionary entry"""
        self.status = DictionaryStatus.APPROVED
        self.approved_by = approved_by
        self.approved_at = str(datetime.utcnow())
        self.version += 1
    
    def reject(self, rejected_by: str, reason: str):
        """Reject the dictionary entry"""
        self.status = DictionaryStatus.REJECTED
        self.rejection_reason = reason
        self.updated_by = rejected_by
    
    def submit_for_review(self):
        """Submit for review"""
        self.status = DictionaryStatus.PENDING_REVIEW
    
    def archive(self):
        """Archive the entry"""
        self.status = DictionaryStatus.ARCHIVED
    
    def increment_usage(self):
        """Increment usage count"""
        self.usage_count += 1
        self.last_used_at = str(datetime.utcnow())
    
    def get_all_terms(self) -> List[str]:
        """Get all terms including synonyms and abbreviations"""
        terms = [self.term]
        if self.synonyms:
            terms.extend(self.synonyms)
        if self.abbreviations:
            terms.extend(self.abbreviations)
        return terms
    
    def matches_term(self, search_term: str, exact: bool = False) -> bool:
        """Check if term matches search term"""
        search_term = search_term.lower().strip()
        all_terms = [term.lower().strip() for term in self.get_all_terms()]
        
        if exact:
            return search_term in all_terms
        else:
            return any(search_term in term or term in search_term for term in all_terms)
    
    def calculate_similarity(self, search_term: str) -> float:
        """Calculate similarity score with search term"""
        from difflib import SequenceMatcher
        
        search_term = search_term.lower().strip()
        all_terms = [term.lower().strip() for term in self.get_all_terms()]
        
        max_similarity = 0.0
        for term in all_terms:
            similarity = SequenceMatcher(None, search_term, term).ratio()
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    @classmethod
    def suggest_from_column(cls, column_name: str, table_name: str, 
                           data_samples: List[str] = None) -> Dict[str, Any]:
        """Suggest dictionary entry from column information"""
        # Clean column name
        clean_name = column_name.replace('_', ' ').replace('-', ' ').title()
        
        # Determine category
        category = DictionaryCategory.BUSINESS_TERM
        if column_name.upper() in ['ID', 'KEY', 'CODE']:
            category = DictionaryCategory.TECHNICAL_TERM
        elif len(column_name.split('_')) == 1 and column_name.isupper():
            category = DictionaryCategory.ABBREVIATION
        
        # Generate definition
        definition = f"Data field '{column_name}' from table '{table_name}'"
        if data_samples:
            definition += f". Sample values: {', '.join(str(s) for s in data_samples[:5])}"
        
        return {
            "term": clean_name,
            "definition": definition,
            "category": category,
            "synonyms": [column_name] if column_name != clean_name else [],
            "context": f"Automatically generated from column analysis",
            "is_auto_generated": True,
            "confidence_score": 0.7,
            "source_type": "column_name",
            "table_mappings": [],
            "column_mappings": []
        }
    
    @classmethod
    def suggest_from_table(cls, table_name: str, column_names: List[str] = None) -> Dict[str, Any]:
        """Suggest dictionary entry from table information"""
        # Clean table name
        clean_name = table_name.replace('_', ' ').replace('-', ' ').title()
        
        # Generate definition
        definition = f"Data entity '{table_name}'"
        if column_names:
            definition += f" with attributes: {', '.join(column_names[:5])}"
            if len(column_names) > 5:
                definition += f" and {len(column_names) - 5} more"
        
        return {
            "term": clean_name,
            "definition": definition,
            "category": DictionaryCategory.ENTITY,
            "synonyms": [table_name] if table_name != clean_name else [],
            "context": f"Automatically generated from table analysis",
            "is_auto_generated": True,
            "confidence_score": 0.8,
            "source_type": "table_name",
            "table_mappings": [],
            "column_mappings": []
        }

