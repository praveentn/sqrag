# services/dictionary_service.py
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter
import json

from models import db, Project, Table, Column, DictionaryEntry, DataSource
from config import Config

logger = logging.getLogger(__name__)

class DictionaryService:
    """Service for managing data dictionary entries and auto-generation"""
    
    def __init__(self):
        # Common business terms that we can auto-detect
        self.business_patterns = {
            'customer': ['customer', 'client', 'user', 'account_holder'],
            'revenue': ['revenue', 'sales', 'income', 'earnings'],
            'transaction': ['transaction', 'payment', 'order', 'purchase'],
            'product': ['product', 'item', 'sku', 'inventory'],
            'date': ['date', 'timestamp', 'created_at', 'updated_at'],
            'amount': ['amount', 'total', 'sum', 'value', 'price', 'cost'],
            'quantity': ['quantity', 'count', 'number', 'qty'],
            'status': ['status', 'state', 'flag', 'active', 'enabled'],
            'identifier': ['id', 'key', 'reference', 'ref', 'code']
        }
        
        # Domain-specific patterns
        self.domain_patterns = {
            'finance': ['balance', 'credit', 'debit', 'interest', 'principal', 'loan'],
            'hr': ['employee', 'salary', 'department', 'manager', 'position'],
            'sales': ['lead', 'opportunity', 'quote', 'commission', 'territory'],
            'marketing': ['campaign', 'conversion', 'impression', 'click', 'engagement'],
            'operations': ['inventory', 'warehouse', 'shipment', 'logistics', 'supply']
        }
    
    def generate_suggestions(self, project_id: int) -> List[Dict[str, Any]]:
        """Generate dictionary suggestions from project data"""
        try:
            suggestions = []
            
            # Validate project exists
            project = db.session.get(Project, project_id)
            if not project:
                logger.warning(f"Project {project_id} not found")
                return []
            
            # Get all tables and columns for the project
            tables = db.session.query(Table).join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            if not tables:
                logger.info(f"No tables found for project {project_id}")
                return []
            
            # Analyze column names and generate suggestions
            for table in tables:
                try:
                    table_suggestions = self._analyze_table_for_terms(table)
                    suggestions.extend(table_suggestions)
                except Exception as e:
                    logger.warning(f"Error analyzing table {table.name}: {str(e)}")
                    continue
            
            # Remove duplicates and rank by confidence
            unique_suggestions = self._deduplicate_suggestions(suggestions)
            ranked_suggestions = sorted(unique_suggestions, 
                                      key=lambda x: x.get('confidence_score', 0), 
                                      reverse=True)
            
            # Limit to top 50 suggestions
            final_suggestions = ranked_suggestions[:50]
            
            logger.info(f"Generated {len(final_suggestions)} dictionary suggestions for project {project_id}")
            return final_suggestions
            
        except Exception as e:
            logger.error(f"Error generating dictionary suggestions: {str(e)}")
            # Return empty list instead of raising exception
            return []
    
    def _analyze_table_for_terms(self, table: Table) -> List[Dict[str, Any]]:
        """Analyze a table and generate term suggestions"""
        suggestions = []
        
        # Analyze table name
        table_suggestions = self._extract_terms_from_name(
            table.name, 'table', table.id, table.name
        )
        suggestions.extend(table_suggestions)
        
        # Analyze column names
        for column in table.columns:
            try:
                column_suggestions = self._extract_terms_from_name(
                    column.name, 'column', column.id, f"{table.name}.{column.name}"
                )
                suggestions.extend(column_suggestions)
            except Exception as e:
                logger.warning(f"Error analyzing column {column.name}: {str(e)}")
                continue
        
        return suggestions
    
    def _extract_terms_from_name(self, name: str, object_type: str, 
                                object_id: int, full_name: str) -> List[Dict[str, Any]]:
        """Extract business terms from a name"""
        suggestions = []
        name_lower = name.lower()
        
        # Check business patterns
        for category, patterns in self.business_patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    confidence = self._calculate_confidence(name_lower, pattern, category)
                    suggestion = self._create_suggestion(
                        name, category, pattern, confidence, object_type, object_id, full_name
                    )
                    suggestions.append(suggestion)
        
        # Check domain patterns
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    confidence = self._calculate_confidence(name_lower, pattern, domain)
                    suggestion = self._create_suggestion(
                        name, domain, pattern, confidence, object_type, object_id, full_name
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_confidence(self, name: str, pattern: str, category: str) -> float:
        """Calculate confidence score for a term suggestion"""
        base_confidence = 0.5
        
        # Exact match
        if name == pattern:
            base_confidence = 0.9
        # Contains pattern
        elif pattern in name:
            base_confidence = 0.7
        # Partial match
        elif any(p in name for p in pattern.split('_')):
            base_confidence = 0.6
        
        # Boost confidence for important categories
        important_categories = ['customer', 'revenue', 'transaction', 'product']
        if category in important_categories:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _create_suggestion(self, name: str, category: str, pattern: str, 
                         confidence: float, object_type: str, object_id: int,
                         full_name: str) -> Dict[str, Any]:
        """Create a dictionary suggestion"""
        
        # Generate term and definition based on pattern
        term_mappings = {
            'customer': {
                'term': 'Customer',
                'definition': 'An individual or entity that purchases goods or services from the business'
            },
            'revenue': {
                'term': 'Revenue',
                'definition': 'The total amount of income generated by the business from its operations'
            },
            'transaction': {
                'term': 'Transaction',
                'definition': 'A business event or exchange involving the transfer of goods, services, or money'
            },
            'product': {
                'term': 'Product',
                'definition': 'A good or service offered by the business to customers'
            },
            'amount': {
                'term': 'Amount',
                'definition': 'A monetary value or quantity associated with a transaction or measurement'
            },
            'quantity': {
                'term': 'Quantity',
                'definition': 'The number or amount of items, units, or measures'
            },
            'status': {
                'term': 'Status',
                'definition': 'The current state or condition of an entity or process'
            },
            'identifier': {
                'term': 'Identifier',
                'definition': 'A unique value used to distinguish and reference a specific entity'
            },
            'date': {
                'term': 'Date',
                'definition': 'A temporal reference point indicating when an event occurred'
            }
        }
        
        mapping = term_mappings.get(category, {
            'term': pattern.title().replace('_', ' '),
            'definition': f'A {category} field containing {pattern} information'
        })
        
        return {
            'term': mapping['term'],
            'definition': mapping['definition'],
            'category': 'business_term',
            'domain': self._infer_domain(name),
            'confidence_score': round(confidence, 3),
            'source_object_type': object_type,
            'source_object_id': object_id,
            'source_name': full_name,
            'synonyms': self._generate_synonyms(category),
            'abbreviations': self._generate_abbreviations(category)
        }
    
    def _infer_domain(self, name: str) -> Optional[str]:
        """Infer business domain from name"""
        name_lower = name.lower()
        
        for domain, patterns in self.domain_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return domain
        
        # Default domain inference
        if any(keyword in name_lower for keyword in ['sale', 'order', 'customer']):
            return 'sales'
        elif any(keyword in name_lower for keyword in ['employee', 'hr', 'staff']):
            return 'hr'
        elif any(keyword in name_lower for keyword in ['finance', 'payment', 'cost']):
            return 'finance'
        elif any(keyword in name_lower for keyword in ['product', 'inventory', 'stock']):
            return 'operations'
        
        return None
    
    def _generate_synonyms(self, term: str) -> List[str]:
        """Generate synonyms for a term"""
        synonym_map = {
            'customer': ['client', 'buyer', 'purchaser'],
            'revenue': ['income', 'earnings', 'sales'],
            'transaction': ['payment', 'order', 'purchase'],
            'product': ['item', 'goods', 'merchandise'],
            'amount': ['value', 'sum', 'total'],
            'quantity': ['count', 'number', 'volume'],
            'status': ['state', 'condition', 'stage'],
            'identifier': ['id', 'key', 'reference'],
            'date': ['timestamp', 'time', 'when']
        }
        
        return synonym_map.get(term.lower(), [])
    
    def _generate_abbreviations(self, term: str) -> List[str]:
        """Generate common abbreviations for a term"""
        abbreviation_map = {
            'customer': ['cust', 'client'],
            'transaction': ['txn', 'trans'],
            'quantity': ['qty', 'qnt'],
            'amount': ['amt'],
            'identifier': ['id'],
            'number': ['num', 'no'],
            'reference': ['ref'],
            'status': ['stat'],
            'date': ['dt']
        }
        
        return abbreviation_map.get(term.lower(), [])
    
    def _deduplicate_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate suggestions and merge similar ones"""
        seen_terms = {}
        unique_suggestions = []
        
        for suggestion in suggestions:
            term_key = suggestion['term'].lower()
            
            if term_key in seen_terms:
                # Merge with existing suggestion if this one has higher confidence
                existing = seen_terms[term_key]
                if suggestion['confidence_score'] > existing['confidence_score']:
                    # Update the existing suggestion
                    existing.update(suggestion)
            else:
                seen_terms[term_key] = suggestion
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def create_entry_from_suggestion(self, project_id: int, 
                                   suggestion: Dict[str, Any]) -> DictionaryEntry:
        """Create a dictionary entry from a suggestion"""
        try:
            entry = DictionaryEntry(
                project_id=project_id,
                term=suggestion['term'],
                definition=suggestion['definition'],
                category=suggestion.get('category', 'business_term'),
                domain=suggestion.get('domain'),
                synonyms=suggestion.get('synonyms', []),
                abbreviations=suggestion.get('abbreviations', []),
                status='draft',
                is_auto_generated=True,
                confidence_score=suggestion.get('confidence_score'),
                source_tables=json.dumps([suggestion.get('source_name')]) if suggestion.get('source_name') else None,
                created_by='system'
            )
            
            db.session.add(entry)
            db.session.commit()
            
            return entry
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating dictionary entry from suggestion: {str(e)}")
            raise
    
    def bulk_create_from_suggestions(self, project_id: int, 
                                   suggestions: List[Dict[str, Any]]) -> int:
        """Create multiple dictionary entries from suggestions"""
        try:
            created_count = 0
            
            for suggestion in suggestions:
                try:
                    # Check if term already exists
                    existing = DictionaryEntry.query.filter_by(
                        project_id=project_id,
                        term=suggestion['term']
                    ).first()
                    
                    if not existing:
                        self.create_entry_from_suggestion(project_id, suggestion)
                        created_count += 1
                    else:
                        logger.info(f"Term '{suggestion['term']}' already exists, skipping")
                        
                except Exception as e:
                    logger.warning(f"Error creating entry for term '{suggestion.get('term', 'unknown')}': {str(e)}")
                    continue
            
            return created_count
            
        except Exception as e:
            logger.error(f"Error in bulk creation: {str(e)}")
            raise
    
    def update_entry(self, entry_id: int, updates: Dict[str, Any]) -> DictionaryEntry:
        """Update a dictionary entry"""
        try:
            entry = db.session.get(DictionaryEntry, entry_id)
            if not entry:
                raise ValueError(f"Dictionary entry {entry_id} not found")
            
            # Update allowed fields
            allowed_fields = [
                'term', 'definition', 'category', 'domain', 
                'synonyms', 'abbreviations', 'status'
            ]
            
            for field in allowed_fields:
                if field in updates:
                    setattr(entry, field, updates[field])
            
            entry.updated_at = datetime.utcnow()
            db.session.commit()
            
            return entry
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating dictionary entry {entry_id}: {str(e)}")
            raise
    
    def delete_entry(self, entry_id: int) -> bool:
        """Delete a dictionary entry"""
        try:
            entry = db.session.get(DictionaryEntry, entry_id)
            if not entry:
                raise ValueError(f"Dictionary entry {entry_id} not found")
            
            db.session.delete(entry)
            db.session.commit()
            
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting dictionary entry {entry_id}: {str(e)}")
            raise
    
    def get_entries_by_project(self, project_id: int, status: Optional[str] = None) -> List[DictionaryEntry]:
        """Get dictionary entries for a project"""
        try:
            query = DictionaryEntry.query.filter_by(project_id=project_id)
            
            if status:
                query = query.filter_by(status=status)
            
            return query.order_by(DictionaryEntry.term).all()
            
        except Exception as e:
            logger.error(f"Error getting dictionary entries: {str(e)}")
            raise
    
    def search_entries(self, project_id: int, search_term: str) -> List[DictionaryEntry]:
        """Search dictionary entries by term or definition"""
        try:
            search_pattern = f"%{search_term}%"
            
            entries = DictionaryEntry.query.filter(
                DictionaryEntry.project_id == project_id,
                db.or_(
                    DictionaryEntry.term.ilike(search_pattern),
                    DictionaryEntry.definition.ilike(search_pattern)
                )
            ).order_by(DictionaryEntry.term).all()
            
            return entries
            
        except Exception as e:
            logger.error(f"Error searching dictionary entries: {str(e)}")
            raise