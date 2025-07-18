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
            
            # Get all tables and columns for the project
            tables = db.session.query(Table).join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            # Analyze column names and generate suggestions
            for table in tables:
                table_suggestions = self._analyze_table_for_terms(table)
                suggestions.extend(table_suggestions)
            
            # Remove duplicates and rank by confidence
            unique_suggestions = self._deduplicate_suggestions(suggestions)
            ranked_suggestions = sorted(unique_suggestions, 
                                      key=lambda x: x['confidence_score'], 
                                      reverse=True)
            
            return ranked_suggestions[:50]  # Return top 50 suggestions
            
        except Exception as e:
            logger.error(f"Error generating dictionary suggestions: {str(e)}")
            raise
    
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
            column_suggestions = self._extract_terms_from_name(
                column.name, 'column', column.id, f"{table.name}.{column.name}"
            )
            suggestions.extend(column_suggestions)
        
        return suggestions
    
    def _extract_terms_from_name(self, name: str, object_type: str, 
                                object_id: int, full_name: str) -> List[Dict[str, Any]]:
        """Extract potential business terms from a name"""
        suggestions = []
        name_lower = name.lower()
        
        # Check against business patterns
        for category, patterns in self.business_patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    confidence = self._calculate_confidence(name_lower, pattern, category)
                    if confidence > 0.3:  # Minimum confidence threshold
                        suggestion = self._create_suggestion(
                            name, category, pattern, confidence, 
                            object_type, object_id, full_name
                        )
                        suggestions.append(suggestion)
        
        # Check against domain patterns
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    confidence = self._calculate_confidence(name_lower, pattern, domain)
                    if confidence > 0.3:
                        suggestion = self._create_domain_suggestion(
                            name, domain, pattern, confidence,
                            object_type, object_id, full_name
                        )
                        suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_confidence(self, name: str, pattern: str, category: str) -> float:
        """Calculate confidence score for a term match"""
        base_confidence = 0.5
        
        # Exact match gets higher confidence
        if name == pattern:
            base_confidence = 0.9
        elif name.startswith(pattern) or name.endswith(pattern):
            base_confidence = 0.8
        elif pattern in name:
            base_confidence = 0.6
        
        # Adjust based on pattern length and specificity
        if len(pattern) > 6:  # Longer patterns are more specific
            base_confidence += 0.1
        
        # Adjust based on category importance
        important_categories = ['customer', 'revenue', 'transaction']
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
            }
        }
        
        mapping = term_mappings.get(category, {
            'term': pattern.title(),
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
            'pattern_matched': pattern,
            'synonyms': self._generate_synonyms(pattern),
            'abbreviations': self._generate_abbreviations(pattern)
        }
    
    def _create_domain_suggestion(self, name: str, domain: str, pattern: str,
                                confidence: float, object_type: str, object_id: int,
                                full_name: str) -> Dict[str, Any]:
        """Create a domain-specific suggestion"""
        
        domain_definitions = {
            'finance': f'A financial term related to {pattern} in accounting and monetary operations',
            'hr': f'A human resources term related to {pattern} and employee management',
            'sales': f'A sales-related term involving {pattern} and customer acquisition',
            'marketing': f'A marketing term related to {pattern} and customer engagement',
            'operations': f'An operational term related to {pattern} and business processes'
        }
        
        return {
            'term': pattern.title(),
            'definition': domain_definitions.get(domain, f'A {domain} term related to {pattern}'),
            'category': 'domain_term',
            'domain': domain,
            'confidence_score': round(confidence, 3),
            'source_object_type': object_type,
            'source_object_id': object_id,
            'source_name': full_name,
            'pattern_matched': pattern,
            'synonyms': self._generate_synonyms(pattern),
            'abbreviations': self._generate_abbreviations(pattern)
        }
    
    def _infer_domain(self, name: str) -> Optional[str]:
        """Infer the business domain from a name"""
        name_lower = name.lower()
        
        for domain, patterns in self.domain_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return domain
        
        # Default inference based on common patterns
        if any(word in name_lower for word in ['price', 'cost', 'revenue', 'profit']):
            return 'finance'
        elif any(word in name_lower for word in ['employee', 'staff', 'manager']):
            return 'hr'
        elif any(word in name_lower for word in ['customer', 'client', 'sale']):
            return 'sales'
        
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
            'id': ['identifier', 'key', 'reference']
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
            'status': ['stat']
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
            entry = DictionaryEntry.query.get(entry_id)
            if not entry:
                raise ValueError(f"Dictionary entry {entry_id} not found")
            
            # Update allowed fields
            allowed_fields = [
                'term', 'definition', 'category', 'domain', 'synonyms',
                'abbreviations', 'status', 'approved_by'
            ]
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(entry, field):
                    setattr(entry, field, value)
            
            # Set approval timestamp if status changes to approved
            if updates.get('status') == 'approved' and entry.status != 'approved':
                entry.approved_at = datetime.utcnow()
            
            entry.updated_at = datetime.utcnow()
            db.session.commit()
            
            return entry
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating dictionary entry: {str(e)}")
            raise
    
    def search_entries(self, project_id: int, query: str, 
                      filters: Optional[Dict[str, Any]] = None) -> List[DictionaryEntry]:
        """Search dictionary entries"""
        try:
            base_query = DictionaryEntry.query.filter_by(project_id=project_id)
            
            # Apply text search
            if query:
                search_filter = (
                    DictionaryEntry.term.ilike(f'%{query}%') |
                    DictionaryEntry.definition.ilike(f'%{query}%')
                )
                base_query = base_query.filter(search_filter)
            
            # Apply filters
            if filters:
                if filters.get('category'):
                    base_query = base_query.filter_by(category=filters['category'])
                if filters.get('domain'):
                    base_query = base_query.filter_by(domain=filters['domain'])
                if filters.get('status'):
                    base_query = base_query.filter_by(status=filters['status'])
            
            return base_query.order_by(DictionaryEntry.term).all()
            
        except Exception as e:
            logger.error(f"Error searching dictionary entries: {str(e)}")
            raise
    
    def get_statistics(self, project_id: int) -> Dict[str, Any]:
        """Get dictionary statistics for a project"""
        try:
            entries = DictionaryEntry.query.filter_by(project_id=project_id).all()
            
            # Count by category
            categories = Counter(entry.category for entry in entries if entry.category)
            
            # Count by domain
            domains = Counter(entry.domain for entry in entries if entry.domain)
            
            # Count by status
            statuses = Counter(entry.status for entry in entries if entry.status)
            
            # Count auto-generated vs manual
            auto_generated = len([e for e in entries if e.is_auto_generated])
            manual = len(entries) - auto_generated
            
            return {
                'total_entries': len(entries),
                'categories': dict(categories),
                'domains': dict(domains),
                'statuses': dict(statuses),
                'auto_generated': auto_generated,
                'manual': manual,
                'completion_rate': round(
                    len([e for e in entries if e.status == 'approved']) / len(entries) * 100, 1
                ) if entries else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting dictionary statistics: {str(e)}")
            raise
    
    def export_dictionary(self, project_id: int, format: str = 'json') -> Dict[str, Any]:
        """Export dictionary entries"""
        try:
            entries = DictionaryEntry.query.filter_by(project_id=project_id).all()
            
            export_data = []
            for entry in entries:
                export_data.append({
                    'term': entry.term,
                    'definition': entry.definition,
                    'category': entry.category,
                    'domain': entry.domain,
                    'synonyms': entry.synonyms or [],
                    'abbreviations': entry.abbreviations or [],
                    'status': entry.status,
                    'created_at': entry.created_at.isoformat() if entry.created_at else None,
                    'updated_at': entry.updated_at.isoformat() if entry.updated_at else None
                })
            
            return {
                'project_id': project_id,
                'exported_at': datetime.utcnow().isoformat(),
                'total_entries': len(export_data),
                'entries': export_data
            }
            
        except Exception as e:
            logger.error(f"Error exporting dictionary: {str(e)}")
            raise