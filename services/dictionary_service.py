# services/dictionary_service.py
import re
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import defaultdict, Counter
import string

from models import db, DictionaryEntry, DataSource, Table, Column
from config import Config

logger = logging.getLogger(__name__)

class DictionaryService:
    """Manages data dictionary and domain encyclopedia"""
    
    def __init__(self):
        self.config = Config()
        self.common_words = {
            'id', 'key', 'name', 'code', 'type', 'status', 'date', 'time', 'timestamp',
            'created', 'updated', 'modified', 'deleted', 'active', 'inactive', 'enabled',
            'disabled', 'count', 'total', 'sum', 'avg', 'min', 'max', 'value', 'amount',
            'price', 'cost', 'rate', 'percent', 'percentage', 'flag', 'indicator',
            'description', 'comment', 'note', 'text', 'number', 'index', 'order',
            'sequence', 'rank', 'level', 'category', 'group', 'class', 'version'
        }
        
    def create_entry(self, term: str, definition: str, category: str = 'general',
                    synonyms: List[str] = None, approved: bool = False,
                    source_table: str = None, source_column: str = None) -> DictionaryEntry:
        """Create a new dictionary entry"""
        try:
            # Check if term already exists
            existing = DictionaryEntry.query.filter_by(term=term.lower()).first()
            if existing:
                raise ValueError(f"Term '{term}' already exists in dictionary")
            
            entry = DictionaryEntry(
                term=term.lower(),
                definition=definition,
                category=category,
                synonyms=synonyms or [],
                approved=approved,
                source_table=source_table,
                source_column=source_column,
                created_at=datetime.utcnow()
            )
            
            db.session.add(entry)
            db.session.commit()
            
            logger.info(f"Created dictionary entry: {term}")
            return entry
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating dictionary entry {term}: {str(e)}")
            raise
    
    def update_entry(self, entry_id: int, data: Dict[str, Any]) -> DictionaryEntry:
        """Update an existing dictionary entry"""
        try:
            entry = DictionaryEntry.query.get_or_404(entry_id)
            
            # Update allowed fields
            if 'definition' in data:
                entry.definition = data['definition']
            if 'category' in data:
                entry.category = data['category']
            if 'synonyms' in data:
                entry.synonyms = data['synonyms']
            if 'abbreviations' in data:
                entry.abbreviations = data['abbreviations']
            if 'approved' in data:
                entry.approved = data['approved']
            
            # Increment version
            entry.version += 1
            entry.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            logger.info(f"Updated dictionary entry: {entry.term}")
            return entry
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating dictionary entry {entry_id}: {str(e)}")
            raise
    
    def delete_entry(self, entry_id: int) -> None:
        """Delete a dictionary entry"""
        try:
            entry = DictionaryEntry.query.get_or_404(entry_id)
            term = entry.term
            
            db.session.delete(entry)
            db.session.commit()
            
            logger.info(f"Deleted dictionary entry: {term}")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting dictionary entry {entry_id}: {str(e)}")
            raise
    
    def auto_generate_dictionary(self) -> Dict[str, Any]:
        """Auto-generate dictionary entries from data sources"""
        try:
            stats = {
                'terms_generated': 0,
                'columns_processed': 0,
                'tables_processed': 0,
                'sources_processed': 0,
                'categories': defaultdict(int)
            }
            
            # Get all data sources
            sources = DataSource.query.all()
            
            for source in sources:
                stats['sources_processed'] += 1
                logger.info(f"Processing source: {source.name}")
                
                # Process each table
                for table in source.tables:
                    stats['tables_processed'] += 1
                    
                    # Generate table-level terms
                    table_terms = self._extract_table_terms(table)
                    for term_data in table_terms:
                        self._create_or_update_term(term_data, stats)
                    
                    # Process each column
                    for column in table.columns:
                        stats['columns_processed'] += 1
                        
                        # Generate column-level terms
                        column_terms = self._extract_column_terms(column)
                        for term_data in column_terms:
                            self._create_or_update_term(term_data, stats)
                        
                        # Extract terms from sample values
                        if column.sample_values:
                            value_terms = self._extract_value_terms(column)
                            for term_data in value_terms:
                                self._create_or_update_term(term_data, stats)
            
            db.session.commit()
            
            logger.info(f"Auto-generated {stats['terms_generated']} dictionary terms")
            return dict(stats)
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error auto-generating dictionary: {str(e)}")
            raise
    
    def _extract_table_terms(self, table: Table) -> List[Dict[str, Any]]:
        """Extract terms from table _metadata"""
        terms = []
        
        # Table name
        table_name = table.name.lower()
        clean_name = self._clean_identifier(table_name)
        
        if self._is_meaningful_term(clean_name):
            terms.append({
                'term': clean_name,
                'definition': f"Data table containing {table.display_name or table.name} information",
                'category': 'table',
                'source_table': table.name,
                'synonyms': [table.display_name.lower()] if table.display_name else []
            })
        
        # Extract terms from table name parts
        name_parts = self._split_identifier(table_name)
        for part in name_parts:
            if self._is_meaningful_term(part):
                terms.append({
                    'term': part,
                    'definition': f"Business entity or concept related to {part}",
                    'category': 'business_term',
                    'source_table': table.name
                })
        
        return terms
    
    def _extract_column_terms(self, column: Column) -> List[Dict[str, Any]]:
        """Extract terms from column _metadata"""
        terms = []
        
        # Column name
        column_name = column.name.lower()
        clean_name = self._clean_identifier(column_name)
        
        if self._is_meaningful_term(clean_name):
            definition = f"Data field: {column.display_name or column.name}"
            if column.data_type:
                definition += f" (Type: {column.data_type})"
            
            terms.append({
                'term': clean_name,
                'definition': definition,
                'category': 'column',
                'source_table': column.table.name,
                'source_column': column.name,
                'synonyms': [column.display_name.lower()] if column.display_name else []
            })
        
        # Extract terms from column name parts
        name_parts = self._split_identifier(column_name)
        for part in name_parts:
            if self._is_meaningful_term(part):
                terms.append({
                    'term': part,
                    'definition': f"Data attribute related to {part}",
                    'category': 'attribute',
                    'source_column': column.name
                })
        
        # Detect common patterns and generate terms
        pattern_terms = self._detect_column_patterns(column)
        terms.extend(pattern_terms)
        
        return terms
    
    def _extract_value_terms(self, column: Column) -> List[Dict[str, Any]]:
        """Extract terms from column sample values"""
        terms = []
        
        if not column.sample_values:
            return terms
        
        # Analyze sample values for enumerated values
        if column.unique_count and column.unique_count <= 20:  # Likely enumerated values
            for value in column.sample_values[:10]:  # Limit processing
                clean_value = self._clean_value(str(value))
                if self._is_meaningful_term(clean_value):
                    terms.append({
                        'term': clean_value,
                        'definition': f"Enumerated value for {column.display_name or column.name}",
                        'category': 'enumerated_value',
                        'source_table': column.table.name,
                        'source_column': column.name
                    })
        
        # Extract abbreviations and codes
        abbreviations = self._extract_abbreviations(column.sample_values)
        for abbrev, expansion in abbreviations.items():
            terms.append({
                'term': abbrev.lower(),
                'definition': f"Abbreviation for {expansion}",
                'category': 'abbreviation',
                'source_column': column.name,
                'abbreviations': [abbrev]
            })
        
        return terms
    
    def _detect_column_patterns(self, column: Column) -> List[Dict[str, Any]]:
        """Detect common column patterns and generate terms"""
        terms = []
        column_name = column.name.lower()
        
        # Date/time patterns
        if any(pattern in column_name for pattern in ['date', 'time', 'timestamp', 'created', 'updated']):
            terms.append({
                'term': 'timestamp',
                'definition': 'Date and time information',
                'category': 'data_type'
            })
        
        # ID patterns
        if column_name.endswith('_id') or column_name == 'id':
            entity = column_name.replace('_id', '') if column_name.endswith('_id') else 'record'
            terms.append({
                'term': f"{entity}_identifier",
                'definition': f"Unique identifier for {entity}",
                'category': 'identifier',
                'source_column': column.name
            })
        
        # Amount/money patterns
        if any(pattern in column_name for pattern in ['amount', 'price', 'cost', 'fee', 'total']):
            terms.append({
                'term': 'monetary_value',
                'definition': 'Financial amount or monetary value',
                'category': 'financial'
            })
        
        # Status/flag patterns
        if any(pattern in column_name for pattern in ['status', 'flag', 'active', 'enabled']):
            terms.append({
                'term': 'status_indicator',
                'definition': 'Status or state information',
                'category': 'status'
            })
        
        return terms
    
    def _clean_identifier(self, identifier: str) -> str:
        """Clean and normalize identifier names"""
        # Remove special characters and numbers
        clean = re.sub(r'[^a-zA-Z\s]', ' ', identifier)
        # Remove extra spaces
        clean = ' '.join(clean.split())
        return clean.strip().lower()
    
    def _split_identifier(self, identifier: str) -> List[str]:
        """Split identifier into meaningful parts"""
        # Handle camelCase and snake_case
        parts = []
        
        # Split on underscores
        underscore_parts = identifier.split('_')
        for part in underscore_parts:
            # Split camelCase
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', part)
            if camel_parts:
                parts.extend([p.lower() for p in camel_parts])
            else:
                parts.append(part.lower())
        
        # Filter out single characters and common words
        meaningful_parts = [
            part for part in parts 
            if len(part) > 1 and part not in self.common_words
        ]
        
        return meaningful_parts
    
    def _clean_value(self, value: str) -> str:
        """Clean and normalize sample values"""
        # Remove extra whitespace
        clean = ' '.join(value.split())
        # Remove special characters for term extraction
        clean = re.sub(r'[^\w\s]', ' ', clean)
        return clean.strip().lower()
    
    def _is_meaningful_term(self, term: str) -> bool:
        """Check if a term is meaningful for the dictionary"""
        if not term or len(term) < 2:
            return False
        
        # Skip common words
        if term in self.common_words:
            return False
        
        # Skip pure numbers
        if term.isdigit():
            return False
        
        # Skip single characters
        if len(term) == 1:
            return False
        
        # Skip terms that are mostly numbers
        if sum(c.isdigit() for c in term) > len(term) * 0.5:
            return False
        
        return True
    
    def _extract_abbreviations(self, values: List[str]) -> Dict[str, str]:
        """Extract potential abbreviations from values"""
        abbreviations = {}
        
        for value in values:
            value_str = str(value).strip()
            
            # Look for patterns like "USA", "NY", "CA" (2-4 uppercase letters)
            if re.match(r'^[A-Z]{2,4}$', value_str):
                # This could be an abbreviation
                expansion = self._guess_abbreviation_expansion(value_str)
                if expansion:
                    abbreviations[value_str] = expansion
        
        return abbreviations
    
    def _guess_abbreviation_expansion(self, abbrev: str) -> Optional[str]:
        """Guess the expansion of an abbreviation"""
        # Common abbreviations dictionary
        common_abbrevs = {
            'USA': 'United States of America',
            'UK': 'United Kingdom',
            'CA': 'California',
            'NY': 'New York',
            'TX': 'Texas',
            'FL': 'Florida',
            'LLC': 'Limited Liability Company',
            'Inc': 'Incorporated',
            'Corp': 'Corporation',
            'Ltd': 'Limited',
            'CEO': 'Chief Executive Officer',
            'CFO': 'Chief Financial Officer',
            'CTO': 'Chief Technology Officer',
            'HR': 'Human Resources',
            'IT': 'Information Technology',
            'AI': 'Artificial Intelligence',
            'ML': 'Machine Learning',
            'API': 'Application Programming Interface',
            'SQL': 'Structured Query Language',
            'DB': 'Database',
            'ID': 'Identifier'
        }
        
        return common_abbrevs.get(abbrev)
    
    def _create_or_update_term(self, term_data: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """Create or update a dictionary term"""
        term = term_data['term']
        
        # Check if term already exists
        existing = DictionaryEntry.query.filter_by(term=term).first()
        
        if existing:
            # Update existing term if new information is available
            if not existing.source_table and term_data.get('source_table'):
                existing.source_table = term_data['source_table']
            if not existing.source_column and term_data.get('source_column'):
                existing.source_column = term_data['source_column']
            
            # Merge synonyms
            existing_synonyms = set(existing.synonyms or [])
            new_synonyms = set(term_data.get('synonyms', []))
            existing.synonyms = list(existing_synonyms | new_synonyms)
            
            existing.updated_at = datetime.utcnow()
        else:
            # Create new term
            entry = DictionaryEntry(
                term=term,
                definition=term_data['definition'],
                category=term_data['category'],
                synonyms=term_data.get('synonyms', []),
                abbreviations=term_data.get('abbreviations', []),
                source_table=term_data.get('source_table'),
                source_column=term_data.get('source_column'),
                approved=False,  # Auto-generated terms need approval
                created_at=datetime.utcnow()
            )
            db.session.add(entry)
            
            stats['terms_generated'] += 1
            stats['categories'][term_data['category']] += 1
    
    def approve_entry(self, entry_id: int, approved_by: str = None) -> DictionaryEntry:
        """Approve a dictionary entry"""
        try:
            entry = DictionaryEntry.query.get_or_404(entry_id)
            entry.approved = True
            entry.approved_by = approved_by
            entry.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            logger.info(f"Approved dictionary entry: {entry.term}")
            return entry
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error approving dictionary entry {entry_id}: {str(e)}")
            raise
    
    def search_terms(self, query: str, category: str = None, approved_only: bool = False) -> List[DictionaryEntry]:
        """Search dictionary terms"""
        try:
            q = DictionaryEntry.query
            
            if query:
                # Search in term, definition, and synonyms
                search_filter = (
                    DictionaryEntry.term.ilike(f'%{query.lower()}%') |
                    DictionaryEntry.definition.ilike(f'%{query}%')
                )
                q = q.filter(search_filter)
            
            if category:
                q = q.filter(DictionaryEntry.category == category)
            
            if approved_only:
                q = q.filter(DictionaryEntry.approved == True)
            
            return q.order_by(DictionaryEntry.term).all()
            
        except Exception as e:
            logger.error(f"Error searching dictionary terms: {str(e)}")
            raise
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """Get all dictionary categories with counts"""
        try:
            categories = db.session.query(
                DictionaryEntry.category,
                db.func.count(DictionaryEntry.id).label('count'),
                db.func.sum(db.case([(DictionaryEntry.approved == True, 1)], else_=0)).label('approved_count')
            ).group_by(DictionaryEntry.category).all()
            
            return [
                {
                    'category': cat.category,
                    'total_count': cat.count,
                    'approved_count': cat.approved_count or 0
                }
                for cat in categories
            ]
            
        except Exception as e:
            logger.error(f"Error getting dictionary categories: {str(e)}")
            raise
    
    def export_dictionary(self, approved_only: bool = False) -> List[Dict[str, Any]]:
        """Export dictionary entries"""
        try:
            q = DictionaryEntry.query
            if approved_only:
                q = q.filter(DictionaryEntry.approved == True)
            
            entries = q.all()
            
            return [entry.to_dict() for entry in entries]
            
        except Exception as e:
            logger.error(f"Error exporting dictionary: {str(e)}")
            raise
    
    def import_dictionary(self, data: List[Dict[str, Any]], overwrite: bool = False) -> Dict[str, Any]:
        """Import dictionary entries from data"""
        try:
            stats = {
                'imported': 0,
                'updated': 0,
                'skipped': 0,
                'errors': 0
            }
            
            for item in data:
                try:
                    term = item.get('term', '').lower()
                    if not term:
                        stats['errors'] += 1
                        continue
                    
                    existing = DictionaryEntry.query.filter_by(term=term).first()
                    
                    if existing and not overwrite:
                        stats['skipped'] += 1
                        continue
                    
                    if existing and overwrite:
                        # Update existing
                        existing.definition = item.get('definition', existing.definition)
                        existing.category = item.get('category', existing.category)
                        existing.synonyms = item.get('synonyms', existing.synonyms)
                        existing.abbreviations = item.get('abbreviations', existing.abbreviations)
                        existing.approved = item.get('approved', existing.approved)
                        existing.version += 1
                        existing.updated_at = datetime.utcnow()
                        stats['updated'] += 1
                    else:
                        # Create new
                        entry = DictionaryEntry(
                            term=term,
                            definition=item.get('definition', ''),
                            category=item.get('category', 'general'),
                            synonyms=item.get('synonyms', []),
                            abbreviations=item.get('abbreviations', []),
                            approved=item.get('approved', False),
                            created_at=datetime.utcnow()
                        )
                        db.session.add(entry)
                        stats['imported'] += 1
                
                except Exception as e:
                    logger.warning(f"Error importing dictionary item: {str(e)}")
                    stats['errors'] += 1
            
            db.session.commit()
            
            logger.info(f"Dictionary import completed: {stats}")
            return stats
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error importing dictionary: {str(e)}")
            raise