# services/dictionary_service.py
import re
import json
import logging
from typing import Dict, List, Any, Optional, Set
from collections import Counter
from datetime import datetime
import openai
from openai import AzureOpenAI

from config import Config
from models import db, Project, Table, Column, DictionaryEntry

logger = logging.getLogger(__name__)

class DictionaryService:
    """Service for managing data dictionary and auto-generating terms"""
    
    def __init__(self):
        self._init_llm_client()
        self.common_words = self._load_common_words()
        
    def _init_llm_client(self):
        """Initialize Azure OpenAI client for term generation"""
        try:
            llm_config = Config.LLM_CONFIG['azure']
            self.client = AzureOpenAI(
                api_key=llm_config['api_key'],
                api_version=llm_config['api_version'],
                azure_endpoint=llm_config['endpoint']
            )
            self.deployment_name = llm_config['deployment_name']
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            self.client = None
    
    def generate_suggestions(self, project_id: int) -> Dict[str, Any]:
        """Generate dictionary suggestions from project data"""
        try:
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            suggestions = {
                'business_terms': [],
                'technical_terms': [], 
                'abbreviations': [],
                'domain_terms': {},
                'auto_generated_count': 0
            }
            
            # Analyze table and column names
            table_suggestions = self._analyze_table_names(project)
            column_suggestions = self._analyze_column_names(project)
            value_suggestions = self._analyze_sample_values(project)
            
            # Combine and categorize suggestions
            all_terms = table_suggestions + column_suggestions + value_suggestions
            categorized = self._categorize_terms(all_terms)
            
            suggestions.update(categorized)
            suggestions['auto_generated_count'] = len(all_terms)
            
            # Use LLM to enhance definitions if available
            if self.client:
                suggestions = self._enhance_with_llm(suggestions, project)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating dictionary suggestions: {str(e)}")
            raise
    
    def _analyze_table_names(self, project: Project) -> List[Dict[str, Any]]:
        """Analyze table names to extract business terms"""
        suggestions = []
        
        for source in project.sources:
            for table in source.tables:
                # Clean table name
                cleaned_name = self._clean_identifier(table.name)
                terms = self._extract_terms_from_name(cleaned_name)
                
                for term in terms:
                    if self._is_meaningful_term(term):
                        suggestion = {
                            'term': term,
                            'category': 'business_term',
                            'source_type': 'table',
                            'source_name': table.name,
                            'confidence': 0.8,
                            'auto_definition': f"Business entity representing {term} data",
                            'context': {
                                'table_name': table.name,
                                'row_count': table.row_count,
                                'source_id': source.id
                            }
                        }
                        suggestions.append(suggestion)
        
        return suggestions
    
    def _analyze_column_names(self, project: Project) -> List[Dict[str, Any]]:
        """Analyze column names to extract technical and business terms"""
        suggestions = []
        column_patterns = self._build_column_patterns()
        
        for source in project.sources:
            for table in source.tables:
                for column in table.columns:
                    # Clean column name
                    cleaned_name = self._clean_identifier(column.name)
                    terms = self._extract_terms_from_name(cleaned_name)
                    
                    for term in terms:
                        if self._is_meaningful_term(term):
                            # Determine category based on patterns
                            category = self._classify_column_term(term, column, column_patterns)
                            
                            suggestion = {
                                'term': term,
                                'category': category,
                                'source_type': 'column',
                                'source_name': f"{table.name}.{column.name}",
                                'confidence': self._calculate_term_confidence(term, column),
                                'auto_definition': self._generate_column_definition(term, column),
                                'context': {
                                    'table_name': table.name,
                                    'column_name': column.name,
                                    'data_type': column.data_type,
                                    'business_category': column.business_category,
                                    'sample_values': column.sample_values[:3] if column.sample_values else []
                                }
                            }
                            suggestions.append(suggestion)
        
        return suggestions
    
    def _analyze_sample_values(self, project: Project) -> List[Dict[str, Any]]:
        """Analyze sample values to identify abbreviations and coded values"""
        suggestions = []
        
        for source in project.sources:
            for table in source.tables:
                for column in table.columns:
                    if column.sample_values:
                        # Look for abbreviations
                        abbrevs = self._extract_abbreviations(column.sample_values)
                        for abbrev in abbrevs:
                            suggestion = {
                                'term': abbrev['code'],
                                'category': 'abbreviation',
                                'source_type': 'value',
                                'source_name': f"{table.name}.{column.name}",
                                'confidence': abbrev['confidence'],
                                'auto_definition': f"Abbreviated code: {abbrev['possible_meaning']}",
                                'context': {
                                    'table_name': table.name,
                                    'column_name': column.name,
                                    'sample_values': column.sample_values[:5],
                                    'frequency': abbrev['frequency']
                                }
                            }
                            suggestions.append(suggestion)
        
        return suggestions
    
    def _categorize_terms(self, terms: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Categorize terms by type and domain"""
        categorized = {
            'business_terms': [],
            'technical_terms': [],
            'abbreviations': [],
            'domain_terms': {}
        }
        
        # Group by category
        for term in terms:
            category = term['category']
            if category in categorized:
                categorized[category].append(term)
        
        # Group by domain
        for term in terms:
            context = term.get('context', {})
            domain = context.get('business_category')
            if domain:
                if domain not in categorized['domain_terms']:
                    categorized['domain_terms'][domain] = []
                categorized['domain_terms'][domain].append(term)
        
        # Remove duplicates and sort by confidence
        for category in categorized:
            if category != 'domain_terms':
                categorized[category] = self._deduplicate_terms(categorized[category])
                categorized[category].sort(key=lambda x: x['confidence'], reverse=True)
        
        # Handle domain terms
        for domain in categorized['domain_terms']:
            categorized['domain_terms'][domain] = self._deduplicate_terms(
                categorized['domain_terms'][domain]
            )
        
        return categorized
    
    def _enhance_with_llm(self, suggestions: Dict[str, Any], project: Project) -> Dict[str, Any]:
        """Use LLM to enhance term definitions and relationships"""
        try:
            # Focus on high-confidence business terms
            high_conf_terms = [
                term for term in suggestions['business_terms'] 
                if term['confidence'] > 0.7
            ][:10]  # Limit to top 10 to avoid long prompts
            
            if not high_conf_terms:
                return suggestions
            
            # Build context about the project
            project_context = self._build_project_context_for_llm(project)
            
            # Create prompt for LLM
            prompt = self._build_enhancement_prompt(high_conf_terms, project_context)
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse and apply enhancements
            enhanced = self._parse_llm_enhancements(response, high_conf_terms)
            
            # Update suggestions with enhancements
            for i, term in enumerate(suggestions['business_terms']):
                if i < len(enhanced):
                    term.update(enhanced[i])
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {str(e)}")
            return suggestions
    
    def _clean_identifier(self, name: str) -> str:
        """Clean table/column identifier"""
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^(tbl_|table_|col_|fld_)', '', name, flags=re.IGNORECASE)
        cleaned = re.sub(r'(_id|_key|_num|_code)$', '', cleaned, flags=re.IGNORECASE)
        
        # Replace underscores and camelCase
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1_\2', cleaned)
        cleaned = cleaned.replace('_', ' ')
        
        return cleaned.strip()
    
    def _extract_terms_from_name(self, name: str) -> List[str]:
        """Extract meaningful terms from a name"""
        # Split on common delimiters
        words = re.split(r'[\s_\-\.]+', name.lower())
        
        # Filter meaningful words
        terms = []
        for word in words:
            if len(word) > 2 and word not in self.common_words:
                terms.append(word.title())
        
        return terms
    
    def _is_meaningful_term(self, term: str) -> bool:
        """Check if a term is meaningful for the dictionary"""
        if len(term) < 3:
            return False
        
        # Skip common words
        if term.lower() in self.common_words:
            return False
        
        # Skip numeric or mostly numeric terms
        if re.match(r'^\d+$', term) or len(re.findall(r'\d', term)) / len(term) > 0.5:
            return False
        
        return True
    
    def _build_column_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for classifying column terms"""
        return {
            'financial': ['amount', 'price', 'cost', 'revenue', 'profit', 'salary', 'budget'],
            'temporal': ['date', 'time', 'year', 'month', 'day', 'created', 'updated', 'modified'],
            'identifier': ['id', 'key', 'code', 'number', 'ref', 'reference'],
            'descriptive': ['name', 'title', 'description', 'comment', 'note', 'label'],
            'quantitative': ['count', 'quantity', 'volume', 'weight', 'length', 'size'],
            'status': ['status', 'state', 'flag', 'active', 'enabled', 'type', 'category']
        }
    
    def _classify_column_term(self, term: str, column: Column, patterns: Dict[str, List[str]]) -> str:
        """Classify a column term by category"""
        term_lower = term.lower()
        
        # Check against patterns
        for category, keywords in patterns.items():
            if any(keyword in term_lower for keyword in keywords):
                return 'technical_term'
        
        # Check data type
        if column.data_type in ['INTEGER', 'FLOAT', 'DECIMAL']:
            return 'technical_term'
        elif column.data_type in ['TIMESTAMP', 'DATE', 'TIME']:
            return 'technical_term'
        
        # Default to business term
        return 'business_term'
    
    def _calculate_term_confidence(self, term: str, column: Column) -> float:
        """Calculate confidence score for a term"""
        confidence = 0.5
        
        # Boost confidence based on column metadata
        if column.description:
            confidence += 0.2
        
        if column.business_category:
            confidence += 0.1
        
        # Boost for meaningful names
        if len(term) > 5:
            confidence += 0.1
        
        # Boost for non-PII data
        if not column.pii_flag:
            confidence += 0.1
        
        return round(min(confidence, 1.0), 3)
    
    def _generate_column_definition(self, term: str, column: Column) -> str:
        """Generate automatic definition for a column term"""
        if column.description:
            return f"Column field: {column.description}"
        
        # Generate based on data type and patterns
        if column.data_type in ['INTEGER', 'FLOAT']:
            return f"Numeric field representing {term} values"
        elif column.data_type in ['TIMESTAMP', 'DATE']:
            return f"Date/time field for {term} information"
        elif column.pii_flag:
            return f"Personal information field for {term}"
        else:
            return f"Data field containing {term} information"
    
    def _extract_abbreviations(self, sample_values: List[str]) -> List[Dict[str, Any]]:
        """Extract potential abbreviations from sample values"""
        abbreviations = []
        
        # Count value frequencies
        value_counts = Counter(sample_values)
        
        for value, count in value_counts.items():
            value_str = str(value).strip().upper()
            
            # Look for abbreviation patterns
            if self._looks_like_abbreviation(value_str):
                abbrev = {
                    'code': value_str,
                    'frequency': count,
                    'confidence': self._calculate_abbrev_confidence(value_str, count, len(sample_values)),
                    'possible_meaning': self._guess_abbreviation_meaning(value_str)
                }
                abbreviations.append(abbrev)
        
        return abbreviations
    
    def _looks_like_abbreviation(self, value: str) -> bool:
        """Check if a value looks like an abbreviation"""
        if not value or len(value) > 10:
            return False
        
        # Patterns for abbreviations
        patterns = [
            r'^[A-Z]{2,5}$',  # All caps, 2-5 letters
            r'^[A-Z]\d{1,3}$',  # Letter followed by digits
            r'^[A-Z]{1,3}_[A-Z]{1,3}$',  # Underscore separated
            r'^[A-Z]{2,3}\-[A-Z]{2,3}$'  # Hyphen separated
        ]
        
        return any(re.match(pattern, value) for pattern in patterns)
    
    def _calculate_abbrev_confidence(self, value: str, frequency: int, total_samples: int) -> float:
        """Calculate confidence for abbreviation"""
        # Base confidence
        confidence = 0.6
        
        # Boost for common patterns
        if re.match(r'^[A-Z]{2,3}$', value):
            confidence += 0.2
        
        # Boost for frequency
        if frequency / total_samples > 0.1:
            confidence += 0.1
        
        return round(min(confidence, 1.0), 3)
    
    def _guess_abbreviation_meaning(self, abbrev: str) -> str:
        """Guess the meaning of an abbreviation"""
        # Common abbreviation mappings
        common_abbrevs = {
            'USA': 'United States of America',
            'UK': 'United Kingdom',
            'NYC': 'New York City',
            'LA': 'Los Angeles',
            'CA': 'California',
            'NY': 'New York',
            'FL': 'Florida',
            'TX': 'Texas',
            'USD': 'US Dollar',
            'EUR': 'Euro',
            'GBP': 'British Pound',
            'CEO': 'Chief Executive Officer',
            'CTO': 'Chief Technology Officer',
            'HR': 'Human Resources',
            'IT': 'Information Technology',
            'QA': 'Quality Assurance'
        }
        
        if abbrev in common_abbrevs:
            return common_abbrevs[abbrev]
        
        # Generate generic meaning
        if len(abbrev) == 2:
            return f"Two-letter code for {abbrev}"
        elif len(abbrev) == 3:
            return f"Three-letter code for {abbrev}"
        else:
            return f"Coded value: {abbrev}"
    
    def _deduplicate_terms(self, terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate terms, keeping highest confidence"""
        seen_terms = {}
        
        for term in terms:
            term_key = term['term'].lower()
            if term_key not in seen_terms or term['confidence'] > seen_terms[term_key]['confidence']:
                seen_terms[term_key] = term
        
        return list(seen_terms.values())
    
    def _load_common_words(self) -> Set[str]:
        """Load common English words to filter out"""
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an',
            'all', 'any', 'some', 'no', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'now', 'how', 'where', 'why', 'when', 'what',
            'who', 'which', 'if', 'then', 'else', 'each', 'every', 'both', 'either',
            'neither', 'more', 'most', 'other', 'such', 'few', 'many', 'much',
            'data', 'info', 'information', 'record', 'field', 'value', 'item'
        }
        return common_words
    
    def _build_project_context_for_llm(self, project: Project) -> str:
        """Build context about project for LLM enhancement"""
        context_parts = [
            f"Project: {project.name}",
            f"Description: {project.description or 'No description'}"
        ]
        
        # Add table overview
        table_names = []
        for source in project.sources:
            for table in source.tables:
                table_names.append(table.name)
        
        if table_names:
            context_parts.append(f"Tables: {', '.join(table_names[:10])}")
        
        return "\n".join(context_parts)
    
    def _build_enhancement_prompt(self, terms: List[Dict], context: str) -> str:
        """Build prompt for LLM to enhance term definitions"""
        terms_text = "\n".join([
            f"- {term['term']}: {term['auto_definition']}"
            for term in terms
        ])
        
        prompt = f"""
You are a data analyst creating a business glossary. Given the following context and automatically generated terms, 
provide enhanced definitions that are clear, business-focused, and accurate.

Context:
{context}

Terms to enhance:
{terms_text}

For each term, provide:
1. An improved definition (1-2 sentences)
2. Any relevant synonyms
3. The business domain it belongs to

Respond in JSON format:
[
  {{
    "term": "term_name",
    "enhanced_definition": "improved definition",
    "synonyms": ["synonym1", "synonym2"],
    "domain": "business_domain"
  }}
]
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call Azure OpenAI for enhancement"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert business analyst and data dictionary curator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM for enhancement: {str(e)}")
            raise
    
    def _parse_llm_enhancements(self, response: str, original_terms: List[Dict]) -> List[Dict]:
        """Parse LLM response and apply enhancements"""
        try:
            enhancements = json.loads(response)
            enhanced_terms = []
            
            for i, enhancement in enumerate(enhancements):
                if i < len(original_terms):
                    enhanced = original_terms[i].copy()
                    enhanced.update({
                        'enhanced_definition': enhancement.get('enhanced_definition'),
                        'suggested_synonyms': enhancement.get('synonyms', []),
                        'suggested_domain': enhancement.get('domain'),
                        'llm_enhanced': True
                    })
                    enhanced_terms.append(enhanced)
            
            return enhanced_terms
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM enhancement response")
            return original_terms
    
    def create_bulk_entries(self, project_id: int, suggestions: List[Dict[str, Any]], 
                           created_by: str = 'system') -> List[int]:
        """Create multiple dictionary entries from suggestions"""
        try:
            created_ids = []
            
            for suggestion in suggestions:
                # Check if term already exists
                existing = DictionaryEntry.query.filter_by(
                    project_id=project_id,
                    term=suggestion['term']
                ).first()
                
                if existing:
                    continue
                
                # Create new entry
                entry = DictionaryEntry(
                    project_id=project_id,
                    term=suggestion['term'],
                    definition=suggestion.get('enhanced_definition') or suggestion.get('auto_definition'),
                    category=suggestion.get('category', 'business_term'),
                    synonyms=suggestion.get('suggested_synonyms', []),
                    domain=suggestion.get('suggested_domain'),
                    is_auto_generated=True,
                    confidence_score=suggestion.get('confidence', 0.5),
                    source_tables=suggestion.get('context', {}).get('table_name'),
                    created_by=created_by,
                    status='draft'
                )
                
                db.session.add(entry)
                db.session.flush()
                created_ids.append(entry.id)
            
            db.session.commit()
            return created_ids
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating bulk dictionary entries: {str(e)}")
            raise
    
    def export_dictionary(self, project_id: int) -> Dict[str, Any]:
        """Export dictionary to JSON format"""
        try:
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            entries = DictionaryEntry.query.filter_by(
                project_id=project_id
            ).filter(
                DictionaryEntry.status != 'archived'
            ).all()
            
            export_data = {
                'project': {
                    'id': project.id,
                    'name': project.name,
                    'description': project.description
                },
                'exported_at': datetime.utcnow().isoformat(),
                'entries': [entry.to_dict() for entry in entries],
                'total_entries': len(entries)
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting dictionary: {str(e)}")
            raise