# services/dictionary_service.py
import re
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
import openai
from openai import AzureOpenAI

from config import Config
from models import db, Project, Table, Column, DictionaryEntry, DataSource

logger = logging.getLogger(__name__)

class DictionaryService:
    """Service for managing data dictionary and auto-generating terms"""
    
    def __init__(self):
        self._init_llm_client()
        self.common_abbreviations = self._load_common_abbreviations()
        self.business_keywords = self._load_business_keywords()
        
    def _init_llm_client(self):
        """Initialize Azure OpenAI client"""
        try:
            llm_config = Config.LLM_CONFIG['azure']
            self.client = AzureOpenAI(
                api_key=llm_config['api_key'],
                api_version=llm_config['api_version'],
                azure_endpoint=llm_config['endpoint']
            )
            self.deployment_name = llm_config['deployment_name']
            logger.info("Azure OpenAI client initialized for dictionary service")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            self.client = None
    
    def generate_suggestions(self, project_id: int) -> Dict[str, Any]:
        """Generate comprehensive dictionary suggestions for a project"""
        try:
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            # Collect all textual data from the project
            project_data = self._collect_project_data(project_id)
            
            suggestions = {
                'business_terms': [],
                'technical_terms': [],
                'abbreviations': [],
                'domain_terms': {}
            }
            
            # Generate different types of suggestions
            suggestions['business_terms'] = self._generate_business_terms(project_data)
            suggestions['technical_terms'] = self._generate_technical_terms(project_data)
            suggestions['abbreviations'] = self._generate_abbreviations(project_data)
            suggestions['domain_terms'] = self._generate_domain_terms(project_data)
            
            # Enhance suggestions with AI if available
            if self.client:
                suggestions = self._enhance_suggestions_with_ai(suggestions, project_data)
            
            # Filter and rank suggestions
            suggestions = self._filter_and_rank_suggestions(suggestions, project_id)
            
            logger.info(f"Generated {self._count_suggestions(suggestions)} suggestions for project {project_id}")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating dictionary suggestions: {str(e)}")
            return {
                'business_terms': [],
                'technical_terms': [],
                'abbreviations': [],
                'domain_terms': {},
                'error': str(e)
            }
    
    def _collect_project_data(self, project_id: int) -> Dict[str, Any]:
        """Collect all relevant textual data from the project"""
        try:
            # Get tables and columns
            tables = db.session.query(Table).join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            columns = db.session.query(Column).join(Table).join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            # Collect existing dictionary terms
            existing_terms = DictionaryEntry.query.filter_by(
                project_id=project_id
            ).all()
            
            data = {
                'table_names': [table.name for table in tables],
                'table_display_names': [table.display_name for table in tables if table.display_name],
                'table_descriptions': [table.description for table in tables if table.description],
                'column_names': [col.name for col in columns],
                'column_display_names': [col.display_name for col in columns if col.display_name],
                'column_descriptions': [col.description for col in columns if col.description],
                'business_categories': [col.business_category for col in columns if col.business_category],
                'sample_values': [],
                'existing_terms': [term.term for term in existing_terms],
                'data_types': list(set(col.data_type for col in columns if col.data_type))
            }
            
            # Collect sample values
            for col in columns:
                if col.sample_values:
                    data['sample_values'].extend([str(val) for val in col.sample_values[:3]])
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting project data: {str(e)}")
            return {}
    
    def _generate_business_terms(self, project_data: Dict) -> List[Dict[str, Any]]:
        """Generate business term suggestions"""
        suggestions = []
        
        # Extract business terms from table names
        for table_name in project_data.get('table_names', []):
            terms = self._extract_business_terms_from_text(table_name)
            for term in terms:
                suggestions.append({
                    'term': term,
                    'auto_definition': f'Business entity or concept related to {table_name}',
                    'category': 'business_term',
                    'confidence': 0.7,
                    'context': f'Derived from table: {table_name}',
                    'suggested_domain': self._infer_domain(table_name),
                    'suggested_synonyms': self._find_synonyms(term)
                })
        
        # Extract from column names
        for col_name in project_data.get('column_names', []):
            terms = self._extract_business_terms_from_text(col_name)
            for term in terms:
                suggestions.append({
                    'term': term,
                    'auto_definition': f'Data attribute or field representing {term}',
                    'category': 'business_term',
                    'confidence': 0.6,
                    'context': f'Derived from column: {col_name}',
                    'suggested_domain': self._infer_domain(col_name),
                    'suggested_synonyms': self._find_synonyms(term)
                })
        
        # Extract from business categories
        for category in project_data.get('business_categories', []):
            if category:
                suggestions.append({
                    'term': category.replace('_', ' ').title(),
                    'auto_definition': f'Business category or classification',
                    'category': 'business_term',
                    'confidence': 0.8,
                    'context': f'Business category: {category}',
                    'suggested_domain': category,
                    'suggested_synonyms': []
                })
        
        return self._deduplicate_suggestions(suggestions)
    
    def _generate_technical_terms(self, project_data: Dict) -> List[Dict[str, Any]]:
        """Generate technical term suggestions"""
        suggestions = []
        
        # Data types as technical terms
        for data_type in project_data.get('data_types', []):
            if data_type and data_type.upper() not in ['VARCHAR', 'TEXT', 'INTEGER', 'FLOAT']:
                suggestions.append({
                    'term': data_type,
                    'auto_definition': f'Data type used in database schema',
                    'category': 'technical_term',
                    'confidence': 0.9,
                    'context': f'Database data type',
                    'suggested_domain': 'database',
                    'suggested_synonyms': []
                })
        
        # Technical patterns in column names
        technical_patterns = [
            r'.*_id$', r'.*_key$', r'.*_ref$', r'.*_code$',
            r'.*_timestamp$', r'.*_date$', r'.*_flag$'
        ]
        
        for col_name in project_data.get('column_names', []):
            for pattern in technical_patterns:
                if re.match(pattern, col_name, re.IGNORECASE):
                    term_type = pattern.split('_')[-1].replace('$', '')
                    suggestions.append({
                        'term': f'{term_type.title()} Field',
                        'auto_definition': f'Technical field type: {term_type}',
                        'category': 'technical_term',
                        'confidence': 0.7,
                        'context': f'Pattern found in: {col_name}',
                        'suggested_domain': 'database',
                        'suggested_synonyms': []
                    })
        
        return self._deduplicate_suggestions(suggestions)
    
    def _generate_abbreviations(self, project_data: Dict) -> List[Dict[str, Any]]:
        """Generate abbreviation suggestions"""
        suggestions = []
        
        # Find potential abbreviations in names
        all_names = (project_data.get('table_names', []) + 
                    project_data.get('column_names', []))
        
        for name in all_names:
            abbrevs = self._extract_abbreviations(name)
            for abbrev in abbrevs:
                full_form = self._expand_abbreviation(abbrev)
                if full_form:
                    suggestions.append({
                        'term': abbrev.upper(),
                        'auto_definition': f'Abbreviation for {full_form}',
                        'category': 'abbreviation',
                        'confidence': 0.6,
                        'context': f'Found in: {name}',
                        'suggested_domain': self._infer_domain(name),
                        'suggested_synonyms': [full_form]
                    })
        
        # Check against common abbreviations
        for name in all_names:
            name_parts = re.split(r'[_\s]+', name.lower())
            for part in name_parts:
                if part in self.common_abbreviations:
                    suggestions.append({
                        'term': part.upper(),
                        'auto_definition': f'Common abbreviation for {self.common_abbreviations[part]}',
                        'category': 'abbreviation',
                        'confidence': 0.8,
                        'context': f'Common abbreviation found in: {name}',
                        'suggested_domain': self._infer_domain(name),
                        'suggested_synonyms': [self.common_abbreviations[part]]
                    })
        
        return self._deduplicate_suggestions(suggestions)
    
    def _generate_domain_terms(self, project_data: Dict) -> Dict[str, List[Dict]]:
        """Generate domain-specific terms"""
        domain_suggestions = defaultdict(list)
        
        # Infer domains from table and column names
        all_names = (project_data.get('table_names', []) + 
                    project_data.get('column_names', []))
        
        domain_keywords = {
            'finance': ['revenue', 'cost', 'profit', 'expense', 'budget', 'price', 'amount', 'payment'],
            'hr': ['employee', 'salary', 'department', 'manager', 'hire', 'staff'],
            'sales': ['customer', 'order', 'product', 'quantity', 'discount'],
            'marketing': ['campaign', 'lead', 'conversion', 'click', 'impression'],
            'inventory': ['stock', 'warehouse', 'supply', 'vendor'],
            'operations': ['process', 'workflow', 'task', 'status']
        }
        
        for name in all_names:
            name_lower = name.lower()
            for domain, keywords in domain_keywords.items():
                for keyword in keywords:
                    if keyword in name_lower:
                        domain_suggestions[domain].append({
                            'term': name.replace('_', ' ').title(),
                            'auto_definition': f'{domain.title()} related term',
                            'category': 'domain_term',
                            'confidence': 0.7,
                            'context': f'Domain: {domain}, found in: {name}',
                            'suggested_domain': domain,
                            'suggested_synonyms': []
                        })
        
        # Deduplicate within each domain
        for domain in domain_suggestions:
            domain_suggestions[domain] = self._deduplicate_suggestions(domain_suggestions[domain])
        
        return dict(domain_suggestions)
    
    def _enhance_suggestions_with_ai(self, suggestions: Dict, project_data: Dict) -> Dict[str, Any]:
        """Enhance suggestions using AI"""
        if not self.client:
            return suggestions
        
        try:
            # Prepare context for AI
            context = {
                'table_names': project_data.get('table_names', [])[:10],
                'column_names': project_data.get('column_names', [])[:20],
                'sample_values': project_data.get('sample_values', [])[:10]
            }
            
            # Enhance business terms
            for suggestion in suggestions.get('business_terms', []):
                enhanced_def = self._enhance_definition_with_ai(
                    suggestion['term'], 
                    suggestion['auto_definition'], 
                    context
                )
                if enhanced_def:
                    suggestion['enhanced_definition'] = enhanced_def
                    suggestion['confidence'] = min(1.0, suggestion['confidence'] + 0.1)
            
            # Enhance technical terms
            for suggestion in suggestions.get('technical_terms', []):
                enhanced_def = self._enhance_definition_with_ai(
                    suggestion['term'], 
                    suggestion['auto_definition'], 
                    context,
                    is_technical=True
                )
                if enhanced_def:
                    suggestion['enhanced_definition'] = enhanced_def
                    suggestion['confidence'] = min(1.0, suggestion['confidence'] + 0.1)
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Error enhancing suggestions with AI: {str(e)}")
            return suggestions
    
    def _enhance_definition_with_ai(self, term: str, basic_definition: str, 
                                  context: Dict, is_technical: bool = False) -> Optional[str]:
        """Enhance a definition using AI"""
        try:
            context_type = "technical database" if is_technical else "business"
            
            prompt = f"""
            Improve this {context_type} definition for the term "{term}":
            
            Current definition: {basic_definition}
            
            Context from data schema:
            Tables: {', '.join(context.get('table_names', []))}
            Columns: {', '.join(context.get('column_names', []))}
            
            Provide a clear, concise definition (max 100 words) that would be useful in a data dictionary.
            """
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a data analyst helping create clear definitions for a data dictionary."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            enhanced_def = response.choices[0].message.content.strip()
            
            # Basic validation
            if len(enhanced_def) > 20 and enhanced_def != basic_definition:
                return enhanced_def
            
            return None
            
        except Exception as e:
            logger.warning(f"Error enhancing definition for {term}: {str(e)}")
            return None
    
    def _extract_business_terms_from_text(self, text: str) -> List[str]:
        """Extract potential business terms from text"""
        terms = []
        
        # Split camelCase and snake_case
        words = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', text)
        words.extend(text.split('_'))
        
        # Clean and filter words
        for word in words:
            word = word.strip().lower()
            if (len(word) > 2 and 
                word not in ['the', 'and', 'or', 'of', 'in', 'to', 'for', 'with'] and
                word in self.business_keywords):
                terms.append(word.title())
        
        return list(set(terms))
    
    def _extract_abbreviations(self, text: str) -> List[str]:
        """Extract potential abbreviations from text"""
        abbreviations = []
        
        # Look for uppercase letter sequences
        matches = re.findall(r'[A-Z]{2,}', text)
        abbreviations.extend(matches)
        
        # Look for patterns like 'id', 'cd', 'no'
        short_patterns = re.findall(r'\b[a-z]{2,3}\b', text.lower())
        for pattern in short_patterns:
            if pattern in self.common_abbreviations:
                abbreviations.append(pattern)
        
        return list(set(abbreviations))
    
    def _expand_abbreviation(self, abbrev: str) -> Optional[str]:
        """Expand abbreviation to full form"""
        abbrev_lower = abbrev.lower()
        
        if abbrev_lower in self.common_abbreviations:
            return self.common_abbreviations[abbrev_lower]
        
        # Try to guess based on context
        expansion_patterns = {
            'id': 'Identifier',
            'cd': 'Code',
            'no': 'Number',
            'qty': 'Quantity',
            'amt': 'Amount',
            'desc': 'Description',
            'addr': 'Address',
            'ref': 'Reference'
        }
        
        return expansion_patterns.get(abbrev_lower)
    
    def _infer_domain(self, text: str) -> Optional[str]:
        """Infer business domain from text"""
        text_lower = text.lower()
        
        domain_indicators = {
            'finance': ['revenue', 'cost', 'profit', 'expense', 'budget', 'price', 'amount', 'payment', 'invoice'],
            'hr': ['employee', 'staff', 'salary', 'department', 'manager', 'hire'],
            'sales': ['customer', 'order', 'product', 'quantity', 'discount', 'sale'],
            'marketing': ['campaign', 'lead', 'conversion', 'click', 'impression', 'ad'],
            'inventory': ['stock', 'warehouse', 'supply', 'vendor', 'item'],
            'operations': ['process', 'workflow', 'task', 'status', 'operation']
        }
        
        for domain, indicators in domain_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return domain
        
        return None
    
    def _find_synonyms(self, term: str) -> List[str]:
        """Find potential synonyms for a term"""
        synonym_map = {
            'customer': ['client', 'buyer'],
            'product': ['item', 'good'],
            'order': ['purchase', 'transaction'],
            'employee': ['staff', 'worker'],
            'revenue': ['income', 'sales'],
            'cost': ['expense', 'expenditure']
        }
        
        return synonym_map.get(term.lower(), [])
    
    def _deduplicate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Remove duplicate suggestions"""
        seen = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            term_key = suggestion['term'].lower()
            if term_key not in seen:
                seen.add(term_key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _filter_and_rank_suggestions(self, suggestions: Dict, project_id: int) -> Dict[str, Any]:
        """Filter and rank suggestions"""
        # Get existing terms to avoid duplicates
        existing_terms = set(
            term.term.lower() for term in 
            DictionaryEntry.query.filter_by(project_id=project_id).all()
        )
        
        # Filter and rank each category
        for category in suggestions:
            if isinstance(suggestions[category], list):
                # Filter out existing terms
                filtered = [
                    s for s in suggestions[category] 
                    if s['term'].lower() not in existing_terms
                ]
                
                # Sort by confidence
                filtered.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Limit results
                suggestions[category] = filtered[:20]
            
            elif isinstance(suggestions[category], dict):
                # Handle domain terms
                for domain in suggestions[category]:
                    filtered = [
                        s for s in suggestions[category][domain]
                        if s['term'].lower() not in existing_terms
                    ]
                    filtered.sort(key=lambda x: x['confidence'], reverse=True)
                    suggestions[category][domain] = filtered[:10]
        
        return suggestions
    
    def _count_suggestions(self, suggestions: Dict) -> int:
        """Count total suggestions"""
        count = 0
        for category, items in suggestions.items():
            if isinstance(items, list):
                count += len(items)
            elif isinstance(items, dict):
                for domain_items in items.values():
                    count += len(domain_items)
        return count
    
    def _load_common_abbreviations(self) -> Dict[str, str]:
        """Load common abbreviations mapping"""
        return {
            'id': 'Identifier',
            'cd': 'Code',
            'no': 'Number',
            'qty': 'Quantity',
            'amt': 'Amount',
            'desc': 'Description',
            'addr': 'Address',
            'ref': 'Reference',
            'dept': 'Department',
            'mgr': 'Manager',
            'emp': 'Employee',
            'cust': 'Customer',
            'prod': 'Product',
            'ord': 'Order',
            'inv': 'Invoice',
            'acct': 'Account',
            'bal': 'Balance',
            'calc': 'Calculated',
            'ctrl': 'Control',
            'stat': 'Status',
            'sys': 'System',
            'tmp': 'Temporary',
            'usr': 'User',
            'grp': 'Group',
            'org': 'Organization',
            'loc': 'Location',
            'cat': 'Category',
            'type': 'Type',
            'src': 'Source',
            'tgt': 'Target',
            'min': 'Minimum',
            'max': 'Maximum',
            'avg': 'Average',
            'cnt': 'Count',
            'sum': 'Sum',
            'pct': 'Percent'
        }
    
    def _load_business_keywords(self) -> Set[str]:
        """Load business keywords for term extraction"""
        return {
            'customer', 'client', 'buyer', 'user', 'account', 'contact',
            'product', 'service', 'item', 'good', 'offering',
            'order', 'purchase', 'transaction', 'sale', 'deal',
            'revenue', 'income', 'profit', 'cost', 'expense', 'budget',
            'price', 'amount', 'value', 'total', 'sum',
            'employee', 'staff', 'worker', 'manager', 'team',
            'department', 'division', 'unit', 'group', 'organization',
            'campaign', 'marketing', 'promotion', 'advertisement',
            'lead', 'prospect', 'opportunity', 'conversion',
            'inventory', 'stock', 'warehouse', 'supply', 'vendor',
            'process', 'workflow', 'task', 'project', 'activity',
            'report', 'dashboard', 'analytics', 'metric', 'kpi',
            'date', 'time', 'period', 'quarter', 'year',
            'status', 'state', 'condition', 'flag', 'indicator',
            'category', 'type', 'class', 'segment', 'group',
            'location', 'region', 'country', 'city', 'address',
            'contract', 'agreement', 'policy', 'terms',
            'payment', 'invoice', 'billing', 'charge', 'fee'
        }