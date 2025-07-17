# services/chat_service.py
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, text
from fuzzywuzzy import fuzz
import openai
from openai import AzureOpenAI

from config import Config
from models import db, Project, Table, Column, DictionaryEntry, DataSource, NLQFeedback
from services.search_service import SearchService

logger = logging.getLogger(__name__)

class ChatService:
    """Service for Natural Language to SQL conversation pipeline"""
    
    def __init__(self):
        self.search_service = SearchService()
        self._init_llm_client()
        self.entity_confidence_threshold = Config.ENTITY_CONFIG['similarity_threshold']
        self.max_entities = Config.ENTITY_CONFIG['max_entities']
        
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
            self.model_name = llm_config['model_name']
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise
    
    def extract_entities(self, query: str, project_id: int) -> Dict[str, Any]:
        """Extract entities from natural language query using LLM"""
        start_time = time.time()
        
        try:
            # Get project context
            context = self._build_project_context(project_id)
            
            # Build prompt for entity extraction
            prompt = Config.PROMPTS['entity_extraction'].format(
                query=query,
                tables=context['tables'],
                columns=context['columns'],
                dictionary_terms=context['dictionary_terms']
            )
            
            # Call Azure OpenAI
            response = self._call_llm(prompt, max_tokens=1000)
            
            # Parse response
            try:
                entities_raw = json.loads(response)
                entities = self._validate_and_enhance_entities(entities_raw, project_id)
            except json.JSONDecodeError:
                # Fallback: extract entities using regex if JSON parsing fails
                logger.warning("LLM response not valid JSON, using fallback extraction")
                entities = self._fallback_entity_extraction(query, project_id)
            
            extraction_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                'entities': entities,
                'extraction_time_ms': extraction_time,
                'query': query,
                'context_used': {
                    'tables_count': len(context['tables']),
                    'columns_count': len(context['columns']),
                    'dictionary_terms_count': len(context['dictionary_terms'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            raise Exception(f"Entity extraction failed: {str(e)}")
    
    def map_entities_to_schema(self, entities: List[Dict], project_id: int) -> Dict[str, Any]:
        """Map extracted entities to actual table/column schema"""
        start_time = time.time()
        
        try:
            mappings = []
            
            for entity in entities:
                entity_text = entity['entity']
                entity_type = entity.get('type', 'unknown')
                confidence = entity.get('confidence', 0.5)
                
                # Find best matches for this entity
                matches = self._find_entity_matches(entity_text, entity_type, project_id)
                
                if matches:
                    mapping = {
                        'entity': entity_text,
                        'entity_type': entity_type,
                        'entity_confidence': confidence,
                        'matches': matches,
                        'best_match': matches[0] if matches else None
                    }
                    mappings.append(mapping)
            
            # Rank and filter mappings
            ranked_mappings = self._rank_mappings(mappings, project_id)
            
            mapping_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                'mappings': ranked_mappings,
                'mapping_time_ms': mapping_time,
                'total_entities': len(entities),
                'mapped_entities': len(ranked_mappings)
            }
            
        except Exception as e:
            logger.error(f"Error mapping entities: {str(e)}")
            raise Exception(f"Entity mapping failed: {str(e)}")
    
    def generate_sql(self, query: str, entities: List[Dict], mappings: List[Dict], project_id: int) -> Dict[str, Any]:
        """Generate SQL query from natural language and mapped entities"""
        start_time = time.time()
        
        try:
            # Build enhanced context with mappings
            context = self._build_sql_generation_context(mappings, project_id)
            
            # Build prompt for SQL generation
            prompt = Config.PROMPTS['sql_generation'].format(
                query=query,
                schemas=context['schemas'],
                entities=json.dumps(entities, indent=2)
            )
            
            # Call Azure OpenAI for SQL generation
            response = self._call_llm(prompt, max_tokens=1500)
            
            # Parse response
            try:
                sql_result = json.loads(response)
            except json.JSONDecodeError:
                # Fallback: extract SQL from text response
                sql_result = self._fallback_sql_extraction(response)
            
            # Validate and clean SQL
            sql_query = sql_result.get('sql', '')
            validated_sql = self._validate_and_clean_sql(sql_query)
            
            generation_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                'sql': validated_sql,
                'rationale': sql_result.get('rationale', ''),
                'confidence': sql_result.get('confidence', 0.7),
                'tables_used': sql_result.get('tables_used', []),
                'assumptions': sql_result.get('assumptions', []),
                'generation_time_ms': generation_time,
                'context_tables': len(context['schemas'])
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise Exception(f"SQL generation failed: {str(e)}")
    
    def execute_sql_safely(self, sql_query: str, project_id: int) -> Dict[str, Any]:
        """Execute SQL query with safety checks"""
        start_time = time.time()
        
        try:
            # Validate SQL safety
            if not self._is_sql_safe(sql_query):
                raise Exception("SQL query failed safety validation")
            
            # Get project's data sources
            project = Project.query.get(project_id)
            if not project:
                raise Exception(f"Project {project_id} not found")
            
            # For now, we'll execute against the first database source
            # In production, you might want to let users choose or have a default
            db_source = None
            for source in project.sources:
                if source.type == 'database':
                    db_source = source
                    break
            
            if not db_source:
                # If no database source, we might need to create a temporary database
                # with file data or use SQLite in-memory database
                return self._execute_on_file_data(sql_query, project_id)
            
            # Execute on database
            connection_string = self._build_connection_string(db_source.connection_config)
            engine = create_engine(connection_string)
            
            with engine.connect() as conn:
                # Set query timeout
                conn = conn.execution_options(
                    autocommit=True,
                    isolation_level="AUTOCOMMIT"
                )
                
                result = conn.execute(text(sql_query))
                
                # Fetch results
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = result.fetchall()
                    data = [dict(zip(columns, row)) for row in rows]
                    
                    # Limit rows returned
                    max_rows = Config.SQL_CONFIG['max_rows']
                    if len(data) > max_rows:
                        data = data[:max_rows]
                        truncated = True
                    else:
                        truncated = False
                else:
                    columns = []
                    data = []
                    truncated = False
            
            execution_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                'success': True,
                'columns': columns,
                'data': data,
                'row_count': len(data),
                'truncated': truncated,
                'execution_time_ms': execution_time,
                'sql_query': sql_query
            }
            
        except Exception as e:
            execution_time = round((time.time() - start_time) * 1000, 2)
            logger.error(f"Error executing SQL: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time,
                'sql_query': sql_query
            }
    
    def refine_sql_with_feedback(self, original_query: str, generated_sql: str, 
                                result: Dict, feedback: str, project_id: int) -> Dict[str, Any]:
        """Refine SQL based on user feedback"""
        try:
            # Build context for refinement
            context = self._build_project_context(project_id)
            
            prompt = Config.PROMPTS['sql_refinement'].format(
                original_query=original_query,
                generated_sql=generated_sql,
                result=json.dumps(result, indent=2),
                feedback=feedback,
                schemas=context['schemas']
            )
            
            response = self._call_llm(prompt, max_tokens=1500)
            
            try:
                refined_result = json.loads(response)
            except json.JSONDecodeError:
                refined_result = self._fallback_sql_extraction(response)
            
            return refined_result
            
        except Exception as e:
            logger.error(f"Error refining SQL: {str(e)}")
            raise Exception(f"SQL refinement failed: {str(e)}")
    
    def _build_project_context(self, project_id: int) -> Dict[str, Any]:
        """Build context about project's tables, columns, and dictionary"""
        try:
            project = Project.query.get(project_id)
            if not project:
                raise Exception(f"Project {project_id} not found")
            
            # Get tables and columns
            tables_info = []
            columns_info = []
            
            for source in project.sources:
                for table in source.tables:
                    table_info = {
                        'name': table.name,
                        'display_name': table.display_name,
                        'description': table.description,
                        'row_count': table.row_count
                    }
                    tables_info.append(table_info)
                    
                    for column in table.columns:
                        column_info = {
                            'table': table.name,
                            'name': column.name,
                            'display_name': column.display_name,
                            'description': column.description,
                            'type': column.data_type,
                            'business_category': column.business_category
                        }
                        columns_info.append(column_info)
            
            # Get dictionary terms
            dictionary_terms = []
            for entry in project.dictionary_entries:
                if entry.status != 'archived':
                    term_info = {
                        'term': entry.term,
                        'definition': entry.definition,
                        'synonyms': entry.synonyms or [],
                        'category': entry.category,
                        'domain': entry.domain
                    }
                    dictionary_terms.append(term_info)
            
            return {
                'tables': tables_info,
                'columns': columns_info,
                'dictionary_terms': dictionary_terms,
                'schemas': self._build_schema_context(project_id)
            }
            
        except Exception as e:
            logger.error(f"Error building project context: {str(e)}")
            raise
    
    def _build_schema_context(self, project_id: int) -> str:
        """Build detailed schema context for SQL generation"""
        try:
            project = Project.query.get(project_id)
            schema_lines = []
            
            for source in project.sources:
                for table in source.tables:
                    schema_lines.append(f"\nTable: {table.name}")
                    if table.description:
                        schema_lines.append(f"Description: {table.description}")
                    
                    schema_lines.append("Columns:")
                    for column in table.columns:
                        col_line = f"  - {column.name} ({column.data_type})"
                        if column.description:
                            col_line += f" - {column.description}"
                        if column.is_primary_key:
                            col_line += " [PRIMARY KEY]"
                        if column.is_foreign_key:
                            col_line += " [FOREIGN KEY]"
                        schema_lines.append(col_line)
                    
                    schema_lines.append(f"Row count: {table.row_count}")
            
            return "\n".join(schema_lines)
            
        except Exception as e:
            logger.error(f"Error building schema context: {str(e)}")
            return ""
    
    def _find_entity_matches(self, entity_text: str, entity_type: str, project_id: int) -> List[Dict]:
        """Find matching tables/columns for an entity"""
        matches = []
        
        # Get all tables and columns for the project
        project = Project.query.get(project_id)
        if not project:
            return matches
        
        # Search in tables
        for source in project.sources:
            for table in source.tables:
                # Exact match
                if entity_text.lower() == table.name.lower():
                    matches.append({
                        'type': 'table',
                        'name': table.name,
                        'display_name': table.display_name,
                        'description': table.description,
                        'match_type': 'exact',
                        'confidence': 1.0,
                        'table_id': table.id
                    })
                # Fuzzy match
                elif fuzz.ratio(entity_text.lower(), table.name.lower()) > 80:
                    confidence = round(fuzz.ratio(entity_text.lower(), table.name.lower()) / 100, 3)
                    matches.append({
                        'type': 'table',
                        'name': table.name,
                        'display_name': table.display_name,
                        'description': table.description,
                        'match_type': 'fuzzy',
                        'confidence': confidence,
                        'table_id': table.id
                    })
                
                # Search in columns
                for column in table.columns:
                    # Exact match
                    if entity_text.lower() == column.name.lower():
                        matches.append({
                            'type': 'column',
                            'name': column.name,
                            'table': table.name,
                            'display_name': column.display_name,
                            'description': column.description,
                            'data_type': column.data_type,
                            'match_type': 'exact',
                            'confidence': 1.0,
                            'column_id': column.id,
                            'table_id': table.id
                        })
                    # Fuzzy match
                    elif fuzz.ratio(entity_text.lower(), column.name.lower()) > 70:
                        confidence = round(fuzz.ratio(entity_text.lower(), column.name.lower()) / 100, 3)
                        matches.append({
                            'type': 'column',
                            'name': column.name,
                            'table': table.name,
                            'display_name': column.display_name,
                            'description': column.description,
                            'data_type': column.data_type,
                            'match_type': 'fuzzy',
                            'confidence': confidence,
                            'column_id': column.id,
                            'table_id': table.id
                        })
        
        # Search in dictionary
        for entry in project.dictionary_entries:
            if entry.status != 'archived':
                # Exact match
                if entity_text.lower() == entry.term.lower():
                    matches.append({
                        'type': 'dictionary_term',
                        'name': entry.term,
                        'definition': entry.definition,
                        'synonyms': entry.synonyms,
                        'match_type': 'exact',
                        'confidence': 1.0,
                        'dictionary_id': entry.id
                    })
                # Synonym match
                elif entry.synonyms and any(entity_text.lower() == syn.lower() for syn in entry.synonyms):
                    matches.append({
                        'type': 'dictionary_term',
                        'name': entry.term,
                        'definition': entry.definition,
                        'synonyms': entry.synonyms,
                        'match_type': 'synonym',
                        'confidence': 0.9,
                        'dictionary_id': entry.id
                    })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches[:10]  # Return top 10 matches
    
    def _rank_mappings(self, mappings: List[Dict], project_id: int) -> List[Dict]:
        """Rank entity mappings by relevance and confidence"""
        weights = Config.ENTITY_CONFIG['ranking_weights']
        
        for mapping in mappings:
            total_score = 0.0
            
            for match in mapping['matches']:
                score = match['confidence']
                
                # Apply weights based on match type
                if match['match_type'] == 'exact':
                    score *= weights['exact_match']
                elif match['match_type'] == 'fuzzy':
                    score *= weights['fuzzy_match']
                
                # Boost table matches
                if match['type'] == 'table':
                    score *= weights['table_importance']
                
                match['weighted_score'] = round(score, 3)
                total_score += score
            
            mapping['total_score'] = round(total_score, 3)
        
        # Sort by total score
        mappings.sort(key=lambda x: x['total_score'], reverse=True)
        
        return mappings
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call Azure OpenAI LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst and SQL developer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise Exception(f"LLM call failed: {str(e)}")
    
    def _validate_and_enhance_entities(self, entities_raw: List[Dict], project_id: int) -> List[Dict]:
        """Validate and enhance extracted entities"""
        enhanced_entities = []
        
        for entity in entities_raw:
            if isinstance(entity, dict) and 'entity' in entity:
                enhanced = {
                    'entity': entity['entity'].strip(),
                    'type': entity.get('type', 'unknown'),
                    'confidence': round(float(entity.get('confidence', 0.5)), 3)
                }
                
                # Filter out very low confidence entities
                if enhanced['confidence'] >= self.entity_confidence_threshold:
                    enhanced_entities.append(enhanced)
        
        # Limit number of entities
        return enhanced_entities[:self.max_entities]
    
    def _fallback_entity_extraction(self, query: str, project_id: int) -> List[Dict]:
        """Fallback entity extraction using regex and keyword matching"""
        entities = []
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Get project context for matching
        context = self._build_project_context(project_id)
        
        # Match against table names
        for table in context['tables']:
            table_name = table['name'].lower()
            if table_name in words or any(word in table_name for word in words):
                entities.append({
                    'entity': table['name'],
                    'type': 'table',
                    'confidence': 0.8
                })
        
        # Match against column names
        for column in context['columns']:
            column_name = column['name'].lower()
            if column_name in words or any(word in column_name for word in words):
                entities.append({
                    'entity': column['name'],
                    'type': 'column',
                    'confidence': 0.7
                })
        
        return entities[:10]
    
    def _fallback_sql_extraction(self, response: str) -> Dict[str, Any]:
        """Extract SQL from text response when JSON parsing fails"""
        # Look for SQL in code blocks or after certain keywords
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?)\s*```',
            r'SQL:\s*(SELECT.*?)(?:\n|$)',
            r'(SELECT.*?)(?:\n|$)'
        ]
        
        sql_query = ""
        for pattern in sql_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sql_query = match.group(1).strip()
                break
        
        return {
            'sql': sql_query,
            'rationale': 'Generated using fallback extraction',
            'confidence': 0.6,
            'tables_used': [],
            'assumptions': ['Response parsing failed, used fallback extraction']
        }
    
    def _validate_and_clean_sql(self, sql_query: str) -> str:
        """Validate and clean SQL query"""
        if not sql_query:
            return sql_query
        
        # Remove common issues
        sql_query = sql_query.strip()
        
        # Remove trailing semicolon if present
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
        
        # Ensure it starts with SELECT
        if not sql_query.upper().startswith('SELECT'):
            raise Exception("Only SELECT queries are allowed")
        
        return sql_query
    
    def _is_sql_safe(self, sql_query: str) -> bool:
        """Check if SQL query is safe to execute"""
        sql_upper = sql_query.upper()
        
        # Check for allowed statements
        allowed = Config.SQL_CONFIG['allowed_statements']
        if not any(sql_upper.strip().startswith(stmt) for stmt in allowed):
            return False
        
        # Check for blocked keywords
        blocked = Config.SQL_CONFIG['blocked_keywords']
        if any(keyword in sql_upper for keyword in blocked):
            return False
        
        return True
    
    def _build_connection_string(self, config: Dict[str, Any]) -> str:
        """Build database connection string"""
        db_type = config['type']
        host = config['host']
        port = config.get('port', 5432)
        database = config['database']
        username = config['username']
        password = config['password']
        
        if db_type == 'postgresql':
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'mysql':
            return f"mysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'sqlite':
            return f"sqlite:///{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _execute_on_file_data(self, sql_query: str, project_id: int) -> Dict[str, Any]:
        """Execute SQL on file-based data using SQLite in-memory database"""
        # This is a placeholder for executing queries on file data
        # In a full implementation, you would:
        # 1. Create an in-memory SQLite database
        # 2. Load file data into tables
        # 3. Execute the query
        
        return {
            'success': False,
            'error': 'File-based SQL execution not implemented yet',
            'execution_time_ms': 0,
            'sql_query': sql_query
        }
    
    def _build_sql_generation_context(self, mappings: List[Dict], project_id: int) -> Dict[str, Any]:
        """Build context specifically for SQL generation"""
        schemas = []
        
        # Get unique table IDs from mappings
        table_ids = set()
        for mapping in mappings:
            for match in mapping['matches']:
                if match['type'] in ['table', 'column'] and 'table_id' in match:
                    table_ids.add(match['table_id'])
        
        # Build schema for each relevant table
        for table_id in table_ids:
            table = Table.query.get(table_id)
            if table:
                schema = {
                    'table': table.name,
                    'description': table.description,
                    'columns': []
                }
                
                for column in table.columns:
                    schema['columns'].append({
                        'name': column.name,
                        'type': column.data_type,
                        'description': column.description,
                        'nullable': column.is_nullable,
                        'primary_key': column.is_primary_key
                    })
                
                schemas.append(schema)
        
        return {'schemas': schemas}