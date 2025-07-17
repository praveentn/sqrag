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
    
    def extract_entities(self, query: str, project_id: int) -> List[Dict[str, Any]]:
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
                entities = json.loads(response)
                if isinstance(entities, dict) and 'entities' in entities:
                    entities = entities['entities']
            except json.JSONDecodeError:
                # Fallback: extract entities using pattern matching
                entities = self._extract_entities_fallback(query, context)
            
            # Validate and enhance entities
            validated_entities = self._validate_entities(entities, context)
            
            logger.info(f"Extracted {len(validated_entities)} entities in {time.time() - start_time:.2f}s")
            return validated_entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            # Fallback to rule-based extraction
            context = self._build_project_context(project_id)
            return self._extract_entities_fallback(query, context)
    
    def map_entities_to_schema(self, entities: List[Dict], project_id: int) -> List[Dict[str, Any]]:
        """Map extracted entities to database schema elements"""
        try:
            mappings = []
            
            # Get schema information
            tables = Table.query.join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            columns = Column.query.join(Table).join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            for entity in entities:
                entity_text = entity.get('text', '').lower()
                entity_type = entity.get('type', 'unknown')
                
                best_mappings = []
                
                # Map to tables
                for table in tables:
                    similarity = self._calculate_similarity(entity_text, table.name.lower())
                    if similarity > 0.6:  # Threshold for table matching
                        best_mappings.append({
                            'entity': entity_text,
                            'type': 'table',
                            'table': table.name,
                            'column': None,
                            'confidence': round(similarity, 3),
                            'object_id': table.id
                        })
                
                # Map to columns
                for column in columns:
                    # Check column name similarity
                    col_similarity = self._calculate_similarity(entity_text, column.name.lower())
                    
                    # Check display name similarity
                    if column.display_name:
                        display_similarity = self._calculate_similarity(
                            entity_text, column.display_name.lower()
                        )
                        col_similarity = max(col_similarity, display_similarity)
                    
                    # Check business category
                    if column.business_category:
                        cat_similarity = self._calculate_similarity(
                            entity_text, column.business_category.lower()
                        )
                        col_similarity = max(col_similarity, cat_similarity)
                    
                    # Check sample values
                    if column.sample_values:
                        for sample in column.sample_values[:5]:  # Check first 5 samples
                            sample_similarity = self._calculate_similarity(
                                entity_text, str(sample).lower()
                            )
                            col_similarity = max(col_similarity, sample_similarity)
                    
                    if col_similarity > 0.5:  # Threshold for column matching
                        best_mappings.append({
                            'entity': entity_text,
                            'type': 'column',
                            'table': column.table.name,
                            'column': column.name,
                            'confidence': round(col_similarity, 3),
                            'object_id': column.id,
                            'data_type': column.data_type
                        })
                
                # Sort by confidence and take top mappings
                best_mappings.sort(key=lambda x: x['confidence'], reverse=True)
                mappings.extend(best_mappings[:3])  # Top 3 mappings per entity
            
            # Remove duplicates and sort by confidence
            unique_mappings = []
            seen = set()
            
            for mapping in sorted(mappings, key=lambda x: x['confidence'], reverse=True):
                key = f"{mapping['table']}.{mapping['column']}"
                if key not in seen:
                    unique_mappings.append(mapping)
                    seen.add(key)
            
            return unique_mappings[:10]  # Return top 10 mappings
            
        except Exception as e:
            logger.error(f"Error mapping entities to schema: {str(e)}")
            return []
    
    def generate_sql(self, query: str, entities: List[Dict], 
                    mappings: List[Dict], project_id: int) -> Dict[str, Any]:
        """Generate SQL query from natural language and mappings"""
        try:
            # Build schema context
            schema_context = self._build_schema_context(mappings, project_id)
            
            # Build SQL generation prompt
            prompt = Config.PROMPTS['sql_generation'].format(
                query=query,
                entities=json.dumps(entities, indent=2),
                mappings=json.dumps(mappings, indent=2),
                schema=schema_context['schema'],
                relationships=schema_context['relationships']
            )
            
            # Call LLM
            response = self._call_llm(prompt, max_tokens=1500)
            
            # Parse response
            try:
                sql_result = json.loads(response)
            except json.JSONDecodeError:
                # Extract SQL from response if JSON parsing fails
                sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
                if sql_match:
                    sql_query = sql_match.group(1).strip()
                else:
                    # Try to find SQL without markdown
                    sql_query = self._extract_sql_from_text(response)
                
                sql_result = {
                    'sql': sql_query,
                    'explanation': 'Generated SQL query from natural language',
                    'confidence': 0.7
                }
            
            # Validate SQL
            sql_result = self._validate_and_enhance_sql(sql_result, mappings)
            
            return sql_result
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            return {
                'sql': '',
                'explanation': f'Error generating SQL: {str(e)}',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def execute_sql_safely(self, sql: str, project_id: int) -> Dict[str, Any]:
        """Execute SQL query with safety checks"""
        try:
            # Security checks
            if not self._is_sql_safe(sql):
                raise ValueError("SQL query contains potentially dangerous operations")
            
            # Get first data source for the project (assuming single DB per project for now)
            data_source = DataSource.query.filter_by(
                project_id=project_id,
                type='database'
            ).first()
            
            if not data_source:
                # For file-based sources, we need to create a temporary database
                return self._execute_on_file_sources(sql, project_id)
            
            # Execute on database source
            connection_string = self._build_connection_string(data_source.connection_config)
            engine = create_engine(connection_string)
            
            start_time = time.time()
            
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                columns = list(result.keys())
                data = [dict(row) for row in result.fetchall()]
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'data': data,
                'columns': columns,
                'row_count': len(data),
                'execution_time_seconds': round(execution_time, 3),
                'sql': sql
            }
            
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'sql': sql
            }
    
    def generate_natural_language_answer(self, query: str, sql: str, 
                                       results: Dict, project_id: int) -> str:
        """Generate natural language answer from SQL results"""
        try:
            # Build prompt for answer generation
            prompt = Config.PROMPTS['answer_generation'].format(
                original_query=query,
                sql_query=sql,
                results=json.dumps(results['data'][:10], indent=2),  # First 10 rows
                row_count=results['row_count']
            )
            
            # Call LLM
            answer = self._call_llm(prompt, max_tokens=500)
            
            # Clean up answer
            answer = answer.strip()
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating natural language answer: {str(e)}")
            return f"I found {results.get('row_count', 0)} results for your query, but I couldn't generate a natural language explanation."
    
    def _build_project_context(self, project_id: int) -> Dict[str, Any]:
        """Build context information for the project"""
        try:
            # Get tables
            tables = Table.query.join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            # Get columns  
            columns = Column.query.join(Table).join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            # Get dictionary terms
            dictionary_terms = DictionaryEntry.query.filter_by(
                project_id=project_id,
                status='approved'
            ).all()
            
            context = {
                'tables': [
                    {
                        'name': table.name,
                        'display_name': table.display_name,
                        'description': table.description
                    } for table in tables
                ],
                'columns': [
                    {
                        'name': column.name,
                        'table': column.table.name,
                        'data_type': column.data_type,
                        'business_category': column.business_category,
                        'display_name': column.display_name
                    } for column in columns
                ],
                'dictionary_terms': [
                    {
                        'term': term.term,
                        'definition': term.definition,
                        'synonyms': term.synonyms
                    } for term in dictionary_terms
                ]
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error building project context: {str(e)}")
            return {'tables': [], 'columns': [], 'dictionary_terms': []}
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call Azure OpenAI LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that converts natural language to SQL and helps analyze data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise
    
    def _extract_entities_fallback(self, query: str, context: Dict) -> List[Dict]:
        """Fallback entity extraction using pattern matching"""
        entities = []
        query_lower = query.lower()
        
        # Look for table names
        for table in context['tables']:
            table_name = table['name'].lower()
            if table_name in query_lower:
                entities.append({
                    'text': table['name'],
                    'type': 'table',
                    'confidence': 0.8,
                    'position': query_lower.find(table_name)
                })
        
        # Look for column names
        for column in context['columns']:
            col_name = column['name'].lower()
            if col_name in query_lower:
                entities.append({
                    'text': column['name'],
                    'type': 'column',
                    'confidence': 0.7,
                    'position': query_lower.find(col_name)
                })
        
        # Look for dictionary terms
        for term in context['dictionary_terms']:
            term_text = term['term'].lower()
            if term_text in query_lower:
                entities.append({
                    'text': term['term'],
                    'type': 'business_term',
                    'confidence': 0.6,
                    'position': query_lower.find(term_text)
                })
        
        # Remove duplicates and sort by confidence
        unique_entities = []
        seen = set()
        
        for entity in sorted(entities, key=lambda x: x['confidence'], reverse=True):
            if entity['text'] not in seen:
                unique_entities.append(entity)
                seen.add(entity['text'])
        
        return unique_entities[:self.max_entities]
    
    def _validate_entities(self, entities: List[Dict], context: Dict) -> List[Dict]:
        """Validate and enhance extracted entities"""
        validated = []
        
        if not isinstance(entities, list):
            return []
        
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            # Ensure required fields
            if 'text' not in entity:
                continue
                
            # Set defaults
            entity.setdefault('type', 'unknown')
            entity.setdefault('confidence', 0.5)
            
            # Validate confidence is a number
            try:
                entity['confidence'] = float(entity['confidence'])
            except (ValueError, TypeError):
                entity['confidence'] = 0.5
            
            # Filter by confidence threshold
            if entity['confidence'] >= self.entity_confidence_threshold:
                validated.append(entity)
        
        return validated[:self.max_entities]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        # Use fuzzy string matching
        return fuzz.ratio(text1, text2) / 100.0
    
    def _build_schema_context(self, mappings: List[Dict], project_id: int) -> Dict[str, Any]:
        """Build schema context for SQL generation"""
        try:
            schema_info = []
            table_names = set()
            
            for mapping in mappings:
                table_name = mapping['table']
                table_names.add(table_name)
                
                if mapping['column']:
                    schema_info.append(f"{table_name}.{mapping['column']} ({mapping.get('data_type', 'unknown')})")
                else:
                    schema_info.append(f"{table_name} (table)")
            
            # Get additional table information
            tables = Table.query.join(DataSource).filter(
                DataSource.project_id == project_id,
                Table.name.in_(table_names)
            ).all()
            
            relationships = []
            for table in tables:
                # Get foreign key relationships (simplified)
                for column in table.columns:
                    if column.is_foreign_key:
                        relationships.append(f"{table.name}.{column.name} references another table")
            
            return {
                'schema': '\n'.join(schema_info),
                'relationships': '\n'.join(relationships) if relationships else 'No explicit relationships found'
            }
            
        except Exception as e:
            logger.error(f"Error building schema context: {str(e)}")
            return {'schema': '', 'relationships': ''}
    
    def _validate_and_enhance_sql(self, sql_result: Dict, mappings: List[Dict]) -> Dict[str, Any]:
        """Validate and enhance SQL result"""
        try:
            sql = sql_result.get('sql', '')
            
            # Basic SQL validation
            if not sql or not sql.strip():
                sql_result['confidence'] = 0.0
                sql_result['error'] = 'No SQL generated'
                return sql_result
            
            # Check if SQL contains mapped tables/columns
            sql_lower = sql.lower()
            mapped_tables = set(mapping['table'].lower() for mapping in mappings)
            
            table_matches = sum(1 for table in mapped_tables if table in sql_lower)
            
            # Adjust confidence based on table matches
            if table_matches > 0:
                original_confidence = sql_result.get('confidence', 0.5)
                boost = min(0.3, table_matches * 0.1)
                sql_result['confidence'] = min(1.0, original_confidence + boost)
            
            # Add metadata
            sql_result['tables_used'] = list(mapped_tables)
            sql_result['table_matches'] = table_matches
            
            return sql_result
            
        except Exception as e:
            logger.error(f"Error validating SQL: {str(e)}")
            sql_result['confidence'] = 0.0
            sql_result['error'] = str(e)
            return sql_result
    
    def _is_sql_safe(self, sql: str) -> bool:
        """Check if SQL query is safe to execute"""
        sql_upper = sql.upper().strip()
        
        # Only allow SELECT statements
        if not sql_upper.startswith('SELECT'):
            return False
        
        # Forbidden keywords/operations
        forbidden = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
            'MERGE', 'BULK', 'OPENROWSET', 'OPENDATASOURCE'
        ]
        
        for keyword in forbidden:
            if keyword in sql_upper:
                return False
        
        return True
    
    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL from text response"""
        # Look for SQL patterns
        sql_patterns = [
            r'SELECT\s+.*?(?=\n\n|\n$|$)',
            r'select\s+.*?(?=\n\n|\n$|$)'
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return text.strip()
    
    def _execute_on_file_sources(self, sql: str, project_id: int) -> Dict[str, Any]:
        """Execute SQL on file-based data sources (create temporary SQLite DB)"""
        try:
            import sqlite3
            import pandas as pd
            import tempfile
            
            # Create temporary SQLite database
            temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            conn = sqlite3.connect(temp_db.name)
            
            # Load data from file sources into SQLite
            data_sources = DataSource.query.filter_by(
                project_id=project_id,
                type='file'
            ).all()
            
            for source in data_sources:
                for table in source.tables:
                    try:
                        # Read file data
                        if source.subtype == 'csv':
                            df = pd.read_csv(source.file_path)
                        elif source.subtype in ['xlsx', 'xls']:
                            df = pd.read_excel(source.file_path)
                        else:
                            continue
                        
                        # Store in SQLite
                        df.to_sql(table.name, conn, if_exists='replace', index=False)
                        
                    except Exception as e:
                        logger.warning(f"Error loading table {table.name}: {str(e)}")
                        continue
            
            # Execute SQL
            start_time = time.time()
            result_df = pd.read_sql_query(sql, conn)
            execution_time = time.time() - start_time
            
            # Convert to dict format
            data = result_df.to_dict('records')
            columns = list(result_df.columns)
            
            # Cleanup
            conn.close()
            os.unlink(temp_db.name)
            
            return {
                'success': True,
                'data': data,
                'columns': columns,
                'row_count': len(data),
                'execution_time_seconds': round(execution_time, 3),
                'sql': sql
            }
            
        except Exception as e:
            logger.error(f"Error executing SQL on file sources: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'sql': sql
            }
    
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
        return None