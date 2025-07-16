# services/rag_pipeline.py
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher
import uuid

from models import db, ChatSession, ChatMessage, Table, Column, DictionaryEntry, QueryExecution
from services.embedding_service import EmbeddingService
from services.llm_client import LLMClient
from services.sql_executor import SQLExecutor
from config import Config

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline for processing natural language queries"""
    
    def __init__(self):
        self.config = Config()
        self.rag_config = self.config.RAG_CONFIG
        self.entity_config = self.config.ENTITY_CONFIG
        self.embedding_service = EmbeddingService()
        self.llm_client = LLMClient()
        self.sql_executor = SQLExecutor()
        
    def process_query(self, query: str, session_id: int = None) -> Dict[str, Any]:
        """Process a natural language query through the RAG pipeline"""
        try:
            correlation_id = str(uuid.uuid4())
            logger.info(f"Processing query [{correlation_id}]: {query[:100]}...")
            
            # Create or get chat session
            if session_id:
                session = ChatSession.query.get(session_id)
                if not session:
                    raise ValueError(f"Session {session_id} not found")
            else:
                session = ChatSession(
                    title=self._generate_session_title(query),
                    created_at=datetime.utcnow()
                )
                db.session.add(session)
                db.session.flush()
                session_id = session.id
            
            # Add user message
            user_message = ChatMessage(
                session_id=session_id,
                role='user',
                content=query,
                metadata={'correlation_id': correlation_id},
                created_at=datetime.utcnow()
            )
            db.session.add(user_message)
            db.session.flush()
            
            # Get conversation context
            context = self._get_conversation_context(session_id)
            
            # Step 1: Entity Extraction
            logger.info(f"[{correlation_id}] Step 1: Extracting entities")
            entities = self._extract_entities(query, context)
            logger.info(f"[{correlation_id}] Found {len(entities)} entities")
            
            # Step 2: Rank candidate tables and schemas
            logger.info(f"[{correlation_id}] Step 2: Ranking candidate tables")
            candidate_tables = self._rank_candidate_tables(entities, query)
            logger.info(f"[{correlation_id}] Found {len(candidate_tables)} candidate tables")
            
            # Step 3: Generate SQL
            logger.info(f"[{correlation_id}] Step 3: Generating SQL")
            sql_result = self._generate_sql(query, entities, candidate_tables, context)
            logger.info(f"[{correlation_id}] SQL generated with confidence: {sql_result.get('confidence', 0)}")
            
            # Step 4: Execute SQL if confidence is high enough
            execution_result = None
            if sql_result.get('sql') and sql_result['confidence'] >= self.rag_config['confidence_threshold']:
                logger.info(f"[{correlation_id}] Step 4: Executing SQL")
                execution_result = self._execute_sql_safely(
                    sql_result['sql'], 
                    candidate_tables[0]['source_id'] if candidate_tables else None,
                    session_id,
                    user_message.id
                )
            else:
                logger.info(f"[{correlation_id}] Skipping SQL execution - low confidence or no SQL")
            
            # Create assistant response
            response_content = self._format_response(sql_result, execution_result, entities, candidate_tables)
            assistant_message = ChatMessage(
                session_id=session_id,
                role='assistant',
                content=response_content,
                metadata={
                    'correlation_id': correlation_id,
                    'entities': entities,
                    'candidate_tables': candidate_tables,
                    'sql_result': sql_result,
                    'execution_result': execution_result,
                    'sql': sql_result.get('sql', ''),
                    'confidence': sql_result.get('confidence', 0)
                },
                created_at=datetime.utcnow()
            )
            db.session.add(assistant_message)
            
            # Update session
            session.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return {
                'session_id': session_id,
                'message_id': assistant_message.id,
                'response': response_content,
                'sql': sql_result.get('sql', ''),
                'confidence': sql_result.get('confidence', 0),
                'entities': entities,
                'candidate_tables': [t['name'] for t in candidate_tables],
                'execution_result': execution_result,
                'correlation_id': correlation_id
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing query [{correlation_id}]: {str(e)}")
            
            # Create error response
            try:
                if session_id:
                    error_message = ChatMessage(
                        session_id=session_id,
                        role='assistant',
                        content=f"I encountered an error while processing your query: {str(e)}",
                        metadata={'error': str(e), 'correlation_id': correlation_id},
                        created_at=datetime.utcnow()
                    )
                    db.session.add(error_message)
                    db.session.commit()
            except:
                pass
            
            raise
    
    def process_feedback(self, feedback: str, session_id: int, message_id: int = None) -> Dict[str, Any]:
        """Process user feedback and refine the query"""
        try:
            correlation_id = str(uuid.uuid4())
            logger.info(f"Processing feedback [{correlation_id}]: {feedback[:100]}...")
            
            # Get session and context
            session = ChatSession.query.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
                
            context = self._get_conversation_context(session_id, include_current=True)
            
            # Get the last assistant message if message_id not provided
            if not message_id:
                last_message = ChatMessage.query.filter_by(
                    session_id=session_id, 
                    role='assistant'
                ).order_by(ChatMessage.created_at.desc()).first()
                if last_message:
                    message_id = last_message.id
            
            # Get original query context
            original_message = ChatMessage.query.get(message_id) if message_id else None
            original_metadata = original_message.metadata if original_message else {}
            
            # Add feedback message
            feedback_message = ChatMessage(
                session_id=session_id,
                role='user',
                content=feedback,
                metadata={
                    'correlation_id': correlation_id,
                    'feedback_for': message_id,
                    'type': 'feedback'
                },
                parent_message_id=message_id,
                created_at=datetime.utcnow()
            )
            db.session.add(feedback_message)
            db.session.flush()
            
            # Re-extract entities with feedback context
            original_query = self._get_original_query_from_context(context)
            entities = self._extract_entities_with_feedback(original_query, feedback, context)
            
            # Re-rank tables with refined entities
            candidate_tables = self._rank_candidate_tables(entities, original_query)
            
            # Generate refined SQL
            sql_result = self._refine_sql_with_feedback(
                original_query, 
                feedback, 
                original_metadata.get('sql_result', {}),
                original_metadata.get('execution_result'),
                entities,
                candidate_tables,
                context
            )
            
            # Execute refined SQL
            execution_result = None
            if sql_result.get('sql') and sql_result['confidence'] >= self.rag_config['confidence_threshold']:
                execution_result = self._execute_sql_safely(
                    sql_result['sql'],
                    candidate_tables[0]['source_id'] if candidate_tables else None,
                    session_id,
                    feedback_message.id
                )
            
            # Create refined response
            response_content = self._format_refined_response(sql_result, execution_result, feedback)
            refined_message = ChatMessage(
                session_id=session_id,
                role='assistant',
                content=response_content,
                metadata={
                    'correlation_id': correlation_id,
                    'entities': entities,
                    'candidate_tables': candidate_tables,
                    'sql_result': sql_result,
                    'execution_result': execution_result,
                    'refined_from': message_id,
                    'feedback': feedback,
                    'sql': sql_result.get('sql', ''),
                    'confidence': sql_result.get('confidence', 0)
                },
                parent_message_id=feedback_message.id,
                created_at=datetime.utcnow()
            )
            db.session.add(refined_message)
            
            # Update session
            session.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return {
                'session_id': session_id,
                'message_id': refined_message.id,
                'response': response_content,
                'sql': sql_result.get('sql', ''),
                'confidence': sql_result.get('confidence', 0),
                'entities': entities,
                'candidate_tables': [t['name'] for t in candidate_tables],
                'execution_result': execution_result,
                'correlation_id': correlation_id,
                'changes_made': sql_result.get('changes', [])
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing feedback [{correlation_id}]: {str(e)}")
            raise
    
    def _extract_entities(self, query: str, context: List[Dict] = None) -> List[Dict[str, Any]]:
        """Extract entities from query using LLM and similarity matching"""
        try:
            # Get available context for entity extraction
            tables = self._get_available_tables()
            columns = self._get_available_columns()
            dictionary_terms = self._get_dictionary_terms()
            
            logger.info(f"Available tables: {len(tables)}, columns: {len(columns)}, terms: {len(dictionary_terms)}")
            
            # Use LLM for entity extraction
            llm_entities = self._llm_extract_entities(query, tables, columns, dictionary_terms, context)
            
            # Use similarity matching for additional entities
            similarity_entities = self._similarity_extract_entities(query, tables, columns, dictionary_terms)
            
            # Combine and rank entities
            all_entities = self._combine_and_rank_entities(llm_entities, similarity_entities)
            
            # Limit to max entities
            max_entities = self.entity_config['max_entities']
            return all_entities[:max_entities]
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _extract_entities_with_feedback(self, original_query: str, feedback: str, 
                                       context: List[Dict] = None) -> List[Dict[str, Any]]:
        """Extract entities considering user feedback"""
        try:
            # Combine original query with feedback for better entity extraction
            combined_query = f"{original_query} {feedback}"
            
            # Extract entities from combined context
            entities = self._extract_entities(combined_query, context)
            
            # Boost entities mentioned in feedback
            feedback_lower = feedback.lower()
            for entity in entities:
                if entity['entity'].lower() in feedback_lower:
                    entity['confidence'] = min(1.0, entity['confidence'] + 0.2)
                    entity['feedback_boost'] = True
            
            # Re-sort by confidence
            entities.sort(key=lambda x: x['confidence'], reverse=True)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities with feedback: {str(e)}")
            return []
    
    def _llm_extract_entities(self, query: str, tables: List[str], columns: List[str], 
                             dictionary_terms: List[str], context: List[Dict] = None) -> List[Dict[str, Any]]:
        """Use LLM to extract entities from query"""
        try:
            prompt = self.config.PROMPTS['entity_extraction'].format(
                query=query,
                tables=', '.join(tables[:20]),  # Limit context size
                columns=', '.join(columns[:50]),
                dictionary_terms=', '.join(dictionary_terms[:30])
            )
            
            response = self.llm_client.generate(prompt, max_tokens=500)
            
            # Parse JSON response
            try:
                entities = json.loads(response)
                if isinstance(entities, list):
                    # Validate entity format
                    validated_entities = []
                    for entity in entities:
                        if isinstance(entity, dict) and 'entity' in entity:
                            entity.setdefault('type', 'unknown')
                            entity.setdefault('confidence', 0.5)
                            entity['method'] = 'llm'
                            validated_entities.append(entity)
                    return validated_entities
            except json.JSONDecodeError:
                logger.warning("LLM response not valid JSON, falling back to regex extraction")
            
            # Fallback: extract entities using regex
            return self._regex_extract_entities(response)
            
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {str(e)}")
            return []
    
    def _similarity_extract_entities(self, query: str, tables: List[str], 
                                   columns: List[str], dictionary_terms: List[str]) -> List[Dict[str, Any]]:
        """Extract entities using fuzzy string matching"""
        entities = []
        query_lower = query.lower()
        threshold = self.entity_config['fuzzy_threshold']
        
        # Check tables
        for table in tables:
            ratio = SequenceMatcher(None, query_lower, table.lower()).ratio()
            if ratio >= threshold or table.lower() in query_lower:
                entities.append({
                    'entity': table,
                    'type': 'table',
                    'confidence': round(ratio, 3),
                    'method': 'similarity'
                })
        
        # Check columns
        for column in columns:
            ratio = SequenceMatcher(None, query_lower, column.lower()).ratio()
            if ratio >= threshold or column.lower() in query_lower:
                entities.append({
                    'entity': column,
                    'type': 'column',
                    'confidence': round(ratio, 3),
                    'method': 'similarity'
                })
        
        # Check dictionary terms
        for term in dictionary_terms:
            ratio = SequenceMatcher(None, query_lower, term.lower()).ratio()
            if ratio >= threshold or term.lower() in query_lower:
                entities.append({
                    'entity': term,
                    'type': 'term',
                    'confidence': round(ratio, 3),
                    'method': 'similarity'
                })
        
        return entities
    
    def _regex_extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns as fallback"""
        entities = []
        
        # Look for JSON-like patterns
        patterns = [
            r'"entity":\s*"([^"]+)".*?"type":\s*"([^"]+)".*?"confidence":\s*([0-9.]+)',
            r'entity:\s*([^\s,]+).*?type:\s*([^\s,]+).*?confidence:\s*([0-9.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    entities.append({
                        'entity': match[0],
                        'type': match[1],
                        'confidence': round(float(match[2]), 3),
                        'method': 'regex'
                    })
                except (ValueError, IndexError):
                    continue
        
        return entities
    
    def _combine_and_rank_entities(self, llm_entities: List[Dict], 
                                  similarity_entities: List[Dict]) -> List[Dict[str, Any]]:
        """Combine entities from different methods and rank them"""
        entity_map = {}
        
        # Process LLM entities
        for entity in llm_entities:
            key = (entity['entity'].lower(), entity['type'])
            if key not in entity_map:
                entity_map[key] = entity.copy()
                entity_map[key]['sources'] = ['llm']
            else:
                # Combine confidence scores
                entity_map[key]['confidence'] = max(entity_map[key]['confidence'], entity['confidence'])
                entity_map[key]['sources'].append('llm')
        
        # Process similarity entities
        for entity in similarity_entities:
            key = (entity['entity'].lower(), entity['type'])
            if key not in entity_map:
                entity_map[key] = entity.copy()
                entity_map[key]['sources'] = ['similarity']
            else:
                # Boost confidence if found by multiple methods
                entity_map[key]['confidence'] = min(1.0, entity_map[key]['confidence'] + 0.1)
                if 'similarity' not in entity_map[key]['sources']:
                    entity_map[key]['sources'].append('similarity')
        
        # Convert back to list and sort by confidence
        ranked_entities = list(entity_map.values())
        ranked_entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return ranked_entities
    
    def _rank_candidate_tables(self, entities: List[Dict], query: str) -> List[Dict[str, Any]]:
        """Rank candidate tables based on entities and relevance"""
        try:
            table_scores = {}
            
            # Get all tables with their metadata
            tables = Table.query.all()
            
            for table in tables:
                score = 0.0
                reasons = []
                
                # Check if table name matches entities
                for entity in entities:
                    if entity['type'] == 'table' and entity['entity'].lower() == table.name.lower():
                        score += entity['confidence'] * self.entity_config['ranking_weights']['exact_match']
                        reasons.append(f"Exact table match: {entity['entity']}")
                    elif entity['entity'].lower() in table.name.lower():
                        score += entity['confidence'] * self.entity_config['ranking_weights']['fuzzy_match']
                        reasons.append(f"Partial table match: {entity['entity']}")
                
                # Check if table columns match entities
                for column in table.columns:
                    for entity in entities:
                        if entity['type'] == 'column' and entity['entity'].lower() == column.name.lower():
                            score += entity['confidence'] * self.entity_config['ranking_weights']['exact_match'] * 0.8
                            reasons.append(f"Exact column match: {entity['entity']}")
                        elif entity['entity'].lower() in column.name.lower():
                            score += entity['confidence'] * self.entity_config['ranking_weights']['fuzzy_match'] * 0.8
                            reasons.append(f"Partial column match: {entity['entity']}")
                
                # Boost score based on table importance (row count, recent usage, etc.)
                importance_boost = min(0.5, (table.row_count or 0) / 10000)  # Normalize by 10k rows
                score += importance_boost * self.entity_config['ranking_weights']['table_importance']
                
                if score > 0:
                    table_scores[table.id] = {
                        'table_id': table.id,
                        'name': table.name,
                        'display_name': table.display_name,
                        'source_id': table.source_id,
                        'score': round(score, 3),
                        'reasons': reasons,
                        'columns': [col.to_dict() for col in table.columns],
                        'row_count': table.row_count
                    }
            
            # Sort by score and return top candidates
            candidates = sorted(table_scores.values(), key=lambda x: x['score'], reverse=True)
            return candidates[:self.rag_config['max_candidate_tables']]
            
        except Exception as e:
            logger.error(f"Error ranking candidate tables: {str(e)}")
            return []
    
    def _generate_sql(self, query: str, entities: List[Dict], candidate_tables: List[Dict], 
                     context: List[Dict] = None) -> Dict[str, Any]:
        """Generate SQL query using LLM"""
        try:
            if not candidate_tables:
                return {
                    'sql': '',
                    'rationale': 'No relevant tables found for the query',
                    'confidence': 0.0,
                    'tables_used': [],
                    'assumptions': [],
                    'error': 'No candidate tables available'
                }
            
            # Prepare schemas for top candidate tables
            schemas = self._format_schemas_for_llm(candidate_tables)
            
            # Format entities for context
            entity_context = self._format_entities_for_llm(entities)
            
            prompt = self.config.PROMPTS['sql_generation'].format(
                query=query,
                schemas=schemas,
                entities=entity_context
            )
            
            response = self.llm_client.generate(prompt, max_tokens=1000)
            
            # Parse JSON response
            try:
                sql_result = json.loads(response)
                
                # Validate required fields
                if 'sql' not in sql_result:
                    raise ValueError("No SQL in response")
                
                # Set defaults and ensure proper formatting
                sql_result.setdefault('confidence', 0.5)
                sql_result.setdefault('rationale', 'Generated SQL query')
                sql_result.setdefault('tables_used', [])
                sql_result.setdefault('assumptions', [])
                
                # Round confidence to 3 decimal places
                sql_result['confidence'] = round(sql_result['confidence'], 3)
                
                return sql_result
                
            except json.JSONDecodeError:
                # Try to extract SQL from response
                sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    return {
                        'sql': sql_match.group(1).strip(),
                        'rationale': 'Extracted SQL from LLM response',
                        'confidence': 0.7,
                        'tables_used': [t['name'] for t in candidate_tables[:2]],
                        'assumptions': ['SQL extracted from text response']
                    }
                
                raise ValueError("Could not parse LLM response")
                
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            return {
                'sql': '',
                'rationale': f'Error generating SQL: {str(e)}',
                'confidence': 0.0,
                'tables_used': [],
                'assumptions': [],
                'error': str(e)
            }
    
    def _refine_sql_with_feedback(self, original_query: str, feedback: str, 
                                 original_sql_result: Dict, execution_result: Dict,
                                 entities: List[Dict], candidate_tables: List[Dict],
                                 context: List[Dict]) -> Dict[str, Any]:
        """Refine SQL based on user feedback"""
        try:
            # Prepare schemas
            schemas = self._format_schemas_for_llm(candidate_tables)
            
            prompt = self.config.PROMPTS['sql_refinement'].format(
                original_query=original_query,
                generated_sql=original_sql_result.get('sql', ''),
                result=json.dumps(execution_result, indent=2) if execution_result else 'No execution result',
                feedback=feedback,
                schemas=schemas
            )
            
            response = self.llm_client.generate(prompt, max_tokens=1000)
            
            # Parse response
            try:
                refined_result = json.loads(response)
                refined_result.setdefault('confidence', 0.7)
                refined_result.setdefault('changes', [])
                refined_result['confidence'] = round(refined_result['confidence'], 3)
                return refined_result
                
            except json.JSONDecodeError:
                # Fallback extraction
                sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    return {
                        'sql': sql_match.group(1).strip(),
                        'rationale': f'Refined SQL based on feedback: {feedback}',
                        'confidence': 0.7,
                        'changes': ['Modified based on user feedback']
                    }
                
                # Return original if refinement fails
                return original_sql_result
                
        except Exception as e:
            logger.error(f"Error refining SQL: {str(e)}")
            return original_sql_result
    
    def _execute_sql_safely(self, sql: str, source_id: int = None, 
                           session_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """Execute SQL safely with validation and logging"""
        try:
            return self.sql_executor.execute_query(sql, source_id, session_id, message_id)
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'sql': sql,
                'execution_time': 0.0
            }
    
    def _get_conversation_context(self, session_id: int, include_current: bool = False, 
                                 limit: int = 10) -> List[Dict]:
        """Get conversation context for the session"""
        try:
            query = ChatMessage.query.filter_by(session_id=session_id)
            if not include_current:
                query = query.filter(ChatMessage.role != 'user')
            
            messages = query.order_by(ChatMessage.created_at.desc()).limit(limit).all()
            
            context = []
            for msg in reversed(messages):  # Reverse to get chronological order
                context.append({
                    'role': msg.role,
                    'content': msg.content,
                    'metadata': msg.metadata or {},
                    'created_at': msg.created_at.isoformat()
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return []
    
    def _get_original_query_from_context(self, context: List[Dict]) -> str:
        """Extract the original user query from context"""
        for msg in reversed(context):
            if msg['role'] == 'user' and msg['metadata'].get('type') != 'feedback':
                return msg['content']
        return ""
    
    def _get_available_tables(self) -> List[str]:
        """Get list of available table names"""
        try:
            tables = db.session.query(Table.name).all()
            return [table.name for table in tables]
        except Exception as e:
            logger.error(f"Error getting available tables: {str(e)}")
            return []
    
    def _get_available_columns(self) -> List[str]:
        """Get list of available column names"""
        try:
            columns = db.session.query(Column.name).distinct().all()
            return [column.name for column in columns]
        except Exception as e:
            logger.error(f"Error getting available columns: {str(e)}")
            return []
    
    def _get_dictionary_terms(self) -> List[str]:
        """Get list of dictionary terms"""
        try:
            terms = db.session.query(DictionaryEntry.term).filter(
                DictionaryEntry.approved == True
            ).all()
            return [term.term for term in terms]
        except Exception as e:
            logger.error(f"Error getting dictionary terms: {str(e)}")
            return []
    
    def _format_schemas_for_llm(self, candidate_tables: List[Dict]) -> str:
        """Format table schemas for LLM context"""
        schemas = []
        
        for table in candidate_tables:
            schema_parts = [f"Table: {table['name']}"]
            if table.get('display_name'):
                schema_parts.append(f"Description: {table['display_name']}")
            
            schema_parts.append("Columns:")
            for column in table.get('columns', []):
                col_info = f"  - {column['name']} ({column.get('data_type', 'unknown')})"
                if column.get('description'):
                    col_info += f": {column['description']}"
                schema_parts.append(col_info)
            
            schemas.append('\n'.join(schema_parts))
        
        return '\n\n'.join(schemas)
    
    def _format_entities_for_llm(self, entities: List[Dict]) -> str:
        """Format entities for LLM context"""
        if not entities:
            return "No entities found"
        
        entity_groups = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(f"{entity['entity']} (confidence: {entity['confidence']:.2f})")
        
        formatted = []
        for entity_type, entity_list in entity_groups.items():
            formatted.append(f"{entity_type.title()}s: {', '.join(entity_list)}")
        
        return '\n'.join(formatted)
    
    def _format_response(self, sql_result: Dict, execution_result: Dict = None, 
                        entities: List[Dict] = None, candidate_tables: List[Dict] = None) -> str:
        """Format the assistant response"""
        response_parts = []
        
        # Show entity extraction results
        if entities:
            response_parts.append(f"**Entities Found:** {', '.join([e['entity'] for e in entities[:5]])}")
        
        # Show candidate tables
        if candidate_tables:
            response_parts.append(f"**Relevant Tables:** {', '.join([t['name'] for t in candidate_tables[:3]])}")
        
        if sql_result.get('sql'):
            response_parts.append(f"**SQL Query:**\n```sql\n{sql_result['sql']}\n```")
            
            if sql_result.get('rationale'):
                response_parts.append(f"**Explanation:** {sql_result['rationale']}")
            
            if execution_result:
                if execution_result.get('status') == 'success':
                    row_count = execution_result.get('row_count', 0)
                    response_parts.append(f"**Result:** Query executed successfully, returned {row_count} rows.")
                    
                    if execution_result.get('preview_data'):
                        response_parts.append("**Preview:**")
                        # Format preview data as table (simplified)
                        preview = execution_result['preview_data'][:5]  # Show first 5 rows
                        if preview:
                            headers = list(preview[0].keys())
                            response_parts.append(f"| {' | '.join(headers)} |")
                            response_parts.append(f"|{' --- |' * len(headers)}")
                            for row in preview:
                                values = [str(row.get(h, '')) for h in headers]
                                response_parts.append(f"| {' | '.join(values)} |")
                else:
                    response_parts.append(f"**Error:** {execution_result.get('error', 'Unknown error')}")
            
            confidence = sql_result.get('confidence', 0.0)
            if confidence < self.rag_config['confidence_threshold']:
                response_parts.append(f"\n*Note: I'm {confidence:.0%} confident about this query. Please provide feedback if it doesn't look right.*")
        else:
            response_parts.append("I couldn't generate a SQL query for your request. Could you please rephrase or provide more details?")
            
            if sql_result.get('error'):
                response_parts.append(f"*Error: {sql_result['error']}*")
        
        return '\n\n'.join(response_parts)
    
    def _format_refined_response(self, sql_result: Dict, execution_result: Dict = None, 
                               feedback: str = None) -> str:
        """Format the refined assistant response"""
        response_parts = []
        
        if feedback:
            response_parts.append(f"Based on your feedback: \"{feedback}\"")
        
        response_parts.append("Here's the refined query:")
        
        main_response = self._format_response(sql_result, execution_result)
        response_parts.append(main_response)
        
        if sql_result.get('changes'):
            response_parts.append(f"**Changes made:** {', '.join(sql_result['changes'])}")
        
        return '\n\n'.join(response_parts)
    
    def _generate_session_title(self, query: str) -> str:
        """Generate a session title from the first query"""
        # Take first 50 characters and clean up
        title = query[:50].strip()
        if len(query) > 50:
            title += "..."
        
        # Remove newlines and excessive whitespace
        title = ' '.join(title.split())
        
        return title or "New Chat"
