# backend/services/llm_service.py
"""
LLM service for entity extraction and SQL generation
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncAzureOpenAI
import time

from config import Config

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM operations using Azure OpenAI"""
    
    def __init__(self):
        self.config = Config.LLM_CONFIG
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        try:
            if self.config['provider'] == 'azure_openai':
                azure_config = self.config['azure']
                self.client = AsyncAzureOpenAI(
                    azure_endpoint=azure_config['endpoint'],
                    api_key=azure_config['api_key'],
                    api_version=azure_config['api_version']
                )
                logger.info("Azure OpenAI client initialized successfully")
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config['provider']}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    async def extract_entities(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract entities from natural language query"""
        
        try:
            # Prepare context
            tables = context.get('tables', []) if context else []
            columns = context.get('columns', []) if context else []
            dictionary_terms = context.get('dictionary_terms', []) if context else []
            
            # Format context for prompt
            tables_str = "\n".join([f"- {t}" for t in tables[:20]])
            columns_str = "\n".join([f"- {c}" for c in columns[:50]])
            dict_str = "\n".join([f"- {term}" for term in dictionary_terms[:30]])
            
            # Build prompt
            prompt = Config.PROMPTS['entity_extraction'].format(
                query=query,
                tables=tables_str,
                columns=columns_str,
                dictionary_terms=dict_str
            )
            
            # Make API call
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.config['azure']['deployment_name'],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data analyst specializing in entity extraction for database queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_time = time.time() - start_time
            content = response.choices[0].message.content
            
            try:
                entities = json.loads(content)
                if isinstance(entities, list):
                    entities = {"entities": entities}
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                entities = {"entities": []}
            
            # Add metadata
            result = {
                "entities": entities.get("entities", []),
                "query": query,
                "response_time": response_time,
                "model_used": self.config['azure']['model_name'],
                "confidence": self._calculate_overall_confidence(entities.get("entities", [])),
                "context_used": {
                    "tables_count": len(tables),
                    "columns_count": len(columns),
                    "dictionary_terms_count": len(dictionary_terms)
                }
            }
            
            logger.info(f"Extracted {len(result['entities'])} entities in {response_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                "entities": [],
                "query": query,
                "error": str(e),
                "response_time": 0,
                "model_used": self.config['azure']['model_name']
            }
    
    async def generate_sql(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        schemas: List[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate SQL from natural language query with context"""
        
        try:
            # Format schemas
            schemas_str = self._format_schemas_for_prompt(schemas)
            
            # Format entities
            entities_str = self._format_entities_for_prompt(entities)
            
            # Build prompt
            prompt = Config.PROMPTS['sql_generation'].format(
                query=query,
                schemas=schemas_str,
                entities=entities_str
            )
            
            # Make API call
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.config['azure']['deployment_name'],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL analyst. Generate accurate, executable SQL queries based on natural language requests."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_time = time.time() - start_time
            content = response.choices[0].message.content
            
            try:
                sql_response = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                sql_response = {
                    "sql": "-- Failed to generate SQL",
                    "rationale": "JSON parsing error",
                    "confidence": 0.0
                }
            
            # Validate and clean SQL
            sql_query = sql_response.get("sql", "")
            validated_sql = self._validate_sql(sql_query)
            
            result = {
                "sql": validated_sql,
                "rationale": sql_response.get("rationale", ""),
                "confidence": sql_response.get("confidence", 0.5),
                "tables_used": sql_response.get("tables_used", []),
                "assumptions": sql_response.get("assumptions", []),
                "query": query,
                "entities_used": entities,
                "response_time": response_time,
                "model_used": self.config['azure']['model_name'],
                "is_valid": self._is_sql_safe(validated_sql)
            }
            
            logger.info(f"Generated SQL in {response_time:.2f}s with confidence {result['confidence']}")
            return result
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return {
                "sql": "-- Error generating SQL",
                "rationale": f"Error: {str(e)}",
                "confidence": 0.0,
                "query": query,
                "error": str(e),
                "response_time": 0,
                "model_used": self.config['azure']['model_name'],
                "is_valid": False
            }
    
    async def refine_sql(
        self,
        original_query: str,
        generated_sql: str,
        execution_result: Dict[str, Any],
        user_feedback: str,
        schemas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Refine SQL based on user feedback"""
        
        try:
            # Format schemas
            schemas_str = self._format_schemas_for_prompt(schemas)
            
            # Format execution result
            result_str = self._format_result_for_prompt(execution_result)
            
            # Build prompt
            prompt = Config.PROMPTS['sql_refinement'].format(
                original_query=original_query,
                generated_sql=generated_sql,
                result=result_str,
                feedback=user_feedback,
                schemas=schemas_str
            )
            
            # Make API call
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.config['azure']['deployment_name'],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL analyst. Refine SQL queries based on user feedback and execution results."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_time = time.time() - start_time
            content = response.choices[0].message.content
            
            try:
                refined_response = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                refined_response = {
                    "sql": generated_sql,  # Return original if parsing fails
                    "rationale": "Failed to parse refinement response",
                    "confidence": 0.3
                }
            
            # Validate refined SQL
            refined_sql = refined_response.get("sql", generated_sql)
            validated_sql = self._validate_sql(refined_sql)
            
            result = {
                "sql": validated_sql,
                "rationale": refined_response.get("rationale", ""),
                "confidence": refined_response.get("confidence", 0.5),
                "changes": refined_response.get("changes", []),
                "original_query": original_query,
                "user_feedback": user_feedback,
                "response_time": response_time,
                "model_used": self.config['azure']['model_name'],
                "is_valid": self._is_sql_safe(validated_sql)
            }
            
            logger.info(f"Refined SQL in {response_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"SQL refinement failed: {e}")
            return {
                "sql": generated_sql,  # Return original on error
                "rationale": f"Refinement error: {str(e)}",
                "confidence": 0.2,
                "error": str(e),
                "response_time": 0,
                "model_used": self.config['azure']['model_name'],
                "is_valid": False
            }
    
    def _format_schemas_for_prompt(self, schemas: List[Dict[str, Any]]) -> str:
        """Format schemas for prompt"""
        schema_parts = []
        
        for schema in schemas:
            table_name = schema.get('table_name', 'unknown')
            columns = schema.get('columns', [])
            
            column_parts = []
            for col in columns:
                col_def = f"{col['name']} {col['data_type']}"
                if col.get('is_primary_key'):
                    col_def += " PRIMARY KEY"
                if not col.get('is_nullable', True):
                    col_def += " NOT NULL"
                if col.get('description'):
                    col_def += f" -- {col['description']}"
                column_parts.append(col_def)
            
            schema_part = f"Table: {table_name}\n"
            schema_part += "\n".join([f"  {col}" for col in column_parts])
            
            if schema.get('description'):
                schema_part += f"\nDescription: {schema['description']}"
            
            schema_parts.append(schema_part)
        
        return "\n\n".join(schema_parts)
    
    def _format_entities_for_prompt(self, entities: List[Dict[str, Any]]) -> str:
        """Format entities for prompt"""
        entity_parts = []
        
        for entity in entities:
            entity_text = entity.get('entity', '')
            entity_type = entity.get('type', '')
            confidence = entity.get('confidence', 0.0)
            
            entity_part = f"- {entity_text} (type: {entity_type}, confidence: {confidence:.2f})"
            entity_parts.append(entity_part)
        
        return "\n".join(entity_parts)
    
    def _format_result_for_prompt(self, result: Dict[str, Any]) -> str:
        """Format execution result for prompt"""
        if result.get('error'):
            return f"Error: {result['error']}"
        
        data = result.get('data', [])
        columns = result.get('columns', [])
        
        if not data:
            return "No data returned"
        
        # Format first few rows
        result_str = f"Columns: {', '.join(columns)}\n"
        result_str += f"Rows returned: {len(data)}\n"
        
        if data:
            result_str += "Sample data (first 3 rows):\n"
            for i, row in enumerate(data[:3]):
                result_str += f"Row {i+1}: {dict(zip(columns, row))}\n"
        
        return result_str
    
    def _validate_sql(self, sql: str) -> str:
        """Basic SQL validation and cleaning"""
        if not sql:
            return ""
        
        # Remove comments and extra whitespace
        lines = sql.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--'):
                cleaned_lines.append(line)
        
        cleaned_sql = ' '.join(cleaned_lines)
        
        # Ensure it ends with semicolon
        if cleaned_sql and not cleaned_sql.endswith(';'):
            cleaned_sql += ';'
        
        return cleaned_sql
    
    def _is_sql_safe(self, sql: str) -> bool:
        """Check if SQL is safe for execution"""
        if not sql:
            return False
        
        sql_upper = sql.upper()
        
        # Check for blocked keywords
        blocked_keywords = Config.SQL_CONFIG['blocked_keywords']
        for keyword in blocked_keywords:
            if keyword.upper() in sql_upper:
                logger.warning(f"Blocked keyword found in SQL: {keyword}")
                return False
        
        # Must contain allowed statements
        allowed_statements = Config.SQL_CONFIG['allowed_statements']
        has_allowed = any(stmt.upper() in sql_upper for stmt in allowed_statements)
        
        if not has_allowed:
            logger.warning("No allowed SQL statements found")
            return False
        
        return True
    
    def _calculate_overall_confidence(self, entities: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence from entity confidences"""
        if not entities:
            return 0.0
        
        confidences = [e.get('confidence', 0.0) for e in entities]
        return sum(confidences) / len(confidences)
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and complexity"""
        
        # Simple intent analysis
        query_lower = query.lower()
        
        # Determine intent
        intent = "unknown"
        if any(word in query_lower for word in ['show', 'list', 'display', 'get', 'find']):
            intent = "retrieve"
        elif any(word in query_lower for word in ['count', 'how many', 'total', 'sum']):
            intent = "aggregate"
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            intent = "compare"
        elif any(word in query_lower for word in ['trend', 'over time', 'change', 'growth']):
            intent = "trend"
        elif any(word in query_lower for word in ['top', 'best', 'highest', 'lowest', 'minimum', 'maximum']):
            intent = "ranking"
        
        # Determine complexity
        complexity = "simple"
        if len(query.split()) > 20:
            complexity = "complex"
        elif any(word in query_lower for word in ['join', 'group by', 'having', 'subquery']):
            complexity = "complex"
        elif len([w for w in query.split() if w.lower() in ['and', 'or', 'where', 'when']]) > 2:
            complexity = "medium"
        
        return {
            "intent": intent,
            "complexity": complexity,
            "word_count": len(query.split()),
            "estimated_difficulty": self._estimate_difficulty(query)
        }
    
    def _estimate_difficulty(self, query: str) -> float:
        """Estimate query difficulty (0.0 to 1.0)"""
        difficulty = 0.1  # Base difficulty
        
        query_lower = query.lower()
        
        # Add difficulty for complexity indicators
        complexity_indicators = [
            'join', 'group by', 'having', 'subquery', 'nested', 'union',
            'case when', 'exists', 'window function', 'partition by'
        ]
        
        for indicator in complexity_indicators:
            if indicator in query_lower:
                difficulty += 0.15
        
        # Add difficulty for multiple conditions
        condition_words = ['and', 'or', 'where', 'when', 'if']
        condition_count = sum(1 for word in condition_words if word in query_lower)
        difficulty += min(condition_count * 0.05, 0.3)
        
        # Add difficulty for length
        word_count = len(query.split())
        if word_count > 30:
            difficulty += 0.2
        elif word_count > 15:
            difficulty += 0.1
        
        return min(difficulty, 1.0)

# Global service instance
llm_service = LLMService()