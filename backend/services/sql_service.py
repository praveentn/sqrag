# backend/services/sql_service.py
"""
SQL execution service with security and safety checks
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, create_engine
from sqlalchemy.exc import SQLAlchemyError
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Token

from backend.models import Source, Table, Column, Project
from config import Config

logger = logging.getLogger(__name__)

class SQLService:
    """Service for safe SQL execution"""
    
    def __init__(self):
        self.config = Config.SQL_CONFIG
        self.max_execution_time = self.config['max_execution_time']
        self.max_rows = self.config['max_rows']
        self.allowed_statements = self.config['allowed_statements']
        self.blocked_keywords = self.config['blocked_keywords']
        self.preview_rows = self.config['preview_rows']
    
    async def execute_query(
        self,
        sql_query: str,
        project_id: int,
        db: AsyncSession,
        limit_rows: bool = True
    ) -> Dict[str, Any]:
        """Execute SQL query with safety checks"""
        
        start_time = time.time()
        
        try:
            # Validate SQL safety
            validation_result = self.validate_sql_safety(sql_query)
            if not validation_result['is_safe']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'execution_time': time.time() - start_time
                }
            
            # Parse and clean SQL
            cleaned_sql = self.clean_sql(sql_query)
            
            # Add row limit if requested
            if limit_rows:
                cleaned_sql = self.add_row_limit(cleaned_sql, self.max_rows)
            
            # Get data source for the project
            source = await self._get_project_source(project_id, db)
            if not source:
                return {
                    'success': False,
                    'error': 'No data source found for project',
                    'execution_time': time.time() - start_time
                }
            
            # Execute query based on source type
            if source.type.value == 'database':
                result = await self._execute_database_query(cleaned_sql, source, db)
            else:
                result = await self._execute_file_query(cleaned_sql, source, db)
            
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            
            logger.info(f"SQL executed successfully in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"SQL execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
    
    def validate_sql_safety(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query for safety"""
        
        if not sql_query or not sql_query.strip():
            return {'is_safe': False, 'error': 'Empty SQL query'}
        
        sql_upper = sql_query.upper()
        
        # Check for blocked keywords
        for keyword in self.blocked_keywords:
            if keyword.upper() in sql_upper:
                return {
                    'is_safe': False,
                    'error': f'Blocked keyword detected: {keyword}'
                }
        
        # Check for allowed statements
        has_allowed = any(
            stmt.upper() in sql_upper 
            for stmt in self.allowed_statements
        )
        
        if not has_allowed:
            return {
                'is_safe': False,
                'error': f'Query must contain one of: {", ".join(self.allowed_statements)}'
            }
        
        # Parse SQL to check structure
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return {'is_safe': False, 'error': 'Invalid SQL syntax'}
            
            # Check each statement
            for statement in parsed:
                if not self._validate_statement(statement):
                    return {
                        'is_safe': False,
                        'error': 'Statement contains unsafe operations'
                    }
        
        except Exception as e:
            return {'is_safe': False, 'error': f'SQL parsing error: {str(e)}'}
        
        return {'is_safe': True, 'error': None}
    
    def _validate_statement(self, statement: Statement) -> bool:
        """Validate individual SQL statement"""
        
        # Check token types for dangerous operations
        for token in statement.flatten():
            if token.ttype in [Token.Keyword.DDL, Token.Keyword.DML]:
                token_value = token.value.upper()
                if token_value in self.blocked_keywords:
                    return False
        
        return True
    
    def clean_sql(self, sql_query: str) -> str:
        """Clean and normalize SQL query"""
        
        # Remove comments
        sql_query = sqlparse.format(
            sql_query,
            strip_comments=True,
            reindent=True
        )
        
        # Remove extra whitespace
        sql_query = ' '.join(sql_query.split())
        
        # Ensure semicolon at end
        if not sql_query.strip().endswith(';'):
            sql_query = sql_query.strip() + ';'
        
        return sql_query
    
    def add_row_limit(self, sql_query: str, limit: int) -> str:
        """Add LIMIT clause to SQL query if not present"""
        
        sql_upper = sql_query.upper()
        
        # Check if LIMIT already exists
        if 'LIMIT' in sql_upper:
            return sql_query
        
        # Remove trailing semicolon
        sql_query = sql_query.rstrip(';')
        
        # Add LIMIT clause
        sql_query += f' LIMIT {limit};'
        
        return sql_query
    
    async def _get_project_source(
        self,
        project_id: int,
        db: AsyncSession
    ) -> Optional[Source]:
        """Get primary data source for project"""
        
        # Get the first available source for the project
        query = select(Source).where(
            Source.project_id == project_id,
            Source.status.in_(['ready', 'connected']),
            Source.is_deleted == False
        ).limit(1)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def _execute_database_query(
        self,
        sql_query: str,
        source: Source,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute query against database source"""
        
        try:
            # Create engine for the specific database
            if not source.connection_uri:
                raise ValueError("No connection URI available for database source")
            
            # Create synchronous engine for SQL execution
            from sqlalchemy import create_engine
            engine = create_engine(source.connection_uri)
            
            # Execute query with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Query execution timeout")
            
            # Set timeout alarm (Unix only)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.max_execution_time)
            except AttributeError:
                # Windows doesn't support signal.alarm
                pass
            
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql_query))
                    
                    # Fetch results
                    if result.returns_rows:
                        columns = list(result.keys())
                        rows = result.fetchall()
                        
                        # Convert rows to list of lists for JSON serialization
                        data = [list(row) for row in rows]
                        
                        return {
                            'success': True,
                            'columns': columns,
                            'data': data,
                            'row_count': len(data),
                            'has_more': len(data) >= self.max_rows,
                            'query': sql_query
                        }
                    else:
                        return {
                            'success': True,
                            'message': 'Query executed successfully (no results)',
                            'query': sql_query
                        }
            
            finally:
                try:
                    signal.alarm(0)  # Cancel timeout
                except AttributeError:
                    pass
                engine.dispose()
        
        except TimeoutError:
            return {
                'success': False,
                'error': f'Query timeout after {self.max_execution_time} seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Database query failed: {str(e)}'
            }
    
    async def _execute_file_query(
        self,
        sql_query: str,
        source: Source,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute query against file-based source using DuckDB"""
        
        try:
            import duckdb
            import pandas as pd
            from pathlib import Path
            
            # Create DuckDB connection
            conn = duckdb.connect(':memory:')
            
            # Load file data into DuckDB
            if source.file_path and Path(source.file_path).exists():
                file_path = source.file_path
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension == '.csv':
                    # Load CSV
                    conn.execute(f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{file_path}')")
                elif file_extension in ['.xlsx', '.xls']:
                    # Load Excel - need to handle multiple sheets
                    tables = await self._get_source_tables(source.id, db)
                    for table in tables:
                        sheet_name = table.name.replace(f"{Path(file_path).stem}_", "")
                        conn.execute(f"""
                            CREATE TABLE {table.name} AS 
                            SELECT * FROM read_excel('{file_path}', sheet_name='{sheet_name}')
                        """)
                elif file_extension == '.json':
                    # Load JSON
                    conn.execute(f"CREATE TABLE data AS SELECT * FROM read_json_auto('{file_path}')")
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Execute the user's query
            result = conn.execute(sql_query).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            # Convert to list of lists
            data = [list(row) for row in result]
            
            conn.close()
            
            return {
                'success': True,
                'columns': columns,
                'data': data,
                'row_count': len(data),
                'has_more': len(data) >= self.max_rows,
                'query': sql_query
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'File query failed: {str(e)}'
            }
    
    async def _get_source_tables(
        self,
        source_id: int,
        db: AsyncSession
    ) -> List[Table]:
        """Get tables for a source"""
        
        query = select(Table).where(Table.source_id == source_id)
        result = await db.execute(query)
        return result.scalars().all()
    
    async def preview_query_results(
        self,
        sql_query: str,
        project_id: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Preview query results with limited rows"""
        
        # Add small limit for preview
        preview_sql = self.add_row_limit(sql_query, self.preview_rows)
        
        result = await self.execute_query(
            preview_sql,
            project_id,
            db,
            limit_rows=False  # We already added the limit
        )
        
        if result.get('success'):
            result['is_preview'] = True
            result['preview_limit'] = self.preview_rows
        
        return result
    
    async def explain_query(
        self,
        sql_query: str,
        project_id: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get query execution plan"""
        
        try:
            # Add EXPLAIN to the query
            explain_sql = f"EXPLAIN {sql_query}"
            
            result = await self.execute_query(
                explain_sql,
                project_id,
                db,
                limit_rows=False
            )
            
            if result.get('success'):
                result['is_explain'] = True
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Query explanation failed: {str(e)}'
            }
    
    def extract_table_names(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query"""
        
        table_names = []
        
        try:
            parsed = sqlparse.parse(sql_query)
            
            for statement in parsed:
                for token in statement.flatten():
                    if token.ttype is None and token.value.upper() not in ['SELECT', 'FROM', 'WHERE', 'AND', 'OR']:
                        # This is a potential table name
                        table_name = token.value.strip('`"[]')
                        if table_name and not table_name.upper() in ['AS', 'ON', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER']:
                            table_names.append(table_name)
            
        except Exception as e:
            logger.error(f"Table name extraction failed: {e}")
        
        return list(set(table_names))  # Remove duplicates
    
    def format_sql(self, sql_query: str) -> str:
        """Format SQL query for better readability"""
        
        return sqlparse.format(
            sql_query,
            reindent=True,
            keyword_case='upper',
            identifier_case='lower',
            strip_comments=False
        )
    
    async def validate_table_access(
        self,
        table_names: List[str],
        project_id: int,
        db: AsyncSession
    ) -> Dict[str, bool]:
        """Validate that tables exist and are accessible"""
        
        access_map = {}
        
        # Get all tables for the project
        tables_query = select(Table.name).join(Source).where(Source.project_id == project_id)
        tables_result = await db.execute(tables_query)
        available_tables = [row[0] for row in tables_result.fetchall()]
        
        for table_name in table_names:
            access_map[table_name] = table_name in available_tables
        
        return access_map

# Global service instance
sql_service = SQLService()