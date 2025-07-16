# services/sql_executor.py
import logging
import time
import re
import sqlparse
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

from models import db, DataSource, QueryExecution
from config import Config

logger = logging.getLogger(__name__)

class SQLExecutor:
    """Safe SQL execution with validation and monitoring"""
    
    def __init__(self):
        self.config = Config()
        self.sql_config = self.config.SQL_CONFIG
        self.engines = {}  # Cache for database engines
        
    def execute_query(self, sql: str, source_id: int = None, session_id: int = None, 
                     message_id: int = None) -> Dict[str, Any]:
        """Execute SQL query safely with validation and logging"""
        try:
            start_time = time.time()
            
            # Validate SQL
            validation_result = self._validate_sql(sql)
            if not validation_result['valid']:
                return {
                    'status': 'error',
                    'error': validation_result['error'],
                    'sql': sql,
                    'execution_time': 0.0
                }
            
            # Get data source
            if source_id:
                source = DataSource.query.get(source_id)
                if not source:
                    return {
                        'status': 'error',
                        'error': f'Data source {source_id} not found',
                        'sql': sql,
                        'execution_time': 0.0
                    }
            else:
                # Use default source (first available)
                source = DataSource.query.first()
                if not source:
                    return {
                        'status': 'error',
                        'error': 'No data sources available',
                        'sql': sql,
                        'execution_time': 0.0
                    }
            
            # Execute query
            result = self._execute_on_source(sql, source)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log execution
            self._log_execution(
                sql=sql,
                source_id=source.id,
                session_id=session_id,
                message_id=message_id,
                status=result['status'],
                execution_time=execution_time,
                row_count=result.get('row_count', 0),
                error_message=result.get('error'),
                result_preview=result.get('preview_data')
            )
            
            result['execution_time'] = round(execution_time, 3)
            result['sql'] = sql
            result['source_name'] = source.name
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'sql': sql,
                'execution_time': 0.0
            }
    
    def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate SQL query for safety"""
        try:
            # Clean and normalize SQL
            sql_clean = sql.strip()
            if not sql_clean:
                return {'valid': False, 'error': 'Empty SQL query'}
            
            # Parse SQL
            try:
                parsed = sqlparse.parse(sql_clean)
                if not parsed:
                    return {'valid': False, 'error': 'Invalid SQL syntax'}
            except Exception as e:
                return {'valid': False, 'error': f'SQL parsing error: {str(e)}'}
            
            # Check allowed statements
            first_token = parsed[0].tokens[0] if parsed[0].tokens else None
            if first_token:
                statement_type = first_token.value.upper()
                if statement_type not in self.sql_config['allowed_statements']:
                    return {'valid': False, 'error': f'Statement type {statement_type} not allowed'}
            
            # Check for blocked keywords
            sql_upper = sql_clean.upper()
            for blocked_keyword in self.sql_config['blocked_keywords']:
                if blocked_keyword in sql_upper:
                    return {'valid': False, 'error': f'Keyword {blocked_keyword} not allowed'}
            
            # Check for dangerous patterns
            dangerous_patterns = [
                r';\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)',
                r'--.*?(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)',
                r'/\*.*?(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE).*?\*/',
                r'EXEC\s*\(',
                r'EXECUTE\s*\(',
                r'xp_cmdshell',
                r'sp_executesql'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, sql_upper, re.IGNORECASE | re.DOTALL):
                    return {'valid': False, 'error': 'Potentially dangerous SQL pattern detected'}
            
            # Check query length
            if len(sql_clean) > 10000:  # 10KB limit
                return {'valid': False, 'error': 'SQL query too long'}
            
            return {'valid': True, 'sql': sql_clean}
            
        except Exception as e:
            logger.error(f"SQL validation error: {str(e)}")
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def _execute_on_source(self, sql: str, source: DataSource) -> Dict[str, Any]:
        """Execute SQL on specific data source"""
        try:
            if source.type in ['csv', 'excel']:
                return self._execute_on_file(sql, source)
            else:
                return self._execute_on_database(sql, source)
                
        except Exception as e:
            logger.error(f"Error executing on source {source.name}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_on_file(self, sql: str, source: DataSource) -> Dict[str, Any]:
        """Execute SQL on file-based data source using pandas"""
        try:
            # Load data into pandas DataFrames
            dfs = {}
            
            if source.type == 'csv':
                table_name = source.tables[0].name if source.tables else 'data'
                dfs[table_name] = pd.read_csv(source.file_path)
            
            elif source.type == 'excel':
                xl_file = pd.ExcelFile(source.file_path)
                for sheet_name in xl_file.sheet_names:
                    dfs[sheet_name] = pd.read_excel(source.file_path, sheet_name=sheet_name)
            
            # Create temporary SQLite database in memory
            engine = create_engine('sqlite:///:memory:')
            
            # Load DataFrames into SQLite
            for table_name, df in dfs.items():
                df.to_sql(table_name, engine, if_exists='replace', index=False)
            
            # Execute query
            return self._execute_with_engine(sql, engine)
            
        except Exception as e:
            logger.error(f"Error executing on file {source.file_path}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_on_database(self, sql: str, source: DataSource) -> Dict[str, Any]:
        """Execute SQL on database source"""
        try:
            # Get or create engine
            engine = self._get_engine(source)
            
            # Execute query
            return self._execute_with_engine(sql, engine)
            
        except Exception as e:
            logger.error(f"Error executing on database {source.name}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_with_engine(self, sql: str, engine) -> Dict[str, Any]:
        """Execute SQL with SQLAlchemy engine"""
        try:
            with engine.connect() as conn:
                # Set timeout
                if hasattr(conn, 'execute'):
                    # Add timeout hint for the query
                    sql_with_timeout = sql
                    if 'LIMIT' not in sql.upper():
                        sql_with_timeout += f" LIMIT {self.sql_config['max_rows']}"
                
                # Execute query
                result = conn.execute(text(sql_with_timeout))
                
                # Fetch results
                rows = result.fetchall()
                columns = list(result.keys()) if hasattr(result, 'keys') else []
                
                # Convert to list of dictionaries
                data = []
                for row in rows:
                    if hasattr(row, '_asdict'):
                        data.append(row._asdict())
                    elif hasattr(row, '_mapping'):
                        data.append(dict(row._mapping))
                    else:
                        data.append(dict(zip(columns, row)))
                
                # Prepare response
                row_count = len(data)
                preview_rows = self.sql_config['preview_rows']
                
                return {
                    'status': 'success',
                    'row_count': row_count,
                    'columns': columns,
                    'data': data,
                    'preview_data': data[:preview_rows],
                    'truncated': row_count > preview_rows
                }
                
        except SQLAlchemyError as e:
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            return {
                'status': 'error',
                'error': f'Database error: {error_msg}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _get_engine(self, source: DataSource):
        """Get or create database engine for source"""
        cache_key = f"{source.id}_{source.name}"
        
        if cache_key not in self.engines:
            try:
                # Configure engine with timeouts and limits
                engine_options = {
                    'pool_pre_ping': True,
                    'pool_recycle': 300,
                    'connect_args': {
                        'timeout': self.sql_config['max_execution_time']
                    }
                }
                
                # Add database-specific options
                if 'sqlite' in source.connection_string:
                    engine_options['connect_args']['timeout'] = self.sql_config['max_execution_time']
                elif 'postgresql' in source.connection_string:
                    engine_options['connect_args']['options'] = f"-c statement_timeout={self.sql_config['max_execution_time']}s"
                elif 'mysql' in source.connection_string:
                    engine_options['connect_args']['read_timeout'] = self.sql_config['max_execution_time']
                
                self.engines[cache_key] = create_engine(
                    source.connection_string,
                    **engine_options
                )
                
                logger.info(f"Created engine for source {source.name}")
                
            except Exception as e:
                logger.error(f"Error creating engine for {source.name}: {str(e)}")
                raise
        
        return self.engines[cache_key]
    
    def _log_execution(self, sql: str, source_id: int, session_id: int = None,
                      message_id: int = None, status: str = 'success',
                      execution_time: float = 0.0, row_count: int = 0,
                      error_message: str = None, result_preview: List[Dict] = None) -> None:
        """Log query execution for audit and monitoring"""
        try:
            # Limit preview data size
            if result_preview and len(result_preview) > 5:
                result_preview = result_preview[:5]
            
            execution_log = QueryExecution(
                session_id=session_id,
                message_id=message_id,
                sql_query=sql,
                source_id=source_id,
                status=status,
                execution_time=round(execution_time, 3),
                row_count=row_count,
                error_message=error_message,
                result_preview=result_preview,
                created_at=datetime.utcnow()
            )
            
            db.session.add(execution_log)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error logging query execution: {str(e)}")
            # Don't raise - logging failure shouldn't break query execution
    
    def get_execution_history(self, session_id: int = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get query execution history"""
        try:
            query = QueryExecution.query
            
            if session_id:
                query = query.filter_by(session_id=session_id)
            
            executions = query.order_by(QueryExecution.created_at.desc()).limit(limit).all()
            
            return [execution.to_dict() for execution in executions]
            
        except Exception as e:
            logger.error(f"Error getting execution history: {str(e)}")
            return []
    
    def get_execution_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get execution statistics for the specified time period"""
        try:
            from datetime import timedelta
            
            since = datetime.utcnow() - timedelta(hours=hours)
            
            executions = QueryExecution.query.filter(
                QueryExecution.created_at >= since
            ).all()
            
            if not executions:
                return {
                    'total_queries': 0,
                    'successful_queries': 0,
                    'failed_queries': 0,
                    'avg_execution_time': 0.0,
                    'total_rows_returned': 0
                }
            
            successful = [e for e in executions if e.status == 'success']
            failed = [e for e in executions if e.status == 'error']
            
            avg_execution_time = sum(e.execution_time or 0 for e in executions) / len(executions)
            total_rows = sum(e.row_count or 0 for e in successful)
            
            return {
                'total_queries': len(executions),
                'successful_queries': len(successful),
                'failed_queries': len(failed),
                'success_rate': round(len(successful) / len(executions) * 100, 1),
                'avg_execution_time': round(avg_execution_time, 3),
                'total_rows_returned': total_rows,
                'period_hours': hours
            }
            
        except Exception as e:
            logger.error(f"Error getting execution stats: {str(e)}")
            return {}
    
    def explain_query(self, sql: str, source_id: int) -> Dict[str, Any]:
        """Get query execution plan"""
        try:
            source = DataSource.query.get(source_id)
            if not source:
                return {'error': 'Data source not found'}
            
            # Add EXPLAIN to the query
            explain_sql = f"EXPLAIN {sql}"
            
            # Validate the explain query
            validation_result = self._validate_sql(explain_sql)
            if not validation_result['valid']:
                return {'error': validation_result['error']}
            
            # Execute explain
            result = self._execute_on_source(explain_sql, source)
            
            return result
            
        except Exception as e:
            logger.error(f"Error explaining query: {str(e)}")
            return {'error': str(e)}
    
    def test_query_syntax(self, sql: str) -> Dict[str, Any]:
        """Test SQL syntax without execution"""
        try:
            validation_result = self._validate_sql(sql)
            
            if validation_result['valid']:
                # Additional syntax checking with sqlparse
                try:
                    parsed = sqlparse.parse(sql)
                    formatted = sqlparse.format(sql, reindent=True, keyword_case='upper')
                    
                    return {
                        'valid': True,
                        'formatted_sql': formatted,
                        'parsed_statements': len(parsed)
                    }
                except Exception as e:
                    return {
                        'valid': False,
                        'error': f'Syntax error: {str(e)}'
                    }
            else:
                return validation_result
                
        except Exception as e:
            logger.error(f"Error testing query syntax: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def get_slow_queries(self, threshold_seconds: float = 5.0, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slow queries above threshold"""
        try:
            slow_queries = QueryExecution.query.filter(
                QueryExecution.execution_time >= threshold_seconds,
                QueryExecution.status == 'success'
            ).order_by(QueryExecution.execution_time.desc()).limit(limit).all()
            
            return [query.to_dict() for query in slow_queries]
            
        except Exception as e:
            logger.error(f"Error getting slow queries: {str(e)}")
            return []
    
    def cleanup_old_executions(self, days: int = 30) -> int:
        """Clean up old execution logs"""
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            old_executions = QueryExecution.query.filter(
                QueryExecution.created_at < cutoff_date
            ).all()
            
            count = len(old_executions)
            
            for execution in old_executions:
                db.session.delete(execution)
            
            db.session.commit()
            
            logger.info(f"Cleaned up {count} old execution logs")
            return count
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error cleaning up old executions: {str(e)}")
            return 0
        