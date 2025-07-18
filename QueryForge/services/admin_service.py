# services/admin_service.py
import os
import time
import logging
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import text, inspect, create_engine
import sqlite3
import json

from models import (
    db, Project, DataSource, Table, Column, DictionaryEntry, 
    Embedding, Index, SearchLog, NLQFeedback
)

logger = logging.getLogger(__name__)

class AdminService:
    """Service for admin operations and system monitoring"""
    
    def __init__(self):
        self.allowed_sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 
            'OUTER', 'ON', 'GROUP', 'BY', 'ORDER', 'LIMIT', 'OFFSET',
            'HAVING', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
            'AS', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL'
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            health = {
                'overall_status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'database': self._get_database_health(),
                'services': self._get_service_health(),
                'system': self._get_system_metrics(),
                'application': self._get_application_metrics()
            }
            
            # Determine overall status
            if any(s.get('status') == 'error' for s in health['services'].values()):
                health['overall_status'] = 'error'
            elif any(s.get('status') == 'warning' for s in health['services'].values()):
                health['overall_status'] = 'warning'
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _get_database_health(self) -> Dict[str, Any]:
        """Get database health and statistics"""
        try:
            # Test connection
            start_time = time.time()
            db.session.execute(text("SELECT 1"))
            connection_time = time.time() - start_time
            
            # Get record counts
            records = {}
            try:
                records['projects'] = Project.query.count()
                records['data_sources'] = DataSource.query.count()
                records['tables'] = Table.query.count()
                records['columns'] = Column.query.count()
                records['dictionary_entries'] = DictionaryEntry.query.count()
                records['embeddings'] = Embedding.query.count()
                records['indexes'] = Index.query.count()
                records['search_logs'] = SearchLog.query.count()
            except Exception as e:
                logger.warning(f"Error getting record counts: {str(e)}")
                records = {'error': 'Could not fetch record counts'}
            
            # Get database size
            db_size_mb = 0
            try:
                if 'sqlite' in str(db.engine.url):
                    # For SQLite
                    db_path = str(db.engine.url).replace('sqlite:///', '')
                    if os.path.exists(db_path):
                        db_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)
                else:
                    # For PostgreSQL/MySQL
                    result = db.session.execute(text(
                        "SELECT pg_size_pretty(pg_database_size(current_database()))"
                    ))
                    db_size_mb = result.scalar()
            except:
                pass
            
            return {
                'status': 'healthy',
                'records': records,
                'total_records': sum(v for v in records.values() if isinstance(v, int)),
                'size_mb': db_size_mb,
                'connection_status': 'connected',
                'connection_time_ms': round(connection_time * 1000, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting database health: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'connection_status': 'disconnected'
            }
    
    def _get_service_health(self) -> Dict[str, Any]:
        """Get health status of various services"""
        services = {}
        
        # Database service
        try:
            db.session.execute(text("SELECT 1"))
            services['database'] = {
                'status': 'healthy',
                'message': 'Database connection active'
            }
        except Exception as e:
            services['database'] = {
                'status': 'error',
                'message': f'Database error: {str(e)}'
            }
        
        # File system service
        try:
            upload_dir = 'uploads'
            if os.path.exists(upload_dir) and os.access(upload_dir, os.W_OK):
                services['file_system'] = {
                    'status': 'healthy',
                    'message': 'Upload directory accessible'
                }
            else:
                services['file_system'] = {
                    'status': 'warning',
                    'message': 'Upload directory not accessible'
                }
        except Exception as e:
            services['file_system'] = {
                'status': 'error',
                'message': f'File system error: {str(e)}'
            }
        
        # Index directory service
        try:
            index_dir = 'indexes'
            if os.path.exists(index_dir) and os.access(index_dir, os.W_OK):
                services['index_storage'] = {
                    'status': 'healthy',
                    'message': 'Index directory accessible'
                }
            else:
                services['index_storage'] = {
                    'status': 'warning',
                    'message': 'Index directory not accessible'
                }
        except Exception as e:
            services['index_storage'] = {
                'status': 'error',
                'message': f'Index storage error: {str(e)}'
            }
        
        # Embedding service (basic check)
        try:
            # Check if we can import required packages
            import sentence_transformers
            services['embedding_service'] = {
                'status': 'healthy',
                'message': 'Embedding dependencies available'
            }
        except ImportError:
            services['embedding_service'] = {
                'status': 'warning',
                'message': 'Embedding dependencies not available'
            }
        except Exception as e:
            services['embedding_service'] = {
                'status': 'error',
                'message': f'Embedding service error: {str(e)}'
            }
        
        return services
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': round(cpu_percent, 1),
                'memory_percent': round(memory.percent, 1),
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_percent': round(disk.percent, 1),
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_total_gb': round(disk.total / (1024**3), 2)
            }
        except Exception as e:
            logger.warning(f"Error getting system metrics: {str(e)}")
            return {
                'error': 'System metrics unavailable'
            }
    
    def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        try:
            # Recent activity (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            
            recent_searches = SearchLog.query.filter(
                SearchLog.timestamp >= yesterday
            ).count()
            
            # Embedding statistics
            total_embeddings = Embedding.query.count()
            embedding_models = db.session.query(Embedding.model_name).distinct().count()
            
            # Index statistics
            ready_indexes = Index.query.filter_by(status='ready').count()
            total_indexes = Index.query.count()
            
            # Project statistics
            active_projects = Project.query.filter_by(status='active').count()
            
            return {
                'embeddings_count': total_embeddings,
                'embedding_models_count': embedding_models,
                'indexes_count': ready_indexes,
                'total_indexes': total_indexes,
                'active_projects': active_projects,
                'activity_24h': {
                    'searches': recent_searches
                }
            }
        except Exception as e:
            logger.warning(f"Error getting application metrics: {str(e)}")
            return {
                'error': 'Application metrics unavailable'
            }
    
    def execute_sql_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query with safety checks"""
        try:
            # Security validation
            if not self._is_safe_sql(sql):
                return {
                    'success': False,
                    'error': 'Only SELECT statements are allowed',
                    'data': [],
                    'columns': [],
                    'row_count': 0
                }
            
            start_time = time.time()
            
            # Execute query
            result = db.session.execute(text(sql))
            
            # Get columns and data
            columns = list(result.keys()) if result.keys() else []
            data = []
            
            for row in result:
                row_dict = {}
                for i, column in enumerate(columns):
                    value = row[i]
                    # Convert datetime objects to strings
                    if isinstance(value, datetime):
                        value = value.isoformat()
                    # Round decimal values
                    elif isinstance(value, (float, int)) and isinstance(value, float):
                        value = round(value, 3)
                    row_dict[column] = value
                data.append(row_dict)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'data': data,
                'columns': columns,
                'row_count': len(data),
                'execution_time_ms': round(execution_time * 1000, 2),
                'sql': sql
            }
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'columns': [],
                'row_count': 0,
                'sql': sql
            }
    
    def _is_safe_sql(self, sql: str) -> bool:
        """Check if SQL query is safe to execute"""
        sql_upper = sql.strip().upper()
        
        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            return False
        
        # Check for dangerous keywords
        dangerous_keywords = {
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
            'TRUNCATE', 'REPLACE', 'MERGE', 'EXEC', 'EXECUTE',
            'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK'
        }
        
        # Split by whitespace and check each word
        words = sql_upper.split()
        for word in words:
            clean_word = word.strip('(),;')
            if clean_word in dangerous_keywords:
                return False
        
        # Check for SQL injection patterns
        injection_patterns = [
            '--', '/*', '*/', 'xp_', 'sp_cmdshell', 'exec(',
            'execute(', 'eval(', 'union select', '; drop',
            '; delete', '; insert', '; update'
        ]
        
        sql_lower = sql.lower()
        for pattern in injection_patterns:
            if pattern in sql_lower:
                return False
        
        return True
    
    def get_table_details(self, table_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        try:
            table = Table.query.get(table_id)
            if not table:
                return {
                    'success': False,
                    'error': f'Table {table_id} not found'
                }
            
            # Get column details
            columns = []
            for col in table.columns:
                col_data = col.to_dict()
                columns.append(col_data)
            
            # Get sample data (first 10 rows)
            try:
                from services.data_source_service import DataSourceService
                data_service = DataSourceService()
                sample_data = data_service.get_table_sample_data(table_id, limit=10)
            except Exception as e:
                logger.warning(f"Could not get sample data: {str(e)}")
                sample_data = {'columns': [], 'data': [], 'total_rows': 0}
            
            return {
                'success': True,
                'table': table.to_dict(),
                'columns': columns,
                'sample_data': sample_data,
                'source': table.source.to_dict() if table.source else None
            }
            
        except Exception as e:
            logger.error(f"Error getting table details: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Get complete database schema information"""
        try:
            schema = {
                'tables': [],
                'relationships': [],
                'statistics': {}
            }
            
            # Get all tables
            tables = Table.query.all()
            for table in tables:
                table_info = {
                    'id': table.id,
                    'name': table.name,
                    'display_name': table.display_name,
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'source': {
                        'id': table.source.id,
                        'name': table.source.name,
                        'type': table.source.type,
                        'project_name': table.source.project.name
                    } if table.source else None,
                    'columns': []
                }
                
                # Get column information
                for col in table.columns:
                    col_info = {
                        'name': col.name,
                        'data_type': col.data_type,
                        'is_nullable': col.is_nullable,
                        'is_primary_key': col.is_primary_key,
                        'is_foreign_key': col.is_foreign_key,
                        'business_category': col.business_category
                    }
                    table_info['columns'].append(col_info)
                
                schema['tables'].append(table_info)
            
            # Get basic statistics
            schema['statistics'] = {
                'total_tables': len(tables),
                'total_columns': Column.query.count(),
                'total_projects': Project.query.count(),
                'total_sources': DataSource.query.count()
            }
            
            return {
                'success': True,
                'schema': schema
            }
            
        except Exception as e:
            logger.error(f"Error getting database schema: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cleanup_orphaned_data(self) -> Dict[str, Any]:
        """Clean up orphaned data and fix inconsistencies"""
        try:
            cleanup_results = {
                'embeddings_cleaned': 0,
                'indexes_cleaned': 0,
                'search_logs_cleaned': 0,
                'files_cleaned': []
            }
            
            # Clean up embeddings for non-existent objects
            orphaned_embeddings = 0
            embeddings = Embedding.query.all()
            for emb in embeddings:
                object_exists = False
                
                if emb.object_type == 'table':
                    object_exists = Table.query.get(emb.object_id) is not None
                elif emb.object_type == 'column':
                    object_exists = Column.query.get(emb.object_id) is not None
                elif emb.object_type == 'dictionary_entry':
                    object_exists = DictionaryEntry.query.get(emb.object_id) is not None
                
                if not object_exists:
                    db.session.delete(emb)
                    orphaned_embeddings += 1
            
            cleanup_results['embeddings_cleaned'] = orphaned_embeddings
            
            # Clean up old search logs (older than 90 days)
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            old_logs = SearchLog.query.filter(SearchLog.timestamp < cutoff_date).count()
            SearchLog.query.filter(SearchLog.timestamp < cutoff_date).delete()
            cleanup_results['search_logs_cleaned'] = old_logs
            
            # Clean up orphaned index files
            index_dir = 'indexes'
            if os.path.exists(index_dir):
                valid_index_ids = {str(idx.id) for idx in Index.query.all()}
                for item in os.listdir(index_dir):
                    item_path = os.path.join(index_dir, item)
                    if os.path.isdir(item_path) and item not in valid_index_ids:
                        try:
                            import shutil
                            shutil.rmtree(item_path)
                            cleanup_results['files_cleaned'].append(item_path)
                        except Exception as e:
                            logger.warning(f"Could not clean up {item_path}: {str(e)}")
            
            db.session.commit()
            
            return {
                'success': True,
                'cleanup_results': cleanup_results
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during cleanup: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_query_suggestions(self, table_name: str = None) -> List[str]:
        """Get suggested SQL queries for exploration"""
        suggestions = [
            "SELECT COUNT(*) as total_records FROM projects;",
            "SELECT type, COUNT(*) as count FROM sources GROUP BY type;",
            "SELECT name, row_count, column_count FROM tables ORDER BY row_count DESC LIMIT 10;",
            "SELECT model_name, COUNT(*) as embedding_count FROM embeddings GROUP BY model_name;",
            "SELECT status, COUNT(*) as count FROM indexes GROUP BY status;",
            "SELECT p.name as project, COUNT(t.id) as table_count FROM projects p LEFT JOIN sources s ON p.id = s.project_id LEFT JOIN tables t ON s.id = t.source_id GROUP BY p.id, p.name;",
            "SELECT data_type, COUNT(*) as column_count FROM columns GROUP BY data_type ORDER BY column_count DESC;",
            "SELECT category, COUNT(*) as term_count FROM dictionary GROUP BY category;",
            "SELECT DATE(created_at) as date, COUNT(*) as searches FROM search_logs WHERE created_at >= DATE('now', '-7 days') GROUP BY DATE(created_at);",
            "SELECT t.name as table_name, COUNT(c.id) as column_count FROM tables t LEFT JOIN columns c ON t.id = c.table_id GROUP BY t.id, t.name ORDER BY column_count DESC;"
        ]
        
        if table_name:
            # Add table-specific suggestions
            table_suggestions = [
                f"SELECT * FROM {table_name} LIMIT 10;",
                f"SELECT COUNT(*) FROM {table_name};",
                f"SELECT * FROM {table_name} WHERE id IS NOT NULL LIMIT 5;",
            ]
            suggestions.extend(table_suggestions)
        
        return suggestions