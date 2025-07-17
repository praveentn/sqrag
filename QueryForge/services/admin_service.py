# services/admin_service.py
import os
import psutil
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import text, inspect
import json

from config import Config
from models import db, Project, DataSource, Table, Column, DictionaryEntry, Embedding, Index, SearchLog, NLQFeedback

logger = logging.getLogger(__name__)

class AdminService:
    """Service for administrative operations and system monitoring"""
    
    def __init__(self):
        self.sql_safety_patterns = self._build_safety_patterns()
    
    def browse_tables(self, page: int = 1, per_page: int = 50) -> Dict[str, Any]:
        """Browse database tables with pagination"""
        try:
            # Get all tables across all projects
            tables_query = Table.query.join(Table.source).order_by(Table.id.desc())
            
            # Apply pagination
            tables_paginated = tables_query.paginate(
                page=page, 
                per_page=per_page, 
                error_out=False
            )
            
            tables_data = []
            for table in tables_paginated.items:
                table_info = table.to_dict()
                table_info.update({
                    'project_name': table.source.project.name if table.source.project else 'Unknown',
                    'source_name': table.source.name if table.source else 'Unknown',
                    'source_type': table.source.type if table.source else 'Unknown'
                })
                tables_data.append(table_info)
            
            return {
                'tables': tables_data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': tables_paginated.total,
                    'pages': tables_paginated.pages,
                    'has_prev': tables_paginated.has_prev,
                    'has_next': tables_paginated.has_next
                }
            }
            
        except Exception as e:
            logger.error(f"Error browsing tables: {str(e)}")
            raise
    
    def browse_table_data(self, table_id: int, page: int = 1, 
                         per_page: int = 100) -> Dict[str, Any]:
        """Browse data within a specific table"""
        try:
            table = Table.query.get(table_id)
            if not table:
                raise ValueError(f"Table {table_id} not found")
            
            # For now, we'll return column metadata
            # In a full implementation, you'd query the actual data
            columns = Column.query.filter_by(table_id=table_id).all()
            
            return {
                'table': table.to_dict(),
                'columns': [col.to_dict() for col in columns],
                'sample_data': self._get_table_sample_data(table),
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': table.row_count or 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error browsing table data: {str(e)}")
            raise
    
    def _get_table_sample_data(self, table: Table) -> List[Dict[str, Any]]:
        """Get sample data from table (placeholder implementation)"""
        try:
            # This is a simplified implementation
            # In practice, you'd query the actual data source
            
            if table.source.type == 'file':
                # For file sources, you might read from the file
                return [
                    {col.name: f"sample_value_{i}" for col in table.columns}
                    for i in range(min(5, table.row_count or 0))
                ]
            elif table.source.type == 'database':
                # For database sources, you'd query the actual table
                # This requires implementing connection logic
                return []
            
            return []
            
        except Exception as e:
            logger.warning(f"Error getting sample data: {str(e)}")
            return []
    
    def execute_sql(self, sql: str, limit_rows: bool = True) -> Dict[str, Any]:
        """Execute SQL query with safety checks"""
        try:
            # Validate SQL safety
            if not self.validate_sql_safety(sql):
                raise ValueError("SQL query failed safety validation")
            
            # Add LIMIT if not present and limit_rows is True
            if limit_rows and not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
                sql = f"{sql.rstrip(';')} LIMIT {Config.SQL_CONFIG['max_rows']}"
            
            # Execute query
            start_time = datetime.utcnow()
            result = db.session.execute(text(sql))
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Fetch results if query returns rows
            if result.returns_rows:
                columns = list(result.keys())
                rows = result.fetchall()
                data = [dict(zip(columns, row)) for row in rows]
            else:
                columns = []
                data = []
                rows = []
            
            return {
                'success': True,
                'columns': columns,
                'data': data,
                'row_count': len(data),
                'execution_time_seconds': round(execution_time, 3),
                'sql_query': sql
            }
            
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'sql_query': sql
            }
    
    def validate_sql_safety(self, sql: str) -> bool:
        """Validate if SQL query is safe to execute"""
        sql_upper = sql.upper().strip()
        
        # Check for allowed statements
        allowed_statements = Config.SQL_CONFIG['allowed_statements']
        if not any(sql_upper.startswith(stmt) for stmt in allowed_statements):
            logger.warning(f"SQL not allowed - must start with: {allowed_statements}")
            return False
        
        # Check for blocked keywords
        blocked_keywords = Config.SQL_CONFIG['blocked_keywords']
        for keyword in blocked_keywords:
            if keyword in sql_upper:
                logger.warning(f"SQL contains blocked keyword: {keyword}")
                return False
        
        # Additional safety patterns
        for pattern, description in self.sql_safety_patterns.items():
            if re.search(pattern, sql, re.IGNORECASE):
                logger.warning(f"SQL failed safety check: {description}")
                return False
        
        return True
    
    def _build_safety_patterns(self) -> Dict[str, str]:
        """Build regex patterns for SQL safety validation"""
        return {
            r'\bEXEC\b|\bEXECUTE\b': 'Execute statements not allowed',
            r'\bSP_\w+': 'Stored procedures not allowed',
            r'\bXP_\w+': 'Extended procedures not allowed',
            r'--.*?(DELETE|DROP|UPDATE|INSERT)': 'Suspicious commented commands',
            r'/\*.*?(DELETE|DROP|UPDATE|INSERT).*?\*/': 'Suspicious block comments',
            r'\bUNION\b.*?\bSELECT\b.*?\bFROM\b': 'Complex UNION queries require review',
            r'\bINTO\b\s+OUTFILE': 'File operations not allowed',
            r'\bLOAD_FILE\b': 'File operations not allowed',
            r'\bINTO\b\s+DUMPFILE': 'File operations not allowed'
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics"""
        try:
            health = {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'healthy',
                'services': {},
                'database': {},
                'system': {},
                'application': {}
            }
            
            # Database health
            health['database'] = self._get_database_health()
            
            # System metrics
            health['system'] = self._get_system_metrics()
            
            # Application metrics
            health['application'] = self._get_application_metrics()
            
            # Service health checks
            health['services'] = self._check_services_health()
            
            # Overall status
            if any(service['status'] != 'healthy' for service in health['services'].values()):
                health['status'] = 'degraded'
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _get_database_health(self) -> Dict[str, Any]:
        """Get database health metrics"""
        try:
            # Basic connection test
            db.session.execute(text("SELECT 1"))
            
            # Get database size info
            inspector = inspect(db.engine)
            table_names = inspector.get_table_names()
            
            # Count records in main tables
            project_count = Project.query.count()
            source_count = DataSource.query.count()
            table_count = Table.query.count()
            column_count = Column.query.count()
            
            return {
                'status': 'healthy',
                'connection': 'ok',
                'tables_count': len(table_names),
                'records': {
                    'projects': project_count,
                    'sources': source_count,
                    'tables': table_count,
                    'columns': column_count
                },
                'engine': str(db.engine.url.drivername)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            return {
                'cpu': {
                    'usage_percent': round(cpu_percent, 2),
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'usage_percent': round(memory.percent, 2)
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'usage_percent': round((disk.used / disk.total) * 100, 2)
                }
            }
            
        except Exception as e:
            logger.warning(f"Error getting system metrics: {str(e)}")
            return {'error': str(e)}
    
    def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        try:
            # Count embeddings and indexes
            embedding_count = Embedding.query.count()
            index_count = Index.query.count()
            
            # Recent activity
            recent_searches = SearchLog.query.filter(
                SearchLog.created_at >= datetime.utcnow() - timedelta(hours=24)
            ).count()
            
            recent_feedback = NLQFeedback.query.filter(
                NLQFeedback.created_at >= datetime.utcnow() - timedelta(hours=24)
            ).count()
            
            # Index status
            index_statuses = db.session.query(
                Index.status,
                db.func.count(Index.id)
            ).group_by(Index.status).all()
            
            return {
                'embeddings_count': embedding_count,
                'indexes_count': index_count,
                'index_statuses': {status: count for status, count in index_statuses},
                'activity_24h': {
                    'searches': recent_searches,
                    'feedback': recent_feedback
                }
            }
            
        except Exception as e:
            logger.warning(f"Error getting application metrics: {str(e)}")
            return {'error': str(e)}
    
    def _check_services_health(self) -> Dict[str, Dict[str, str]]:
        """Check health of various services"""
        services = {}
        
        # Database service
        try:
            db.session.execute(text("SELECT 1"))
            services['database'] = {'status': 'healthy', 'message': 'Connection OK'}
        except Exception as e:
            services['database'] = {'status': 'error', 'message': str(e)}
        
        # File system (uploads directory)
        try:
            upload_dir = Config.UPLOAD_FOLDER
            if os.path.exists(upload_dir) and os.access(upload_dir, os.W_OK):
                services['filesystem'] = {'status': 'healthy', 'message': 'Upload directory accessible'}
            else:
                services['filesystem'] = {'status': 'error', 'message': 'Upload directory not accessible'}
        except Exception as e:
            services['filesystem'] = {'status': 'error', 'message': str(e)}
        
        # Indexes directory
        try:
            index_dir = 'indexes'
            if os.path.exists(index_dir) and os.access(index_dir, os.W_OK):
                services['indexes'] = {'status': 'healthy', 'message': 'Index directory accessible'}
            else:
                services['indexes'] = {'status': 'error', 'message': 'Index directory not accessible'}
        except Exception as e:
            services['indexes'] = {'status': 'error', 'message': str(e)}
        
        return services
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old logs and temporary data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Clean old search logs
            old_searches = SearchLog.query.filter(
                SearchLog.created_at < cutoff_date
            ).count()
            
            SearchLog.query.filter(
                SearchLog.created_at < cutoff_date
            ).delete()
            
            # Clean old NLQ feedback (keep only rated ones)
            old_feedback = NLQFeedback.query.filter(
                NLQFeedback.created_at < cutoff_date,
                NLQFeedback.rating.is_(None)
            ).count()
            
            NLQFeedback.query.filter(
                NLQFeedback.created_at < cutoff_date,
                NLQFeedback.rating.is_(None)
            ).delete()
            
            db.session.commit()
            
            return {
                'search_logs_deleted': old_searches,
                'feedback_deleted': old_feedback
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error cleaning up old data: {str(e)}")
            raise
    
    def export_project_data(self, project_id: int) -> Dict[str, Any]:
        """Export all project data for backup"""
        try:
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            export_data = {
                'project': project.to_dict(),
                'sources': [],
                'dictionary': [],
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
            # Export sources and their tables/columns
            for source in project.sources:
                source_data = source.to_dict()
                source_data['tables'] = []
                
                for table in source.tables:
                    table_data = table.to_dict()
                    table_data['columns'] = [col.to_dict() for col in table.columns]
                    source_data['tables'].append(table_data)
                
                export_data['sources'].append(source_data)
            
            # Export dictionary
            for entry in project.dictionary_entries:
                export_data['dictionary'].append(entry.to_dict())
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting project data: {str(e)}")
            raise
    
    def get_usage_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for the specified period"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Search statistics
            search_stats = db.session.query(
                db.func.date(SearchLog.created_at).label('date'),
                db.func.count(SearchLog.id).label('count'),
                db.func.avg(SearchLog.response_time_ms).label('avg_response_time')
            ).filter(
                SearchLog.created_at >= start_date
            ).group_by(
                db.func.date(SearchLog.created_at)
            ).all()
            
            # Feedback statistics
            feedback_stats = db.session.query(
                db.func.date(NLQFeedback.created_at).label('date'),
                db.func.count(NLQFeedback.id).label('count'),
                db.func.avg(NLQFeedback.rating).label('avg_rating')
            ).filter(
                NLQFeedback.created_at >= start_date,
                NLQFeedback.rating.isnot(None)
            ).group_by(
                db.func.date(NLQFeedback.created_at)
            ).all()
            
            # Top queries
            top_queries = db.session.query(
                SearchLog.query_text,
                db.func.count(SearchLog.id).label('count')
            ).filter(
                SearchLog.created_at >= start_date
            ).group_by(
                SearchLog.query_text
            ).order_by(
                db.func.count(SearchLog.id).desc()
            ).limit(10).all()
            
            return {
                'period_days': days,
                'search_activity': [
                    {
                        'date': stat.date.isoformat(),
                        'searches': stat.count,
                        'avg_response_time_ms': round(stat.avg_response_time or 0, 2)
                    }
                    for stat in search_stats
                ],
                'feedback_activity': [
                    {
                        'date': stat.date.isoformat(),
                        'feedback_count': stat.count,
                        'avg_rating': round(stat.avg_rating or 0, 2)
                    }
                    for stat in feedback_stats
                ],
                'top_queries': [
                    {
                        'query': query,
                        'count': count
                    }
                    for query, count in top_queries
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {str(e)}")
            raise
    
    def rebuild_all_indexes(self, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Rebuild all indexes for a project or all projects"""
        try:
            query = Index.query
            if project_id:
                query = query.filter_by(project_id=project_id)
            
            indexes = query.all()
            results = {
                'total_indexes': len(indexes),
                'rebuilt': 0,
                'failed': 0,
                'errors': []
            }
            
            for index in indexes:
                try:
                    # This would need to be implemented with proper job queuing
                    # For now, just mark as needing rebuild
                    index.status = 'building'
                    db.session.commit()
                    results['rebuilt'] += 1
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Index {index.id}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error rebuilding indexes: {str(e)}")
            raise