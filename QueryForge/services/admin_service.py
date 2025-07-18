# services/admin_service.py
import os
import logging
import psutil  # FIXED: Use psutil instead of Memory for system monitoring
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import text, inspect
import json

from models import db, Project, DataSource, Table, Column, DictionaryEntry, Embedding, Index, SearchLog, NLQFeedback

logger = logging.getLogger(__name__)

class AdminService:
    """Service for system administration and monitoring"""
    
    def __init__(self):
        self.logger = logger
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            stats = {
                'database': self._get_database_stats(),
                'system': self._get_system_stats(),
                'projects': self._get_project_stats(),
                'usage': self._get_usage_stats(),
                'health': self._get_health_status()
            }
            
            return {
                'success': True,
                'stats': stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            # Count records in each table
            stats['record_counts'] = {
                'projects': Project.query.count(),
                'data_sources': DataSource.query.count(),
                'tables': Table.query.count(),
                'columns': Column.query.count(),
                'dictionary_entries': DictionaryEntry.query.count(),
                'embeddings': Embedding.query.count(),
                'indexes': Index.query.count(),
                'search_logs': SearchLog.query.count(),
                'nlq_feedback': NLQFeedback.query.count()
            }
            
            # Calculate total records
            stats['total_records'] = sum(stats['record_counts'].values())
            
            # Get database file size (for SQLite)
            try:
                db_path = 'queryforge.db'  # Default SQLite path
                if os.path.exists(db_path):
                    stats['database_size_mb'] = round(os.path.getsize(db_path) / (1024 * 1024), 2)
                else:
                    stats['database_size_mb'] = 0
            except:
                stats['database_size_mb'] = 0
            
            # Get database engine info
            stats['database_type'] = str(db.engine.url).split('://')[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {'error': str(e)}
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get system resource statistics"""
        try:
            # Use psutil for system monitoring - FIXED: No more Memory import error
            stats = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory': {
                    'total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                    'available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                    'used_percent': psutil.virtual_memory().percent
                },
                'disk': {
                    'total_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
                    'free_gb': round(psutil.disk_usage('/').free / (1024**3), 2),
                    'used_percent': psutil.disk_usage('/').percent
                },
                'uptime_hours': round((datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600, 1)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                'cpu_percent': 0,
                'memory': {'total_gb': 0, 'available_gb': 0, 'used_percent': 0},
                'disk': {'total_gb': 0, 'free_gb': 0, 'used_percent': 0},
                'uptime_hours': 0,
                'error': str(e)
            }
    
    def _get_project_stats(self) -> Dict[str, Any]:
        """Get project-level statistics"""
        try:
            projects = Project.query.all()
            
            stats = {
                'total_projects': len(projects),
                'active_projects': len([p for p in projects if p.status == 'active']),
                'projects_by_status': {},
                'top_projects': []
            }
            
            # Count by status
            for project in projects:
                status = project.status or 'unknown'
                stats['projects_by_status'][status] = stats['projects_by_status'].get(status, 0) + 1
            
            # Get top projects by data volume
            for project in projects:
                sources_count = len(project.sources)
                entries_count = len(project.dictionary_entries)
                total_items = sources_count + entries_count
                
                stats['top_projects'].append({
                    'id': project.id,
                    'name': project.name,
                    'sources_count': sources_count,
                    'dictionary_entries_count': entries_count,
                    'total_items': total_items,
                    'created_at': project.created_at.isoformat() if project.created_at else None
                })
            
            # Sort by total items
            stats['top_projects'].sort(key=lambda x: x['total_items'], reverse=True)
            stats['top_projects'] = stats['top_projects'][:10]  # Top 10
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting project stats: {str(e)}")
            return {'error': str(e)}
    
    def _get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        try:
            # Search activity (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            recent_searches = SearchLog.query.filter(
                SearchLog.created_at >= thirty_days_ago
            ).all()
            
            stats = {
                'search_activity': {
                    'total_searches_30d': len(recent_searches),
                    'avg_search_time_ms': round(sum(s.search_time_ms or 0 for s in recent_searches) / len(recent_searches), 2) if recent_searches else 0,
                    'most_searched_projects': self._get_most_searched_projects(recent_searches)
                },
                'data_ingestion': {
                    'total_sources': DataSource.query.count(),
                    'successful_ingests': DataSource.query.filter_by(ingest_status='completed').count(),
                    'failed_ingests': DataSource.query.filter_by(ingest_status='failed').count(),
                    'pending_ingests': DataSource.query.filter_by(ingest_status='pending').count()
                },
                'embeddings': {
                    'total_embeddings': Embedding.query.count(),
                    'total_indexes': Index.query.count(),
                    'ready_indexes': Index.query.filter_by(status='ready').count()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting usage stats: {str(e)}")
            return {'error': str(e)}
    
    def _get_most_searched_projects(self, search_logs: List[SearchLog]) -> List[Dict]:
        """Get most searched projects from search logs"""
        try:
            project_counts = {}
            for log in search_logs:
                project_counts[log.project_id] = project_counts.get(log.project_id, 0) + 1
            
            # Get project names
            result = []
            for project_id, count in sorted(project_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                project = Project.query.get(project_id)
                if project:
                    result.append({
                        'project_id': project_id,
                        'project_name': project.name,
                        'search_count': count
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting most searched projects: {str(e)}")
            return []
    
    def _get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            health = {
                'overall_status': 'healthy',
                'checks': {}
            }
            
            # Database connectivity
            try:
                db.session.execute(text('SELECT 1'))
                health['checks']['database'] = {'status': 'healthy', 'message': 'Database connection OK'}
            except Exception as e:
                health['checks']['database'] = {'status': 'unhealthy', 'message': f'Database error: {str(e)}'}
                health['overall_status'] = 'unhealthy'
            
            # File system space
            try:
                disk_usage = psutil.disk_usage('/')
                free_percent = (disk_usage.free / disk_usage.total) * 100
                if free_percent < 10:
                    health['checks']['disk_space'] = {'status': 'warning', 'message': f'Low disk space: {free_percent:.1f}% free'}
                    if health['overall_status'] == 'healthy':
                        health['overall_status'] = 'warning'
                else:
                    health['checks']['disk_space'] = {'status': 'healthy', 'message': f'Disk space OK: {free_percent:.1f}% free'}
            except Exception as e:
                health['checks']['disk_space'] = {'status': 'unknown', 'message': f'Cannot check disk space: {str(e)}'}
            
            # Memory usage
            try:
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    health['checks']['memory'] = {'status': 'warning', 'message': f'High memory usage: {memory.percent:.1f}%'}
                    if health['overall_status'] == 'healthy':
                        health['overall_status'] = 'warning'
                else:
                    health['checks']['memory'] = {'status': 'healthy', 'message': f'Memory usage OK: {memory.percent:.1f}%'}
            except Exception as e:
                health['checks']['memory'] = {'status': 'unknown', 'message': f'Cannot check memory: {str(e)}'}
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                'overall_status': 'unknown',
                'checks': {},
                'error': str(e)
            }
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query (admin only)"""
        try:
            # Security check - only allow SELECT statements for safety
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                return {
                    'success': False,
                    'error': 'Only SELECT queries are allowed for security reasons'
                }
            
            # Execute query
            result = db.session.execute(text(query))
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            if rows:
                columns = list(result.keys())
                data = [dict(zip(columns, row)) for row in rows]
            else:
                columns = []
                data = []
            
            return {
                'success': True,
                'columns': columns,
                'data': data,
                'row_count': len(data),
                'execution_time': 'N/A'
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_table_details(self, table_id: int) -> Dict[str, Any]:
        """Get detailed information about a table"""
        try:
            table = Table.query.get(table_id)
            if not table:
                return {
                    'success': False,
                    'error': f'Table {table_id} not found'
                }
            
            # Get columns
            columns = Column.query.filter_by(table_id=table_id).all()
            
            # Get related embeddings
            embeddings = Embedding.query.filter_by(
                object_type='table',
                object_id=table_id
            ).all()
            
            column_embeddings = Embedding.query.filter_by(
                object_type='column'
            ).filter(
                Embedding.object_id.in_([c.id for c in columns])
            ).all()
            
            return {
                'success': True,
                'table': table.to_dict(),
                'columns': [col.to_dict() for col in columns],
                'embeddings': {
                    'table_embeddings': len(embeddings),
                    'column_embeddings': len(column_embeddings),
                    'total_embeddings': len(embeddings) + len(column_embeddings)
                },
                'statistics': {
                    'column_count': len(columns),
                    'pii_columns': len([c for c in columns if c.pii_flag]),
                    'nullable_columns': len([c for c in columns if c.is_nullable]),
                    'primary_key_columns': len([c for c in columns if c.is_primary_key])
                }
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
            # Get all tables with their relationships
            projects = Project.query.all()
            
            schema = {
                'projects': [],
                'summary': {
                    'total_projects': len(projects),
                    'total_sources': 0,
                    'total_tables': 0,
                    'total_columns': 0
                }
            }
            
            for project in projects:
                project_data = {
                    'id': project.id,
                    'name': project.name,
                    'sources': []
                }
                
                for source in project.sources:
                    source_data = {
                        'id': source.id,
                        'name': source.name,
                        'type': source.type,
                        'status': source.ingest_status,
                        'tables': []
                    }
                    
                    for table in source.tables:
                        table_data = {
                            'id': table.id,
                            'name': table.name,
                            'row_count': table.row_count,
                            'column_count': table.column_count,
                            'columns': [
                                {
                                    'id': col.id,
                                    'name': col.name,
                                    'data_type': col.data_type,
                                    'is_nullable': col.is_nullable,
                                    'is_primary_key': col.is_primary_key,
                                    'pii_flag': col.pii_flag
                                }
                                for col in table.columns
                            ]
                        }
                        source_data['tables'].append(table_data)
                        schema['summary']['total_columns'] += len(table.columns)
                    
                    project_data['sources'].append(source_data)
                    schema['summary']['total_tables'] += len(source.tables)
                
                schema['projects'].append(project_data)
                schema['summary']['total_sources'] += len(project.sources)
            
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
        """Clean up orphaned data and inconsistencies"""
        try:
            cleanup_stats = {
                'orphaned_embeddings': 0,
                'orphaned_indexes': 0,
                'invalid_search_logs': 0,
                'total_cleaned': 0
            }
            
            # Find orphaned embeddings (embeddings with no corresponding objects)
            orphaned_embeddings = []
            
            embeddings = Embedding.query.all()
            for embedding in embeddings:
                object_exists = False
                
                if embedding.object_type == 'table':
                    object_exists = Table.query.get(embedding.object_id) is not None
                elif embedding.object_type == 'column':
                    object_exists = Column.query.get(embedding.object_id) is not None
                elif embedding.object_type == 'dictionary_entry':
                    object_exists = DictionaryEntry.query.get(embedding.object_id) is not None
                
                if not object_exists:
                    orphaned_embeddings.append(embedding)
            
            # Delete orphaned embeddings
            for embedding in orphaned_embeddings:
                db.session.delete(embedding)
                cleanup_stats['orphaned_embeddings'] += 1
            
            # Find orphaned indexes (indexes for non-existent projects)
            orphaned_indexes = []
            indexes = Index.query.all()
            for index in indexes:
                if not Project.query.get(index.project_id):
                    orphaned_indexes.append(index)
            
            # Delete orphaned indexes
            for index in orphaned_indexes:
                db.session.delete(index)
                cleanup_stats['orphaned_indexes'] += 1
            
            # Find invalid search logs (logs for non-existent projects)
            invalid_logs = []
            search_logs = SearchLog.query.all()
            for log in search_logs:
                if not Project.query.get(log.project_id):
                    invalid_logs.append(log)
            
            # Delete invalid search logs
            for log in invalid_logs:
                db.session.delete(log)
                cleanup_stats['invalid_search_logs'] += 1
            
            # Commit all changes
            db.session.commit()
            
            cleanup_stats['total_cleaned'] = (
                cleanup_stats['orphaned_embeddings'] + 
                cleanup_stats['orphaned_indexes'] + 
                cleanup_stats['invalid_search_logs']
            )
            
            return {
                'success': True,
                'cleanup_stats': cleanup_stats,
                'message': f"Cleaned up {cleanup_stats['total_cleaned']} orphaned records"
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during cleanup: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_project_data(self, project_id: int) -> Dict[str, Any]:
        """Export project data for backup"""
        try:
            project = Project.query.get(project_id)
            if not project:
                return {
                    'success': False,
                    'error': f'Project {project_id} not found'
                }
            
            # Export all project data
            export_data = {
                'project': project.to_dict(),
                'sources': [source.to_dict() for source in project.sources],
                'tables': [],
                'columns': [],
                'dictionary_entries': [entry.to_dict() for entry in project.dictionary_entries],
                'embeddings_count': len(project.embeddings),
                'indexes': [index.to_dict() for index in project.indexes],
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
            # Get tables and columns
            for source in project.sources:
                for table in source.tables:
                    export_data['tables'].append(table.to_dict())
                    for column in table.columns:
                        export_data['columns'].append(column.to_dict())
            
            return {
                'success': True,
                'export_data': export_data,
                'summary': {
                    'sources': len(export_data['sources']),
                    'tables': len(export_data['tables']),
                    'columns': len(export_data['columns']),
                    'dictionary_entries': len(export_data['dictionary_entries']),
                    'embeddings': export_data['embeddings_count'],
                    'indexes': len(export_data['indexes'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error exporting project data: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }