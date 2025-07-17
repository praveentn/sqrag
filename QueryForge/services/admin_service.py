# services/admin_service.py
import os
import psutil
import logging
import sqlalchemy
from sqlalchemy import create_engine, text
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from models import db, Project, DataSource, Table, Column, DictionaryEntry, Embedding, Index, SearchLog

logger = logging.getLogger(__name__)

class AdminService:
    """Service for system administration and monitoring"""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        try:
            health = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
                'system': self._get_system_metrics(),
                'database': self._get_database_health(),
                'services': self._get_service_health(),
                'application': self._get_application_metrics()
            }
            
            # Determine overall health status
            health['status'] = self._determine_health_status(health)
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_total_gb = round(memory.total / (1024**3), 2)
            memory_available_gb = round(memory.available / (1024**3), 2)
            memory_usage_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_total_gb = round(disk.total / (1024**3), 2)
            disk_free_gb = round(disk.free / (1024**3), 2)
            disk_usage_percent = round((disk.used / disk.total) * 100, 2)
            
            # Network metrics (basic)
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'usage_percent': round(cpu_percent, 2),
                    'count': cpu_count
                },
                'memory': {
                    'total_gb': memory_total_gb,
                    'available_gb': memory_available_gb,
                    'usage_percent': round(memory_usage_percent, 2)
                },
                'disk': {
                    'total_gb': disk_total_gb,
                    'free_gb': disk_free_gb,
                    'usage_percent': disk_usage_percent
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    def _get_database_health(self) -> Dict[str, Any]:
        """Get database health and statistics"""
        try:
            # Basic connection test
            db.session.execute(text("SELECT 1"))
            
            # Count records in main tables
            records = {
                'projects': Project.query.count(),
                'data_sources': DataSource.query.count(),
                'tables': Table.query.count(),
                'columns': Column.query.count(),
                'dictionary_entries': DictionaryEntry.query.count(),
                'embeddings': Embedding.query.count(),
                'indexes': Index.query.count(),
                'search_logs': SearchLog.query.count()
            }
            
            # Database size estimation (for SQLite)
            db_size_mb = 0
            try:
                db_file = 'queryforge.db'  # Adjust based on your config
                if os.path.exists(db_file):
                    db_size_mb = round(os.path.getsize(db_file) / (1024**2), 2)
            except:
                pass
            
            return {
                'status': 'healthy',
                'records': records,
                'total_records': sum(records.values()),
                'size_mb': db_size_mb,
                'connection_status': 'connected'
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
            logger.error(f"Error getting application metrics: {str(e)}")
            return {}
    
    def _determine_health_status(self, health: Dict[str, Any]) -> str:
        """Determine overall health status"""
        try:
            # Check critical services
            if health.get('database', {}).get('status') == 'error':
                return 'critical'
            
            # Check system resources
            system = health.get('system', {})
            cpu_usage = system.get('cpu', {}).get('usage_percent', 0)
            memory_usage = system.get('memory', {}).get('usage_percent', 0)
            disk_usage = system.get('disk', {}).get('usage_percent', 0)
            
            if cpu_usage > 90 or memory_usage > 90 or disk_usage > 95:
                return 'critical'
            
            if cpu_usage > 80 or memory_usage > 80 or disk_usage > 90:
                return 'warning'
            
            # Check service health
            services = health.get('services', {})
            error_count = sum(1 for service in services.values() if service.get('status') == 'error')
            warning_count = sum(1 for service in services.values() if service.get('status') == 'warning')
            
            if error_count > 0:
                return 'warning'
            
            if warning_count > 1:
                return 'warning'
            
            return 'healthy'
            
        except Exception as e:
            logger.error(f"Error determining health status: {str(e)}")
            return 'unknown'
    
    def execute_sql_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query safely for admin purposes"""
        try:
            start_time = datetime.utcnow()
            
            # Additional safety checks for admin queries
            sql_upper = sql.strip().upper()
            if not sql_upper.startswith('SELECT'):
                raise ValueError("Only SELECT statements are allowed")
            
            # Execute query
            result = db.session.execute(text(sql))
            
            # Fetch results
            columns = list(result.keys()) if result.keys() else []
            data = []
            
            for row in result:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i] if i < len(row) else None
                    # Convert complex types to string for JSON serialization
                    if isinstance(value, (datetime, timedelta)):
                        value = str(value)
                    row_dict[col] = value
                data.append(row_dict)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                'success': True,
                'data': data,
                'columns': columns,
                'row_count': len(data),
                'execution_time_seconds': round(execution_time, 3)
            }
            
        except Exception as e:
            logger.error(f"Error executing admin SQL: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_system_logs(self, level: str = 'INFO', limit: int = 100) -> List[Dict[str, Any]]:
        """Get system logs (if logging to file)"""
        try:
            logs = []
            log_file = 'logs/app.log'
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Take last 'limit' lines
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                
                for line in recent_lines:
                    # Simple log parsing (adjust based on your log format)
                    if level.upper() in line:
                        logs.append({
                            'timestamp': datetime.utcnow().isoformat(),
                            'level': level,
                            'message': line.strip()
                        })
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting system logs: {str(e)}")
            return []
    
    def cleanup_old_data(self, days: int = 30) -> Dict[str, Any]:
        """Clean up old data (logs, temporary files, etc.)"""
        try:
            cleanup_results = {
                'search_logs_deleted': 0,
                'temp_files_deleted': 0,
                'old_indexes_cleaned': 0
            }
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean old search logs
            old_search_logs = SearchLog.query.filter(
                SearchLog.timestamp < cutoff_date
            )
            cleanup_results['search_logs_deleted'] = old_search_logs.count()
            old_search_logs.delete()
            
            # Clean temporary files
            temp_dirs = ['uploads', 'logs']
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for filename in os.listdir(temp_dir):
                        filepath = os.path.join(temp_dir, filename)
                        if os.path.isfile(filepath):
                            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                            if file_time < cutoff_date:
                                try:
                                    os.remove(filepath)
                                    cleanup_results['temp_files_deleted'] += 1
                                except:
                                    pass
            
            # Clean up failed/error indexes
            failed_indexes = Index.query.filter_by(status='error')
            cleanup_results['old_indexes_cleaned'] = failed_indexes.count()
            for index in failed_indexes:
                # Remove index files
                if index.index_file_path and os.path.exists(index.index_file_path):
                    try:
                        os.remove(index.index_file_path)
                    except:
                        pass
                if index.metadata_file_path and os.path.exists(index.metadata_file_path):
                    try:
                        os.remove(index.metadata_file_path)
                    except:
                        pass
            failed_indexes.delete()
            
            db.session.commit()
            
            return {
                'success': True,
                'cleanup_results': cleanup_results,
                'cutoff_date': cutoff_date.isoformat()
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during cleanup: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance"""
        try:
            optimization_results = []
            
            # For SQLite, run VACUUM
            try:
                db.session.execute(text("VACUUM"))
                optimization_results.append("Database VACUUM completed")
            except Exception as e:
                optimization_results.append(f"VACUUM failed: {str(e)}")
            
            # Analyze tables
            try:
                db.session.execute(text("ANALYZE"))
                optimization_results.append("Database ANALYZE completed")
            except Exception as e:
                optimization_results.append(f"ANALYZE failed: {str(e)}")
            
            # Reindex
            try:
                db.session.execute(text("REINDEX"))
                optimization_results.append("Database REINDEX completed")
            except Exception as e:
                optimization_results.append(f"REINDEX failed: {str(e)}")
            
            db.session.commit()
            
            return {
                'success': True,
                'optimization_results': optimization_results
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error optimizing database: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def backup_database(self, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Create database backup"""
        try:
            if not backup_path:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                backup_path = f"backups/queryforge_backup_{timestamp}.db"
            
            # Create backup directory
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # For SQLite, copy the file
            db_file = 'queryforge.db'  # Adjust based on your config
            if os.path.exists(db_file):
                import shutil
                shutil.copy2(db_file, backup_path)
                
                backup_size = os.path.getsize(backup_path)
                
                return {
                    'success': True,
                    'backup_path': backup_path,
                    'backup_size_mb': round(backup_size / (1024**2), 2),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Database file not found'
                }
                
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics over time"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Search performance
            search_metrics = db.session.query(
                db.func.date(SearchLog.timestamp).label('date'),
                db.func.count(SearchLog.id).label('search_count'),
                db.func.avg(SearchLog.search_time_seconds).label('avg_search_time'),
                db.func.avg(SearchLog.result_count).label('avg_results')
            ).filter(
                SearchLog.timestamp >= cutoff_date
            ).group_by(
                db.func.date(SearchLog.timestamp)
            ).all()
            
            # Convert to list of dicts
            search_data = []
            for metric in search_metrics:
                search_data.append({
                    'date': str(metric.date),
                    'search_count': metric.search_count,
                    'avg_search_time': round(float(metric.avg_search_time or 0), 3),
                    'avg_results': round(float(metric.avg_results or 0), 1)
                })
            
            # Resource usage (current snapshot)
            system_metrics = self._get_system_metrics()
            
            return {
                'search_performance': search_data,
                'current_resources': system_metrics,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {
                'search_performance': [],
                'current_resources': {},
                'period_days': days,
                'error': str(e)
            }