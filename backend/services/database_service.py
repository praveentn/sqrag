# backend/services/database_service.py
"""
Database service for handling external database connections and operations
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from pathlib import Path

from backend.models import Source, Table, Column
from config import Config

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database connection and introspection"""
    
    def __init__(self):
        self.connectors = Config.DATABASE_CONNECTORS
        
    async def test_connection(
        self,
        db_type: str,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        schema_name: str = None
    ) -> Dict[str, Any]:
        """Test database connection"""
        
        try:
            connection_uri = self.build_connection_uri(
                db_type, host, port, database, username, password
            )
            
            # Test connection
            engine = create_engine(connection_uri, pool_pre_ping=True)
            with engine.connect() as conn:
                # Simple test query
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Get database info
                inspector = inspect(engine)
                
                db_info = {
                    "version": self._get_db_version(conn, db_type),
                    "schemas": inspector.get_schema_names() if hasattr(inspector, 'get_schema_names') else [],
                    "tables_count": len(inspector.get_table_names(schema=schema_name)),
                }
                
            engine.dispose()
            
            return {
                "success": True,
                "message": "Connection successful",
                "database_info": db_info
            }
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "details": {"error_type": type(e).__name__}
            }
    
    def build_connection_uri(
        self,
        db_type: str,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        **kwargs
    ) -> str:
        """Build database connection URI"""
        
        if db_type not in self.connectors:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        connector = self.connectors[db_type]
        driver = connector['driver']
        
        # Handle special cases
        if db_type == 'sqlite':
            return f"sqlite:///{database}"
        
        # Build standard connection string
        if port is None:
            port = connector['default_port']
        
        # URL encode password if it contains special characters
        from urllib.parse import quote_plus
        encoded_password = quote_plus(password) if password else ""
        encoded_username = quote_plus(username) if username else ""
        
        if db_type == 'mssql':
            # SQL Server requires special handling
            return f"{driver}://{encoded_username}:{encoded_password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        elif db_type == 'oracle':
            # Oracle connection string
            return f"{driver}://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
        else:
            # PostgreSQL, MySQL
            return f"{driver}://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
    
    async def introspect_schema_async(self, source_id: int, db_session):
        """Introspect database schema asynchronously"""
        
        try:
            from backend.database import async_session_factory
            
            async with async_session_factory() as db:
                # Get source
                from sqlalchemy import select
                source_query = select(Source).where(Source.id == source_id)
                source_result = await db.execute(source_query)
                source = source_result.scalar_one_or_none()
                
                if not source:
                    logger.error(f"Source {source_id} not found")
                    return
                
                # Update status
                source.status = source.status.__class__.CONNECTING
                source.update_ingest_progress(10.0, "Starting schema introspection")
                await db.commit()
                
                # Introspect schema
                schema_info = await self._introspect_database_schema(source)
                
                if schema_info['success']:
                    # Create tables and columns
                    await self._create_tables_from_schema(source, schema_info['schema'], db)
                    
                    # Update source
                    source.set_ready()
                    source.schema_metadata = schema_info['metadata']
                    source.total_tables = len(schema_info['schema']['tables'])
                    source.total_columns = sum(len(t['columns']) for t in schema_info['schema']['tables'])
                    
                else:
                    source.set_error(schema_info['error'])
                
                await db.commit()
                logger.info(f"Schema introspection completed for source {source_id}")
                
        except Exception as e:
            logger.error(f"Schema introspection failed for source {source_id}: {e}")
            
            # Update source to error state
            async with async_session_factory() as db:
                source_query = select(Source).where(Source.id == source_id)
                source_result = await db.execute(source_query)
                source = source_result.scalar_one_or_none()
                if source:
                    source.set_error(str(e))
                    await db.commit()
    
    async def _introspect_database_schema(self, source: Source) -> Dict[str, Any]:
        """Introspect database schema"""
        
        try:
            # Create engine
            engine = create_engine(source.connection_uri, pool_pre_ping=True)
            
            with engine.connect() as conn:
                inspector = inspect(engine)
                
                # Get schema name
                schema_name = None
                if source.connection_params:
                    schema_name = source.connection_params.get('schema_name')
                
                # Get tables
                table_names = inspector.get_table_names(schema=schema_name)
                
                tables = []
                total_tables = len(table_names)
                
                for i, table_name in enumerate(table_names):
                    try:
                        # Get table info
                        columns_info = inspector.get_columns(table_name, schema=schema_name)
                        primary_keys = inspector.get_pk_constraint(table_name, schema=schema_name)
                        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema_name)
                        indexes = inspector.get_indexes(table_name, schema=schema_name)
                        
                        # Process columns
                        columns = []
                        for col_info in columns_info:
                            column = {
                                'name': col_info['name'],
                                'data_type': str(col_info['type']),
                                'nullable': col_info['nullable'],
                                'default': str(col_info['default']) if col_info['default'] is not None else None,
                                'primary_key': col_info['name'] in (primary_keys.get('constrained_columns', []) if primary_keys else []),
                                'foreign_key': any(col_info['name'] in fk['constrained_columns'] for fk in foreign_keys),
                                'autoincrement': col_info.get('autoincrement', False),
                            }
                            columns.append(column)
                        
                        # Get row count (estimate)
                        try:
                            count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                            if schema_name:
                                count_query = text(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")
                            
                            count_result = conn.execute(count_query)
                            row_count = count_result.scalar()
                        except:
                            row_count = 0
                        
                        table_info = {
                            'name': table_name,
                            'schema': schema_name,
                            'columns': columns,
                            'row_count': row_count,
                            'primary_keys': primary_keys.get('constrained_columns', []) if primary_keys else [],
                            'foreign_keys': foreign_keys,
                            'indexes': indexes
                        }
                        
                        tables.append(table_info)
                        
                        # Update progress
                        progress = 20.0 + (70.0 * (i + 1) / total_tables)
                        source.update_ingest_progress(progress, f"Introspected table: {table_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to introspect table {table_name}: {e}")
                        continue
                
                # Get views
                try:
                    view_names = inspector.get_view_names(schema=schema_name)
                    views = []
                    
                    for view_name in view_names:
                        try:
                            columns_info = inspector.get_columns(view_name, schema=schema_name)
                            
                            columns = []
                            for col_info in columns_info:
                                column = {
                                    'name': col_info['name'],
                                    'data_type': str(col_info['type']),
                                    'nullable': col_info['nullable'],
                                }
                                columns.append(column)
                            
                            view_info = {
                                'name': view_name,
                                'schema': schema_name,
                                'columns': columns,
                                'type': 'view'
                            }
                            views.append(view_info)
                            
                        except Exception as e:
                            logger.warning(f"Failed to introspect view {view_name}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Failed to get views: {e}")
                    views = []
                
                schema_info = {
                    'tables': tables,
                    'views': views,
                    'schema_name': schema_name
                }
                
                metadata = {
                    'total_tables': len(tables),
                    'total_views': len(views),
                    'total_columns': sum(len(t['columns']) for t in tables),
                    'database_type': source.connection_params.get('db_type') if source.connection_params else 'unknown',
                    'introspection_date': str(pd.Timestamp.now())
                }
                
            engine.dispose()
            
            return {
                'success': True,
                'schema': schema_info,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Database schema introspection failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_tables_from_schema(
        self,
        source: Source,
        schema_info: Dict[str, Any],
        db_session
    ):
        """Create table and column records from schema info"""
        
        # Process tables
        for table_info in schema_info['tables']:
            # Create table record
            table = Table(
                source_id=source.id,
                name=table_info['name'],
                schema_name=table_info.get('schema'),
                row_count=table_info.get('row_count', 0),
                column_count=len(table_info['columns']),
                table_type='table',
                schema_json={
                    'primary_keys': table_info.get('primary_keys', []),
                    'foreign_keys': table_info.get('foreign_keys', []),
                    'indexes': table_info.get('indexes', [])
                }
            )
            
            db_session.add(table)
            await db_session.flush()  # Get table ID
            
            # Create column records
            for col_info in table_info['columns']:
                column = Column(
                    table_id=table.id,
                    name=col_info['name'],
                    data_type=col_info['data_type'],
                    is_nullable=col_info['nullable'],
                    is_primary_key=col_info['primary_key'],
                    is_foreign_key=col_info['foreign_key']
                )
                
                db_session.add(column)
        
        # Process views
        for view_info in schema_info.get('views', []):
            # Create table record for view
            table = Table(
                source_id=source.id,
                name=view_info['name'],
                schema_name=view_info.get('schema'),
                column_count=len(view_info['columns']),
                table_type='view'
            )
            
            db_session.add(table)
            await db_session.flush()
            
            # Create column records for view
            for col_info in view_info['columns']:
                column = Column(
                    table_id=table.id,
                    name=col_info['name'],
                    data_type=col_info['data_type'],
                    is_nullable=col_info['nullable']
                )
                
                db_session.add(column)
    
    async def preview_table_data(
        self,
        source: Source,
        table: Table,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Preview data from database table"""
        
        try:
            engine = create_engine(source.connection_uri, pool_pre_ping=True)
            
            with engine.connect() as conn:
                # Build query
                table_name = table.name
                if table.schema_name:
                    table_name = f"{table.schema_name}.{table.name}"
                
                query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
                
                # Execute query
                result = conn.execute(query)
                columns = list(result.keys())
                rows = result.fetchall()
                
                # Convert to list of dicts
                data = []
                for row in rows:
                    record = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Handle None values and convert to JSON-serializable types
                        if value is None:
                            record[col] = None
                        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                            record[col] = str(value)
                        else:
                            record[col] = value
                    data.append(record)
            
            engine.dispose()
            
            return {
                "table_name": table.name,
                "columns": columns,
                "data": data,
                "total_rows": table.row_count,
                "sample_size": len(data)
            }
            
        except Exception as e:
            logger.error(f"Preview failed for table {table.name}: {e}")
            return {
                "table_name": table.name,
                "columns": [],
                "data": [],
                "total_rows": 0,
                "sample_size": 0,
                "error": str(e)
            }
    
    def _get_db_version(self, connection, db_type: str) -> str:
        """Get database version"""
        
        try:
            if db_type == 'postgresql':
                result = connection.execute(text("SELECT version()"))
                return result.scalar()
            elif db_type == 'mysql':
                result = connection.execute(text("SELECT VERSION()"))
                return result.scalar()
            elif db_type == 'sqlite':
                result = connection.execute(text("SELECT sqlite_version()"))
                return f"SQLite {result.scalar()}"
            elif db_type == 'mssql':
                result = connection.execute(text("SELECT @@VERSION"))
                return result.scalar()
            elif db_type == 'oracle':
                result = connection.execute(text("SELECT * FROM v$version WHERE rownum = 1"))
                return result.scalar()
            else:
                return "Unknown"
        except:
            return "Unknown"

# Global service instance
database_service = DatabaseService()