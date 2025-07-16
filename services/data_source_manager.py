# services/data_source_manager.py
import pandas as pd
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine, inspect, text
import os
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import uuid

from models import db, DataSource, Table, Column
from config import Config

logger = logging.getLogger(__name__)

class DataSourceManager:
    """Manages data source connections and _metadata extraction"""
    
    def __init__(self):
        self.config = Config()
        self.supported_types = ['csv', 'excel', 'postgresql', 'mysql', 'sqlite', 'mssql']
        
    def create_source(self, name: str, type: str, connection_string: str = '', 
                     file_path: str = '', _metadata: Dict = None) -> DataSource:
        """Create a new data source"""
        try:
            # Validate type
            if type not in self.supported_types:
                raise ValueError(f"Unsupported source type: {type}")
            
            # Create data source record
            source = DataSource(
                name=name,
                type=type,
                connection_string=connection_string,
                file_path=file_path,
                _metadata=_metadata or {},
                created_at=datetime.utcnow()
            )
            
            db.session.add(source)
            db.session.commit()
            
            # Extract _metadata immediately
            self.refresh__metadata(source)
            
            logger.info(f"Created data source: {name} (type: {type})")
            return source
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating data source {name}: {str(e)}")
            raise
    
    def test_connection(self, source: DataSource) -> Dict[str, Any]:
        """Test connection to a data source"""
        try:
            if source.type in ['csv', 'excel']:
                return self._test_file_connection(source)
            else:
                return self._test_database_connection(source)
                
        except Exception as e:
            logger.error(f"Connection test failed for {source.name}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _test_file_connection(self, source: DataSource) -> Dict[str, Any]:
        """Test file-based data source"""
        file_path = source.file_path
        
        if not file_path or not os.path.exists(file_path):
            return {
                'status': 'error',
                'message': f'File not found: {file_path}',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        try:
            if source.type == 'csv':
                # Test CSV reading
                df = pd.read_csv(file_path, nrows=5)
                row_count = len(pd.read_csv(file_path))
            elif source.type == 'excel':
                # Test Excel reading
                xl_file = pd.ExcelFile(file_path)
                sheet_names = xl_file.sheet_names
                df = pd.read_excel(file_path, sheet_name=sheet_names[0], nrows=5)
                row_count = len(pd.read_excel(file_path, sheet_name=sheet_names[0]))
            
            return {
                'status': 'success',
                'message': 'Connection successful',
                'row_count': row_count,
                'columns': df.columns.tolist(),
                'sample_data': df.to_dict('records'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to read file: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _test_database_connection(self, source: DataSource) -> Dict[str, Any]:
        """Test database connection"""
        try:
            engine = create_engine(source.connection_string)
            
            # Test connection with a simple query
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
            
            # Get basic database info
            inspector = inspect(engine)
            schemas = inspector.get_schema_names()
            
            return {
                'status': 'success',
                'message': 'Connection successful',
                'schemas': schemas[:10],  # Limit to first 10
                'database_type': engine.dialect.name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Database connection failed: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def refresh__metadata(self, source: DataSource) -> None:
        """Refresh _metadata for a data source"""
        try:
            if source.type in ['csv', 'excel']:
                self._extract_file__metadata(source)
            else:
                self._extract_database__metadata(source)
            
            source.last_refresh = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"Refreshed _metadata for source: {source.name}")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error refreshing _metadata for {source.name}: {str(e)}")
            raise
    
    def _extract_file__metadata(self, source: DataSource) -> None:
        """Extract _metadata from file-based sources"""
        if source.type == 'csv':
            self._extract_csv__metadata(source)
        elif source.type == 'excel':
            self._extract_excel__metadata(source)
    
    def _extract_csv__metadata(self, source: DataSource) -> None:
        """Extract _metadata from CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(source.file_path)
            
            # Create or update table record
            table_name = os.path.splitext(os.path.basename(source.file_path))[0]
            table = Table.query.filter_by(source_id=source.id, name=table_name).first()
            
            if not table:
                table = Table(
                    name=table_name,
                    display_name=table_name.replace('_', ' ').title(),
                    source_id=source.id,
                    created_at=datetime.utcnow()
                )
                db.session.add(table)
                db.session.flush()  # Get table ID
            
            table.row_count = len(df)
            table.updated_at = datetime.utcnow()
            
            # Update source row count
            source.row_count = len(df)
            
            # Extract column _metadata
            self._extract_columns_from_dataframe(table, df)
            
        except Exception as e:
            logger.error(f"Error extracting CSV _metadata: {str(e)}")
            raise
    
    def _extract_excel__metadata(self, source: DataSource) -> None:
        """Extract _metadata from Excel file"""
        try:
            xl_file = pd.ExcelFile(source.file_path)
            total_rows = 0
            
            for sheet_name in xl_file.sheet_names:
                # Read each sheet as a separate table
                df = pd.read_excel(source.file_path, sheet_name=sheet_name)
                
                # Create or update table record
                table = Table.query.filter_by(source_id=source.id, name=sheet_name).first()
                
                if not table:
                    table = Table(
                        name=sheet_name,
                        display_name=sheet_name.replace('_', ' ').title(),
                        source_id=source.id,
                        created_at=datetime.utcnow()
                    )
                    db.session.add(table)
                    db.session.flush()  # Get table ID
                
                table.row_count = len(df)
                table.updated_at = datetime.utcnow()
                total_rows += len(df)
                
                # Extract column _metadata
                self._extract_columns_from_dataframe(table, df)
            
            # Update source row count
            source.row_count = total_rows
            
        except Exception as e:
            logger.error(f"Error extracting Excel _metadata: {str(e)}")
            raise
    
    def _extract_database__metadata(self, source: DataSource) -> None:
        """Extract _metadata from database source"""
        try:
            engine = create_engine(source.connection_string)
            inspector = inspect(engine)
            
            total_rows = 0
            
            # Get all tables
            for schema_name in inspector.get_schema_names():
                table_names = inspector.get_table_names(schema=schema_name)
                
                for table_name in table_names:
                    # Create or update table record
                    full_table_name = f"{schema_name}.{table_name}" if schema_name != 'public' else table_name
                    table = Table.query.filter_by(source_id=source.id, name=full_table_name).first()
                    
                    if not table:
                        table = Table(
                            name=full_table_name,
                            display_name=table_name.replace('_', ' ').title(),
                            schema_name=schema_name,
                            source_id=source.id,
                            created_at=datetime.utcnow()
                        )
                        db.session.add(table)
                        db.session.flush()  # Get table ID
                    
                    table.updated_at = datetime.utcnow()
                    
                    # Get row count
                    try:
                        with engine.connect() as conn:
                            result = conn.execute(text(f"SELECT COUNT(*) FROM {full_table_name}"))
                            row_count = result.fetchone()[0]
                            table.row_count = row_count
                            total_rows += row_count
                    except:
                        table.row_count = 0
                    
                    # Extract column _metadata
                    self._extract_database_columns(table, inspector, table_name, schema_name, engine)
            
            # Update source row count
            source.row_count = total_rows
            
        except Exception as e:
            logger.error(f"Error extracting database _metadata: {str(e)}")
            raise
    
    def _extract_columns_from_dataframe(self, table: Table, df: pd.DataFrame) -> None:
        """Extract column _metadata from pandas DataFrame"""
        for col_name in df.columns:
            # Create or update column record
            column = Column.query.filter_by(table_id=table.id, name=col_name).first()
            
            if not column:
                column = Column(
                    name=col_name,
                    display_name=col_name.replace('_', ' ').title(),
                    table_id=table.id,
                    created_at=datetime.utcnow()
                )
                db.session.add(column)
            
            # Infer data type
            column.data_type = str(df[col_name].dtype)
            column.is_nullable = df[col_name].isnull().any()
            column.null_count = int(df[col_name].isnull().sum())
            column.unique_count = int(df[col_name].nunique())
            column.updated_at = datetime.utcnow()
            
            # Get sample values (non-null, unique values)
            sample_values = df[col_name].dropna().unique()
            if len(sample_values) > 10:
                sample_values = sample_values[:10]
            column.sample_values = [str(val) for val in sample_values.tolist()]
    
    def _extract_database_columns(self, table: Table, inspector, table_name: str, 
                                 schema_name: str, engine) -> None:
        """Extract column _metadata from database table"""
        columns_info = inspector.get_columns(table_name, schema=schema_name)
        pk_columns = inspector.get_pk_constraint(table_name, schema=schema_name)
        fk_columns = inspector.get_foreign_keys(table_name, schema=schema_name)
        
        pk_column_names = pk_columns.get('constrained_columns', [])
        fk_column_names = [fk['constrained_columns'][0] for fk in fk_columns 
                          if fk.get('constrained_columns')]
        
        for col_info in columns_info:
            col_name = col_info['name']
            
            # Create or update column record
            column = Column.query.filter_by(table_id=table.id, name=col_name).first()
            
            if not column:
                column = Column(
                    name=col_name,
                    display_name=col_name.replace('_', ' ').title(),
                    table_id=table.id,
                    created_at=datetime.utcnow()
                )
                db.session.add(column)
            
            column.data_type = str(col_info.get('type', 'unknown'))
            column.is_nullable = col_info.get('nullable', True)
            column.is_primary_key = col_name in pk_column_names
            column.is_foreign_key = col_name in fk_column_names
            column.updated_at = datetime.utcnow()
            
            # Get sample values
            try:
                full_table_name = f"{schema_name}.{table_name}" if schema_name != 'public' else table_name
                with engine.connect() as conn:
                    # Get distinct values (limited)
                    query = text(f"SELECT DISTINCT {col_name} FROM {full_table_name} WHERE {col_name} IS NOT NULL LIMIT 10")
                    result = conn.execute(query)
                    sample_values = [str(row[0]) for row in result.fetchall()]
                    column.sample_values = sample_values
                    
                    # Get unique count
                    count_query = text(f"SELECT COUNT(DISTINCT {col_name}) FROM {full_table_name}")
                    count_result = conn.execute(count_query)
                    column.unique_count = count_result.fetchone()[0]
                    
                    # Get null count
                    null_query = text(f"SELECT COUNT(*) FROM {full_table_name} WHERE {col_name} IS NULL")
                    null_result = conn.execute(null_query)
                    column.null_count = null_result.fetchone()[0]
                    
            except Exception as e:
                logger.warning(f"Could not get sample values for {col_name}: {str(e)}")
                column.sample_values = []
                column.unique_count = 0
                column.null_count = 0
    
    def get_sample_data(self, source: DataSource, table_name: str = None, limit: int = 100) -> Dict[str, Any]:
        """Get sample data from source"""
        try:
            if source.type == 'csv':
                df = pd.read_csv(source.file_path, nrows=limit)
                return {
                    'columns': df.columns.tolist(),
                    'data': df.to_dict('records'),
                    'total_rows': len(pd.read_csv(source.file_path))
                }
            
            elif source.type == 'excel':
                if table_name:
                    df = pd.read_excel(source.file_path, sheet_name=table_name, nrows=limit)
                    total_rows = len(pd.read_excel(source.file_path, sheet_name=table_name))
                else:
                    xl_file = pd.ExcelFile(source.file_path)
                    sheet_name = xl_file.sheet_names[0]
                    df = pd.read_excel(source.file_path, sheet_name=sheet_name, nrows=limit)
                    total_rows = len(pd.read_excel(source.file_path, sheet_name=sheet_name))
                
                return {
                    'columns': df.columns.tolist(),
                    'data': df.to_dict('records'),
                    'total_rows': total_rows
                }
            
            else:
                # Database source
                engine = create_engine(source.connection_string)
                table_to_query = table_name or source.tables[0].name
                
                with engine.connect() as conn:
                    # Get sample data
                    query = text(f"SELECT * FROM {table_to_query} LIMIT {limit}")
                    result = conn.execute(query)
                    columns = result.keys()
                    data = [dict(row) for row in result.fetchall()]
                    
                    # Get total count
                    count_query = text(f"SELECT COUNT(*) FROM {table_to_query}")
                    count_result = conn.execute(count_query)
                    total_rows = count_result.fetchone()[0]
                
                return {
                    'columns': list(columns),
                    'data': data,
                    'total_rows': total_rows
                }
                
        except Exception as e:
            logger.error(f"Error getting sample data: {str(e)}")
            raise
    
    def delete_source(self, source_id: int) -> None:
        """Delete a data source and all its _metadata"""
        try:
            source = DataSource.query.get_or_404(source_id)
            
            # Cascade delete will handle tables and columns
            db.session.delete(source)
            db.session.commit()
            
            logger.info(f"Deleted data source: {source.name}")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting data source {source_id}: {str(e)}")
            raise
    
    def update_table_description(self, table_id: int, description: str) -> None:
        """Update table description"""
        try:
            table = Table.query.get_or_404(table_id)
            table.description = description
            table.updated_at = datetime.utcnow()
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating table description: {str(e)}")
            raise
    
    def update_column_description(self, column_id: int, description: str) -> None:
        """Update column description"""
        try:
            column = Column.query.get_or_404(column_id)
            column.description = description
            column.updated_at = datetime.utcnow()
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating column description: {str(e)}")
            raise
    
    def get_source_statistics(self, source_id: int) -> Dict[str, Any]:
        """Get comprehensive statistics for a data source"""
        try:
            source = DataSource.query.get_or_404(source_id)
            tables = Table.query.filter_by(source_id=source_id).all()
            
            stats = {
                'source_name': source.name,
                'source_type': source.type,
                'total_tables': len(tables),
                'total_rows': sum(table.row_count or 0 for table in tables),
                'total_columns': sum(len(table.columns) for table in tables),
                'last_refresh': source.last_refresh.isoformat() if source.last_refresh else None,
                'tables': []
            }
            
            for table in tables:
                table_stats = {
                    'name': table.name,
                    'display_name': table.display_name,
                    'row_count': table.row_count,
                    'column_count': len(table.columns),
                    'has_description': bool(table.description),
                    'columns': [
                        {
                            'name': col.name,
                            'data_type': col.data_type,
                            'is_nullable': col.is_nullable,
                            'unique_count': col.unique_count,
                            'has_description': bool(col.description)
                        }
                        for col in table.columns
                    ]
                }
                stats['tables'].append(table_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting source statistics: {str(e)}")
            raise