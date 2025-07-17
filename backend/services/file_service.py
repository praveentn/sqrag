# backend/services/file_service.py
"""
File ingestion service for handling CSV/Excel uploads
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models import Source, Table, Column, SourceStatus
from backend.database import async_session_factory
from config import Config

logger = logging.getLogger(__name__)

class FileIngestService:
    """Service for ingesting file data into database"""
    
    def __init__(self):
        self.supported_extensions = ['.csv', '.xlsx', '.xls', '.json']
        self.chunk_size = 10000  # Process files in chunks
    
    async def ingest_file_async(
        self,
        source_id: int,
        file_path: str,
        db_session: AsyncSession = None
    ):
        """Asynchronously ingest file data"""
        
        try:
            # Use provided session or create new one
            if db_session:
                await self._ingest_file_internal(source_id, file_path, db_session)
            else:
                async with async_session_factory() as session:
                    await self._ingest_file_internal(source_id, file_path, session)
                    
        except Exception as e:
            logger.error(f"File ingestion failed for source {source_id}: {e}")
            # Update source status to error
            async with async_session_factory() as session:
                source_query = select(Source).where(Source.id == source_id)
                result = await session.execute(source_query)
                source = result.scalar_one_or_none()
                if source:
                    source.set_error(str(e))
                    await session.commit()
    
    async def _ingest_file_internal(
        self,
        source_id: int,
        file_path: str,
        db: AsyncSession
    ):
        """Internal file ingestion logic"""
        
        # Get source
        source_query = select(Source).where(Source.id == source_id)
        result = await db.execute(source_query)
        source = result.scalar_one_or_none()
        
        if not source:
            raise ValueError(f"Source {source_id} not found")
        
        # Update status
        source.status = SourceStatus.INGESTING
        source.ingest_status = "started"
        source.update_ingest_progress(0.0, "Starting file ingestion")
        await db.commit()
        
        try:
            # Detect file type and read data
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                await self._ingest_csv(source, file_path, db)
            elif file_extension in ['.xlsx', '.xls']:
                await self._ingest_excel(source, file_path, db)
            elif file_extension == '.json':
                await self._ingest_json(source, file_path, db)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Mark as ready
            source.set_ready()
            await db.commit()
            
            logger.info(f"Successfully ingested file for source {source_id}")
            
        except Exception as e:
            source.set_error(str(e))
            await db.commit()
            raise
    
    async def _ingest_csv(
        self,
        source: Source,
        file_path: str,
        db: AsyncSession
    ):
        """Ingest CSV file"""
        
        source.update_ingest_progress(10.0, "Reading CSV file")
        await db.commit()
        
        try:
            # Read CSV with pandas
            df = pd.read_csv(file_path)
            
            # Create table name from filename
            table_name = Path(file_path).stem
            
            # Create single table for CSV
            await self._create_table_from_dataframe(
                source, table_name, df, db
            )
            
            source.total_tables = 1
            source.total_rows = len(df)
            source.update_ingest_progress(100.0, "CSV ingestion completed")
            
        except Exception as e:
            raise Exception(f"CSV ingestion failed: {e}")
    
    async def _ingest_excel(
        self,
        source: Source,
        file_path: str,
        db: AsyncSession
    ):
        """Ingest Excel file"""
        
        source.update_ingest_progress(10.0, "Reading Excel file")
        await db.commit()
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            source.update_ingest_progress(20.0, f"Found {len(sheet_names)} sheets")
            await db.commit()
            
            total_rows = 0
            for i, sheet_name in enumerate(sheet_names):
                # Read sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Create table for each sheet
                table_name = f"{Path(file_path).stem}_{sheet_name}"
                await self._create_table_from_dataframe(
                    source, table_name, df, db
                )
                
                total_rows += len(df)
                
                # Update progress
                progress = 20.0 + (70.0 * (i + 1) / len(sheet_names))
                source.update_ingest_progress(
                    progress, 
                    f"Processed sheet: {sheet_name}"
                )
                await db.commit()
            
            source.total_tables = len(sheet_names)
            source.total_rows = total_rows
            source.update_ingest_progress(100.0, "Excel ingestion completed")
            
        except Exception as e:
            raise Exception(f"Excel ingestion failed: {e}")
    
    async def _ingest_json(
        self,
        source: Source,
        file_path: str,
        db: AsyncSession
    ):
        """Ingest JSON file"""
        
        source.update_ingest_progress(10.0, "Reading JSON file")
        await db.commit()
        
        try:
            # Read JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects -> single table
                df = pd.json_normalize(data)
                table_name = Path(file_path).stem
                await self._create_table_from_dataframe(
                    source, table_name, df, db
                )
                source.total_tables = 1
                source.total_rows = len(df)
                
            elif isinstance(data, dict):
                # Object with multiple arrays -> multiple tables
                table_count = 0
                total_rows = 0
                
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        df = pd.json_normalize(value)
                        table_name = f"{Path(file_path).stem}_{key}"
                        await self._create_table_from_dataframe(
                            source, table_name, df, db
                        )
                        table_count += 1
                        total_rows += len(df)
                
                source.total_tables = table_count
                source.total_rows = total_rows
            
            source.update_ingest_progress(100.0, "JSON ingestion completed")
            
        except Exception as e:
            raise Exception(f"JSON ingestion failed: {e}")
    
    async def _create_table_from_dataframe(
        self,
        source: Source,
        table_name: str,
        df: pd.DataFrame,
        db: AsyncSession
    ):
        """Create table and columns from pandas DataFrame"""
        
        # Clean table name
        clean_table_name = self._clean_name(table_name)
        
        # Create table record
        table = Table(
            source_id=source.id,
            name=clean_table_name,
            display_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            table_type="table"
        )
        
        db.add(table)
        await db.flush()  # Get table ID
        
        # Create columns
        for col_name in df.columns:
            column_info = await self._analyze_column(df, col_name)
            
            column = Column(
                table_id=table.id,
                name=self._clean_name(col_name),
                display_name=col_name,
                data_type=column_info['data_type'],
                max_length=column_info.get('max_length'),
                is_nullable=column_info['is_nullable'],
                unique_count=column_info['unique_count'],
                null_count=column_info['null_count'],
                min_value=column_info.get('min_value'),
                max_value=column_info.get('max_value'),
                sample_values=column_info['sample_values']
            )
            
            # Detect PII
            column.detect_pii(column_info['sample_values'])
            
            db.add(column)
        
        # Update table column count
        table.column_count = len(df.columns)
        
        # Store schema as JSON
        schema_json = {
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "nullable": bool(df[col].isnull().any())
                }
                for col in df.columns
            ],
            "shape": list(df.shape),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        table.schema_json = schema_json
        
        await db.commit()
        
        logger.info(f"Created table '{clean_table_name}' with {len(df.columns)} columns")
    
    async def _analyze_column(
        self,
        df: pd.DataFrame,
        column_name: str
    ) -> Dict[str, Any]:
        """Analyze column data and determine metadata"""
        
        series = df[column_name]
        
        # Basic statistics
        null_count = series.isnull().sum()
        unique_count = series.nunique()
        total_count = len(series)
        
        # Data type mapping
        dtype = str(series.dtype)
        sql_type = self._map_pandas_to_sql_type(series)
        
        # Sample values (non-null)
        sample_values = []
        non_null_series = series.dropna()
        if len(non_null_series) > 0:
            sample_size = min(10, len(non_null_series))
            sample_values = non_null_series.sample(n=sample_size, random_state=42).tolist()
            # Convert to strings for storage
            sample_values = [str(v) for v in sample_values]
        
        # Min/max for numeric columns
        min_value = None
        max_value = None
        max_length = None
        
        if pd.api.types.is_numeric_dtype(series):
            if len(non_null_series) > 0:
                min_value = str(non_null_series.min())
                max_value = str(non_null_series.max())
        elif pd.api.types.is_string_dtype(series):
            if len(non_null_series) > 0:
                string_lengths = non_null_series.astype(str).str.len()
                max_length = int(string_lengths.max())
                min_value = str(string_lengths.min())
                max_value = str(string_lengths.max())
        
        return {
            "data_type": sql_type,
            "max_length": max_length,
            "is_nullable": null_count > 0,
            "unique_count": int(unique_count),
            "null_count": int(null_count),
            "min_value": min_value,
            "max_value": max_value,
            "sample_values": sample_values,
            "pandas_dtype": dtype
        }
    
    def _map_pandas_to_sql_type(self, series: pd.Series) -> str:
        """Map pandas dtype to SQL type"""
        
        dtype = series.dtype
        
        if pd.api.types.is_integer_dtype(dtype):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            return "FLOAT"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        elif pd.api.types.is_object_dtype(dtype):
            # Check if it's actually numeric stored as object
            try:
                pd.to_numeric(series.dropna(), errors='raise')
                return "FLOAT"
            except (ValueError, TypeError):
                pass
            
            # Check if it's datetime stored as object
            try:
                pd.to_datetime(series.dropna(), errors='raise')
                return "TIMESTAMP"
            except (ValueError, TypeError):
                pass
            
            # Default to text
            max_length = series.astype(str).str.len().max()
            if max_length and max_length > 255:
                return "TEXT"
            else:
                return f"VARCHAR({max_length or 255})"
        else:
            return "TEXT"
    
    def _clean_name(self, name: str) -> str:
        """Clean name for database usage"""
        
        # Replace spaces and special characters with underscores
        import re
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        
        # Remove consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        # Ensure it starts with a letter
        if cleaned and cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"
        
        # Handle empty names
        if not cleaned:
            cleaned = "unnamed_column"
        
        return cleaned.lower()
    
    async def preview_table_data(
        self,
        source: Source,
        table: Table,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Preview data from file-based table"""
        
        try:
            # Read the original file
            file_path = source.file_path
            if not file_path or not Path(file_path).exists():
                raise ValueError("Source file not found")
            
            file_extension = Path(file_path).suffix.lower()
            
            # Read data based on file type
            if file_extension == '.csv':
                df = pd.read_csv(file_path, nrows=limit)
            elif file_extension in ['.xlsx', '.xls']:
                # Extract sheet name from table name
                sheet_name = table.name.replace(f"{Path(file_path).stem}_", "")
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=limit)
            elif file_extension == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    df = pd.json_normalize(data[:limit])
                elif isinstance(data, dict):
                    # Find the relevant array
                    key = table.name.replace(f"{Path(file_path).stem}_", "")
                    if key in data and isinstance(data[key], list):
                        df = pd.json_normalize(data[key][:limit])
                    else:
                        raise ValueError(f"Unable to find data for table {table.name}")
            else:
                raise ValueError(f"Unsupported file type for preview: {file_extension}")
            
            # Convert to records
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    # Handle NaN values
                    if pd.isna(value):
                        record[col] = None
                    else:
                        record[col] = value
                records.append(record)
            
            return {
                "table_name": table.name,
                "columns": list(df.columns),
                "data": records,
                "total_rows": table.row_count,
                "sample_size": len(records)
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
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information"""
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            "filename": path.name,
            "size": path.stat().st_size,
            "extension": path.suffix.lower(),
            "modified": path.stat().st_mtime
        }
        
        # Quick peek at file structure
        try:
            if info["extension"] == ".csv":
                df = pd.read_csv(file_path, nrows=5)
                info["preview"] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample_data": df.head(3).to_dict('records')
                }
            elif info["extension"] in [".xlsx", ".xls"]:
                excel_file = pd.ExcelFile(file_path)
                info["preview"] = {
                    "sheets": excel_file.sheet_names,
                    "sheet_count": len(excel_file.sheet_names)
                }
        except Exception as e:
            info["preview_error"] = str(e)
        
        return info