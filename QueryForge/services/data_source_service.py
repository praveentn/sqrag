# services/data_source_service.py
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import json
import logging
import re
from typing import Dict, List, Any, Optional
import traceback
import numpy as np

from models import db, DataSource, Table, Column

logger = logging.getLogger(__name__)

class DataSourceService:
    """Service for handling data source operations"""
    
    def __init__(self):
        self.supported_file_types = {
            'csv': self._process_csv,
            'xlsx': self._process_excel,
            'xls': self._process_excel,
            'json': self._process_json
        }
    
    def process_uploaded_file(self, source_id: int, file_path: str) -> int:
        """Process uploaded file and create tables"""
        try:
            source = DataSource.query.get(source_id)
            if not source:
                raise ValueError(f"Data source {source_id} not found")
            
            file_extension = source.subtype.lower()
            
            if file_extension not in self.supported_file_types:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            logger.info(f"Processing {file_extension} file: {file_path}")
            
            # Process based on file type
            processor = self.supported_file_types[file_extension]
            tables_created = processor(source, file_path)
            
            logger.info(f"Successfully processed file {file_path}, created {tables_created} tables")
            return tables_created
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _process_csv(self, source: DataSource, file_path: str) -> int:
        """Process CSV file"""
        try:
            logger.info(f"Processing CSV file: {file_path}")
            
            # Try multiple encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    logger.info(f"Successfully read CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Error reading CSV with {encoding}: {str(e)}")
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")
            
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError("CSV file is empty")
            
            logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Clean column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Improve data type inference
            df = self._improve_data_types(df)
            
            # Create table name from filename
            table_name = self._clean_table_name(os.path.splitext(os.path.basename(file_path))[0])
            
            # Create table record
            table = Table(
                source_id=source.id,
                name=table_name,
                display_name=table_name.replace('_', ' ').title(),
                row_count=len(df),
                column_count=len(df.columns),
                schema_json=self._extract_schema(df),
                description=f"Imported from CSV file: {os.path.basename(file_path)}"
            )
            db.session.add(table)
            db.session.flush()  # Get table ID
            
            logger.info(f"Created table record: {table_name} (ID: {table.id})")
            
            # Create column records
            columns_created = 0
            for col_name in df.columns:
                try:
                    column = self._create_column_from_series(table.id, col_name, df[col_name])
                    db.session.add(column)
                    columns_created += 1
                    logger.debug(f"Created column: {col_name}")
                except Exception as e:
                    logger.warning(f"Error creating column {col_name}: {str(e)}")
            
            db.session.commit()
            logger.info(f"Successfully created table with {columns_created} columns")
            return 1
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing CSV file: {str(e)}")
            raise Exception(f"Error processing CSV file: {str(e)}")
    
    def _process_excel(self, source: DataSource, file_path: str) -> int:
        """Process Excel file (can have multiple sheets)"""
        try:
            logger.info(f"Processing Excel file: {file_path}")
            
            # Read all sheets
            try:
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            except Exception:
                # Try with xlrd for older Excel files
                try:
                    excel_file = pd.ExcelFile(file_path, engine='xlrd')
                except Exception as e:
                    raise ValueError(f"Could not read Excel file: {str(e)}")
            
            tables_created = 0
            base_name = self._clean_table_name(os.path.splitext(os.path.basename(file_path))[0])
            
            logger.info(f"Found {len(excel_file.sheet_names)} sheets: {excel_file.sheet_names}")
            
            for sheet_name in excel_file.sheet_names:
                try:
                    logger.info(f"Processing sheet: {sheet_name}")
                    
                    # Read sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name, engine=excel_file.engine)
                    
                    # Skip empty sheets
                    if df.empty:
                        logger.warning(f"Skipping empty sheet: {sheet_name}")
                        continue
                    
                    # Clean column names
                    df.columns = [self._clean_column_name(col) for col in df.columns]
                    
                    # Improve data type inference
                    df = self._improve_data_types(df)
                    
                    # Create table name from filename and sheet
                    if len(excel_file.sheet_names) > 1:
                        table_name = f"{base_name}_{self._clean_table_name(sheet_name)}"
                        display_name = f"{base_name} - {sheet_name}"
                    else:
                        table_name = base_name
                        display_name = base_name.replace('_', ' ').title()
                    
                    # Create table record
                    table = Table(
                        source_id=source.id,
                        name=table_name,
                        display_name=display_name,
                        row_count=len(df),
                        column_count=len(df.columns),
                        schema_json=self._extract_schema(df),
                        description=f"Imported from Excel sheet '{sheet_name}' in file: {os.path.basename(file_path)}"
                    )
                    db.session.add(table)
                    db.session.flush()
                    
                    logger.info(f"Created table: {table_name} (ID: {table.id}) with {len(df)} rows, {len(df.columns)} columns")
                    
                    # Create column records
                    columns_created = 0
                    for col_name in df.columns:
                        try:
                            column = self._create_column_from_series(table.id, col_name, df[col_name])
                            db.session.add(column)
                            columns_created += 1
                        except Exception as e:
                            logger.warning(f"Error creating column {col_name}: {str(e)}")
                    
                    logger.info(f"Created {columns_created} columns for table {table_name}")
                    tables_created += 1
                    
                except Exception as e:
                    logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
                    continue
            
            db.session.commit()
            logger.info(f"Successfully processed Excel file: created {tables_created} tables")
            return tables_created
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing Excel file: {str(e)}")
            raise Exception(f"Error processing Excel file: {str(e)}")
    
    def _process_json(self, source: DataSource, file_path: str) -> int:
        """Process JSON file"""
        try:
            logger.info(f"Processing JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                # If it's a dict, try to find the main data array
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        df = pd.json_normalize(value)
                        break
                else:
                    # If no array found, create single row DataFrame
                    df = pd.json_normalize([data])
            else:
                raise ValueError("JSON file format not supported")
            
            if df.empty:
                raise ValueError("JSON file contains no data")
            
            # Clean column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Improve data type inference
            df = self._improve_data_types(df)
            
            # Create table name from filename
            table_name = self._clean_table_name(os.path.splitext(os.path.basename(file_path))[0])
            
            # Create table record
            table = Table(
                source_id=source.id,
                name=table_name,
                display_name=table_name.replace('_', ' ').title(),
                row_count=len(df),
                column_count=len(df.columns),
                schema_json=self._extract_schema(df),
                description=f"Imported from JSON file: {os.path.basename(file_path)}"
            )
            db.session.add(table)
            db.session.flush()
            
            # Create column records
            for col_name in df.columns:
                column = self._create_column_from_series(table.id, col_name, df[col_name])
                db.session.add(column)
            
            db.session.commit()
            return 1
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing JSON file: {str(e)}")
            raise Exception(f"Error processing JSON file: {str(e)}")
    
    def _clean_table_name(self, name: str) -> str:
        """Clean table name to be database-friendly"""
        # Remove special characters and spaces
        name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure it starts with a letter
        if name and name[0].isdigit():
            name = 'table_' + name
        # Limit length
        return name[:50] if name else 'unnamed_table'
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name to be database-friendly"""
        # Handle unnamed columns
        if pd.isna(name) or str(name).strip() == '':
            return 'unnamed_column'
        
        name = str(name).strip()
        
        # Remove special characters and spaces
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure it starts with a letter or underscore
        if name and name[0].isdigit():
            name = 'col_' + name
        # Limit length
        return name[:50] if name else 'unnamed_column'
    
    def _improve_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improve pandas data type inference"""
        for col in df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Try to convert to numeric
            try:
                # First, handle common string representations of nulls
                df[col] = df[col].replace(['', 'null', 'NULL', 'None', 'NaN', 'nan'], pd.NA)
                
                # Try to convert to numeric
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if not numeric_col.isna().all():
                    # If we have some valid numbers, check if they're all integers
                    if numeric_col.dropna().apply(lambda x: x.is_integer()).all():
                        df[col] = numeric_col.astype('Int64')  # Nullable integer
                    else:
                        df[col] = numeric_col
                    continue
            except:
                pass
            
            # Try to convert to datetime
            try:
                datetime_col = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                if not datetime_col.isna().all():
                    df[col] = datetime_col
                    continue
            except:
                pass
            
            # Try to convert to boolean
            try:
                if df[col].dropna().str.lower().isin(['true', 'false', '1', '0', 'yes', 'no']).all():
                    bool_map = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False}
                    df[col] = df[col].str.lower().map(bool_map)
                    continue
            except:
                pass
        
        return df
    
    def _extract_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract schema information from DataFrame"""
        schema = {
            'columns': [],
            'data_types': {},
            'sample_data': {}
        }
        
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'memory_usage': int(df[col].memory_usage(deep=True))
            }
            
            # Add sample values (non-null, unique)
            sample_values = df[col].dropna().unique()[:5]
            col_info['sample_values'] = [str(v) for v in sample_values]
            
            schema['columns'].append(col_info)
            schema['data_types'][col] = str(df[col].dtype)
            schema['sample_data'][col] = col_info['sample_values']
        
        return schema
    
    def _create_column_from_series(self, table_id: int, col_name: str, series: pd.Series) -> Column:
        """Create Column object from pandas Series"""
        
        # Infer data type
        data_type = self._pandas_to_sql_type(series.dtype)
        
        # Calculate statistics
        null_count = int(series.isnull().sum())
        distinct_count = int(series.nunique())
        
        # Get sample values (non-null, unique, first 10)
        sample_values = []
        try:
            non_null_values = series.dropna().unique()[:10]
            sample_values = [str(v) for v in non_null_values if str(v) != 'nan']
        except:
            pass
        
        # Calculate min/max for numeric columns
        min_value = None
        max_value = None
        avg_value = None
        
        if pd.api.types.is_numeric_dtype(series):
            try:
                if not series.empty and not series.isna().all():
                    min_value = float(series.min())
                    max_value = float(series.max())
                    avg_value = float(series.mean())
                    
                    # Round to 3 decimal places
                    min_value = round(min_value, 3)
                    max_value = round(max_value, 3)
                    avg_value = round(avg_value, 3)
            except:
                pass
        
        # Detect PII
        pii_flag = self._detect_pii(col_name, sample_values)
        
        # Infer business category
        business_category = self._infer_business_category(col_name)
        
        return Column(
            table_id=table_id,
            name=col_name,
            display_name=col_name.replace('_', ' ').title(),
            data_type=data_type,
            is_nullable=null_count > 0,
            distinct_count=distinct_count,
            null_count=null_count,
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            sample_values=sample_values,
            pii_flag=pii_flag,
            business_category=business_category,
            description=f"Column from imported data: {data_type}"
        )
    
    def _pandas_to_sql_type(self, dtype) -> str:
        """Convert pandas dtype to SQL type string"""
        dtype_str = str(dtype).lower()
        
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'FLOAT'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        elif 'datetime' in dtype_str or 'timestamp' in dtype_str:
            return 'DATETIME'
        elif 'date' in dtype_str:
            return 'DATE'
        elif 'object' in dtype_str or 'string' in dtype_str:
            return 'TEXT'
        else:
            return 'TEXT'
    
    def _detect_pii(self, column_name: str, sample_values: List[str]) -> bool:
        """Detect if column might contain PII"""
        pii_keywords = [
            'email', 'phone', 'ssn', 'social', 'credit_card', 'password',
            'address', 'name', 'firstname', 'lastname', 'dob', 'birth'
        ]
        
        column_lower = column_name.lower()
        for keyword in pii_keywords:
            if keyword in column_lower:
                return True
        
        # Check sample values for patterns
        for value in sample_values[:5]:  # Check first 5 samples
            if self._looks_like_email(value) or self._looks_like_phone(value):
                return True
        
        return False
    
    def _looks_like_email(self, value: str) -> bool:
        """Check if value looks like an email"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, str(value)) is not None
    
    def _looks_like_phone(self, value: str) -> bool:
        """Check if value looks like a phone number"""
        phone_pattern = r'^[\+]?[\d\s\-\(\)]{10,}$'
        return re.match(phone_pattern, str(value)) is not None
    
    def _infer_business_category(self, column_name: str) -> str:
        """Infer business category from column name"""
        name_lower = column_name.lower()
        
        if any(word in name_lower for word in ['id', 'key', 'ref']):
            return 'identifier'
        elif any(word in name_lower for word in ['name', 'title', 'description']):
            return 'descriptive'
        elif any(word in name_lower for word in ['date', 'time', 'created', 'updated']):
            return 'temporal'
        elif any(word in name_lower for word in ['amount', 'price', 'cost', 'total', 'sum']):
            return 'financial'
        elif any(word in name_lower for word in ['count', 'quantity', 'number', 'size']):
            return 'quantitative'
        elif any(word in name_lower for word in ['status', 'state', 'flag', 'type']):
            return 'categorical'
        else:
            return 'general'
    
    def get_table_sample_data(self, table_id: int, limit: int = 100) -> Dict[str, Any]:
        """Get sample data from a table"""
        try:
            table = Table.query.get(table_id)
            if not table:
                raise ValueError(f"Table {table_id} not found")
            
            source = table.source
            
            if source.type == 'file':
                # For file sources, read from the original file
                return self._get_file_sample_data(source, table, limit)
            elif source.type == 'database':
                # For database sources, query the actual table
                return self._get_database_sample_data(source, table, limit)
            else:
                raise ValueError(f"Unsupported source type: {source.type}")
                
        except Exception as e:
            logger.error(f"Error getting sample data for table {table_id}: {str(e)}")
            raise
    
    def _get_file_sample_data(self, source: DataSource, table: Table, limit: int) -> Dict[str, Any]:
        """Get sample data from file source"""
        try:
            if source.subtype == 'csv':
                df = pd.read_csv(source.file_path, nrows=limit)
            elif source.subtype in ['xlsx', 'xls']:
                # For Excel files with multiple sheets, determine which sheet
                sheet_name = None
                if '_' in table.name:
                    # Try to extract sheet name from table name
                    parts = table.name.split('_')
                    if len(parts) > 1:
                        sheet_name = '_'.join(parts[1:])
                
                try:
                    df = pd.read_excel(source.file_path, sheet_name=sheet_name, nrows=limit)
                except:
                    # Fallback to first sheet
                    df = pd.read_excel(source.file_path, nrows=limit)
            else:
                raise ValueError(f"Unsupported file type for sampling: {source.subtype}")
            
            # Clean up the data for JSON serialization
            df = df.fillna('')  # Replace NaN with empty string
            
            # Convert to records
            data = []
            for _, row in df.iterrows():
                row_dict = {}
                for col, value in row.items():
                    # Handle different data types for JSON serialization
                    if pd.isna(value):
                        row_dict[col] = None
                    elif isinstance(value, (pd.Timestamp, np.datetime64)):
                        row_dict[col] = str(value)
                    elif isinstance(value, (np.integer, np.floating)):
                        row_dict[col] = float(value) if np.isfinite(value) else None
                    elif isinstance(value, bool):
                        row_dict[col] = bool(value)
                    else:
                        row_dict[col] = str(value)
                data.append(row_dict)
            
            return {
                'columns': df.columns.tolist(),
                'data': data,
                'total_rows': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error getting file sample data: {str(e)}")
            raise
    
    def _get_database_sample_data(self, source: DataSource, table: Table, limit: int) -> Dict[str, Any]:
        """Get sample data from database source"""
        connection_string = self._build_connection_string(source.connection_config)
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table.name} LIMIT {limit}"))
            columns = list(result.keys())
            data = [dict(row) for row in result]
        
        return {
            'columns': columns,
            'data': data,
            'total_rows': len(data)
        }
    
    def _build_connection_string(self, config: Dict[str, Any]) -> str:
        """Build database connection string"""
        db_type = config['type']
        host = config['host']
        port = config.get('port', 5432)
        database = config['database']
        username = config['username']
        password = config['password']
        
        if db_type == 'postgresql':
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'mysql':
            return f"mysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'sqlite':
            return f"sqlite:///{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def import_database_schema(self, source_id: int) -> int:
        """Import schema from database source"""
        try:
            source = DataSource.query.get(source_id)
            if not source:
                raise ValueError(f"Data source {source_id} not found")
            
            connection_string = self._build_connection_string(source.connection_config)
            engine = create_engine(connection_string)
            
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
            tables_created = 0
            
            for table_name in table_names:
                # Get table metadata
                columns_info = inspector.get_columns(table_name)
                pk_constraint = inspector.get_pk_constraint(table_name)
                fk_constraints = inspector.get_foreign_keys(table_name)
                
                # Get row count
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = result.scalar()
                
                # Create table record
                table = Table(
                    source_id=source.id,
                    name=table_name,
                    display_name=table_name.replace('_', ' ').title(),
                    row_count=row_count,
                    column_count=len(columns_info),
                    schema_json={
                        'columns': columns_info,
                        'primary_keys': pk_constraint.get('constrained_columns', []),
                        'foreign_keys': fk_constraints
                    }
                )
                db.session.add(table)
                db.session.flush()
                
                # Create column records
                for col_info in columns_info:
                    column = Column(
                        table_id=table.id,
                        name=col_info['name'],
                        display_name=col_info['name'].replace('_', ' ').title(),
                        data_type=str(col_info['type']),
                        is_nullable=col_info.get('nullable', True),
                        is_primary_key=col_info['name'] in pk_constraint.get('constrained_columns', []),
                        is_foreign_key=any(col_info['name'] in fk['constrained_columns'] 
                                         for fk in fk_constraints),
                        pii_flag=self._detect_pii(col_info['name'], []),
                        business_category=self._infer_business_category(col_info['name'])
                    )
                    db.session.add(column)
                
                tables_created += 1
            
            db.session.commit()
            return tables_created
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error importing database schema: {str(e)}")
            raise