# services/data_source_service.py
import os
import pandas as pd
import numpy as np
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from werkzeug.utils import secure_filename

from models import db, Project, DataSource, Table, Column
from sqlalchemy import create_engine, text, inspect
from config import Config

logger = logging.getLogger(__name__)

class DataSourceService:
    """Service for handling data source operations"""
    
    def __init__(self):
        self.logger = logger
        self.upload_folder = Config.UPLOAD_FOLDER
        self.allowed_extensions = Config.ALLOWED_EXTENSIONS
        
        # Ensure upload directory exists
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def process_uploaded_file(self, file, project_id: int) -> Dict[str, Any]:
        """Process an uploaded file and create data source"""
        try:
            # Validate file
            if not self._allowed_file(file.filename):
                return {
                    'success': False,
                    'error': f'File type not allowed. Allowed: {", ".join(self.allowed_extensions)}'
                }
            
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(self.upload_folder, filename)
            file.save(file_path)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_extension = filename.split('.')[-1].lower()
            
            # Create data source record
            source = DataSource(
                project_id=project_id,
                name=file.filename,  # Original filename
                type='file',
                subtype=file_extension,
                connection_string=file_path,
                file_size_bytes=file_size,
                ingest_status='processing'
            )
            db.session.add(source)
            db.session.flush()  # Get ID
            
            # Process the file based on type
            if file_extension in ['csv', 'txt']:
                tables_created = self._process_csv_file(source, file_path)
            elif file_extension in ['xlsx', 'xls']:
                tables_created = self._process_excel_file(source, file_path)
            elif file_extension == 'json':
                tables_created = self._process_json_file(source, file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Update source status
            source.ingest_status = 'completed'
            db.session.commit()
            
            # Get table info for response
            table = Table.query.filter_by(source_id=source.id).first()
            columns_created = Column.query.filter_by(table_id=table.id).count() if table else 0
            
            # Generate preview data
            preview_data = self._generate_preview(file_path, file_extension)
            
            return {
                'success': True,
                'source': {
                    'id': source.id,
                    'name': source.name,
                    'type': source.type,
                    'ingest_status': source.ingest_status
                },
                'processing': {
                    'tables_created': tables_created,
                    'columns_created': columns_created
                },
                'table': {
                    'id': table.id,
                    'name': table.name,
                    'row_count': table.row_count,
                    'column_count': table.column_count
                },
                'preview': preview_data,
                'file_info': {
                    'size_bytes': os.path.getsize(file_path),
                    'size_mb': round(os.path.getsize(file_path) / (1024*1024), 2),
                    'extension': file_extension,
                    'shape': f"{len(preview_data)} rows × {len(preview_data[0]) if preview_data else 0} columns" if preview_data else "Unknown"
                }
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error processing file {file.filename}: {str(e)}")
            
            # Update source status to failed if it exists
            try:
                if 'source' in locals():
                    source.ingest_status = 'failed'
                    db.session.commit()
            except:
                pass
            
            return {
                'success': False,
                'error': f'Error processing file: {str(e)}'
            }
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _process_csv_file(self, source: DataSource, file_path: str) -> int:
        """Process CSV file and create table/columns"""
        try:
            # Read CSV with improved data type inference
            df = self._read_file_to_dataframe(file_path, 'csv')
            if df is None or df.empty:
                raise ValueError("Could not read CSV file or file is empty")
            
            # Improve data types
            df = self._improve_data_types(df)
            
            # Clean column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                raise ValueError("File contains no data after cleaning")
            
            # Create table
            table_name = self._clean_table_name(os.path.splitext(source.name)[0])
            table = Table(
                source_id=source.id,
                name=table_name,
                display_name=source.name,
                row_count=len(df),
                column_count=len(df.columns),
                description=f"Imported from {source.name}"
            )
            db.session.add(table)
            db.session.flush()
            
            # Create columns
            for col_name in df.columns:
                try:
                    column = self._create_column_from_series(table.id, col_name, df[col_name])
                    db.session.add(column)
                except Exception as col_error:
                    logger.warning(f"Error creating column {col_name}: {str(col_error)}")
                    continue
            
            db.session.commit()
            return 1
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing CSV file: {str(e)}")
            raise Exception(f"Error processing CSV file: {str(e)}")
    
    def _process_excel_file(self, source: DataSource, file_path: str) -> int:
        """Process Excel file and create table/columns"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            tables_created = 0
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    if df.empty:
                        continue
                    
                    # Improve data types
                    df = self._improve_data_types(df)
                    
                    # Clean column names
                    df.columns = [self._clean_column_name(col) for col in df.columns]
                    
                    # Remove completely empty rows and columns
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if df.empty:
                        continue
                    
                    # Create table
                    table_name = self._clean_table_name(f"{os.path.splitext(source.name)[0]}_{sheet_name}")
                    table = Table(
                        source_id=source.id,
                        name=table_name,
                        display_name=f"{source.name} - {sheet_name}",
                        row_count=len(df),
                        column_count=len(df.columns),
                        description=f"Sheet '{sheet_name}' from {source.name}"
                    )
                    db.session.add(table)
                    db.session.flush()
                    
                    # Create columns
                    for col_name in df.columns:
                        try:
                            column = self._create_column_from_series(table.id, col_name, df[col_name])
                            db.session.add(column)
                        except Exception as col_error:
                            logger.warning(f"Error creating column {col_name}: {str(col_error)}")
                            continue
                    
                    tables_created += 1
                    
                except Exception as sheet_error:
                    logger.warning(f"Error processing sheet {sheet_name}: {str(sheet_error)}")
                    continue
            
            db.session.commit()
            return tables_created
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing Excel file: {str(e)}")
            raise Exception(f"Error processing Excel file: {str(e)}")
    
    def _process_json_file(self, source: DataSource, file_path: str) -> int:
        """Process JSON file and create table/columns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                # Check if it's a single object or has arrays
                arrays = {k: v for k, v in data.items() if isinstance(v, list)}
                if arrays:
                    # Take the first array found
                    key, array_data = next(iter(arrays.items()))
                    df = pd.json_normalize(array_data)
                else:
                    # Single object, convert to single-row DataFrame
                    df = pd.json_normalize([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            if df.empty:
                raise ValueError("No data found in JSON file")
            
            # Improve data types
            df = self._improve_data_types(df)
            
            # Clean column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Create table
            table_name = self._clean_table_name(os.path.splitext(source.name)[0])
            table = Table(
                source_id=source.id,
                name=table_name,
                display_name=source.name,
                row_count=len(df),
                column_count=len(df.columns),
                description=f"Imported from {source.name}"
            )
            db.session.add(table)
            db.session.flush()
            
            # Create columns
            for col_name in df.columns:
                column = self._create_column_from_series(table.id, col_name, df[col_name])
                db.session.add(column)
            
            db.session.commit()
            return 1
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing JSON file: {str(e)}")
            raise Exception(f"Error processing JSON file: {str(e)}")
    
    def _read_file_to_dataframe(self, file_path: str, file_extension: str) -> Optional[pd.DataFrame]:
        """Read various file formats into DataFrame"""
        try:
            if file_extension in ['csv', 'txt']:
                # Try different separators and encodings
                separators = [',', ';', '\t', '|']
                encodings = ['utf-8', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, sep=sep, encoding=encoding, 
                                           low_memory=False, na_values=['', 'NULL', 'null', 'N/A', 'n/a'])
                            
                            # Check if this looks like a good parse
                            if len(df.columns) > 1 and len(df) > 0:
                                return df
                        except:
                            continue
                
                # Fallback: try with default settings
                return pd.read_csv(file_path, low_memory=False)
                
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(file_path)
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
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
                        # Convert to Int64 (nullable integer)
                        df[col] = numeric_col.astype('Int64')
                    else:
                        # Keep as float
                        df[col] = numeric_col
                    continue
            except:
                pass
            
            # Try to convert to datetime
            try:
                # Only try datetime conversion if there are some values that look like dates
                sample_values = df[col].dropna().astype(str).head(100)
                if any(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', str(val)) for val in sample_values):
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    if not datetime_series.isna().all():
                        df[col] = datetime_series
                        continue
            except:
                pass
        
        return df
    
    def _create_column_from_series(self, table_id: int, col_name: str, series: pd.Series) -> Column:
        """Create a Column object from a pandas Series"""
        # Basic statistics
        null_count = int(series.isnull().sum())
        distinct_count = int(series.nunique())
        total_count = len(series)
        
        # Data type detection
        data_type = self._determine_sql_data_type(series)
        
        # Sample values (non-null, unique, first 10)
        sample_values = []
        try:
            non_null_values = series.dropna().unique()[:10]
            sample_values = [str(v) for v in non_null_values if str(v) not in ['nan', 'NaT', 'None']]
        except Exception:
            pass
        
        # Statistical measures for numeric columns
        min_value = None
        max_value = None
        avg_value = None
        
        if pd.api.types.is_numeric_dtype(series):
            try:
                if not series.empty and not series.isna().all():
                    min_value = round(float(series.min()), 3)
                    max_value = round(float(series.max()), 3)
                    avg_value = round(float(series.mean()), 3)
            except Exception:
                pass
        
        # PII detection
        pii_flag = self._detect_pii(col_name, series)
        
        # Business context hints
        business_context = self._infer_business_context(col_name, series)
        
        # FIXED: Use correct column names that match the Column model
        return Column(
            table_id=table_id,
            name=col_name,
            data_type=data_type,
            is_nullable=null_count > 0,
            null_count=null_count,
            distinct_count=distinct_count,
            sample_values=sample_values,  # Store as list, not JSON string
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            pii_flag=pii_flag,  # FIXED: was is_pii
            business_category=business_context  # FIXED: was business_context
        )
    
    def _determine_sql_data_type(self, series: pd.Series) -> str:
        """Determine appropriate SQL data type"""
        if pd.api.types.is_integer_dtype(series):
            return 'INTEGER'
        elif pd.api.types.is_float_dtype(series):
            return 'FLOAT'
        elif pd.api.types.is_bool_dtype(series):
            return 'BOOLEAN'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'DATETIME'
        elif pd.api.types.is_categorical_dtype(series):
            return 'VARCHAR'
        else:
            # Check max length for strings
            try:
                max_length = series.astype(str).str.len().max()
                if max_length > 255:
                    return 'TEXT'
                else:
                    return 'VARCHAR'
            except:
                return 'TEXT'
    
    def _detect_pii(self, col_name: str, series: pd.Series) -> bool:
        """Detect potential PII in column"""
        import re
        
        col_lower = col_name.lower()
        
        # Common PII column names
        pii_keywords = [
            'email', 'phone', 'ssn', 'social', 'security', 'passport',
            'license', 'credit', 'card', 'account', 'routing', 'tax',
            'firstname', 'lastname', 'fullname', 'address', 'zip', 'postal'
        ]
        
        if any(keyword in col_lower for keyword in pii_keywords):
            return True
        
        # Pattern-based detection (sample a few values)
        try:
            sample_values = series.dropna().astype(str).head(50)
            
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if sample_values.str.match(email_pattern).any():
                return True
            
            # Phone pattern
            phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
            if sample_values.str.match(phone_pattern).any():
                return True
            
            # SSN pattern (US)
            ssn_pattern = r'\d{3}-?\d{2}-?\d{4}'
            if sample_values.str.match(ssn_pattern).any():
                return True
                
        except Exception:
            pass
        
        return False
    
    def _infer_business_context(self, col_name: str, series: pd.Series) -> str:
        """Infer business context/category for the column"""
        col_lower = col_name.lower()
        
        # ID/Key patterns
        if any(keyword in col_lower for keyword in ['id', 'key', 'ref', 'pk', 'fk']):
            return 'identifier'
        
        # Date/Time patterns
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'modified']):
            return 'temporal'
        
        # Amount/Money patterns
        if any(keyword in col_lower for keyword in ['amount', 'price', 'cost', 'total', 'sum', 'revenue', 'sales']):
            return 'financial'
        
        # Count/Quantity patterns
        if any(keyword in col_lower for keyword in ['count', 'quantity', 'qty', 'number', 'num']):
            return 'quantity'
        
        # Status/Flag patterns
        if any(keyword in col_lower for keyword in ['status', 'state', 'flag', 'active', 'enabled', 'type']):
            return 'categorical'
        
        # Name/Description patterns
        if any(keyword in col_lower for keyword in ['name', 'title', 'description', 'comment', 'note']):
            return 'descriptive'
        
        # Geographic patterns
        if any(keyword in col_lower for keyword in ['address', 'city', 'state', 'country', 'zip', 'postal', 'region']):
            return 'geographic'
        
        # Check data characteristics
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            if unique_ratio < 0.1:  # Low cardinality numeric
                return 'categorical'
            else:
                return 'measure'
        
        return 'attribute'
    
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
    
    def _generate_preview(self, file_path: str, file_extension: str, max_rows: int = 10) -> List[List]:
        """Generate preview data for the frontend"""
        try:
            df = self._read_file_to_dataframe(file_path, file_extension)
            if df is None or df.empty:
                return []
            
            # Clean column names for consistency
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Take first few rows
            preview_df = df.head(max_rows)
            
            # Convert to list of lists for JSON serialization
            # Include headers as first row
            preview_data = []
            
            # Add header row
            preview_data.append(list(df.columns))
            
            # Add data rows
            for _, row in preview_df.iterrows():
                # Convert each value to string, handling NaN/None
                row_data = []
                for val in row:
                    if pd.isna(val):
                        row_data.append("")
                    else:
                        row_data.append(str(val))
                preview_data.append(row_data)
            
            return preview_data
            
        except Exception as e:
            logger.error(f"Error generating preview: {str(e)}")
            return []
    
    def import_database_schema(self, project_id: int, connection_string: str) -> int:
        """Import schema from an existing database"""
        try:
            # Create engine and inspector
            engine = create_engine(connection_string)
            inspector = inspect(engine)
            
            # Create data source record
            source = DataSource(
                project_id=project_id,
                name="Database Import",
                type='database',
                connection_string=connection_string,
                ingest_status='processing'
            )
            db.session.add(source)
            db.session.flush()
            
            tables_created = 0
            table_names = inspector.get_table_names()
            
            for table_name in table_names:
                # Get table info
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
    
    def _infer_business_category(self, col_name: str) -> str:
        """Infer business category from column name (for database import)"""
        col_lower = col_name.lower()
        
        if any(keyword in col_lower for keyword in ['id', 'key']):
            return 'identifier'
        elif any(keyword in col_lower for keyword in ['name', 'title']):
            return 'descriptive'
        elif any(keyword in col_lower for keyword in ['date', 'time']):
            return 'temporal'
        elif any(keyword in col_lower for keyword in ['amount', 'price', 'cost']):
            return 'financial'
        else:
            return 'attribute'