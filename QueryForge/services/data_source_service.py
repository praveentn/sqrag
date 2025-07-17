# services/data_source_service.py
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import json
import logging
from typing import Dict, List, Any, Optional
import traceback

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
            # Read CSV with pandas
            df = pd.read_csv(file_path)
            
            # Create table name from filename
            table_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Create table record
            table = Table(
                source_id=source.id,
                name=table_name,
                display_name=table_name.replace('_', ' ').title(),
                row_count=len(df),
                column_count=len(df.columns),
                schema_json=self._extract_schema(df)
            )
            db.session.add(table)
            db.session.flush()  # Get table ID
            
            # Create column records
            for col_name in df.columns:
                column = self._create_column_from_series(table.id, col_name, df[col_name])
                db.session.add(column)
            
            db.session.commit()
            return 1
            
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Error processing CSV file: {str(e)}")
    
    def _process_excel(self, source: DataSource, file_path: str) -> int:
        """Process Excel file (can have multiple sheets)"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            tables_created = 0
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty:
                    continue
                
                # Create table name from filename and sheet
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                table_name = f"{base_name}_{sheet_name}" if len(excel_file.sheet_names) > 1 else base_name
                
                # Create table record
                table = Table(
                    source_id=source.id,
                    name=table_name,
                    display_name=f"{base_name} - {sheet_name}",
                    row_count=len(df),
                    column_count=len(df.columns),
                    schema_json=self._extract_schema(df)
                )
                db.session.add(table)
                db.session.flush()
                
                # Create column records
                for col_name in df.columns:
                    column = self._create_column_from_series(table.id, col_name, df[col_name])
                    db.session.add(column)
                
                tables_created += 1
            
            db.session.commit()
            return tables_created
            
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Error processing Excel file: {str(e)}")
    
    def _process_json(self, source: DataSource, file_path: str) -> int:
        """Process JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle nested JSON
                df = pd.json_normalize(data)
            else:
                raise ValueError("Unsupported JSON structure")
            
            # Create table name from filename
            table_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Create table record
            table = Table(
                source_id=source.id,
                name=table_name,
                display_name=table_name.replace('_', ' ').title(),
                row_count=len(df),
                column_count=len(df.columns),
                schema_json=self._extract_schema(df)
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
            raise Exception(f"Error processing JSON file: {str(e)}")
    
    def _extract_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract schema information from DataFrame"""
        schema = {
            'columns': [],
            'data_types': {},
            'nullable_columns': [],
            'sample_data': {}
        }
        
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'non_null_count': int(df[col].count()),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique())
            }
            
            # Add sample values
            sample_values = df[col].dropna().head(5).tolist()
            col_info['sample_values'] = [str(v) for v in sample_values]
            
            schema['columns'].append(col_info)
            schema['data_types'][col] = str(df[col].dtype)
            
            if df[col].isnull().any():
                schema['nullable_columns'].append(col)
        
        return schema
    
    def _create_column_from_series(self, table_id: int, col_name: str, series: pd.Series) -> Column:
        """Create Column object from pandas Series"""
        # Infer data type
        dtype = str(series.dtype)
        python_type = self._map_pandas_dtype_to_sql(dtype)
        
        # Calculate statistics
        null_count = int(series.isnull().sum())
        distinct_count = int(series.nunique())
        
        # Get sample values
        sample_values = series.dropna().head(10).tolist()
        sample_values = [str(v) for v in sample_values]
        
        # Calculate min/max for numeric columns
        min_value = None
        max_value = None
        avg_value = None
        
        if pd.api.types.is_numeric_dtype(series):
            min_value = str(series.min()) if not pd.isna(series.min()) else None
            max_value = str(series.max()) if not pd.isna(series.max()) else None
            avg_value = float(series.mean()) if not pd.isna(series.mean()) else None
        
        # Detect PII
        pii_flag = self._detect_pii(col_name, sample_values)
        
        return Column(
            table_id=table_id,
            name=col_name,
            display_name=col_name.replace('_', ' ').title(),
            data_type=python_type,
            is_nullable=null_count > 0,
            distinct_count=distinct_count,
            null_count=null_count,
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            sample_values=sample_values,
            pii_flag=pii_flag,
            business_category=self._infer_business_category(col_name)
        )
    
    def _map_pandas_dtype_to_sql(self, pandas_dtype: str) -> str:
        """Map pandas dtype to SQL data type"""
        mapping = {
            'int64': 'INTEGER',
            'int32': 'INTEGER',
            'float64': 'FLOAT',
            'float32': 'FLOAT',
            'object': 'VARCHAR',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'category': 'VARCHAR'
        }
        return mapping.get(pandas_dtype.lower(), 'VARCHAR')
    
    def _detect_pii(self, col_name: str, sample_values: List[str]) -> bool:
        """Detect if column might contain PII"""
        pii_keywords = [
            'email', 'phone', 'ssn', 'social', 'password', 'credit_card',
            'address', 'zip', 'postal', 'name', 'surname', 'firstname',
            'lastname', 'birth', 'age', 'salary', 'income'
        ]
        
        col_lower = col_name.lower()
        for keyword in pii_keywords:
            if keyword in col_lower:
                return True
        
        # Check sample values for patterns
        for value in sample_values:
            value = str(value).lower()
            # Email pattern
            if '@' in value and '.' in value:
                return True
            # Phone pattern (simple check)
            if len(value.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')) >= 10 and \
               any(c.isdigit() for c in value):
                return True
        
        return False
    
    def _infer_business_category(self, col_name: str) -> Optional[str]:
        """Infer business category from column name"""
        categories = {
            'finance': ['amount', 'price', 'cost', 'revenue', 'profit', 'budget', 'expense'],
            'hr': ['employee', 'salary', 'department', 'manager', 'hire', 'termination'],
            'sales': ['customer', 'order', 'product', 'quantity', 'discount', 'commission'],
            'marketing': ['campaign', 'lead', 'conversion', 'impression', 'click'],
            'operations': ['inventory', 'supply', 'vendor', 'shipping', 'delivery']
        }
        
        col_lower = col_name.lower()
        for category, keywords in categories.items():
            if any(keyword in col_lower for keyword in keywords):
                return category
        
        return None
    
    def test_database_connection(self, connection_config: Dict[str, Any]) -> bool:
        """Test database connection"""
        try:
            connection_string = self._build_connection_string(connection_config)
            engine = create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
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
        if source.subtype == 'csv':
            df = pd.read_csv(source.file_path).head(limit)
        elif source.subtype in ['xlsx', 'xls']:
            # For Excel files with multiple sheets, we need to determine which sheet
            sheet_name = table.name.split('_', 1)[-1] if '_' in table.name else 0
            df = pd.read_excel(source.file_path, sheet_name=sheet_name).head(limit)
        else:
            raise ValueError(f"Unsupported file type for sampling: {source.subtype}")
        
        return {
            'columns': df.columns.tolist(),
            'data': df.fillna('').to_dict('records'),
            'total_rows': len(df)
        }
    
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