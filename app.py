# app.py
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
import json
import yaml
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import uuid
from services.data_source_manager import DataSourceManager
from services.dictionary_service import DictionaryService
from services.embedding_service import EmbeddingService
from services.rag_pipeline import RAGPipeline
from services.chat_service import ChatService
from services.sql_executor import SQLExecutor
from models import db, DataSource, Table, Column, DictionaryEntry, ChatSession, ChatMessage
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Initialize services
data_source_manager = DataSourceManager()
dictionary_service = DictionaryService()
embedding_service = EmbeddingService()
rag_pipeline = RAGPipeline()
chat_service = ChatService()
sql_executor = SQLExecutor()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database initialization flag
database_initialized = False

@app.before_request
def create_tables():
    """Create database tables if they don't exist"""
    global database_initialized
    if not database_initialized:
        try:
            db.create_all()
            logger.info("Database tables created/verified")
            database_initialized = True
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")

# Routes

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/data-sources')
def data_sources():
    """Data Source Manager Interface"""
    sources = DataSource.query.all()
    return render_template('data_sources.html', sources=sources)

@app.route('/dictionary')
def dictionary():
    """Data Dictionary & Encyclopedia Interface"""
    entries = DictionaryEntry.query.all()
    return render_template('dictionary.html', entries=entries)

@app.route('/embeddings')
def embeddings():
    """Embedding & Index Console Interface"""
    return render_template('embeddings.html')

@app.route('/chat')
def chat():
    """Chat Workspace Interface"""
    return render_template('chat.html')

@app.route('/admin')
def admin():
    """Admin Control Panel Interface"""
    return render_template('admin.html')

# API Routes

@app.route('/api/data-sources', methods=['GET'])
def get_data_sources():
    """Get all data sources with their tables"""
    try:
        sources = DataSource.query.all()
        sources_data = []
        
        for source in sources:
            source_dict = source.to_dict()
            # Add table and column counts for summary
            source_dict['table_count'] = len(source.tables)
            source_dict['column_count'] = sum(len(table.columns) for table in source.tables)
            sources_data.append(source_dict)
        
        return jsonify(sources_data)
    except Exception as e:
        logger.error(f"Error getting data sources: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources', methods=['POST'])
def create_data_source():
    """Create new data source"""
    try:
        # Handle both JSON and form data
        if request.content_type and 'multipart/form-data' in request.content_type:
            # File upload case
            data = {}
            data['name'] = request.form.get('name')
            data['type'] = request.form.get('type')
            data['description'] = request.form.get('description', '')
            
            # Handle file upload
            if 'file' in request.files:
                file = request.files['file']
                if file and file.filename:
                    # Validate file type
                    allowed_extensions = {'.csv', '.xlsx', '.xls'}
                    file_ext = os.path.splitext(file.filename)[1].lower()
                    
                    if file_ext not in allowed_extensions:
                        return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed.'}), 400
                    
                    # Save file
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4().hex}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    
                    # Ensure upload directory exists
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    
                    file.save(file_path)
                    data['file_path'] = file_path
                    
                    logger.info(f"File saved to: {file_path}")
            
            # Handle database connection
            if data['type'] not in ['csv', 'excel']:
                data['connection_string'] = request.form.get('connection_string', '')
        else:
            # JSON case
            data = request.get_json()
        
        # Validate required fields
        if not data.get('name') or not data.get('type'):
            return jsonify({'error': 'Name and type are required'}), 400
        
        # Create data source
        source = data_source_manager.create_source(
            name=data['name'],
            type=data['type'],
            connection_string=data.get('connection_string', ''),
            file_path=data.get('file_path', ''),
            metadata={'description': data.get('description', '')}
        )
        
        return jsonify({'id': source.id, 'message': 'Data source created successfully'})
        
    except Exception as e:
        logger.error(f"Error creating data source: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources/<int:source_id>/test', methods=['POST'])
def test_connection(source_id):
    """Test connection to data source"""
    try:
        source = DataSource.query.get_or_404(source_id)
        result = data_source_manager.test_connection(source)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error testing connection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources/<int:source_id>/refresh', methods=['POST'])
def refresh_metadata(source_id):
    """Refresh metadata for data source"""
    try:
        source = DataSource.query.get_or_404(source_id)
        data_source_manager.refresh_metadata(source)
        return jsonify({'message': 'Metadata refreshed successfully'})
    except Exception as e:
        logger.error(f"Error refreshing metadata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources/<int:source_id>/sample', methods=['GET'])
def get_sample_data(source_id):
    """Get sample data from source"""
    try:
        source = DataSource.query.get_or_404(source_id)
        table_name = request.args.get('table')
        limit = int(request.args.get('limit', 100))
        
        sample = data_source_manager.get_sample_data(source, table_name, limit)
        return jsonify(sample)
    except Exception as e:
        logger.error(f"Error getting sample data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources/<int:source_id>/stats', methods=['GET'])
def get_source_statistics(source_id):
    """Get comprehensive statistics for a data source"""
    try:
        stats = data_source_manager.get_source_statistics(source_id)
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting source statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources/<int:source_id>', methods=['DELETE'])
def delete_data_source(source_id):
    """Delete a data source"""
    try:
        data_source_manager.delete_source(source_id)
        return jsonify({'message': 'Data source deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting data source: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Selective Dictionary Generation API
@app.route('/api/dictionary/generate-selective', methods=['POST'])
def generate_selective_dictionary():
    """Generate dictionary entries for selected tables/columns"""
    try:
        data = request.get_json()
        table_ids = data.get('table_ids', [])
        column_ids = data.get('column_ids', [])
        
        if not table_ids and not column_ids:
            return jsonify({'error': 'Please select at least one table or column'}), 400
        
        result = dictionary_service.generate_selective_dictionary(table_ids, column_ids)
        return jsonify({
            'message': 'Dictionary generated successfully',
            'terms_generated': result.get('terms_generated', 0),
            'tables_processed': result.get('tables_processed', 0),
            'columns_processed': result.get('columns_processed', 0)
        })
    except Exception as e:
        logger.error(f"Error generating selective dictionary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    """Get all dictionary entries"""
    try:
        entries = DictionaryEntry.query.all()
        return jsonify([entry.to_dict() for entry in entries])
    except Exception as e:
        logger.error(f"Error getting dictionary entries: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dictionary', methods=['POST'])
def create_dictionary_entry():
    """Create dictionary entry"""
    try:
        data = request.get_json()
        entry = dictionary_service.create_entry(
            term=data['term'],
            definition=data['definition'],
            category=data.get('category', 'general'),
            synonyms=data.get('synonyms', []),
            approved=data.get('approved', False)
        )
        return jsonify({'id': entry.id, 'message': 'Dictionary entry created successfully'})
    except Exception as e:
        logger.error(f"Error creating dictionary entry: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dictionary/<int:entry_id>', methods=['PUT'])
def update_dictionary_entry(entry_id):
    """Update dictionary entry"""
    try:
        data = request.get_json()
        entry = dictionary_service.update_entry(entry_id, data)
        return jsonify({'message': 'Dictionary entry updated successfully'})
    except Exception as e:
        logger.error(f"Error updating dictionary entry: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dictionary/<int:entry_id>', methods=['DELETE'])
def delete_dictionary_entry(entry_id):
    """Delete dictionary entry"""
    try:
        dictionary_service.delete_entry(entry_id)
        return jsonify({'message': 'Dictionary entry deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting dictionary entry: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dictionary/generate', methods=['POST'])
def generate_dictionary():
    """Auto-generate dictionary from data sources"""
    try:
        result = dictionary_service.auto_generate_dictionary()
        return jsonify({
            'message': 'Dictionary generated successfully',
            'terms_generated': result.get('terms_generated', 0),
            'tables_processed': result.get('tables_processed', 0),
            'columns_processed': result.get('columns_processed', 0)
        })
    except Exception as e:
        logger.error(f"Error generating dictionary: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Selective Embedding Generation API
@app.route('/api/embeddings/create-selective', methods=['POST'])
def create_selective_embeddings():
    """Create embeddings for selected tables/columns"""
    try:
        data = request.get_json()
        scope = data.get('scope')  # 'table' or 'column'
        table_ids = data.get('table_ids', [])
        column_ids = data.get('column_ids', [])
        backend = data.get('backend', 'faiss')
        model = data.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        name = data.get('name')
        
        if scope == 'table' and not table_ids:
            return jsonify({'error': 'Please select at least one table'}), 400
        elif scope == 'column' and not column_ids:
            return jsonify({'error': 'Please select at least one column'}), 400
        
        job_id = embedding_service.create_selective_index(
            scope=scope,
            table_ids=table_ids,
            column_ids=column_ids,
            backend=backend,
            model=model,
            name=name
        )
        return jsonify({'job_id': job_id, 'message': 'Embedding job started'})
    except Exception as e:
        logger.error(f"Error creating selective embeddings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/create', methods=['POST'])
def create_embeddings():
    """Create embeddings index"""
    try:
        data = request.get_json()
        job_id = embedding_service.create_index(
            scope=data['scope'],
            backend=data['backend'],
            model=data.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        return jsonify({'job_id': job_id, 'message': 'Embedding job started'})
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/status/<job_id>', methods=['GET'])
def get_embedding_status(job_id):
    """Get embedding job status"""
    try:
        status = embedding_service.get_job_status(job_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting embedding status: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Get chat sessions"""
    try:
        sessions = chat_service.get_sessions(limit=50)
        return jsonify([session.to_dict() for session in sessions])
    except Exception as e:
        logger.error(f"Error getting chat sessions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sessions', methods=['POST'])
def create_chat_session():
    """Create new chat session"""
    try:
        data = request.get_json()
        session = chat_service.create_session(title=data.get('title', 'New Chat'))
        return jsonify({'id': session.id, 'message': 'Chat session created successfully'})
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sessions/<int:session_id>/messages', methods=['GET'])
def get_chat_messages(session_id):
    """Get messages for chat session"""
    try:
        messages = chat_service.get_messages(session_id)
        return jsonify([message.to_dict() for message in messages])
    except Exception as e:
        logger.error(f"Error getting chat messages: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sessions/<int:session_id>', methods=['GET'])
def get_chat_session(session_id):
    """Get a specific chat session"""
    try:
        session = chat_service.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        return jsonify(session.to_dict())
    except Exception as e:
        logger.error(f"Error getting chat session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sessions/<int:session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    """Delete a chat session"""
    try:
        result = chat_service.delete_session(session_id)
        if result:
            return jsonify({'message': 'Chat session deleted successfully'})
        else:
            return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/query', methods=['POST'])
def process_query():
    """Process natural language query with step-by-step confirmation"""
    try:
        data = request.get_json()
        query = data['query']
        session_id = data.get('session_id')
        step = data.get('step', 'extract_entities')  # extract_entities, generate_sql, execute_sql
        context = data.get('context', {})
        
        if step == 'extract_entities':
            # Step 1: Extract entities and show to user
            result = rag_pipeline.extract_entities_step(query, session_id)
        elif step == 'generate_sql':
            # Step 2: Generate SQL with entities and show to user
            result = rag_pipeline.generate_sql_step(query, session_id, context)
        elif step == 'execute_sql':
            # Step 3: Execute SQL after user confirmation
            result = rag_pipeline.execute_sql_step(query, session_id, context)
        else:
            # Process complete query (backward compatibility)
            result = rag_pipeline.process_query(query, session_id)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/feedback', methods=['POST'])
def process_feedback():
    """Process user feedback on query results"""
    try:
        data = request.get_json()
        feedback = data['feedback']
        session_id = data['session_id']
        message_id = data.get('message_id')
        
        # Process feedback through RAG pipeline
        result = rag_pipeline.process_feedback(feedback, session_id, message_id)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sql/execute', methods=['POST'])
def execute_sql():
    """Execute SQL query"""
    try:
        data = request.get_json()
        sql = data['sql']
        source_id = data.get('source_id')
        
        result = sql_executor.execute_query(sql, source_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error executing SQL: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/health', methods=['GET'])
def get_system_health():
    """Get system health metrics"""
    try:
        # Check database connection
        db_healthy = True
        try:
            db.session.execute('SELECT 1')
        except:
            db_healthy = False
        
        # Check LLM service
        llm_healthy = True
        try:
            from services.llm_client import LLMClient
            llm_client = LLMClient()
            test_result = llm_client.test_connection()
            llm_healthy = test_result.get('status') == 'success'
        except:
            llm_healthy = False
        
        # Check embedding service
        embedding_healthy = True
        try:
            # Simple check - this could be more sophisticated
            embedding_service.get_available_indexes()
        except:
            embedding_healthy = False
        
        health = {
            'database': 'healthy' if db_healthy else 'error',
            'llm_service': 'healthy' if llm_healthy else 'error',
            'embedding_service': 'healthy' if embedding_healthy else 'error',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(health)
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        stats = {
            'data_sources': DataSource.query.count(),
            'tables': Table.query.count(),
            'columns': Column.query.count(),
            'dictionary_entries': DictionaryEntry.query.count(),
            'chat_sessions': ChatSession.query.count(),
            'chat_messages': ChatMessage.query.count()
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/dictionary', methods=['GET'])
def export_dictionary():
    """Export dictionary as JSON"""
    try:
        entries = dictionary_service.export_dictionary()
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(entries, f, indent=2)
            temp_path = f.name
        
        return send_file(temp_path, as_attachment=True, download_name='dictionary.json')
    except Exception as e:
        logger.error(f"Error exporting dictionary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/import/dictionary', methods=['POST'])
def import_dictionary():
    """Import dictionary from JSON file"""
    try:
        file = request.files['file']
        if file and file.filename.endswith('.json'):
            data = json.load(file)
            result = dictionary_service.import_dictionary(data)
            return jsonify({
                'message': 'Dictionary imported successfully',
                'stats': result
            })
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    except Exception as e:
        logger.error(f"Error importing dictionary: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Additional API endpoints for getting tables and columns for selection
@app.route('/api/tables', methods=['GET'])
def get_tables():
    """Get all tables for selection"""
    try:
        tables = Table.query.all()
        return jsonify([{
            'id': table.id,
            'name': table.name,
            'display_name': table.display_name,
            'source_id': table.source_id,
            'source_name': table.data_source.name,
            'row_count': table.row_count,
            'column_count': len(table.columns)
        } for table in tables])
    except Exception as e:
        logger.error(f"Error getting tables: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/columns', methods=['GET'])
def get_columns():
    """Get all columns for selection"""
    try:
        table_id = request.args.get('table_id')
        query = Column.query
        if table_id:
            query = query.filter_by(table_id=int(table_id))
        
        columns = query.all()
        return jsonify([{
            'id': column.id,
            'name': column.name,
            'display_name': column.display_name,
            'table_id': column.table_id,
            'table_name': column.table.name,
            'data_type': column.data_type,
            'unique_count': column.unique_count
        } for column in columns])
    except Exception as e:
        logger.error(f"Error getting columns: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data-sources/<int:source_id>/tables/<table_name>/data', methods=['GET'])
def get_table_data(source_id, table_name):
    """Get data from a specific table"""
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        source = DataSource.query.get_or_404(source_id)
        table = Table.query.filter_by(source_id=source_id, name=table_name).first()
        
        if not table:
            return jsonify({'error': f'Table {table_name} not found in source {source.name}'}), 404
        
        # Use the SQL executor to get table data
        sql = f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}"
        result = sql_executor.execute_query(sql, source_id)
        
        if result['status'] == 'success':
            return jsonify({
                'data': result.get('data', []),
                'row_count': result.get('row_count', 0),
                'total_rows': table.row_count,
                'table_name': table_name,
                'source_name': source.name,
                'columns': [col.to_dict() for col in table.columns]
            })
        else:
            return jsonify({'error': result.get('error', 'Unknown error')}), 500
            
    except Exception as e:
        logger.error(f"Error getting table data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources/<int:source_id>/tables', methods=['GET'])
def get_source_tables(source_id):
    """Get all tables for a data source with their metadata"""
    try:
        source = DataSource.query.get_or_404(source_id)
        tables = Table.query.filter_by(source_id=source_id).all()
        
        tables_data = []
        for table in tables:
            table_dict = table.to_dict()
            table_dict['columns'] = [col.to_dict() for col in table.columns]
            tables_data.append(table_dict)
        
        return jsonify({
            'source': source.to_dict(),
            'tables': tables_data
        })
        
    except Exception as e:
        logger.error(f"Error getting source tables: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/indexes', methods=['GET'])
def get_embedding_indexes():
    """Get all embedding indexes"""
    try:
        from models import EmbeddingIndex
        
        indexes = EmbeddingIndex.query.all()
        indexes_data = []
        
        for index in indexes:
            index_dict = {
                'id': index.id,
                'name': index.name,
                'scope': index.scope,
                'backend': index.backend,
                'model_name': index.model_name,
                'status': index.status,
                'progress': round(index.progress or 0.0, 2),
                'item_count': index.item_count or 0,
                'dimensions': index.dimensions,
                'index_path': index.index_path,
                'error_message': index.error_message,
                'created_at': index.created_at.isoformat() if index.created_at else None,
                'updated_at': index.updated_at.isoformat() if index.updated_at else None
            }
            indexes_data.append(index_dict)
        
        return jsonify(indexes_data)
        
    except Exception as e:
        logger.error(f"Error getting embedding indexes: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/indexes/<int:index_id>', methods=['GET'])
def get_embedding_index(index_id):
    """Get specific embedding index details"""
    try:
        from models import EmbeddingIndex
        
        index = EmbeddingIndex.query.get_or_404(index_id)
        
        index_dict = {
            'id': index.id,
            'name': index.name,
            'scope': index.scope,
            'backend': index.backend,
            'model_name': index.model_name,
            'status': index.status,
            'progress': round(index.progress or 0.0, 2),
            'item_count': index.item_count or 0,
            'dimensions': index.dimensions,
            'index_path': index.index_path,
            'error_message': index.error_message,
            'created_at': index.created_at.isoformat() if index.created_at else None,
            'updated_at': index.updated_at.isoformat() if index.updated_at else None
        }
        
        return jsonify(index_dict)
        
    except Exception as e:
        logger.error(f"Error getting embedding index: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/indexes/<int:index_id>', methods=['DELETE'])
def delete_embedding_index(index_id):
    """Delete an embedding index"""
    try:
        from models import EmbeddingIndex
        import os
        
        index = EmbeddingIndex.query.get_or_404(index_id)
        
        # Delete index files if they exist
        if index.index_path and os.path.exists(index.index_path):
            try:
                os.remove(index.index_path)
                logger.info(f"Deleted index file: {index.index_path}")
            except Exception as e:
                logger.warning(f"Could not delete index file {index.index_path}: {str(e)}")
        
        # Delete database record
        db.session.delete(index)
        db.session.commit()
        
        return jsonify({'message': 'Index deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting embedding index: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/job-status/<job_id>', methods=['GET'])
def get_embedding_job_status(job_id):
    """Get status of an embedding job"""
    try:
        status = embedding_service.get_job_status(job_id)
        if status:
            return jsonify(status)
        else:
            return jsonify({'error': 'Job not found'}), 404
            
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources/<int:source_id>', methods=['PUT'])
def update_data_source(source_id):
    """Update data source metadata"""
    try:
        source = DataSource.query.get_or_404(source_id)
        data = request.get_json()
        
        # Update allowed fields
        if 'name' in data:
            source.name = data['name']
        if 'description' in data:
            if source.metadata is None:
                source.metadata = {}
            source.metadata['description'] = data['description']
        
        source.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({'message': 'Data source updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating data source: {str(e)}")
        return jsonify({'error': str(e)}), 500



# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)
