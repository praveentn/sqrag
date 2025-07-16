# app.py
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
import json
import yaml
from datetime import datetime
import logging
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
    """Get all data sources"""
    sources = DataSource.query.all()
    return jsonify([{
        'id': s.id,
        'name': s.name,
        'type': s.type,
        'connection_string': s.connection_string,
        'created_at': s.created_at.isoformat(),
        'last_refresh': s.last_refresh.isoformat() if s.last_refresh else None
    } for s in sources])

@app.route('/api/data-sources', methods=['POST'])
def create_data_source():
    """Create new data source"""
    try:
        data = request.get_json()
        source = data_source_manager.create_source(
            name=data['name'],
            type=data['type'],
            connection_string=data.get('connection_string', ''),
            file_path=data.get('file_path', ''),
            _metadata=data.get('_metadata', {})
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
def refresh__metadata(source_id):
    """Refresh _metadata for data source"""
    try:
        source = DataSource.query.get_or_404(source_id)
        data_source_manager.refresh__metadata(source)
        return jsonify({'message': '_metadata refreshed successfully'})
    except Exception as e:
        logger.error(f"Error refreshing _metadata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-sources/<int:source_id>/sample', methods=['GET'])
def get_sample_data(source_id):
    """Get sample data from source"""
    try:
        source = DataSource.query.get_or_404(source_id)
        sample = data_source_manager.get_sample_data(source)
        return jsonify(sample)
    except Exception as e:
        logger.error(f"Error getting sample data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    """Get all dictionary entries"""
    entries = DictionaryEntry.query.all()
    return jsonify([{
        'id': e.id,
        'term': e.term,
        'definition': e.definition,
        'category': e.category,
        'synonyms': e.synonyms,
        'approved': e.approved,
        'created_at': e.created_at.isoformat(),
        'updated_at': e.updated_at.isoformat()
    } for e in entries])

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
        dictionary_service.auto_generate_dictionary()
        return jsonify({'message': 'Dictionary generated successfully'})
    except Exception as e:
        logger.error(f"Error generating dictionary: {str(e)}")
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
    sessions = ChatSession.query.order_by(ChatSession.created_at.desc()).all()
    return jsonify([{
        'id': s.id,
        'title': s.title,
        'created_at': s.created_at.isoformat(),
        'updated_at': s.updated_at.isoformat()
    } for s in sessions])

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
    messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.asc()).all()
    return jsonify([{
        'id': m.id,
        'role': m.role,
        'content': m.content,
        '_metadata': m._metadata,
        'created_at': m.created_at.isoformat()
    } for m in messages])

@app.route('/api/chat/query', methods=['POST'])
def process_query():
    """Process natural language query"""
    try:
        data = request.get_json()
        query = data['query']
        session_id = data.get('session_id')
        
        # Process query through RAG pipeline
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
        # Basic health check
        health = {
            'database': 'healthy',
            'embedding_service': 'healthy',
            'llm_service': 'healthy',
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
        entries = DictionaryEntry.query.all()
        data = [{
            'term': e.term,
            'definition': e.definition,
            'category': e.category,
            'synonyms': e.synonyms,
            'approved': e.approved
        } for e in entries]
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
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
            dictionary_service.import_dictionary(data)
            return jsonify({'message': 'Dictionary imported successfully'})
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    except Exception as e:
        logger.error(f"Error importing dictionary: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize database
@app.before_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)
