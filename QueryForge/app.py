# app.py
import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

# Import configuration and models
from config import get_config
from models import db, Project, DataSource, Table, Column, DictionaryEntry, Embedding, Index, SearchLog, NLQFeedback

# Import services
from services.data_source_service import DataSourceService
from services.dictionary_service import DictionaryService
from services.embedding_service import EmbeddingService
from services.search_service import SearchService
from services.chat_service import ChatService
from services.admin_service import AdminService

def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app, origins="*")
    
    # Configure logging
    if not app.debug:
        logging.basicConfig(level=logging.INFO)
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('indexes', exist_ok=True)
    
    # Initialize services
    data_source_service = DataSourceService()
    dictionary_service = DictionaryService()
    embedding_service = EmbeddingService()
    search_service = SearchService()
    chat_service = ChatService()
    admin_service = AdminService()
    
    # =================== ERROR HANDLERS ===================
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found', 'message': str(error)}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        app.logger.error(f'Server Error: {error}')
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500
    
    # =================== TAB 0: PROJECTS ===================
    
    @app.route('/api/projects', methods=['GET'])
    def get_projects():
        """Get all projects"""
        try:
            projects = Project.query.all()
            return jsonify({
                'success': True,
                'projects': [project.to_dict() for project in projects]
            })
        except Exception as e:
            app.logger.error(f'Error fetching projects: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects', methods=['POST'])
    def create_project():
        """Create a new project"""
        try:
            data = request.get_json()
            project = Project(
                name=data['name'],
                description=data.get('description', ''),
                owner=data.get('owner', 'default_user')
            )
            db.session.add(project)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'project': project.to_dict()
            }), 201
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating project: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>', methods=['GET'])
    def get_project(project_id):
        """Get project by ID"""
        try:
            project = Project.query.get_or_404(project_id)
            return jsonify({
                'success': True,
                'project': project.to_dict()
            })
        except Exception as e:
            app.logger.error(f'Error fetching project {project_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>', methods=['PUT'])
    def update_project(project_id):
        """Update project"""
        try:
            project = Project.query.get_or_404(project_id)
            data = request.get_json()
            
            project.name = data.get('name', project.name)
            project.description = data.get('description', project.description)
            
            db.session.commit()
            return jsonify({
                'success': True,
                'project': project.to_dict()
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error updating project {project_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>', methods=['DELETE'])
    def delete_project(project_id):
        """Delete project"""
        try:
            project = Project.query.get_or_404(project_id)
            project.status = 'deleted'
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Project archived'})
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error deleting project {project_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/clone', methods=['POST'])
    def clone_project(project_id):
        """Clone project with all data"""
        try:
            original = Project.query.get_or_404(project_id)
            data = request.get_json()
            
            # Create new project
            cloned = Project(
                name=data.get('name', f"{original.name} (Copy)"),
                description=f"Cloned from {original.name}",
                owner=original.owner
            )
            db.session.add(cloned)
            db.session.flush()  # Get the ID
            
            # Clone dictionary entries
            for entry in original.dictionary_entries:
                cloned_entry = DictionaryEntry(
                    project_id=cloned.id,
                    term=entry.term,
                    definition=entry.definition,
                    category=entry.category,
                    synonyms=entry.synonyms,
                    abbreviations=entry.abbreviations,
                    domain=entry.domain,
                    status=entry.status
                )
                db.session.add(cloned_entry)
            
            db.session.commit()
            return jsonify({
                'success': True,
                'project': cloned.to_dict()
            }), 201
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error cloning project {project_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 1: DATA SOURCES ===================
    
    @app.route('/api/projects/<int:project_id>/sources', methods=['GET'])
    def get_data_sources(project_id):
        """Get all data sources for a project"""
        try:
            sources = DataSource.query.filter_by(project_id=project_id).all()
            return jsonify({
                'success': True,
                'sources': [source.to_dict() for source in sources]
            })
        except Exception as e:
            app.logger.error(f'Error fetching sources for project {project_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/sources/upload', methods=['POST'])
    def upload_file_source(project_id):
        """Upload and process file data source"""
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            # Secure filename and save
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Create data source entry
            source = DataSource(
                project_id=project_id,
                name=filename,
                type='file',
                subtype=filename.split('.')[-1].lower(),
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                ingest_status='processing'
            )
            db.session.add(source)
            db.session.commit()
            
            # Process file asynchronously
            try:
                tables_created = data_source_service.process_uploaded_file(source.id, file_path)
                source.ingest_status = 'completed'
                source.ingest_progress = 1.0
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'source': source.to_dict(),
                    'tables_created': tables_created
                })
            except Exception as processing_error:
                source.ingest_status = 'failed'
                source.error_message = str(processing_error)
                db.session.commit()
                raise processing_error
                
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error uploading file: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/sources/database', methods=['POST'])
    def add_database_source(project_id):
        """Add database connection as data source"""
        try:
            data = request.get_json()
            
            # Test connection first
            connection_valid = data_source_service.test_database_connection(data['connection_config'])
            if not connection_valid:
                return jsonify({'success': False, 'error': 'Database connection failed'}), 400
            
            source = DataSource(
                project_id=project_id,
                name=data['name'],
                type='database',
                subtype=data['db_type'],
                connection_config=data['connection_config'],
                ingest_status='completed'
            )
            db.session.add(source)
            db.session.commit()
            
            # Import schema
            tables_imported = data_source_service.import_database_schema(source.id)
            
            return jsonify({
                'success': True,
                'source': source.to_dict(),
                'tables_imported': tables_imported
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error adding database source: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/sources/<int:source_id>/tables', methods=['GET'])
    def get_source_tables(source_id):
        """Get tables for a data source"""
        try:
            tables = Table.query.filter_by(source_id=source_id).all()
            return jsonify({
                'success': True,
                'tables': [table.to_dict() for table in tables]
            })
        except Exception as e:
            app.logger.error(f'Error fetching tables for source {source_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/tables/<int:table_id>/columns', methods=['GET'])
    def get_table_columns(table_id):
        """Get columns for a table"""
        try:
            columns = Column.query.filter_by(table_id=table_id).all()
            return jsonify({
                'success': True,
                'columns': [column.to_dict() for column in columns]
            })
        except Exception as e:
            app.logger.error(f'Error fetching columns for table {table_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/columns/<int:column_id>', methods=['PUT'])
    def update_column(column_id):
        """Update column metadata"""
        try:
            column = Column.query.get_or_404(column_id)
            data = request.get_json()
            
            column.display_name = data.get('display_name', column.display_name)
            column.description = data.get('description', column.description)
            column.business_category = data.get('business_category', column.business_category)
            column.pii_flag = data.get('pii_flag', column.pii_flag)
            
            db.session.commit()
            return jsonify({
                'success': True,
                'column': column.to_dict()
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error updating column {column_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 2: DATA DICTIONARY ===================
    
    @app.route('/api/projects/<int:project_id>/dictionary', methods=['GET'])
    def get_dictionary_entries(project_id):
        """Get all dictionary entries for a project"""
        try:
            entries = DictionaryEntry.query.filter_by(project_id=project_id).all()
            return jsonify({
                'success': True,
                'entries': [entry.to_dict() for entry in entries]
            })
        except Exception as e:
            app.logger.error(f'Error fetching dictionary for project {project_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/dictionary', methods=['POST'])
    def create_dictionary_entry(project_id):
        """Create new dictionary entry"""
        try:
            data = request.get_json()
            entry = DictionaryEntry(
                project_id=project_id,
                term=data['term'],
                definition=data['definition'],
                category=data.get('category', 'business_term'),
                synonyms=data.get('synonyms', []),
                abbreviations=data.get('abbreviations', []),
                domain=data.get('domain'),
                created_by=data.get('created_by', 'user')
            )
            db.session.add(entry)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'entry': entry.to_dict()
            }), 201
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating dictionary entry: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/dictionary/<int:entry_id>', methods=['PUT'])
    def update_dictionary_entry(entry_id):
        """Update dictionary entry"""
        try:
            entry = DictionaryEntry.query.get_or_404(entry_id)
            data = request.get_json()
            
            entry.term = data.get('term', entry.term)
            entry.definition = data.get('definition', entry.definition)
            entry.category = data.get('category', entry.category)
            entry.synonyms = data.get('synonyms', entry.synonyms)
            entry.abbreviations = data.get('abbreviations', entry.abbreviations)
            entry.domain = data.get('domain', entry.domain)
            
            db.session.commit()
            return jsonify({
                'success': True,
                'entry': entry.to_dict()
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error updating dictionary entry {entry_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/dictionary/<int:entry_id>', methods=['DELETE'])
    def delete_dictionary_entry(entry_id):
        """Delete dictionary entry"""
        try:
            entry = DictionaryEntry.query.get_or_404(entry_id)
            entry.status = 'archived'
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Dictionary entry archived'})
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error deleting dictionary entry {entry_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/dictionary/suggest', methods=['POST'])
    def suggest_dictionary_terms(project_id):
        """Auto-generate dictionary suggestions from data"""
        try:
            suggestions = dictionary_service.generate_suggestions(project_id)
            return jsonify({
                'success': True,
                'suggestions': suggestions
            })
        except Exception as e:
            app.logger.error(f'Error generating dictionary suggestions: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 3: EMBEDDINGS & INDEXING ===================
    
    @app.route('/api/projects/<int:project_id>/embeddings', methods=['GET'])
    def get_embeddings(project_id):
        """Get embeddings for a project"""
        try:
            embeddings = Embedding.query.filter_by(project_id=project_id).all()
            return jsonify({
                'success': True,
                'embeddings': [emb.to_dict() for emb in embeddings]
            })
        except Exception as e:
            app.logger.error(f'Error fetching embeddings: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/embeddings/batch', methods=['POST'])
    def create_embeddings_batch(project_id):
        """Create embeddings in batch"""
        try:
            data = request.get_json()
            model_name = data['model_name']
            object_types = data['object_types']  # ['tables', 'columns', 'dictionary']
            
            # Start batch embedding process
            job_id = embedding_service.create_embeddings_batch(
                project_id, model_name, object_types
            )
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'message': 'Embedding creation started'
            })
        except Exception as e:
            app.logger.error(f'Error creating embeddings: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/indexes', methods=['GET'])
    def get_indexes(project_id):
        """Get indexes for a project"""
        try:
            indexes = Index.query.filter_by(project_id=project_id).all()
            return jsonify({
                'success': True,
                'indexes': [idx.to_dict() for idx in indexes]
            })
        except Exception as e:
            app.logger.error(f'Error fetching indexes: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/indexes', methods=['POST'])
    def create_index(project_id):
        """Create a new search index"""
        try:
            data = request.get_json()
            
            index = Index(
                project_id=project_id,
                name=data['name'],
                description=data.get('description', ''),
                index_type=data['index_type'],
                metric=data.get('metric', 'cosine'),
                object_scope=data['object_scope'],
                embedding_model=data.get('embedding_model'),
                build_params=data.get('build_params', {}),
                status='building'
            )
            db.session.add(index)
            db.session.commit()
            
            # Start index building
            embedding_service.build_index(index.id)
            
            return jsonify({
                'success': True,
                'index': index.to_dict()
            }), 201
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating index: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 4: SEARCH PLAYGROUND ===================
    
    @app.route('/api/search', methods=['POST'])
    def perform_search():
        """Perform search across indexes"""
        try:
            data = request.get_json()
            query = data['query']
            index_id = data.get('index_id')
            search_type = data.get('search_type', 'hybrid')
            top_k = data.get('top_k', 10)
            
            results = search_service.search(
                query=query,
                index_id=index_id,
                search_type=search_type,
                top_k=top_k
            )
            
            return jsonify({
                'success': True,
                'results': results
            })
        except Exception as e:
            app.logger.error(f'Error performing search: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 5: CHAT (NL → SQL) ===================
    
    @app.route('/api/chat/entities', methods=['POST'])
    def extract_entities():
        """Extract entities from natural language query"""
        try:
            data = request.get_json()
            query = data['query']
            project_id = data['project_id']
            
            entities = chat_service.extract_entities(query, project_id)
            
            return jsonify({
                'success': True,
                'entities': entities
            })
        except Exception as e:
            app.logger.error(f'Error extracting entities: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/chat/mapping', methods=['POST'])
    def map_entities():
        """Map entities to tables/columns"""
        try:
            data = request.get_json()
            entities = data['entities']
            project_id = data['project_id']
            
            mappings = chat_service.map_entities_to_schema(entities, project_id)
            
            return jsonify({
                'success': True,
                'mappings': mappings
            })
        except Exception as e:
            app.logger.error(f'Error mapping entities: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/chat/sql', methods=['POST'])
    def generate_sql():
        """Generate SQL from natural language query and context"""
        try:
            data = request.get_json()
            query = data['query']
            entities = data['entities']
            mappings = data['mappings']
            project_id = data['project_id']
            
            sql_result = chat_service.generate_sql(query, entities, mappings, project_id)
            
            return jsonify({
                'success': True,
                'sql_result': sql_result
            })
        except Exception as e:
            app.logger.error(f'Error generating SQL: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/chat/execute', methods=['POST'])
    def execute_sql():
        """Execute SQL query safely"""
        try:
            data = request.get_json()
            sql_query = data['sql_query']
            project_id = data['project_id']
            
            results = chat_service.execute_sql_safely(sql_query, project_id)
            
            return jsonify({
                'success': True,
                'results': results
            })
        except Exception as e:
            app.logger.error(f'Error executing SQL: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/chat/feedback', methods=['POST'])
    def submit_feedback():
        """Submit user feedback for NLQ pipeline"""
        try:
            data = request.get_json()
            
            feedback = NLQFeedback(
                project_id=data['project_id'],
                nlq_text=data['nlq_text'],
                extracted_entities=data.get('extracted_entities'),
                mapped_tables=data.get('mapped_tables'),
                generated_sql=data.get('generated_sql'),
                sql_results=data.get('sql_results'),
                rating=data.get('rating'),
                feedback_type=data.get('feedback_type'),
                comment=data.get('comment'),
                user_id=data.get('user_id', 'anonymous')
            )
            db.session.add(feedback)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Feedback submitted successfully'
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error submitting feedback: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 6: ADMIN PANEL ===================
    
    @app.route('/api/admin/tables', methods=['GET'])
    def admin_browse_tables():
        """Browse database tables with pagination"""
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 50))
            
            tables = admin_service.browse_tables(page=page, per_page=per_page)
            return jsonify({
                'success': True,
                'tables': tables
            })
        except Exception as e:
            app.logger.error(f'Error browsing tables: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/admin/execute', methods=['POST'])
    def admin_execute_sql():
        """Execute SQL with admin privileges"""
        try:
            data = request.get_json()
            sql = data['sql']
            
            # Validate SQL for safety
            if not admin_service.validate_sql_safety(sql):
                return jsonify({'success': False, 'error': 'Unsafe SQL query'}), 400
            
            results = admin_service.execute_sql(sql)
            return jsonify({
                'success': True,
                'results': results
            })
        except Exception as e:
            app.logger.error(f'Error executing admin SQL: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/admin/health', methods=['GET'])
    def system_health():
        """Get system health metrics"""
        try:
            health = admin_service.get_system_health()
            return jsonify({
                'success': True,
                'health': health
            })
        except Exception as e:
            app.logger.error(f'Error getting system health: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== STATIC FILES ===================
    
    @app.route('/')
    def serve_frontend():
        """Serve React frontend"""
        return send_from_directory('static', 'index.html')
    
    @app.route('/<path:path>')
    def serve_static(path):
        """Serve static files"""
        return send_from_directory('static', path)
    
    # =================== UTILITY ENDPOINTS ===================
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @app.route('/api/config', methods=['GET'])
    def get_client_config():
        """Get client-side configuration"""
        return jsonify({
            'success': True,
            'config': {
                'embedding_models': list(app.config['EMBEDDING_CONFIG']['backends'].keys()),
                'search_types': ['keyword', 'semantic', 'hybrid'],
                'max_file_size': app.config['SECURITY_CONFIG']['max_file_size'],
                'allowed_extensions': app.config['SECURITY_CONFIG']['allowed_file_extensions']
            }
        })
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app

# Run application
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)