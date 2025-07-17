# app.py
import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback
import uuid
from datetime import datetime

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

def create_app(config_name='development'):
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
    
    # Initialize services with app context
    with app.app_context():
        data_source_service = DataSourceService()
        dictionary_service = DictionaryService()
        embedding_service = EmbeddingService(app)  # Pass app for context
        search_service = SearchService()
        chat_service = ChatService()
        admin_service = AdminService()
    
    # =================== ERROR HANDLERS ===================
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'success': False, 'error': 'Bad request', 'message': str(error)}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'success': False, 'error': 'Not found', 'message': str(error)}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        app.logger.error(f'Server Error: {error}')
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    # =================== UTILITY ROUTES ===================
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        })
    
    @app.route('/')
    def serve_index():
        """Serve the main application"""
        return send_from_directory('static', 'index.html')
    
    @app.route('/<path:filename>')
    def serve_static(filename):
        """Serve static files"""
        return send_from_directory('static', filename)
    
    # =================== TAB 1: PROJECTS ===================
    
    @app.route('/api/projects', methods=['GET'])
    def get_projects():
        """Get all projects"""
        try:
            projects = Project.query.all()
            project_list = []
            
            for project in projects:
                project_data = {
                    'id': project.id,
                    'name': project.name,
                    'description': project.description,
                    'status': project.status,
                    'created_at': project.created_at.isoformat() if project.created_at else None,
                    'updated_at': project.updated_at.isoformat() if project.updated_at else None,
                    'sources_count': project.sources,
                    'dictionary_entries_count': project.dictionary_entries
                }
                project_list.append(project_data)
            
            return jsonify({
                'success': True,
                'projects': project_list
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
                owner=data.get('owner', 'default_user'),  # Provide default owner
                status='active'
            )
            
            db.session.add(project)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'project': {
                    'id': project.id,
                    'name': project.name,
                    'description': project.description,
                    'owner': project.owner,
                    'status': project.status,
                    'created_at': project.created_at.isoformat(),
                    'sources_count': 0,
                    'dictionary_entries_count': 0
                }
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating project: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>', methods=['PUT'])
    def update_project(project_id):
        """Update a project"""
        try:
            project = Project.query.get_or_404(project_id)
            data = request.get_json()
            
            project.name = data.get('name', project.name)
            project.description = data.get('description', project.description)
            project.status = data.get('status', project.status)
            # Note: owner can be updated if provided, otherwise keep existing
            if 'owner' in data:
                project.owner = data['owner']
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'project': {
                    'id': project.id,
                    'name': project.name,
                    'description': project.description,
                    'owner': project.owner,
                    'status': project.status,
                    'updated_at': project.updated_at.isoformat()
                }
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error updating project: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>', methods=['DELETE'])
    def delete_project(project_id):
        """Delete a project"""
        try:
            project = Project.query.get_or_404(project_id)
            db.session.delete(project)
            db.session.commit()
            
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error deleting project: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 2: DATA SOURCES ===================
    
    @app.route('/api/projects/<int:project_id>/sources', methods=['GET'])
    def get_data_sources(project_id):
        """Get data sources for a project"""
        try:
            sources = DataSource.query.filter_by(project_id=project_id).all()
            source_list = []
            
            for source in sources:
                source_data = {
                    'id': source.id,
                    'name': source.name,
                    'type': source.type,
                    'subtype': source.subtype,
                    'status': source.ingest_status,  # Use the actual column but return as 'status'
                    'file_path': source.file_path,
                    'created_at': source.created_at.isoformat() if source.created_at else None,
                    'table_count': len(source.tables)
                }
                source_list.append(source_data)
            
            return jsonify({
                'success': True,
                'sources': source_list
            })
        except Exception as e:
            app.logger.error(f'Error fetching data sources: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/sources/upload', methods=['POST'])
    def upload_file(project_id):
        """Upload a file data source"""
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            # Validate file type
            allowed_extensions = {'csv', 'xlsx', 'xls', 'json'}
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            
            if file_extension not in allowed_extensions:
                return jsonify({
                    'success': False, 
                    'error': f'Unsupported file type. Allowed: {", ".join(allowed_extensions)}'
                }), 400
            
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Create data source record
            source = DataSource(
                project_id=project_id,
                name=filename,
                type='file',
                subtype=file_extension,
                file_path=file_path,
                ingest_status='uploaded'  # Use the correct column name
            )
            
            db.session.add(source)
            db.session.flush()  # Get the ID
            
            try:
                # Process the file
                tables_created = data_source_service.process_uploaded_file(source.id, file_path)
                source.ingest_status = 'processed'  # Use correct column name
                
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'source': {
                        'id': source.id,
                        'name': source.name,
                        'type': source.type,
                        'subtype': source.subtype,
                        'status': source.ingest_status,  # Return as 'status' for API consistency
                        'tables_created': tables_created
                    }
                })
                
            except Exception as e:
                source.ingest_status = 'error'  # Use correct column name
                db.session.commit()
                app.logger.error(f'Error processing file: {str(e)}')
                return jsonify({
                    'success': False, 
                    'error': f'Failed to process file: {str(e)}'
                }), 500
                
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error uploading file: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/tables', methods=['GET'])
    def get_tables(project_id):
        """Get tables for a project"""
        try:
            tables = db.session.query(Table).join(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            
            table_list = []
            for table in tables:
                table_data = {
                    'id': table.id,
                    'name': table.name,
                    'display_name': table.display_name,
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'source_name': table.source.name,
                    'source_type': table.source.type
                }
                table_list.append(table_data)
            
            return jsonify({
                'success': True,
                'tables': table_list
            })
        except Exception as e:
            app.logger.error(f'Error fetching tables: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/tables/<int:table_id>/columns', methods=['GET'])
    def get_columns(table_id):
        """Get columns for a table"""
        try:
            columns = Column.query.filter_by(table_id=table_id).all()
            column_list = []
            
            for column in columns:
                column_data = {
                    'id': column.id,
                    'name': column.name,
                    'display_name': column.display_name,
                    'data_type': column.data_type,
                    'is_nullable': column.is_nullable,
                    'distinct_count': column.distinct_count,
                    'sample_values': column.sample_values,
                    'business_category': column.business_category,
                    'pii_flag': column.pii_flag
                }
                column_list.append(column_data)
            
            return jsonify({
                'success': True,
                'columns': column_list
            })
        except Exception as e:
            app.logger.error(f'Error fetching columns: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 3: DICTIONARY ===================
    
    @app.route('/api/projects/<int:project_id>/dictionary', methods=['GET'])
    def get_dictionary_entries(project_id):
        """Get dictionary entries for a project"""
        try:
            entries = DictionaryEntry.query.filter_by(project_id=project_id).all()
            entry_list = []
            
            for entry in entries:
                entry_data = {
                    'id': entry.id,
                    'term': entry.term,
                    'definition': entry.definition,
                    'category': entry.category,
                    'domain': entry.domain,
                    'synonyms': entry.synonyms,
                    'abbreviations': entry.abbreviations,
                    'status': entry.status,
                    'is_auto_generated': entry.is_auto_generated,
                    'confidence_score': entry.confidence_score,
                    'created_at': entry.created_at.isoformat() if entry.created_at else None,
                    'created_by': entry.created_by
                }
                entry_list.append(entry_data)
            
            return jsonify({
                'success': True,
                'entries': entry_list
            })
        except Exception as e:
            app.logger.error(f'Error fetching dictionary entries: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/dictionary', methods=['POST'])
    def create_dictionary_entry(project_id):
        """Create a dictionary entry"""
        try:
            data = request.get_json()
            
            entry = DictionaryEntry(
                project_id=project_id,
                term=data['term'],
                definition=data['definition'],
                category=data.get('category', 'business_term'),
                domain=data.get('domain'),
                synonyms=data.get('synonyms', []),
                abbreviations=data.get('abbreviations', []),
                status=data.get('status', 'draft'),
                is_auto_generated=data.get('is_auto_generated', False),
                confidence_score=data.get('confidence_score'),
                created_by=data.get('created_by')
            )
            
            db.session.add(entry)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'entry': {
                    'id': entry.id,
                    'term': entry.term,
                    'definition': entry.definition,
                    'category': entry.category,
                    'status': entry.status
                }
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating dictionary entry: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/dictionary/<int:entry_id>', methods=['PUT'])
    def update_dictionary_entry(entry_id):
        """Update a dictionary entry"""
        try:
            entry = DictionaryEntry.query.get_or_404(entry_id)
            data = request.get_json()
            
            entry.term = data.get('term', entry.term)
            entry.definition = data.get('definition', entry.definition)
            entry.category = data.get('category', entry.category)
            entry.domain = data.get('domain', entry.domain)
            entry.synonyms = data.get('synonyms', entry.synonyms)
            entry.abbreviations = data.get('abbreviations', entry.abbreviations)
            entry.status = data.get('status', entry.status)
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'entry': {
                    'id': entry.id,
                    'term': entry.term,
                    'definition': entry.definition,
                    'status': entry.status
                }
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error updating dictionary entry: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/dictionary/<int:entry_id>', methods=['DELETE'])
    def delete_dictionary_entry(entry_id):
        """Delete a dictionary entry"""
        try:
            entry = DictionaryEntry.query.get_or_404(entry_id)
            db.session.delete(entry)
            db.session.commit()
            
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error deleting dictionary entry: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/dictionary/suggest', methods=['POST'])
    def generate_dictionary_suggestions(project_id):
        """Generate dictionary suggestions from data sources"""
        try:
            suggestions = dictionary_service.generate_suggestions(project_id)
            
            return jsonify({
                'success': True,
                'suggestions': suggestions
            })
        except Exception as e:
            app.logger.error(f'Error generating dictionary suggestions: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 4: EMBEDDINGS ===================
    
    @app.route('/api/projects/<int:project_id>/embeddings', methods=['GET'])
    def get_embeddings_status(project_id):
        """Get embeddings status for a project"""
        try:
            embeddings = Embedding.query.filter_by(project_id=project_id).all()
            
            # Count by model and object type
            model_counts = {}
            object_type_counts = {}
            
            for embedding in embeddings:
                # Count by model
                if embedding.model_name not in model_counts:
                    model_counts[embedding.model_name] = 0
                model_counts[embedding.model_name] += 1
                
                # Count by object type
                if embedding.object_type not in object_type_counts:
                    object_type_counts[embedding.object_type] = 0
                object_type_counts[embedding.object_type] += 1
            
            return jsonify({
                'success': True,
                'total_embeddings': len(embeddings),
                'models_used': len(model_counts),
                'object_types': len(object_type_counts),
                'model_breakdown': model_counts,
                'object_type_breakdown': object_type_counts
            })
        except Exception as e:
            app.logger.error(f'Error getting embeddings status: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/embeddings/batch', methods=['POST'])
    def create_embeddings_batch(project_id):
        """Create embeddings in batch"""
        try:
            data = request.get_json()
            model_name = data.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            object_types = data.get('object_types', ['tables', 'columns', 'dictionary'])
            
            # Start background job
            job_id = embedding_service.create_embeddings_batch(
                project_id, model_name, object_types
            )
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'message': 'Embedding creation started'
            })
        except Exception as e:
            app.logger.error(f'Error creating embeddings batch: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/embeddings/job/<job_id>/status', methods=['GET'])
    def get_embedding_job_status(job_id):
        """Get status of embedding job"""
        try:
            status = embedding_service.get_job_status(job_id)
            return jsonify({
                'success': True,
                'status': status
            })
        except Exception as e:
            app.logger.error(f'Error getting job status: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/embeddings/models', methods=['GET'])
    def get_available_models():
        """Get available embedding models"""
        try:
            models = embedding_service.get_available_models()
            return jsonify({
                'success': True,
                'models': models
            })
        except Exception as e:
            app.logger.error(f'Error getting available models: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== INDEXES ===================
    
    @app.route('/api/projects/<int:project_id>/indexes', methods=['GET'])
    def get_indexes(project_id):
        """Get indexes for a project"""
        try:
            indexes = Index.query.filter_by(project_id=project_id).all()
            index_list = []
            
            for index in indexes:
                index_data = {
                    'id': index.id,
                    'name': index.name,
                    'index_type': index.index_type,
                    'status': index.status,
                    'total_vectors': index.total_vectors,
                    'embedding_model': index.embedding_model,
                    'build_progress': index.build_progress,
                    'created_at': index.created_at.isoformat() if index.created_at else None
                }
                index_list.append(index_data)
            
            return jsonify({
                'success': True,
                'indexes': index_list
            })
        except Exception as e:
            app.logger.error(f'Error fetching indexes: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/indexes', methods=['POST'])
    def create_index(project_id):
        """Create a new index"""
        try:
            data = request.get_json()
            
            # Create index record
            index = Index(
                project_id=project_id,
                name=data['name'],
                index_type=data.get('index_type', 'faiss'),
                embedding_model=data.get('embedding_model'),
                object_scope=data.get('object_scope', {}),
                build_params=data.get('build_params', {}),
                status='created'
            )
            
            db.session.add(index)
            db.session.flush()  # Get ID
            
            try:
                # Build the index
                result = embedding_service.build_index(index.id)
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'index': {
                        'id': index.id,
                        'name': index.name,
                        'status': index.status,
                        'total_vectors': result['total_vectors']
                    }
                })
                
            except Exception as e:
                db.session.rollback()
                app.logger.error(f'Error building index {index.id}: {str(e)}')
                return jsonify({
                    'success': False, 
                    'error': f'Failed to build index: {str(e)}'
                }), 500
                
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating index: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== SEARCH ===================
    
    @app.route('/api/search', methods=['POST'])
    def perform_search():
        """Perform search across indexes"""
        try:
            data = request.get_json()
            query = data.get('query')
            project_id = data.get('project_id')  # Required for search
            search_params = data.get('search_params', {})
            
            if not project_id:
                return jsonify({
                    'success': False, 
                    'error': 'Project ID required for multi-index search'
                }), 400
            
            if not query:
                return jsonify({
                    'success': False, 
                    'error': 'Query is required'
                }), 400
            
            # Perform search
            results = search_service.search(query, project_id, search_params)
            
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
            sql_query = data['sql']
            project_id = data['project_id']
            
            results = chat_service.execute_sql_safely(sql_query, project_id)
            
            return jsonify({
                'success': True,
                'results': results
            })
        except Exception as e:
            app.logger.error(f'Error executing SQL: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/chat/answer', methods=['POST'])
    def generate_answer():
        """Generate natural language answer from SQL results"""
        try:
            data = request.get_json()
            query = data['query']
            sql = data['sql']
            results = data['results']
            project_id = data['project_id']
            
            answer = chat_service.generate_natural_language_answer(
                query, sql, results, project_id
            )
            
            return jsonify({
                'success': True,
                'answer': answer
            })
        except Exception as e:
            app.logger.error(f'Error generating answer: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== TAB 6: ADMIN ===================
    
    @app.route('/api/admin/health', methods=['GET'])
    def get_system_health():
        """Get system health status"""
        try:
            health = admin_service.get_system_health()
            return jsonify({
                'success': True,
                'health': health
            })
        except Exception as e:
            app.logger.error(f'Error getting system health: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/admin/tables', methods=['GET'])
    def get_admin_tables():
        """Get all tables with pagination for admin view"""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 10, type=int)
            
            tables_query = db.session.query(Table).join(DataSource).join(Project)
            
            # Paginate
            paginated = tables_query.paginate(
                page=page, per_page=per_page, error_out=False
            )
            
            table_list = []
            for table in paginated.items:
                table_data = {
                    'id': table.id,
                    'name': table.name,
                    'project_name': table.source.project.name,
                    'source_name': table.source.name,
                    'source_type': table.source.type,
                    'row_count': table.row_count,
                    'column_count': table.column_count
                }
                table_list.append(table_data)
            
            return jsonify({
                'success': True,
                'tables': table_list,
                'pagination': {
                    'page': page,
                    'pages': paginated.pages,
                    'per_page': per_page,
                    'total': paginated.total,
                    'has_next': paginated.has_next,
                    'has_prev': paginated.has_prev
                }
            })
        except Exception as e:
            app.logger.error(f'Error getting admin tables: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/admin/execute', methods=['POST'])
    def admin_execute_sql():
        """Execute SQL query for admin (with safety checks)"""
        try:
            data = request.get_json()
            sql = data['sql']
            
            # Safety check - only allow SELECT statements
            sql_upper = sql.strip().upper()
            if not sql_upper.startswith('SELECT'):
                return jsonify({
                    'success': False,
                    'error': 'Only SELECT statements are allowed'
                }), 400
            
            results = admin_service.execute_sql_query(sql)
            
            return jsonify({
                'success': True,
                'results': results
            })
        except Exception as e:
            app.logger.error(f'Error executing admin SQL: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== DATABASE INITIALIZATION ===================
    
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("Database tables created successfully")
        except Exception as e:
            app.logger.error(f"Error creating database tables: {str(e)}")
    
    return app

# Create the application
app = create_app('development')  # Explicitly use development config

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)