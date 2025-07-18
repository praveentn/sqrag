# C:\Users\Praveen.TN\Downloads\Experiments\sqrag\QueryForge\app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os
import logging

# Database and models
from models import db, Project, DataSource, Table, Column, DictionaryEntry, Embedding, Index, SearchLog

# Services
from services.data_source_service import DataSourceService
from services.embedding_service import EmbeddingService
from services.search_service import SearchService
from services.dictionary_service import DictionaryService
from services.admin_service import AdminService
from services.chat_service import ChatService

# Configuration
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
_services_initialized = False
data_source_service = None
embedding_service = None
search_service = None
dictionary_service = None
admin_service = None
chat_service = None

def create_app(config_name='development'):
    """Application factory"""
    global _services_initialized, data_source_service, embedding_service, search_service, dictionary_service, admin_service, chat_service
    
    app = Flask(__name__, static_folder='static')
    
    # Configuration
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize database
    db.init_app(app)
    
    # Initialize services within app context
    with app.app_context():
        if not _services_initialized:
            try:
                data_source_service = DataSourceService()
                embedding_service = EmbeddingService(app)
                search_service = SearchService()
                dictionary_service = DictionaryService()
                admin_service = AdminService()
                chat_service = ChatService()
                _services_initialized = True
                app.logger.info("All services initialized successfully")
            except Exception as e:
                app.logger.error(f"Error initializing services: {str(e)}")
                # Continue with app creation even if services fail
    
    # Routes
    
    @app.route('/')
    def serve_frontend():
        """Serve the main frontend application"""
        return send_from_directory(app.static_folder, 'index.html')
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services_initialized': _services_initialized
        })
    
    # =================== PROJECTS ===================
    
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
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating project: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>', methods=['GET'])
    def get_project(project_id):
        """Get a specific project"""
        try:
            project = db.session.get(Project, project_id)
            if not project:
                return jsonify({'success': False, 'error': 'Project not found'}), 404
            
            return jsonify({
                'success': True,
                'project': project.to_dict()
            })
        except Exception as e:
            app.logger.error(f'Error fetching project: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== DATA SOURCES ===================
    
    @app.route('/api/projects/<int:project_id>/sources', methods=['GET'])
    def get_sources(project_id):
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
                    'ingest_status': source.ingest_status,
                    'created_at': source.created_at.isoformat() if source.created_at else None,
                    'row_count': getattr(source, 'row_count', 0),
                    'file_size_mb': round(getattr(source, 'file_size_bytes', 0) / (1024*1024), 2) if hasattr(source, 'file_size_bytes') else 0
                }
                source_list.append(source_data)
            
            return jsonify({
                'success': True,
                'sources': source_list
            })
        except Exception as e:
            app.logger.error(f'Error fetching sources: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/sources', methods=['POST'])
    def upload_file(project_id):
        """Upload and process a file"""
        try:
            # Validate project exists
            project = db.session.get(Project, project_id)
            if not project:
                return jsonify({
                    'success': False,
                    'error': f'Project {project_id} not found'
                }), 404
            
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No file provided'
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            # Process the file
            result = data_source_service.process_uploaded_file(file, project_id)
            
            return jsonify(result)
            
        except Exception as e:
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
                    'synonyms': entry.synonyms or [],
                    'abbreviations': entry.abbreviations or [],
                    'status': entry.status,
                    'is_auto_generated': entry.is_auto_generated,
                    'confidence_score': round(float(entry.confidence_score), 3) if entry.confidence_score else None,
                    'created_by': entry.created_by,
                    'created_at': entry.created_at.isoformat() if entry.created_at else None
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
        """Create a new dictionary entry"""
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
                status='approved',
                created_by=data.get('created_by', 'user')
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
                },
                'message': 'Dictionary entry created successfully'
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating dictionary entry: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/dictionary/<int:entry_id>', methods=['PUT'])
    def update_dictionary_entry(entry_id):
        """Update a dictionary entry"""
        try:
            entry = db.session.get(DictionaryEntry, entry_id)
            if not entry:
                return jsonify({'success': False, 'error': 'Entry not found'}), 404
                
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
            entry = db.session.get(DictionaryEntry, entry_id)
            if not entry:
                return jsonify({'success': False, 'error': 'Entry not found'}), 404
                
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
            
            # Return embeddings list as well for debugging
            embeddings_list = []
            for emb in embeddings[:50]:  # Limit to first 50 for performance
                embeddings_list.append({
                    'id': emb.id,
                    'object_type': emb.object_type,
                    'object_id': emb.object_id,
                    'model_name': emb.model_name,
                    'created_at': emb.created_at.isoformat() if emb.created_at else None
                })
         
            return jsonify({
                'success': True,
                'total_embeddings': len(embeddings),
                'models_used': len(model_counts),
                'object_types': len(object_type_counts),
                'model_breakdown': model_counts,
                'object_type_breakdown': object_type_counts,
                'embeddings': embeddings_list  # For debugging
            })
        except Exception as e:
            app.logger.error(f'Error getting embeddings status: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # ADD MISSING DELETE ENDPOINT FOR EMBEDDINGS
    @app.route('/api/projects/<int:project_id>/embeddings', methods=['DELETE'])
    def delete_embeddings(project_id):
        """Delete embeddings for a project"""
        try:
            object_type = request.args.get('object_type')
            
            # Validate project exists
            project = db.session.get(Project, project_id)
            if not project:
                return jsonify({
                    'success': False,
                    'error': f'Project {project_id} not found'
                }), 404
            
            # Delete embeddings
            deleted_count = embedding_service.delete_embeddings(project_id, object_type)
            
            return jsonify({
                'success': True,
                'deleted_count': deleted_count,
                'message': f'Deleted {deleted_count} embeddings'
            })
        except Exception as e:
            app.logger.error(f'Error deleting embeddings: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/projects/<int:project_id>/embeddings/batch', methods=['POST'])
    def create_embeddings_batch(project_id):
        """Create embeddings in batch"""
        try:
            data = request.get_json()
            model_name = data.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            object_types = data.get('object_types', ['tables', 'columns', 'dictionary'])
            
            # Validate project exists
            project = db.session.get(Project, project_id)
            if not project:
                return jsonify({
                    'success': False,
                    'error': f'Project {project_id} not found'
                }), 404
            
            # Start background job
            job_id = embedding_service.create_embeddings_batch(
                project_id, model_name, object_types
            )
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'message': 'Embedding creation started',
                'model_name': model_name,
                'object_types': object_types
            })
        except Exception as e:
            app.logger.error(f'Error creating embeddings batch: {str(e)}')
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
            
            # FIXED: Use object_types instead of object_scope
            index = Index(
                project_id=project_id,
                name=data['name'],
                index_type=data.get('index_type', 'faiss'),
                embedding_model=data.get('embedding_model'),
                object_types=data.get('object_types', []),  # FIXED: was object_scope
                config_json=data.get('config_json', {}),    # FIXED: was build_params
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
                    'index': index.to_dict(),
                    'build_result': result
                })
            except Exception as build_error:
                db.session.rollback()
                app.logger.error(f'Error building index: {str(build_error)}')
                return jsonify({
                    'success': False,
                    'error': f'Index creation failed: {str(build_error)}'
                }), 500
                
        except Exception as e:
            db.session.rollback()
            app.logger.error(f'Error creating index: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== SEARCH ===================
    
    @app.route('/api/projects/<int:project_id>/search', methods=['POST'])
    def search_project(project_id):
        """Search within a project"""
        try:
            data = request.get_json()
            query = data.get('query', '')
            search_params = data.get('params', {})
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'Query is required'
                }), 400
            
            # Perform search
            results = search_service.search(query, project_id, search_params)
            
            return jsonify({
                'success': True,
                **results
            })
        except Exception as e:
            app.logger.error(f'Error searching project {project_id}: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== CHAT ===================
    
    @app.route('/api/projects/<int:project_id>/chat', methods=['POST'])
    def chat_with_data(project_id):
        """Chat with project data using NL to SQL"""
        try:
            data = request.get_json()
            query = data.get('query', '')
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'Query is required'
                }), 400
            
            # Process natural language query
            result = chat_service.process_query(query, project_id)
            
            return jsonify({
                'success': True,
                **result
            })
        except Exception as e:
            app.logger.error(f'Error processing chat query: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # =================== ADMIN ===================
    
    @app.route('/api/admin/stats', methods=['GET'])
    def get_admin_stats():
        """Get system statistics"""
        try:
            result = admin_service.get_system_stats()
            return jsonify(result)
        except Exception as e:
            app.logger.error(f'Error getting admin stats: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/admin/execute', methods=['POST'])
    def execute_admin_query():
        """Execute admin SQL query"""
        try:
            data = request.get_json()
            query = data.get('query', '')
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'Query is required'
                }), 400
            
            result = admin_service.execute_query(query)
            return jsonify(result)
        except Exception as e:
            app.logger.error(f'Error executing admin query: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/admin/table/<int:table_id>', methods=['GET'])
    def get_admin_table_details(table_id):
        """Get detailed information about a table"""
        try:
            result = admin_service.get_table_details(table_id)
            return jsonify(result)
        except Exception as e:
            app.logger.error(f'Error getting table details: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/admin/schema', methods=['GET'])
    def get_admin_schema():
        """Get complete database schema"""
        try:
            result = admin_service.get_database_schema()
            return jsonify(result)
        except Exception as e:
            app.logger.error(f'Error getting schema: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/admin/cleanup', methods=['POST'])
    def admin_cleanup():
        """Clean up orphaned data"""
        try:
            result = admin_service.cleanup_orphaned_data()
            return jsonify(result)
        except Exception as e:
            app.logger.error(f'Error during cleanup: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500

    # =================== DEBUG ENDPOINTS ===================

    @app.route('/api/debug/info', methods=['GET'])
    def debug_info():
        """Debug information endpoint"""
        try:
            # Basic counts
            projects_count = Project.query.count()
            sources_count = DataSource.query.count()
            
            return jsonify({
                'success': True,
                'debug_info': {
                    'services_initialized': _services_initialized,
                    'projects_count': projects_count,
                    'sources_count': sources_count,
                    'config_name': app.config.get('ENV', 'unknown'),
                    'database_url': str(db.engine.url).split('@')[0] + '@***' if '@' in str(db.engine.url) else str(db.engine.url)
                }
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    

    @app.route('/api/debug/embedding-service', methods=['GET'])
    def debug_embedding_service():
        """Debug endpoint to check embedding service status"""
        try:
            models = embedding_service.get_available_models()
            job_statuses = embedding_service.job_status
            
            return jsonify({
                'success': True,
                'available_models': models,
                'active_jobs': len(job_statuses),
                'job_statuses': job_statuses,
                'sentence_model_loaded': embedding_service.sentence_model is not None,
                'openai_client_available': embedding_service.openai_client is not None
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/debug/database', methods=['GET'])
    def debug_database():
        """Debug endpoint to check database status"""
        try:
            # Test basic queries
            projects_count = Project.query.count()
            tables_count = Table.query.count()
            columns_count = Column.query.count()
            embeddings_count = Embedding.query.count()
            dictionary_count = DictionaryEntry.query.count()
            
            # Test a simple join
            tables_with_sources = db.session.query(Table).join(DataSource).count()
            
            return jsonify({
                'success': True,
                'counts': {
                    'projects': projects_count,
                    'tables': tables_count,
                    'columns': columns_count,
                    'embeddings': embeddings_count,
                    'dictionary_entries': dictionary_count,
                    'tables_with_sources': tables_with_sources
                },
                'database_url': str(db.engine.url).replace(str(db.engine.url).split('@')[0].split('://')[-1] + '@', '***@') if '@' in str(db.engine.url) else str(db.engine.url)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

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
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)