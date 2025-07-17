# services/embedding_service.py
import os
import numpy as np
import json
import pickle
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Optional: OpenAI embeddings
try:
    import openai
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config import Config
from models import db, Project, Table, Column, DictionaryEntry, Embedding, Index

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating and managing embeddings and search indexes"""
    
    def __init__(self, app=None):
        self.app = app
        self.embedding_models = {}
        self.indexes = {}
        self.job_status = {}  # Track async jobs
        
        # Initialize default models
        self._init_sentence_transformer()
        if OPENAI_AVAILABLE:
            self._init_openai_client()
        
        # Ensure index directory exists
        os.makedirs('indexes', exist_ok=True)
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer model"""
        try:
            default_model = Config.EMBEDDING_CONFIG['default_model']
            self.sentence_model = SentenceTransformer(default_model)
            logger.info(f"Loaded sentence transformer: {default_model}")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {str(e)}")
            self.sentence_model = None
    
    def _init_openai_client(self):
        """Initialize OpenAI client for embeddings"""
        try:
            llm_config = Config.LLM_CONFIG['azure']
            self.openai_client = AzureOpenAI(
                api_key=llm_config['api_key'],
                api_version=llm_config['api_version'],
                azure_endpoint=llm_config['endpoint']
            )
            logger.info("OpenAI client initialized for embeddings")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
            self.openai_client = None
    
    def create_embeddings_batch(self, project_id: int, model_name: str, 
                               object_types: List[str]) -> str:
        """Create embeddings in batch for multiple object types"""
        job_id = str(uuid.uuid4())
        
        # Track job status
        self.job_status[job_id] = {
            'status': 'running',
            'progress': 0.0,
            'message': 'Starting embedding creation',
            'created_embeddings': 0,
            'total_objects': 0,
            'started_at': datetime.utcnow()
        }
        
        # Start background job with proper Flask context
        if self.app:
            thread = threading.Thread(
                target=self._run_embedding_job_with_context,
                args=(job_id, project_id, model_name, object_types)
            )
        else:
            thread = threading.Thread(
                target=self._run_embedding_job,
                args=(job_id, project_id, model_name, object_types)
            )
        
        thread.daemon = True
        thread.start()
        
        return job_id
    
    def _run_embedding_job_with_context(self, job_id: str, project_id: int, 
                                      model_name: str, object_types: List[str]):
        """Run embedding job with Flask application context"""
        with self.app.app_context():
            self._run_embedding_job(job_id, project_id, model_name, object_types)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of embedding job"""
        return self.job_status.get(job_id, {
            'status': 'not_found',
            'message': 'Job not found'
        })
    
    def _run_embedding_job(self, job_id: str, project_id: int, 
                          model_name: str, object_types: List[str]):
        """Run embedding creation job in background"""
        try:
            self.job_status[job_id]['status'] = 'running'
            
            # Get all objects to embed
            objects_to_embed = self._get_objects_for_embedding(project_id, object_types)
            total_objects = len(objects_to_embed)
            
            self.job_status[job_id].update({
                'total_objects': total_objects,
                'message': f'Processing {total_objects} objects'
            })
            
            if total_objects == 0:
                self.job_status[job_id].update({
                    'status': 'completed',
                    'progress': 1.0,
                    'message': 'No objects found to embed',
                    'completed_at': datetime.utcnow()
                })
                return
            
            created_count = 0
            batch_size = Config.EMBEDDING_CONFIG['batch_size']
            
            # Process in batches
            for i in range(0, total_objects, batch_size):
                batch = objects_to_embed[i:i + batch_size]
                
                # Create embeddings for batch
                batch_embeddings = self._create_embeddings_for_batch(
                    batch, model_name, project_id
                )
                
                # Save to database
                for embedding_data in batch_embeddings:
                    embedding = Embedding(**embedding_data)
                    db.session.add(embedding)
                
                created_count += len(batch_embeddings)
                
                # Update progress
                progress = created_count / total_objects
                self.job_status[job_id].update({
                    'progress': round(progress, 3),
                    'created_embeddings': created_count,
                    'message': f'Created {created_count}/{total_objects} embeddings'
                })
                
                # Commit batch
                try:
                    db.session.commit()
                except Exception as e:
                    logger.error(f"Error committing embeddings batch: {str(e)}")
                    db.session.rollback()
                    raise
            
            # Job completed
            self.job_status[job_id].update({
                'status': 'completed',
                'progress': 1.0,
                'message': f'Successfully created {created_count} embeddings',
                'completed_at': datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Embedding job {job_id} failed: {str(e)}")
            self.job_status[job_id].update({
                'status': 'failed',
                'message': f'Job failed: {str(e)}',
                'error': str(e),
                'failed_at': datetime.utcnow()
            })
            try:
                db.session.rollback()
            except:
                pass  # Session might not be available
    
    def _get_objects_for_embedding(self, project_id: int, object_types: List[str]) -> List[Dict]:
        """Get all objects that need embeddings"""
        objects = []
        
        if 'tables' in object_types:
            tables = db.session.query(Table).join(
                Table.source
            ).filter(
                Table.source.has(project_id=project_id)
            ).all()
            
            for table in tables:
                text = self._build_table_text(table)
                objects.append({
                    'object_type': 'table',
                    'object_id': table.id,
                    'text': text,
                    'metadata': {
                        'table_name': table.name,
                        'source_id': table.source_id
                    }
                })
        
        if 'columns' in object_types:
            columns = db.session.query(Column).join(
                Column.table
            ).join(
                Table.source
            ).filter(
                Table.source.has(project_id=project_id)
            ).all()
            
            for column in columns:
                text = self._build_column_text(column)
                objects.append({
                    'object_type': 'column',
                    'object_id': column.id,
                    'text': text,
                    'metadata': {
                        'column_name': column.name,
                        'table_name': column.table.name,
                        'table_id': column.table_id
                    }
                })
        
        if 'dictionary' in object_types:
            dictionary_entries = DictionaryEntry.query.filter_by(
                project_id=project_id
            ).filter(
                DictionaryEntry.status != 'archived'
            ).all()
            
            for entry in dictionary_entries:
                text = self._build_dictionary_text(entry)
                objects.append({
                    'object_type': 'dictionary_entry',
                    'object_id': entry.id,
                    'text': text,
                    'metadata': {
                        'term': entry.term,
                        'category': entry.category,
                        'domain': entry.domain
                    }
                })
        
        return objects
    
    def _build_table_text(self, table: Table) -> str:
        """Build text representation of table for embedding"""
        parts = [table.name]
        
        if table.display_name and table.display_name != table.name:
            parts.append(table.display_name)
        
        if table.description:
            parts.append(table.description)
        
        # Add column names for context
        column_names = [col.name for col in table.columns[:10]]  # Limit to first 10
        if column_names:
            parts.append(f"Columns: {', '.join(column_names)}")
        
        return ' '.join(parts)
    
    def _build_column_text(self, column: Column) -> str:
        """Build text representation of column for embedding"""
        parts = [column.name]
        
        if column.display_name and column.display_name != column.name:
            parts.append(column.display_name)
        
        if column.description:
            parts.append(column.description)
        
        # Add context
        parts.append(f"Table: {column.table.name}")
        parts.append(f"Type: {column.data_type}")
        
        if column.business_category:
            parts.append(f"Category: {column.business_category}")
        
        # Add sample values for context
        if column.sample_values:
            sample_text = ', '.join(str(v) for v in column.sample_values[:5])
            parts.append(f"Examples: {sample_text}")
        
        return ' '.join(parts)
    
    def _build_dictionary_text(self, entry: DictionaryEntry) -> str:
        """Build text representation of dictionary entry for embedding"""
        parts = [entry.term, entry.definition]
        
        if entry.synonyms:
            parts.append(f"Synonyms: {', '.join(entry.synonyms)}")
        
        if entry.abbreviations:
            parts.append(f"Abbreviations: {', '.join(entry.abbreviations)}")
        
        if entry.domain:
            parts.append(f"Domain: {entry.domain}")
        
        return ' '.join(parts)
    
    def _create_embeddings_for_batch(self, batch: List[Dict], model_name: str, 
                                   project_id: int) -> List[Dict]:
        """Create embeddings for a batch of objects"""
        texts = [obj['text'] for obj in batch]
        
        # Generate embeddings based on model type
        if model_name.startswith('sentence-transformers/'):
            vectors = self._create_sentence_transformer_embeddings(texts, model_name)
        elif model_name.startswith('openai/'):
            vectors = self._create_openai_embeddings(texts, model_name)
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        
        # Create embedding records
        embeddings = []
        for i, obj in enumerate(batch):
            vector = vectors[i]
            
            embedding_data = {
                'project_id': project_id,
                'object_type': obj['object_type'],
                'object_id': obj['object_id'],
                'object_text': obj['text'][:1000],  # Limit text length
                'model_name': model_name,
                'vector_dimension': len(vector),
                'vector': pickle.dumps(vector),  # Serialize vector
                'vector_norm': round(float(np.linalg.norm(vector)), 3)
            }
            embeddings.append(embedding_data)
        
        return embeddings
    
    def _create_sentence_transformer_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Create embeddings using sentence transformers"""
        if not self.sentence_model:
            raise ValueError("Sentence transformer not available")
        
        # Load specific model if different from default
        if model_name != Config.EMBEDDING_CONFIG['default_model']:
            model = SentenceTransformer(model_name.replace('sentence-transformers/', ''))
        else:
            model = self.sentence_model
        
        # Create embeddings
        embeddings = model.encode(texts, batch_size=Config.EMBEDDING_CONFIG['batch_size'])
        return embeddings
    
    def _create_openai_embeddings(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Create embeddings using OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI client not available")
        
        embeddings = []
        
        # Process in smaller batches for API limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            response = self.openai_client.embeddings.create(
                input=batch_texts,
                model=model_name.replace('openai/', '')
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def build_index(self, index_id: int) -> Dict[str, Any]:
        """Build search index from embeddings"""
        try:
            index_record = Index.query.get(index_id)
            if not index_record:
                raise ValueError(f"Index {index_id} not found")
            
            # Update status
            index_record.status = 'building'
            index_record.build_progress = 0.0
            db.session.commit()
            
            # Get embeddings for this index
            embeddings = self._get_embeddings_for_index(index_record)
            
            if not embeddings:
                raise ValueError("No embeddings found for index")
            
            # Build index based on type
            if index_record.index_type == 'faiss':
                index_data = self._build_faiss_index(embeddings, index_record)
            elif index_record.index_type == 'tfidf':
                index_data = self._build_tfidf_index(embeddings, index_record)
            elif index_record.index_type == 'bm25':
                index_data = self._build_bm25_index(embeddings, index_record)
            else:
                raise ValueError(f"Unsupported index type: {index_record.index_type}")
            
            # Save index files
            index_path = self._save_index_files(index_record.id, index_data)
            
            # Update index record
            index_record.status = 'ready'
            index_record.build_progress = 1.0
            index_record.total_vectors = len(embeddings)
            index_record.index_file_path = index_path['index']
            index_record.metadata_file_path = index_path['metadata']
            index_record.index_size_mb = round(
                os.path.getsize(index_path['index']) / (1024 * 1024), 2
            ) if os.path.exists(index_path['index']) else 0
            
            db.session.commit()
            
            # Cache index in memory
            self.indexes[index_id] = index_data
            
            return {
                'success': True,
                'index_id': index_id,
                'total_vectors': len(embeddings),
                'index_type': index_record.index_type
            }
            
        except Exception as e:
            logger.error(f"Error building index {index_id}: {str(e)}")
            
            # Update error status
            if index_record:
                index_record.status = 'error'
                db.session.commit()
            
            raise Exception(f"Index building failed: {str(e)}")
    
    def _get_embeddings_for_index(self, index_record: Index) -> List[Dict]:
        """Get embeddings that should be included in the index"""
        query = Embedding.query.filter_by(project_id=index_record.project_id)
        
        # Filter by object scope if specified
        object_scope = index_record.object_scope or {}
        if 'object_types' in object_scope:
            query = query.filter(Embedding.object_type.in_(object_scope['object_types']))
        
        # Filter by embedding model if specified
        if index_record.embedding_model:
            query = query.filter_by(model_name=index_record.embedding_model)
        
        embeddings = query.all()
        
        # Convert to dict format with deserialized vectors
        embedding_data = []
        for emb in embeddings:
            try:
                vector = pickle.loads(emb.vector)
                embedding_data.append({
                    'id': emb.id,
                    'object_type': emb.object_type,
                    'object_id': emb.object_id,
                    'object_text': emb.object_text,
                    'vector': vector,
                    'metadata': {
                        'model_name': emb.model_name,
                        'vector_dimension': emb.vector_dimension,
                        'vector_norm': emb.vector_norm
                    }
                })
            except Exception as e:
                logger.warning(f"Error deserializing embedding {emb.id}: {str(e)}")
                continue
        
        return embedding_data
    
    def _build_faiss_index(self, embeddings: List[Dict], index_record: Index) -> Dict[str, Any]:
        """Build FAISS index for vector similarity search"""
        vectors = np.array([emb['vector'] for emb in embeddings]).astype('float32')
        dimension = vectors.shape[1]
        
        # Choose FAISS index type based on configuration
        index_type = index_record.build_params.get('index_type', 'IndexFlatIP')
        
        if index_type == 'IndexFlatIP':
            # Inner product for cosine similarity (normalize vectors first)
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(dimension)
        elif index_type == 'IndexFlatL2':
            # L2 distance
            index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IndexIVFFlat':
            # IVF for larger datasets
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(vectors)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")
        
        # Add vectors to index
        index.add(vectors)
        
        # Build metadata mapping
        metadata = {
            'embedding_ids': [emb['id'] for emb in embeddings],
            'object_types': [emb['object_type'] for emb in embeddings],
            'object_ids': [emb['object_id'] for emb in embeddings],
            'object_texts': [emb['object_text'] for emb in embeddings],
            'dimension': dimension,
            'total_vectors': len(embeddings),
            'index_type': index_type
        }
        
        return {
            'type': 'faiss',
            'index': index,
            'metadata': metadata
        }
    
    def _build_tfidf_index(self, embeddings: List[Dict], index_record: Index) -> Dict[str, Any]:
        """Build TF-IDF index for text similarity search"""
        texts = [emb['object_text'] for emb in embeddings]
        
        # Configure TF-IDF
        max_features = index_record.build_params.get('max_features', 10000)
        ngram_range = tuple(index_record.build_params.get('ngram_range', [1, 2]))
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
        
        # Fit and transform texts
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        metadata = {
            'embedding_ids': [emb['id'] for emb in embeddings],
            'object_types': [emb['object_type'] for emb in embeddings],
            'object_ids': [emb['object_id'] for emb in embeddings],
            'object_texts': [emb['object_text'] for emb in embeddings],
            'vocabulary_size': len(vectorizer.vocabulary_),
            'total_documents': len(texts)
        }
        
        return {
            'type': 'tfidf',
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'metadata': metadata
        }
    
    def _build_bm25_index(self, embeddings: List[Dict], index_record: Index) -> Dict[str, Any]:
        """Build BM25 index (simplified using TF-IDF with BM25-like parameters)"""
        texts = [emb['object_text'] for emb in embeddings]
        
        # Use binary term frequency (similar to BM25)
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            binary=True,  # Binary term frequency
            sublinear_tf=True  # Use log normalization
        )
        
        matrix = vectorizer.fit_transform(texts)
        
        metadata = {
            'embedding_ids': [emb['id'] for emb in embeddings],
            'object_types': [emb['object_type'] for emb in embeddings],
            'object_ids': [emb['object_id'] for emb in embeddings],
            'object_texts': [emb['object_text'] for emb in embeddings],
            'vocabulary_size': len(vectorizer.vocabulary_)
        }
        
        return {
            'type': 'bm25',
            'vectorizer': vectorizer,
            'matrix': matrix,
            'metadata': metadata
        }
    
    def _save_index_files(self, index_id: int, index_data: Dict[str, Any]) -> Dict[str, str]:
        """Save index and metadata files"""
        index_dir = os.path.join('indexes', str(index_id))
        os.makedirs(index_dir, exist_ok=True)
        
        index_path = os.path.join(index_dir, 'index.pkl')
        metadata_path = os.path.join(index_dir, 'metadata.json')
        
        # Save based on index type
        if index_data['type'] == 'faiss':
            faiss_path = os.path.join(index_dir, 'faiss.index')
            faiss.write_index(index_data['index'], faiss_path)
            
            # Save serializable parts
            serializable_data = {
                'type': 'faiss',
                'faiss_path': faiss_path,
                'metadata': index_data['metadata']
            }
            
            with open(index_path, 'wb') as f:
                pickle.dump(serializable_data, f)
        
        else:
            # For TF-IDF and BM25, save the whole structure
            with open(index_path, 'wb') as f:
                pickle.dump(index_data, f)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(index_data['metadata'], f, indent=2)
        
        return {
            'index': index_path,
            'metadata': metadata_path
        }
    
    def load_index(self, index_id: int) -> Dict[str, Any]:
        """Load index from files"""
        if index_id in self.indexes:
            return self.indexes[index_id]
        
        index_record = Index.query.get(index_id)
        if not index_record or index_record.status != 'ready':
            raise ValueError(f"Index {index_id} not ready")
        
        try:
            # Load from file
            with open(index_record.index_file_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Special handling for FAISS
            if index_data['type'] == 'faiss':
                faiss_index = faiss.read_index(index_data['faiss_path'])
                index_data['index'] = faiss_index
            
            # Cache in memory
            self.indexes[index_id] = index_data
            
            return index_data
            
        except Exception as e:
            logger.error(f"Error loading index {index_id}: {str(e)}")
            raise
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available embedding models"""
        models = {
            'sentence_transformers': [
                'sentence-transformers/all-MiniLM-L6-v2',
                'sentence-transformers/all-mpnet-base-v2',
                'sentence-transformers/distilbert-base-nli-mean-tokens',
                'sentence-transformers/paraphrase-MiniLM-L6-v2'
            ],
            'openai': []
        }
        
        if OPENAI_AVAILABLE and self.openai_client:
            models['openai'] = [
                'openai/text-embedding-ada-002',
                'openai/text-embedding-3-small',
                'openai/text-embedding-3-large'
            ]
        
        return models
    
    def delete_embeddings(self, project_id: int, object_type: Optional[str] = None) -> int:
        """Delete embeddings for a project or specific object type"""
        try:
            query = Embedding.query.filter_by(project_id=project_id)
            
            if object_type:
                query = query.filter_by(object_type=object_type)
            
            count = query.count()
            query.delete()
            db.session.commit()
            
            return count
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting embeddings: {str(e)}")
            raise
    
    def rebuild_index(self, index_id: int) -> Dict[str, Any]:
        """Rebuild an existing index"""
        try:
            # Remove from cache
            if index_id in self.indexes:
                del self.indexes[index_id]
            
            # Rebuild
            return self.build_index(index_id)
            
        except Exception as e:
            logger.error(f"Error rebuilding index {index_id}: {str(e)}")
            raise