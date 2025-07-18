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
        """Create embeddings in batch for specified object types"""
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        self.job_status[job_id] = {
            'status': 'starting',
            'progress': 0.0,
            'total_objects': 0,
            'processed_objects': 0,
            'created_embeddings': 0,
            'message': 'Initializing embedding job',
            'started_at': datetime.utcnow(),
            'error': None
        }
        
        # Start background thread
        thread = threading.Thread(
            target=self._create_embeddings_worker,
            args=(job_id, project_id, model_name, object_types)
        )
        thread.daemon = True
        thread.start()
        
        return job_id
    
    def _create_embeddings_worker(self, job_id: str, project_id: int, 
                                model_name: str, object_types: List[str]):
        """Background worker for creating embeddings"""
        try:
            # Update status
            self.job_status[job_id]['status'] = 'gathering_objects'
            self.job_status[job_id]['message'] = 'Gathering objects for embedding'
            
            # Get objects to embed
            objects = self._get_objects_for_embedding(project_id, object_types)
            
            if not objects:
                self.job_status[job_id]['status'] = 'completed'
                self.job_status[job_id]['message'] = 'No objects found for embedding'
                self.job_status[job_id]['progress'] = 1.0
                return
            
            self.job_status[job_id]['total_objects'] = len(objects)
            self.job_status[job_id]['status'] = 'creating_embeddings'
            self.job_status[job_id]['message'] = f'Creating embeddings for {len(objects)} objects'
            
            # Create embeddings in batches
            batch_size = Config.EMBEDDING_CONFIG.get('batch_size', 32)
            created_count = 0
            
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i + batch_size]
                
                try:
                    batch_embeddings = self._create_embeddings_for_batch(
                        batch, model_name, project_id
                    )
                    created_count += len(batch_embeddings)
                    
                    # Update progress
                    processed = min(i + batch_size, len(objects))
                    progress = processed / len(objects)
                    
                    self.job_status[job_id]['processed_objects'] = processed
                    self.job_status[job_id]['created_embeddings'] = created_count
                    self.job_status[job_id]['progress'] = progress
                    self.job_status[job_id]['message'] = f'Processed {processed}/{len(objects)} objects'
                    
                except Exception as batch_error:
                    logger.warning(f"Error processing batch {i}: {str(batch_error)}")
                    continue
            
            # Complete
            self.job_status[job_id]['status'] = 'completed'
            self.job_status[job_id]['progress'] = 1.0
            self.job_status[job_id]['message'] = f'Successfully created {created_count} embeddings'
            self.job_status[job_id]['completed_at'] = datetime.utcnow()
            
            logger.info(f"Embedding job {job_id} completed: {created_count} embeddings created")
            
        except Exception as e:
            logger.error(f"Embedding job {job_id} failed: {str(e)}")
            self.job_status[job_id]['status'] = 'failed'
            self.job_status[job_id]['error'] = str(e)
            self.job_status[job_id]['message'] = f'Job failed: {str(e)}'
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of an embedding job"""
        if job_id not in self.job_status:
            return {
                'status': 'not_found',
                'message': f'Job {job_id} not found'
            }
        
        status = self.job_status[job_id].copy()
        
        # Add duration if job is running or completed
        if 'started_at' in status:
            if status['status'] == 'completed' and 'completed_at' in status:
                duration = (status['completed_at'] - status['started_at']).total_seconds()
            else:
                duration = (datetime.utcnow() - status['started_at']).total_seconds()
            
            status['duration_seconds'] = round(duration, 2)
        
        return status
    
    def _get_objects_for_embedding(self, project_id: int, object_types: List[str]) -> List[Dict]:
        """Get objects that need embeddings"""
        objects = []
        
        # Clean up existing embeddings for selected object types
        for object_type in object_types:
            existing = Embedding.query.filter_by(
                project_id=project_id,
                object_type=object_type
            ).delete()
            if existing > 0:
                logger.info(f"Removed {existing} existing {object_type} embeddings")
        
        db.session.commit()
        
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
        
        logger.info(f"Found {len(objects)} objects to embed for project {project_id}")
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
            # Handle both JSON string and list formats
            if isinstance(column.sample_values, str):
                try:
                    sample_values = json.loads(column.sample_values)
                except:
                    sample_values = [column.sample_values]
            else:
                sample_values = column.sample_values
            
            if sample_values:
                sample_text = ', '.join(str(v) for v in sample_values[:5])
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
        
        # Create embeddings
        if model_name.startswith('openai/'):
            embeddings = self._create_openai_embeddings(texts, model_name)
        else:
            embeddings = self._create_sentence_transformer_embeddings(texts, model_name)
        
        # Save to database
        created_embeddings = []
        for i, (obj, embedding_vector) in enumerate(zip(batch, embeddings)):
            try:
                # Calculate vector norm
                vector_norm = np.linalg.norm(embedding_vector) if embedding_vector is not None else 0
                
                embedding = Embedding(
                    project_id=project_id,
                    object_type=obj['object_type'],
                    object_id=obj['object_id'],
                    object_text=obj['text'],
                    model_name=model_name,
                    vector_dimension=len(embedding_vector) if embedding_vector is not None else 0,
                    vector_norm=round(float(vector_norm), 3),
                    emb__metadata=json.dumps(obj['metadata'])
                )
                
                if embedding_vector is not None:
                    embedding.set_vector(embedding_vector)
                
                db.session.add(embedding)
                created_embeddings.append(embedding)
                
            except Exception as e:
                logger.warning(f"Error creating embedding for object {obj['object_id']}: {str(e)}")
                continue
        
        db.session.commit()
        return created_embeddings
    
    def _create_sentence_transformer_embeddings(self, texts: List[str], model_name: str) -> List[np.ndarray]:
        """Create embeddings using sentence transformers"""
        if not self.sentence_model:
            raise ValueError("Sentence transformer model not available")
        
        try:
            # Load specific model if different from default
            if model_name != Config.EMBEDDING_CONFIG['default_model']:
                if model_name not in self.embedding_models:
                    self.embedding_models[model_name] = SentenceTransformer(model_name)
                model = self.embedding_models[model_name]
            else:
                model = self.sentence_model
            
            # Create embeddings
            embeddings = model.encode(texts, convert_to_numpy=True)
            return [emb.astype('float32') for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Error creating sentence transformer embeddings: {str(e)}")
            raise
    
    def _create_openai_embeddings(self, texts: List[str], model_name: str) -> List[np.ndarray]:
        """Create embeddings using OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI client not available")
        
        try:
            # Clean model name
            clean_model_name = model_name.replace('openai/', '')
            
            # Process in smaller batches for OpenAI
            embeddings = []
            batch_size = 20  # OpenAI recommended batch size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = self.openai_client.embeddings.create(
                    input=batch_texts,
                    model=clean_model_name
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return [np.array(emb, dtype='float32') for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Error creating OpenAI embeddings: {str(e)}")
            raise
    
    def build_index(self, index_id: int) -> Dict[str, Any]:
        """Build search index from embeddings"""
        try:
            index_record = db.session.get(Index, index_id)
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
        
        # Filter by object types if specified
        if index_record.object_types:
            query = query.filter(Embedding.object_type.in_(index_record.object_types))
        
        # Filter by embedding model if specified
        if index_record.embedding_model:
            query = query.filter_by(model_name=index_record.embedding_model)
        
        embeddings = query.all()
        
        results = []
        for emb in embeddings:
            try:
                vector = emb.get_vector()
                if vector is not None:
                    results.append({
                        'id': emb.id,
                        'object_type': emb.object_type,
                        'object_id': emb.object_id,
                        'text': emb.object_text,
                        'vector': vector,
                        'metadata': json.loads(emb.emb__metadata) if emb.emb__metadata else {}
                    })
            except Exception as e:
                logger.warning(f"Error processing embedding {emb.id}: {str(e)}")
                continue
        
        return results
    
    def _build_faiss_index(self, embeddings: List[Dict], index_record: Index) -> Dict:
        """Build FAISS index from embeddings"""
        vectors = np.array([emb['vector'] for emb in embeddings])
        dimension = vectors.shape[1]
        
        # Create FAISS index
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        index.add(vectors)
        
        # Prepare metadata
        metadata = {
            'embedding_ids': [emb['id'] for emb in embeddings],
            'object_types': [emb['object_type'] for emb in embeddings],
            'object_ids': [emb['object_id'] for emb in embeddings],
            'object_texts': [emb['text'] for emb in embeddings]
        }
        
        return {
            'type': 'faiss',
            'index': index,
            'metadata': metadata,
            'dimension': dimension
        }
    
    def _build_tfidf_index(self, embeddings: List[Dict], index_record: Index) -> Dict:
        """Build TF-IDF index from embeddings"""
        texts = [emb['text'] for emb in embeddings]
        
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Prepare metadata
        metadata = {
            'embedding_ids': [emb['id'] for emb in embeddings],
            'object_types': [emb['object_type'] for emb in embeddings],
            'object_ids': [emb['object_id'] for emb in embeddings],
            'object_texts': [emb['text'] for emb in embeddings]
        }
        
        return {
            'type': 'tfidf',
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'metadata': metadata
        }
    
    def _build_bm25_index(self, embeddings: List[Dict], index_record: Index) -> Dict:
        """Build BM25 index (similar to TF-IDF)"""
        return self._build_tfidf_index(embeddings, index_record)
    
    def _save_index_files(self, index_id: int, index_data: Dict) -> Dict[str, str]:
        """Save index files to disk"""
        index_dir = os.path.join('indexes', str(index_id))
        os.makedirs(index_dir, exist_ok=True)
        
        paths = {}
        
        if index_data['type'] == 'faiss':
            # Save FAISS index
            faiss_path = os.path.join(index_dir, 'index.faiss')
            faiss.write_index(index_data['index'], faiss_path)
            
            # Save complete index data (without FAISS object)
            index_copy = index_data.copy()
            index_copy['faiss_path'] = faiss_path
            del index_copy['index']  # Remove FAISS object for pickling
            
            index_path = os.path.join(index_dir, 'index.pkl')
            with open(index_path, 'wb') as f:
                pickle.dump(index_copy, f)
            
            paths['index'] = index_path
            
        else:
            # Save other index types
            index_path = os.path.join(index_dir, 'index.pkl')
            with open(index_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            paths['index'] = index_path
        
        # Save metadata separately
        metadata_path = os.path.join(index_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(index_data['metadata'], f, indent=2)
        
        paths['metadata'] = metadata_path
        
        return paths
    
    def load_index(self, index_id: int) -> Dict:
        """Load index from disk"""
        try:
            index_record = db.session.get(Index, index_id)
            if not index_record or not index_record.index_file_path:
                raise ValueError(f"Index {index_id} not found or no file path")
            
            # Load index data
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
            
            logger.info(f"Deleted {count} embeddings for project {project_id}")
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