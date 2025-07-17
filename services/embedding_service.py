# services/embedding_service.py
import numpy as np
import logging
import os
import pickle
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from models import db, EmbeddingIndex, Table, Column, DictionaryEntry
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Manages embedding creation and similarity search"""
    
    def __init__(self):
        self.config = Config()
        self.embedding_config = self.config.EMBEDDING_CONFIG
        self.models = {}  # Cache for loaded models
        self.indexes = {}  # Cache for loaded indexes
        self.job_status = {}  # Track async job status
        
        # Ensure index directory exists
        self.index_dir = Path(self.embedding_config['backends']['faiss']['index_path'])
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
    def create_index(self, scope: str, backend: str = 'faiss', 
                    model: str = None, name: str = None) -> str:
        """Create embedding index asynchronously"""
        try:
            # Validate inputs
            if scope not in ['table', 'column', 'dictionary']:
                raise ValueError(f"Invalid scope: {scope}")
            
            if backend not in ['faiss', 'tfidf', 'pgvector']:
                raise ValueError(f"Invalid backend: {backend}")
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Create index record
            index_name = name or f"{scope}_{backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_name = model or self.embedding_config['default_model']
            
            index_record = EmbeddingIndex(
                name=index_name,
                scope=scope,
                backend=backend,
                model_name=model_name,
                status='pending',
                created_at=datetime.utcnow()
            )
            
            db.session.add(index_record)
            db.session.commit()
            
            # Initialize job status
            self.job_status[job_id] = {
                'index_id': index_record.id,
                'status': 'pending',
                'progress': 0.0,
                'message': 'Job queued',
                'started_at': datetime.utcnow().isoformat()
            }
            
            # Start async job
            thread = threading.Thread(
                target=self._build_index_async,
                args=(job_id, index_record.id, scope, backend, model_name)
            )
            thread.start()
            
            logger.info(f"Started embedding job {job_id} for {scope} index")
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating embedding index: {str(e)}")
            raise
    
    def _build_index_async(self, job_id: str, index_id: int, scope: str, 
                          backend: str, model_name: str) -> None:
        """Build embedding index in background thread"""
        try:
            # Update status
            self._update_job_status(job_id, 'building', 0.0, 'Extracting text data')
            
            # Get index record
            index_record = EmbeddingIndex.query.get(index_id)
            index_record.status = 'building'
            db.session.commit()
            
            # Extract text data based on scope
            texts, _metadata = self._extract_text_data(scope)
            
            if not texts:
                self._update_job_status(job_id, 'error', 0.0, 'No data found for indexing')
                index_record.status = 'error'
                index_record.error_message = 'No data found for indexing'
                db.session.commit()
                return
            
            self._update_job_status(job_id, 'building', 25.0, f'Processing {len(texts)} items')
            
            # Build index based on backend
            if backend == 'faiss':
                index_path, dimensions = self._build_faiss_index(texts, _metadata, model_name, job_id)
            elif backend == 'tfidf':
                index_path, dimensions = self._build_tfidf_index(texts, _metadata, job_id)
            elif backend == 'pgvector':
                index_path, dimensions = self._build_pgvector_index(texts, _metadata, model_name, job_id)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            
            # Update index record
            index_record.status = 'ready'
            index_record.progress = 100.0
            index_record.index_path = index_path
            index_record.dimensions = dimensions
            index_record.item_count = len(texts)
            index_record.updated_at = datetime.utcnow()
            db.session.commit()
            
            self._update_job_status(job_id, 'completed', 100.0, 'Index built successfully')
            
            logger.info(f"Completed embedding job {job_id}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in embedding job {job_id}: {error_msg}")
            
            # Update status
            self._update_job_status(job_id, 'error', 0.0, error_msg)
            
            # Update index record
            try:
                index_record = EmbeddingIndex.query.get(index_id)
                index_record.status = 'error'
                index_record.error_message = error_msg
                db.session.commit()
            except:
                pass
    
    def _extract_text_data(self, scope: str) -> Tuple[List[str], List[Dict]]:
        """Extract text data for indexing based on scope"""
        texts = []
        _metadata = []
        
        if scope == 'table':
            tables = Table.query.all()
            for table in tables:
                # Combine table name, display name, and description
                text_parts = [table.name, table.display_name or '', table.description or '']
                text = ' '.join(filter(None, text_parts))
                
                texts.append(text)
                _metadata.append({
                    'type': 'table',
                    'id': table.id,
                    'name': table.name,
                    'display_name': table.display_name,
                    'source_id': table.source_id
                })
        
        elif scope == 'column':
            columns = Column.query.all()
            for column in columns:
                # Combine column name, display name, description, and sample values
                text_parts = [
                    column.name,
                    column.display_name or '',
                    column.description or '',
                    column.data_type or ''
                ]
                
                # Add sample values (limited)
                if column.sample_values:
                    sample_text = ' '.join(str(v) for v in column.sample_values[:5])
                    text_parts.append(sample_text)
                
                text = ' '.join(filter(None, text_parts))
                
                texts.append(text)
                _metadata.append({
                    'type': 'column',
                    'id': column.id,
                    'name': column.name,
                    'display_name': column.display_name,
                    'table_id': column.table_id,
                    'table_name': column.table.name,
                    'data_type': column.data_type
                })
        
        elif scope == 'dictionary':
            entries = DictionaryEntry.query.all()
            for entry in entries:
                # Combine term, definition, and synonyms
                text_parts = [
                    entry.term,
                    entry.definition,
                    ' '.join(entry.synonyms or []),
                    ' '.join(entry.abbreviations or [])
                ]
                text = ' '.join(filter(None, text_parts))
                
                texts.append(text)
                _metadata.append({
                    'type': 'dictionary',
                    'id': entry.id,
                    'term': entry.term,
                    'definition': entry.definition,
                    'category': entry.category,
                    'approved': entry.approved
                })
        
        return texts, _metadata
    
    def _build_faiss_index(self, texts: List[str], _metadata: List[Dict], 
                          model_name: str, job_id: str) -> Tuple[str, int]:
        """Build FAISS index"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not available")
        
        # Load model
        self._update_job_status(job_id, 'building', 30.0, 'Loading embedding model')
        model = self._get_model(model_name)
        
        # Generate embeddings
        self._update_job_status(job_id, 'building', 50.0, 'Generating embeddings')
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Build FAISS index
        self._update_job_status(job_id, 'building', 75.0, 'Building FAISS index')
        dimensions = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        index = faiss.IndexFlatIP(dimensions)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        
        # Save index and _metadata
        index_filename = f"faiss_index_{uuid.uuid4().hex}.idx"
        index_path = self.index_dir / index_filename
        
        faiss.write_index(index, str(index_path))
        
        # Save _metadata
        _metadata_path = index_path.with_suffix('._metadata')
        with open(_metadata_path, 'wb') as f:
            pickle.dump({
                '_metadata': _metadata,
                'model_name': model_name,
                'dimensions': dimensions,
                'text_samples': texts[:10]  # Keep some samples for debugging
            }, f)
        
        self._update_job_status(job_id, 'building', 95.0, 'Saving index')
        
        return str(index_path), dimensions
    
    def _build_tfidf_index(self, texts: List[str], _metadata: List[Dict], 
                          job_id: str) -> Tuple[str, int]:
        """Build TF-IDF index"""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
        
        # Build TF-IDF vectorizer
        self._update_job_status(job_id, 'building', 50.0, 'Building TF-IDF vectors')
        
        config = self.embedding_config['backends']['tfidf']
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            stop_words='english',
            lowercase=True
        )
        
        # Fit and transform texts
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Save index and _metadata
        index_filename = f"tfidf_index_{uuid.uuid4().hex}.pkl"
        index_path = self.index_dir / index_filename
        
        self._update_job_status(job_id, 'building', 90.0, 'Saving TF-IDF index')
        
        with open(index_path, 'wb') as f:
            pickle.dump({
                'vectorizer': vectorizer,
                'tfidf_matrix': tfidf_matrix,
                '_metadata': _metadata,
                'text_samples': texts[:10]
            }, f)
        
        return str(index_path), tfidf_matrix.shape[1]
    
    def _build_pgvector_index(self, texts: List[str], _metadata: List[Dict], 
                             model_name: str, job_id: str) -> Tuple[str, int]:
        """Build pgvector index (placeholder - requires PostgreSQL with pgvector)"""
        # This would require a PostgreSQL connection with pgvector extension
        # For now, we'll fall back to FAISS
        logger.warning("pgvector not implemented, falling back to FAISS")
        return self._build_faiss_index(texts, _metadata, model_name, job_id)
    
    def _get_model(self, model_name: str):
        """Get or load embedding model"""
        if model_name not in self.models:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError("sentence-transformers not available")
            
            logger.info(f"Loading embedding model: {model_name}")
            self.models[model_name] = SentenceTransformer(model_name)
        
        return self.models[model_name]
        
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an embedding job"""
        return self.job_status.get(job_id)

    def _update_job_status(self, job_id: str, status: str, progress: float, message: str = None):
        """Update job status"""
        self.job_status[job_id] = {
            'job_id': job_id,
            'status': status,
            'progress': progress,
            'message': message,
            'updated_at': datetime.utcnow().isoformat()
        }
    
    def search_similar(self, query: str, index_name: str, top_k: int = 10, 
                      threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar items using embedding index"""
        try:
            # Get index record
            index_record = EmbeddingIndex.query.filter_by(name=index_name).first()
            if not index_record or index_record.status != 'ready':
                raise ValueError(f"Index {index_name} not found or not ready")
            
            # Load index if not cached
            if index_name not in self.indexes:
                self._load_index(index_record)
            
            index_data = self.indexes[index_name]
            
            # Perform search based on backend
            if index_record.backend == 'faiss':
                results = self._search_faiss(query, index_data, top_k, threshold)
            elif index_record.backend == 'tfidf':
                results = self._search_tfidf(query, index_data, top_k, threshold)
            else:
                raise ValueError(f"Unsupported backend: {index_record.backend}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index {index_name}: {str(e)}")
            raise
    
    def _load_index(self, index_record: EmbeddingIndex) -> None:
        """Load index into memory"""
        try:
            index_path = Path(index_record.index_path)
            
            if index_record.backend == 'faiss':
                # Load FAISS index
                index = faiss.read_index(str(index_path))
                
                # Load _metadata
                _metadata_path = index_path.with_suffix('._metadata')
                with open(_metadata_path, 'rb') as f:
                    _metadata_info = pickle.load(f)
                
                self.indexes[index_record.name] = {
                    'type': 'faiss',
                    'index': index,
                    '_metadata': _metadata_info['_metadata'],
                    'model_name': _metadata_info['model_name']
                }
            
            elif index_record.backend == 'tfidf':
                # Load TF-IDF index
                with open(index_path, 'rb') as f:
                    index_data = pickle.load(f)
                
                self.indexes[index_record.name] = {
                    'type': 'tfidf',
                    'vectorizer': index_data['vectorizer'],
                    'tfidf_matrix': index_data['tfidf_matrix'],
                    '_metadata': index_data['_metadata']
                }
            
            logger.info(f"Loaded index: {index_record.name}")
            
        except Exception as e:
            logger.error(f"Error loading index {index_record.name}: {str(e)}")
            raise
    
    def _search_faiss(self, query: str, index_data: Dict, top_k: int, 
                     threshold: float) -> List[Dict[str, Any]]:
        """Search using FAISS index"""
        # Generate query embedding
        model = self._get_model(index_data['model_name'])
        query_embedding = model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding.astype(np.float32))
        
        # Search
        scores, indices = index_data['index'].search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= threshold and idx != -1:
                _metadata = index_data['_metadata'][idx]
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    '_metadata': _metadata
                })
        
        return results
    
    def _search_tfidf(self, query: str, index_data: Dict, top_k: int, 
                     threshold: float) -> List[Dict[str, Any]]:
        """Search using TF-IDF index"""
        # Transform query
        query_vector = index_data['vectorizer'].transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, index_data['tfidf_matrix']).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            if score >= threshold:
                _metadata = index_data['_metadata'][idx]
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    '_metadata': _metadata
                })
        
        return results
    
    def get_available_indexes(self) -> List[Dict[str, Any]]:
        """Get list of available indexes"""
        try:
            indexes = EmbeddingIndex.query.filter_by(status='ready').all()
            return [index.to_dict() for index in indexes]
        except Exception as e:
            logger.error(f"Error getting available indexes: {str(e)}")
            raise
    
    def delete_index(self, index_name: str) -> None:
        """Delete an embedding index"""
        try:
            index_record = EmbeddingIndex.query.filter_by(name=index_name).first()
            if not index_record:
                raise ValueError(f"Index {index_name} not found")
            
            # Remove from cache
            if index_name in self.indexes:
                del self.indexes[index_name]
            
            # Delete files
            if index_record.index_path and os.path.exists(index_record.index_path):
                os.remove(index_record.index_path)
                
                # Delete _metadata file for FAISS
                if index_record.backend == 'faiss':
                    _metadata_path = Path(index_record.index_path).with_suffix('._metadata')
                    if _metadata_path.exists():
                        os.remove(_metadata_path)
            
            # Delete database record
            db.session.delete(index_record)
            db.session.commit()
            
            logger.info(f"Deleted index: {index_name}")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting index {index_name}: {str(e)}")
            raise
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index"""
        try:
            index_record = EmbeddingIndex.query.filter_by(name=index_name).first()
            if not index_record:
                raise ValueError(f"Index {index_name} not found")
            
            stats = index_record.to_dict()
            
            # Add file size if available
            if index_record.index_path and os.path.exists(index_record.index_path):
                file_size = os.path.getsize(index_record.index_path)
                stats['file_size_mb'] = round(file_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            raise
    
    def cleanup_failed_jobs(self) -> None:
        """Clean up failed or stale jobs"""
        try:
            # Remove job status for completed jobs older than 1 hour
            current_time = datetime.utcnow()
            expired_jobs = []
            
            for job_id, status in self.job_status.items():
                if status['status'] in ['completed', 'error']:
                    started_at = datetime.fromisoformat(status['started_at'].replace('Z', '+00:00'))
                    if (current_time - started_at.replace(tzinfo=None)).total_seconds() > 3600:
                        expired_jobs.append(job_id)
            
            for job_id in expired_jobs:
                del self.job_status[job_id]
            
            # Update stale index records
            stale_indexes = EmbeddingIndex.query.filter(
                EmbeddingIndex.status == 'building',
                EmbeddingIndex.updated_at < datetime.utcnow() - datetime.timedelta(hours=1)
            ).all()
            
            for index in stale_indexes:
                index.status = 'error'
                index.error_message = 'Job timed out'
            
            db.session.commit()
            
            logger.info(f"Cleaned up {len(expired_jobs)} expired jobs and {len(stale_indexes)} stale indexes")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error cleaning up jobs: {str(e)}")

def get_all_tables_for_admin():
    """Get all tables from all data sources for admin panel"""
    try:
        from models import DataSource, Table
        
        all_tables = []
        sources = DataSource.query.all()
        
        for source in sources:
            for table in source.tables:
                all_tables.append({
                    'name': table.name,
                    'display_name': table.display_name or table.name,
                    'source_name': source.name,
                    'source_id': source.id,
                    'row_count': table.row_count or 0,
                    'column_count': len(table.columns) if table.columns else 0
                })
        
        return all_tables
    except Exception as e:
        logger.error(f"Error getting all tables: {str(e)}")
        return []