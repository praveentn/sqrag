# backend/services/embedding_service.py
"""
Embedding service for creating and managing vector embeddings
"""

import asyncio
import logging
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import pickle
import time

from backend.models import (
    Embedding, Index, ObjectType, IndexType, IndexStatus,
    Table, Column, DictionaryEntry
)
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating and managing embeddings"""
    
    def __init__(self):
        self.config = Config.EMBEDDING_CONFIG
        self.models = {}  # Cache for loaded models
        self.indexes = {}  # Cache for loaded indexes
    
    def _get_model(self, model_name: str) -> SentenceTransformer:
        """Get or load embedding model"""
        if model_name not in self.models:
            try:
                self.models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Fallback to default model
                default_model = self.config['default_model']
                if model_name != default_model:
                    self.models[model_name] = SentenceTransformer(default_model)
                    logger.warning(f"Using fallback model: {default_model}")
                else:
                    raise
        
        return self.models[model_name]
    
    async def create_embedding(
        self,
        text: str,
        model_name: str = None
    ) -> np.ndarray:
        """Create embedding for text"""
        
        if not model_name:
            model_name = self.config['default_model']
        
        try:
            model = self._get_model(model_name)
            
            # Create embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: model.encode([text])[0]
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise
    
    async def create_embeddings_batch(
        self,
        texts: List[str],
        model_name: str = None,
        batch_size: int = None
    ) -> List[np.ndarray]:
        """Create embeddings for multiple texts"""
        
        if not model_name:
            model_name = self.config['default_model']
        
        if not batch_size:
            batch_size = self.config['batch_size']
        
        try:
            model = self._get_model(model_name)
            
            # Process in batches
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Create embeddings in thread pool
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None,
                    lambda: model.encode(batch_texts)
                )
                
                embeddings.extend(batch_embeddings)
                
                # Log progress
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create batch embeddings: {e}")
            raise
    
    async def create_table_embedding(
        self,
        table: Table,
        model_name: str = None
    ) -> Dict[str, Any]:
        """Create embedding for table metadata"""
        
        # Compose text from table metadata
        text_parts = [table.name]
        
        if table.display_name and table.display_name != table.name:
            text_parts.append(table.display_name)
        
        if table.description:
            text_parts.append(table.description)
        
        # Add column names
        if table.columns:
            column_names = [col.name for col in table.columns]
            text_parts.append(" ".join(column_names))
        
        text_content = " | ".join(text_parts)
        
        # Create embedding
        embedding = await self.create_embedding(text_content, model_name)
        
        return {
            "text_content": text_content,
            "embedding": embedding,
            "metadata": {
                "table_id": table.id,
                "table_name": table.name,
                "schema_name": table.schema_name,
                "row_count": table.row_count,
                "column_count": table.column_count
            }
        }
    
    async def create_column_embedding(
        self,
        column: Column,
        model_name: str = None
    ) -> Dict[str, Any]:
        """Create embedding for column metadata"""
        
        # Compose text from column metadata
        text_parts = [column.name]
        
        if column.display_name and column.display_name != column.name:
            text_parts.append(column.display_name)
        
        if column.description:
            text_parts.append(column.description)
        
        # Add data type
        text_parts.append(column.data_type)
        
        # Add sample values if available
        if column.sample_values:
            sample_text = " ".join(str(v) for v in column.sample_values[:5])
            text_parts.append(sample_text)
        
        text_content = " | ".join(text_parts)
        
        # Create embedding
        embedding = await self.create_embedding(text_content, model_name)
        
        return {
            "text_content": text_content,
            "embedding": embedding,
            "metadata": {
                "column_id": column.id,
                "column_name": column.name,
                "table_id": column.table_id,
                "data_type": column.data_type,
                "is_primary_key": column.is_primary_key,
                "pii_flag": column.pii_flag
            }
        }
    
    async def create_dictionary_embedding(
        self,
        entry: DictionaryEntry,
        model_name: str = None
    ) -> Dict[str, Any]:
        """Create embedding for dictionary entry"""
        
        # Compose text from dictionary entry
        text_parts = [entry.term]
        
        if entry.definition:
            text_parts.append(entry.definition)
        
        # Add synonyms
        if entry.synonyms:
            text_parts.extend(entry.synonyms)
        
        # Add abbreviations
        if entry.abbreviations:
            text_parts.extend(entry.abbreviations)
        
        # Add context
        if entry.context:
            text_parts.append(entry.context)
        
        text_content = " | ".join(text_parts)
        
        # Create embedding
        embedding = await self.create_embedding(text_content, model_name)
        
        return {
            "text_content": text_content,
            "embedding": embedding,
            "metadata": {
                "entry_id": entry.id,
                "term": entry.term,
                "category": entry.category.value,
                "domain_tags": entry.domain_tags,
                "confidence_score": entry.confidence_score
            }
        }
    
    async def build_faiss_index(
        self,
        embeddings: List[Embedding],
        index_config: Dict[str, Any] = None
    ) -> Tuple[faiss.Index, Dict[str, Any]]:
        """Build FAISS index from embeddings"""
        
        if not embeddings:
            raise ValueError("No embeddings provided for index building")
        
        # Get dimensions from first embedding
        first_embedding = embeddings[0].get_vector()
        dimensions = len(first_embedding)
        
        # Default configuration
        config = index_config or {}
        metric = config.get('metric', 'cosine')
        index_type = config.get('index_type', 'IndexFlatIP')
        
        try:
            # Create FAISS index
            if metric == 'cosine':
                # For cosine similarity, use inner product with normalized vectors
                index = faiss.IndexFlatIP(dimensions)
            else:
                # For L2 distance
                index = faiss.IndexFlatL2(dimensions)
            
            # Prepare vectors
            vectors = []
            metadata = []
            
            for emb in embeddings:
                vector = emb.get_vector()
                
                # Normalize for cosine similarity
                if metric == 'cosine':
                    vector = vector / np.linalg.norm(vector)
                
                vectors.append(vector)
                metadata.append({
                    "embedding_id": emb.id,
                    "object_type": emb.object_type.value,
                    "object_id": emb.object_id,
                    "text_content": emb.text_content
                })
            
            # Convert to numpy array
            vectors_array = np.array(vectors).astype('float32')
            
            # Add vectors to index
            index.add(vectors_array)
            
            logger.info(f"Built FAISS index with {len(vectors)} vectors, {dimensions} dimensions")
            
            return index, {
                "total_vectors": len(vectors),
                "dimensions": dimensions,
                "metric": metric,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise
    
    async def search_faiss_index(
        self,
        index: faiss.Index,
        query_vector: np.ndarray,
        metadata: List[Dict[str, Any]],
        k: int = 10,
        metric: str = 'cosine'
    ) -> List[Dict[str, Any]]:
        """Search FAISS index"""
        
        try:
            # Normalize query vector for cosine similarity
            if metric == 'cosine':
                query_vector = query_vector / np.linalg.norm(query_vector)
            
            # Reshape for FAISS
            query_vector = query_vector.reshape(1, -1).astype('float32')
            
            # Search
            similarities, indices = index.search(query_vector, k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(metadata):  # Valid index
                    result = metadata[idx].copy()
                    result['similarity_score'] = float(similarities[0][i])
                    result['rank'] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def save_faiss_index(
        self,
        index: faiss.Index,
        metadata: Dict[str, Any],
        index_path: str
    ):
        """Save FAISS index to disk"""
        
        try:
            # Create directory if it doesn't exist
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(index, index_path)
            
            # Save metadata
            metadata_path = index_path.replace('.index', '.metadata')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved FAISS index to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    async def load_faiss_index(
        self,
        index_path: str
    ) -> Tuple[faiss.Index, Dict[str, Any]]:
        """Load FAISS index from disk"""
        
        try:
            # Load FAISS index
            index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = index_path.replace('.index', '.metadata')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            logger.info(f"Loaded FAISS index from {index_path}")
            return index, metadata
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    async def create_tfidf_index(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]],
        config: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Create TF-IDF index"""
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Default configuration
        tfidf_config = config or {}
        max_features = tfidf_config.get('max_features', 10000)
        ngram_range = tuple(tfidf_config.get('ngram_range', [1, 2]))
        stop_words = tfidf_config.get('stop_words', 'english')
        
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=stop_words,
                lowercase=True
            )
            
            # Fit and transform texts
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Create index object
            index_data = {
                'vectorizer': vectorizer,
                'tfidf_matrix': tfidf_matrix,
                'texts': texts,
                'metadata': metadata
            }
            
            logger.info(f"Built TF-IDF index with {len(texts)} documents, {tfidf_matrix.shape[1]} features")
            
            return index_data, {
                "total_documents": len(texts),
                "features": tfidf_matrix.shape[1],
                "max_features": max_features,
                "ngram_range": ngram_range
            }
            
        except Exception as e:
            logger.error(f"Failed to create TF-IDF index: {e}")
            raise
    
    async def search_tfidf_index(
        self,
        index_data: Dict[str, Any],
        query: str,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search TF-IDF index"""
        
        try:
            vectorizer = index_data['vectorizer']
            tfidf_matrix = index_data['tfidf_matrix']
            metadata = index_data['metadata']
            
            # Transform query
            query_vector = vectorizer.transform([query])
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Format results
            results = []
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:  # Only include non-zero similarities
                    result = metadata[idx].copy()
                    result['similarity_score'] = float(similarities[idx])
                    result['rank'] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return []
    
    def calculate_similarity(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """Calculate similarity between two vectors"""
        
        try:
            if metric == 'cosine':
                from sklearn.metrics.pairwise import cosine_similarity
                return cosine_similarity(
                    vector1.reshape(1, -1),
                    vector2.reshape(1, -1)
                )[0][0]
            elif metric == 'euclidean':
                from scipy.spatial.distance import euclidean
                return 1.0 / (1.0 + euclidean(vector1, vector2))
            elif metric == 'dot':
                return np.dot(vector1, vector2)
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
                
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def hybrid_search(
        self,
        query: str,
        faiss_index: faiss.Index = None,
        faiss_metadata: List[Dict[str, Any]] = None,
        tfidf_index: Dict[str, Any] = None,
        model_name: str = None,
        k: int = 10,
        alpha: float = 0.7  # Weight for semantic search vs keyword search
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search"""
        
        results = {}
        
        try:
            # Semantic search with FAISS
            if faiss_index is not None and faiss_metadata is not None:
                query_embedding = await self.create_embedding(query, model_name)
                semantic_results = await self.search_faiss_index(
                    faiss_index, query_embedding, faiss_metadata, k * 2
                )
                
                for result in semantic_results:
                    key = f"{result['object_type']}_{result['object_id']}"
                    if key not in results:
                        results[key] = result.copy()
                        results[key]['semantic_score'] = result['similarity_score']
                        results[key]['keyword_score'] = 0.0
                    else:
                        results[key]['semantic_score'] = result['similarity_score']
            
            # Keyword search with TF-IDF
            if tfidf_index is not None:
                keyword_results = await self.search_tfidf_index(
                    tfidf_index, query, k * 2
                )
                
                for result in keyword_results:
                    key = f"{result['object_type']}_{result['object_id']}"
                    if key not in results:
                        results[key] = result.copy()
                        results[key]['semantic_score'] = 0.0
                        results[key]['keyword_score'] = result['similarity_score']
                    else:
                        results[key]['keyword_score'] = result['similarity_score']
            
            # Combine scores
            final_results = []
            for key, result in results.items():
                semantic_score = result.get('semantic_score', 0.0)
                keyword_score = result.get('keyword_score', 0.0)
                
                # Hybrid score
                hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score
                
                result['hybrid_score'] = hybrid_score
                result['similarity_score'] = hybrid_score  # Use as main score
                final_results.append(result)
            
            # Sort by hybrid score and limit results
            final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # Re-rank
            for i, result in enumerate(final_results[:k]):
                result['rank'] = i + 1
            
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

# Global service instance
embedding_service = EmbeddingService()