# services/search_service.py
import numpy as np
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import Config
from models import db, Project, Table, Column, DictionaryEntry, Embedding, Index, SearchLog
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class SearchService:
    """Service for performing searches across multiple index types"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.loaded_indexes = {}  # Cache for loaded indexes
        self.sentence_model = None
        self._init_sentence_transformer()
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer for query encoding"""
        try:
            default_model = Config.EMBEDDING_CONFIG['default_model']
            self.sentence_model = SentenceTransformer(default_model)
            logger.info(f"Loaded sentence transformer: {default_model}")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {str(e)}")
    
    def search(self, query: str, project_id: int, search_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform search across project indexes
        
        Args:
            query: Search query string
            project_id: Project ID for search scope
            search_params: Optional search parameters
                - index_ids: List of specific index IDs to search
                - top_k: Number of results to return (default: 10)
                - min_score: Minimum similarity score threshold
                - search_type: 'semantic', 'keyword', or 'hybrid'
        """
        start_time = time.time()
        
        if not project_id:
            raise ValueError("Project ID required for search")
        
        # Validate project exists
        project = db.session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Default search parameters
        search_params = search_params or {}
        top_k = search_params.get('top_k', 10)
        min_score = search_params.get('min_score', 0.1)
        search_type = search_params.get('search_type', 'hybrid')
        index_ids = search_params.get('index_ids')
        
        try:
            # Get available indexes for project
            if index_ids:
                indexes = Index.query.filter(
                    Index.id.in_(index_ids),
                    Index.project_id == project_id,
                    Index.status == 'ready'
                ).all()
            else:
                indexes = Index.query.filter_by(
                    project_id=project_id,
                    status='ready'
                ).all()
            
            if not indexes:
                return {
                    'results': [],
                    'total_results': 0,
                    'search_time': time.time() - start_time,
                    'message': 'No ready indexes found for this project'
                }
            
            # Perform search across all available indexes
            all_results = []
            
            for index in indexes:
                try:
                    index_results = self._search_single_index(
                        query, index, top_k, min_score, search_type
                    )
                    all_results.extend(index_results)
                except Exception as e:
                    logger.warning(f"Error searching index {index.id}: {str(e)}")
                    continue
            
            # Merge and rank results
            final_results = self._merge_and_rank_results(all_results, top_k)
            
            # Log search
            self._log_search(query, project_id, len(final_results), time.time() - start_time)
            
            return {
                'results': final_results,
                'total_results': len(final_results),
                'search_time': round(time.time() - start_time, 3),
                'indexes_searched': len(indexes),
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Search failed for project {project_id}: {str(e)}")
            raise Exception(f"Search failed: {str(e)}")
    
    def _search_single_index(self, query: str, index: Index, top_k: int, 
                           min_score: float, search_type: str) -> List[Dict]:
        """Search a single index"""
        try:
            # Load index if not cached
            if index.id not in self.loaded_indexes:
                index_data = self.embedding_service.load_index(index.id)
                self.loaded_indexes[index.id] = index_data
            else:
                index_data = self.loaded_indexes[index.id]
            
            # Perform search based on index type
            if index_data['type'] == 'faiss':
                return self._search_faiss_index(query, index_data, top_k, min_score)
            elif index_data['type'] == 'tfidf':
                return self._search_tfidf_index(query, index_data, top_k, min_score)
            elif index_data['type'] == 'bm25':
                return self._search_bm25_index(query, index_data, top_k, min_score)
            else:
                logger.warning(f"Unsupported index type: {index_data['type']}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching index {index.id}: {str(e)}")
            return []
    
    def _search_faiss_index(self, query: str, index_data: Dict, 
                          top_k: int, min_score: float) -> List[Dict]:
        """Search FAISS vector index"""
        if not self.sentence_model:
            raise ValueError("Sentence transformer not available for vector search")
        
        # Encode query
        query_vector = self.sentence_model.encode([query]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = index_data['index'].search(query_vector, top_k)
        
        results = []
        metadata = index_data['metadata']
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or score < min_score:  # FAISS returns -1 for invalid results
                continue
                
            try:
                result = {
                    'id': metadata['embedding_ids'][idx],
                    'object_type': metadata['object_types'][idx],
                    'object_id': metadata['object_ids'][idx],
                    'text': metadata['object_texts'][idx],
                    'score': float(score),
                    'rank': i + 1,
                    'search_type': 'semantic'
                }
                results.append(result)
            except IndexError:
                logger.warning(f"Index out of bounds for FAISS result {idx}")
                continue
        
        return results
    
    def _search_tfidf_index(self, query: str, index_data: Dict, 
                          top_k: int, min_score: float) -> List[Dict]:
        """Search TF-IDF index"""
        # Transform query
        query_vector = index_data['vectorizer'].transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, index_data['matrix']).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        metadata = index_data['metadata']
        
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            if score < min_score:
                continue
                
            try:
                result = {
                    'id': metadata['embedding_ids'][idx],
                    'object_type': metadata['object_types'][idx],
                    'object_id': metadata['object_ids'][idx],
                    'text': metadata['object_texts'][idx],
                    'score': float(score),
                    'rank': i + 1,
                    'search_type': 'keyword'
                }
                results.append(result)
            except IndexError:
                logger.warning(f"Index out of bounds for TF-IDF result {idx}")
                continue
        
        return results
    
    def _search_bm25_index(self, query: str, index_data: Dict, 
                         top_k: int, min_score: float) -> List[Dict]:
        """Search BM25 index (similar to TF-IDF but with different scoring)"""
        return self._search_tfidf_index(query, index_data, top_k, min_score)
    
    def _merge_and_rank_results(self, all_results: List[Dict], top_k: int) -> List[Dict]:
        """Merge results from multiple indexes and re-rank"""
        if not all_results:
            return []
        
        # Group by object_id and object_type to deduplicate
        unique_results = {}
        for result in all_results:
            key = f"{result['object_type']}_{result['object_id']}"
            if key not in unique_results or result['score'] > unique_results[key]['score']:
                unique_results[key] = result
        
        # Convert back to list and sort by score
        merged_results = list(unique_results.values())
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(merged_results):
            result['rank'] = i + 1
        
        return merged_results[:top_k]
    
    def _log_search(self, query: str, project_id: int, results_count: int, search_time: float):
        """Log search for analytics"""
        try:
            search_log = SearchLog(
                project_id=project_id,
                query=query,
                results_count=results_count,
                search_time_ms=round(search_time * 1000, 2),
                created_at=datetime.utcnow()
            )
            db.session.add(search_log)
            db.session.commit()
        except Exception as e:
            logger.warning(f"Error logging search: {str(e)}")
    
    def get_similar_objects(self, object_type: str, object_id: int, 
                          project_id: int, top_k: int = 5) -> List[Dict]:
        """Find objects similar to a given object"""
        try:
            # Get the embedding for the target object
            target_embedding = Embedding.query.filter_by(
                project_id=project_id,
                object_type=object_type,
                object_id=object_id
            ).first()
            
            if not target_embedding:
                return []
            
            # Use the object text as query
            return self.search(
                target_embedding.object_text, 
                project_id, 
                {'top_k': top_k + 1}  # +1 to exclude self
            )['results'][1:]  # Exclude the first result (self)
            
        except Exception as e:
            logger.error(f"Error finding similar objects: {str(e)}")
            return []
    
    def search_by_category(self, category: str, project_id: int, 
                         top_k: int = 10) -> List[Dict]:
        """Search for objects by category"""
        try:
            # Get all objects of the specified category
            if category == 'tables':
                tables = db.session.query(Table).join(
                    Table.source
                ).filter(
                    Table.source.has(project_id=project_id)
                ).all()
                
                results = []
                for table in tables:
                    results.append({
                        'object_type': 'table',
                        'object_id': table.id,
                        'text': table.name,
                        'score': 1.0,
                        'metadata': {
                            'name': table.name,
                            'row_count': table.row_count,
                            'column_count': table.column_count
                        }
                    })
                
            elif category == 'columns':
                columns = db.session.query(Column).join(
                    Column.table
                ).join(
                    Table.source
                ).filter(
                    Table.source.has(project_id=project_id)
                ).all()
                
                results = []
                for column in columns:
                    results.append({
                        'object_type': 'column',
                        'object_id': column.id,
                        'text': column.name,
                        'score': 1.0,
                        'metadata': {
                            'name': column.name,
                            'table_name': column.table.name,
                            'data_type': column.data_type,
                            'business_category': column.business_category
                        }
                    })
                
            elif category == 'dictionary':
                entries = DictionaryEntry.query.filter_by(
                    project_id=project_id
                ).filter(
                    DictionaryEntry.status != 'archived'
                ).all()
                
                results = []
                for entry in entries:
                    results.append({
                        'object_type': 'dictionary_entry',
                        'object_id': entry.id,
                        'text': entry.term,
                        'score': 1.0,
                        'metadata': {
                            'term': entry.term,
                            'definition': entry.definition,
                            'category': entry.category,
                            'domain': entry.domain
                        }
                    })
            else:
                results = []
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching by category: {str(e)}")
            return []
    
    def get_search_suggestions(self, partial_query: str, project_id: int, 
                             max_suggestions: int = 5) -> List[str]:
        """Get search suggestions based on partial query"""
        try:
            suggestions = set()
            partial_lower = partial_query.lower()
            
            # Get table names
            tables = db.session.query(Table).join(
                Table.source
            ).filter(
                Table.source.has(project_id=project_id)
            ).all()
            
            for table in tables:
                if partial_lower in table.name.lower():
                    suggestions.add(table.name)
            
            # Get column names
            columns = db.session.query(Column).join(
                Column.table
            ).join(
                Table.source
            ).filter(
                Table.source.has(project_id=project_id)
            ).all()
            
            for column in columns:
                if partial_lower in column.name.lower():
                    suggestions.add(column.name)
            
            # Get dictionary terms
            entries = DictionaryEntry.query.filter_by(project_id=project_id).all()
            for entry in entries:
                if partial_lower in entry.term.lower():
                    suggestions.add(entry.term)
            
            return sorted(list(suggestions))[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            return []
    
    def get_search_stats(self, project_id: int, days: int = 30) -> Dict[str, Any]:
        """Get search statistics for a project"""
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get search logs
            logs = SearchLog.query.filter(
                SearchLog.project_id == project_id,
                SearchLog.created_at >= cutoff_date
            ).all()
            
            if not logs:
                return {
                    'total_searches': 0,
                    'avg_search_time': 0,
                    'avg_results_count': 0,
                    'popular_queries': [],
                    'search_trends': []
                }
            
            # Calculate statistics
            total_searches = len(logs)
            avg_search_time = sum(log.search_time_ms for log in logs) / total_searches
            avg_results_count = sum(log.results_count for log in logs) / total_searches
            
            # Popular queries
            query_counts = {}
            for log in logs:
                query_counts[log.query] = query_counts.get(log.query, 0) + 1
            
            popular_queries = sorted(query_counts.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'total_searches': total_searches,
                'avg_search_time': round(avg_search_time, 2),
                'avg_results_count': round(avg_results_count, 1),
                'popular_queries': [{'query': q, 'count': c} for q, c in popular_queries],
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting search stats: {str(e)}")
            return {
                'total_searches': 0,
                'avg_search_time': 0,
                'avg_results_count': 0,
                'popular_queries': [],
                'error': str(e)
            }