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
        project = Project.query.get(project_id)
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
        
        # Group by object to avoid duplicates
        object_groups = {}
        for result in all_results:
            key = f"{result['object_type']}_{result['object_id']}"
            if key not in object_groups:
                object_groups[key] = []
            object_groups[key].append(result)
        
        # Take best score for each object
        merged_results = []
        for object_key, results in object_groups.items():
            best_result = max(results, key=lambda x: x['score'])
            
            # Add information about multiple matches
            if len(results) > 1:
                best_result['match_count'] = len(results)
                best_result['search_types'] = list(set(r['search_type'] for r in results))
            
            merged_results.append(best_result)
        
        # Sort by score and limit
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        return merged_results[:top_k]
    
    def _log_search(self, query: str, project_id: int, result_count: int, search_time: float):
        """Log search query for analytics"""
        try:
            search_log = SearchLog(
                project_id=project_id,
                query=query,
                result_count=result_count,
                search_time_seconds=round(search_time, 3),
                timestamp=datetime.utcnow()
            )
            db.session.add(search_log)
            db.session.commit()
        except Exception as e:
            logger.warning(f"Failed to log search: {str(e)}")
            db.session.rollback()
    
    def get_search_suggestions(self, query: str, project_id: int, limit: int = 5) -> List[str]:
        """Get search suggestions based on dictionary and previous searches"""
        suggestions = []
        
        try:
            # Get dictionary terms that match
            dictionary_matches = DictionaryEntry.query.filter(
                DictionaryEntry.project_id == project_id,
                DictionaryEntry.term.ilike(f'%{query}%')
            ).limit(limit).all()
            
            suggestions.extend([entry.term for entry in dictionary_matches])
            
            # Get popular searches
            if len(suggestions) < limit:
                popular_searches = db.session.query(SearchLog.query)\
                    .filter(SearchLog.project_id == project_id)\
                    .filter(SearchLog.query.ilike(f'%{query}%'))\
                    .group_by(SearchLog.query)\
                    .order_by(db.func.count(SearchLog.query).desc())\
                    .limit(limit - len(suggestions))\
                    .all()
                
                suggestions.extend([log.query for log in popular_searches])
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            return suggestions
    
    def get_search_analytics(self, project_id: int, days: int = 30) -> Dict[str, Any]:
        """Get search analytics for a project"""
        try:
            from sqlalchemy import func
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Basic stats
            total_searches = SearchLog.query.filter(
                SearchLog.project_id == project_id,
                SearchLog.timestamp >= cutoff_date
            ).count()
            
            # Average results and search time
            avg_stats = db.session.query(
                func.avg(SearchLog.result_count).label('avg_results'),
                func.avg(SearchLog.search_time_seconds).label('avg_time')
            ).filter(
                SearchLog.project_id == project_id,
                SearchLog.timestamp >= cutoff_date
            ).first()
            
            # Top queries
            top_queries = db.session.query(
                SearchLog.query,
                func.count(SearchLog.query).label('count')
            ).filter(
                SearchLog.project_id == project_id,
                SearchLog.timestamp >= cutoff_date
            ).group_by(SearchLog.query)\
            .order_by(func.count(SearchLog.query).desc())\
            .limit(10).all()
            
            # Search volume by day
            daily_volume = db.session.query(
                func.date(SearchLog.timestamp).label('date'),
                func.count(SearchLog.id).label('count')
            ).filter(
                SearchLog.project_id == project_id,
                SearchLog.timestamp >= cutoff_date
            ).group_by(func.date(SearchLog.timestamp))\
            .order_by(func.date(SearchLog.timestamp)).all()
            
            return {
                'total_searches': total_searches,
                'avg_results': round(float(avg_stats.avg_results or 0), 2),
                'avg_search_time': round(float(avg_stats.avg_time or 0), 3),
                'top_queries': [{'query': q.query, 'count': q.count} for q in top_queries],
                'daily_volume': [{'date': str(d.date), 'count': d.count} for d in daily_volume],
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting search analytics: {str(e)}")
            return {
                'total_searches': 0,
                'avg_results': 0,
                'avg_search_time': 0,
                'top_queries': [],
                'daily_volume': [],
                'period_days': days
            }
    
    def clear_index_cache(self, index_id: Optional[int] = None):
        """Clear cached indexes"""
        if index_id:
            self.loaded_indexes.pop(index_id, None)
        else:
            self.loaded_indexes.clear()
        logger.info(f"Cleared index cache for index {index_id if index_id else 'all'}")
    
    def get_available_indexes(self, project_id: int) -> List[Dict[str, Any]]:
        """Get list of available indexes for a project"""
        try:
            indexes = Index.query.filter_by(project_id=project_id).all()
            
            result = []
            for index in indexes:
                index_info = {
                    'id': index.id,
                    'name': index.name,
                    'index_type': index.index_type,
                    'status': index.status,
                    'total_vectors': index.total_vectors,
                    'embedding_model': index.embedding_model,
                    'created_at': index.created_at.isoformat() if index.created_at else None,
                    'build_progress': index.build_progress
                }
                result.append(index_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting available indexes: {str(e)}")
            return []