# services/search_service.py
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import faiss

from config import Config
from models import db, Index, SearchLog, Table, Column, DictionaryEntry, Project
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class SearchService:
    """Service for performing searches across different index types"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.sentence_model = None
        self._init_sentence_transformer()
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer for query embedding"""
        try:
            model_name = Config.EMBEDDING_CONFIG['default_model']
            self.sentence_model = SentenceTransformer(model_name)
            logger.info("Search service initialized with sentence transformer")
        except Exception as e:
            logger.warning(f"Failed to initialize sentence transformer: {str(e)}")
    
    def search(self, query: str, index_id: Optional[int] = None, 
               search_type: str = 'hybrid', top_k: int = 10, 
               project_id: Optional[int] = None) -> Dict[str, Any]:
        """Perform search across indexes"""
        start_time = time.time()
        
        try:
            if index_id:
                # Search specific index
                results = self._search_single_index(query, index_id, search_type, top_k)
            else:
                # Search across all relevant indexes
                results = self._search_all_indexes(query, project_id, search_type, top_k)
            
            search_time_ms = round((time.time() - start_time) * 1000, 2)
            
            # Log search
            if project_id:
                self._log_search(query, index_id, search_type, results, search_time_ms, project_id)
            
            return {
                'query': query,
                'search_type': search_type,
                'results': results,
                'total_results': len(results),
                'search_time_ms': search_time_ms,
                'index_id': index_id
            }
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise Exception(f"Search failed: {str(e)}")
    
    def _search_single_index(self, query: str, index_id: int, 
                           search_type: str, top_k: int) -> List[Dict[str, Any]]:
        """Search within a single index"""
        index_record = Index.query.get(index_id)
        if not index_record or index_record.status != 'ready':
            raise ValueError(f"Index {index_id} not ready")
        
        # Load index
        index_data = self.embedding_service.load_index(index_id)
        
        # Perform search based on index type and search type
        if index_data['type'] == 'faiss':
            return self._search_faiss_index(query, index_data, search_type, top_k)
        elif index_data['type'] in ['tfidf', 'bm25']:
            return self._search_text_index(query, index_data, search_type, top_k)
        else:
            raise ValueError(f"Unsupported index type: {index_data['type']}")
    
    def _search_all_indexes(self, query: str, project_id: int, 
                          search_type: str, top_k: int) -> List[Dict[str, Any]]:
        """Search across all indexes for a project"""
        if not project_id:
            raise ValueError("Project ID required for multi-index search")
        
        # Get all ready indexes for the project
        indexes = Index.query.filter_by(
            project_id=project_id,
            status='ready'
        ).all()
        
        if not indexes:
            return []
        
        all_results = []
        
        # Search each index
        for index_record in indexes:
            try:
                index_results = self._search_single_index(
                    query, index_record.id, search_type, top_k
                )
                
                # Add index metadata to results
                for result in index_results:
                    result['index_id'] = index_record.id
                    result['index_name'] = index_record.name
                    result['index_type'] = index_record.index_type
                
                all_results.extend(index_results)
                
            except Exception as e:
                logger.warning(f"Error searching index {index_record.id}: {str(e)}")
                continue
        
        # Merge and rank results
        merged_results = self._merge_and_rank_results(all_results, query, top_k)
        
        return merged_results
    
    def _search_faiss_index(self, query: str, index_data: Dict, 
                           search_type: str, top_k: int) -> List[Dict[str, Any]]:
        """Search FAISS vector index"""
        if not self.sentence_model:
            raise ValueError("Sentence transformer not available for FAISS search")
        
        # Embed query
        query_vector = self.sentence_model.encode([query])[0].astype('float32')
        
        # Normalize for cosine similarity if using IndexFlatIP
        if index_data['metadata']['index_type'] == 'IndexFlatIP':
            faiss.normalize_L2(query_vector.reshape(1, -1))
        
        # Search
        faiss_index = index_data['index']
        scores, indices = faiss_index.search(query_vector.reshape(1, -1), top_k)
        
        # Build results
        results = []
        metadata = index_data['metadata']
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for not found
                continue
            
            result = {
                'rank': i + 1,
                'score': float(score),
                'similarity': float(score),  # For FAISS, score is similarity
                'object_type': metadata['object_types'][idx],
                'object_id': metadata['object_ids'][idx],
                'object_text': metadata['object_texts'][idx],
                'embedding_id': metadata['embedding_ids'][idx],
                'search_method': 'semantic'
            }
            
            # Add object details
            result.update(self._get_object_details(
                result['object_type'], 
                result['object_id']
            ))
            
            results.append(result)
        
        return results
    
    def _search_text_index(self, query: str, index_data: Dict, 
                          search_type: str, top_k: int) -> List[Dict[str, Any]]:
        """Search TF-IDF or BM25 text index"""
        vectorizer = index_data['vectorizer']
        matrix = index_data['matrix']
        metadata = index_data['metadata']
        
        # Transform query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            
            if score <= 0:  # Skip zero similarities
                continue
            
            result = {
                'rank': i + 1,
                'score': float(score),
                'similarity': float(score),
                'object_type': metadata['object_types'][idx],
                'object_id': metadata['object_ids'][idx],
                'object_text': metadata['object_texts'][idx],
                'embedding_id': metadata['embedding_ids'][idx],
                'search_method': 'lexical'
            }
            
            # Add object details
            result.update(self._get_object_details(
                result['object_type'], 
                result['object_id']
            ))
            
            results.append(result)
        
        return results
    
    def _get_object_details(self, object_type: str, object_id: int) -> Dict[str, Any]:
        """Get additional details about the search result object"""
        details = {}
        
        try:
            if object_type == 'table':
                table = Table.query.get(object_id)
                if table:
                    details.update({
                        'table_name': table.name,
                        'table_display_name': table.display_name,
                        'table_description': table.description,
                        'source_name': table.source.name if table.source else None,
                        'row_count': table.row_count,
                        'column_count': table.column_count
                    })
            
            elif object_type == 'column':
                column = Column.query.get(object_id)
                if column:
                    details.update({
                        'column_name': column.name,
                        'column_display_name': column.display_name,
                        'column_description': column.description,
                        'table_name': column.table.name if column.table else None,
                        'data_type': column.data_type,
                        'business_category': column.business_category,
                        'is_primary_key': column.is_primary_key,
                        'sample_values': column.sample_values[:3] if column.sample_values else []
                    })
            
            elif object_type == 'dictionary_entry':
                entry = DictionaryEntry.query.get(object_id)
                if entry:
                    details.update({
                        'term': entry.term,
                        'definition': entry.definition,
                        'category': entry.category,
                        'domain': entry.domain,
                        'synonyms': entry.synonyms or [],
                        'status': entry.status
                    })
        
        except Exception as e:
            logger.warning(f"Error getting object details: {str(e)}")
        
        return details
    
    def _merge_and_rank_results(self, all_results: List[Dict], 
                               query: str, top_k: int) -> List[Dict[str, Any]]:
        """Merge results from multiple indexes and re-rank"""
        if not all_results:
            return []
        
        # Group by object
        object_groups = {}
        for result in all_results:
            key = f"{result['object_type']}_{result['object_id']}"
            if key not in object_groups:
                object_groups[key] = []
            object_groups[key].append(result)
        
        # Merge scores for each object
        merged_results = []
        for key, group in object_groups.items():
            # Take the best result as base
            best_result = max(group, key=lambda x: x['score'])
            
            # Calculate combined score
            scores = [r['score'] for r in group]
            methods = list(set(r['search_method'] for r in group))
            
            # Weighted combination
            if len(methods) > 1:
                # Hybrid score: weighted average with bonus for multi-method match
                combined_score = np.mean(scores) * 1.2  # 20% bonus for multi-method
                search_method = 'hybrid'
            else:
                combined_score = max(scores)
                search_method = methods[0]
            
            # Add fuzzy matching bonus
            fuzzy_bonus = self._calculate_fuzzy_bonus(query, best_result)
            combined_score = min(combined_score + fuzzy_bonus, 1.0)
            
            merged_result = best_result.copy()
            merged_result.update({
                'score': round(combined_score, 3),
                'search_method': search_method,
                'index_count': len(group),
                'fuzzy_bonus': fuzzy_bonus
            })
            
            merged_results.append(merged_result)
        
        # Sort by combined score
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Re-rank and limit
        for i, result in enumerate(merged_results[:top_k]):
            result['rank'] = i + 1
        
        return merged_results[:top_k]
    
    def _calculate_fuzzy_bonus(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate fuzzy matching bonus"""
        query_lower = query.lower()
        bonus = 0.0
        
        # Check against different text fields
        text_fields = [
            result.get('object_text', ''),
            result.get('table_name', ''),
            result.get('column_name', ''),
            result.get('term', ''),
            result.get('table_display_name', ''),
            result.get('column_display_name', '')
        ]
        
        max_fuzzy_score = 0
        for text in text_fields:
            if text:
                fuzzy_score = fuzz.partial_ratio(query_lower, text.lower())
                max_fuzzy_score = max(max_fuzzy_score, fuzzy_score)
        
        # Convert to bonus (0.0 to 0.3)
        if max_fuzzy_score > 90:
            bonus = 0.3
        elif max_fuzzy_score > 80:
            bonus = 0.2
        elif max_fuzzy_score > 70:
            bonus = 0.1
        
        return round(bonus, 3)
    
    def search_by_type(self, query: str, object_type: str, 
                      project_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for specific object type (tables, columns, dictionary)"""
        try:
            if object_type == 'tables':
                return self._search_tables(query, project_id, top_k)
            elif object_type == 'columns':
                return self._search_columns(query, project_id, top_k)
            elif object_type == 'dictionary':
                return self._search_dictionary(query, project_id, top_k)
            else:
                raise ValueError(f"Unsupported object type: {object_type}")
        
        except Exception as e:
            logger.error(f"Type-specific search error: {str(e)}")
            raise
    
    def _search_tables(self, query: str, project_id: int, top_k: int) -> List[Dict[str, Any]]:
        """Search tables using fuzzy matching"""
        tables = db.session.query(Table).join(
            Table.source
        ).filter(
            Table.source.has(project_id=project_id)
        ).all()
        
        results = []
        query_lower = query.lower()
        
        for table in tables:
            # Calculate relevance score
            scores = []
            
            # Name matching
            name_score = fuzz.ratio(query_lower, table.name.lower()) / 100
            scores.append(name_score * 1.0)  # Full weight for name
            
            # Display name matching
            if table.display_name:
                display_score = fuzz.ratio(query_lower, table.display_name.lower()) / 100
                scores.append(display_score * 0.8)
            
            # Description matching
            if table.description:
                desc_score = fuzz.partial_ratio(query_lower, table.description.lower()) / 100
                scores.append(desc_score * 0.6)
            
            # Combined score
            combined_score = max(scores) if scores else 0
            
            if combined_score > 0.3:  # Threshold
                result = {
                    'object_type': 'table',
                    'object_id': table.id,
                    'score': round(combined_score, 3),
                    'table_name': table.name,
                    'table_display_name': table.display_name,
                    'table_description': table.description,
                    'source_name': table.source.name if table.source else None,
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'search_method': 'fuzzy'
                }
                results.append(result)
        
        # Sort and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _search_columns(self, query: str, project_id: int, top_k: int) -> List[Dict[str, Any]]:
        """Search columns using fuzzy matching"""
        columns = db.session.query(Column).join(
            Column.table
        ).join(
            Table.source
        ).filter(
            Table.source.has(project_id=project_id)
        ).all()
        
        results = []
        query_lower = query.lower()
        
        for column in columns:
            scores = []
            
            # Name matching
            name_score = fuzz.ratio(query_lower, column.name.lower()) / 100
            scores.append(name_score * 1.0)
            
            # Display name matching
            if column.display_name:
                display_score = fuzz.ratio(query_lower, column.display_name.lower()) / 100
                scores.append(display_score * 0.8)
            
            # Description matching
            if column.description:
                desc_score = fuzz.partial_ratio(query_lower, column.description.lower()) / 100
                scores.append(desc_score * 0.6)
            
            # Business category matching
            if column.business_category:
                cat_score = fuzz.ratio(query_lower, column.business_category.lower()) / 100
                scores.append(cat_score * 0.4)
            
            combined_score = max(scores) if scores else 0
            
            if combined_score > 0.3:
                result = {
                    'object_type': 'column',
                    'object_id': column.id,
                    'score': round(combined_score, 3),
                    'column_name': column.name,
                    'column_display_name': column.display_name,
                    'column_description': column.description,
                    'table_name': column.table.name,
                    'data_type': column.data_type,
                    'business_category': column.business_category,
                    'search_method': 'fuzzy'
                }
                results.append(result)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _search_dictionary(self, query: str, project_id: int, top_k: int) -> List[Dict[str, Any]]:
        """Search dictionary entries using fuzzy matching"""
        entries = DictionaryEntry.query.filter_by(
            project_id=project_id
        ).filter(
            DictionaryEntry.status != 'archived'
        ).all()
        
        results = []
        query_lower = query.lower()
        
        for entry in entries:
            scores = []
            
            # Term matching
            term_score = fuzz.ratio(query_lower, entry.term.lower()) / 100
            scores.append(term_score * 1.0)
            
            # Definition matching
            def_score = fuzz.partial_ratio(query_lower, entry.definition.lower()) / 100
            scores.append(def_score * 0.7)
            
            # Synonym matching
            if entry.synonyms:
                syn_scores = [
                    fuzz.ratio(query_lower, syn.lower()) / 100 
                    for syn in entry.synonyms
                ]
                if syn_scores:
                    scores.append(max(syn_scores) * 0.9)
            
            # Abbreviation matching
            if entry.abbreviations:
                abbrev_scores = [
                    fuzz.ratio(query_lower, abbrev.lower()) / 100 
                    for abbrev in entry.abbreviations
                ]
                if abbrev_scores:
                    scores.append(max(abbrev_scores) * 0.8)
            
            combined_score = max(scores) if scores else 0
            
            if combined_score > 0.3:
                result = {
                    'object_type': 'dictionary_entry',
                    'object_id': entry.id,
                    'score': round(combined_score, 3),
                    'term': entry.term,
                    'definition': entry.definition,
                    'category': entry.category,
                    'domain': entry.domain,
                    'synonyms': entry.synonyms or [],
                    'search_method': 'fuzzy'
                }
                results.append(result)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _log_search(self, query: str, index_id: Optional[int], search_type: str, 
                   results: List[Dict], search_time_ms: float, project_id: int):
        """Log search for analytics"""
        try:
            search_log = SearchLog(
                project_id=project_id,
                query_text=query,
                query_type=search_type,
                index_id=index_id,
                results_count=len(results),
                top_score=results[0]['score'] if results else 0.0,
                response_time_ms=search_time_ms,
                user_id='anonymous',  # TODO: Get from session
                session_id='session_123'  # TODO: Get from session
            )
            
            db.session.add(search_log)
            db.session.commit()
            
        except Exception as e:
            logger.warning(f"Failed to log search: {str(e)}")
            db.session.rollback()
    
    def get_search_suggestions(self, partial_query: str, project_id: int, 
                             limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query"""
        try:
            suggestions = set()
            
            # Get suggestions from table names
            tables = db.session.query(Table.name).join(
                Table.source
            ).filter(
                Table.source.has(project_id=project_id)
            ).all()
            
            for (name,) in tables:
                if partial_query.lower() in name.lower():
                    suggestions.add(name)
            
            # Get suggestions from column names
            columns = db.session.query(Column.name).join(
                Column.table
            ).join(
                Table.source
            ).filter(
                Table.source.has(project_id=project_id)
            ).all()
            
            for (name,) in columns:
                if partial_query.lower() in name.lower():
                    suggestions.add(name)
            
            # Get suggestions from dictionary terms
            terms = db.session.query(DictionaryEntry.term).filter_by(
                project_id=project_id
            ).filter(
                DictionaryEntry.status != 'archived'
            ).all()
            
            for (term,) in terms:
                if partial_query.lower() in term.lower():
                    suggestions.add(term)
            
            return list(suggestions)[:limit]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            return []