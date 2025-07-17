# backend/api/routes/search.py
"""
Search API routes for semantic and keyword search
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional, Dict, Any
import logging
import time
import pickle
from pathlib import Path

from backend.database import get_db
from backend.models import (
    Index, Embedding, Project, Table, Column, DictionaryEntry,
    ObjectType, IndexType, IndexStatus
)
from backend.schemas.embedding import SearchRequest, SearchResponse, SearchResult
from backend.api.deps import get_current_user, get_project_access
from backend.services.embedding_service import embedding_service
from config import Config

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/", response_model=SearchResponse)
async def search(
    search_request: SearchRequest,
    project_id: int = Query(..., description="Project ID"),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Perform search across project indexes"""
    
    start_time = time.time()
    
    try:
        # Get available indexes
        indexes_query = select(Index).where(
            Index.project_id == project_id,
            Index.status == IndexStatus.READY,
            Index.is_deleted == False
        )
        
        # Filter by specific indexes if requested
        if search_request.index_ids:
            indexes_query = indexes_query.where(Index.id.in_(search_request.index_ids))
        
        indexes_result = await db.execute(indexes_query)
        indexes = indexes_result.scalars().all()
        
        if not indexes:
            return SearchResponse(
                results=[],
                total_found=0,
                query=search_request.query,
                search_time=time.time() - start_time,
                indexes_used=[],
                search_type=search_request.search_type
            )
        
        # Perform search based on type
        all_results = []
        indexes_used = []
        
        for index in indexes:
            try:
                if search_request.search_type == "semantic" and index.type in [IndexType.FAISS, IndexType.PGVECTOR]:
                    results = await _search_semantic_index(index, search_request, db)
                elif search_request.search_type == "keyword" and index.type == IndexType.TFIDF:
                    results = await _search_keyword_index(index, search_request, db)
                elif search_request.search_type == "hybrid":
                    results = await _search_hybrid_index(index, search_request, db)
                else:
                    continue  # Skip incompatible index types
                
                all_results.extend(results)
                indexes_used.append(index.id)
                
            except Exception as e:
                logger.error(f"Search failed for index {index.id}: {e}")
                continue
        
        # Merge and rank results
        merged_results = _merge_search_results(all_results, search_request.limit)
        
        # Apply threshold filter
        filtered_results = [
            r for r in merged_results
            if r['similarity_score'] >= search_request.threshold
        ]
        
        # Enhance results with object details
        enhanced_results = await _enhance_search_results(filtered_results, db)
        
        search_time = time.time() - start_time
        
        logger.info(f"Search completed: {len(enhanced_results)} results in {search_time:.3f}s")
        
        return SearchResponse(
            results=enhanced_results,
            total_found=len(enhanced_results),
            query=search_request.query,
            search_time=search_time,
            indexes_used=indexes_used,
            search_type=search_request.search_type
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

async def _search_semantic_index(
    index: Index,
    search_request: SearchRequest,
    db: AsyncSession
) -> List[Dict[str, Any]]:
    """Search semantic (vector) index"""
    
    try:
        # Create query embedding
        query_embedding = await embedding_service.create_embedding(
            search_request.query,
            index.model_name
        )
        
        if index.type == IndexType.FAISS:
            # Load FAISS index
            if not index.index_path or not Path(index.index_path).exists():
                logger.warning(f"FAISS index file not found: {index.index_path}")
                return []
            
            faiss_index, metadata = await embedding_service.load_faiss_index(
                index.index_path
            )
            
            # Search FAISS index
            results = await embedding_service.search_faiss_index(
                faiss_index,
                query_embedding,
                metadata['metadata'],
                k=search_request.limit * 2,  # Get more for filtering
                metric=index.get_build_params().get('metric', 'cosine')
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'object_type': ObjectType(result['object_type']),
                    'object_id': result['object_id'],
                    'similarity_score': result['similarity_score'],
                    'rank': result['rank'],
                    'matched_text': result.get('text_content', ''),
                    'index_id': index.id,
                    'search_method': 'faiss'
                })
            
            return formatted_results
            
        elif index.type == IndexType.PGVECTOR:
            # TODO: Implement pgvector search
            logger.warning("pgvector search not implemented yet")
            return []
        
        return []
        
    except Exception as e:
        logger.error(f"Semantic search failed for index {index.id}: {e}")
        return []

async def _search_keyword_index(
    index: Index,
    search_request: SearchRequest,
    db: AsyncSession
) -> List[Dict[str, Any]]:
    """Search keyword (TF-IDF) index"""
    
    try:
        if index.type != IndexType.TFIDF:
            return []
        
        # Load TF-IDF index
        if not index.index_path or not Path(index.index_path).exists():
            logger.warning(f"TF-IDF index file not found: {index.index_path}")
            return []
        
        with open(index.index_path, 'rb') as f:
            tfidf_index = pickle.load(f)
        
        # Search TF-IDF index
        results = await embedding_service.search_tfidf_index(
            tfidf_index,
            search_request.query,
            k=search_request.limit * 2
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'object_type': ObjectType(result['object_type']),
                'object_id': result['object_id'],
                'similarity_score': result['similarity_score'],
                'rank': result['rank'],
                'matched_text': result.get('text_content', ''),
                'index_id': index.id,
                'search_method': 'tfidf'
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Keyword search failed for index {index.id}: {e}")
        return []

async def _search_hybrid_index(
    index: Index,
    search_request: SearchRequest,
    db: AsyncSession
) -> List[Dict[str, Any]]:
    """Search with hybrid approach (if index supports it)"""
    
    try:
        # For now, just use the primary search method of the index
        if index.type in [IndexType.FAISS, IndexType.PGVECTOR]:
            return await _search_semantic_index(index, search_request, db)
        elif index.type == IndexType.TFIDF:
            return await _search_keyword_index(index, search_request, db)
        
        return []
        
    except Exception as e:
        logger.error(f"Hybrid search failed for index {index.id}: {e}")
        return []

def _merge_search_results(
    all_results: List[Dict[str, Any]],
    limit: int
) -> List[Dict[str, Any]]:
    """Merge and deduplicate search results from multiple indexes"""
    
    # Use a dictionary to deduplicate by (object_type, object_id)
    unique_results = {}
    
    for result in all_results:
        key = (result['object_type'], result['object_id'])
        
        if key not in unique_results:
            unique_results[key] = result
        else:
            # Keep the result with higher similarity score
            existing = unique_results[key]
            if result['similarity_score'] > existing['similarity_score']:
                unique_results[key] = result
    
    # Sort by similarity score and limit
    sorted_results = sorted(
        unique_results.values(),
        key=lambda x: x['similarity_score'],
        reverse=True
    )
    
    # Re-rank
    for i, result in enumerate(sorted_results[:limit]):
        result['rank'] = i + 1
    
    return sorted_results[:limit]

async def _enhance_search_results(
    results: List[Dict[str, Any]],
    db: AsyncSession
) -> List[SearchResult]:
    """Enhance search results with object details"""
    
    enhanced_results = []
    
    for result in results:
        try:
            object_type = result['object_type']
            object_id = result['object_id']
            
            # Get object details
            object_name = ""
            context_metadata = {}
            
            if object_type == ObjectType.TABLE:
                query = select(Table).where(Table.id == object_id)
                obj_result = await db.execute(query)
                table = obj_result.scalar_one_or_none()
                
                if table:
                    object_name = table.display_name or table.name
                    context_metadata = {
                        "schema_name": table.schema_name,
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "table_type": table.table_type
                    }
            
            elif object_type == ObjectType.COLUMN:
                query = select(Column).where(Column.id == object_id)
                obj_result = await db.execute(query)
                column = obj_result.scalar_one_or_none()
                
                if column:
                    object_name = column.display_name or column.name
                    context_metadata = {
                        "table_id": column.table_id,
                        "data_type": column.data_type,
                        "is_primary_key": column.is_primary_key,
                        "is_nullable": column.is_nullable,
                        "pii_flag": column.pii_flag
                    }
            
            elif object_type == ObjectType.DICTIONARY_ENTRY:
                query = select(DictionaryEntry).where(DictionaryEntry.id == object_id)
                obj_result = await db.execute(query)
                entry = obj_result.scalar_one_or_none()
                
                if entry:
                    object_name = entry.term
                    context_metadata = {
                        "definition": entry.definition[:200] + "..." if len(entry.definition) > 200 else entry.definition,
                        "category": entry.category.value,
                        "status": entry.status.value,
                        "domain_tags": entry.domain_tags or []
                    }
            
            # Create enhanced result
            enhanced_result = SearchResult(
                object_type=object_type,
                object_id=object_id,
                object_name=object_name,
                similarity_score=result['similarity_score'],
                rank=result['rank'],
                matched_text=result.get('matched_text'),
                context_metadata=context_metadata,
                index_id=result['index_id'],
                search_method=result['search_method']
            )
            
            enhanced_results.append(enhanced_result)
            
        except Exception as e:
            logger.error(f"Failed to enhance search result: {e}")
            continue
    
    return enhanced_results

@router.get("/suggest")
async def search_suggestions(
    project_id: int = Query(..., description="Project ID"),
    query: str = Query(..., min_length=1, description="Partial query for suggestions"),
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Get search suggestions based on partial query"""
    
    suggestions = []
    
    try:
        # Get suggestions from dictionary entries
        dict_query = select(DictionaryEntry).where(
            DictionaryEntry.project_id == project_id,
            DictionaryEntry.is_deleted == False,
            DictionaryEntry.term.ilike(f"%{query}%")
        ).limit(limit)
        
        dict_result = await db.execute(dict_query)
        dict_entries = dict_result.scalars().all()
        
        for entry in dict_entries:
            suggestions.append({
                "text": entry.term,
                "type": "dictionary_term",
                "description": entry.definition[:100] + "..." if len(entry.definition) > 100 else entry.definition,
                "category": entry.category.value
            })
        
        # Get suggestions from table names
        from backend.models import Source
        tables_query = select(Table).join(Source).where(
            Source.project_id == project_id,
            Table.name.ilike(f"%{query}%")
        ).limit(limit - len(suggestions))
        
        tables_result = await db.execute(tables_query)
        tables = tables_result.scalars().all()
        
        for table in tables:
            suggestions.append({
                "text": table.display_name or table.name,
                "type": "table",
                "description": f"Table with {table.row_count} rows, {table.column_count} columns",
                "category": "data_source"
            })
        
        # Get suggestions from column names
        if len(suggestions) < limit:
            columns_query = select(Column).join(Table).join(Source).where(
                Source.project_id == project_id,
                Column.name.ilike(f"%{query}%")
            ).limit(limit - len(suggestions))
            
            columns_result = await db.execute(columns_query)
            columns = columns_result.scalars().all()
            
            for column in columns:
                suggestions.append({
                    "text": column.display_name or column.name,
                    "type": "column",
                    "description": f"{column.data_type} column",
                    "category": "data_field"
                })
        
        return {
            "suggestions": suggestions[:limit],
            "query": query,
            "total": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Search suggestions failed: {e}")
        return {
            "suggestions": [],
            "query": query,
            "total": 0,
            "error": str(e)
        }

@router.get("/similar")
async def find_similar_objects(
    project_id: int = Query(..., description="Project ID"),
    object_type: ObjectType = Query(..., description="Object type"),
    object_id: int = Query(..., description="Object ID"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Find similar objects to a given object"""
    
    try:
        # Get the object's embedding
        embedding_query = select(Embedding).where(
            Embedding.project_id == project_id,
            Embedding.object_type == object_type,
            Embedding.object_id == object_id
        ).limit(1)
        
        embedding_result = await db.execute(embedding_query)
        source_embedding = embedding_result.scalar_one_or_none()
        
        if not source_embedding:
            raise HTTPException(
                status_code=404,
                detail="No embedding found for the specified object"
            )
        
        # Find indexes that can search this object type
        indexes_query = select(Index).where(
            Index.project_id == project_id,
            Index.status == IndexStatus.READY,
            Index.is_deleted == False,
            Index.object_scope.contains({"object_types": [object_type.value]})
        )
        
        indexes_result = await db.execute(indexes_query)
        indexes = indexes_result.scalars().all()
        
        if not indexes:
            return {
                "similar_objects": [],
                "total_found": 0,
                "object_type": object_type,
                "object_id": object_id
            }
        
        # Use the first suitable index for similarity search
        index = indexes[0]
        
        # Create a search request using the object's text content
        search_request = SearchRequest(
            query=source_embedding.text_content,
            limit=limit + 1,  # +1 to exclude the source object
            threshold=0.3,
            search_type="semantic"
        )
        
        # Perform search
        if index.type == IndexType.FAISS:
            results = await _search_semantic_index(index, search_request, db)
        else:
            results = []
        
        # Filter out the source object
        filtered_results = [
            r for r in results
            if not (r['object_type'] == object_type and r['object_id'] == object_id)
        ]
        
        # Enhance results
        enhanced_results = await _enhance_search_results(filtered_results[:limit], db)
        
        return {
            "similar_objects": enhanced_results,
            "total_found": len(enhanced_results),
            "object_type": object_type,
            "object_id": object_id
        }
        
    except Exception as e:
        logger.error(f"Similar objects search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indexes/status")
async def get_search_indexes_status(
    project_id: int = Query(..., description="Project ID"),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Get status of all search indexes for a project"""
    
    indexes_query = select(Index).where(
        Index.project_id == project_id,
        Index.is_deleted == False
    )
    
    indexes_result = await db.execute(indexes_query)
    indexes = indexes_result.scalars().all()
    
    status_summary = {
        "total_indexes": len(indexes),
        "ready_indexes": 0,
        "building_indexes": 0,
        "error_indexes": 0,
        "indexes": []
    }
    
    for index in indexes:
        index_info = {
            "id": index.id,
            "name": index.name,
            "type": index.type.value,
            "status": index.status.value,
            "progress": index.build_progress,
            "object_types": index.object_scope.get("object_types", []),
            "total_objects": index.total_objects,
            "indexed_objects": index.indexed_objects
        }
        
        status_summary["indexes"].append(index_info)
        
        if index.status == IndexStatus.READY:
            status_summary["ready_indexes"] += 1
        elif index.status == IndexStatus.BUILDING:
            status_summary["building_indexes"] += 1
        elif index.status == IndexStatus.ERROR:
            status_summary["error_indexes"] += 1
    
    return status_summary