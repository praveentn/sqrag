# backend/api/routes/embeddings.py
"""
Embeddings and indexing API routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from typing import List, Optional, Dict, Any
import logging
import asyncio

from backend.database import get_db
from backend.models import (
    Embedding, Index, Project, Table, Column, DictionaryEntry,
    ObjectType, IndexType, IndexStatus, EmbeddingModel
)
from backend.schemas.embedding import (
    EmbeddingCreateRequest, EmbeddingResponse, EmbeddingListResponse,
    IndexCreateRequest, IndexResponse, IndexListResponse, IndexStatsResponse,
    EmbeddingBatchRequest, IndexBuildResponse
)
from backend.api.deps import get_current_user, get_project_access
from backend.services.embedding_service import embedding_service
from config import Config

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=EmbeddingListResponse)
async def list_embeddings(
    project_id: int = Query(..., description="Project ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    object_type: Optional[ObjectType] = Query(None),
    model_name: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """List embeddings for a project"""
    
    # Build query
    query = select(Embedding).where(Embedding.project_id == project_id)
    
    # Apply filters
    if object_type:
        query = query.where(Embedding.object_type == object_type)
    
    if model_name:
        query = query.where(Embedding.model_name == model_name)
    
    # Get total count
    count_query = select(func.count(Embedding.id)).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.order_by(Embedding.created_at.desc()).offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    embeddings = result.scalars().all()
    
    return EmbeddingListResponse(
        embeddings=[EmbeddingResponse.from_orm(e) for e in embeddings],
        total=total,
        skip=skip,
        limit=limit
    )

@router.post("/create", response_model=EmbeddingResponse)
async def create_embedding(
    embedding_request: EmbeddingCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Create a single embedding"""
    
    try:
        # Check if embedding already exists
        existing_query = select(Embedding).where(
            Embedding.project_id == embedding_request.project_id,
            Embedding.object_type == embedding_request.object_type,
            Embedding.object_id == embedding_request.object_id,
            Embedding.model_name == embedding_request.model_name
        )
        existing_result = await db.execute(existing_query)
        existing_embedding = existing_result.scalar_one_or_none()
        
        if existing_embedding:
            return EmbeddingResponse.from_orm(existing_embedding)
        
        # Get object data for embedding
        text_content, metadata = await _get_object_data(
            embedding_request.object_type,
            embedding_request.object_id,
            db
        )
        
        # Create embedding
        vector = await embedding_service.create_embedding(
            text_content,
            embedding_request.model_name
        )
        
        # Save to database
        embedding = Embedding.create_from_text(
            project_id=embedding_request.project_id,
            object_type=embedding_request.object_type,
            object_id=embedding_request.object_id,
            text_content=text_content,
            model_name=embedding_request.model_name,
            vector=vector,
            metadata=metadata
        )
        
        db.add(embedding)
        await db.commit()
        await db.refresh(embedding)
        
        logger.info(f"Created embedding for {embedding_request.object_type} {embedding_request.object_id}")
        return EmbeddingResponse.from_orm(embedding)
        
    except Exception as e:
        logger.error(f"Failed to create embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=Dict[str, Any])
async def create_embeddings_batch(
    batch_request: EmbeddingBatchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Create embeddings in batch"""
    
    # Start background task
    background_tasks.add_task(
        _create_embeddings_batch_task,
        batch_request,
        current_user
    )
    
    return {
        "message": "Batch embedding creation started",
        "project_id": batch_request.project_id,
        "object_types": batch_request.object_types,
        "model_name": batch_request.model_name
    }

async def _create_embeddings_batch_task(
    batch_request: EmbeddingBatchRequest,
    user_id: str
):
    """Background task for batch embedding creation"""
    
    try:
        from backend.database import async_session_factory
        
        async with async_session_factory() as db:
            created_count = 0
            total_objects = 0
            
            for object_type in batch_request.object_types:
                # Get objects to embed
                objects = await _get_objects_by_type(
                    batch_request.project_id,
                    object_type,
                    db
                )
                
                total_objects += len(objects)
                
                # Process in batches
                batch_size = Config.EMBEDDING_CONFIG['batch_size']
                for i in range(0, len(objects), batch_size):
                    batch_objects = objects[i:i + batch_size]
                    
                    # Prepare texts and metadata
                    texts = []
                    object_data = []
                    
                    for obj in batch_objects:
                        text_content, metadata = await _get_object_data_from_object(
                            object_type, obj
                        )
                        texts.append(text_content)
                        object_data.append((obj, text_content, metadata))
                    
                    # Create embeddings
                    vectors = await embedding_service.create_embeddings_batch(
                        texts,
                        batch_request.model_name,
                        len(texts)
                    )
                    
                    # Save to database
                    for (obj, text_content, metadata), vector in zip(object_data, vectors):
                        # Check if embedding already exists
                        existing_query = select(Embedding).where(
                            Embedding.project_id == batch_request.project_id,
                            Embedding.object_type == object_type,
                            Embedding.object_id == obj.id,
                            Embedding.model_name == batch_request.model_name
                        )
                        existing_result = await db.execute(existing_query)
                        existing_embedding = existing_result.scalar_one_or_none()
                        
                        if not existing_embedding:
                            embedding = Embedding.create_from_text(
                                project_id=batch_request.project_id,
                                object_type=object_type,
                                object_id=obj.id,
                                text_content=text_content,
                                model_name=batch_request.model_name,
                                vector=vector,
                                metadata=metadata
                            )
                            db.add(embedding)
                            created_count += 1
                    
                    await db.commit()
                    
                    logger.info(f"Created {len(vectors)} embeddings in batch")
            
            logger.info(f"Batch embedding creation completed: {created_count}/{total_objects} created")
            
    except Exception as e:
        logger.error(f"Batch embedding creation failed: {e}")

@router.get("/indexes", response_model=IndexListResponse)
async def list_indexes(
    project_id: int = Query(..., description="Project ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    index_type: Optional[IndexType] = Query(None),
    status: Optional[IndexStatus] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """List indexes for a project"""
    
    # Build query
    query = select(Index).where(
        Index.project_id == project_id,
        Index.is_deleted == False
    )
    
    # Apply filters
    if index_type:
        query = query.where(Index.type == index_type)
    
    if status:
        query = query.where(Index.status == status)
    
    # Get total count
    count_query = select(func.count(Index.id)).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.order_by(Index.created_at.desc()).offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    indexes = result.scalars().all()
    
    return IndexListResponse(
        indexes=[IndexResponse.from_orm(i) for i in indexes],
        total=total,
        skip=skip,
        limit=limit
    )

@router.post("/indexes", response_model=IndexResponse)
async def create_index(
    index_request: IndexCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Create a new search index"""
    
    # Check if index name already exists
    existing_query = select(Index).where(
        Index.project_id == index_request.project_id,
        Index.name == index_request.name,
        Index.is_deleted == False
    )
    existing_result = await db.execute(existing_query)
    existing_index = existing_result.scalar_one_or_none()
    
    if existing_index:
        raise HTTPException(
            status_code=400,
            detail=f"Index with name '{index_request.name}' already exists"
        )
    
    # Create index record
    if index_request.type == IndexType.FAISS:
        index = Index.create_faiss_index(
            project_id=index_request.project_id,
            name=index_request.name,
            model_name=index_request.model_name,
            object_types=index_request.object_types,
            dimensions=index_request.dimensions or 384,  # Default for sentence transformers
            metric=index_request.build_params.get('metric', 'cosine')
        )
    elif index_request.type == IndexType.TFIDF:
        index = Index.create_tfidf_index(
            project_id=index_request.project_id,
            name=index_request.name,
            object_types=index_request.object_types,
            max_features=index_request.build_params.get('max_features', 10000)
        )
    elif index_request.type == IndexType.PGVECTOR:
        index = Index.create_pgvector_index(
            project_id=index_request.project_id,
            name=index_request.name,
            model_name=index_request.model_name,
            object_types=index_request.object_types,
            dimensions=index_request.dimensions or 384
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported index type: {index_request.type}"
        )
    
    index.created_by = current_user
    
    db.add(index)
    await db.commit()
    await db.refresh(index)
    
    # Start background index building
    background_tasks.add_task(
        _build_index_task,
        index.id
    )
    
    logger.info(f"Created index: {index.name} (ID: {index.id})")
    return IndexResponse.from_orm(index)

async def _build_index_task(index_id: int):
    """Background task for building search index"""
    
    try:
        from backend.database import async_session_factory
        
        async with async_session_factory() as db:
            # Get index
            index_query = select(Index).where(Index.id == index_id)
            index_result = await db.execute(index_query)
            index = index_result.scalar_one_or_none()
            
            if not index:
                logger.error(f"Index {index_id} not found")
                return
            
            # Update status
            index.status = IndexStatus.BUILDING
            index.update_build_progress(0.0, "Starting index build")
            await db.commit()
            
            # Get embeddings for the index
            embeddings_query = select(Embedding).where(
                Embedding.project_id == index.project_id,
                Embedding.object_type.in_([ObjectType(ot) for ot in index.object_scope['object_types']])
            )
            
            if index.model_name != "tfidf":
                embeddings_query = embeddings_query.where(
                    Embedding.model_name == index.model_name
                )
            
            embeddings_result = await db.execute(embeddings_query)
            embeddings = embeddings_result.scalars().all()
            
            if not embeddings:
                index.set_error("No embeddings found for index")
                await db.commit()
                return
            
            index.update_build_progress(20.0, f"Found {len(embeddings)} embeddings")
            await db.commit()
            
            # Build index based on type
            if index.type == IndexType.FAISS:
                await _build_faiss_index(index, embeddings, db)
            elif index.type == IndexType.TFIDF:
                await _build_tfidf_index(index, embeddings, db)
            else:
                index.set_error(f"Index type {index.type} not implemented")
                await db.commit()
                return
            
            logger.info(f"Index build completed: {index.name}")
            
    except Exception as e:
        logger.error(f"Index build failed for index {index_id}: {e}")
        # Update index status to error
        async with async_session_factory() as db:
            index_query = select(Index).where(Index.id == index_id)
            index_result = await db.execute(index_query)
            index = index_result.scalar_one_or_none()
            if index:
                index.set_error(str(e))
                await db.commit()

async def _build_faiss_index(index: Index, embeddings: List[Embedding], db: AsyncSession):
    """Build FAISS index"""
    
    import time
    start_time = time.time()
    
    try:
        # Build FAISS index
        faiss_index, metadata = await embedding_service.build_faiss_index(
            embeddings,
            index.get_build_params()
        )
        
        index.update_build_progress(80.0, "Saving FAISS index")
        await db.commit()
        
        # Save index to disk
        index_path = f"indexes/faiss/{index.id}/index.faiss"
        await embedding_service.save_faiss_index(
            faiss_index,
            metadata,
            index_path
        )
        
        # Update index record
        build_time = time.time() - start_time
        index.set_ready(len(embeddings), build_time)
        index.index_path = index_path
        index.dimensions = metadata['dimensions']
        
        await db.commit()
        
    except Exception as e:
        index.set_error(f"FAISS index build failed: {e}")
        await db.commit()
        raise

async def _build_tfidf_index(index: Index, embeddings: List[Embedding], db: AsyncSession):
    """Build TF-IDF index"""
    
    import time
    import pickle
    from pathlib import Path
    
    start_time = time.time()
    
    try:
        # Prepare texts
        texts = [emb.text_content for emb in embeddings]
        metadata = [emb.metadata for emb in embeddings]
        
        index.update_build_progress(40.0, "Creating TF-IDF vectors")
        await db.commit()
        
        # Build TF-IDF index
        tfidf_index, index_metadata = await embedding_service.create_tfidf_index(
            texts,
            metadata,
            index.get_build_params()
        )
        
        index.update_build_progress(80.0, "Saving TF-IDF index")
        await db.commit()
        
        # Save index to disk
        index_path = f"indexes/tfidf/{index.id}/index.pkl"
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(index_path, 'wb') as f:
            pickle.dump(tfidf_index, f)
        
        # Update index record
        build_time = time.time() - start_time
        index.set_ready(len(embeddings), build_time)
        index.index_path = index_path
        
        await db.commit()
        
    except Exception as e:
        index.set_error(f"TF-IDF index build failed: {e}")
        await db.commit()
        raise

@router.get("/indexes/{index_id}", response_model=IndexResponse)
async def get_index(
    index_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get index by ID"""
    
    query = select(Index).where(
        Index.id == index_id,
        Index.is_deleted == False
    )
    result = await db.execute(query)
    index = result.scalar_one_or_none()
    
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == index.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return IndexResponse.from_orm(index)

@router.delete("/indexes/{index_id}")
async def delete_index(
    index_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete index"""
    
    query = select(Index).where(
        Index.id == index_id,
        Index.is_deleted == False
    )
    result = await db.execute(query)
    index = result.scalar_one_or_none()
    
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == index.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Soft delete
    index.soft_delete(current_user)
    
    await db.commit()
    
    logger.info(f"Deleted index: {index.name} (ID: {index.id})")
    return {"message": "Index deleted successfully"}

@router.get("/indexes/{index_id}/stats", response_model=IndexStatsResponse)
async def get_index_stats(
    index_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get index statistics"""
    
    query = select(Index).where(
        Index.id == index_id,
        Index.is_deleted == False
    )
    result = await db.execute(query)
    index = result.scalar_one_or_none()
    
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    
    return IndexStatsResponse(
        index_id=index_id,
        name=index.name,
        type=index.type,
        status=index.status,
        total_objects=index.total_objects,
        indexed_objects=index.indexed_objects,
        dimensions=index.dimensions,
        build_time=index.build_time,
        index_size=index.index_size,
        build_progress=index.build_progress,
        created_at=index.created_at,
        updated_at=index.updated_at
    )

# Helper functions

async def _get_object_data(
    object_type: ObjectType,
    object_id: int,
    db: AsyncSession
) -> tuple[str, dict]:
    """Get object data for embedding"""
    
    if object_type == ObjectType.TABLE:
        query = select(Table).where(Table.id == object_id)
        result = await db.execute(query)
        table = result.scalar_one_or_none()
        if not table:
            raise ValueError(f"Table {object_id} not found")
        
        # Get columns for more context
        columns_query = select(Column).where(Column.table_id == object_id)
        columns_result = await db.execute(columns_query)
        columns = columns_result.scalars().all()
        
        text_content = await _get_table_text_content(table, columns)
        metadata = {
            "table_id": table.id,
            "table_name": table.name,
            "schema_name": table.schema_name,
            "row_count": table.row_count,
            "column_count": len(columns)
        }
        
    elif object_type == ObjectType.COLUMN:
        query = select(Column).where(Column.id == object_id)
        result = await db.execute(query)
        column = result.scalar_one_or_none()
        if not column:
            raise ValueError(f"Column {object_id} not found")
        
        text_content = await _get_column_text_content(column)
        metadata = {
            "column_id": column.id,
            "column_name": column.name,
            "table_id": column.table_id,
            "data_type": column.data_type,
            "is_primary_key": column.is_primary_key
        }
        
    elif object_type == ObjectType.DICTIONARY_ENTRY:
        query = select(DictionaryEntry).where(DictionaryEntry.id == object_id)
        result = await db.execute(query)
        entry = result.scalar_one_or_none()
        if not entry:
            raise ValueError(f"Dictionary entry {object_id} not found")
        
        text_content = await _get_dictionary_text_content(entry)
        metadata = {
            "entry_id": entry.id,
            "term": entry.term,
            "category": entry.category.value,
            "domain_tags": entry.domain_tags
        }
        
    else:
        raise ValueError(f"Unsupported object type: {object_type}")
    
    return text_content, metadata

async def _get_objects_by_type(
    project_id: int,
    object_type: ObjectType,
    db: AsyncSession
) -> List[Any]:
    """Get all objects of given type for a project"""
    
    if object_type == ObjectType.TABLE:
        from backend.models import Source
        query = select(Table).join(Source).where(Source.project_id == project_id)
        
    elif object_type == ObjectType.COLUMN:
        from backend.models import Source
        query = select(Column).join(Table).join(Source).where(Source.project_id == project_id)
        
    elif object_type == ObjectType.DICTIONARY_ENTRY:
        query = select(DictionaryEntry).where(
            DictionaryEntry.project_id == project_id,
            DictionaryEntry.is_deleted == False
        )
        
    else:
        return []
    
    result = await db.execute(query)
    return result.scalars().all()

async def _get_object_data_from_object(
    object_type: ObjectType,
    obj: Any
) -> tuple[str, dict]:
    """Get text content and metadata from object instance"""
    
    if object_type == ObjectType.TABLE:
        text_content = await _get_table_text_content(obj, obj.columns)
        metadata = {
            "table_id": obj.id,
            "table_name": obj.name,
            "schema_name": obj.schema_name,
            "row_count": obj.row_count,
            "column_count": len(obj.columns) if obj.columns else 0
        }
        
    elif object_type == ObjectType.COLUMN:
        text_content = await _get_column_text_content(obj)
        metadata = {
            "column_id": obj.id,
            "column_name": obj.name,
            "table_id": obj.table_id,
            "data_type": obj.data_type,
            "is_primary_key": obj.is_primary_key
        }
        
    elif object_type == ObjectType.DICTIONARY_ENTRY:
        text_content = await _get_dictionary_text_content(obj)
        metadata = {
            "entry_id": obj.id,
            "term": obj.term,
            "category": obj.category.value,
            "domain_tags": obj.domain_tags or []
        }
        
    else:
        raise ValueError(f"Unsupported object type: {object_type}")
    
    return text_content, metadata

async def _get_table_text_content(table: Table, columns: List[Column] = None) -> str:
    """Generate text content for table embedding"""
    
    parts = [table.name]
    
    if table.display_name and table.display_name != table.name:
        parts.append(table.display_name)
    
    if table.description:
        parts.append(table.description)
    
    if columns:
        column_names = [col.name for col in columns]
        parts.append(" ".join(column_names))
    
    return " | ".join(parts)

async def _get_column_text_content(column: Column) -> str:
    """Generate text content for column embedding"""
    
    parts = [column.name]
    
    if column.display_name and column.display_name != column.name:
        parts.append(column.display_name)
    
    if column.description:
        parts.append(column.description)
    
    parts.append(column.data_type)
    
    if column.sample_values:
        sample_text = " ".join(str(v) for v in column.sample_values[:5])
        parts.append(sample_text)
    
    return " | ".join(parts)

async def _get_dictionary_text_content(entry: DictionaryEntry) -> str:
    """Generate text content for dictionary entry embedding"""
    
    parts = [entry.term]
    
    if entry.definition:
        parts.append(entry.definition)
    
    if entry.synonyms:
        parts.extend(entry.synonyms)
    
    if entry.abbreviations:
        parts.extend(entry.abbreviations)
    
    if entry.context:
        parts.append(entry.context)
    
    return " | ".join(parts)