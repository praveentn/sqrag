# backend/api/routes/dictionary.py
"""
Data dictionary management API routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from typing import List, Optional, Dict, Any
import logging

from backend.database import get_db
from backend.models import (
    DictionaryEntry, DictionaryStatus, DictionaryCategory, 
    Project, Table, Column
)
from backend.schemas.dictionary import (
    DictionaryEntryCreate, DictionaryEntryUpdate, DictionaryEntryResponse,
    DictionaryListResponse, DictionaryBulkCreate, DictionarySearchResponse,
    DictionaryStatsResponse, DictionaryApprovalRequest
)
from backend.api.deps import get_current_user, get_project_access

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=DictionaryListResponse)
async def list_dictionary_entries(
    project_id: int = Query(..., description="Project ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
    category: Optional[DictionaryCategory] = Query(None),
    status: Optional[DictionaryStatus] = Query(None),
    domain_tag: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """List dictionary entries for a project"""
    
    # Build query
    query = select(DictionaryEntry).where(
        DictionaryEntry.project_id == project_id,
        DictionaryEntry.is_deleted == False
    )
    
    # Apply filters
    if search:
        query = query.where(
            or_(
                DictionaryEntry.term.ilike(f"%{search}%"),
                DictionaryEntry.definition.ilike(f"%{search}%"),
                DictionaryEntry.context.ilike(f"%{search}%")
            )
        )
    
    if category:
        query = query.where(DictionaryEntry.category == category)
    
    if status:
        query = query.where(DictionaryEntry.status == status)
    
    if domain_tag:
        query = query.where(
            DictionaryEntry.domain_tags.contains([domain_tag])
        )
    
    # Get total count
    count_query = select(func.count(DictionaryEntry.id)).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination and ordering
    query = query.order_by(DictionaryEntry.term).offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    entries = result.scalars().all()
    
    return DictionaryListResponse(
        entries=[DictionaryEntryResponse.from_orm(e) for e in entries],
        total=total,
        skip=skip,
        limit=limit
    )

@router.post("/", response_model=DictionaryEntryResponse)
async def create_dictionary_entry(
    entry_data: DictionaryEntryCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Create a new dictionary entry"""
    
    # Check if term already exists in project
    existing_query = select(DictionaryEntry).where(
        DictionaryEntry.project_id == entry_data.project_id,
        DictionaryEntry.term.ilike(entry_data.term),
        DictionaryEntry.is_deleted == False
    )
    existing_result = await db.execute(existing_query)
    existing_entry = existing_result.scalar_one_or_none()
    
    if existing_entry:
        raise HTTPException(
            status_code=400,
            detail=f"Term '{entry_data.term}' already exists in dictionary"
        )
    
    # Create entry
    entry = DictionaryEntry(
        project_id=entry_data.project_id,
        term=entry_data.term,
        definition=entry_data.definition,
        category=entry_data.category,
        synonyms=entry_data.synonyms or [],
        abbreviations=entry_data.abbreviations or [],
        domain_tags=entry_data.domain_tags or [],
        context=entry_data.context,
        examples=entry_data.examples or [],
        created_by=current_user
    )
    
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    
    logger.info(f"Created dictionary entry: {entry.term} (ID: {entry.id})")
    return DictionaryEntryResponse.from_orm(entry)

@router.get("/{entry_id}", response_model=DictionaryEntryResponse)
async def get_dictionary_entry(
    entry_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get dictionary entry by ID"""
    
    query = select(DictionaryEntry).where(
        DictionaryEntry.id == entry_id,
        DictionaryEntry.is_deleted == False
    )
    result = await db.execute(query)
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Dictionary entry not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == entry.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return DictionaryEntryResponse.from_orm(entry)

@router.put("/{entry_id}", response_model=DictionaryEntryResponse)
async def update_dictionary_entry(
    entry_id: int,
    entry_update: DictionaryEntryUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update dictionary entry"""
    
    query = select(DictionaryEntry).where(
        DictionaryEntry.id == entry_id,
        DictionaryEntry.is_deleted == False
    )
    result = await db.execute(query)
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Dictionary entry not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == entry.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update fields
    update_data = entry_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(entry, field, value)
    
    entry.updated_by = current_user
    
    # If updating approved entry, reset to draft
    if entry.status == DictionaryStatus.APPROVED and update_data:
        entry.status = DictionaryStatus.DRAFT
        entry.version += 1
    
    await db.commit()
    await db.refresh(entry)
    
    logger.info(f"Updated dictionary entry: {entry.term} (ID: {entry.id})")
    return DictionaryEntryResponse.from_orm(entry)

@router.delete("/{entry_id}")
async def delete_dictionary_entry(
    entry_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete dictionary entry"""
    
    query = select(DictionaryEntry).where(
        DictionaryEntry.id == entry_id,
        DictionaryEntry.is_deleted == False
    )
    result = await db.execute(query)
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Dictionary entry not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == entry.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Soft delete
    entry.soft_delete(current_user)
    
    await db.commit()
    
    logger.info(f"Deleted dictionary entry: {entry.term} (ID: {entry.id})")
    return {"message": "Dictionary entry deleted successfully"}

@router.post("/bulk", response_model=List[DictionaryEntryResponse])
async def create_bulk_dictionary_entries(
    bulk_data: DictionaryBulkCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Create multiple dictionary entries"""
    
    created_entries = []
    
    for entry_data in bulk_data.entries:
        # Check if term already exists
        existing_query = select(DictionaryEntry).where(
            DictionaryEntry.project_id == bulk_data.project_id,
            DictionaryEntry.term.ilike(entry_data.term),
            DictionaryEntry.is_deleted == False
        )
        existing_result = await db.execute(existing_query)
        existing_entry = existing_result.scalar_one_or_none()
        
        if existing_entry:
            logger.warning(f"Skipping duplicate term: {entry_data.term}")
            continue
        
        # Create entry
        entry = DictionaryEntry(
            project_id=bulk_data.project_id,
            term=entry_data.term,
            definition=entry_data.definition,
            category=entry_data.category,
            synonyms=entry_data.synonyms or [],
            abbreviations=entry_data.abbreviations or [],
            domain_tags=entry_data.domain_tags or [],
            context=entry_data.context,
            examples=entry_data.examples or [],
            is_auto_generated=bulk_data.is_auto_generated,
            created_by=current_user
        )
        
        db.add(entry)
        created_entries.append(entry)
    
    await db.commit()
    
    # Refresh all entries
    for entry in created_entries:
        await db.refresh(entry)
    
    logger.info(f"Created {len(created_entries)} dictionary entries in bulk")
    return [DictionaryEntryResponse.from_orm(e) for e in created_entries]

@router.post("/auto-generate")
async def auto_generate_dictionary(
    project_id: int = Query(..., description="Project ID"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Auto-generate dictionary entries from project data"""
    
    # Start background task for auto-generation
    background_tasks.add_task(
        _auto_generate_dictionary_task,
        project_id,
        current_user,
        db
    )
    
    return {"message": "Dictionary auto-generation started"}

async def _auto_generate_dictionary_task(
    project_id: int,
    user_id: str,
    db_session: AsyncSession
):
    """Background task for auto-generating dictionary entries"""
    
    try:
        from backend.database import async_session_factory
        
        async with async_session_factory() as db:
            # Get all tables and columns for the project
            tables_query = select(Table).join(
                Source, Table.source_id == Source.id
            ).where(Source.project_id == project_id)
            
            tables_result = await db.execute(tables_query)
            tables = tables_result.scalars().all()
            
            created_count = 0
            
            for table in tables:
                # Generate entry for table
                table_suggestion = DictionaryEntry.suggest_from_table(
                    table.name,
                    [col.name for col in table.columns] if table.columns else []
                )
                
                # Check if term already exists
                existing_query = select(DictionaryEntry).where(
                    DictionaryEntry.project_id == project_id,
                    DictionaryEntry.term.ilike(table_suggestion['term']),
                    DictionaryEntry.is_deleted == False
                )
                existing_result = await db.execute(existing_query)
                existing_entry = existing_result.scalar_one_or_none()
                
                if not existing_entry:
                    entry = DictionaryEntry(
                        project_id=project_id,
                        created_by=user_id,
                        **table_suggestion
                    )
                    db.add(entry)
                    created_count += 1
                
                # Generate entries for columns
                for column in table.columns:
                    column_suggestion = DictionaryEntry.suggest_from_column(
                        column.name,
                        table.name,
                        column.sample_values
                    )
                    
                    # Check if term already exists
                    existing_query = select(DictionaryEntry).where(
                        DictionaryEntry.project_id == project_id,
                        DictionaryEntry.term.ilike(column_suggestion['term']),
                        DictionaryEntry.is_deleted == False
                    )
                    existing_result = await db.execute(existing_query)
                    existing_entry = existing_result.scalar_one_or_none()
                    
                    if not existing_entry:
                        entry = DictionaryEntry(
                            project_id=project_id,
                            created_by=user_id,
                            **column_suggestion
                        )
                        db.add(entry)
                        created_count += 1
            
            await db.commit()
            logger.info(f"Auto-generated {created_count} dictionary entries for project {project_id}")
            
    except Exception as e:
        logger.error(f"Dictionary auto-generation failed: {e}")

@router.get("/search", response_model=DictionarySearchResponse)
async def search_dictionary(
    project_id: int = Query(..., description="Project ID"),
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    exact_match: bool = Query(False, description="Exact match only"),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Search dictionary entries"""
    
    # Get all entries for the project
    entries_query = select(DictionaryEntry).where(
        DictionaryEntry.project_id == project_id,
        DictionaryEntry.is_deleted == False,
        DictionaryEntry.status.in_([DictionaryStatus.APPROVED, DictionaryStatus.DRAFT])
    )
    
    entries_result = await db.execute(entries_query)
    entries = entries_result.scalars().all()
    
    # Search and rank entries
    results = []
    for entry in entries:
        if exact_match:
            if entry.matches_term(query, exact=True):
                results.append({
                    "entry": entry,
                    "similarity_score": 1.0,
                    "match_type": "exact"
                })
        else:
            similarity = entry.calculate_similarity(query)
            if similarity > 0.3:  # Threshold for relevance
                match_type = "exact" if entry.matches_term(query, exact=True) else "fuzzy"
                results.append({
                    "entry": entry,
                    "similarity_score": similarity,
                    "match_type": match_type
                })
    
    # Sort by similarity score
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Limit results
    results = results[:limit]
    
    # Format response
    search_results = []
    for result in results:
        entry_data = DictionaryEntryResponse.from_orm(result['entry'])
        search_results.append({
            **entry_data.dict(),
            "similarity_score": result['similarity_score'],
            "match_type": result['match_type']
        })
    
    return DictionarySearchResponse(
        results=search_results,
        total_found=len(results),
        query=query,
        exact_match=exact_match
    )

@router.post("/{entry_id}/approve")
async def approve_dictionary_entry(
    entry_id: int,
    approval_data: DictionaryApprovalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Approve dictionary entry"""
    
    query = select(DictionaryEntry).where(
        DictionaryEntry.id == entry_id,
        DictionaryEntry.is_deleted == False
    )
    result = await db.execute(query)
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Dictionary entry not found")
    
    if approval_data.approved:
        entry.approve(current_user)
    else:
        entry.reject(current_user, approval_data.reason or "No reason provided")
    
    await db.commit()
    
    action = "approved" if approval_data.approved else "rejected"
    logger.info(f"{action.title()} dictionary entry: {entry.term} (ID: {entry.id})")
    
    return {"message": f"Dictionary entry {action} successfully"}

@router.get("/stats", response_model=DictionaryStatsResponse)
async def get_dictionary_stats(
    project_id: int = Query(..., description="Project ID"),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Get dictionary statistics"""
    
    # Total entries
    total_query = select(func.count(DictionaryEntry.id)).where(
        DictionaryEntry.project_id == project_id,
        DictionaryEntry.is_deleted == False
    )
    total_result = await db.execute(total_query)
    total_entries = total_result.scalar()
    
    # By status
    status_counts = {}
    for status in DictionaryStatus:
        status_query = select(func.count(DictionaryEntry.id)).where(
            DictionaryEntry.project_id == project_id,
            DictionaryEntry.status == status,
            DictionaryEntry.is_deleted == False
        )
        status_result = await db.execute(status_query)
        status_counts[status.value] = status_result.scalar()
    
    # By category
    category_counts = {}
    for category in DictionaryCategory:
        category_query = select(func.count(DictionaryEntry.id)).where(
            DictionaryEntry.project_id == project_id,
            DictionaryEntry.category == category,
            DictionaryEntry.is_deleted == False
        )
        category_result = await db.execute(category_query)
        category_counts[category.value] = category_result.scalar()
    
    # Auto-generated vs manual
    auto_generated_query = select(func.count(DictionaryEntry.id)).where(
        DictionaryEntry.project_id == project_id,
        DictionaryEntry.is_auto_generated == True,
        DictionaryEntry.is_deleted == False
    )
    auto_generated_result = await db.execute(auto_generated_query)
    auto_generated_count = auto_generated_result.scalar()
    
    return DictionaryStatsResponse(
        project_id=project_id,
        total_entries=total_entries,
        status_counts=status_counts,
        category_counts=category_counts,
        auto_generated_count=auto_generated_count,
        manual_count=total_entries - auto_generated_count
    )