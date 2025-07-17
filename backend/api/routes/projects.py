# backend/api/routes/projects.py
"""
Project management API routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from typing import List, Optional, Dict, Any
import logging

from backend.database import get_db
from backend.models import Project, ProjectStatus
from backend.schemas.project import (
    ProjectCreate, ProjectUpdate, ProjectResponse, ProjectListResponse,
    ProjectStatsResponse, ProjectCloneRequest
)
from backend.api.deps import get_current_user
from backend.utils.helpers import paginate_query

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
    status: Optional[ProjectStatus] = Query(None),
    owner: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """List projects with filtering and pagination"""
    
    # Build query
    query = select(Project).where(Project.is_deleted == False)
    
    # Apply filters
    if search:
        query = query.where(
            or_(
                Project.name.ilike(f"%{search}%"),
                Project.description.ilike(f"%{search}%")
            )
        )
    
    if status:
        query = query.where(Project.status == status)
    
    if owner:
        query = query.where(Project.owner == owner)
    
    # Get total count
    count_query = select(func.count(Project.id)).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.order_by(Project.created_at.desc()).offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    projects = result.scalars().all()
    
    return ProjectListResponse(
        projects=[ProjectResponse.from_orm(p) for p in projects],
        total=total,
        skip=skip,
        limit=limit
    )

@router.post("/", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a new project"""
    
    # Check if project name already exists for this user
    existing_query = select(Project).where(
        Project.name == project.name,
        Project.owner == current_user,
        Project.is_deleted == False
    )
    existing_result = await db.execute(existing_query)
    existing_project = existing_result.scalar_one_or_none()
    
    if existing_project:
        raise HTTPException(
            status_code=400,
            detail=f"Project with name '{project.name}' already exists"
        )
    
    # Create project
    db_project = Project(
        name=project.name,
        description=project.description,
        owner=current_user,
        settings=project.settings or {},
        created_by=current_user
    )
    
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)
    
    logger.info(f"Created project: {db_project.name} (ID: {db_project.id})")
    return ProjectResponse.from_orm(db_project)

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get project by ID"""
    
    query = select(Project).where(
        Project.id == project_id,
        Project.is_deleted == False
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check permissions (owner or collaborator)
    if project.owner != current_user:
        # In a real app, check if user is a collaborator
        raise HTTPException(status_code=403, detail="Access denied")
    
    return ProjectResponse.from_orm(project)

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update project"""
    
    query = select(Project).where(
        Project.id == project_id,
        Project.is_deleted == False
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update fields
    update_data = project_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    
    project.updated_by = current_user
    
    await db.commit()
    await db.refresh(project)
    
    logger.info(f"Updated project: {project.name} (ID: {project.id})")
    return ProjectResponse.from_orm(project)

@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete project (soft delete)"""
    
    query = select(Project).where(
        Project.id == project_id,
        Project.is_deleted == False
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Soft delete
    project.soft_delete(current_user)
    
    await db.commit()
    
    logger.info(f"Deleted project: {project.name} (ID: {project.id})")
    return {"message": "Project deleted successfully"}

@router.post("/{project_id}/clone", response_model=ProjectResponse)
async def clone_project(
    project_id: int,
    clone_request: ProjectCloneRequest,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Clone project"""
    
    # Get source project
    query = select(Project).where(
        Project.id == project_id,
        Project.is_deleted == False
    )
    result = await db.execute(query)
    source_project = result.scalar_one_or_none()
    
    if not source_project:
        raise HTTPException(status_code=404, detail="Source project not found")
    
    if source_project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if new name already exists
    existing_query = select(Project).where(
        Project.name == clone_request.name,
        Project.owner == current_user,
        Project.is_deleted == False
    )
    existing_result = await db.execute(existing_query)
    existing_project = existing_result.scalar_one_or_none()
    
    if existing_project:
        raise HTTPException(
            status_code=400,
            detail=f"Project with name '{clone_request.name}' already exists"
        )
    
    # Clone project
    cloned_project = source_project.clone(
        new_name=clone_request.name,
        new_owner=current_user,
        user_id=current_user
    )
    
    if clone_request.description:
        cloned_project.description = clone_request.description
    
    db.add(cloned_project)
    await db.commit()
    await db.refresh(cloned_project)
    
    logger.info(f"Cloned project: {cloned_project.name} (ID: {cloned_project.id})")
    return ProjectResponse.from_orm(cloned_project)

@router.post("/{project_id}/archive")
async def archive_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Archive project"""
    
    query = select(Project).where(
        Project.id == project_id,
        Project.is_deleted == False
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    project.archive(current_user)
    
    await db.commit()
    
    logger.info(f"Archived project: {project.name} (ID: {project.id})")
    return {"message": "Project archived successfully"}

@router.post("/{project_id}/activate")
async def activate_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Activate project"""
    
    query = select(Project).where(
        Project.id == project_id,
        Project.is_deleted == False
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    project.activate(current_user)
    
    await db.commit()
    
    logger.info(f"Activated project: {project.name} (ID: {project.id})")
    return {"message": "Project activated successfully"}

@router.get("/{project_id}/stats", response_model=ProjectStatsResponse)
async def get_project_stats(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get project statistics"""
    
    query = select(Project).where(
        Project.id == project_id,
        Project.is_deleted == False
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get statistics
    from backend.models import Source, Table, Column, DictionaryEntry, Index, NLQSession
    
    # Sources count
    sources_query = select(func.count(Source.id)).where(Source.project_id == project_id)
    sources_result = await db.execute(sources_query)
    sources_count = sources_result.scalar()
    
    # Tables count
    tables_query = select(func.count(Table.id)).join(Source).where(Source.project_id == project_id)
    tables_result = await db.execute(tables_query)
    tables_count = tables_result.scalar()
    
    # Columns count
    columns_query = select(func.count(Column.id)).join(Table).join(Source).where(Source.project_id == project_id)
    columns_result = await db.execute(columns_query)
    columns_count = columns_result.scalar()
    
    # Dictionary entries count
    dict_query = select(func.count(DictionaryEntry.id)).where(DictionaryEntry.project_id == project_id)
    dict_result = await db.execute(dict_query)
    dict_count = dict_result.scalar()
    
    # Indexes count
    indexes_query = select(func.count(Index.id)).where(Index.project_id == project_id)
    indexes_result = await db.execute(indexes_query)
    indexes_count = indexes_result.scalar()
    
    # NLQ sessions count
    nlq_query = select(func.count(NLQSession.id)).where(NLQSession.project_id == project_id)
    nlq_result = await db.execute(nlq_query)
    nlq_count = nlq_result.scalar()
    
    return ProjectStatsResponse(
        project_id=project_id,
        sources_count=sources_count,
        tables_count=tables_count,
        columns_count=columns_count,
        dictionary_entries_count=dict_count,
        indexes_count=indexes_count,
        nlq_sessions_count=nlq_count
    )

@router.put("/{project_id}/settings")
async def update_project_settings(
    project_id: int,
    settings: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update project settings"""
    
    query = select(Project).where(
        Project.id == project_id,
        Project.is_deleted == False
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update settings
    current_settings = project.settings or {}
    current_settings.update(settings)
    project.settings = current_settings
    project.updated_by = current_user
    
    await db.commit()
    
    logger.info(f"Updated settings for project: {project.name} (ID: {project.id})")
    return {"message": "Settings updated successfully", "settings": project.settings}
