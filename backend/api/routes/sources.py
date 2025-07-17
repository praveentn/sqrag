# backend/api/routes/sources.py
"""
Data sources management API routes
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from typing import List, Optional, Dict, Any
import logging
import os
from pathlib import Path
import aiofiles

from backend.database import get_db
from backend.models import Source, Table, Column, Project, SourceType, SourceStatus
from backend.schemas.source import (
    SourceCreate, SourceUpdate, SourceResponse, SourceListResponse,
    DatabaseSourceCreate, FileUploadResponse, TableResponse, ColumnResponse,
    TableListResponse, ColumnListResponse, TableUpdate, ColumnUpdate,
    ConnectionTestResponse, SchemaIntrospectionResponse, DataPreviewResponse,
    IngestStatusResponse
)
from backend.api.deps import get_current_user, get_project_access
from backend.services.file_service import FileIngestService
from backend.services.database_service import DatabaseService
from config import Config

logger = logging.getLogger(__name__)
router = APIRouter()

# File service instance
file_service = FileIngestService()
db_service = DatabaseService()

@router.get("/", response_model=SourceListResponse)
async def list_sources(
    project_id: int = Query(..., description="Project ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
    source_type: Optional[SourceType] = Query(None),
    status: Optional[SourceStatus] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """List sources for a project"""
    
    # Build query
    query = select(Source).where(
        Source.project_id == project_id,
        Source.is_deleted == False
    )
    
    # Apply filters
    if search:
        query = query.where(
            or_(
                Source.name.ilike(f"%{search}%"),
                Source.file_type.ilike(f"%{search}%")
            )
        )
    
    if source_type:
        query = query.where(Source.type == source_type)
    
    if status:
        query = query.where(Source.status == status)
    
    # Get total count
    count_query = select(func.count(Source.id)).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.order_by(Source.created_at.desc()).offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    sources = result.scalars().all()
    
    return SourceListResponse(
        sources=[SourceResponse.from_orm(s) for s in sources],
        total=total,
        skip=skip,
        limit=limit
    )

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    project_id: int = Query(..., description="Project ID"),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Upload file and create source"""
    
    # Validate file type
    allowed_extensions = Config.SECURITY_CONFIG['allowed_file_extensions']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not allowed. Allowed types: {allowed_extensions}"
        )
    
    # Validate file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > Config.SECURITY_CONFIG['max_file_size']:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds maximum allowed size"
        )
    
    # Create upload directory
    upload_dir = Path(Config.UPLOAD_FOLDER) / str(project_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = upload_dir / file.filename
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    # Create source record
    source = Source(
        project_id=project_id,
        name=file.filename,
        type=SourceType.FILE_UPLOAD,
        status=SourceStatus.PENDING,
        file_path=str(file_path),
        file_size=file_size,
        file_type=file_extension,
        created_by=current_user
    )
    
    db.add(source)
    await db.commit()
    await db.refresh(source)
    
    # Start background ingestion
    background_tasks.add_task(
        file_service.ingest_file_async,
        source.id,
        str(file_path),
        db
    )
    
    logger.info(f"Uploaded file: {file.filename} for project {project_id}")
    
    return FileUploadResponse(
        filename=file.filename,
        file_size=file_size,
        file_type=file_extension,
        upload_path=str(file_path),
        tables_detected=[]  # Will be populated after ingestion
    )

@router.post("/database", response_model=SourceResponse)
async def add_database_source(
    source_data: DatabaseSourceCreate,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Add database source"""
    
    # Test connection first
    connection_success = await db_service.test_connection(
        db_type=source_data.db_type,
        host=source_data.host,
        port=source_data.port,
        database=source_data.database,
        username=source_data.username,
        password=source_data.password,
        schema_name=source_data.schema_name
    )
    
    if not connection_success['success']:
        raise HTTPException(
            status_code=400,
            detail=f"Database connection failed: {connection_success['message']}"
        )
    
    # Create connection URI
    connection_uri = db_service.build_connection_uri(
        db_type=source_data.db_type,
        host=source_data.host,
        port=source_data.port,
        database=source_data.database,
        username=source_data.username,
        password=source_data.password
    )
    
    # Create source record
    source = Source(
        project_id=source_data.project_id,
        name=source_data.name,
        type=SourceType.DATABASE,
        status=SourceStatus.CONNECTING,
        connection_uri=connection_uri,
        connection_params={
            "db_type": source_data.db_type,
            "host": source_data.host,
            "port": source_data.port,
            "database": source_data.database,
            "username": source_data.username,
            "schema_name": source_data.schema_name,
            "ssl_mode": source_data.ssl_mode
        },
        created_by=current_user
    )
    
    db.add(source)
    await db.commit()
    await db.refresh(source)
    
    # Start background schema introspection
    background_tasks.add_task(
        db_service.introspect_schema_async,
        source.id,
        db
    )
    
    logger.info(f"Added database source: {source_data.name} for project {source_data.project_id}")
    
    return SourceResponse.from_orm(source)

@router.post("/test-connection", response_model=ConnectionTestResponse)
async def test_database_connection(
    source_data: DatabaseSourceCreate,
    current_user: str = Depends(get_current_user)
):
    """Test database connection"""
    
    result = await db_service.test_connection(
        db_type=source_data.db_type,
        host=source_data.host,
        port=source_data.port,
        database=source_data.database,
        username=source_data.username,
        password=source_data.password,
        schema_name=source_data.schema_name
    )
    
    return ConnectionTestResponse(**result)

@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get source by ID"""
    
    query = select(Source).where(
        Source.id == source_id,
        Source.is_deleted == False
    )
    result = await db.execute(query)
    source = result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == source.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return SourceResponse.from_orm(source)

@router.put("/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: int,
    source_update: SourceUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update source"""
    
    query = select(Source).where(
        Source.id == source_id,
        Source.is_deleted == False
    )
    result = await db.execute(query)
    source = result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == source.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update fields
    update_data = source_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(source, field, value)
    
    source.updated_by = current_user
    
    await db.commit()
    await db.refresh(source)
    
    logger.info(f"Updated source: {source.name} (ID: {source.id})")
    return SourceResponse.from_orm(source)

@router.delete("/{source_id}")
async def delete_source(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete source"""
    
    query = select(Source).where(
        Source.id == source_id,
        Source.is_deleted == False
    )
    result = await db.execute(query)
    source = result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == source.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Soft delete
    source.soft_delete(current_user)
    
    await db.commit()
    
    logger.info(f"Deleted source: {source.name} (ID: {source.id})")
    return {"message": "Source deleted successfully"}

@router.get("/{source_id}/tables", response_model=TableListResponse)
async def list_tables(
    source_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """List tables for a source"""
    
    # Verify source access
    source_query = select(Source).where(Source.id == source_id)
    source_result = await db.execute(source_query)
    source = source_result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Build query
    query = select(Table).where(Table.source_id == source_id)
    
    if search:
        query = query.where(
            or_(
                Table.name.ilike(f"%{search}%"),
                Table.display_name.ilike(f"%{search}%"),
                Table.description.ilike(f"%{search}%")
            )
        )
    
    # Get total count
    count_query = select(func.count(Table.id)).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.order_by(Table.name).offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    tables = result.scalars().all()
    
    return TableListResponse(
        tables=[TableResponse.from_orm(t) for t in tables],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/{source_id}/tables/{table_id}/columns", response_model=ColumnListResponse)
async def list_columns(
    source_id: int,
    table_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """List columns for a table"""
    
    # Verify table access
    table_query = select(Table).where(
        Table.id == table_id,
        Table.source_id == source_id
    )
    table_result = await db.execute(table_query)
    table = table_result.scalar_one_or_none()
    
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    
    # Build query
    query = select(Column).where(Column.table_id == table_id)
    
    if search:
        query = query.where(
            or_(
                Column.name.ilike(f"%{search}%"),
                Column.display_name.ilike(f"%{search}%"),
                Column.description.ilike(f"%{search}%")
            )
        )
    
    # Get total count
    count_query = select(func.count(Column.id)).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.order_by(Column.name).offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    columns = result.scalars().all()
    
    return ColumnListResponse(
        columns=[ColumnResponse.from_orm(c) for c in columns],
        total=total,
        skip=skip,
        limit=limit
    )

@router.put("/tables/{table_id}", response_model=TableResponse)
async def update_table(
    table_id: int,
    table_update: TableUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update table metadata"""
    
    query = select(Table).where(Table.id == table_id)
    result = await db.execute(query)
    table = result.scalar_one_or_none()
    
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    
    # Update fields
    update_data = table_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(table, field, value)
    
    await db.commit()
    await db.refresh(table)
    
    return TableResponse.from_orm(table)

@router.put("/columns/{column_id}", response_model=ColumnResponse)
async def update_column(
    column_id: int,
    column_update: ColumnUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update column metadata"""
    
    query = select(Column).where(Column.id == column_id)
    result = await db.execute(query)
    column = result.scalar_one_or_none()
    
    if not column:
        raise HTTPException(status_code=404, detail="Column not found")
    
    # Update fields
    update_data = column_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(column, field, value)
    
    await db.commit()
    await db.refresh(column)
    
    return ColumnResponse.from_orm(column)

@router.get("/{source_id}/schema", response_model=SchemaIntrospectionResponse)
async def get_schema(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get source schema information"""
    
    source_query = select(Source).where(Source.id == source_id)
    source_result = await db.execute(source_query)
    source = source_result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Get tables and columns
    tables_query = select(Table).where(Table.source_id == source_id)
    tables_result = await db.execute(tables_query)
    tables = tables_result.scalars().all()
    
    schema_data = {
        "tables": [],
        "views": [],
        "total_tables": len(tables),
        "total_columns": 0,
        "schema_metadata": source.schema_metadata or {}
    }
    
    for table in tables:
        table_data = {
            "id": table.id,
            "name": table.name,
            "display_name": table.display_name,
            "description": table.description,
            "schema_name": table.schema_name,
            "table_type": table.table_type,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "columns": []
        }
        
        # Get columns for this table
        columns_query = select(Column).where(Column.table_id == table.id)
        columns_result = await db.execute(columns_query)
        columns = columns_result.scalars().all()
        
        for column in columns:
            table_data["columns"].append({
                "id": column.id,
                "name": column.name,
                "data_type": column.data_type,
                "is_nullable": column.is_nullable,
                "is_primary_key": column.is_primary_key,
                "description": column.description
            })
        
        schema_data["total_columns"] += len(columns)
        
        if table.table_type == "view":
            schema_data["views"].append(table_data)
        else:
            schema_data["tables"].append(table_data)
    
    return SchemaIntrospectionResponse(**schema_data)

@router.get("/{source_id}/tables/{table_id}/preview", response_model=DataPreviewResponse)
async def preview_table_data(
    source_id: int,
    table_id: int,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Preview table data"""
    
    # Verify table access
    table_query = select(Table).where(
        Table.id == table_id,
        Table.source_id == source_id
    )
    table_result = await db.execute(table_query)
    table = table_result.scalar_one_or_none()
    
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    
    # Get source
    source_query = select(Source).where(Source.id == source_id)
    source_result = await db.execute(source_query)
    source = source_result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Get preview data based on source type
    if source.type == SourceType.DATABASE:
        preview_data = await db_service.preview_table_data(
            source, table, limit
        )
    else:
        # For file sources, use file service
        preview_data = await file_service.preview_table_data(
            source, table, limit
        )
    
    return DataPreviewResponse(**preview_data)

@router.get("/{source_id}/ingest-status", response_model=IngestStatusResponse)
async def get_ingest_status(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get ingestion status"""
    
    query = select(Source).where(Source.id == source_id)
    result = await db.execute(query)
    source = result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Parse log messages
    log_messages = []
    if source.ingest_log:
        log_messages = source.ingest_log.split('\n')
    
    return IngestStatusResponse(
        source_id=source_id,
        status=source.ingest_status,
        progress=source.ingest_progress,
        log_messages=log_messages,
        tables_processed=len(source.tables) if source.tables else 0,
        tables_total=source.total_tables
    )