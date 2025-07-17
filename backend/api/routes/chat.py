# backend/api/routes/chat.py
"""
Natural Language Query (Chat) API routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from typing import List, Optional, Dict, Any
import logging
import time
import uuid
import json

from backend.database import get_db
from backend.models import (
    NLQSession, EntityExtraction, EntityMapping, NLQFeedback,
    Project, Table, Column, DictionaryEntry, Source,
    SessionStatus, EntityType, MappingConfidence, FeedbackType
)
from backend.schemas.chat import (
    ChatRequest, ChatResponse, EntityExtractionResponse,
    EntityMappingResponse, SQLGenerationResponse, ChatSessionResponse,
    FeedbackRequest, SessionListResponse
)
from backend.api.deps import get_current_user, get_project_access
from backend.services.llm_service import llm_service
from backend.services.entity_service import entity_service
from backend.services.sql_service import sql_service
from config import Config

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/query", response_model=ChatResponse)
async def process_natural_language_query(
    chat_request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Process a complete natural language query pipeline"""
    
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Create NLQ session
        session = NLQSession(
            project_id=chat_request.project_id,
            session_id=session_id,
            query_text=chat_request.query,
            status=SessionStatus.STARTED
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        # Step 1: Entity Extraction
        extraction_start = time.time()
        entities_result = await _extract_entities(chat_request, session, db)
        session.add_step_time("entity_extraction", time.time() - extraction_start)
        
        if not entities_result.get("entities"):
            session.set_status(SessionStatus.ERROR, "No entities extracted from query")
            await db.commit()
            
            return ChatResponse(
                session_id=session_id,
                query=chat_request.query,
                status="error",
                error_message="Could not extract entities from the query",
                entities=entities_result,
                processing_time=time.time() - start_time
            )
        
        # Step 2: Entity Mapping
        mapping_start = time.time()
        mappings_result = await _map_entities(entities_result, session, db)
        session.add_step_time("entity_mapping", time.time() - mapping_start)
        
        if not mappings_result.get("mappings"):
            session.set_status(SessionStatus.ERROR, "No suitable entity mappings found")
            await db.commit()
            
            return ChatResponse(
                session_id=session_id,
                query=chat_request.query,
                status="error",
                error_message="Could not map entities to database objects",
                entities=entities_result,
                mappings=mappings_result,
                processing_time=time.time() - start_time
            )
        
        # Step 3: SQL Generation
        sql_start = time.time()
        sql_result = await _generate_sql(session, mappings_result, db)
        session.add_step_time("sql_generation", time.time() - sql_start)
        
        if not sql_result.get("sql") or not sql_result.get("is_valid"):
            session.set_status(SessionStatus.ERROR, "Failed to generate valid SQL")
            await db.commit()
            
            return ChatResponse(
                session_id=session_id,
                query=chat_request.query,
                status="error",
                error_message="Could not generate valid SQL query",
                entities=entities_result,
                mappings=mappings_result,
                sql_generation=sql_result,
                processing_time=time.time() - start_time
            )
        
        # Step 4: SQL Execution (if auto_execute is True)
        execution_result = None
        if chat_request.auto_execute:
            exec_start = time.time()
            execution_result = await _execute_sql(session, sql_result, db)
            session.add_step_time("sql_execution", time.time() - exec_start)
        
        # Complete session
        total_time = time.time() - start_time
        session.complete_session()
        session.total_time = total_time
        await db.commit()
        
        response_status = "completed" if chat_request.auto_execute else "sql_ready"
        
        return ChatResponse(
            session_id=session_id,
            query=chat_request.query,
            status=response_status,
            entities=entities_result,
            mappings=mappings_result,
            sql_generation=sql_result,
            execution_result=execution_result,
            processing_time=total_time,
            step_times=session.step_times
        )
        
    except Exception as e:
        logger.error(f"NLQ processing failed: {e}")
        
        # Update session status
        if 'session' in locals():
            session.set_status(SessionStatus.ERROR, str(e))
            await db.commit()
        
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities_only(
    chat_request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """Extract entities from natural language query only"""
    
    session_id = str(uuid.uuid4())
    
    # Create temporary session
    session = NLQSession(
        project_id=chat_request.project_id,
        session_id=session_id,
        query_text=chat_request.query,
        status=SessionStatus.ENTITY_EXTRACTION
    )
    
    db.add(session)
    await db.commit()
    
    try:
        result = await _extract_entities(chat_request, session, db)
        
        return EntityExtractionResponse(
            session_id=session_id,
            query=chat_request.query,
            entities=result.get("entities", []),
            confidence=result.get("confidence", 0.0),
            processing_time=result.get("response_time", 0.0),
            model_used=result.get("model_used", "")
        )
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/map-entities", response_model=EntityMappingResponse)
async def map_entities_to_schema(
    session_id: str = Query(..., description="Session ID"),
    entities: List[Dict[str, Any]] = [],
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Map extracted entities to database schema objects"""
    
    # Get session
    session_query = select(NLQSession).where(NLQSession.session_id == session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Use provided entities or session entities
        entities_data = {"entities": entities} if entities else {"entities": session.extracted_entities or []}
        
        result = await _map_entities(entities_data, session, db)
        
        return EntityMappingResponse(
            session_id=session_id,
            mappings=result.get("mappings", []),
            confidence_scores=result.get("confidence_scores", {}),
            unmapped_entities=result.get("unmapped_entities", []),
            processing_time=result.get("processing_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Entity mapping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-sql", response_model=SQLGenerationResponse)
async def generate_sql_query(
    session_id: str = Query(..., description="Session ID"),
    mappings: Optional[Dict[str, Any]] = None,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Generate SQL query from entity mappings"""
    
    # Get session
    session_query = select(NLQSession).where(NLQSession.session_id == session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Use provided mappings or session mappings
        mappings_data = mappings or session.entity_mappings or {}
        
        result = await _generate_sql(session, mappings_data, db)
        
        return SQLGenerationResponse(
            session_id=session_id,
            sql=result.get("sql", ""),
            rationale=result.get("rationale", ""),
            confidence=result.get("confidence", 0.0),
            tables_used=result.get("tables_used", []),
            assumptions=result.get("assumptions", []),
            is_valid=result.get("is_valid", False),
            processing_time=result.get("response_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute-sql")
async def execute_sql_query(
    session_id: str = Query(..., description="Session ID"),
    sql_query: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Execute generated SQL query"""
    
    # Get session
    session_query = select(NLQSession).where(NLQSession.session_id == session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Use provided SQL or session SQL
        sql_to_execute = sql_query or session.generated_sql
        
        if not sql_to_execute:
            raise HTTPException(status_code=400, detail="No SQL query to execute")
        
        sql_data = {"sql": sql_to_execute, "is_valid": True}
        result = await _execute_sql(session, sql_data, db)
        
        return result
        
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(
    feedback_request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Submit feedback for NLQ session"""
    
    # Get session
    session_query = select(NLQSession).where(NLQSession.session_id == feedback_request.session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Create feedback record
        feedback = NLQFeedback(
            session_id=feedback_request.session_id,
            project_id=session.project_id,
            feedback_type=feedback_request.feedback_type,
            rating=feedback_request.rating,
            comment=feedback_request.comment,
            feedback_step=feedback_request.step,
            nlq_text=session.query_text,
            sql_text=session.generated_sql,
            suggested_entities=feedback_request.suggested_entities,
            suggested_mappings=feedback_request.suggested_mappings,
            suggested_sql=feedback_request.suggested_sql
        )
        
        db.add(feedback)
        
        # Update session user feedback
        if not session.user_feedback:
            session.user_feedback = {}
        
        session.user_feedback[feedback_request.step] = {
            "rating": feedback_request.rating,
            "comment": feedback_request.comment,
            "feedback_type": feedback_request.feedback_type.value,
            "timestamp": time.time()
        }
        
        await db.commit()
        
        logger.info(f"Feedback submitted for session {feedback_request.session_id}")
        
        return {"message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=SessionListResponse)
async def list_chat_sessions(
    project_id: int = Query(..., description="Project ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[SessionStatus] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user),
    project: Project = Depends(get_project_access)
):
    """List chat sessions for a project"""
    
    # Build query
    query = select(NLQSession).where(NLQSession.project_id == project_id)
    
    if status:
        query = query.where(NLQSession.status == status)
    
    # Get total count
    count_query = select(func.count(NLQSession.id)).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.order_by(NLQSession.created_at.desc()).offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    sessions = result.scalars().all()
    
    session_responses = []
    for session in sessions:
        session_responses.append(ChatSessionResponse(
            session_id=session.session_id,
            query=session.query_text,
            status=session.status.value,
            created_at=session.created_at,
            total_time=session.total_time,
            has_result=bool(session.sql_result),
            has_feedback=bool(session.user_feedback)
        ))
    
    return SessionListResponse(
        sessions=session_responses,
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/sessions/{session_id}", response_model=ChatResponse)
async def get_chat_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get detailed chat session information"""
    
    session_query = select(NLQSession).where(NLQSession.session_id == session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check project access
    project_query = select(Project).where(Project.id == session.project_id)
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()
    
    if not project or project.owner != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return ChatResponse(
        session_id=session.session_id,
        query=session.query_text,
        status=session.status.value,
        entities={"entities": session.extracted_entities or []},
        mappings=session.entity_mappings or {},
        sql_generation={
            "sql": session.generated_sql,
            "is_valid": bool(session.generated_sql)
        },
        execution_result=session.sql_result,
        processing_time=session.total_time,
        step_times=session.step_times,
        user_feedback=session.user_feedback,
        error_message=session.error_message
    )

# Helper functions

async def _extract_entities(
    chat_request: ChatRequest,
    session: NLQSession,
    db: AsyncSession
) -> Dict[str, Any]:
    """Extract entities from natural language query"""
    
    try:
        # Get context for entity extraction
        context = await _build_extraction_context(chat_request.project_id, db)
        
        # Extract entities using LLM
        result = await llm_service.extract_entities(chat_request.query, context)
        
        # Save entities to session
        session.set_entities(result.get("entities", []))
        await db.commit()
        
        return result
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {"entities": [], "error": str(e)}

async def _map_entities(
    entities_result: Dict[str, Any],
    session: NLQSession,
    db: AsyncSession
) -> Dict[str, Any]:
    """Map extracted entities to database objects"""
    
    try:
        # Use entity service for mapping
        from backend.services.entity_service import entity_service
        
        entities = entities_result.get("entities", [])
        mappings_result = await entity_service.map_entities_to_schema(
            entities,
            session.project_id,
            db
        )
        
        # Save mappings to session
        session.set_mappings(mappings_result)
        await db.commit()
        
        return mappings_result
        
    except Exception as e:
        logger.error(f"Entity mapping failed: {e}")
        return {"mappings": [], "error": str(e)}

async def _generate_sql(
    session: NLQSession,
    mappings_result: Dict[str, Any],
    db: AsyncSession
) -> Dict[str, Any]:
    """Generate SQL query from entity mappings"""
    
    try:
        # Get schemas for mapped tables
        schemas = await _build_schemas_from_mappings(mappings_result, db)
        
        # Generate SQL using LLM
        result = await llm_service.generate_sql(
            query=session.query_text,
            entities=session.extracted_entities or [],
            schemas=schemas
        )
        
        # Save SQL to session
        if result.get("sql"):
            tables_used = result.get("tables_used", [])
            session.set_sql(result["sql"], tables_used)
            await db.commit()
        
        return result
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return {"sql": "", "error": str(e), "is_valid": False}

async def _execute_sql(
    session: NLQSession,
    sql_result: Dict[str, Any],
    db: AsyncSession
) -> Dict[str, Any]:
    """Execute SQL query"""
    
    try:
        # Use SQL service for execution
        from backend.services.sql_service import sql_service
        
        execution_result = await sql_service.execute_query(
            sql_result["sql"],
            session.project_id,
            db
        )
        
        # Save result to session
        if execution_result:
            session.set_result(execution_result, execution_result.get("execution_time", 0.0))
            await db.commit()
        
        return execution_result
        
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return {"error": str(e), "success": False}

async def _build_extraction_context(
    project_id: int,
    db: AsyncSession
) -> Dict[str, Any]:
    """Build context for entity extraction"""
    
    # Get table names
    tables_query = select(Table.name).join(Source).where(Source.project_id == project_id)
    tables_result = await db.execute(tables_query)
    tables = [row[0] for row in tables_result.fetchall()]
    
    # Get column names
    columns_query = select(Column.name).join(Table).join(Source).where(
        Source.project_id == project_id
    ).limit(100)  # Limit to avoid token overflow
    columns_result = await db.execute(columns_query)
    columns = [row[0] for row in columns_result.fetchall()]
    
    # Get dictionary terms
    dict_query = select(DictionaryEntry.term).where(
        DictionaryEntry.project_id == project_id,
        DictionaryEntry.is_deleted == False
    ).limit(50)
    dict_result = await db.execute(dict_query)
    dictionary_terms = [row[0] for row in dict_result.fetchall()]
    
    return {
        "tables": tables,
        "columns": columns,
        "dictionary_terms": dictionary_terms
    }

async def _build_schemas_from_mappings(
    mappings_result: Dict[str, Any],
    db: AsyncSession
) -> List[Dict[str, Any]]:
    """Build schema information from entity mappings"""
    
    schemas = []
    table_ids = set()
    
    # Extract table IDs from mappings
    for mapping in mappings_result.get("mappings", []):
        if mapping.get("target_type") == "table":
            table_ids.add(mapping.get("target_id"))
        elif mapping.get("target_type") == "column":
            # Get table ID from column
            column_query = select(Column.table_id).where(Column.id == mapping.get("target_id"))
            column_result = await db.execute(column_query)
            table_id = column_result.scalar_one_or_none()
            if table_id:
                table_ids.add(table_id)
    
    # Build schemas for each table
    for table_id in table_ids:
        table_query = select(Table).where(Table.id == table_id)
        table_result = await db.execute(table_query)
        table = table_result.scalar_one_or_none()
        
        if not table:
            continue
        
        # Get columns for this table
        columns_query = select(Column).where(Column.table_id == table_id)
        columns_result = await db.execute(columns_query)
        columns = columns_result.scalars().all()
        
        schema = {
            "table_name": table.get_qualified_name(),
            "table_id": table.id,
            "description": table.description,
            "columns": []
        }
        
        for column in columns:
            column_info = {
                "name": column.name,
                "data_type": column.data_type,
                "is_nullable": column.is_nullable,
                "is_primary_key": column.is_primary_key,
                "description": column.description
            }
            schema["columns"].append(column_info)
        
        schemas.append(schema)
    
    return schemas