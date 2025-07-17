# backend/api/deps.py
"""
API dependencies for authentication and authorization
"""

from fastapi import Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import logging

from backend.database import get_db
from backend.models import Project
from config import Config

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Get current user from token
    For demo purposes, this is simplified. In production, implement proper JWT validation.
    """
    
    # For development/demo, we'll use a simple approach
    # In production, validate JWT token here
    
    if not credentials:
        # For demo, allow requests without auth
        return "demo_user"
    
    try:
        # In production, decode and validate JWT token
        # For now, just return the token as username
        token = credentials.credentials
        
        # Simple validation - check if token exists
        if not token or len(token) < 3:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # For demo, just return token as user identifier
        return token
        
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_project_access(
    project_id: int = Query(..., description="Project ID"),
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Project:
    """
    Verify user has access to the project and return project
    """
    
    try:
        # Get project
        query = select(Project).where(
            Project.id == project_id,
            Project.is_deleted == False
        )
        result = await db.execute(query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Check access (simplified for demo)
        # In production, implement proper RBAC
        if project.owner != current_user:
            # For demo, allow access to all projects
            # In production, check user permissions
            pass
        
        return project
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Project access check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify project access"
        )

async def get_admin_user(
    current_user: str = Depends(get_current_user)
) -> str:
    """
    Verify user has admin privileges
    """
    
    # For demo, all users are admin
    # In production, check user roles
    admin_users = ["admin", "demo_user"]
    
    if current_user not in admin_users:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user

class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.requests = {}
    
    def __call__(self, current_user: str = Depends(get_current_user)):
        import time
        
        now = time.time()
        user_requests = self.requests.get(current_user, [])
        
        # Remove old requests
        user_requests = [req_time for req_time in user_requests if now - req_time < self.period]
        
        # Check rate limit
        if len(user_requests) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.calls} calls per {self.period} seconds"
            )
        
        # Add current request
        user_requests.append(now)
        self.requests[current_user] = user_requests
        
        return current_user

# Rate limiter instances
rate_limit_strict = RateLimiter(calls=10, period=60)  # 10 calls per minute
rate_limit_moderate = RateLimiter(calls=100, period=3600)  # 100 calls per hour

def get_pagination_params(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return")
) -> tuple[int, int]:
    """Get pagination parameters"""
    return skip, limit

def get_search_params(
    search: Optional[str] = Query(None, min_length=1, max_length=100, description="Search query"),
    sort_by: Optional[str] = Query(None, description="Sort field"),
    sort_order: Optional[str] = Query("asc", regex="^(asc|desc)$", description="Sort order")
) -> dict:
    """Get search and sort parameters"""
    return {
        "search": search,
        "sort_by": sort_by,
        "sort_order": sort_order
    }