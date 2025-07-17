# backend/utils/helpers.py
"""
Utility functions and helpers
"""

import re
import hashlib
import secrets
import string
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

def clean_identifier(name: str) -> str:
    """Clean name to be database-safe identifier"""
    
    # Replace spaces and special characters with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    
    # Remove consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    # Ensure it starts with a letter
    if cleaned and cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    
    # Handle empty names
    if not cleaned:
        cleaned = "unnamed"
    
    return cleaned.lower()

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return secrets.token_urlsafe(32)

def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(64)

def hash_password(password: str) -> str:
    """Hash password using SHA-256 (for demo purposes)"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = 255 - len(ext) - 1 if ext else 255
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text"""
    if not text:
        return []
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those'
    }
    
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    # Remove duplicates and limit
    return list(dict.fromkeys(keywords))[:max_keywords]

def parse_connection_string(connection_string: str) -> Dict[str, str]:
    """Parse database connection string"""
    
    # Simple regex-based parsing for common formats
    # postgresql://user:password@host:port/database
    # mysql://user:password@host:port/database
    
    pattern = r'(\w+)://(?:([^:]+):([^@]+)@)?([^:/]+)(?::(\d+))?/(.+)'
    match = re.match(pattern, connection_string)
    
    if not match:
        return {}
    
    return {
        'driver': match.group(1),
        'username': match.group(2) or '',
        'password': match.group(3) or '',
        'host': match.group(4) or 'localhost',
        'port': int(match.group(5)) if match.group(5) else None,
        'database': match.group(6) or ''
    }

def build_connection_string(
    driver: str,
    host: str,
    database: str,
    username: str = None,
    password: str = None,
    port: int = None,
    **kwargs
) -> str:
    """Build database connection string"""
    
    connection_parts = [f"{driver}://"]
    
    if username:
        if password:
            connection_parts.append(f"{username}:{password}@")
        else:
            connection_parts.append(f"{username}@")
    
    connection_parts.append(host)
    
    if port:
        connection_parts.append(f":{port}")
    
    connection_parts.append(f"/{database}")
    
    # Add additional parameters
    if kwargs:
        params = "&".join([f"{k}={v}" for k, v in kwargs.items()])
        connection_parts.append(f"?{params}")
    
    return "".join(connection_parts)

async def paginate_query(
    query,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = None
) -> Tuple[List[Any], int]:
    """Paginate a SQLAlchemy query"""
    
    if db is None:
        raise ValueError("Database session is required")
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    paginated_query = query.offset(skip).limit(limit)
    result = await db.execute(paginated_query)
    items = result.scalars().all()
    
    return items, total

def validate_email(email: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def mask_sensitive_data(data: str, mask_char: str = '*', visible_chars: int = 4) -> str:
    """Mask sensitive data showing only last few characters"""
    if not data or len(data) <= visible_chars:
        return mask_char * len(data) if data else ''
    
    masked_length = len(data) - visible_chars
    return mask_char * masked_length + data[-visible_chars:]

def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding"""
    try:
        import chardet
        
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    except Exception:
        return 'utf-8'

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string"""
    try:
        import json
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = '{}') -> str:
    """Safely dump object to JSON string"""
    try:
        import json
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return default

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using Levenshtein distance"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace common separators with spaces
    text = re.sub(r'[_\-\.]+', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def generate_slug(text: str, max_length: int = 50) -> str:
    """Generate URL-friendly slug from text"""
    if not text:
        return ''
    
    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s_-]+', '-', slug)
    slug = slug.strip('-')
    
    # Truncate if too long
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    
    return slug

def parse_query_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate query filters"""
    
    parsed_filters = {}
    
    for key, value in filters.items():
        if value is None or value == '':
            continue
        
        # Handle different filter types
        if key.endswith('_min') or key.endswith('_max'):
            try:
                parsed_filters[key] = float(value)
            except (ValueError, TypeError):
                continue
        elif key.endswith('_date'):
            try:
                if isinstance(value, str):
                    parsed_filters[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                else:
                    parsed_filters[key] = value
            except ValueError:
                continue
        elif key.endswith('_list'):
            if isinstance(value, str):
                parsed_filters[key] = [item.strip() for item in value.split(',') if item.strip()]
            elif isinstance(value, list):
                parsed_filters[key] = value
        else:
            parsed_filters[key] = value
    
    return parsed_filters

def get_current_timestamp() -> datetime:
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc)

def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """Format datetime to string"""
    if not dt:
        return ''
    
    # Convert to UTC if timezone-aware
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    
    return dt.strftime(format_str)