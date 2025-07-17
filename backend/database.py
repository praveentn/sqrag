# backend/database.py
"""
Database configuration and session management
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from config import Config

logger = logging.getLogger(__name__)

# Create async engine
# engine = create_async_engine(
#     Config.SQLALCHEMY_DATABASE_URI.replace('sqlite://', 'sqlite+aiosqlite://'),
#     echo=Config.DEBUG,
#     pool_pre_ping=True,
#     pool_recycle=3600,
#     **Config.SQLALCHEMY_ENGINE_OPTIONS
# )

engine = create_async_engine(
    Config.SQLALCHEMY_DATABASE_URI.replace("sqlite://", "sqlite+aiosqlite://"),
    echo=Config.DEBUG,
    **Config.SQLALCHEMY_ENGINE_OPTIONS,   # holds pool_pre_ping, pool_recycle, etc.
)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for all models
class Base(DeclarativeBase):
    """Base class for all database models"""
    _metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )

# Database dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()

# Context manager for database transactions
@asynccontextmanager
async def db_transaction():
    """Database transaction context manager"""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Transaction error: {e}")
            raise
        finally:
            await session.close()

# Database initialization
async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base._metadata.create_all)
    logger.info("✅ Database initialized successfully!")

# Database health check
async def check_db_health():
    """Check database connection health"""
    try:
        from sqlalchemy import text
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False