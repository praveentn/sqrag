# migrations/versions/001_initial_schema.py
"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create data_sources table
    op.create_table('data_sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('connection_string', sa.Text(), nullable=True),
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('last_refresh', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create tables table
    op.create_table('tables',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('schema_name', sa.String(length=255), nullable=True),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('sampling_stats', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['source_id'], ['data_sources.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create columns table
    op.create_table('columns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('data_type', sa.String(length=50), nullable=True),
        sa.Column('is_nullable', sa.Boolean(), nullable=True),
        sa.Column('is_primary_key', sa.Boolean(), nullable=True),
        sa.Column('is_foreign_key', sa.Boolean(), nullable=True),
        sa.Column('table_id', sa.Integer(), nullable=False),
        sa.Column('sample_values', sa.JSON(), nullable=True),
        sa.Column('unique_count', sa.Integer(), nullable=True),
        sa.Column('null_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['table_id'], ['tables.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create dictionary_entries table
    op.create_table('dictionary_entries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('term', sa.String(length=255), nullable=False),
        sa.Column('definition', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('synonyms', sa.JSON(), nullable=True),
        sa.Column('abbreviations', sa.JSON(), nullable=True),
        sa.Column('source_table', sa.String(length=255), nullable=True),
        sa.Column('source_column', sa.String(length=255), nullable=True),
        sa.Column('approved', sa.Boolean(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=True),
        sa.Column('approved_by', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('term')
    )
    
    # Create embedding_indexes table
    op.create_table('embedding_indexes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('scope', sa.String(length=50), nullable=False),
        sa.Column('backend', sa.String(length=50), nullable=False),
        sa.Column('model_name', sa.String(length=255), nullable=True),
        sa.Column('dimensions', sa.Integer(), nullable=True),
        sa.Column('index_path', sa.String(length=500), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('item_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create chat_sessions table
    op.create_table('chat_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create chat_messages table
    op.create_table('chat_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('parent_message_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['parent_message_id'], ['chat_messages.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create query_executions table
    op.create_table('query_executions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=True),
        sa.Column('message_id', sa.Integer(), nullable=True),
        sa.Column('sql_query', sa.Text(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('execution_time', sa.Float(), nullable=True),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('result_preview', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['message_id'], ['chat_messages.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ),
        sa.ForeignKeyConstraint(['source_id'], ['data_sources.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create system_logs table
    op.create_table('system_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('level', sa.String(length=20), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('correlation_id', sa.String(length=255), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for better performance
    op.create_index('idx_data_sources_type', 'data_sources', ['type'])
    op.create_index('idx_tables_source_id', 'tables', ['source_id'])
    op.create_index('idx_columns_table_id', 'columns', ['table_id'])
    op.create_index('idx_dictionary_category', 'dictionary_entries', ['category'])
    op.create_index('idx_dictionary_approved', 'dictionary_entries', ['approved'])
    op.create_index('idx_embedding_status', 'embedding_indexes', ['status'])
    op.create_index('idx_chat_messages_session', 'chat_messages', ['session_id'])
    op.create_index('idx_chat_messages_created', 'chat_messages', ['created_at'])
    op.create_index('idx_query_executions_session', 'query_executions', ['session_id'])
    op.create_index('idx_system_logs_level', 'system_logs', ['level'])
    op.create_index('idx_system_logs_category', 'system_logs', ['category'])
    op.create_index('idx_system_logs_created', 'system_logs', ['created_at'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('system_logs')
    op.drop_table('query_executions')
    op.drop_table('chat_messages')
    op.drop_table('chat_sessions')
    op.drop_table('embedding_indexes')
    op.drop_table('dictionary_entries')
    op.drop_table('columns')
    op.drop_table('tables')
    op.drop_table('data_sources')

