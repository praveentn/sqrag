# services/chat_service.py
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from models import db, ChatSession, ChatMessage
from config import Config

logger = logging.getLogger(__name__)

class ChatService:
    """Manages chat sessions and conversations"""
    
    def __init__(self):
        self.config = Config()
        
    def create_session(self, title: str = "New Chat", user_id: str = None, 
                      _metadata: Dict[str, Any] = None) -> ChatSession:
        """Create a new chat session"""
        try:
            session = ChatSession(
                title=title,
                user_id=user_id,
                _metadata=_metadata or {},
                created_at=datetime.utcnow()
            )
            
            db.session.add(session)
            db.session.commit()
            
            logger.info(f"Created chat session: {session.id}")
            return session
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating chat session: {str(e)}")
            raise
    
    def get_session(self, session_id: int) -> Optional[ChatSession]:
        """Get a chat session by ID"""
        try:
            return ChatSession.query.get(session_id)
        except Exception as e:
            logger.error(f"Error getting chat session {session_id}: {str(e)}")
            return None
    
    def get_sessions(self, user_id: str = None, limit: int = 50, 
                    offset: int = 0) -> List[ChatSession]:
        """Get chat sessions with optional filtering"""
        try:
            query = ChatSession.query
            
            if user_id:
                query = query.filter_by(user_id=user_id)
            
            sessions = query.order_by(ChatSession.updated_at.desc())\
                          .offset(offset).limit(limit).all()
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting chat sessions: {str(e)}")
            return []
    
    def update_session(self, session_id: int, title: str = None, 
                      _metadata: Dict[str, Any] = None) -> Optional[ChatSession]:
        """Update a chat session"""
        try:
            session = ChatSession.query.get(session_id)
            if not session:
                return None
            
            if title is not None:
                session.title = title
            
            if _metadata is not None:
                # Merge _metadata
                current__metadata = session._metadata or {}
                current__metadata.update(_metadata)
                session._metadata = current__metadata
            
            session.updated_at = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"Updated chat session: {session_id}")
            return session
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating chat session {session_id}: {str(e)}")
            return None
    
    def delete_session(self, session_id: int) -> bool:
        """Delete a chat session and all its messages"""
        try:
            session = ChatSession.query.get(session_id)
            if not session:
                return False
            
            # Cascade delete will handle messages
            db.session.delete(session)
            db.session.commit()
            
            logger.info(f"Deleted chat session: {session_id}")
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting chat session {session_id}: {str(e)}")
            return False
    
    def add_message(self, session_id: int, role: str, content: str,
                   _metadata: Dict[str, Any] = None, parent_message_id: int = None) -> ChatMessage:
        """Add a message to a chat session"""
        try:
            # Validate role
            if role not in ['user', 'assistant', 'system']:
                raise ValueError(f"Invalid role: {role}")
            
            # Check if session exists
            session = ChatSession.query.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                _metadata=_metadata or {},
                parent_message_id=parent_message_id,
                created_at=datetime.utcnow()
            )
            
            db.session.add(message)
            
            # Update session timestamp
            session.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            logger.info(f"Added {role} message to session {session_id}")
            return message
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding message to session {session_id}: {str(e)}")
            raise
    
    def get_messages(self, session_id: int, limit: int = 100, 
                    offset: int = 0, include__metadata: bool = True) -> List[ChatMessage]:
        """Get messages for a chat session"""
        try:
            query = ChatMessage.query.filter_by(session_id=session_id)
            
            messages = query.order_by(ChatMessage.created_at.asc())\
                          .offset(offset).limit(limit).all()
            
            # Optionally strip _metadata for lighter responses
            if not include__metadata:
                for message in messages:
                    message._metadata = {}
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {str(e)}")
            return []
    
    def get_message(self, message_id: int) -> Optional[ChatMessage]:
        """Get a specific message by ID"""
        try:
            return ChatMessage.query.get(message_id)
        except Exception as e:
            logger.error(f"Error getting message {message_id}: {str(e)}")
            return None
    
    def update_message(self, message_id: int, content: str = None,
                      _metadata: Dict[str, Any] = None) -> Optional[ChatMessage]:
        """Update a message"""
        try:
            message = ChatMessage.query.get(message_id)
            if not message:
                return None
            
            if content is not None:
                message.content = content
            
            if _metadata is not None:
                # Merge _metadata
                current__metadata = message._metadata or {}
                current__metadata.update(_metadata)
                message._metadata = current__metadata
            
            db.session.commit()
            
            logger.info(f"Updated message: {message_id}")
            return message
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating message {message_id}: {str(e)}")
            return None
    
    def delete_message(self, message_id: int) -> bool:
        """Delete a message"""
        try:
            message = ChatMessage.query.get(message_id)
            if not message:
                return False
            
            db.session.delete(message)
            db.session.commit()
            
            logger.info(f"Deleted message: {message_id}")
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting message {message_id}: {str(e)}")
            return False
    
    def search_messages(self, query: str, session_id: int = None, 
                       role: str = None, limit: int = 50) -> List[ChatMessage]:
        """Search messages by content"""
        try:
            search_query = ChatMessage.query.filter(
                ChatMessage.content.ilike(f'%{query}%')
            )
            
            if session_id:
                search_query = search_query.filter_by(session_id=session_id)
            
            if role:
                search_query = search_query.filter_by(role=role)
            
            messages = search_query.order_by(ChatMessage.created_at.desc())\
                                 .limit(limit).all()
            
            return messages
            
        except Exception as e:
            logger.error(f"Error searching messages: {str(e)}")
            return []
    
    def get_conversation_summary(self, session_id: int) -> Dict[str, Any]:
        """Get summary statistics for a conversation"""
        try:
            session = ChatSession.query.get(session_id)
            if not session:
                return {}
            
            messages = ChatMessage.query.filter_by(session_id=session_id).all()
            
            user_messages = [m for m in messages if m.role == 'user']
            assistant_messages = [m for m in messages if m.role == 'assistant']
            
            # Count SQL queries executed
            sql_queries = 0
            successful_queries = 0
            for msg in assistant_messages:
                _metadata = msg._metadata or {}
                if _metadata.get('sql_result', {}).get('sql'):
                    sql_queries += 1
                    if _metadata.get('execution_result', {}).get('status') == 'success':
                        successful_queries += 1
            
            return {
                'session_id': session_id,
                'title': session.title,
                'total_messages': len(messages),
                'user_messages': len(user_messages),
                'assistant_messages': len(assistant_messages),
                'sql_queries_generated': sql_queries,
                'successful_queries': successful_queries,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'duration_minutes': round((session.updated_at - session.created_at).total_seconds() / 60, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {str(e)}")
            return {}
    
    def export_conversation(self, session_id: int, format: str = 'json') -> Dict[str, Any]:
        """Export conversation in specified format"""
        try:
            session = ChatSession.query.get(session_id)
            if not session:
                return {'error': 'Session not found'}
            
            messages = self.get_messages(session_id, limit=1000)
            
            if format.lower() == 'json':
                return {
                    'session': session.to_dict(),
                    'messages': [message.to_dict() for message in messages],
                    'exported_at': datetime.utcnow().isoformat()
                }
            
            elif format.lower() == 'markdown':
                md_lines = [
                    f"# Chat Session: {session.title}",
                    f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"**Updated:** {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "---",
                    ""
                ]
                
                for message in messages:
                    role_emoji = "🧑" if message.role == 'user' else "🤖" if message.role == 'assistant' else "⚙️"
                    md_lines.append(f"## {role_emoji} {message.role.title()}")
                    md_lines.append(f"*{message.created_at.strftime('%H:%M:%S')}*")
                    md_lines.append("")
                    md_lines.append(message.content)
                    md_lines.append("")
                    
                    # Add SQL info if available
                    _metadata = message._metadata or {}
                    if _metadata.get('sql_result', {}).get('sql'):
                        sql = _metadata['sql_result']['sql']
                        md_lines.append("**SQL Query:**")
                        md_lines.append("```sql")
                        md_lines.append(sql)
                        md_lines.append("```")
                        md_lines.append("")
                    
                    md_lines.append("---")
                    md_lines.append("")
                
                return {
                    'content': '\n'.join(md_lines),
                    'format': 'markdown'
                }
            
            else:
                return {'error': f'Unsupported format: {format}'}
                
        except Exception as e:
            logger.error(f"Error exporting conversation: {str(e)}")
            return {'error': str(e)}
    
    def get_recent_sessions(self, user_id: str = None, days: int = 7, 
                           limit: int = 10) -> List[ChatSession]:
        """Get recently active chat sessions"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            query = ChatSession.query.filter(ChatSession.updated_at >= since_date)
            
            if user_id:
                query = query.filter_by(user_id=user_id)
            
            sessions = query.order_by(ChatSession.updated_at.desc())\
                          .limit(limit).all()
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting recent sessions: {str(e)}")
            return []
    
    def cleanup_old_sessions(self, days: int = 90, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up old chat sessions"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            old_sessions = ChatSession.query.filter(
                ChatSession.updated_at < cutoff_date
            ).all()
            
            if dry_run:
                return {
                    'dry_run': True,
                    'sessions_to_delete': len(old_sessions),
                    'oldest_session': old_sessions[-1].updated_at.isoformat() if old_sessions else None,
                    'cutoff_date': cutoff_date.isoformat()
                }
            
            # Actually delete sessions
            deleted_count = 0
            for session in old_sessions:
                db.session.delete(session)
                deleted_count += 1
            
            db.session.commit()
            
            logger.info(f"Cleaned up {deleted_count} old chat sessions")
            
            return {
                'dry_run': False,
                'sessions_deleted': deleted_count,
                'cutoff_date': cutoff_date.isoformat()
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error cleaning up old sessions: {str(e)}")
            return {'error': str(e)}
    
    def get_chat_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics for chat usage"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Get sessions in period
            sessions = ChatSession.query.filter(
                ChatSession.created_at >= since_date
            ).all()
            
            # Get messages in period
            messages = ChatMessage.query.filter(
                ChatMessage.created_at >= since_date
            ).all()
            
            # Calculate analytics
            total_sessions = len(sessions)
            total_messages = len(messages)
            user_messages = len([m for m in messages if m.role == 'user'])
            assistant_messages = len([m for m in messages if m.role == 'assistant'])
            
            # SQL query analytics
            sql_queries = 0
            successful_queries = 0
            
            for message in assistant_messages:
                if isinstance(message, ChatMessage):
                    _metadata = message._metadata or {}
                    if _metadata.get('sql_result', {}).get('sql'):
                        sql_queries += 1
                        if _metadata.get('execution_result', {}).get('status') == 'success':
                            successful_queries += 1
            
            # Average messages per session
            avg_messages_per_session = total_messages / total_sessions if total_sessions > 0 else 0
            
            # Daily breakdown
            daily_stats = {}
            for i in range(days):
                date = (datetime.utcnow() - timedelta(days=i)).date()
                date_str = date.isoformat()
                
                day_sessions = [s for s in sessions if s.created_at.date() == date]
                day_messages = [m for m in messages if m.created_at.date() == date]
                
                daily_stats[date_str] = {
                    'sessions': len(day_sessions),
                    'messages': len(day_messages)
                }
            
            return {
                'period_days': days,
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'sql_queries_generated': sql_queries,
                'successful_queries': successful_queries,
                'sql_success_rate': round(successful_queries / sql_queries * 100, 1) if sql_queries > 0 else 0,
                'avg_messages_per_session': round(avg_messages_per_session, 1),
                'daily_breakdown': daily_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting chat analytics: {str(e)}")
            return {}
    
    def regenerate_session_title(self, session_id: int) -> Optional[str]:
        """Regenerate session title based on conversation content"""
        try:
            session = ChatSession.query.get(session_id)
            if not session:
                return None
            
            # Get first user message
            first_message = ChatMessage.query.filter_by(
                session_id=session_id,
                role='user'
            ).order_by(ChatMessage.created_at.asc()).first()
            
            if first_message:
                # Generate title from first query
                content = first_message.content
                
                # Clean and truncate
                title = content.strip()
                if len(title) > 50:
                    title = title[:47] + "..."
                
                # Remove newlines
                title = ' '.join(title.split())
                
                # Update session
                session.title = title
                session.updated_at = datetime.utcnow()
                db.session.commit()
                
                return title
            
            return None
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error regenerating session title: {str(e)}")
            return None
