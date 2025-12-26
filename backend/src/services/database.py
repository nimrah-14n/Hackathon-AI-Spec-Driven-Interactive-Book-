from sqlalchemy.orm import Session
from ..models.user import User, ChatSession, ChatMessage
from typing import Optional
import uuid
from datetime import datetime


class DatabaseService:
    def __init__(self, db: Session):
        self.db = db

    def create_user(self, email: str, name: str, hashed_password: str,
                    software_background: str = "none", hardware_background: str = "none") -> User:
        """Create a new user in the database"""
        db_user = User(
            email=email,
            name=name,
            hashed_password=hashed_password,
            software_background=software_background,
            hardware_background=hardware_background
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()

    def create_chat_session(self, user_id: Optional[int] = None) -> ChatSession:
        """Create a new chat session"""
        session_token = str(uuid.uuid4())
        db_session = ChatSession(
            user_id=user_id,
            session_token=session_token
        )
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        return db_session

    def create_chat_message(self, session_id: int, role: str, content: str,
                           context_used: str = "book_wide") -> ChatMessage:
        """Create a new chat message"""
        db_message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            context_used=context_used
        )
        self.db.add(db_message)
        self.db.commit()
        self.db.refresh(db_message)
        return db_message

    def get_chat_messages_by_session(self, session_id: int) -> list[ChatMessage]:
        """Get all messages for a session"""
        return self.db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at).all()

    def update_user_background(self, user_id: int, software_background: str, hardware_background: str) -> User:
        """Update user background information"""
        db_user = self.db.query(User).filter(User.id == user_id).first()
        if db_user:
            db_user.software_background = software_background
            db_user.hardware_background = hardware_background
            db_user.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(db_user)
        return db_user