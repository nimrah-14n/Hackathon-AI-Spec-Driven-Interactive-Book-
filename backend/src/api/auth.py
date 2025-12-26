from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..services.database import DatabaseService
from ..services.feature_flags import FeatureFlagsService
from ..config.database import get_db
from ..config.settings import settings
from passlib.context import CryptContext
from jose import JWTError, jwt
import uuid

router = APIRouter()

# Feature flags service
feature_flags = FeatureFlagsService()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token helper functions
def get_authorization_header(authorization: str = None):
    """Get authorization header from request"""
    from fastapi import Header
    return Header(default=None)(authorization)


def get_token_from_header(authorization: str = Depends(get_authorization_header)):
    """Extract token from Authorization header"""
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    return authorization[7:]  # Remove "Bearer " prefix


def verify_token(token: str = Depends(get_token_from_header)):
    """Verify JWT token and return user ID"""
    if not settings.better_auth_secret:
        raise HTTPException(status_code=500, detail="Better Auth secret not configured")

    try:
        payload = jwt.decode(token, settings.better_auth_secret, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return int(user_id)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Pydantic models for auth
class UserRegistration(BaseModel):
    email: str
    password: str
    name: str
    software_background: Optional[str] = "none"
    hardware_background: Optional[str] = "none"

class UserLogin(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    id: int
    email: str
    name: str
    software_background: str
    hardware_background: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class UserLoginResponse(BaseModel):
    user: UserProfile
    token: str

@router.post("/register", response_model=UserProfile)
async def register_user(user_data: UserRegistration, db: Session = Depends(get_db)):
    # Check if authentication feature is enabled
    if not feature_flags.is_enabled("auth"):
        raise HTTPException(status_code=403, detail="Authentication is not enabled")

    db_service = DatabaseService(db)

    # Check if user already exists
    existing_user = db.query(db_service.User).filter(db_service.User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash the password
    hashed_password = pwd_context.hash(user_data.password)

    # Create new user with background information
    new_user = db_service.User(
        email=user_data.email,
        name=user_data.name,
        hashed_password=hashed_password,
        software_background=user_data.software_background,
        hardware_background=user_data.hardware_background
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Convert SQLAlchemy object to response format
    return UserProfile(
        id=new_user.id,
        email=new_user.email,
        name=new_user.name,
        software_background=new_user.software_background,
        hardware_background=new_user.hardware_background,
        created_at=new_user.created_at
    )

@router.post("/login", response_model=UserLoginResponse)
async def login_user(credentials: UserLogin, db: Session = Depends(get_db)):
    # Check if authentication feature is enabled
    if not feature_flags.is_enabled("auth"):
        raise HTTPException(status_code=403, detail="Authentication is not enabled")

    db_service = DatabaseService(db)

    # Find user by email
    user = db.query(db_service.User).filter(db_service.User.email == credentials.email).first()

    if not user or not pwd_context.verify(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Convert SQLAlchemy object to response format
    user_profile = UserProfile(
        id=user.id,
        email=user.email,
        name=user.name,
        software_background=user.software_background,
        hardware_background=user.hardware_background,
        created_at=user.created_at,
        updated_at=user.updated_at
    )

    # Generate JWT token using Better Auth secret from settings
    if not settings.better_auth_secret:
        raise HTTPException(status_code=500, detail="Better Auth secret not configured")

    # Create JWT token with user information
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "name": user.name,
        "exp": datetime.utcnow() + timedelta(days=7)  # Token expires in 7 days
    }
    token = jwt.encode(token_data, settings.better_auth_secret, algorithm="HS256")

    return UserLoginResponse(user=user_profile, token=token)

@router.get("/profile", response_model=UserProfile)
async def get_profile(user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    """Get user profile with authentication"""
    db_service = DatabaseService(db)

    # Find user by ID from token
    user = db.query(db_service.User).filter(db_service.User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Convert SQLAlchemy object to response format
    return UserProfile(
        id=user.id,
        email=user.email,
        name=user.name,
        software_background=user.software_background,
        hardware_background=user.hardware_background,
        created_at=user.created_at,
        updated_at=user.updated_at
    )