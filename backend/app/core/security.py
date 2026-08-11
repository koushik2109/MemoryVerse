from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel
from app.config.settings import settings
from app.core.db import get_supabase_client

security_scheme = HTTPBearer()

class CurrentUser(BaseModel):
    id: str
    email: str
    full_name: str | None = None

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
) -> CurrentUser:
    token = credentials.credentials
    try:
        supabase = get_supabase_client()
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = user_response.user
        metadata = user.user_metadata or {}
        user_email = user.email or ""
        return CurrentUser(
            id=str(user.id),
            email=user_email,
            full_name=metadata.get("full_name") or (user_email.split("@")[0] if user_email else "User"),
        )
    except Exception as e:
        # Fallback to decode JWT directly if standalone JWT mode
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM], options={"verify_aud": False})
            user_id: str = str(payload.get("sub") or payload.get("user_id"))
            email: str = payload.get("email", "")
            if not user_id:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
            return CurrentUser(id=user_id, email=email)
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Could not validate credentials: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
