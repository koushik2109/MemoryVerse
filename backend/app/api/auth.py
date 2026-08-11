from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.domain import SignUpRequest, SignInRequest, ForgotPasswordRequest, AuthResponse, UserProfile
from app.core.db import get_supabase_client
from app.core.security import get_current_user, CurrentUser

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup", response_model=AuthResponse)
async def signup(payload: SignUpRequest):
    supabase = get_supabase_client()
    try:
        res = supabase.auth.sign_up({
            "email": payload.email,
            "password": payload.password,
            "options": {
                "data": {"full_name": payload.full_name or payload.email.split("@")[0]}
            }
        })
        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed")
        
        session = res.session
        token = session.access_token if session else "stub_token"
        return AuthResponse(
            access_token=token,
            user_id=str(res.user.id),
            email=str(res.user.email),
            full_name=payload.full_name
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=AuthResponse)
async def login(payload: SignInRequest):
    supabase = get_supabase_client()
    try:
        res = supabase.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password
        })
        if not res.user or not res.session:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        meta = res.user.user_metadata or {}
        return AuthResponse(
            access_token=res.session.access_token,
            user_id=str(res.user.id),
            email=str(res.user.email),
            full_name=meta.get("full_name")
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest):
    supabase = get_supabase_client()
    try:
        supabase.auth.reset_password_for_email(payload.email)
        return {"message": "Password reset link sent to email"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/me", response_model=CurrentUser)
async def get_me(user: CurrentUser = Depends(get_current_user)):
    return user
