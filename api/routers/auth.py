from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_db_manager
from uuid import UUID

router = APIRouter()

class LoginRequest(BaseModel):
    user_name: str
    password: str

class LogoutRequest(BaseModel):
    user_id: int
    session_token: str
    target_user_id: Optional[int] = None

@router.post("/auth/login")
def login(request: LoginRequest, db_manager = Depends(get_db_manager)):
    user_id, session_token = db_manager.login_user(request.user_name, request.password)
    return {"user_id": user_id, "session_token": str(session_token)}

@router.post("/auth/logout")
def logout(request: LogoutRequest, db_manager = Depends(get_db_manager)):
    return {"user_id": db_manager.logout_user(request.user_id, UUID(request.session_token), request.target_user_id)}
