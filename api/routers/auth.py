from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.dependencies import get_db_manager

router = APIRouter()

class LoginRequest(BaseModel):
    user_name: str
    password: str

@router.post("/auth/login")
def login(request: LoginRequest, db_manager = Depends(get_db_manager)):
    user_id, session_token = db_manager.login_user(request.user_name, request.password)
    return {"user_id": user_id, "session_token": str(session_token)}
