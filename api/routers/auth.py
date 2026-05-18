from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_db_manager
from uuid import UUID
from fastapi import HTTPException

router = APIRouter()

class LoginRequest(BaseModel):
    user_name: str
    password: str

class LogoutRequest(BaseModel):
    user_id: int
    session_token: str
    target_user_id: Optional[int] = None

class ChangePasswordRequest(BaseModel):
    user_id: int
    session_token: str
    new_password: str
    repeated_password: str
    target_user_id: Optional[int] = None

class GetRoleRequest(BaseModel):
    user_id: int
    session_token: str

@router.post("/auth/login")
def login(request: LoginRequest, db_manager = Depends(get_db_manager)):
    try:
        user_id, session_token = db_manager.login_user(request.user_name, request.password)
        return {
            "user_id": user_id,
            "session_token": str(session_token)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auth/logout")
def logout(request: LogoutRequest, db_manager = Depends(get_db_manager)):
    try:
        return {
            "user_id": db_manager.logout_user(
                request.user_id,
                UUID(request.session_token),
                request.target_user_id
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/auth/change_password")
def change_password(request: ChangePasswordRequest, db_manager = Depends(get_db_manager)):
    try:
        return {
            "success": db_manager.change_password(
                request.user_id,
                UUID(request.session_token),
                request.new_password,
                request.repeated_password,
                request.target_user_id
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/auth/get_role")
def get_role(request: GetRoleRequest, db_manager = Depends(get_db_manager)):
    try:
        return {
            "role_name": db_manager.get_role(
                request.user_id,
                UUID(request.session_token)
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
