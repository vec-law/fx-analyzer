from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_db_manager
from uuid import UUID

router = APIRouter()

class LoginRequest(BaseModel):
    user_name: str
    password: str

class ChangePasswordRequest(BaseModel):
    new_password: str
    repeated_password: str

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

@router.delete("/auth/logout")
def logout(
    authorization: str = Header(...),
    x_user_id: str = Header(...),
    x_target_user_id: str = Header(None),
    db_manager = Depends(get_db_manager)
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        target_user_id = int(x_target_user_id) if x_target_user_id else None

        return {
            "user_id": db_manager.logout_user(
                int(x_user_id),
                session_token,
                target_user_id
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.patch("/auth/change-password")
def change_password(
    request: ChangePasswordRequest,
    authorization: str = Header(...),
    x_user_id: str = Header(...),
    x_target_user_id: str = Header(None),
    db_manager = Depends(get_db_manager)
    ):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        target_user_id = int(x_target_user_id) if x_target_user_id else None

        return {
            "success": db_manager.change_password(
                int(x_user_id),
                session_token,
                request.new_password,
                request.repeated_password,
                target_user_id
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/auth/get-role")
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
