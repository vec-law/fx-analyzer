from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_db_manager
from uuid import UUID
from api.db_manager import DBManager
from api.utils import validate_actor

router = APIRouter()

class LoginRequest(BaseModel):
    user_name: str
    password: str

class ChangePasswordRequest(BaseModel):
    new_password: str
    repeated_password: str

@router.post("/auth/login")
def login(request: LoginRequest, db_manager: DBManager = Depends(get_db_manager)):
    try:
        if not request.user_name or not request.password:
            raise ValueError("Żadne z pól (login, hasło) nie może być puste")
        
        if not (user_id := db_manager.get_user_id(request.user_name)):
            raise ValueError(f"Użytkownik {request.user_name} nie jest zarejestrowany")
        
        if db_manager.is_blocked(user_id):
            raise ValueError(f"Użytkownik {request.user_name} jest zablokowany")

        session_token = db_manager.login_user(user_id, request.user_name, request.password)
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
    db_manager: DBManager = Depends(get_db_manager)
):
    try:
        user_id = validate_actor(
            authorization,
            x_user_id,
            db_manager,
            x_target_user_id
        )

        return {
            "success": db_manager.logout_user(
                user_id
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
    db_manager: DBManager = Depends(get_db_manager)
    ):
    try:
        user_id = validate_actor(
            authorization,
            x_user_id,
            db_manager,
            x_target_user_id
        )
        
        if not request.new_password or not request.repeated_password or \
            request.new_password != request.repeated_password:
            raise ValueError("Hasła nie mogą być puste i muszą być takie same")

        return {
            "success": db_manager.change_password(
                user_id,
                request.new_password
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
