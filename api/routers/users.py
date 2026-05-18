from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_db_manager
from uuid import UUID

router = APIRouter()

@router.get("/users/role")
def get_role(
    authorization: str = Header(...),
    x_user_id: str = Header(...),
    db_manager = Depends(get_db_manager)
    ):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        return {
            "role_name": db_manager.get_role(
                int(x_user_id),
                session_token
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users")
def get_users(
    authorization: str = Header(...),
    x_user_id: str = Header(...),
    db_manager = Depends(get_db_manager)
    ):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        return {
            "users": db_manager.get_users(
                int(x_user_id),
                session_token
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))