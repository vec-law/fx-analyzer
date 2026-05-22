from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from api.dependencies import get_db_manager
from api.utils import get_actor_role, actor_is_admin

class AddUserRequest(BaseModel):
    user_name: str
    password: str
    repeated_password: str
    is_admin: bool

router = APIRouter()

@router.get("/users/role")
def get_role(
    authorization: str = Header(...),
    x_user_id: str = Header(...),
    db_manager = Depends(get_db_manager)
    ):
    try:
        return {
            "role_name": get_actor_role(
                authorization,
                x_user_id,
                db_manager
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
        if actor_is_admin(authorization, x_user_id, db_manager):
            return {"users": db_manager.get_users()}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users")
def add_user(
    request: AddUserRequest,
    authorization: str = Header(...),
    x_user_id: str = Header(...),
    db_manager = Depends(get_db_manager)
    ):
    try:
        if actor_is_admin(authorization, x_user_id, db_manager):
            if not (request.user_name and request.password and request.repeated_password):
                raise ValueError("Żadne z pól (login, hasła) nie może być puste")
            
            if request.password != request.repeated_password:
                raise ValueError("Hasła muszą być takie same")
            
            return {
                "success": db_manager.add_user(
                    request.user_name,
                    request.password,
                    request.is_admin
                )
            }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
