from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from api.dependencies import get_db_manager
from api.utils import get_actor_role, actor_is_admin
from api.db_manager import DBManager

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
    db_manager : DBManager = Depends(get_db_manager)
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
    db_manager : DBManager = Depends(get_db_manager)
    ):
    try:
        if actor_is_admin(authorization, x_user_id, db_manager):
            if not (request.user_name and request.password and request.repeated_password):
                raise ValueError("Żadne z pól (login, hasła) nie może być puste")
            
            if db_manager.get_user_id(request.user_name):
                raise ValueError(f"Użytkownik {request.user_name} już istnieje")
            
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
    
@router.delete("/users")
def del_user(
    authorization: str = Header(...),
    x_user_id: str = Header(...),
    x_target_user_id: str = Header(...),
    db_manager: DBManager = Depends(get_db_manager)
):
    try:
        if actor_is_admin(authorization, x_user_id, db_manager):
            if x_target_user_id == x_user_id:
                raise ValueError("Niedozwolona operacja")
            
            if not db_manager.user_exists(target_user_id := int(x_target_user_id)):
                raise ValueError("Użytkownik nie istnieje")
            
            db_manager.block_user(target_user_id)
            db_manager.logout_user(target_user_id)
            # TODO: zakończyć wszystkie zadania treningu i predykcji
            db_manager.del_user(target_user_id)
            
            return {"success": True}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
