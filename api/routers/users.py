from db.queries.trainings import get_user_trainings, update_training_status
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from uuid import UUID
from db.queries.users import (
    user_exists, is_blocked, get_role, get_session_token,
    change_password, get_users, add_user, get_user_id,
    del_user, block_user, unblock_user, logout_user
)

class AddUserRequest(BaseModel):
    user_name: str
    password: str
    repeated_password: str
    is_admin: bool

class ChangePasswordRequest(BaseModel):
    new_password: str
    repeated_password: str

router = APIRouter()

@router.patch("/users/{user_id}/password")
def change_password_endpoint(
    user_id: int,
    request: ChangePasswordRequest,
    authorization: str = Header(...),
    requester_id: str | None = Header(None, alias="x-user-id")
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        if is_blocked(user_id):
            raise ValueError("Użytkownik zablokowany")

        if requester_id is None:
            requester_id = user_id
        else:
            if not requester_id.isdigit():
                raise ValueError("Nieprawidłowy identyfikator")
            if (requester_id := int(requester_id)) != user_id:
                if not user_exists(requester_id):
                    raise ValueError("Użytkownik nie istnieje")
                if is_blocked(requester_id):
                    raise ValueError("Użytkownik zablokowany")
                if get_role(requester_id) != "admin":
                    raise HTTPException(status_code=403, detail="Brak uprawnień")
        
        if (db_token := get_session_token(requester_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        if not request.new_password or not request.repeated_password or \
            request.new_password != request.repeated_password:
            raise ValueError("Hasła nie mogą być puste i muszą być takie same")

        return {"success": change_password(user_id, request.new_password)}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/role")
def get_role_endpoint(
    user_id: int,
    authorization: str = Header(...)
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if (db_token := get_session_token(user_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        if is_blocked(user_id):
            raise ValueError("Użytkownik zablokowany")

        return {"role_name": get_role(user_id)}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users")
def get_users_endpoint(
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id")
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if not user_exists(requester_id := int(requester_id)):
            raise ValueError("Użytkownik nie istnieje")
        
        if (db_token := get_session_token(requester_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        if is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")

        return {"users": get_users()}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users")
def add_user_endpoint(
    request: AddUserRequest,
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id")
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if not user_exists(requester_id := int(requester_id)):
            raise ValueError("Użytkownik nie istnieje")
        
        if (db_token := get_session_token(requester_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        if is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")
        
        if not (request.user_name and request.password and request.repeated_password):
            raise ValueError("Żadne z pól (login, hasła) nie może być puste")
            
        if get_user_id(request.user_name):
            raise ValueError(f"Użytkownik {request.user_name} już istnieje")
            
        if request.password != request.repeated_password:
            raise ValueError("Hasła muszą być takie same")
            
        return {"success": add_user(request.user_name, request.password, request.is_admin)}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/users/{user_id}")
def del_user_endpoint(
    user_id: int,
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id")
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if (requester_id := int(requester_id)) == user_id:
            raise ValueError("Nieprawidłowa operacja")
        
        if not user_exists(requester_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if (db_token := get_session_token(requester_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
 
        if get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")
    
        block_user(user_id)
        logout_user(user_id)

        trainings = get_user_trainings(user_id)
        for training in trainings:
            if training['status'] in ('running', 'stopping', 'pending'):
                update_training_status(training['train_uuid'], 'stopping')
                raise ValueError("Treningi użytkownika są zatrzymywane — spróbuj ponownie za chwilę")

        del_user(user_id)
            
        return {"success": True}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/users/{user_id}/block")
def block_user_endpoint(
    user_id: int,
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id")
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if (requester_id := int(requester_id)) == user_id:
            raise ValueError("Nieprawidłowa operacja")
        
        if not user_exists(requester_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if (db_token := get_session_token(requester_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
 
        if get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")
    
        block_user(user_id)
        logout_user(user_id)

        trainings = get_user_trainings(user_id)
        for training in trainings:
            if training['status'] in ('running', 'stopping', 'pending'):
                update_training_status(training['train_uuid'], 'stopping')
            
        return {"success": True}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/users/{user_id}/unblock")
def unblock_user_endpoint(
    user_id: int,
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id")
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if (requester_id := int(requester_id)) == user_id:
            raise ValueError("Nieprawidłowa operacja")
        
        if not user_exists(requester_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if (db_token := get_session_token(requester_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
 
        if get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")
    
        unblock_user(user_id)
            
        return {"success": True}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
