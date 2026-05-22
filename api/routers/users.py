from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from api.dependencies import get_db_manager
from api.db_manager import DBManager
from uuid import UUID

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
def change_password(
    user_id: int,
    request: ChangePasswordRequest,
    authorization: str = Header(...),
    requester_id: str | None = Header(None, alias="x-user-id"),
    db_manager: DBManager = Depends(get_db_manager)
    ):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not db_manager.user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        if db_manager.is_blocked(user_id):
            raise ValueError("Użytkownik zablokowany")

        if requester_id is None:
            requester_id = user_id
        else:
            if not requester_id.isdigit():
                raise ValueError("Nieprawidłowy identyfikator")
            
            if (requester_id := int(requester_id)) != user_id:
                if not db_manager.user_exists(requester_id):
                    raise ValueError("Użytkownik nie istnieje")
                if db_manager.is_blocked(requester_id):
                    raise ValueError("Użytkownik zablokowany")
                
                if db_manager.get_role(requester_id) != "admin":
                    raise HTTPException(status_code=403, detail="Brak uprawnień")
        
        if session_token != db_manager.get_session_token(requester_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
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

@router.get("/users/{user_id}/role")
def get_role(
    user_id: int,
    authorization: str = Header(...),
    db_manager: DBManager = Depends(get_db_manager)
    ):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not db_manager.user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if session_token != db_manager.get_session_token(user_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        if db_manager.is_blocked(user_id):
            raise ValueError("Użytkownik zablokowany")

        return {"role_name": db_manager.get_role(user_id)}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users")
def get_users(
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id"),
    db_manager : DBManager = Depends(get_db_manager)
    ):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if not db_manager.user_exists(requester_id := int(requester_id)):
            raise ValueError("Użytkownik nie istnieje")
        
        if session_token != db_manager.get_session_token(requester_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        if db_manager.is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if db_manager.get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")

        return {"users": db_manager.get_users()}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users")
def add_user(
    request: AddUserRequest,
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id"),
    db_manager : DBManager = Depends(get_db_manager)
    ):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if not db_manager.user_exists(requester_id := int(requester_id)):
            raise ValueError("Użytkownik nie istnieje")
        
        if session_token != db_manager.get_session_token(requester_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        if db_manager.is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if db_manager.get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")
        
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
    
@router.delete("/users/{user_id}")
def del_user(
    user_id: int,
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id"),
    db_manager: DBManager = Depends(get_db_manager)
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not db_manager.user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if (requester_id := int(requester_id)) == user_id:
            raise ValueError("Nieprawidłowa operacja")
        
        if not db_manager.user_exists(requester_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if db_manager.is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if session_token != db_manager.get_session_token(requester_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")
 
        if db_manager.get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")
    
        db_manager.block_user(user_id)
        db_manager.logout_user(user_id)
        # TODO: zakończyć wszystkie zadania treningu i predykcji
        db_manager.del_user(user_id)
            
        return {"success": True}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/users/{user_id}/block")
def block_user(
    user_id: int,
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id"),
    db_manager: DBManager = Depends(get_db_manager)
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not db_manager.user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if (requester_id := int(requester_id)) == user_id:
            raise ValueError("Nieprawidłowa operacja")
        
        if not db_manager.user_exists(requester_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if db_manager.is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if session_token != db_manager.get_session_token(requester_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")
 
        if db_manager.get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")
    
        db_manager.block_user(user_id)
        db_manager.logout_user(user_id)
        # TODO: zakończyć wszystkie zadania treningu i predykcji
            
        return {"success": True}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/users/{user_id}/unblock")
def unblock_user(
    user_id: int,
    authorization: str = Header(...),
    requester_id: str = Header(..., alias="x-user-id"),
    db_manager: DBManager = Depends(get_db_manager)
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        if not db_manager.user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if not requester_id.isdigit():
            raise ValueError("Nieprawidłowy identyfikator")
        
        if (requester_id := int(requester_id)) == user_id:
            raise ValueError("Nieprawidłowa operacja")
        
        if not db_manager.user_exists(requester_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if db_manager.is_blocked(requester_id):
            raise ValueError("Użytkownik zablokowany")
        
        if session_token != db_manager.get_session_token(requester_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")
 
        if db_manager.get_role(requester_id) != "admin":
            raise HTTPException(status_code=403, detail="Brak uprawnień")
    
        db_manager.unblock_user(user_id)
            
        return {"success": True}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    