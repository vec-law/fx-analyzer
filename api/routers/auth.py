from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from uuid import UUID
from db.queries.users import (
    get_user_id, is_blocked, login_user,
    user_exists, get_role, get_session_token, logout_user
)

router = APIRouter()

class LoginRequest(BaseModel):
    user_name: str
    password: str

@router.post("/auth/login")
def login(request: LoginRequest):
    try:
        if not request.user_name or not request.password:
            raise ValueError("Żadne z pól (login, hasło) nie może być puste")
        
        if not (user_id := get_user_id(request.user_name)):
            raise ValueError(f"Użytkownik {request.user_name} nie jest zarejestrowany")
        
        if is_blocked(user_id):
            raise ValueError(f"Użytkownik {request.user_name} jest zablokowany")

        session_token = login_user(user_id, request.user_name, request.password)
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
    x_target_user_id: str = Header(None)
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        user_id = int(x_user_id)
        target_user_id = int(x_target_user_id) if x_target_user_id else None

        if not user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if target_user_id is not None and target_user_id != user_id:
            if get_role(user_id) != "admin":
                raise HTTPException(status_code=403, detail="Brak uprawnień")
            user_id = target_user_id

        if session_token != get_session_token(user_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")

        return {"success": logout_user(user_id)}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
