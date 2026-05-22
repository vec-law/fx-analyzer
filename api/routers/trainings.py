from fastapi import APIRouter, Header, Depends, HTTPException
from api.dependencies import get_db_manager
from api.db_manager import DBManager
from uuid import UUID

router = APIRouter()

@router.get("/users/{user_id}/trainings")
def get_trainings(
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
        if db_manager.is_blocked(user_id):
            raise ValueError("Użytkownik zablokowany")
        
        if session_token != db_manager.get_session_token(user_id):
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        return {"trainings": db_manager.get_trainings(user_id)}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
