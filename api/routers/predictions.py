from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from uuid import UUID
from db.queries.users import user_exists, is_blocked, get_session_token
from typing import Optional

from db.queries.predictions import (
    get_user_predictions, add_prediction, get_prediction_status,
    del_prediction
)

router = APIRouter()

class AddPredictionRequest(BaseModel):
    train_uuid: UUID
    all_samples: int
    predicted_samples: int

@router.get("/users/{user_id}/predictions")
def get_predictions_endpoint(
    user_id: int,
    authorization: str = Header(...),
    status: Optional[str] = None
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
        if (db_token := get_session_token(user_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        return {"predictions": get_user_predictions(user_id, status)}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/{user_id}/predictions")
def add_prediction_endpoint(
    user_id: int,
    request: AddPredictionRequest,
    authorization: str = Header(...)
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
        if (db_token := get_session_token(user_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        config = {
            "train_uuid": request.train_uuid,
            "all_samples": request.all_samples,
            "predicted_samples": request.predicted_samples,
        }
        
        return {"pred_uuid": add_prediction(user_id, config)}
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/users/{user_id}/predictions/{pred_uuid}")
def del_prediction_endpoint(
    user_id: int,
    pred_uuid: UUID,
    authorization: str = Header(...)
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
        if (db_token := get_session_token(user_id)) is None or session_token != db_token:
            raise HTTPException(status_code=401, detail="Brak dostępu")
        
        if get_prediction_status(pred_uuid) in ("running", "stopping"):
            raise ValueError("Nie można usunąć predykcji która jest uruchomiona")
            
        return {"success": del_prediction(user_id, pred_uuid)}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.get("/users/{user_id}/trainings/{train_uuid}/config")
# def get_training_config_endpoint(
#     user_id: int,
#     train_uuid: UUID,
#     authorization: str = Header(...)
# ):
#     try:
#         if authorization.startswith("Bearer "):
#             session_token = UUID(authorization.removeprefix("Bearer "))
#         else:
#             raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
#         if not user_exists(user_id):
#             raise ValueError("Użytkownik nie istnieje")
#         if is_blocked(user_id):
#             raise ValueError("Użytkownik zablokowany")
#         if (db_token := get_session_token(user_id)) is None or session_token != db_token:
#             raise HTTPException(status_code=401, detail="Brak dostępu")
        
#         return {"config": get_training_config(train_uuid)}
    
#     except HTTPException:
#         raise
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/users/{user_id}/trainings/{train_uuid}/run")
# def run_training_endpoint(
#     user_id: int,
#     train_uuid: UUID,
#     authorization: str = Header(...)
# ):
#     try:
#         if authorization.startswith("Bearer "):
#             session_token = UUID(authorization.removeprefix("Bearer "))
#         else:
#             raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
#         if not user_exists(user_id):
#             raise ValueError("Użytkownik nie istnieje")
#         if is_blocked(user_id):
#             raise ValueError("Użytkownik zablokowany")
#         if (db_token := get_session_token(user_id)) is None or session_token != db_token:
#             raise HTTPException(status_code=401, detail="Brak dostępu")
        
#         status = get_training_status(train_uuid)
        
#         if status in ("created", "pending", "failed", 'stopped'):
#             update_training_status(train_uuid, "pending")
#         else:
#             raise ValueError("Nie można uruchomić treningu o tym statusie")

#         return {"success": True}
        
#     except HTTPException:
#         raise
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.patch("/users/{user_id}/trainings/{train_uuid}/stop")
# def stop_training_endpoint(
#     user_id: int,
#     train_uuid: UUID,
#     authorization: str = Header(...)
# ):
#     try:
#         if authorization.startswith("Bearer "):
#             session_token = UUID(authorization.removeprefix("Bearer "))
#         else:
#             raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
#         if not user_exists(user_id):
#             raise ValueError("Użytkownik nie istnieje")
#         if is_blocked(user_id):
#             raise ValueError("Użytkownik zablokowany")
#         if (db_token := get_session_token(user_id)) is None or session_token != db_token:
#             raise HTTPException(status_code=401, detail="Brak dostępu")
        
#         status = get_training_status(train_uuid)
        
#         if status == "running":
#             update_training_status(train_uuid, "stopping")
#         else:
#             raise ValueError("Trening nie jest uruchomiony")

#         return {"success": True}
        
#     except HTTPException:
#         raise
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/users/{user_id}/trainings/{train_uuid}/logs")
# def get_training_logs_endpoint(
#     user_id: int,
#     train_uuid: UUID,
#     authorization: str = Header(...)
# ):
#     try:
#         if authorization.startswith("Bearer "):
#             session_token = UUID(authorization.removeprefix("Bearer "))
#         else:
#             raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
#         if not user_exists(user_id):
#             raise ValueError("Użytkownik nie istnieje")
#         if is_blocked(user_id):
#             raise ValueError("Użytkownik zablokowany")
        
#         if (db_token := get_session_token(user_id)) is None or session_token != db_token:
#             raise HTTPException(status_code=401, detail="Brak dostępu")
        
#         return {"logs": get_training_logs(train_uuid)}
    
#     except HTTPException:
#         raise
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))