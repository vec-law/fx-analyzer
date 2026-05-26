from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from uuid import UUID
from db.queries.users import user_exists, is_blocked, get_session_token
from typing import Optional

from db.queries.predictions import (
    get_user_predictions, add_prediction, get_prediction_status, get_prediction_logs,
    del_prediction, get_prediction_config, update_prediction_status, load_prediction_result
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
        
        if get_prediction_status(user_id, pred_uuid) in ("running", "stopping"):
            raise ValueError("Nie można usunąć predykcji która jest uruchomiona")
            
        return {"success": del_prediction(user_id, pred_uuid)}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/predictions/{pred_uuid}/config")
def get_prediction_config_endpoint(
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
        
        return {"config": get_prediction_config(user_id, pred_uuid)}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/users/{user_id}/predictions/{pred_uuid}/run")
def run_prediction_endpoint(
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
        
        status = get_prediction_status(user_id, pred_uuid)
        
        if status in ("created", "pending", "failed", 'stopped'):
            update_prediction_status(user_id, pred_uuid, "pending")
        else:
            raise ValueError("Nie można uruchomić predykcji o tym statusie")

        return {"success": True}
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.patch("/users/{user_id}/predictions/{pred_uuid}/stop")
def stop_prediction_endpoint(
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
        
        status = get_prediction_status(user_id, pred_uuid)
        
        if status == "running":
            update_prediction_status(user_id, pred_uuid, "stopping")
        else:
            raise ValueError("Predykcja nie jest uruchomiona")

        return {"success": True}
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/predictions/{pred_uuid}/logs")
def get_prediction_logs_endpoint(
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
        
        return {"logs": get_prediction_logs(user_id, pred_uuid)}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/predictions/{pred_uuid}/result/{arch_name}")
def get_prediction_results_endpoint(
    user_id: int,
    pred_uuid: UUID,
    arch_name: str,
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
        
        data = load_prediction_result(user_id, pred_uuid, arch_name)
        if data is None:
            raise ValueError("Brak wyników predykcji")
        
        import base64
        return {"data": base64.b64encode(data).decode()}
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))