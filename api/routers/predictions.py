from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from uuid import UUID
from db.queries.users import user_exists, is_blocked, get_session_token
from typing import Optional

from db.queries.predictions import (
    get_user_predictions
)

router = APIRouter()

class AddTrainingRequest(BaseModel):
    instrument_name: str
    timeframe_name: str
    data_source_name: str
    all_samples: int
    test_samples: int
    seed: int
    epochs: int
    train_noise: float
    learning_rate: float
    features: list
    targets: list
    architectures: list

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

# @router.post("/users/{user_id}/trainings")
# def add_training_endpoint(
#     user_id: int,
#     request: AddTrainingRequest,
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
        
#         if not request.features or not request.targets or not request.architectures:
#             raise ValueError("Brak featurów, targetów lub architektur")
        
#         config = {
#             "instrument_name": request.instrument_name,
#             "timeframe_name": request.timeframe_name,
#             "data_source_name": request.data_source_name,
#             "all_samples": request.all_samples,
#             "test_samples": request.test_samples,
#             "seed": request.seed,
#             "epochs": request.epochs,
#             "train_noise": request.train_noise,
#             "learning_rate": request.learning_rate,
#             "features": request.features,
#             "targets": request.targets,
#             "architectures": request.architectures
#         }
        
#         return {"train_uuid": add_training(user_id, config)}
        
#     except HTTPException:
#         raise
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.delete("/users/{user_id}/trainings/{train_uuid}")
# def del_training_endpoint(
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
        
#         if get_training_status(train_uuid) in ("running", "stopping"):
#             raise ValueError("Nie można usunąć treningu który jest uruchomiony")
            
#         return {"success": del_training(train_uuid)}
    
#     except HTTPException:
#         raise
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

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