from fastapi import APIRouter, Header, Depends, HTTPException
from pydantic import BaseModel
from api.dependencies import get_db_manager
from db.manager import DatabaseManager
from uuid import UUID

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

@router.get("/users/{user_id}/trainings")
def get_trainings(
    user_id: int,
    authorization: str = Header(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
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
        
        return {"trainings": db_manager.get_user_trainings(user_id)}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/{user_id}/trainings")
def add_training(
    user_id: int,
    request: AddTrainingRequest,
    authorization: str = Header(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
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
        
        if not request.features or not request.targets or not request.architectures:
            raise ValueError("Brak featurów, targetów lub architektur")
        
        config = {
            "instrument_name": request.instrument_name,
            "timeframe_name": request.timeframe_name,
            "data_source_name": request.data_source_name,
            "all_samples": request.all_samples,
            "test_samples": request.test_samples,
            "seed": request.seed,
            "epochs": request.epochs,
            "train_noise": request.train_noise,
            "learning_rate": request.learning_rate,
            "features": request.features,
            "targets": request.targets,
            "architectures": request.architectures
        }
        
        return {"train_uuid": db_manager.add_training(user_id, config)}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/users/{user_id}/trainings/{train_uuid}")
def del_training(
    user_id: int,
    train_uuid: UUID,
    authorization: str = Header(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
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
            
        return {"success": db_manager.del_training(train_uuid)}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/users/{user_id}/trainings/{train_uuid}/config")
def get_training_config(
    user_id: int,
    train_uuid: UUID,
    authorization: str = Header(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
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
        
        return {"config": db_manager.get_training_config(train_uuid)}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/users/{user_id}/trainings/{train_uuid}/run")
def run_training(
    user_id: int,
    train_uuid: UUID,
    authorization: str = Header(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
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
        
        status = db_manager.get_training_status(train_uuid)
        
        if status in ("created", "pending", "failed"):
            db_manager.update_training_status(train_uuid, "pending")
        else:
            raise ValueError("Nie można uruchomić treningu o tym statusie")

        return {"success": True}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.patch("/users/{user_id}/trainings/{train_uuid}/stop")
def stop_training(
    user_id: int,
    train_uuid: UUID,
    authorization: str = Header(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
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
        
        status = db_manager.get_training_status(train_uuid)
        
        if status == "running":
            db_manager.update_training_status(train_uuid, "stopping")
        else:
            raise ValueError("Trening nie jest uruchomiony")

        return {"success": True}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))