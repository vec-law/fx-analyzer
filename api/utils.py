from uuid import UUID
from api.db_manager import DBManager

def actor_is_admin(
    authorization: str,
    x_user_id: str,
    db_manager : DBManager
    ):
    try:
        if get_actor_role(
            authorization,
            x_user_id,
            db_manager
        ) != "admin":
            raise ValueError("Brak uprawnień")
        return True
        
    except ValueError as e:
        raise e
    except Exception as e:
        raise e

def get_actor_role(
    authorization: str,
    x_user_id: str,
    db_manager : DBManager
    ):
    try:
        user_id = validate_actor(
            authorization,
            x_user_id,
            db_manager
        )
        return db_manager.get_role(user_id)
    
    except ValueError as e:
        raise e
    except Exception as e:
        raise e

def validate_actor(
        authorization: str,
        x_user_id: str,
        db_manager: DBManager,
        x_target_user_id: str | None = None
):
    try:
        if authorization.startswith("Bearer "):
            session_token = UUID(authorization.removeprefix("Bearer "))
        else:
            raise ValueError("Nieprawidłowy format nagłówka Authorization")
        
        user_id = int(x_user_id)
        
        target_user_id = int(x_target_user_id) if x_target_user_id else None

        if not db_manager.user_exists(user_id):
            raise ValueError("Użytkownik nie istnieje")
        
        if target_user_id is not None and target_user_id != user_id:
            if db_manager.get_role(user_id) != "admin": raise ValueError("Brak uprawnień")
            user_id = target_user_id

        if session_token != db_manager.get_session_token(user_id):
            raise ValueError("Brak dostępu")
        
        return user_id
    except ValueError as e:
        raise e
    except Exception as e:
        raise e
