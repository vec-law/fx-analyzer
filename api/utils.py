from uuid import UUID
from api.db_manager import DBManager

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