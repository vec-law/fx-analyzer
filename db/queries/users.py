import psycopg2
import bcrypt
import uuid
import os
from db.config import DB_CONFIG

def add_user(user_name, password, is_admin):
    try:           
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_type = "admin" if is_admin else "user"
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO app_user (name, password_hash, role_id)
                    VALUES (%s, %s, (SELECT id FROM role WHERE name = %s))
                """, (user_name, password_hash, user_type))
                conn.commit()
                return True
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def del_user(user_id):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM app_user WHERE id = %s", (user_id, ))
                if cur.rowcount == 0:
                    raise Exception("Użytkownik nie istnieje w bazie danych")
                conn.commit()
                return True
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def block_user(user_id):
    try:                
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE app_user SET is_blocked = TRUE WHERE id = %s
                """, (user_id, ))
                if cur.rowcount == 0:
                    raise Exception("Użytkownik nie istnieje w bazie danych")
                conn.commit()
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def unblock_user(user_id):
    try:                
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE app_user SET is_blocked = FALSE WHERE id = %s
                """, (user_id, ))
                if cur.rowcount == 0:
                    raise Exception("Użytkownik nie istnieje w bazie danych")
                conn.commit()
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def login_user(user_id, user_name, password):
    try:           
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT password_hash FROM app_user WHERE app_user.name = %s
                """, (user_name, ))
                result = cur.fetchone()
                password_hash = bytes(result[0])
                if not bcrypt.checkpw(password.encode('utf-8'), password_hash):
                    raise ValueError(f"Podano nieprawidłowe dane logowania")
                session_token = uuid.uuid4()
                cur.execute("""
                    UPDATE app_user SET session_token = %s WHERE app_user.id = %s
                """, (session_token, user_id))
                conn.commit()
                return session_token
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def logout_user(user_id):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE app_user SET session_token = NULL WHERE app_user.id = %s
                """, (user_id, ))
                conn.commit()
                return True
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def change_password(user_id, new_password):
    try:
        new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE app_user SET password_hash = %s WHERE id = %s
                """, (new_password_hash, user_id))
                conn.commit()
                return True
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def get_session_token(user_id):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_token FROM app_user WHERE id = %s
                """, (user_id, ))
                result = cur.fetchone()
                if result is None: return None
                else: return result[0]
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def ensure_admin():
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT app_user.id FROM app_user
                    JOIN role ON role_id = role.id
                    WHERE role.name = 'admin'
                """)
                if cur.fetchone() is None:
                    add_user(
                        os.getenv("ADMIN_LOGIN"),
                        os.getenv("ADMIN_PASSWORD"),
                        is_admin=True
                    )
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def is_blocked(user_id):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT is_blocked FROM app_user WHERE id = %s
                """, (user_id, ))
                result = cur.fetchone()
                if result is None: return None
                else: return result[0]
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def get_users():
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT app_user.id, app_user.name, role.name, is_blocked
                    FROM app_user
                    JOIN role ON role_id = role.id
                """)
                return cur.fetchall()
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def get_user_id(user_name):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM app_user WHERE name = %s
                """, (user_name, ))
                user_id = cur.fetchone()
                if user_id is None: return None
                else: return user_id[0]
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def get_user_name(user_id):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT name FROM app_user WHERE id = %s
                """, (user_id, ))
                user_name = cur.fetchone()
                if user_name is None: return None
                else: return user_name[0]
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def user_exists(user_id):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM app_user WHERE id = %s
                """, (user_id, ))
                result = cur.fetchone()
                if result is None: return False
                else: return True
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def get_role(user_id):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT role.name FROM role
                    JOIN app_user ON role_id = role.id
                    WHERE app_user.id = %s
                """, (user_id, ))
                result = cur.fetchone()
                if result is None: return None
                else: return result[0]
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")