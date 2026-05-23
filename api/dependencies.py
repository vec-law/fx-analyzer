import os
from dotenv import load_dotenv
from db.manager import DatabaseManager

load_dotenv()

db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT"))
}

db_manager = DatabaseManager(db_config)

def get_db_manager():
    return db_manager
