import sys
import os
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from db.manager import DatabaseManager
from ui.gui import GUI

load_dotenv()

def main():
    app = QApplication(sys.argv)

    db_config = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT"))
    }

    db_manager = DatabaseManager(db_config)

    gui = GUI(db_manager)
    gui.show()

    sys.exit(app.exec())
