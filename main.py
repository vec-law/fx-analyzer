__version__ = "v1.1.0-beta"

import sys
import os
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from src.database_manager import DatabaseManager
from src.ui.gui import GUI

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

if __name__ == "__main__":
    main()