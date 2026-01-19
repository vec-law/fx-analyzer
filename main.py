import sys
from PyQt6.QtWidgets import QApplication
from src.database_manager import DatabaseManager
from src.ui.gui import GUI

def main():
    app = QApplication(sys.argv)

    db_config = {
        "dbname": "fx_analyzer_db",
        "user": "postgres",
        "password": "1111",
        "host": "localhost",
        "port": 5432
    }

    db_manager = DatabaseManager(db_config)
    gui = GUI(db_manager)

    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()