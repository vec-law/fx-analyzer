import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget

def main():
    app = QApplication(sys.argv)

    db_manager = DatabaseManager(config)
    window = MainWindow(db_manager)

    window.show()
    sys.exit(app.exec())

__version__ = "2.0.0-alpha"

import sys
from PyQt6.QtWidgets import QApplication
from src.user_interface import UserInterface 
from src.utils import clear_console

def main():
    clear_console()
    print(f"fx-analyzer v{__version__}")

    app = QApplication(sys.argv)

    db_config = {
        "dbname": "fx_analyzer_db",
        "user": "postgres",
        "password": "1111",
        "host": "localhost",
        "port": 5432
    }
    
    ui = UserInterface(db_config)
    ui.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()