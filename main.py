import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget

def main():
    app = QApplication(sys.argv)

    db_manager = DatabaseManager(config)
    window = MainWindow(db_manager)

    window.show()
    sys.exit(app.exec())
