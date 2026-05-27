import sys
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from gui.gui import GUI

load_dotenv()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec())
