from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal
from src.database_manager import DatabaseManager
from src.ui.utils import show_message

class DelUserPopup(QWidget):
    user_deleted = pyqtSignal()

    def __init__(self, user_id, db_manager: DatabaseManager):
        super().__init__()
        
        self.db_manager = db_manager
        self.user_id = user_id

        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Usuwanie użytkownika")
        self.setFixedWidth(300)
        
        layout = QVBoxLayout()

        self.confirm_label = QLabel(
            f"Czy na pewno chcesz usunąć użytkownika {self.db_manager.get_user_name(self.user_id)}?"
        )
        layout.addWidget(self.confirm_label)

        buttons_layout = QHBoxLayout()
        
        self.yes_button = QPushButton("Tak")
        buttons_layout.addWidget(self.yes_button)

        self.no_button = QPushButton("Nie")
        buttons_layout.addWidget(self.no_button)
        layout.addLayout(buttons_layout)

        self.del_user_message = QLabel("")
        self.del_user_message.setWordWrap(True)
        self.del_user_message.hide()
        layout.addWidget(self.del_user_message)

        self.setLayout(layout)

        self.yes_button.clicked.connect(self.on_del_user)
        self.no_button.clicked.connect(self.close)

    def on_del_user(self):
        try:
            show_message(self.del_user_message, "")

            self.db_manager.del_user(self.user_id)

            self.user_deleted.emit()
            self.close()

        except Exception as e:
            self.del_user_message.show()
            show_message(self.del_user_message, str(e))
