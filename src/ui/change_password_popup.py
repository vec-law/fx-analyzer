from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import pyqtSignal
from src.database_manager import DatabaseManager
from src.ui.utils import show_message

class ChangePasswordPopup(QWidget):
    password_changed = pyqtSignal()

    def __init__(self, user_id, session_token, required_role, db_manager: DatabaseManager):
        super().__init__()
        self.user_id = user_id
        self.session_token = session_token
        self.required_role = required_role
        self.db_manager = db_manager
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Zmiana hasła")
        self.setFixedWidth(300)
        self.setMinimumHeight(200)
        
        layout = QVBoxLayout()

        self.new_password_label = QLabel("Nowe hasło:")
        layout.addWidget(self.new_password_label)

        self.new_password_input = QLineEdit()
        self.new_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.new_password_input)

        self.repeat_password_label = QLabel("Powtórz hasło:")
        layout.addWidget(self.repeat_password_label)

        self.repeat_password_input = QLineEdit()
        self.repeat_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.repeat_password_input)
        
        self.save_password_button = QPushButton("Zapisz")
        layout.addWidget(self.save_password_button)

        self.save_password_message = QLabel("")
        self.save_password_message.setWordWrap(True)
        layout.addWidget(self.save_password_message)

        self.setLayout(layout)

        self.save_password_button.clicked.connect(self.handle_save_password)

    def handle_save_password(self):
        try:
            show_message(self.save_password_message, "")

            new_password = self.new_password_input.text()
            repeated_password = self.repeat_password_input.text()

            self.db_manager.validate_access(self.user_id, self.session_token, self.required_role)

            if not new_password or not repeated_password or new_password != repeated_password:
                raise ValueError("Hasła nie mogą być puste i muszą być takie same")
            
            self.new_password_input.clear()
            self.repeat_password_input.clear()
            
            self.db_manager.change_password(self.user_id, new_password)

            self.password_changed.emit()
            self.close()

        except Exception as e:
            show_message(self.save_password_message, str(e))
