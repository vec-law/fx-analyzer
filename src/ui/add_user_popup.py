from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt6.QtCore import pyqtSignal
from src.database_manager import DatabaseManager
from src.ui.utils import show_message

class AddUserPopup(QWidget):
    user_added = pyqtSignal()

    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Dodawanie użytkownika")
        self.setFixedWidth(300)
        self.setMinimumHeight(200)
        
        layout = QVBoxLayout()

        user_name_label = QLabel("Nazwa użytkownika:")
        layout.addWidget(user_name_label)

        self.user_name_input = QLineEdit()
        layout.addWidget(self.user_name_input)

        self.password_label = QLabel("Hasło:")
        layout.addWidget(self.password_label)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_input)

        self.repeat_password_label = QLabel("Powtórz hasło:")
        layout.addWidget(self.repeat_password_label)

        self.repeat_password_input = QLineEdit()
        self.repeat_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.repeat_password_input)

        self.is_admin_checkbox = QCheckBox("Administrator")
        layout.addWidget(self.is_admin_checkbox)
        
        self.add_user_button = QPushButton("Dodaj")
        layout.addWidget(self.add_user_button)

        self.add_user_message = QLabel("")
        self.add_user_message.setWordWrap(True)
        layout.addWidget(self.add_user_message)

        self.setLayout(layout)

        self.add_user_button.clicked.connect(self.handle_add_user)

    def handle_add_user(self):
        try:
            show_message(self.add_user_message, "")

            password = self.password_input.text()
            repeated_password = self.repeat_password_input.text()

            if not password or not repeated_password or password != repeated_password:
                raise ValueError("Hasła nie mogą być puste i muszą być takie same")
            
            self.password_input.clear()
            self.repeat_password_input.clear()
            
            self.db_manager.add_user(self.user_name_input.text(), password, self.is_admin_checkbox.isChecked())

            self.user_added.emit()
            self.close()

        except Exception as e:
            show_message(self.add_user_message, str(e))
