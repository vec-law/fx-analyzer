from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel
from src.ui.popup.change_password_popup import ChangePasswordPopup
from src.ui.utils import show_message
from PyQt6.QtCore import pyqtSignal
from src.database_manager import DatabaseManager

class LoginPanel(QWidget):
    user_logged_in = pyqtSignal(int, object)
    user_logged_out = pyqtSignal()

    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db_manager = db_manager
        
        self.user_id = None
        self.session_token = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()

        login_label = QLabel("LOGOWANIE")
        layout.addWidget(login_label)

        user_name_label = QLabel("Nazwa użytkownika:")
        layout.addWidget(user_name_label)

        self.user_name_input = QLineEdit()
        layout.addWidget(self.user_name_input)

        password_label = QLabel("Hasło:")
        layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_input)

        self.login_button = QPushButton("Zaloguj")
        layout.addWidget(self.login_button)

        self.logout_button = QPushButton("Wyloguj")
        layout.addWidget(self.logout_button)
        self.logout_button.setEnabled(False)

        self.change_password_button = QPushButton("Zmień hasło")
        layout.addWidget(self.change_password_button)
        self.change_password_button.setEnabled(False)

        self.login_message = QLabel("")
        self.login_message.setWordWrap(True)
        layout.addWidget(self.login_message)

        layout.addStretch()
        self.setFixedWidth(300)
        self.setLayout(layout)

        self.login_button.clicked.connect(self.on_login)
        self.logout_button.clicked.connect(self.on_logout)
        self.change_password_button.clicked.connect(self.on_change_password)

    def on_login(self):
        try:
            show_message(self.login_message, "")

            user_name = self.user_name_input.text()
            password = self.password_input.text()

            if not user_name or not password:
                raise ValueError("Żadne z pól (login, hasło) nie może być puste")

            self.user_id, self.session_token = self.db_manager.login_user(user_name, password)

            self.user_name_input.setEnabled(False)
            self.password_input.setEnabled(False)
            self.login_button.setEnabled(False)
            self.logout_button.setEnabled(True)
            self.change_password_button.setEnabled(True)
            
            show_message(self.login_message, f"Zalogowano użytkownika {user_name}", True)

            self.user_logged_in.emit(self.user_id, self.session_token)

        except Exception as e:
            show_message(self.login_message, str(e))

    def on_logout(self):
        try:
            show_message(self.login_message, "")

            user_name = self.user_name_input.text()

            self.db_manager.logout_user(user_name)

            self.user_id, self.session_token = None, None

            self.user_name_input.setEnabled(True)
            self.password_input.setEnabled(True)
            self.login_button.setEnabled(True)
            self.logout_button.setEnabled(False)
            self.change_password_button.setEnabled(False)

            self.password_input.clear()

            show_message(self.login_message, f"Wylogowano użytkownika {user_name}", True)

            self.user_logged_out.emit()

        except Exception as e:
            show_message(self.login_message, str(e))

    def on_change_password(self):
        self.change_password_popup = ChangePasswordPopup(self.user_id, self.db_manager)
        self.change_password_popup.password_changed.connect(self.on_password_changed)
        self.change_password_popup.show()

    def on_password_changed(self):
        show_message(self.login_message, "Zmieniono hasło użytkownika", True)
