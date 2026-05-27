import requests
import os
from uuid import UUID
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel
from gui.popup.change_password_popup import ChangePasswordPopup
from gui.utils import show_message
from PyQt6.QtCore import pyqtSignal
from dotenv import load_dotenv

load_dotenv()

class LoginPanel(QWidget):
    user_logged_in = pyqtSignal(int, object)
    user_logged_out = pyqtSignal()

    def __init__(self):
        super().__init__()
       
        self.user_id = None
        self.session_token = None
        self.api_url = os.getenv("API_URL")
        
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

            response = requests.post(
                self.api_url + "/auth/login",
                json={"user_name": user_name, "password": password}
            )
            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                self.user_id = response["user_id"]
                self.session_token = UUID(response["session_token"])
            
            self.user_name_input.setEnabled(False)
            self.password_input.setEnabled(False)
            self.login_button.setEnabled(False)
            self.logout_button.setEnabled(True)
            self.change_password_button.setEnabled(True)

            self.user_logged_in.emit(self.user_id, self.session_token)

        except requests.exceptions.ConnectionError:
            show_message(self.login_message, "Nie można połączyć się z serwerem")
        except Exception as e:
            show_message(self.login_message, str(e))

    def on_logout(self):
        try:
            show_message(self.login_message, "")

            self.user_name_input.setEnabled(True)
            self.password_input.setEnabled(True)
            self.login_button.setEnabled(True)
            self.logout_button.setEnabled(False)
            self.change_password_button.setEnabled(False)

            self.password_input.clear()

            self.user_logged_out.emit()

            response = requests.delete(
                self.api_url + "/auth/logout",
                headers={
                    "Authorization": f"Bearer {str(self.session_token)}",
                    "X-User-ID": f"{self.user_id}"
                }
            )
            if response.status_code != 200:
                raise ValueError(response.json()["detail"])
            self.user_id, self.session_token = None, None

        except requests.exceptions.ConnectionError:
            show_message(self.login_message, "Nie można połączyć się z serwerem")
        except Exception as e:
            show_message(self.login_message, str(e))

    def on_change_password(self):
        self.change_password_popup = ChangePasswordPopup(self.user_id, self.session_token)
        self.change_password_popup.show()
