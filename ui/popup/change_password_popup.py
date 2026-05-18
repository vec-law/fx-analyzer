import requests
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import pyqtSignal
from ui.utils import show_message
from dotenv import load_dotenv

load_dotenv()

class ChangePasswordPopup(QWidget):
    password_changed = pyqtSignal()

    def __init__(self, user_id, session_token):
        super().__init__()
        self.user_id = user_id
        self.session_token = session_token
        self.api_url = os.getenv("API_URL")
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

        self.save_password_button.clicked.connect(self.on_save_password)

    def on_save_password(self):
        try:
            show_message(self.save_password_message, "")

            new_password = self.new_password_input.text()
            repeated_password = self.repeat_password_input.text()

            response = requests.post(
                self.api_url + "/auth/change-password",
                json={
                    "user_id": self.user_id,
                    "new_password": new_password,
                    "repeated_password": repeated_password,
                    "session_token": str(self.session_token)
                }
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                self.new_password_input.clear()
                self.repeat_password_input.clear()

            self.close()

        except requests.exceptions.ConnectionError:
                show_message(self.save_password_message, "Nie można połączyć się z serwerem")
        except Exception as e:
            show_message(self.save_password_message, str(e))
