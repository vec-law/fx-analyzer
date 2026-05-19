import requests
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt6.QtCore import pyqtSignal
from ui.utils import show_message
from dotenv import load_dotenv

load_dotenv()

class AddUserPopup(QWidget):
    user_added = pyqtSignal()

    def __init__(self, user_id, session_token):
        super().__init__()
        self.api_url = os.getenv("API_URL")
        self.user_id = user_id
        self.session_token = session_token
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

        self.add_user_button.clicked.connect(self.on_add_user)

    def on_add_user(self):
        try:
            show_message(self.add_user_message, "")
            
            user_name = self.user_name_input.text()
            password = self.password_input.text()
            repeated_password = self.repeat_password_input.text()
            is_admin = self.is_admin_checkbox.isChecked()

            response = requests.post(
                self.api_url + "/users",
                headers={
                    "Authorization": f"Bearer {str(self.session_token)}",
                    "X-User-ID": f"{self.user_id}"
                },
                json={
                    "user_name": user_name,
                    "password": password,
                    "repeated_password": repeated_password,
                    "is_admin": is_admin
                }
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                self.user_added.emit()
                
            self.close()

        except Exception as e:
            show_message(self.add_user_message, str(e))
