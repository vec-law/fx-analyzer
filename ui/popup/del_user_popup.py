import requests
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal
from ui.utils import show_message
from dotenv import load_dotenv

load_dotenv()

class DelUserPopup(QWidget):
    user_deleted = pyqtSignal()

    def __init__(self, user_id, session_token, target_user_id):
        super().__init__()

        self.api_url = os.getenv("API_URL")
        self.user_id = user_id
        self.session_token = session_token
        self.target_user_id = target_user_id

        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Usuwanie użytkownika")
        self.setFixedWidth(300)
        
        layout = QVBoxLayout()

        self.confirm_label = QLabel(
            f"Czy na pewno chcesz usunąć użytkownika?"
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

            response = requests.delete(
                self.api_url + "/users",
                headers={
                    "Authorization": f"Bearer {str(self.session_token)}",
                    "X-User-ID": f"{self.user_id}",
                    "X-Target-User-ID": f"{self.target_user_id}"
                }
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                self.user_deleted.emit()

            self.close()

        except Exception as e:
            self.del_user_message.show()
            show_message(self.del_user_message, str(e))
