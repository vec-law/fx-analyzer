import requests
import os
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QTabWidget
from ui.tab.user_management_tab import UserManagementTab
from ui.tab.training_tab import TrainingTab
from ui.tab.prediction_tab import PredictionTab
from ui.login_panel import LoginPanel
from api.db_manager import DBManager
from ui.utils import show_message
from dotenv import load_dotenv

load_dotenv()

class GUI(QWidget):
    def __init__(self, db_manager: DBManager):
        super().__init__()
        self.db_manager = db_manager
        self.api_url = os.getenv("API_URL")
        self.init_ui()

    def init_ui(self):
        self.login_panel = LoginPanel()

        self.tabs = QTabWidget()

        self.user_management_tab = UserManagementTab(self.db_manager)
        self.training_tab = TrainingTab(self.db_manager, self.tabs)
        self.prediction_tab = PredictionTab(self.db_manager, self.tabs)

        self.tabs.addTab(self.user_management_tab, "Zarządzanie użytkownikami")
        self.tabs.addTab(self.training_tab, "Trening modeli")
        self.tabs.addTab(self.prediction_tab, "Predykcja wartości docelowych")

        self.tabs.setTabVisible(0, False)
        self.tabs.setTabVisible(1, False)
        self.tabs.setTabVisible(2, False)

        layout = QHBoxLayout()
        layout.addWidget(self.login_panel)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.setWindowTitle("fx-analyzer")
        self.resize(1200, 700)

        self.login_panel.user_logged_in.connect(self.on_user_logged_in)
        self.login_panel.user_logged_out.connect(self.on_user_logged_out)

        self.login_panel.user_logged_in.connect(self.user_management_tab.set_session)
        self.login_panel.user_logged_out.connect(self.user_management_tab.clear_session)

        self.login_panel.user_logged_in.connect(self.training_tab.set_session)
        self.login_panel.user_logged_out.connect(self.training_tab.clear_session)

    def on_user_logged_in(self, user_id, session_token):
        try:
            response = requests.get(
                self.api_url + "/users/role",
                headers={
                    "Authorization": f"Bearer {str(session_token)}",
                    "X-User-ID": f"{user_id}"
                }
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                role_name = response["role_name"]

            if role_name == 'admin':
                self.tabs.setTabVisible(0, True)
                self.tabs.setTabVisible(1, False)
                self.tabs.setTabVisible(2, False)
                
            elif role_name == 'user':
                self.tabs.setTabVisible(0, False)
                self.tabs.setTabVisible(1, True)
                self.tabs.setTabVisible(2, True)

        except requests.exceptions.ConnectionError:
            show_message(self.login_panel.login_message, "Nie można połączyć się z serwerem")
        except Exception as e:
            show_message(self.login_panel.login_message, str(e))

    def on_user_logged_out(self):
        self.tabs.setTabVisible(0, False)
        self.tabs.setTabVisible(1, False)
        self.tabs.setTabVisible(2, False)
