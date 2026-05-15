from PyQt6.QtWidgets import QWidget, QHBoxLayout, QTabWidget
from src.ui.user_management_tab import UserManagementTab
from src.ui.training_tab import TrainingTab
from src.ui.prediction_tab import PredictionTab
from src.ui.login_panel import LoginPanel

class GUI(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()
        self.db_manager.ensure_admin()

    def init_ui(self):
        self.login_panel = LoginPanel(self.db_manager)

        self.tabs = QTabWidget()

        self.admin_tab = UserManagementTab(self.db_manager)
        self.training_tab = TrainingTab(self.db_manager, self.tabs)
        self.prediction_tab = PredictionTab(self.db_manager, self.tabs)

        self.tabs.addTab(self.admin_tab, "Zarządzanie użytkownikami")
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

    def on_user_logged_in(self, role_name):
        if role_name == 'admin':
            self.tabs.setTabVisible(0, True)
            self.tabs.setTabVisible(1, False)
            self.tabs.setTabVisible(2, False)
        elif role_name == 'user':
            self.tabs.setTabVisible(0, False)
            self.tabs.setTabVisible(1, True)
            self.tabs.setTabVisible(2, True)

    def on_user_logged_out(self):
        self.tabs.setTabVisible(0, False)
        self.tabs.setTabVisible(1, False)
        self.tabs.setTabVisible(2, False)