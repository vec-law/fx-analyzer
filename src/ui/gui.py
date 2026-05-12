from PyQt6.QtWidgets import QWidget, QHBoxLayout, QTabWidget
from src.ui.training_tab import TrainingTab
from src.ui.prediction_tab import PredictionTab
from src.ui.login_panel import LoginPanel

class GUI(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()

    def init_ui(self):
        self.login_panel = LoginPanel(self.db_manager)

        self.tabs = QTabWidget()

        self.training_tab = TrainingTab(self.db_manager, self.tabs)
        self.prediction_tab = PredictionTab(self.db_manager, self.tabs)

        self.tabs.addTab(self.training_tab, "Trening modeli")
        self.tabs.addTab(self.prediction_tab, "Predykcja wartości docelowych")

        layout = QHBoxLayout()
        layout.addWidget(self.login_panel)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.setWindowTitle("fx-analyzer")