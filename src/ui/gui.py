from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from src.ui.training_tab import TrainingTab
from src.ui.simulation_tab import SimulationTab


class GUI(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()

        tabs.addTab(TrainingTab(self.db_manager), "Trening")
        tabs.addTab(SimulationTab(self.db_manager), "Symulacja")

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

        self.setWindowTitle("fx-analyzer")
