from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from src.ui.training_tab import TrainingTab
from src.ui.simulation_tab import SimulationTab

class GUI(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()

        self.training_tab = TrainingTab(self.db_manager, self.tabs)
        self.simulation_tab = SimulationTab(self.db_manager, self.tabs)

        self.tabs.addTab(self.training_tab, "Trening")
        self.tabs.addTab(self.simulation_tab, "Symulacja")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.setWindowTitle("fx-analyzer")