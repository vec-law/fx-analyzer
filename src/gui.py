from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from src.trening_tab import TreningTab
from src.simulation_tab import SimulationTab


class GUI(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()

        # przekazujemy db_manager dalej, GUI nic z nim nie robi
        tabs.addTab(TreningTab(self.db_manager), "Trening")
        tabs.addTab(SimulationTab(self.db_manager), "Symulacja")

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

        self.setWindowTitle("fx-analyzer")
