from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from src.ui.training_tab import TrainingTab
from src.ui.simulation_tab import SimulationTab

# /* Zaktualizowana klasa GUI przekazująca referencję do zakładek do obu modułów */

class GUI(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()

    def init_ui(self):
        # 1. Tworzymy widget zakładek
        self.tabs = QTabWidget()

        # 2. Tworzymy instancje zakładek
        # Przekazujemy self.tabs do OBU zakładek, aby każda mogła blokować nawigację
        # // ZMIANA: Dodano self.tabs do TrainingTab
        self.training_tab = TrainingTab(self.db_manager, self.tabs)
        self.simulation_tab = SimulationTab(self.db_manager, self.tabs)

        # 3. Dodajemy je do widgetu
        self.tabs.addTab(self.training_tab, "Trening")
        self.tabs.addTab(self.simulation_tab, "Symulacja")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.setWindowTitle("fx-analyzer")