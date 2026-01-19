from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class SimulationTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Symulacja – do uzupełnienia"))
        self.setLayout(layout)
