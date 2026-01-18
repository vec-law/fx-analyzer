from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class SimulationTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Symulacja – do uzupełnienia"))
        self.setLayout(layout)
