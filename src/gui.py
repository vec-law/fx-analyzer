import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget
from trening_tab import TreningTab
from simulation_tab import SimulationTab

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()
        tabs.addTab(TreningTab(), "Trening")
        tabs.addTab(SimulationTab(), "Symulacja")

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

        self.setWindowTitle("fx-analyzer")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec())
