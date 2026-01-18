from PyQt6.QtWidgets import (
    QWidget, QPushButton, QTextEdit, QTableWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFormLayout
)

class TreningTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()

    def init_ui(self):
        # ===== Lewy panel =====
        left_layout = QVBoxLayout()

        self.load_tasks_btn = QPushButton("Wczytaj zadania")
        self.add_task_btn   = QPushButton("Dodaj zadanie")
        self.remove_task_btn= QPushButton("Usuń zadanie")
        self.load_params_btn= QPushButton("Wczytaj parametry")
        self.run_task_btn   = QPushButton("Uruchom zadanie")
        self.stop_task_btn  = QPushButton("Zatrzymaj zadanie")

        for btn in [
            self.load_tasks_btn, self.add_task_btn, self.remove_task_btn,
            self.load_params_btn, self.run_task_btn, self.stop_task_btn
        ]:
            left_layout.addWidget(btn)

        left_layout.addWidget(QLabel("<b>Parametry</b>"))

        self.param_fields = {}
        params_layout = QFormLayout()
        for name in [
            "epochs", "samples_limit", "architecture", "features",
            "targets", "data_source", "train_ratio", "noise_ratio"
        ]:
            field = QLineEdit()
            self.param_fields[name] = field
            params_layout.addRow(name, field)

        left_layout.addLayout(params_layout)
        left_layout.addStretch()

        # ===== Prawy panel =====
        right_layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["UUID"])

        self.console = QTextEdit()
        self.console.setReadOnly(True)

        right_layout.addWidget(self.table)
        right_layout.addWidget(self.console)

        # ===== Layout zakładki =====
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.setLayout(main_layout)
