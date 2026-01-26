from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QTextEdit, QTableWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFormLayout,
    QTableWidgetItem
)

class SimulationTab(QWidget):
    # Aktualizacja mapowania parametrów formularza
    PARAM_MAP = {
        "Liczba próbek": "sample_count",
        "Liczba próbek przewidywanych": "predicted_samples",
        "Strategie": "strategies"
    }

    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.last_clicked_training_uuid = None
        self.last_clicked_sim_uuid = None
        self.thread = None
        self.worker = None
        self.init_ui()
        self.init_actions()
        # Wstępne ładowanie danych przy starcie
        self.on_load_completed_trainings()

    def init_ui(self):
        # Główny układ horyzontalny
        main_layout = QHBoxLayout()

        # --- LEWY PANEL: PRZYCISKI I PARAMETRY ---
        left_layout = QVBoxLayout()
        
        self.add_sim_btn = QPushButton("Dodaj symulację")
        self.remove_sim_btn = QPushButton("Usuń symulację")
        self.run_sim_btn = QPushButton("Uruchom symulację")
        self.stop_sim_btn = QPushButton("Zatrzymaj symulację")
        self.clear_console_btn = QPushButton("Wyczyść konsolę")

        self.buttons = [
            self.add_sim_btn, self.remove_sim_btn,
            self.run_sim_btn, self.stop_sim_btn, self.clear_console_btn
        ]
        for btn in self.buttons:
            left_layout.addWidget(btn)

        left_layout.addWidget(QLabel("<b>Parametry Symulacji</b>"))
        self.param_fields = {}
        params_layout = QFormLayout()
        for label_pl, param in self.PARAM_MAP.items():
            field = QLineEdit()
            self.param_fields[param] = field
            params_layout.addRow(label_pl, field)
        
        left_layout.addLayout(params_layout)
        left_layout.addStretch()

        # --- PRAWY PANEL: TABELE (GÓRA) I KONSOLA (DÓŁ) ---
        right_layout = QVBoxLayout()

        # Kontener na dwie tabele obok siebie
        tables_layout = QHBoxLayout()

        # Tabela 1: Odzwierciedlenie TrainingTab (BEZ statusu)
        self.source_table = QTableWidget()
        self.source_table.setColumnCount(4)
        self.source_table.setHorizontalHeaderLabels([
            "UUID Treningu", "Instrument", "Interwał", "Utworzono"
        ])
        self.source_table.setSelectionBehavior(self.source_table.SelectionBehavior.SelectRows)
        self.source_table.setSelectionMode(self.source_table.SelectionMode.SingleSelection)

        # Tabela 2: Zadania symulacji
        self.sim_table = QTableWidget()
        self.sim_table.setColumnCount(5)
        self.sim_table.setHorizontalHeaderLabels([
            "UUID Symulacji", "UUID Treningu", "Status", "Wynik", "Utworzono"
        ])
        self.sim_table.setSelectionBehavior(self.sim_table.SelectionBehavior.SelectRows)
        self.sim_table.setSelectionMode(self.sim_table.SelectionMode.SingleSelection)

        tables_layout.addWidget(self.source_table)
        tables_layout.addWidget(self.sim_table)

        # Konsola na dole
        self.console = QTextEdit()
        self.console.setReadOnly(True)

        # Proporcja 1:1
        right_layout.addLayout(tables_layout, 1) 
        right_layout.addWidget(self.console, 1)  

        # Złożenie wszystkiego
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 4)
        self.setLayout(main_layout)

    def init_actions(self):
        self.clear_console_btn.clicked.connect(lambda: self.console.clear())
        self.source_table.cellClicked.connect(self.on_source_table_clicked)
        self.sim_table.cellClicked.connect(self.on_sim_table_clicked)

    def showEvent(self, event):
        """Automatyczne odświeżanie przy wejściu w zakładkę."""
        super().showEvent(event)
        self.on_load_completed_trainings()

    def on_load_completed_trainings(self):
        """Pobiera zadania i filtruje zakończone."""
        try:
            all_tasks = self.db_manager.get_training_jobs()
            completed_tasks = [t for t in all_tasks if str(t.get("status")).lower() == 'completed']
            self.fill_source_table(completed_tasks)
        except Exception as e:
            self.log_to_console(f"Błąd ładowania: {e}")

    def fill_source_table(self, tasks: list):
        self.source_table.setRowCount(0)
        self.source_table.setRowCount(len(tasks))
        for row, t in enumerate(tasks):
            self.source_table.setItem(row, 0, QTableWidgetItem(str(t.get("job_uuid", ""))))
            self.source_table.setItem(row, 1, QTableWidgetItem(str(t.get("instrument", ""))))
            self.source_table.setItem(row, 2, QTableWidgetItem(str(t.get("timeframe_name", ""))))
            
            created_at = t.get("created_at")
            created_str = created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else ""
            self.source_table.setItem(row, 3, QTableWidgetItem(created_str))
            
        self.source_table.resizeColumnsToContents()

    def on_source_table_clicked(self, row, column):
        item = self.source_table.item(row, 0)
        if item:
            self.last_clicked_training_uuid = item.text()

    def on_sim_table_clicked(self, row, column):
        item = self.sim_table.item(row, 0)
        if item:
            self.last_clicked_sim_uuid = item.text()

    def log_to_console(self, message: str):
        self.console.append(message)