from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QTextEdit, QTableWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFormLayout,
    QTableWidgetItem
)
from src.worker.simulation_worker import SimulationWorker


class SimulationTab(QWidget):
    PARAM_MAP = {
        "Liczba próbek": "samples_simulation",
        "Liczba próbek przewidywanych": "predicted_samples",
        "Strategie": "strategies"
    }

    def __init__(self, db_manager, tab_widget=None):
        super().__init__()
        self.db_manager = db_manager
        self.tab_widget = tab_widget
        self.last_clicked_training_uuid = None
        self.last_clicked_sim_uuid = None
        self.thread = None
        self.worker = None
        self.init_ui()
        self.init_actions()

    def showEvent(self, event):
        super().showEvent(event)
        self.on_load_completed_trainings()
        self.fill_sim_table()

    def init_ui(self):
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.add_sim_btn = QPushButton("Dodaj symulację")
        self.remove_sim_btn = QPushButton("Usuń symulację")
        self.load_params_btn = QPushButton("Wczytaj parametry")
        self.run_sim_btn = QPushButton("Uruchom symulację")
        self.stop_sim_btn = QPushButton("Zatrzymaj symulację")
        self.clear_console_btn = QPushButton("Wyczyść konsolę")

        self.buttons = [
            self.add_sim_btn, self.remove_sim_btn, self.load_params_btn,
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

        right_layout = QVBoxLayout()
        tables_layout = QHBoxLayout()

        self.source_table = QTableWidget()
        self.source_table.setColumnCount(4)
        self.source_table.setHorizontalHeaderLabels(["UUID Treningu", "Instrument", "Interwał", "Utworzono"])
        self.source_table.setSelectionBehavior(self.source_table.SelectionBehavior.SelectRows)
        self.source_table.setSelectionMode(self.source_table.SelectionMode.SingleSelection)

        self.sim_table = QTableWidget()
        self.sim_table.setColumnCount(4)
        self.sim_table.setHorizontalHeaderLabels(["UUID Symulacji", "UUID Treningu", "Status", "Utworzono"])
        self.sim_table.setSelectionBehavior(self.sim_table.SelectionBehavior.SelectRows)
        self.sim_table.setSelectionMode(self.sim_table.SelectionMode.SingleSelection)

        tables_layout.addWidget(self.source_table, 2)
        tables_layout.addWidget(self.sim_table, 3)

        self.console = QTextEdit()
        self.console.setReadOnly(True)

        right_layout.addLayout(tables_layout, 1) 
        right_layout.addWidget(self.console, 1)  

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 5)
        self.setLayout(main_layout)

    def init_actions(self):
        self.clear_console_btn.clicked.connect(lambda: self.console.clear())
        self.source_table.cellClicked.connect(self.on_source_table_clicked)
        self.sim_table.cellClicked.connect(self.on_sim_table_clicked)
        self.add_sim_btn.clicked.connect(self.on_add_simulation)
        self.remove_sim_btn.clicked.connect(self.on_remove_simulation)
        self.load_params_btn.clicked.connect(self.on_load_params_to_fields)
        self.run_sim_btn.clicked.connect(self.on_run_simulation)
        self.stop_sim_btn.clicked.connect(self.on_stop_simulation)

    def toggle_ui_lock(self, is_running: bool):
        if self.tab_widget:
            self.tab_widget.tabBar().setEnabled(not is_running)
        
        self.add_sim_btn.setEnabled(not is_running)
        self.remove_sim_btn.setEnabled(not is_running)
        self.run_sim_btn.setEnabled(not is_running)

    def on_run_simulation(self):
        if not self.last_clicked_sim_uuid:
            self.log_to_console("Błąd: Nie wybrano symulacji.")
            return

        if self.thread is not None and self.thread.isRunning():
            return

        self.toggle_ui_lock(True)

        self.thread = QThread()
        self.worker = SimulationWorker(self.db_manager, self.last_clicked_sim_uuid)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log_signal.connect(self.log_to_console)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: setattr(self, 'thread', None))
        self.thread.finished.connect(lambda: setattr(self, 'worker', None))
        self.thread.finished.connect(self.fill_sim_table)
        self.thread.finished.connect(lambda: self.toggle_ui_lock(False))

        self.thread.start()
        self.log_to_console(f"Uruchomiono symulację: {self.last_clicked_sim_uuid}")

    def on_stop_simulation(self):
        if self.worker:
            self.worker.stop()
            self.log_to_console("Zatrzymywanie symulacji...")

    def on_load_params_to_fields(self):
        if not self.last_clicked_sim_uuid:
            self.log_to_console("Nie wybrano symulacji do wczytania parametrów.")
            return
        
        try:
            config = self.db_manager.get_simulation_config(self.last_clicked_sim_uuid)
            if config:
                self.param_fields["samples_simulation"].setText(str(config.get("samples_simulation", "")))
                self.param_fields["predicted_samples"].setText(str(config.get("predicted_samples", "")))
                self.param_fields["strategies"].setText(", ".join(config.get("strategies", [])))
                self.log_to_console(f"Wczytano parametry symulacji: {self.last_clicked_sim_uuid}")
        except Exception as e:
            self.log_to_console(f"Błąd wczytywania parametrów: {e}")

    def on_load_completed_trainings(self):
        try:
            all_tasks = self.db_manager.get_training_jobs()
            completed = [t for t in all_tasks if str(t.get("status")).lower() == 'completed']
            self.fill_source_table(completed)
        except Exception as e:
            self.log_to_console(f"Błąd wczytywania treningów: {e}")

    def fill_source_table(self, tasks: list):
        self.source_table.setRowCount(len(tasks))
        for row, t in enumerate(tasks):
            self.source_table.setItem(row, 0, QTableWidgetItem(str(t.get("job_uuid", ""))))
            self.source_table.setItem(row, 1, QTableWidgetItem(str(t.get("instrument", ""))))
            self.source_table.setItem(row, 2, QTableWidgetItem(str(t.get("timeframe_name", ""))))
            dt = t.get("created_at")
            self.source_table.setItem(row, 3, QTableWidgetItem(dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""))
        
        if len(tasks) > 0:
            self.last_clicked_training_uuid = self.source_table.item(0, 0).text()
            self.source_table.selectRow(0)
        else:
            self.last_clicked_training_uuid = None
            
        self.source_table.resizeColumnsToContents()

    def on_source_table_clicked(self, row, column):
        item = self.source_table.item(row, 0)
        if item:
            self.last_clicked_training_uuid = item.text()

    def on_sim_table_clicked(self, row, column):
        item = self.sim_table.item(row, 0)
        if item:
            self.last_clicked_sim_uuid = item.text()

    def on_add_simulation(self):
        if not self.last_clicked_training_uuid:
            self.log_to_console("Błąd: Nie zaznaczono ukończonego treningu w lewej tabeli.")
            return
        try:
            vals = {p: self.param_fields[p].text().strip() for p in self.PARAM_MAP.values()}
            for key, val in vals.items():
                if not val:
                    raise ValueError(f"Pole {key} nie może być puste.")

            self.db_manager.add_simulation_job(
                training_uuid=self.last_clicked_training_uuid,
                samples_simulation=int(vals["samples_simulation"]),
                predicted_samples=int(vals["predicted_samples"]),
                strategies=[s.strip() for s in vals["strategies"].split(",") if s.strip()]
            )
            self.log_to_console("Dodano nową symulację.")
            self.fill_sim_table()
        except Exception as e:
            self.log_to_console(f"Błąd dodawania symulacji: {e}")

    def on_remove_simulation(self):
        if not self.last_clicked_sim_uuid:
            return
        try:
            self.db_manager.del_simulation_job(self.last_clicked_sim_uuid)
            self.log_to_console(f"Usunięto symulację: {self.last_clicked_sim_uuid}")
            self.last_clicked_sim_uuid = None
            self.fill_sim_table()
        except Exception as e:
            self.log_to_console(f"Błąd usuwania: {e}")

    def fill_sim_table(self):
        try:
            simulations = self.db_manager.get_simulations()
            self.sim_table.setRowCount(len(simulations))
            
            for row, sim in enumerate(simulations):
                curr_uuid = str(sim.get("sim_uuid", ""))
                
                self.sim_table.setItem(row, 0, QTableWidgetItem(curr_uuid))
                self.sim_table.setItem(row, 1, QTableWidgetItem(str(sim.get("training_job_uuid", ""))))
                self.sim_table.setItem(row, 2, QTableWidgetItem(str(sim.get("status", ""))))
                dt = sim.get("created_at")
                self.sim_table.setItem(row, 3, QTableWidgetItem(dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""))

            if simulations and len(simulations) > 0:
                first_uuid = self.sim_table.item(0, 0).text()
                self.last_clicked_sim_uuid = first_uuid
                self.sim_table.selectRow(0)
            else:
                self.last_clicked_sim_uuid = None

            self.sim_table.resizeColumnsToContents()
        except Exception as e:
            self.log_to_console(f"Błąd odświeżania tabeli symulacji: {e}")

    def log_to_console(self, message: str):
        self.console.append(message)