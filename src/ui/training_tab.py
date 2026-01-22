from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QTextEdit, QTableWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFormLayout,
    QTableWidgetItem
)
from src.worker.training_worker import TrainingWorker

class TrainingTab(QWidget):
    PARAM_MAP = {
        "Instrument": "instrument_name", "Interwał": "timeframe_name",
        "Źródło danych": "data_source", "Limit próbek": "samples_limit",
        "Współczynnik podziału": "train_ratio", "Losowość": "seed",
        "Liczba epok": "epochs", "Współczynnik szumu": "train_noise",
        "Współczynnik uczenia": "learning_rate", "Wartości docelowe": "targets",
        "Cechy": "features", "Architektury modeli": "architectures"
    }

    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.last_clicked_uuid = None
        self.thread = None
        self.worker = None
        self.init_ui()
        self.init_actions()

    def init_ui(self):
        left_layout = QVBoxLayout()

        self.load_tasks_btn = QPushButton("Wczytaj zadania")
        self.add_task_btn = QPushButton("Dodaj zadanie")
        self.remove_task_btn = QPushButton("Usuń zadanie")
        self.load_params_btn = QPushButton("Wczytaj parametry")
        self.update_params_btn = QPushButton("Zmień parametry")
        self.clear_console_btn = QPushButton("Wyczyść konsolę")
        self.run_task_btn = QPushButton("Uruchom zadanie")
        self.stop_task_btn = QPushButton("Zatrzymaj zadanie")

        self.buttons = [
            self.load_tasks_btn, self.add_task_btn, self.remove_task_btn,
            self.load_params_btn, self.update_params_btn, self.clear_console_btn,
            self.run_task_btn, self.stop_task_btn
        ]
        for btn in self.buttons: left_layout.addWidget(btn)

        left_layout.addWidget(QLabel("<b>Parametry</b>"))
        self.param_fields = {}
        params_layout = QFormLayout()
        
        for label_pl, param in self.PARAM_MAP.items():
            field = QLineEdit()
            self.param_fields[param] = field
            params_layout.addRow(label_pl, field)

        left_layout.addLayout(params_layout)
        left_layout.addStretch()

        right_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["UUID", "Instrument", "Interwał", "Status", "Utworzono"])
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(self.table.SelectionMode.SingleSelection)
        self.console = QTextEdit()
        self.console.setReadOnly(True)

        right_layout.addWidget(self.table)
        right_layout.addWidget(self.console)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)
        self.setLayout(main_layout)

    def init_actions(self):
        self.load_tasks_btn.clicked.connect(lambda: self.on_load_tasks(show_log=True))
        self.remove_task_btn.clicked.connect(self.on_remove_tasks)
        self.load_params_btn.clicked.connect(self.on_load_params)
        # /* Podpięto akcję aktualizacji */
        self.update_params_btn.clicked.connect(self.on_update_params)
        self.clear_console_btn.clicked.connect(self.on_clear_console)
        self.add_task_btn.clicked.connect(self.on_add_task)
        self.run_task_btn.clicked.connect(self.on_run_task)
        self.stop_task_btn.clicked.connect(self.on_stop_task)
        self.table.cellClicked.connect(self.on_table_cell_clicked)

    def on_load_tasks(self, show_log=True):
        tasks = self.db_manager.get_training_jobs()
        self.fill_tasks_table(tasks)
        if show_log:
            self.console.append("Wczytano zadania")

    def on_remove_tasks(self):
        if not self.last_clicked_uuid:
            return self.console.append("Nie wybrano zadania do usunięcia")

        if self.db_manager.del_training_job(self.last_clicked_uuid):
            self.console.append(f"Usunięto zadanie: {self.last_clicked_uuid}")
            self.on_load_tasks(show_log=False)
            if self.table.rowCount() > 0:
                self.table.selectRow(0)
                self.last_clicked_uuid = self.table.item(0, 0).text()
            else:
                self.last_clicked_uuid = None
        else:
            self.console.append(f"Błąd usuwania: {self.last_clicked_uuid}")

    def on_table_cell_clicked(self, row, column):
        item = self.table.item(row, 0)
        if item: self.last_clicked_uuid = item.text()

    def fill_tasks_table(self, tasks: list[dict]):
        self.table.setRowCount(len(tasks))
        for row, t in enumerate(tasks):
            self.table.setItem(row, 0, QTableWidgetItem(str(t["job_uuid"])))
            self.table.setItem(row, 1, QTableWidgetItem(t["instrument"]))
            self.table.setItem(row, 2, QTableWidgetItem(t["timeframe_name"]))
            self.table.setItem(row, 3, QTableWidgetItem(t["status"]))
            self.table.setItem(row, 4, QTableWidgetItem(t["created_at"].strftime("%Y-%m-%d %H:%M:%S")))
        self.table.resizeColumnsToContents()

    def on_load_params(self):
        if not self.last_clicked_uuid: return self.console.append("Nie wybrano zadania")
        config = self.db_manager.get_training_config(self.last_clicked_uuid)
        
        if not config: return self.console.append("Brak parametrów")

        try:
            self.param_fields["instrument_name"].setText(str(config['instrument']['name']))
            self.param_fields["timeframe_name"].setText(str(config['timeframe']['name']))
            self.param_fields["data_source"].setText(str(config['data_source']))

            self.param_fields["samples_limit"].setText(str(config['parameter_set']['samples_limit']))
            self.param_fields["train_ratio"].setText(str(config['parameter_set']['train_ratio']))
            self.param_fields["seed"].setText(str(config['parameter_set']['seed']))
            self.param_fields["epochs"].setText(str(config['parameter_set']['epochs']))
            self.param_fields["train_noise"].setText(str(config['parameter_set']['train_noise']))
            self.param_fields["learning_rate"].setText(str(config['parameter_set']['learning_rate']))
            
            self.param_fields["targets"].setText(", ".join([f"{target['type']}:{target['shift']}" for target in config['targets']]))
            
            self.param_fields["features"].setText(", ".join([
                f"{feature['type']}:{feature['start_from']}:{feature['stop_at']}:{feature['step']}:{feature['shift']}" 
                for feature in config['features']
            ]))

            self.param_fields["architectures"].setText(", ".join(config['architectures']))

            self.console.append(f"Wczytano parametry zadania: {self.last_clicked_uuid}")
            
        except Exception as e:
            self.console.append(f"Błąd mapowania parametrów: {e}")

    def on_update_params(self):
        if not self.last_clicked_uuid:
            return self.console.append("Nie wybrano zadania do modyfikacji")

        try:
            field_values = {param: self.param_fields[param].text().strip() for param in self.PARAM_MAP.values()}

            config = {
                "instrument": {"name": field_values["instrument_name"]},
                "timeframe": {"name": field_values["timeframe_name"]},
                "data_source": field_values["data_source"],
                "parameter_set": {
                    "samples_limit": int(field_values["samples_limit"]),
                    "train_ratio":   float(field_values["train_ratio"]),
                    "seed":          int(field_values["seed"]),
                    "epochs":        int(field_values["epochs"]),
                    "train_noise":   float(field_values["train_noise"]),
                    "learning_rate": float(field_values["learning_rate"]),
                },
                "targets": [],
                "features": [],
                "architectures": []
            }

            if field_values["targets"]:
                for target in field_values["targets"].split(","):
                    target_parts = target.strip().split(":")
                    config["targets"].append({
                        "type":  target_parts[0].strip(),
                        "shift": int(target_parts[1].strip())
                    })

            if field_values["features"]:
                for feature in field_values["features"].split(","):
                    feature_parts = feature.strip().split(":")
                    config["features"].append({
                        "type":       feature_parts[0].strip(),
                        "start_from": int(feature_parts[1].strip()),
                        "stop_at":    int(feature_parts[2].strip()),
                        "step":       int(feature_parts[3].strip()),
                        "shift":      int(feature_parts[4].strip())
                    })

            if field_values["architectures"]:
                config["architectures"] = [
                    a.strip() for a in field_values["architectures"].split(",") if a.strip()
                ]

            if self.db_manager.update_training_config(self.last_clicked_uuid, config):
                self.console.append(f"Zaktualizowano parametry zadania: {self.last_clicked_uuid}")
                
                # Zapamiętujemy UUID przed odświeżeniem
                current_uuid = self.last_clicked_uuid
                self.on_load_tasks(show_log=False)
                
                # Przywracamy zaznaczenie w tabeli
                for row in range(self.table.rowCount()):
                    item = self.table.item(row, 0)
                    if item and item.text() == current_uuid:
                        self.table.selectRow(row)
                        break
            else:
                self.console.append(f"Błąd aktualizacji zadania: {self.last_clicked_uuid}")

        except (ValueError, IndexError, KeyError):
            self.console.append("Podano nieprawidłowe parametry")

    def on_add_task(self):
        try:
            field_values = {param: self.param_fields[param].text().strip() for param in self.PARAM_MAP.values()}

            config = {
                "instrument": {"name": field_values["instrument_name"]},
                "timeframe": {"name": field_values["timeframe_name"]},
                "data_source": field_values["data_source"],
                "parameter_set": {
                    "samples_limit": int(field_values["samples_limit"]),
                    "train_ratio":   float(field_values["train_ratio"]),
                    "seed":          int(field_values["seed"]),
                    "epochs":        int(field_values["epochs"]),
                    "train_noise":   float(field_values["train_noise"]),
                    "learning_rate": float(field_values["learning_rate"]),
                },
                "targets": [],
                "features": [],
                "architectures": []
            }

            if field_values["targets"]:
                for target in field_values["targets"].split(","):
                    target_parts = target.strip().split(":")
                    config["targets"].append({
                        "type":  target_parts[0].strip(),
                        "shift": int(target_parts[1].strip())
                    })

            if field_values["features"]:
                for feature in field_values["features"].split(","):
                    feature_parts = feature.strip().split(":")
                    config["features"].append({
                        "type":       feature_parts[0].strip(),
                        "start_from": int(feature_parts[1].strip()),
                        "stop_at":    int(feature_parts[2].strip()),
                        "step":       int(feature_parts[3].strip()),
                        "shift":      int(feature_parts[4].strip())
                    })

            if field_values["architectures"]:
                config["architectures"] = [
                    a.strip() for a in field_values["architectures"].split(",") if a.strip()
                ]

            new_uuid = self.db_manager.add_training_job(config)
            if new_uuid:
                self.console.append(f"Dodano zadanie: {new_uuid}")
                self.on_load_tasks(show_log=False)

        except (ValueError, IndexError, KeyError):
            self.console.append("Podano nieprawidłowe parametry")

    def on_clear_console(self):
        self.console.clear()

    def on_run_task(self):
        if not self.last_clicked_uuid: return self.console.append("Nie zaznaczono zadania")
        try:
            if self.thread and self.thread.isRunning():
                return self.console.append("Zadanie już działa")
        except RuntimeError: pass

        self.thread = QThread()
        self.worker = TrainingWorker(self.last_clicked_uuid, self.db_manager)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.destroyed.connect(lambda: setattr(self, 'thread', None))
        self.thread.destroyed.connect(lambda: setattr(self, 'worker', None))

        self.worker.log.connect(self.console.append)
        self.thread.finished.connect(lambda: (self.set_running_ui(False)))
        self.thread.finished.connect(lambda: self.on_load_tasks(show_log=False))
        
        if self.last_clicked_uuid:
            self.db_manager.update_training_status(self.last_clicked_uuid, 'running')
            self.on_load_tasks(show_log=False)

        self.load_tasks_btn.setFocus()
        self.set_running_ui(True)
        self.thread.start()

    def on_stop_task(self):
        if self.worker:
            try:
                self.set_running_ui(False)
                self.worker.stop()
                self.console.append("Zatrzymywanie...")
            except RuntimeError:
                self.worker = self.thread = None

    def set_running_ui(self, running: bool):
        # /* Dodano self.update_params_btn do listy blokowanych przycisków */
        for btn in [self.load_tasks_btn, self.add_task_btn, self.remove_task_btn, 
                    self.load_params_btn, self.update_params_btn, self.run_task_btn]:
            btn.setEnabled(not running)