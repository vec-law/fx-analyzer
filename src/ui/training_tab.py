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
        "Współczynnik uczenia": "learning_rate", "Cechy": "features",
        "Wartości docelowe": "targets", "Architektury modeli": "architectures"
    }

    def log_to_console(self, message: str):
        self.console.append(message)

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
        self.remove_task_btn.clicked.connect(self.on_remove_task)
        self.load_params_btn.clicked.connect(self.on_load_params)
        self.update_params_btn.clicked.connect(self.on_update_params)
        self.clear_console_btn.clicked.connect(self.on_clear_console)
        self.add_task_btn.clicked.connect(self.on_add_task)
        self.run_task_btn.clicked.connect(self.on_run_task)
        self.stop_task_btn.clicked.connect(self.on_stop_task)
        self.table.cellClicked.connect(self.on_table_cell_clicked)

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

    def on_clear_console(self):
        self.console.clear()

    def on_run_task(self):
        if not self.last_clicked_uuid:
            self.log_to_console("Nie zaznaczono zadania")
            return

        if self.thread and self.thread.isRunning():
            self.log_to_console("Zadanie już działa")
            return

        self.thread = QThread()

        self.worker = TrainingWorker(self.last_clicked_uuid, self.db_manager)
        self.worker.log_signal.connect(self.log_to_console)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.destroyed.connect(lambda: setattr(self, 'thread', None))
        self.thread.destroyed.connect(lambda: setattr(self, 'worker', None))

        self.thread.finished.connect(lambda: (self.set_running_ui(False)))
        self.thread.finished.connect(lambda: self.on_load_tasks(show_log=False))

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
            except RuntimeError:
                self.worker = self.thread = None

    def set_running_ui(self, running: bool):
        for btn in [self.load_tasks_btn, self.add_task_btn, self.remove_task_btn, 
                    self.load_params_btn, self.update_params_btn, self.run_task_btn]:
            btn.setEnabled(not running)

    def on_add_task(self):
        try:
            field_values = {param: self.param_fields[param].text().strip() for param in self.PARAM_MAP.values()}

            required_fields = [
                "instrument_name", "timeframe_name", "data_source",
                "samples_limit", "train_ratio", "seed", "epochs",
                "train_noise", "learning_rate", "features", "targets", "architectures"
            ]
            for field in required_fields:
                if not field_values.get(field):
                    raise ValueError(f"Pole '{field}' nie może być puste")

            config = {
                "instrument": {"name": field_values["instrument_name"]},
                "timeframe": {"name": field_values["timeframe_name"]},
                "data_source": field_values["data_source"],
                "parameter_set": {
                    "samples_limit": int(field_values["samples_limit"]),
                    "train_ratio": float(field_values["train_ratio"]),
                    "seed": int(field_values["seed"]),
                    "epochs": int(field_values["epochs"]),
                    "train_noise": float(field_values["train_noise"]),
                    "learning_rate": float(field_values["learning_rate"]),
                },
                "features": [],
                "targets": [],
                "architectures": []
            }

            # Nowy blok parsowania features (typ:param-param:shift):
            for feature in field_values["features"].split(","):
                f_str = feature.strip()
                if not f_str: continue
                
                parts = f_str.split(":")
                if len(parts) != 3:
                    raise ValueError(f"Niepoprawny format feature: {f_str}. Oczekiwano typ:parametry:shift")
                
                # Rozbicie okresów po myślniku, np. "12-26-9" -> [12, 26, 9]
                periods = [int(p.strip()) for p in parts[1].split("-") if p.strip()]
                
                config["features"].append({
                    "feature_type": parts[0].strip(),
                    "feature_periods": periods,
                    "shift": int(parts[2].strip())
                })

            for target in field_values["targets"].split(","):
                parts = target.strip().split(":")
                if len(parts) != 2:
                    raise ValueError(f"Niepoprawny format target: {target}")
                config["targets"].append({
                    "base_column": parts[0].strip(),
                    "shift": int(parts[1].strip())
                })

            for architecture in field_values["architectures"].split(","):
                config["architectures"].append(architecture.strip())

            new_uuid = self.db_manager.add_training_job(config)
            self.log_to_console(f"Dodano zadanie: {new_uuid}")
            self.on_load_tasks(show_log=False)

        except (ValueError, IndexError, KeyError) as e:
            self.log_to_console(f"Błąd danych: {e}")
        except Exception as e:
            self.log_to_console(f"{e}")


    def on_load_params(self):
        if not self.last_clicked_uuid:
            self.log_to_console("Nie wybrano zadania")
            return
        try:
            config = self.db_manager.get_training_config(self.last_clicked_uuid)

            self.param_fields["instrument_name"].setText(config["instrument"]["name"])
            self.param_fields["timeframe_name"].setText(config["timeframe"]["name"])
            self.param_fields["data_source"].setText(config["data_source"])

            ps = config["parameter_set"]
            self.param_fields["samples_limit"].setText(str(ps["samples_limit"]))
            self.param_fields["train_ratio"].setText(str(ps["train_ratio"]))
            self.param_fields["seed"].setText(str(ps["seed"]))
            self.param_fields["epochs"].setText(str(ps["epochs"]))
            self.param_fields["train_noise"].setText(str(ps["train_noise"]))
            self.param_fields["learning_rate"].setText(str(ps["learning_rate"]))

            # Nowy blok renderowania features (typ:p1-p2:shift):
            feature_strings = []
            for f in config["features"]:
                # Łączymy listę okresów myślnikiem, np. [12, 26, 9] -> "12-26-9"
                periods_str = "-".join(map(str, f["feature_periods"]))
                feature_strings.append(f"{f['feature_type']}:{periods_str}:{f['shift']}")
            
            self.param_fields["features"].setText(", ".join(feature_strings))

            self.param_fields["targets"].setText(
                ", ".join(
                    f"{t['base_column']}:{t['shift']}"
                    for t in config["targets"]
                )
            )

            self.param_fields["architectures"].setText(
                ", ".join(config["architectures"])
            )

            self.log_to_console(f"Wczytano parametry zadania: {self.last_clicked_uuid}")
        except Exception as e:
            self.log_to_console(f"Błąd wczytywania parametrów: {e}")

    def on_update_params(self):
        if not self.last_clicked_uuid:
            self.log_to_console("Nie wybrano zadania do modyfikacji")
            return
        try:
            field_values = {param: self.param_fields[param].text().strip() for param in self.PARAM_MAP.values()}

            config = {
                "instrument": {"name": field_values["instrument_name"]},
                "timeframe": {"name": field_values["timeframe_name"]},
                "data_source": field_values["data_source"],
                "parameter_set": {
                    "samples_limit": int(field_values["samples_limit"]),
                    "train_ratio": float(field_values["train_ratio"]),
                    "seed": int(field_values["seed"]),
                    "epochs": int(field_values["epochs"]),
                    "train_noise": float(field_values["train_noise"]),
                    "learning_rate": float(field_values["learning_rate"]),
                },
                "features": [],
                "targets": [],
                "architectures": []
            }

            if field_values["features"]:
                # Nowy blok parsowania features (typ:p1-p2:shift):
                for feature in field_values["features"].split(","):
                    f_str = feature.strip()
                    if not f_str: continue
                    
                    parts = f_str.split(":")
                    if len(parts) != 3:
                        raise ValueError(f"Niepoprawny format feature: {f_str} (oczekiwano formatu typ:parametry:shift)")
                    
                    # Konwersja np. "12-26-9" na [12, 26, 9]
                    periods = [int(p.strip()) for p in parts[1].split("-") if p.strip()]
                    
                    config["features"].append({
                        "feature_type": parts[0].strip(),
                        "feature_periods": periods,
                        "shift": int(parts[2].strip())
                    })

            if field_values["targets"]:
                for target in field_values["targets"].split(","):
                    target_parts = target.strip().split(":")
                    if len(target_parts) != 2:
                        raise ValueError(f"Niepoprawny format target: {target}")
                    config["targets"].append({
                        "base_column": target_parts[0].strip(),
                        "shift": int(target_parts[1].strip())
                    })

            if field_values["architectures"]:
                config["architectures"] = [
                    a.strip() for a in field_values["architectures"].split(",") if a.strip()
                ]

            self.db_manager.update_training_config(self.last_clicked_uuid, config)
            self.log_to_console(f"Zaktualizowano parametry: {self.last_clicked_uuid}")
            
            current_uuid = self.last_clicked_uuid
            self.on_load_tasks(show_log=False)
            
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item and item.text() == current_uuid:
                    self.table.selectRow(row)
                    break

        except ValueError as e:
            self.log_to_console(f"Błąd walidacji: {e}")
        except Exception as e:
            self.log_to_console(f"{e}")

    def on_remove_task(self):
        if not self.last_clicked_uuid:
            self.log_to_console("Nie wybrano zadania do usunięcia")
            return
        try:
            self.db_manager.del_training_job(self.last_clicked_uuid)
            self.log_to_console(f"Usunięto zadanie: {self.last_clicked_uuid}")

            self.on_load_tasks(show_log=False)

            if self.table.rowCount() > 0:
                self.table.selectRow(0)
                self.last_clicked_uuid = self.table.item(0, 0).text()
            else:
                self.last_clicked_uuid = None
        except Exception as e:
            self.log_to_console(f"{e}")

    def on_load_tasks(self, show_log=True):
        try:
            tasks = self.db_manager.get_training_jobs()

            if not tasks:
                self.table.setRowCount(0)
                if show_log:
                    self.log_to_console("Brak zadań w bazie danych.")
                return

            self.fill_tasks_table(tasks)

            if show_log:
                self.log_to_console("Wczytano listę zadań.")
        except Exception as e:
            self.log_to_console(f"Błąd wczytywania zadań: {e}")
