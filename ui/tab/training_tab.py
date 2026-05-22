import requests
import os
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QPushButton, QTextEdit, QTableWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtWidgets import QLabel, QLineEdit, QFormLayout, QTableWidgetItem
from ui.tab.base_tab import BaseTab
from src.worker.training_worker import TrainingWorker
from api.db_manager import DBManager

from dotenv import load_dotenv

load_dotenv()

class TrainingTab(BaseTab):
    PARAM_MAP = {
        "Instrument": "instrument_name", "Interwał": "timeframe_name",
        "Źródło danych": "data_source_name", "Liczba próbek": "all_samples",
        "Liczba próbek testowych": "test_samples", "Losowość": "seed",
        "Liczba epok": "epochs", "Współczynnik szumu": "train_noise",
        "Współczynnik uczenia": "learning_rate", "Cechy": "features",
        "Wartości docelowe": "targets", "Architektury modeli": "architectures"
    }

    def log_to_console(self, message: str):
        self.console.append(message)

    def __init__(self, db_manager: DBManager, tab_widget=None):
        super().__init__()
        self.db_manager = db_manager

        self.api_url = os.getenv("API_URL")

        self.tab_widget = tab_widget
        self.last_clicked_uuid = None
        self.thread = None
        self.worker = None
        self.init_ui()
        self.init_actions()

    def init_ui(self):
        left_layout = QVBoxLayout()
        self.add_task_btn = QPushButton("Dodaj trening")
        self.remove_task_btn = QPushButton("Usuń trening")
        self.load_params_btn = QPushButton("Wczytaj parametry")
        self.clear_console_btn = QPushButton("Wyczyść konsolę")
        self.run_task_btn = QPushButton("Uruchom trening")
        self.stop_task_btn = QPushButton("Zatrzymaj trening")

        self.buttons = [
            self.add_task_btn, self.remove_task_btn,
            self.load_params_btn, self.clear_console_btn,
            self.run_task_btn, self.stop_task_btn
        ]
        for btn in self.buttons:
            left_layout.addWidget(btn)

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
        self.table.setHorizontalHeaderLabels(["UUID Treningu", "Instrument", "Interwał", "Status", "Utworzono"])
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(self.table.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.console = QTextEdit()
        self.console.setReadOnly(True)

        right_layout.addWidget(self.table)
        right_layout.addWidget(self.console)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.setLayout(main_layout)

    def init_actions(self):
        self.remove_task_btn.clicked.connect(self.on_remove_task)
        self.load_params_btn.clicked.connect(self.on_load_params)
        self.clear_console_btn.clicked.connect(self.on_clear_console)
        self.add_task_btn.clicked.connect(self.on_add_task)
        self.run_task_btn.clicked.connect(self.on_run_task)
        self.stop_task_btn.clicked.connect(self.on_stop_task)
        self.table.cellClicked.connect(self.on_table_cell_clicked)

    def on_table_cell_clicked(self, row, column):
        item = self.table.item(row, 0)
        if item:
            self.last_clicked_uuid = item.text()

    def fill_tasks_table(self, tasks: list[dict]):
        self.table.setRowCount(len(tasks))
        for row, t in enumerate(tasks):
            self.table.setItem(row, 0, QTableWidgetItem(str(t["train_uuid"])))
            self.table.setItem(row, 1, QTableWidgetItem(t["instrument"]))
            self.table.setItem(row, 2, QTableWidgetItem(t["timeframe_name"]))
            self.table.setItem(row, 3, QTableWidgetItem(t["status"]))
            self.table.setItem(row, 4, QTableWidgetItem(str(t["created_at"])))
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
        
        try:
            current_status = self.db_manager.get_training_status(self.last_clicked_uuid)
            if current_status == 'completed':
                self.log_to_console(f"Zadanie {self.last_clicked_uuid} jest już ukończone. Nie można go uruchomić ponownie.")
                return
            if current_status == 'running':
                self.log_to_console(f"Zadanie {self.last_clicked_uuid} jest już w trakcie wykonywania.")
                return
        except Exception as e:
            self.log_to_console(f"Błąd sprawdzania statusu: {e}")
            return

        self.thread = QThread()
        self.worker = TrainingWorker(self.db_manager, self.last_clicked_uuid)
        self.worker.log_signal.connect(self.log_to_console)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.destroyed.connect(lambda: setattr(self, 'thread', None))
        self.thread.destroyed.connect(lambda: setattr(self, 'worker', None))

        self.thread.finished.connect(lambda: self.set_running_ui(False))
        self.thread.finished.connect(lambda: self.on_load_training_tasks(show_log=False))

        self.db_manager.update_training_status(self.last_clicked_uuid, 'running')
        self.on_load_training_tasks(show_log=False)

        self.set_running_ui(True)
        self.thread.start()

    def on_stop_task(self):
        if self.worker and self.thread and self.thread.isRunning():
            try:
                self.log_to_console("Zatrzymywanie treningu...")
                self.worker.stop()
            except Exception as e:
                self.log_to_console(f"Błąd podczas zatrzymywania: {e}")

    def set_running_ui(self, running: bool):
        for btn in [self.add_task_btn, self.remove_task_btn, 
                    self.load_params_btn, self.run_task_btn]:
            btn.setEnabled(not running)
            
        if self.tab_widget:
            self.tab_widget.tabBar().setEnabled(not running)

    def on_load_training_tasks(self, show_log=True):
        try:
            response = requests.get(
                self.api_url + f"/users/{self.user_id}/trainings",
                headers={
                    "Authorization": f"Bearer {str(self.session_token)}",
                }
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                trainings = response["trainings"]

            if not trainings:
                self.table.setRowCount(0)
                if show_log:
                    self.log_to_console("Brak zadań w bazie.")
                return
            
            self.fill_tasks_table(trainings)
            
            if not self.last_clicked_uuid and self.table.rowCount() > 0:
                self.table.selectRow(0)
                item = self.table.item(0, 0)
                if item:
                    self.last_clicked_uuid = item.text()
            
            if show_log:
                self.log_to_console("Wczytano listę zadań.")
        except Exception as e:
            self.log_to_console(f"Błąd wczytywania: {e}")

    def set_session(self, user_id, session_token):
        super().set_session(user_id, session_token)
        self.on_load_training_tasks()

    def clear_session(self):
        super().clear_session()
        self.console.clear()
        self.table.setRowCount(0)
        self.last_clicked_uuid = None
        for field in self.param_fields.values():
            field.clear()

    def on_add_task(self):
        try:
            field_values = {param: self.param_fields[param].text().strip() for param in self.PARAM_MAP.values()}
            required_fields = list(self.PARAM_MAP.values())

            for field in required_fields:
                if not field_values.get(field):
                    raise ValueError(f"Pole '{field}' nie może być puste")
                
            train_config = {
                "instrument_name": field_values["instrument_name"],
                "timeframe_name": field_values["timeframe_name"],
                "data_source_name": field_values["data_source_name"],
                "all_samples": int(field_values["all_samples"]),
                "test_samples": int(field_values["test_samples"]),
                "seed": int(field_values["seed"]),
                "epochs": int(field_values["epochs"]),
                "train_noise": float(field_values["train_noise"]),
                "learning_rate": float(field_values["learning_rate"]),
                "features": [],
                "targets": [],
                "architectures": []
            }

            for feature in field_values["features"].split(","):
                f_str = feature.strip()
                if not f_str:
                    continue
                parts = f_str.split(":")
                if len(parts) != 3:
                    raise ValueError(f"Format feature: typ:parametry:shift")
                train_config["features"].append({
                    "feature_type": parts[0].strip(),
                    "feature_periods": [int(p.strip()) for p in parts[1].split("-") if p.strip()],
                    "shift": int(parts[2].strip())
                })

            for target in field_values["targets"].split(","):
                target_str = target.strip()
                if not target_str:
                    continue
                parts = target_str.split(":")
                if len(parts) != 2:
                    raise ValueError("Format targetu to 'nazwa_kolumny:shift'")
                train_config["targets"].append({
                    "column": parts[0].strip(),
                    "shift": int(parts[1].strip())
                })

            for architecture in field_values["architectures"].split(","):
                train_config["architectures"].append(architecture.strip())

            response = requests.post(
                self.api_url + f"/users/{self.user_id}/trainings",
                headers={
                    "Authorization": f"Bearer {str(self.session_token)}"
                },
                json=train_config
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                new_uuid = response["train_uuid"]

            self.table.clearSelection()
            self.last_clicked_uuid = new_uuid
            self.on_load_training_tasks(show_log=False)
            
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item and item.text() == new_uuid:
                    self.table.selectRow(row)
                    self.table.setCurrentItem(item)
                    break

            self.log_to_console(f"Dodano trening: {new_uuid}")

        except Exception as e:
            self.log_to_console(f"Błąd dodawania: {e}")

    def on_remove_task(self):
        if not self.last_clicked_uuid:
            self.log_to_console("Nie wybrano zadania do usunięcia")
            return
        
        try:
            response = requests.delete(
                self.api_url + f"/users/{self.user_id}/trainings/{self.last_clicked_uuid}",
                headers={
                    "Authorization": f"Bearer {str(self.session_token)}"
                }
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                self.log_to_console(f"Usunięto zadanie: {self.last_clicked_uuid}")
                self.on_load_training_tasks(show_log=False)

                if self.table.rowCount() > 0:
                    self.table.selectRow(0)
                    item = self.table.item(0, 0)
                    self.last_clicked_uuid = item.text() if item else None
                else:
                    self.last_clicked_uuid = None

        except Exception as e:
            self.log_to_console(f"Błąd usuwania: {e}")

    def on_load_params(self):
        if not self.last_clicked_uuid:
            self.log_to_console("Nie wybrano zadania")
            return
        
        try:
            response = requests.get(
                self.api_url + f"/users/{self.user_id}/trainings/{self.last_clicked_uuid}/config",
                headers={
                    "Authorization": f"Bearer {str(self.session_token)}"
                }
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            if (response := response.json()) is not None:
                train_config = response["config"]
            
                self.param_fields["instrument_name"].setText(train_config["instrument_name"])
                self.param_fields["timeframe_name"].setText(train_config["timeframe_name"])
                self.param_fields["data_source_name"].setText(train_config["data_source_name"])
                self.param_fields["all_samples"].setText(str(train_config["all_samples"]))
                self.param_fields["test_samples"].setText(str(train_config["test_samples"]))
                self.param_fields["seed"].setText(str(train_config["seed"]))
                self.param_fields["epochs"].setText(str(train_config["epochs"]))
                self.param_fields["train_noise"].setText(str(train_config["train_noise"]))
                self.param_fields["learning_rate"].setText(str(train_config["learning_rate"]))
                self.param_fields["features"].setText(", ".join(train_config["features"]))
                self.param_fields["targets"].setText(", ".join(train_config["targets"]))
                self.param_fields["architectures"].setText(", ".join(train_config["architectures"]))

                self.log_to_console(f"Wczytano parametry: {self.last_clicked_uuid}")
        except Exception as e:
            self.log_to_console(f"Błąd parametrów: {e}")

