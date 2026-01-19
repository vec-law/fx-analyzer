import traceback
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QTextEdit, QTableWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFormLayout,
    QTableWidgetItem
)
from src.worker.training_worker import TrainingWorker

class TrainingTab(QWidget):
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
        self.clear_console_btn = QPushButton("Wyczyść konsolę")
        self.run_task_btn = QPushButton("Uruchom zadanie")
        self.stop_task_btn = QPushButton("Zatrzymaj zadanie")

        for btn in [
            self.load_tasks_btn,
            self.add_task_btn,
            self.remove_task_btn,
            self.load_params_btn,
            self.clear_console_btn,
            self.run_task_btn,
            self.stop_task_btn
        ]:
            left_layout.addWidget(btn)

        left_layout.addWidget(QLabel("<b>Parametry</b>"))

        self.param_fields = {}
        params_layout = QFormLayout()
        for name in [
            'Instrument', 'Interwał', 'Źródło danych', 'Limit próbek',
            'Współczynnik podziału', 'Losowość', 'Liczba epok', 'Współczynnik szumu',
            'Współczynnik uczenia', 'Wartości docelowe', 'Cechy', 'Architektury modeli'
        ]:
            field = QLineEdit()
            self.param_fields[name] = field
            params_layout.addRow(name, field)

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

        self.stop_task_btn.setEnabled(False)


    def init_actions(self):
        self.load_tasks_btn.clicked.connect(self.on_load_tasks)
        self.remove_task_btn.clicked.connect(self.on_remove_tasks)
        self.load_params_btn.clicked.connect(self.on_load_params)
        self.clear_console_btn.clicked.connect(self.on_clear_console)
        self.add_task_btn.clicked.connect(self.on_add_task)
        self.run_task_btn.clicked.connect(self.on_run_task)
        self.stop_task_btn.clicked.connect(self.on_stop_task)
        self.table.cellClicked.connect(self.on_table_cell_clicked)

    def on_load_tasks(self):
        tasks = self.db_manager.get_training_jobs()
        self.fill_tasks_table(tasks)
        self.console.append(f"Wczytano {len(tasks)} zadanie(-a/-ń)")

    def on_remove_tasks(self):
        if not self.last_clicked_uuid:
            self.console.append("Nie wybrano zadania do usunięcia!")
            return

        success = self.db_manager.del_training_job(self.last_clicked_uuid)

        if success:
            self.console.append(f"Usunięto zadanie: {self.last_clicked_uuid}")
            self.fill_tasks_table(self.db_manager.get_training_jobs())
            
            if self.table.rowCount() > 0:
                self.table.selectRow(0)
                self.last_clicked_uuid = self.table.item(0, 0).text()
            else:
                self.last_clicked_uuid = None
        else:
            self.console.append(f"Nie udało się usunąć zadania: {self.last_clicked_uuid}")

    def on_table_cell_clicked(self, row, column):
        uuid_item = self.table.item(row, 0)
        if uuid_item:
            self.last_clicked_uuid = uuid_item.text()

    def fill_tasks_table(self, tasks: list[dict]):
        self.table.setRowCount(len(tasks))
        for row, task in enumerate(tasks):
            self.table.setItem(row, 0, QTableWidgetItem(str(task["job_uuid"])))
            self.table.setItem(row, 1, QTableWidgetItem(task["instrument"]))
            self.table.setItem(row, 2, QTableWidgetItem(task["timeframe"]))
            self.table.setItem(row, 3, QTableWidgetItem(task["status"]))
            self.table.setItem(row, 4, QTableWidgetItem(task["created_at"].strftime("%Y-%m-%d %H:%M:%S")))
        self.table.resizeColumnsToContents()

    def on_load_params(self):
        if not self.last_clicked_uuid:
            self.console.append("Nie wybrano zadania!")
            return

        config = self.db_manager.get_training_config(self.last_clicked_uuid)
        if not config:
            self.console.append("Nie znaleziono parametrów dla wybranego zadania!")
            return

        param_mapping = {
            "instrument": "Instrument",
            "timeframe": "Interwał",
            "data_source": "Źródło danych",
            "train_ratio": "Współczynnik podziału",
            "seed": "Losowość",
            "epochs": "Liczba epok",
            "samples_limit": "Limit próbek",
            "train_noise": "Współczynnik szumu",
            "learning_rate": "Współczynnik uczenia",
            "targets": "Wartości docelowe",
            "features": "Cechy",
            "architectures": "Architektury modeli"
        }

        for db_key, gui_name in param_mapping.items():
            field = self.param_fields.get(gui_name)
            value = config.get(db_key, "")
            if field:
                field.setText(str(value))

        self.console.append(f"Wczytano parametry zadania: {self.last_clicked_uuid}")

    def on_add_task(self):
        gui_to_db = {
            "Instrument": "instrument",
            "Interwał": "timeframe",
            "Źródło danych": "data_source",
            "Limit próbek": "samples_limit",
            "Współczynnik podziału": "train_ratio",
            "Losowość": "seed",
            "Liczba epok": "epochs",
            "Współczynnik szumu": "train_noise",
            "Współczynnik uczenia": "learning_rate",
            "Wartości docelowe": "targets",
            "Cechy": "features",
            "Architektury modeli": "architectures"
        }

        task_data = { db_key: self.param_fields[gui_key].text()
                    for gui_key, db_key in gui_to_db.items() }

        try:
            job_uuid = self.db_manager.add_training_job(task_data)
            if job_uuid:
                self.console.append(f"Dodano zadanie: {job_uuid}")
                tasks = self.db_manager.get_training_jobs()
                self.fill_tasks_table(tasks)

                for row in range(self.table.rowCount()):
                    if self.table.item(row, 0).text() == str(job_uuid):
                        self.table.selectRow(row)
                        self.last_clicked_uuid = str(job_uuid)
                        break
            else:
                self.console.append("Błąd: nie udało się dodać zadania")
        except Exception as e:
            self.console.append(f"Błąd przy dodawaniu zadania: {e}")

    def on_clear_console(self):
        self.console.clear()

    def on_run_task(self):
        if not self.last_clicked_uuid:
            self.console.append("Nie zaznaczono zadania!")
            return

        # Bezpieczne sprawdzenie czy wątek żyje
        try:
            if self.thread and self.thread.isRunning():
                self.console.append("Zadanie już jest uruchomione!")
                return
        except RuntimeError:
            self.thread = None
            self.worker = None

        self.console.append("Uruchamiam zadanie...")

        self.thread = QThread() # usunięto self jako parent, by deleteLater działało szybciej
        self.worker = TrainingWorker(self.last_clicked_uuid, self.db_manager)
        self.worker.moveToThread(self.thread)

        # Łączenie sygnałów
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Czyszczenie zmiennych po faktycznym usunięciu obiektów C++
        self.thread.destroyed.connect(lambda: setattr(self, 'thread', None))
        self.thread.destroyed.connect(lambda: setattr(self, 'worker', None))

        self.worker.log.connect(self.console.append)
        self.thread.finished.connect(lambda: self.console.append("Zadanie zakończone"))
        self.thread.finished.connect(lambda: self.set_running_ui(False))

        self.set_running_ui(True)
        self.thread.start()

    def on_stop_task(self):
        self.set_running_ui(False)
        try:
            # Sprawdzamy czy worker istnieje i czy jego obiekt C++ nie został usunięty
            if self.worker:
                self.worker.stop()
                self.console.append("Wysłano sygnał zatrzymania...")
            else:
                self.console.append("Brak aktywnego zadania.")
        except RuntimeError:
            self.worker = None
            self.thread = None
            self.console.append("Zadanie nie było aktywne (obiekt usunięty).")

    def set_running_ui(self, running: bool):
        self.run_task_btn.setEnabled(not running)
        self.stop_task_btn.setEnabled(running)

        self.load_tasks_btn.setEnabled(not running)
        self.add_task_btn.setEnabled(not running)
        self.remove_task_btn.setEnabled(not running)
        self.load_params_btn.setEnabled(not running)
