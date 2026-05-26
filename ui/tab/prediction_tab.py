import base64
from PyQt6.QtWidgets import (
    QPushButton, QTextEdit, QTableWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFormLayout,
    QTableWidgetItem
)
from PyQt6.QtWidgets import QFileDialog
import pandas as pd
import io
import os
import requests
import matplotlib.pyplot as plt
from ui.tab.base_tab import BaseTab
from ui.workers.prediction_status_poller import PredictionStatusPoller
from ui.workers.prediction_logs_poller import PredictionLogsPoller
from dotenv import load_dotenv

load_dotenv()

class PredictionTab(BaseTab):
    PARAM_MAP = {
        "Liczba próbek": "all_samples",
        "Liczba próbek przewidywanych": "predicted_samples"
    }

    def __init__(self, tab_widget=None):
        super().__init__()
        self.api_url = os.getenv("API_URL")
        self.tab_widget = tab_widget
        self.last_clicked_train_uuid = None
        self.last_clicked_pred_uuid = None
        self.status_poller = None
        self.logs_pollers = {}
        self.init_ui()
        self.init_actions()

    def init_ui(self):
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.add_pred_btn = QPushButton("Dodaj predykcję")
        self.remove_pred_btn = QPushButton("Usuń predykcję")
        self.load_params_btn = QPushButton("Wczytaj parametry")
        self.run_pred_btn = QPushButton("Uruchom predykcję")
        self.stop_pred_btn = QPushButton("Zatrzymaj predykcję")
        self.plot_charts_btn = QPushButton("Generuj wykresy")
        self.clear_console_btn = QPushButton("Wyczyść konsolę")

        self.buttons = [
            self.add_pred_btn, self.remove_pred_btn, self.load_params_btn, self.run_pred_btn,
            self.stop_pred_btn, self.plot_charts_btn, self.clear_console_btn
        ]
        for btn in self.buttons:
            left_layout.addWidget(btn)

        left_layout.addWidget(QLabel("<b>Parametry Predykcji</b>"))
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

        self.train_table = QTableWidget()
        self.train_table.setColumnCount(4)
        self.train_table.setHorizontalHeaderLabels(["UUID Treningu", "Instrument", "Interwał", "Utworzono"])
        self.train_table.setSelectionBehavior(self.train_table.SelectionBehavior.SelectRows)
        self.train_table.setSelectionMode(self.train_table.SelectionMode.SingleSelection)

        self.pred_table = QTableWidget()
        self.pred_table.setColumnCount(4)
        self.pred_table.setHorizontalHeaderLabels(["UUID Predykcji", "UUID Treningu", "Status", "Utworzono"])
        self.pred_table.setSelectionBehavior(self.pred_table.SelectionBehavior.SelectRows)
        self.pred_table.setSelectionMode(self.pred_table.SelectionMode.SingleSelection)

        tables_layout.addWidget(self.train_table, 5)
        tables_layout.addWidget(self.pred_table, 6)

        self.console = QTextEdit()
        self.console.setReadOnly(True)

        right_layout.addLayout(tables_layout, 1) 
        right_layout.addWidget(self.console, 1)  

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 5)
        self.setLayout(main_layout)

    def init_actions(self):
        self.clear_console_btn.clicked.connect(lambda: self.console.clear())
        self.train_table.cellClicked.connect(self.on_train_table_clicked)
        self.pred_table.cellClicked.connect(self.on_pred_table_clicked)
        self.add_pred_btn.clicked.connect(self.on_add_prediction)
        self.remove_pred_btn.clicked.connect(self.on_remove_prediction)
        self.load_params_btn.clicked.connect(self.on_load_prediction_params)
        self.run_pred_btn.clicked.connect(self.on_run_prediction)
        self.stop_pred_btn.clicked.connect(self.on_stop_prediction)
        self.plot_charts_btn.clicked.connect(self.on_plot_charts)

    def toggle_ui_lock(self, is_running: bool):
        if self.tab_widget:
            self.tab_widget.tabBar().setEnabled(not is_running)
        
        self.add_pred_btn.setEnabled(not is_running)
        self.remove_pred_btn.setEnabled(not is_running)
        self.run_pred_btn.setEnabled(not is_running)

    def on_run_prediction(self):
        if not self.last_clicked_pred_uuid:
            self.log_to_console("Nie zaznaczono predykcji")
            return
        try:
            response = requests.post(
                self.api_url + f"/users/{self.user_id}/predictions/{self.last_clicked_pred_uuid}/run",
                headers={"Authorization": f"Bearer {str(self.session_token)}"}
            )
            if response.status_code != 200:
                raise ValueError(response.json()["detail"])
            self.log_to_console(f"Uruchomiono predykcję: {self.last_clicked_pred_uuid}")
            self.start_logs_poller()
            self.on_load_predictions(show_log=False)
        except Exception as e:
            self.log_to_console(f"Błąd uruchamiania: {e}")

    def on_stop_prediction(self):
        if not self.last_clicked_pred_uuid:
            self.log_to_console("Nie zaznaczono predykcji")
            return
        try:
            response = requests.patch(
                self.api_url + f"/users/{self.user_id}/predictions/{self.last_clicked_pred_uuid}/stop",
                headers={"Authorization": f"Bearer {str(self.session_token)}"}
            )
            if response.status_code != 200:
                raise ValueError(response.json()["detail"])
            self.log_to_console(f"Zatrzymano predykcję: {self.last_clicked_pred_uuid}")
            self.on_load_predictions(show_log=False)
        except Exception as e:
            self.log_to_console(f"Błąd zatrzymywania: {e}")

    def on_load_prediction_params(self):
        if not self.last_clicked_pred_uuid:
            self.log_to_console("Nie wybrano predykcji")
            return
        
        try:
            response = requests.get(
                self.api_url + f"/users/{self.user_id}/predictions/{self.last_clicked_pred_uuid}/config",
                headers={"Authorization": f"Bearer {str(self.session_token)}"}
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            pred_config = response.json()["config"]

            self.param_fields["all_samples"].setText(str(pred_config["all_samples"]))
            self.param_fields["predicted_samples"].setText(str(pred_config["predicted_samples"]))

            self.log_to_console(f"Wczytano parametry predykcji: {self.last_clicked_pred_uuid}")
        except Exception as e:
            self.log_to_console(f"Błąd parametrów: {e}")

    def on_load_trainings(self, show_log=True):    
        if not self.user_id:
            return   
        try:
            response = requests.get(
                self.api_url + f"/users/{self.user_id}/trainings",
                headers={"Authorization": f"Bearer {str(self.session_token)}"},
                params={"status": "completed"}
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"]) 

            trainings = response.json()["trainings"]
            
            self.fill_train_table(trainings)

            return trainings
        
        except Exception as e:
            self.log_to_console(f"Błąd wczytywania: {e}")

    def on_load_predictions(self, show_log=True):       
        try:
            response = requests.get(
                self.api_url + f"/users/{self.user_id}/predictions",
                headers={"Authorization": f"Bearer {str(self.session_token)}"}
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"]) 

            predictions = response.json()["predictions"]
            
            self.fill_pred_table(predictions)

            return predictions
        
        except Exception as e:
            self.log_to_console(f"Błąd wczytywania: {e}")

    def fill_train_table(self, trainings: list):
        self.train_table.setRowCount(len(trainings))
        for row, t in enumerate(trainings):
            self.train_table.setItem(row, 0, QTableWidgetItem(str(t.get("train_uuid", ""))))
            self.train_table.setItem(row, 1, QTableWidgetItem(str(t.get("instrument", ""))))
            self.train_table.setItem(row, 2, QTableWidgetItem(str(t.get("timeframe_name", ""))))
            self.train_table.setItem(row, 3, QTableWidgetItem(str(t.get("created_at", "")).replace("T", " ").split(".")[0]))
        
        if len(trainings) > 0:
            self.last_clicked_train_uuid = self.train_table.item(0, 0).text()
            self.train_table.selectRow(0)
        else:
            self.last_clicked_train_uuid = None
            
        self.train_table.resizeColumnsToContents()

    def fill_pred_table(self, predictions: list):
        self.pred_table.setRowCount(len(predictions))
        
        for row, pred in enumerate(predictions):
            curr_uuid = str(pred.get("pred_uuid", ""))
            
            self.pred_table.setItem(row, 0, QTableWidgetItem(curr_uuid))
            self.pred_table.setItem(row, 1, QTableWidgetItem(str(pred.get("train_uuid", ""))))
            self.pred_table.setItem(row, 2, QTableWidgetItem(str(pred.get("status", ""))))
            self.pred_table.setItem(row, 3, QTableWidgetItem(str(pred.get("created_at", "")).replace("T", " ").split(".")[0]))

        if predictions and len(predictions) > 0:
            first_uuid = self.pred_table.item(0, 0).text()
            self.last_clicked_pred_uuid = first_uuid
            self.pred_table.selectRow(0)
        else:
            self.last_clicked_pred_uuid = None

        self.pred_table.resizeColumnsToContents()

    def on_train_table_clicked(self, row, column):
        item = self.train_table.item(row, 0)
        if item:
            self.last_clicked_train_uuid = item.text()

    def on_pred_table_clicked(self, row, column):
        item = self.pred_table.item(row, 0)
        if item:
            self.last_clicked_pred_uuid = item.text()

    def on_add_prediction(self):
        if not self.last_clicked_train_uuid:
            self.log_to_console("Nie zaznaczono żadnego treningu")
            return
        try:
            pred_config = {p: self.param_fields[p].text().strip() for p in self.PARAM_MAP.values()}
            for key, val in pred_config.items():
                if not val:
                    raise ValueError(f"Pole {key} nie może być puste.")
                
            pred_config["train_uuid"] = self.last_clicked_train_uuid
                
            response = requests.post(
                self.api_url + f"/users/{self.user_id}/predictions",
                headers={"Authorization": f"Bearer {str(self.session_token)}"},
                json=pred_config
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            pred_uuid = response.json()["pred_uuid"]
            
            self.pred_table.clearSelection()
            self.last_clicked_pred_uuid = pred_uuid
            self.on_load_predictions(show_log=False)
            
            for row in range(self.pred_table.rowCount()):
                item = self.pred_table.item(row, 0)
                if item and item.text() == pred_uuid:
                    self.pred_table.selectRow(row)
                    self.pred_table.setCurrentItem(item)
                    break

            self.log_to_console(f"Dodano predykcję: {pred_uuid}")

        except Exception as e:
            self.log_to_console(f"Błąd dodawania predykcji: {e}")

    def on_remove_prediction(self):
        if not self.last_clicked_pred_uuid:
            self.log_to_console("Nie wybrano predykcji do usunięcia")
            return
        
        try:
            response = requests.delete(
                self.api_url + f"/users/{self.user_id}/predictions/{self.last_clicked_pred_uuid}",
                headers={"Authorization": f"Bearer {str(self.session_token)}"}
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            self.log_to_console(f"Usunięto zadanie: {self.last_clicked_pred_uuid}")
            self.on_load_predictions(show_log=False)

            if self.pred_table.rowCount() > 0:
                self.pred_table.selectRow(0)
                item = self.pred_table.item(0, 0)
                self.last_clicked_pred_uuid = item.text() if item else None
            else:
                self.last_clicked_pred_uuid = None

        except Exception as e:
            self.log_to_console(f"Błąd usuwania: {e}")

    def on_plot_charts(self):
        pred_uuid = self.last_clicked_pred_uuid

        if not pred_uuid:
            self.log_to_console("Nie wybrano predykcji.")
            return
        
        plots_dir = QFileDialog.getExistingDirectory(self, "Wybierz folder do zapisu")
        if not plots_dir: return

        try:
            response = requests.get(
                self.api_url + f"/users/{self.user_id}/predictions/{pred_uuid}/config",
                headers={"Authorization": f"Bearer {str(self.session_token)}"}
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            pred_config = response.json()["config"]

            if pred_config["status"] != 'completed': return
            
            response = requests.get(
                self.api_url + f"/users/{self.user_id}/trainings/{pred_config['train_uuid']}/config",
                headers={"Authorization": f"Bearer {str(self.session_token)}"}
            )

            if response.status_code != 200:
                raise ValueError(response.json()["detail"])

            train_config = response.json()["config"]
            
            for arch_name in train_config['architectures']:
                response = requests.get(
                    self.api_url + f"/users/{self.user_id}/predictions/{pred_uuid}/result/{arch_name}",
                    headers={"Authorization": f"Bearer {str(self.session_token)}"}
                )

                if response.status_code != 200:
                    raise ValueError(response.json()["detail"])

                data = base64.b64decode(response.json()["data"])
                
                df = pd.read_parquet(io.BytesIO(data))
                if df.empty: continue

                plt.figure(figsize=(12, 6)) 
                plt.grid(True, color='gray', linestyle=':', alpha=0.5)

                plt.scatter(df.index, df['close'], color='#CC0000', marker='o', s=2, label='Cena bieżąca close', zorder=1)

                for target in train_config["targets"]:
                    plt.plot(df.index, df[target], linewidth=1, label=f"Predykcja {target}", zorder=2)
                
                plt.title(f"{train_config['instrument_name']} - {arch_name}")
                plt.legend(loc='best', fontsize='small')
                plt.xlabel("Numer próbki")
                plt.ylabel("Cena")
                
                plt.tight_layout()

                plot_name = f"{train_config['instrument_name']}_{arch_name}_{pred_uuid[:6]}.png".replace(" ", "")
                plt.savefig(os.path.join(plots_dir, plot_name), dpi=120)
                plt.close('all')

            self.log_to_console(f"Wygenerowano wykresy w: {plots_dir}")
            
        except Exception as e:
            self.log_to_console(f"Błąd: {e}")

    def log_to_console(self, message: str):
        self.console.append(message)

    def set_session(self, user_id, session_token):
        super().set_session(user_id, session_token)
        trainings = self.on_load_trainings()
        predictions = self.on_load_predictions()
        if not predictions:
            return
        for p in predictions:
            if p["status"] in ("running", "pending", "stopping"):
                self.start_logs_poller(p["pred_uuid"])

    def clear_session(self):
        if self.status_poller:
            self.status_poller.stop()
            self.status_poller.wait()
            self.status_poller = None

        for poller in self.logs_pollers.values():
            poller.stop()
            poller.wait()
        self.logs_pollers.clear()

        super().clear_session()

        self.console.clear()

        self.train_table.setRowCount(0)
        self.pred_table.setRowCount(0)

        self.last_clicked_train_uuid = None
        self.last_clicked_pred_uuid = None

        # for field in self.param_fields.values():
        #     field.clear()

    def start_status_poller(self):
        self.status_poller = PredictionStatusPoller(self.api_url, self.user_id, self.session_token)
        self.status_poller.status_received.connect(self.fill_pred_table)
        self.status_poller.start()

    def start_logs_poller(self, pred_uuid=None):
        pred_uuid = pred_uuid or self.last_clicked_pred_uuid
        if pred_uuid not in self.logs_pollers:
            poller = PredictionLogsPoller(self.api_url, self.user_id, self.session_token, pred_uuid)
            poller.logs_received.connect(self.log_list_to_console)
            poller.start()
            self.logs_pollers[pred_uuid] = poller

    def log_to_console(self, message: str):
        self.console.append(message)

    def log_list_to_console(self, messages: list):
        for message in messages:
            self.log_to_console(message)