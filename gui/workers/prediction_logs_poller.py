import requests
import threading
from PyQt6.QtCore import QThread, pyqtSignal

class PredictionLogsPoller(QThread):
    logs_received = pyqtSignal(list)

    def __init__(self, api_url, user_id, session_token, pred_uuid):
        super().__init__()

        self.api_url = api_url
        self.user_id = user_id
        self.session_token = session_token
        self.pred_uuid = pred_uuid
        self._running = True
        self._last_log_count = 0
        self._stop_event = threading.Event()

    def run(self):
        while self._running:
            if not self.session_token:
                break
            try:
                response = requests.get(
                    self.api_url + f"/users/{self.user_id}/predictions/{self.pred_uuid}/logs",
                    headers={
                        "Authorization": f"Bearer {str(self.session_token)}",
                    }
                )

                if response.status_code != 200:
                    raise ValueError(response.json()["detail"])

                logs = response.json()["logs"]
                new_logs = logs[self._last_log_count:]
                if new_logs:
                    self.logs_received.emit(new_logs)
                    self._last_log_count = len(logs)

            except Exception as e:
                pass
            
            self._stop_event.wait(1)

    def stop(self):
        self._running = False
        self.session_token = None
        self._stop_event.set()