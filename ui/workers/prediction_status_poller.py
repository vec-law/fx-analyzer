import requests
import threading
from PyQt6.QtCore import QThread, pyqtSignal

class PredictionStatusPoller(QThread):
    status_received = pyqtSignal(list)

    def __init__(self, api_url, user_id, session_token):
        super().__init__()

        self.api_url = api_url
        self.user_id = user_id
        self.session_token = session_token
        self._running = True
        self._last_statuses = None
        self._stop_event = threading.Event()

    def run(self):
        while self._running:
            if not self.session_token:
                break
            try:
                response = requests.get(
                    self.api_url + f"/users/{self.user_id}/predictions",
                    headers={"Authorization": f"Bearer {str(self.session_token)}"}
                )

                if response.status_code != 200:
                    raise ValueError(response.json()["detail"])

                predictions = response.json()["predictions"]
                statuses = [(str(t["pred_uuid"]), t["status"]) for t in predictions]
                if statuses != self._last_statuses:
                    self.status_received.emit(predictions)
                    self._last_statuses = statuses
                    
            except Exception as e:
                pass
            
            self._stop_event.wait(1)

    def stop(self):
        self._running = False
        self.session_token = None
        self._stop_event.set()