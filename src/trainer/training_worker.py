import time  # Potrzebujemy zaimportować time do opóźnienia
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QWidget, QPushButton, QTextEdit, QVBoxLayout, QLabel

# Worker
class TrainingWorker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, job_uuid, db_manager):
        super().__init__()
        self.job_uuid = job_uuid
        self.db_manager = db_manager
        self._running = True  # do obsługi stop

    def run(self):
        try:
            # Symulacja treningu
            self.log.emit("Start treningu")
            for epoch in range(1, 11):  # symulacja 10 epok
                if not self._running:
                    self.log.emit("Trening przerwany")
                    return
                self.log.emit(f"Epoka {epoch}/10")
                QThread.sleep(1)  # Każda epoka trwa 5 sekund
            self.log.emit("Trening zakończony")
        except Exception as e:
            self.log.emit(f"Błąd: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._running = False  # zatrzymanie treningu

