from PyQt6.QtCore import QObject, pyqtSignal
from src.trainer.training_pipeline import TrainingPipeline

class TrainingWorker(QObject):
    log = pyqtSignal(str)  # Zdefiniowany sygnał logowania
    finished = pyqtSignal()

    def __init__(self, job_uuid, db_manager):
        super().__init__()
        self.job_uuid = job_uuid
        self.db_manager = db_manager
        self.train_pipeline = None

    def run(self):
        try:
            self.log.emit("Start treningu")
            
            # Pobierz konfigurację
            config = self.db_manager.get_training_config(self.job_uuid)

            # Tworzymy i uruchamiamy pipeline
            self.train_pipeline = TrainingPipeline(config, log_signal=self.log)  # Przekazujemy sygnał logowania
            self.train_pipeline.run()

        except Exception as e:
            self.log.emit(f"Błąd: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        if self.train_pipeline:
            self.train_pipeline.stop()
            self.log.emit("Zatrzymano trening")
