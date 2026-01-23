from PyQt6.QtCore import QObject, pyqtSignal
from src.trainer.training_pipeline import TrainingPipeline

class TrainingWorker(QObject):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, job_uuid, db_manager):
        super().__init__()
        self.job_uuid = job_uuid
        self.db_manager = db_manager
        self.train_pipeline = None

    def run(self):
        try:
            config = self.db_manager.get_training_config(self.job_uuid)
            self.train_pipeline = TrainingPipeline(
                config=config, 
                log_signal=self.log_signal,
                db_manager=self.db_manager, 
                job_uuid=self.job_uuid
            )
            self.train_pipeline.run()

        except Exception as e:
            self.log_signal.emit(f"Błąd: {e}")  # Używamy self.log_to_console do logowania błędów
        finally:
            self.finished.emit()

    def stop(self):
        if self.train_pipeline:
            self.train_pipeline.stop()
