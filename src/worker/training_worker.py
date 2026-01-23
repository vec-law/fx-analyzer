from PyQt6.QtCore import QObject, pyqtSignal
from src.trainer.training_pipeline import TrainingPipeline

class TrainingWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, job_uuid, db_manager, log_to_console):
        super().__init__()
        self.job_uuid = job_uuid
        self.db_manager = db_manager
        self.train_pipeline = None
        self.log_to_console = log_to_console  # Przypisujemy log_to_console do instancji

    def run(self):
        try:
            config = self.db_manager.get_training_config(self.job_uuid)
            self.train_pipeline = TrainingPipeline(
                config=config, 
                log_to_console=self.log_to_console,  # Przekazujemy log_to_console
                db_manager=self.db_manager, 
                job_uuid=self.job_uuid
            )
            self.train_pipeline.run()

        except Exception as e:
            self.log_to_console(f"Błąd: {e}")  # Używamy self.log_to_console do logowania błędów
        finally:
            self.finished.emit()

    def stop(self):
        if self.train_pipeline:
            self.train_pipeline.stop()
