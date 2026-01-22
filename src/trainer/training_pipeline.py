import inspect

from src.loader import Loader
from src.preprocessor import Preprocessor

class TrainingPipeline:
    def __init__(self, config: dict, log_signal, db_manager, job_uuid):
        self.config = config
        self.log_signal = log_signal
        self.db_manager = db_manager
        self.job_uuid = job_uuid
        self.df = None

    def run(self):
        f_name = inspect.currentframe().f_code.co_name
        try:
            self.db_manager.update_training_status(self.job_uuid, 'running')
            self.log_signal.emit(f"[{f_name}] Rozpoczynanie treningu")

            loader = Loader(self.config, log_callback=self.log_signal.emit)

            self.df = loader.load_data()
            if self.df is None or self.df.empty:
                self.db_manager.update_training_status(self.job_uuid, "failed")
                self.stop()
                return

            self.db_manager.update_training_status(self.job_uuid, "completed")
            self.log_signal.emit(f"[{f_name}] Koniec treningu")

        except Exception as e:
            self.db_manager.update_training_status(self.job_uuid, 'failed')
            self.log_signal.emit(f"[{f_name}] Błąd: {e}")

    def stop(self):
        f_name = inspect.currentframe().f_code.co_name
        self.log_signal.emit(f"[{f_name}] Zatrzymano trening")
