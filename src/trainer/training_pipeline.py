import inspect
from src.loader import Loader
from src.cleaner import Cleaner

class TrainingPipeline:
    def __init__(self, config: dict, log_signal, db_manager, job_uuid):
        self.config = config
        self.log_signal = log_signal
        self.db_manager = db_manager
        self.job_uuid = job_uuid
        self.df = None
        self._is_stopped = False

    def run(self):
        f_name = inspect.currentframe().f_code.co_name
        try:
            self.db_manager.update_training_status(self.job_uuid, 'running')
            self.log_signal.emit(f"[{f_name}] Rozpoczynanie treningu")

            loader = Loader(self.config, log_signal=self.log_signal)
            self.df = loader.load_data()

            if self._is_stopped:
                self.db_manager.update_training_status(self.job_uuid, 'failed')
                self.log_signal.emit(f"[{f_name}] Proces przerwany przez użytkownika")
                return

            if self.df is None or self.df.empty:
                self.db_manager.update_training_status(self.job_uuid, "failed")
                self.log_signal.emit(f"[{f_name}] Przerwano: Loader nie zwrócił danych")
                return
            
            cleaner = Cleaner(self.config, log_signal=self.log_signal)
            self.df = cleaner.clean_data(self.df)

            if self._is_stopped:
                self.db_manager.update_training_status(self.job_uuid, 'failed')
                self.log_signal.emit(f"[{f_name}] Proces przerwany przez użytkownika")
                return

            if self.df is None or self.df.empty:
                self.db_manager.update_training_status(self.job_uuid, "failed")
                self.log_signal.emit(f"[{f_name}] Przerwano: Cleaner usunął wszystkie dane")
                return

            self.db_manager.update_training_status(self.job_uuid, "completed")
            self.log_signal.emit(f"[{f_name}] Koniec treningu")

        except Exception as e:
            self.db_manager.update_training_status(self.job_uuid, 'failed')
            self.log_signal.emit(f"[{f_name}] Błąd: {e}")

    def stop(self):
        f_name = inspect.currentframe().f_code.co_name
        self._is_stopped = True
        self.log_signal.emit(f"[{f_name}] Zatrzymywanie...")
