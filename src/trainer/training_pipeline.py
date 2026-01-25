import inspect
from src.loader import Loader
from src.cleaner import Cleaner
from src.data_extractor import DataExtractor
from src.preprocessor import Preprocessor

class TrainingPipeline:
    def __init__(self, config: dict, log_signal, db_manager, job_uuid):
        self.config = config
        self.log_signal = log_signal
        self.db_manager = db_manager
        self.job_uuid = job_uuid
        self._is_stopped = False
        self.df = None
        self.df_train = None
        self.df_test = None
        self.df_mean = None
        self.df_std = None

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
                raise ValueError("Loader nie zwrócił danych")
            
            cleaner = Cleaner(self.config, log_signal=self.log_signal)

            self.df = cleaner.clean_data(self.df)

            if self._is_stopped:
                self.db_manager.update_training_status(self.job_uuid, 'failed')
                self.log_signal.emit(f"[{f_name}] Proces przerwany przez użytkownika")
                return

            if self.df is None or self.df.empty:
                raise ValueError("Cleaner usunął wszystkie dane")
            
            data_extractor = DataExtractor(self.config, log_signal=self.log_signal)

            self.df = data_extractor.add_features(self.df)

            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano cech")
            
            self.df = data_extractor.add_targets(self.df)

            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano wartości docelowych")
            
            self.df = data_extractor.dropna_and_cut(self.df, self.config['parameter_set']['samples_limit'])

            if self.df is None or self.df.empty:
                raise ValueError("Nie ucięto df")
            
            preprocessor = Preprocessor(self.config, self.log_signal)

            self.df_train, self.df_test = preprocessor.split_data(
                self.df,
                self.config['parameter_set']['test_samples'],
                self.config['feature_names'] + self.config['target_names']
            )

            if self.df_train is None:
                raise ValueError("Nie wykonano splitu")
            
            self.df_mean, self.df_std = preprocessor.calculate_stats(
                self.df_train,
                self.config['feature_names'] + self.config['target_names']
            )

            if self.df_mean is None or self.df_std is None:
                raise ValueError("Nie obliczono statystyk")
            
            self.db_manager.save_training_stats(self.job_uuid, self.df_mean, self.df_std)

            self.db_manager.update_training_status(self.job_uuid, "completed")
            self.log_signal.emit(f"[{f_name}] Koniec treningu")

        except Exception as e:
            try:
                self.db_manager.update_training_status(self.job_uuid, 'failed')
            except:
                pass
            
            self.log_signal.emit(f"[{f_name}] Przerwano z powodu błędu: {e}")

    def stop(self):
        f_name = inspect.currentframe().f_code.co_name
        self._is_stopped = True
        self.log_signal.emit(f"[{f_name}] Zatrzymywanie...")
