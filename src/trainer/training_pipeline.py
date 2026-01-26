import inspect
from src.loader import Loader
from src.cleaner import Cleaner
from src.data_extractor import DataExtractor
from src.preprocessor import Preprocessor
import torch
from src.model.model_manager import ModelManager

class TrainingPipeline:
    def __init__(self, config: dict, log_signal, db_manager, job_uuid):
        self.config = config
        self.log_signal = log_signal
        self.db_manager = db_manager
        self.job_uuid = job_uuid
        self._is_stopped = False
        self.df = None
        self.df_train = None
        self.df_train_norm = None
        self.df_test = None
        self.df_test_norm = None
        self.ser_mean = None
        self.ser_std = None
        self.ten_train_norm_x = None
        self.ten_train_norm_y = None
        self.ten_test_norm_x = None
        self.ten_test_norm_y = None
        self.device = None

    def run(self):
        f_name = inspect.currentframe().f_code.co_name
        try:
            self.db_manager.update_training_status(self.job_uuid, 'running')
            self.log_signal.emit(f"[{f_name}] Rozpoczynanie treningu")

            loader = Loader(self.config, log_signal=self.log_signal)

            self.df = loader.load_data()

            if self.df is None or self.df.empty:
                raise ValueError("Loader nie zwrócił danych")

            if self._handle_stop(f_name): return
            
            cleaner = Cleaner(self.config, log_signal=self.log_signal)

            self.df = cleaner.clean_data(self.df)

            if self.df is None or self.df.empty:
                raise ValueError("Cleaner usunął wszystkie dane")

            if self._handle_stop(f_name): return

            data_extractor = DataExtractor(self.config, log_signal=self.log_signal)

            self.df = data_extractor.add_features(self.df)

            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano cech")
            
            if self._handle_stop(f_name): return
            
            self.df = data_extractor.add_targets(self.df)

            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano wartości docelowych")
            
            if self._handle_stop(f_name): return
            
            self.df = data_extractor.dropna_and_cut(self.df, self.config['parameter_set']['samples_limit'])

            if self.df is None or self.df.empty:
                raise ValueError("Nie ucięto df")
            
            if self._handle_stop(f_name): return
            
            preprocessor = Preprocessor(self.config, self.log_signal)

            self.df_train, self.df_test = preprocessor.split_data(
                self.df,
                self.config['parameter_set']['test_samples'],
                self.config['feature_names'] + self.config['target_names']
            )

            if self.df_train is None:
                raise ValueError("Nie wykonano splitu")
            
            if self._handle_stop(f_name): return
            
            self.ser_mean, self.ser_std = preprocessor.calculate_stats(
                self.df_train,
                self.config['feature_names'] + self.config['target_names']
            )

            if self.ser_mean is None or self.ser_std is None:
                raise ValueError("Nie obliczono statystyk")
            
            if self._handle_stop(f_name): return
            
            self.db_manager.save_training_stats(self.job_uuid, self.ser_mean, self.ser_std)

            if self._handle_stop(f_name): return

            self.df_train_norm = preprocessor.scale_data(
                self.df_train,
                self.ser_mean,
                self.ser_std,
                self.config['feature_names'] + self.config['target_names']
            )

            if self.df_train_norm is None or self.df_train_norm.empty:
                raise ValueError("Nie znormalizowano df_train")
            
            if self._handle_stop(f_name): return
            
            if self.df_test is not None and not self.df_test.empty:
                self.df_test_norm = preprocessor.scale_data(
                    self.df_test,
                    self.ser_mean,
                    self.ser_std,
                    self.config['feature_names'] + self.config['target_names']
                )

                if self.df_test_norm is None or self.df_test_norm.empty:
                    raise ValueError("Nie znormalizowano df_test")
                
                if self._handle_stop(f_name): return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
                
            self.ten_train_norm_x = preprocessor.create_tensors(
                self.df_train_norm,
                self.config['feature_names'],
                self.device
            )

            self.ten_train_norm_y = preprocessor.create_tensors(
                self.df_train_norm,
                self.config['target_names'],
                self.device
            )

            if self.ten_train_norm_x is None or self.ten_train_norm_y is None:
                raise ValueError("Nie utworzono ten_train_norm")
            
            if self._handle_stop(f_name): return
            
            if self.df_test_norm is not None and not self.df_test_norm.empty:
                self.ten_test_norm_x = preprocessor.create_tensors(
                    self.df_test_norm,
                    self.config['feature_names'],
                    self.device
                )

                self.ten_test_norm_y = preprocessor.create_tensors(
                    self.df_test_norm,
                    self.config['target_names'],
                    self.device
                )

                if self.ten_test_norm_x is None or self.ten_test_norm_y is None:
                    raise ValueError("Nie utworzono ten_test_norm")
                
                if self._handle_stop(f_name): return

            model_manager = ModelManager(self.config, self.log_signal)

            for arch in self.config['architectures']:
                if self._handle_stop(f_name): return

                model, optimizer, loss_function = model_manager.create_model(
                    len(self.config['feature_names']),
                    len(self.config['target_names']),
                    self.config['parameter_set'],
                    arch,
                    self.device
                )

                if model is None or optimizer is None or loss_function is None:
                    raise ValueError("Nie utworzono modelu")
                
                model = model_manager.train_model(
                    model,
                    optimizer,
                    loss_function,
                    self.ten_train_norm_x,
                    self.ten_train_norm_y,
                    self.config['parameter_set'],
                    self.device
                )

                if model is None:
                    raise ValueError("Nie wykonano uczenia modelu")
                
                if self.ten_test_norm_x is not None or self.ten_test_norm_y is not None:
                    mse_loss, mae_loss = model_manager.evaluate_model(
                        model,
                        loss_function,
                        self.ten_test_norm_x,
                        self.ten_test_norm_y,
                    )

                    if mse_loss is None or mae_loss is None:
                        raise ValueError("Nie wykonano ewaluacji modelu")

            self.db_manager.update_training_status(self.job_uuid, "completed")
            self.log_signal.emit(f"[{f_name}] Koniec treningu")

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Przerwano z powodu błędu: {e}")
            try:
                self.db_manager.update_training_status(self.job_uuid, 'failed')
            except Exception as db_err:
                self.log_signal.emit(f"[{f_name}] Błąd bazy danych: {db_err}")

    def stop(self):
        f_name = inspect.currentframe().f_code.co_name
        self._is_stopped = True
        self.log_signal.emit(f"[{f_name}] Zatrzymywanie...")

    def _handle_stop(self, f_name):
        if self._is_stopped:
            self.db_manager.update_training_status(self.job_uuid, 'failed')
            self.log_signal.emit(f"[{f_name}] Proces przerwany przez użytkownika")
            return True
        return False
